// nz_dashboard: demo web server that runs on the QIDK (Android, aarch64).
//   GET  /                  static dashboard from <www>
//   GET  /api/status        device + server info
//   GET  /api/demo?cmd=...   benchmark demo (baseline/guided/CNN-LSTM, NPU vs CPU), see benchmark_qidk/board/demo.sh
//   GET  /api/models        built-in models + the ones in the registry file
//   POST /api/warm?models=a,b&ttl=60   start resident runners (nz_runner) for those models on the HTP; each one
//        finalizes its graph once (~2 s), then /api/run uses it (~5 ms per window instead of ~1.8 s per qnn-net-run).
//        A runner exits after <ttl> s without requests. GET /api/warm lists the running ones.
//   POST /api/run?model=<all_sensors_1h|all_sensors_24h|single_node|<registry id>>&variant=<aimet_w8a16|fp32>
//        body = one input window, float32 little-endian [1, 96, F], normalized
//        -> runs qnn-net-run on the HTP (W8A16) or CPU (FP32) and returns the outputs + timings as JSON
// Live runs are serialized (one NPU job at a time). Build: dashboard/build.sh (Android NDK, inside qairt image).
// windows are 22 KB (58 features); clients that send them form-encoded must not hit the 8 KB default
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH (4 * 1024 * 1024)
#include "httplib.h"

#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <signal.h>
#include <thread>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

static std::string ROOT = "/data/local/tmp/nz_aimet";
static std::string WWW = ROOT + "/dashboard/www";
static std::string BENCH = "/data/local/tmp/nz_bench";   // benchmark_qidk files + demo.sh (pushed by demo_setup.sh)
static std::mutex run_mutex;

static const std::map<std::string, int> FEATURES = {
    {"all_sensors_1h", 58}, {"all_sensors_24h", 58}, {"single_node", 10}};

// Extra models (plan_model_switcher.md §3.2): one line per model in the registry file,
//   <id> <n_features> <dir>      with <dir>/libw8a16.so (HTP) and <dir>/libfp32.so (CPU); runs in <dir>/live
struct Extra { int features; std::string dir; };
static std::map<std::string, Extra> EXTRA;
static std::string REGISTRY = "/data/local/tmp/nz_live/models/registry.txt";
// ---- resident runners (nz_runner = QNN SampleApp with a stdin request loop, nz_live/runner/)
static std::string RUNNER = "/data/local/tmp/nz_live/bin/nz_runner";
struct Runner { pid_t pid; FILE* in; FILE* out; std::chrono::steady_clock::time_point last; int ttl; };
static std::map<std::string, Runner> RUNNERS;  // key: model id; guarded by run_mutex

static void stop_runner(const std::string& id) {
  auto it = RUNNERS.find(id);
  if (it == RUNNERS.end()) return;
  fputs("QUIT\n", it->second.in); fflush(it->second.in);
  fclose(it->second.in); fclose(it->second.out);
  for (int i = 0; i < 20 && waitpid(it->second.pid, nullptr, WNOHANG) == 0; ++i) usleep(50000);
  if (waitpid(it->second.pid, nullptr, WNOHANG) == 0) { kill(it->second.pid, SIGKILL); waitpid(it->second.pid, nullptr, 0); }
  RUNNERS.erase(it);
}

// start a runner for <lib> (HTP); returns "" or an error. Caller holds run_mutex.
static std::string start_runner(const std::string& id, const std::string& lib, const std::string& live, int nfeat, int ttl) {
  if (RUNNERS.count(id)) { RUNNERS[id].ttl = ttl; RUNNERS[id].last = std::chrono::steady_clock::now(); return ""; }
  mkdir(live.c_str(), 0777);
  std::string win = live + "/window.raw", list = live + "/list.txt";
  { std::ifstream f(win); if (!f) { std::vector<float> z(96 * nfeat, 0.f); std::ofstream(win, std::ios::binary).write((char*)z.data(), z.size() * 4); } }
  { std::ofstream(list) << win << "\n"; }
  int to[2], from[2];
  if (pipe(to) || pipe(from)) return "pipe failed";
  pid_t pid = fork();
  if (pid == 0) {
    dup2(to[0], 0); dup2(from[1], 1);
    int devnull = open("/dev/null", O_WRONLY); dup2(devnull, 2);
    close(to[1]); close(from[0]);
    if (chdir(ROOT.c_str()) != 0) _exit(127);
    setenv("LD_LIBRARY_PATH", (ROOT + "/lib").c_str(), 1);
    setenv("ADSP_LIBRARY_PATH", (ROOT + "/lib").c_str(), 1);
    setenv("NZ_SERVE", "1", 1);
    std::string out0 = live + "/out_runner";
    execl(RUNNER.c_str(), "nz_runner", "--model", lib.c_str(), "--backend", "lib/libQnnHtp.so", "--input_list", list.c_str(),
          "--output_dir", out0.c_str(), "--log_level", "error", (char*)nullptr);
    _exit(127);
  }
  close(to[0]); close(from[1]);
  Runner rn{pid, fdopen(to[1], "w"), fdopen(from[0], "r"), std::chrono::steady_clock::now(), ttl};
  char buf[256];
  while (fgets(buf, sizeof buf, rn.out)) {
    if (strncmp(buf, "READY", 5) == 0) { RUNNERS[id] = rn; return ""; }
  }
  fclose(rn.in); fclose(rn.out); waitpid(pid, nullptr, 0);
  return "runner did not start";
}

// one window through a warm runner; returns execute microseconds, or -1. Caller holds run_mutex.
static long runner_exec(const std::string& id, const std::string& list, const std::string& outdir) {
  auto it = RUNNERS.find(id);
  if (it == RUNNERS.end()) return -1;
  fprintf(it->second.in, "%s %s\n", list.c_str(), outdir.c_str()); fflush(it->second.in);
  char buf[256];
  if (!fgets(buf, sizeof buf, it->second.out)) { stop_runner(id); return -1; }
  it->second.last = std::chrono::steady_clock::now();
  if (strncmp(buf, "OK ", 3) != 0) return -1;
  return atol(buf + 3);
}

static void load_registry() {
  std::ifstream f(REGISTRY);
  std::string id, dir;
  int n;
  while (f >> id >> n >> dir)
    if (!FEATURES.count(id) && n > 0 && n <= 1024) EXTRA[id] = {n, dir};
}

static std::string sh(const std::string& cmd) {
  std::string out;
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) return out;
  char buf[4096];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, p)) > 0) out.append(buf, n);
  pclose(p);
  return out;
}

static std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (c == '\n') o += "\\n";
    else if (static_cast<unsigned char>(c) >= 0x20) o += c;
  }
  return o;
}

static long find_us(const std::string& text, const std::string& label) {
  // qnn-profile-viewer lines look like: "    Backend (Accelerator (execute) time): 3419 us"
  std::regex re(std::regex_replace(label, std::regex(R"([()])"), R"(\$&)") + R"(\):\s*(\d+)\s*us)");
  std::smatch m;
  // take the last match = the "Average"/per-run section that follows the overall stats
  long v = -1;
  for (auto it = std::sregex_iterator(text.begin(), text.end(), re); it != std::sregex_iterator(); ++it)
    v = std::stol((*it)[1]);
  return v;
}

static std::string first_raw(const std::string& dir) {
  DIR* d = opendir(dir.c_str());
  if (!d) return "";
  std::string found;
  while (dirent* e = readdir(d)) {
    std::string n = e->d_name;
    if (n.size() > 4 && n.substr(n.size() - 4) == ".raw") { found = dir + "/" + n; break; }
  }
  closedir(d);
  return found;
}

static void run(const httplib::Request& req, httplib::Response& res) {
  auto model = req.get_param_value("model");
  auto variant = req.has_param("variant") ? req.get_param_value("variant") : "aimet_w8a16";
  auto fit = FEATURES.find(model);
  auto xit = EXTRA.find(model);
  if ((fit == FEATURES.end() && xit == EXTRA.end()) || (variant != "aimet_w8a16" && variant != "fp32")) {
    res.status = 400;
    res.set_content(R"({"ok":false,"error":"unknown model or variant"})", "application/json");
    return;
  }
  bool builtin = fit != FEATURES.end();
  int nfeat = builtin ? fit->second : xit->second.features;
  std::string live = builtin ? ROOT + "/" + model + "/live" : xit->second.dir + "/live";
  std::string lib = builtin ? ROOT + "/" + model + "/models/lib" + variant + ".so"
                            : xit->second.dir + "/lib" + (variant == "aimet_w8a16" ? "w8a16" : "fp32") + ".so";
  size_t expect = 96u * nfeat * sizeof(float);
  if (req.body.size() != expect) {
    res.status = 400;
    res.set_content("{\"ok\":false,\"error\":\"expected " + std::to_string(expect) + " bytes (float32 [1,96," +
                        std::to_string(nfeat) + "]), got " + std::to_string(req.body.size()) + "\"}",
                    "application/json");
    return;
  }
  std::lock_guard<std::mutex> lock(run_mutex);
  if (variant == "aimet_w8a16" && RUNNERS.count(model)) {  // warm path: resident runner, no process start, no finalize
    sh("mkdir -p " + live + " && rm -rf " + live + "/out");
    { std::ofstream(live + "/window.raw", std::ios::binary).write(req.body.data(), req.body.size()); }
    { std::ofstream(live + "/list.txt") << live << "/window.raw\n"; }
    auto t0 = std::chrono::steady_clock::now();
    long us = runner_exec(model, live + "/list.txt", live + "/out");
    double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::string raw = us >= 0 ? first_raw(live + "/out/Result_0") : "";
    if (!raw.empty()) {
      std::ifstream f(raw, std::ios::binary | std::ios::ate);
      size_t bytes = f.tellg();
      f.seekg(0);
      std::vector<float> vals(bytes / sizeof(float));
      f.read(reinterpret_cast<char*>(vals.data()), bytes);
      std::ostringstream js;
      js << "{\"ok\":true,\"model\":\"" << model << "\",\"variant\":\"" << variant << "\",\"backend\":\"HTP (Hexagon NPU)\",\"runner\":\"resident\",\"wall_ms\":" << wall_ms
         << ",\"execute_us\":" << us << ",\"finalize_us\":0,\"out\":[";
      for (size_t i = 0; i < vals.size(); ++i) js << (i ? "," : "") << vals[i];
      js << "]}";
      res.set_content(js.str(), "application/json");
      return;
    }
    stop_runner(model);  // fall through to the one-shot path
  }
  sh("mkdir -p " + live + " && rm -rf " + live + "/out");
  { std::ofstream(live + "/window.raw", std::ios::binary).write(req.body.data(), req.body.size()); }
  { std::ofstream(live + "/list.txt") << live << "/window.raw\n"; }
  bool htp = variant == "aimet_w8a16";
  std::string cmd = "cd " + ROOT + " && LD_LIBRARY_PATH=" + ROOT + "/lib ADSP_LIBRARY_PATH=" + ROOT + "/lib ./bin/qnn-net-run" +
                    " --backend lib/" + (htp ? "libQnnHtp.so" : "libQnnCpu.so") + " --model " + lib +
                    " --input_list " + live + "/list.txt --output_dir " + live + "/out --profiling_level basic" +
                    (htp ? " --perf_profile burst" : "") + " > " + live + "/run.log 2>&1; echo rc=$?";
  auto t0 = std::chrono::steady_clock::now();
  std::string rc = sh(cmd);
  double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::string prof = sh("cd " + ROOT + " && ./bin/qnn-profile-viewer --input_log " + live + "/out/qnn-profiling-data_0.log 2>&1");
  std::string raw = first_raw(live + "/out/Result_0");
  std::ostringstream js;
  if (raw.empty()) {
    std::string log = sh("tail -5 " + live + "/run.log");
    res.status = 500;
    js << "{\"ok\":false,\"error\":\"inference failed\",\"log\":\"" << json_escape(log) << "\"}";
    res.set_content(js.str(), "application/json");
    return;
  }
  std::ifstream f(raw, std::ios::binary | std::ios::ate);
  size_t bytes = f.tellg();
  f.seekg(0);
  std::vector<float> vals(bytes / sizeof(float));
  f.read(reinterpret_cast<char*>(vals.data()), bytes);
  js << "{\"ok\":true,\"model\":\"" << model << "\",\"variant\":\"" << variant << "\",\"backend\":\""
     << (htp ? "HTP (Hexagon NPU)" : "CPU") << "\",\"wall_ms\":" << wall_ms
     << ",\"execute_us\":" << find_us(prof, htp ? "Accelerator (execute) time" : "GRAPH_EXECUTE")
     << ",\"finalize_us\":" << find_us(prof, htp ? "QNN (finalize) time" : "GRAPH_FINALIZE") << ",\"out\":[";
  for (size_t i = 0; i < vals.size(); ++i) js << (i ? "," : "") << vals[i];
  js << "]}";
  res.set_content(js.str(), "application/json");
}

int main(int argc, char** argv) {
  int port = 8080;
  bool daemon = false;
  std::string logfile;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--daemon") daemon = true;
    else if (i + 1 < argc && k == "--port") port = std::stoi(argv[++i]);
    else if (i + 1 < argc && k == "--root") { ROOT = argv[++i]; WWW = ROOT + "/dashboard/www"; }
    else if (i + 1 < argc && k == "--log") logfile = argv[++i];
    else if (i + 1 < argc && k == "--registry") REGISTRY = argv[++i];
  }
  load_registry();
  if (daemon) {  // detach completely so `adb shell` returns immediately (it waits for anything holding its pty)
    if (fork() > 0) return 0;
    setsid();
    if (fork() > 0) return 0;
    int in = open("/dev/null", O_RDONLY);
    int out = open(logfile.empty() ? "/dev/null" : logfile.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    dup2(in, 0); dup2(out, 1); dup2(out, 2);
    if (in > 2) close(in);
    if (out > 2) close(out);
  }
  httplib::Server svr;
  svr.set_payload_max_length(4 * 1024 * 1024);
  if (!svr.set_mount_point("/", WWW)) { fprintf(stderr, "www dir missing: %s\n", WWW.c_str()); return 1; }
  svr.set_file_extension_and_mimetype_mapping("f32", "application/octet-stream");
  svr.Get("/api/status", [](const httplib::Request&, httplib::Response& res) {
    std::string m = sh("getprop ro.product.model"), soc = sh("getprop ro.soc.model"), up = sh("uptime");
    std::ostringstream js;
    js << "{\"ok\":true,\"device\":\"" << json_escape(m) << "\",\"soc\":\"" << json_escape(soc) << "\",\"uptime\":\""
       << json_escape(up) << "\",\"server\":\"nz_dashboard 1.0 (cpp-httplib " CPPHTTPLIB_VERSION ")\"}";
    res.set_content(js.str(), "application/json");
  });
  svr.Post("/api/run", run);
  svr.Post("/api/warm", [](const httplib::Request& req, httplib::Response& res) {
    int ttl = req.has_param("ttl") ? std::max(5, std::min(3600, atoi(req.get_param_value("ttl").c_str()))) : 60;
    std::string list = req.has_param("models") ? req.get_param_value("models") : "";
    std::ostringstream js;
    js << "{\"ok\":true,\"ttl\":" << ttl << ",\"started\":[";
    std::lock_guard<std::mutex> lock(run_mutex);
    std::istringstream is(list);
    bool first = true;
    for (std::string id; std::getline(is, id, ',');) {
      auto fit = FEATURES.find(id);
      auto xit = EXTRA.find(id);
      if (fit == FEATURES.end() && xit == EXTRA.end()) continue;
      bool builtin = fit != FEATURES.end();
      std::string live = builtin ? ROOT + "/" + id + "/live" : xit->second.dir + "/live";
      std::string lib = builtin ? ROOT + "/" + id + "/models/libaimet_w8a16.so" : xit->second.dir + "/libw8a16.so";
      auto t0 = std::chrono::steady_clock::now();
      std::string err = start_runner(id, lib, live, builtin ? fit->second : xit->second.features, ttl);
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      js << (first ? "" : ",") << "{\"id\":\"" << json_escape(id) << "\",\"ok\":" << (err.empty() ? "true" : "false")
         << ",\"ms\":" << ms << (err.empty() ? "" : ",\"error\":\"" + json_escape(err) + "\"") << "}";
      first = false;
    }
    js << "]}";
    res.set_content(js.str(), "application/json");
  });
  svr.Get("/api/warm", [](const httplib::Request&, httplib::Response& res) {
    std::lock_guard<std::mutex> lock(run_mutex);
    std::ostringstream js;
    js << "{\"ok\":true,\"running\":[";
    bool first = true;
    auto now = std::chrono::steady_clock::now();
    for (auto& rn : RUNNERS) {
      js << (first ? "" : ",") << "{\"id\":\"" << json_escape(rn.first) << "\",\"idle_s\":"
         << std::chrono::duration_cast<std::chrono::seconds>(now - rn.second.last).count() << ",\"ttl\":" << rn.second.ttl << "}";
      first = false;
    }
    js << "]}";
    res.set_content(js.str(), "application/json");
  });
  // reaper: runners exit after their ttl without requests (the NPU and memory are free between checks)
  std::thread([] {
    for (;;) {
      sleep(2);
      std::lock_guard<std::mutex> lock(run_mutex);
      auto now = std::chrono::steady_clock::now();
      std::vector<std::string> old;
      for (auto& rn : RUNNERS)
        if (std::chrono::duration_cast<std::chrono::seconds>(now - rn.second.last).count() >= rn.second.ttl) old.push_back(rn.first);
      for (auto& id : old) stop_runner(id);
    }
  }).detach();
  svr.Get("/api/models", [](const httplib::Request&, httplib::Response& res) {
    std::ostringstream js;
    js << "{\"ok\":true,\"models\":[";
    bool first = true;
    for (auto& m : FEATURES) { js << (first ? "" : ",") << "{\"id\":\"" << m.first << "\",\"features\":" << m.second << ",\"builtin\":true}"; first = false; }
    for (auto& m : EXTRA) { js << (first ? "" : ",") << "{\"id\":\"" << json_escape(m.first) << "\",\"features\":" << m.second.features << ",\"builtin\":false}"; first = false; }
    js << "]}";
    res.set_content(js.str(), "application/json");
  });
  // GET /api/demo?cmd=<models|forecast|speed|accuracy>&model=..&window=..&prec=..&n=..
  //   runs benchmark_qidk's demo.sh on the board; returns {"ok":..,"text":<its printout>,"data":<its JSON line>}
  svr.Get("/api/demo", [](const httplib::Request& req, httplib::Response& res) {
    static const std::regex safe("^[a-z0-9_]{1,24}$");
    auto arg = [&](const char* k, const char* def) {
      std::string v = req.has_param(k) ? req.get_param_value(k) : def;
      return std::regex_match(v, safe) ? v : std::string("");
    };
    std::string cmd = arg("cmd", "models");
    std::string args;
    if (cmd == "forecast") args = arg("model", "guided_h4") + " " + arg("window", "0") + " " + arg("prec", "w8a16");
    else if (cmd == "speed") args = arg("prec", "w8a16");
    else if (cmd == "accuracy") args = arg("model", "guided_h4") + " " + arg("prec", "w8a16") + " " + arg("n", "500");
    else if (cmd != "models") { res.status = 400; res.set_content("{\"ok\":false,\"error\":\"bad cmd\"}", "application/json"); return; }
    std::lock_guard<std::mutex> lock(run_mutex);
    std::string out = sh("cd " + BENCH + " && JSON=1 sh demo.sh " + cmd + " " + args + " 2>&1");
    std::string text, data = "null";
    std::istringstream is(out);
    for (std::string line; std::getline(is, line);) {
      if (line.rfind("JSON:", 0) == 0) data = line.substr(5);
      else text += line + "\n";
    }
    res.set_content("{\"ok\":" + std::string(data == "null" ? "false" : "true") + ",\"text\":\"" + json_escape(text) +
                        "\",\"data\":" + data + "}", "application/json");
  });
  fprintf(stderr, "nz_dashboard listening on 0.0.0.0:%d, www=%s\n", port, WWW.c_str());
  svr.listen("0.0.0.0", port);
  return 0;
}
