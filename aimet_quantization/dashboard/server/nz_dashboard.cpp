// nz_dashboard: demo web server that runs on the QIDK (Android, aarch64).
//   GET  /                  static dashboard from <www>
//   GET  /api/status        device + server info
//   POST /api/run?model=<all_sensors_1h|all_sensors_24h|single_node>&variant=<aimet_w8a16|fp32>
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

#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

static std::string ROOT = "/data/local/tmp/nz_aimet";
static std::string WWW = ROOT + "/dashboard/www";
static std::mutex run_mutex;

static const std::map<std::string, int> FEATURES = {
    {"all_sensors_1h", 58}, {"all_sensors_24h", 58}, {"single_node", 10}};

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
  if (fit == FEATURES.end() || (variant != "aimet_w8a16" && variant != "fp32")) {
    res.status = 400;
    res.set_content(R"({"ok":false,"error":"unknown model or variant"})", "application/json");
    return;
  }
  size_t expect = 96u * fit->second * sizeof(float);
  if (req.body.size() != expect) {
    res.status = 400;
    res.set_content("{\"ok\":false,\"error\":\"expected " + std::to_string(expect) + " bytes (float32 [1,96," +
                        std::to_string(fit->second) + "]), got " + std::to_string(req.body.size()) + "\"}",
                    "application/json");
    return;
  }
  std::lock_guard<std::mutex> lock(run_mutex);
  std::string live = ROOT + "/" + model + "/live";
  sh("mkdir -p " + live + " && rm -rf " + live + "/out");
  { std::ofstream(live + "/window.raw", std::ios::binary).write(req.body.data(), req.body.size()); }
  { std::ofstream(live + "/list.txt") << live << "/window.raw\n"; }
  bool htp = variant == "aimet_w8a16";
  std::string cmd = "cd " + ROOT + " && LD_LIBRARY_PATH=" + ROOT + "/lib ADSP_LIBRARY_PATH=" + ROOT + "/lib ./bin/qnn-net-run" +
                    " --backend lib/" + (htp ? "libQnnHtp.so" : "libQnnCpu.so") + " --model " + model + "/models/lib" + variant +
                    ".so --input_list " + live + "/list.txt --output_dir " + live + "/out --profiling_level basic" +
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
  }
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
  fprintf(stderr, "nz_dashboard listening on 0.0.0.0:%d, www=%s\n", port, WWW.c_str());
  svr.listen("0.0.0.0", port);
  return 0;
}
