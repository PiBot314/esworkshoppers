// ONNX Runtime CPU benchmark for the QIDK (Android aarch64).
//   ort_bench <model.onnx> <threads> latency <input.raw> <warmup> <runs>
//       -> per-inference times (mean / p50 / p95 / min, ms)
//   ort_bench <model.onnx> <threads> loop <input.raw> <seconds>
//       -> runs back-to-back for <seconds>, prints count and rate (used for the power measurement)
//   ort_bench <model.onnx> <threads> outputs <list.txt> <out.bin>
//       -> runs every raw file in the list, writes all outputs (float32) to out.bin (accuracy check)
// Input files are raw float32 matching the model's (fixed) input shape.
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static std::vector<float> read_raw(const std::string& p, size_t n) {
  std::vector<float> v(n);
  std::ifstream f(p, std::ios::binary);
  if (!f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float))) {
    std::fprintf(stderr, "cannot read %zu floats from %s\n", n, p.c_str());
    std::exit(2);
  }
  return v;
}

int main(int argc, char** argv) {
  if (argc < 5) { std::fprintf(stderr, "usage: see source\n"); return 1; }
  std::string model = argv[1], mode = argv[3];
  int threads = std::atoi(argv[2]);

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bench");
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(threads);
  so.SetInterOpNumThreads(1);
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  auto t_load = Clock::now();
  Ort::Session sess(env, model.c_str(), so);
  double load_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_load).count();

  Ort::AllocatorWithDefaultOptions alloc;
  auto in_name = sess.GetInputNameAllocated(0, alloc);
  auto out_name = sess.GetOutputNameAllocated(0, alloc);
  auto shape = sess.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  size_t n = 1;
  for (auto& d : shape) { if (d < 1) d = 1; n *= d; }
  const char* in_names[] = {in_name.get()};
  const char* out_names[] = {out_name.get()};
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  auto run_once = [&](std::vector<float>& x) {
    Ort::Value t = Ort::Value::CreateTensor<float>(mem, x.data(), x.size(), shape.data(), shape.size());
    return sess.Run(Ort::RunOptions{nullptr}, in_names, &t, 1, out_names, 1);
  };

  if (mode == "latency") {
    auto x = read_raw(argv[4], n);
    int warm = std::atoi(argv[5]), runs = std::atoi(argv[6]);
    for (int i = 0; i < warm; i++) run_once(x);
    std::vector<double> ms(runs);
    for (int i = 0; i < runs; i++) {
      auto t0 = Clock::now();
      run_once(x);
      ms[i] = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }
    std::vector<double> s = ms;
    std::sort(s.begin(), s.end());
    double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / runs;
    std::printf("load_ms=%.1f runs=%d mean_ms=%.4f p50_ms=%.4f p95_ms=%.4f min_ms=%.4f\n", load_ms, runs, mean,
                s[runs / 2], s[(int)(runs * 0.95)], s[0]);
  } else if (mode == "loop") {
    auto x = read_raw(argv[4], n);
    double secs = std::atof(argv[5]);
    for (int i = 0; i < 20; i++) run_once(x);
    long count = 0;
    auto t0 = Clock::now();
    double el = 0;
    while (el < secs) {
      run_once(x);
      count++;
      el = std::chrono::duration<double>(Clock::now() - t0).count();
    }
    std::printf("loop count=%ld seconds=%.3f rate_per_s=%.2f\n", count, el, count / el);
  } else if (mode == "outputs") {
    std::ifstream lst(argv[4]);
    std::ofstream out(argv[5], std::ios::binary);
    std::string p;
    long k = 0;
    while (std::getline(lst, p)) {
      if (p.empty()) continue;
      auto x = read_raw(p, n);
      auto o = run_once(x);
      size_t m = o[0].GetTensorTypeAndShapeInfo().GetElementCount();
      out.write(reinterpret_cast<const char*>(o[0].GetTensorData<float>()), m * sizeof(float));
      k++;
    }
    std::printf("outputs written=%ld\n", k);
  }
  return 0;
}
