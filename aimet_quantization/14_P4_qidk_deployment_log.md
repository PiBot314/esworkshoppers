# 14 — P4 Log: Conversion and Deployment to the QIDK (executed 2026-10-05)

Scripts: `07_prepare_raw.py` (raw inputs + `--encodings` handoff filter), `08_qairt_build.sh` (QNN model libraries), `09_run_on_qidk.sh` (push/run/pull), `10_compare_device.py` (V3/V4/V5), `11_benchmark_latency.sh`, `12_qidk_forecast.py` (end-user runner).
Target: QIDK `d42f06b`, SM8650, HTP v75, QAIRT runtime 2.47 in `/data/local/tmp/nz_aimet`.

## 1. The path that works (and why it differs from the plan in doc 07)
```
AIMET export (model.onnx + model.encodings)
  └─ 07 --encodings: keep only real activations (input, ReLU, output) + Gemm weight encodings → model.qairt.encodings
qnn-onnx-converter  --quantization_overrides model.qairt.encodings  --input_list calib(512)  --act_bitwidth 16
                    --weights_bitwidth 8 --bias_bitwidth 32  --expand_lstm_op_structure  --preserve_io layout
  └─ qnn-model-lib-generator -t aarch64-android → lib<variant>.so
adb push → qnn-net-run --backend libQnnHtp.so --model lib<variant>.so --input_list … → adb pull → compare
```
Every deviation from doc 07 came from a concrete failure:

| # | Attempt | Failure | Diagnosis | Fix |
|---|---|---|---|---|
| 1 | `qairt-converter` + overrides → `qairt-quantizer` (doc 07 plan) | `quantizer`: *"Activation bitwidth conversion from 8 to 32 is not supported"*. Converter: *"Following OPs fallback to float: node_LSTM_*_step_*"* | Ops without override encodings (the unrolled LSTM internals) are tagged float-fallback, and the quantizer can't reconcile them. Also, AIMET encodes the constant LSTM weight/bias subgraph as *activations*, which QAIRT folds into static params | Use the classic one-step `qnn-onnx-converter` (overrides **+** `--input_list`: calibrates what the overrides don't cover). Filter out encodings on constant-folded tensors |
| 2 | `qairt-quantizer` run in container | `Could not create output directory: ./output/` | cwd `/` not writable as non-root | `cd` into the output folder |
| 3 | Fused quantized `Lstm` op (default unrolling) | On device: `MODEL_GRAPH_OP_VALIDATION_ERROR` for `node_LSTM_65_step_0`. Also: the recurrent hidden state between steps was **8-bit** despite `--act_bitwidth 16` (192 tensors) | HTP rejects that quantized Lstm configuration | **`--expand_lstm_op_structure`**: each step becomes MatMul/Add/Sigmoid/Tanh/Mul (standard HTP ops). All 3,853 activations then come out 16-bit |
| 4 | Expanded graph on HTP | 4.9 dB vs FP32 (garbage) | Converter re-laid the input as `[1, 58, 96]` (channel-first) while the raw files are `[1, 96, 58]` | **`--preserve_io layout`** keeps the ONNX layout (as apps expect) |
| 5 | (same) | 14.6 dB. Per-step hidden tensors on the HTP were exactly **×1.44** too large (1/0.6947: ratio of the calibrated range to the [−1,1] override) | Overriding LSTM hidden outputs in a graph where per-step views alias that tensor | Don't pass LSTM-hidden overrides to QNN (they stay in the AIMET simulation) |
| 6 | (same) | still **14.6 dB** for AIMET W8A16, while QAIRT-only W8A16 = 44.9 dB and AIMET W8A8 = 32.8 dB | Only difference: AIMET W8A16 carried **per-channel Gemm weights** (axis 0). Same model with Gemm param encodings removed → **46.2 dB** | Per-channel Gemm mis-executes in this path → re-ran AutoQuant **per-tensor** (doc 13 §6), so AdaRound optimizes the scales actually deployed |
| 7 | Long device runs | `adb: device not found` twice (device uptime unaffected) | Host adb connection drops | Retry wrapper (kill/start server, wait-for-device) in all device scripts |

Debugging method that found #4–#6: **`qnn-net-run --debug`** dumps every intermediate tensor. One window run on HTP-INT and on CPU-FP32, then compared tensor-by-tensor (SQNR per tensor, per LSTM step). This showed that the recurrence itself is stable (gate pre-activation SQNR stays ≈37–45 dB over all 96 steps) and pinpointed the scaled tensors.

## 2. Results on the device (held-out TEST windows: 4,832 / 4,740 / 4,861)
SQNR and max|Δ| are against the FP32 ONNX model on the same windows (normalized output units; 0.0045 = 0.45 % of capacity). "vs sim" = agreement with the AIMET simulation of the same quantized model. Latency is the mean of 300 inferences of one window (`--perf_profile burst`). "Accel" = HTP execute time, "QNN exec" = time through the QNN API incl. RPC. IPS includes file I/O.

| Model | Variant | SQNR vs FP32 (dB) | max\|Δ\| | vs sim (dB) | Headline R² FP32 → device | Flag agreement natural / injected | Accel (ms) | QNN exec (ms) | IPS |
|---|---|---|---|---|---|---|---|---|---|
| 1 h | CPU FP32 (V3) | 100.0 | 0 | – | 0.8522 → 0.8522 | 100 / 100 % | – | – | 60 |
| 1 h | **HTP AIMET W8A16** | **48.4** | **0.0045** | 47.6 | 0.8522 → **0.8529** | **99.66 / 100 %** | **3.42** | 5.11 | 169 |
| 1 h | HTP QAIRT-only W8A16 | 44.9 | 0.0261 | 46.1 | 0.8522 → 0.8548 | 99.69 / 100 % | 3.41 | 5.08 | 170 |
| 1 h | HTP AIMET W8A8 | 32.8 | 0.0402 | 32.3 | 0.8522 → 0.8500 | 98.37 / 99.75 % | 2.51 | 4.15 | 208 |
| 24 h | CPU FP32 (V3) | 100.0 | 0 | – | 0.3388 → 0.3388 | 100 / 100 % | – | – | 32 |
| 24 h | **HTP AIMET W8A16** | **46.4** | **0.0126** | 45.1 | 0.3388 → **0.3349** | **99.76 / 99.96 %** | **3.44** | 5.11 | 170 |
| 24 h | HTP QAIRT-only W8A16 | 43.8 | 0.0147 | 42.3 | 0.3388 → 0.3344 | 99.77 / 99.98 % | 3.43 | 5.11 | 169 |
| 24 h | HTP AIMET W8A8 | **14.7 ✗** | 0.4201 | 14.7 | 0.3388 → 0.1670 | 89.27 / 94.89 % | 2.79 | 4.38 | 197 |
| single | CPU FP32 (V3) | 100.0 | 0 | – | 0.9499 → 0.9499 | 100 / 100 % | – | – | 68 |
| single | **HTP AIMET W8A16** | **52.1** | **0.0045** | 54.6 | 0.9499 → **0.9498** | **99.92 / 99.92 %** | **3.93** | 5.48 | 165 |
| single | HTP QAIRT-only W8A16 | 46.3 | 0.0065 | 45.2 | 0.9499 → 0.9495 | 99.90 / 99.88 % | 3.89 | 5.44 | 165 |
| single | HTP AIMET W8A8 | 36.2 | 0.0368 | 36.1 | 0.9499 → 0.9482 | 99.88 / 99.84 % | 1.84 | 3.27 | 264 |

Per-domain accuracy of the deployed W8A16 models on the device (rebuilt data): 1 h Solar R² 0.943 / MAE 23.3 kW, Grid 0.763 / 36.0 kW; 24 h Solar 0.798 / 42.1 kW; single node 0.950 / 1.97 kW. All are within ±0.008 of FP32.

### Verification ladder (doc 08)
| Rung | Check | Result |
|---|---|---|
| V0 | Rebuilt data reproduces reported FP32 accuracy | ✅ within 0.007–0.03 R² (doc 12) |
| V1 | AIMET graph == ORT | ✅ 1.2e-7 (doc 10) |
| V2 | AIMET W8A16 sim ≥ 30 dB, ΔR² ≈ 0 | ✅ 45.8–51.8 dB (doc 13 §6) |
| V3 | Converted FP32 graph on device CPU == ONNX | ✅ **bit-exact** (100 dB, max\|Δ\| 0) for all 3 models |
| V4 | HTP INT ≈ AIMET sim | ✅ 45–55 dB agreement for W8A16. Device SQNR vs FP32 is even slightly *higher* than the simulation's on the 1 h/24 h models. Likely reason: on the device the LSTM hidden outputs use their calibrated range (finer steps), while the simulation used the conservative [−1, 1] prior (§1 #5) |
| V5 | AIMET better than QAIRT-only | ✅ **+3.5 / +2.6 / +5.9 dB**. Worst-case error 6× lower on 1 h (0.0045 vs 0.026) |

### Interpretation
* **W8A16 is the right deliverable.** On all three models, the AIMET-quantized W8A16 graph on the HTP matches FP32 to 46–52 dB, R² changes by ≤ 0.004, and ≥ 99.66 % of anomaly decisions are identical.
* **AIMET's contribution is measurable.** Same bitwidths, same converter, same calibration windows: the AIMET version (range priors + AdaRound) beats QNN's own calibration everywhere, most clearly in the worst-case error.
* **W8A8 is not deployable for the 24 h model** (14.7 dB, R² 0.34 → 0.17). It only saves ≈0.7–2 ms. AIMET's simulation (40.6 dB) didn't predict this, because it can't simulate the 8-bit LSTM internals that the HTP uses. That's exactly why rung V4 exists.
* **Latency is not the bottleneck:** ≈3.4–3.9 ms per 24-hour window on the NPU vs a 15-minute data cadence. The report's original "0.092 ms" was off by ~40× (it counted one LSTM step, doc 01).
* **Size:** quantized weights 71–297 KB (vs 236–1,137 KB FP32, ≈3.8× smaller). The model libraries are ≈3.8–4.0 MB because they include the code for the 96 unrolled steps.

## 3. Operational notes
* The laptop **suspended** (09:57 → ~15:28) in the middle of the build, and the host adb connection dropped twice. Disable suspend for long runs. The device scripts retry automatically.
* `qnn-net-run --profiling_level basic` writes the data to `qnn-profiling-data_0.log` (the un-suffixed file is a special file).
* Everything on the device lives in `/data/local/tmp/nz_aimet/`, and nothing else on the QIDK was modified.
