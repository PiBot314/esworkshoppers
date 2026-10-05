# Edge Net-Zero Forecasting and Sensor Anomaly Detection on the Qualcomm QIDK
### Post-training quantization of LSTM forecasters with Qualcomm AIMET, deployed on the Hexagon NPU

---

## 1. Overview
A campus microgrid is monitored by 11 sensor nodes: six solar plants/inverters, two grid/substation meters, and three room sub-meters. Three LSTM models trained on that telemetry forecast campus **solar generation, grid draw and room demand** in 15-minute steps. This project takes those trained models **as they are (no retraining)**, quantizes them with **Qualcomm AIMET**, and runs them on the **Qualcomm Innovators Development Kit (QIDK, Snapdragon 8 Gen 3 / SM8650, Hexagon HTP v75)**.

On top of the forecasts, a **residual-based anomaly check** compares each incoming reading with what the model expected 15 minutes earlier. Large deviations indicate a faulty sensor (dropout, spike) or abnormal generation.

**Result:** all three models run on the NPU in **3.4–3.9 ms** per inference with **8-bit weights and 16-bit activations (W8A16)**. Their outputs match the original FP32 models to **46–52 dB SQNR**. Accuracy (R²) changes by at most 0.004, and **≥ 99.66 %** of anomaly decisions are identical to those of the FP32 models.

---

## 2. Models
All three are stacked 2-layer unidirectional LSTMs followed by a 2-layer MLP head. Input: the last 24 hours (96 × 15-min steps). Static batch size 1.

| Model | Input | Output | Hidden | Parameters | Purpose |
|---|---|---|---|---|---|
| `all_sensors_1h` | [1, 96, 58] | [1, 12] = 4 steps × (Solar, Grid, Demand) | 96 | 139,749 | Real-time dispatch, anomaly check |
| `all_sensors_24h` | [1, 96, 58] | [1, 288] = 96 steps × 3 | 128 | 282,153 | Day-ahead planning |
| `single_node` | [1, 96, 10] | [1, 4] = 4 steps of one solar plant | 64 | 55,021 | Plant-level anomaly check |

**Input features** (order in `release/<model>/model_meta.json`): per-node electrical channels (power, reactive power, voltages, currents, frequency, power factor), a binary `was_imputed` flag per node, and six cyclic time encodings (hour-of-day, day-of-week, day-of-year as sin/cos in Indian Standard Time). Every feature is MinMax-normalized to [0, 1], **including the cyclic encodings**, using each node's own min/max.

**Outputs** are fractions of capacity. Multiply by `capacities_W` in the metadata to get Watts. Solar forecasts are clamped at 0 W.

---

## 3. Anomaly detection method
No extra model is needed. The forecaster itself is the reference:

```
r      = actual(t + 15 min) − forecast(t + 15 min)              per domain, Watts
flag   = | r − median_h | > 4 · 1.4826 · MAD_h                  h = hour of day (IST)
limit  ≥ 2 % of capacity                                         (suppresses night-time noise)
```
Medians and MADs (a robust standard deviation) are estimated per hour of day on validation data, from FP32 residuals. The hour is decoded from the window's own cyclic features, so no timestamps are needed.

Detection on synthetic faults injected into 300 daylight test readings:

| Fault | `all_sensors_1h` (campus total) | `single_node` (one plant) |
|---|---|---|
| Sensor dropout (reading → 0) | 69 % | 92 % |
| Sensor spike (+50 % of capacity) | 91 % | 83 % |
| Partial panel damage (−40 %) | 0 % | 17 % |
| False-flag rate on normal data | 2.3 % | 2.7 % |

Sensor faults are detected well. Partial panel damage is masked by normal forecast error at campus-total level and is only weakly visible per plant. The Demand forecast is not reliable enough for anomaly checks.

---

## 4. Quantization method (Qualcomm AIMET, post-training)
Tool: **aimet-onnx 2.12** (the last release that includes ONNX AutoQuant), HTP v75 quantization-simulation configuration.

1. **Calibration data:** 1,024 windows from the training period, stratified over hour of day (night, dawn, noon, dusk equally represented).
2. **Target format W8A16:** 8-bit weights (LSTM and dense layers), 16-bit activations. On the device: dense weights symmetric 8-bit with 32-bit biases; LSTM weights asymmetric 8-bit with 16-bit biases (expanded-LSTM Add operands). 16-bit activations protect the LSTM state, which is carried through 96 recurrent steps.
3. **LSTM weights treated as parameters.** In the exported graph, the LSTM gate weights pass through a constant reordering subgraph. Its quantizers are explicitly set to 8-bit symmetric so the simulation matches the hardware.
4. **Range priors** applied after calibration:
   * LSTM hidden outputs are mathematically bounded (h = o·tanh(c) ∈ (−1, 1)), so they're fixed to [−1, 1].
   * Model outputs get the calibrated range ± 10 % headroom, because real forecasts over- and undershoot the calibration range.
   * Effect: activation-only (W32A16) fidelity 43.8 → 77.5 dB on the 1 h model, i.e. calibration-range clipping removed.
5. **AutoQuant** (quant-scheme selection → batch-norm folding → cross-layer equalization → **AdaRound**), with the same priors injected into every simulation it builds. AdaRound learns the rounding of the dense-layer weights (10,000 iterations per layer). Batch-norm folding and cross-layer equalization have nothing to act on in an LSTM + MLP.
6. **Selection metric:** SQNR of the quantized outputs vs FP32 on 600 validation windows. Fidelity to FP32 measures what quantization changes, independently of ground-truth quality.

| Validation SQNR vs FP32 (dB) | 1 h | 24 h | single |
|---|---|---|---|
| W8A16, calibration only | 42.4 | 43.3 | 49.9 |
| + range priors | 47.6 | 43.3 | 49.9 |
| **+ AutoQuant / AdaRound (deployed)** | **50.3** | **47.1** | **51.1** |
| W8A8 (for comparison) | 43.3 | 41.2 | 42.4 |

---

## 5. Deployment pipeline
```
FP32 ONNX ─► AIMET (05) ─► model.onnx + model.encodings
                              │  keep encodings of real activations (input, ReLU, output) + dense weights (07)
                              ▼
 qnn-onnx-converter --quantization_overrides … --input_list calib(512)
                    --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32
                    --expand_lstm_op_structure      (LSTM → MatMul/Add/Sigmoid/Tanh/Mul per step)
                    --preserve_io layout            (keep the [1, 96, F] input layout)
                              ▼
 qnn-model-lib-generator -t aarch64-android ─► lib<model>_w8a16_htp.so   (08)
                              ▼
 QIDK: qnn-net-run --backend libQnnHtp.so --model lib….so --input_list …  (09, 12)
```
* SDK: Qualcomm AI Runtime (QAIRT/QNN) **2.47**. The same version's runtime libraries are pushed to the device (`release/qidk_runtime_2.47`: `qnn-net-run`, `libQnnHtp.so`, `libQnnHtpV75Stub.so`, `libQnnHtpV75Skel.so`, `libQnnHtpPrepare.so`, `libQnnSystem.so`, `libQnnCpu.so`, …).
* Tensors not covered by AIMET encodings (the LSTM internals created when the LSTM is expanded per time step) are calibrated by the converter on 512 of the same calibration windows, at 16 bit.
* Input to the device: raw float32 `[1, 96, F]`. Output: raw float32. Quantization and dequantization happen inside the graph.

---

## 6. Results
Held-out test period (Nov 10 – Dec 31 2025; 4,832 / 4,740 / 4,861 windows), measured **on the QIDK**.

| Model | Variant | SQNR vs FP32 | Max error (of capacity) | R² FP32 → device | Same anomaly decisions | NPU time |
|---|---|---|---|---|---|---|
| 1 h | **AIMET W8A16** | **48.4 dB** | **0.45 %** | 0.8522 → 0.8529 | **99.66 %** | **3.42 ms** |
| 1 h | QNN calibration only, W8A16 | 44.9 dB | 2.61 % | 0.8522 → 0.8548 | 99.69 % | 3.41 ms |
| 1 h | AIMET W8A8 | 32.8 dB | 4.02 % | 0.8522 → 0.8500 | 98.37 % | 2.51 ms |
| 24 h | **AIMET W8A16** | **46.4 dB** | **1.26 %** | 0.3388 → 0.3349 | **99.76 %** | **3.44 ms** |
| 24 h | QNN calibration only, W8A16 | 43.8 dB | 1.47 % | 0.3388 → 0.3344 | 99.77 % | 3.43 ms |
| 24 h | AIMET W8A8 | 14.7 dB ✗ | 42 % | 0.3388 → 0.1670 | 89.3 % | 2.79 ms |
| single | **AIMET W8A16** | **52.1 dB** | **0.45 %** | 0.9499 → 0.9498 | **99.92 %** | **3.93 ms** |
| single | QNN calibration only, W8A16 | 46.3 dB | 0.65 % | 0.9499 → 0.9495 | 99.90 % | 3.89 ms |
| single | AIMET W8A8 | 36.2 dB | 3.68 % | 0.9499 → 0.9482 | 99.88 % | 1.84 ms |

R² is the mean of Solar and Grid R² (Solar only for `single_node`). Demand is excluded because the FP32 model's Demand R² is already negative.

Observations:
* **AIMET improves on the converter's own calibration** by 2.6–5.9 dB at identical bitwidths, and reduces the worst-case error by up to 6×.
* **W8A16 is required.** W8A8 saves ≈ 1–2 ms but degrades the day-ahead model severely on hardware, because LSTM internals at 8 bit accumulate error over 96 steps.
* **Latency** (3.4–3.9 ms) is about 230,000× shorter than the 15-minute data interval. For comparison, the FP32 graph on the device CPU manages 32–68 inferences/s including file I/O (≈15–31 ms each).
* **Size:** quantized weights 71 KB / 153 KB / 297 KB (≈ 3.8× smaller than FP32).

Absolute R² values are measured on a reconstruction of the original training data from raw telemetry. They're within 0.007–0.03 of the originally reported FP32 accuracy (Solar R² 0.943 / 0.798 / 0.950). Comparisons between quantized and FP32 models use identical inputs and are exact.

---

## 7. Verification
| Check | Result |
|---|---|
| Rebuilt data reproduces the reported FP32 accuracy | Solar R² within 0.012, 1 h Grid within 0.032 |
| AIMET's graph equals the original (quantizers off) | max difference 1.2 × 10⁻⁷ |
| Quantized simulation ≥ 30 dB vs FP32 | 45.8–51.8 dB on test data |
| Converted FP32 graph on the device CPU equals ONNX | **bit-exact** on all test windows |
| Quantized NPU output agrees with the AIMET simulation | 45–55 dB |
| AIMET vs converter-only quantization on the NPU | AIMET better on all models |
| End-to-end demo (fault injected → flagged on device) | ✓ |

---

## 8. Using the deployed model
```bash
cd mideval/aimet_quantization
../.venv-aimet/bin/python scripts/12_qidk_forecast.py --model all_sensors_1h --test-index 10
../.venv-aimet/bin/python scripts/12_qidk_forecast.py --model all_sensors_1h --test-index 10 --inject sensor_dropout_zero
../.venv-aimet/bin/python scripts/12_qidk_forecast.py --model single_node --input window_1x96x10.npy
```
Direct use on the device (any app or shell):
```bash
D=/data/local/tmp/nz_aimet
adb push release/all_sensors_1h/liball_sensors_1h_w8a16_htp.so $D/
adb push window.raw $D/                       # float32 [1, 96, 58], normalized features
adb shell "cd $D && echo window.raw > list.txt && LD_LIBRARY_PATH=$D/lib ADSP_LIBRARY_PATH=$D/lib \
  ./bin/qnn-net-run --backend lib/libQnnHtp.so --model liball_sensors_1h_w8a16_htp.so --input_list list.txt --output_dir out"
adb pull $D/out/Result_0/forecast_all_pillars.raw   # float32 [12] → reshape (4, 3) × capacities_W
```

---

## 9. Live demo dashboard
A browser dashboard **served by the QIDK itself** for live demonstrations:
* **Server:** `nz_dashboard`, a 1.9 MB native C++ binary (cpp-httplib, Android NDK) running on the QIDK. It serves the web page and `POST /api/run`, which executes a submitted window with `qnn-net-run` on the **Hexagon NPU** (W8A16) or the CPU (FP32 reference) and returns the forecast with timings.
* **Views:** campus replay of the test period (actual vs NPU forecast with a shaded "typical" band; readings outside it are flagged), residual vs alarm limit, plant-level model, day-ahead 24 h forecast with energy totals and a typical-day envelope, verification summary.
* **Interactive tests:** inject sensor dropout / spike / stuck value / panel damage into the replay; download any input window as CSV, edit it or upload your own sensor readings (96 rows at 15-min steps, physical units), and **run it live on the NPU**. The next reading (row 97) is checked for anomalies.
* **Timing of a live run:** ≈4.4–5.6 ms of NPU execution, ≈1.5–2 s end-to-end, because `qnn-net-run` prepares the graph on each launch. A persistent QNN context would remove the preparation time.
* **Access:** USB (`adb forward`, `http://localhost:8080`), Wi-Fi (`http://<QIDK-IP>:8080`), or Chrome on the QIDK's own screen. Start/stop: `scripts/14_dashboard_start.sh [--stop]`.
* **Verified:** live outputs equal the recorded device outputs (≤ 6×10⁻⁷); a CSV round trip reproduces the model input to 1.5×10⁻⁸ (time features recomputed from timestamps to 3×10⁻¹⁸); fault injection flags as expected; no browser console errors.

## 10. Reproducing everything
Environment:
* **AIMET:** Python 3.10 venv: `aimet-onnx==2.12.0 onnxruntime==1.19.2 onnx==1.16.2 "numpy<2" "protobuf>=3.20.2,<5" torch pandas pyarrow scikit-learn` (CPU is sufficient).
* **QAIRT 2.47** (Docker image from `qidk/Tools/qairt_docker`, with `protobuf==3.20.3` in its Python env: `docker/Dockerfile.qairt-fixed`), Android NDK r26c.
* QIDK over ADB, with `release/qidk_runtime_2.47` pushed to `/data/local/tmp/nz_aimet`.

```bash
cd mideval/aimet_quantization; PY=../.venv-aimet/bin/python
$PY scripts/00_smoke_test_aimet.py          # AIMET loads/simulates the models
$PY scripts/01_rebuild_dataset.py           # raw telemetry → normalized dataset
$PY scripts/02_build_windows.py             # windows, calibration set, fidelity report
$PY scripts/03_fp32_baseline.py             # FP32 reference outputs and metrics
$PY scripts/04_anomaly_baseline.py          # anomaly thresholds
$PY scripts/05_aimet_quantize.py            # AIMET grid + AutoQuant + export   (~7 min)
$PY scripts/06_evaluate_sim.py              # test-set evaluation of the exports
$PY scripts/07_prepare_raw.py && $PY scripts/07_prepare_raw.py --encodings
scripts/08_qairt_build.sh all_sensors_1h all_sensors_24h single_node    # model libraries (~10 min)
for m in all_sensors_1h all_sensors_24h single_node; do
  scripts/09_run_on_qidk.sh $m && $PY scripts/10_compare_device.py $m cpu_fp32 htp_aimet_w8a16 htp_qairt_w8a16 htp_aimet_w8a8
  scripts/11_benchmark_latency.sh $m
done
$PY scripts/13_dashboard_pack.py && dashboard/build.sh && scripts/14_dashboard_start.sh     # demo dashboard
```

---

## 11. Repository layout
```
aimet_quantization/
  README.md, README_DETAILED.md, TECHNICAL_QA.md     summaries
  00–15_*.md                                        step-by-step engineering log
  scripts/                                          pipeline 00–12 (+ common.py, anomaly.py)
  configs/                                          AIMET HTP v75 configs (LSTM per-tensor variants)
  docker/Dockerfile.qairt-fixed                     QAIRT build image
  release/                                          deliverables per model + device runtime
  dashboard/                                        demo server (C++), web page, build script, screenshots
  data/  out/                                       generated datasets, results, device outputs
```

## 12. Limitations
* Accuracy figures use a reconstruction of the training data (not the original files). Quantization comparisons are unaffected.
* Partial panel damage isn't detectable from campus-level forecasts. Plant-level forecasting models would be needed.
* The Demand forecast head is not accurate enough for anomaly detection.
* Dense layers use per-tensor weight scales on the NPU. Per-channel scales simulate better (+1.5–3 dB) but aren't used in this deployment path.
