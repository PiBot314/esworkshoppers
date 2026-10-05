# Campus Net-Zero Forecasting & Sensor Anomaly Detection on the Qualcomm QIDK
### Full project explanation, from raw sensor data to a live dashboard served by the Hexagon NPU

*State as of 2026-10-05. Every number here comes from the project's own logs and result files (`aimet_quantization/out/*.json`, docs `00`–`17`). Links point to the source of each claim.*

---

## 0. The whole project in 60 seconds

* **What it does:** a campus microgrid has **11 sensor nodes** (6 solar, 2 grid/substation, 3 room sub-meters). Three **LSTM forecasters** read the last **24 h** of their data (96 steps × 15 min) and predict **Solar generation, Grid draw and room Demand** for the next **1 h** or **24 h**. A **residual check** compares each new reading with what the model expected. Readings that are too far off are flagged as **anomalies** (e.g. a dead or spiking sensor).
* **Where it runs:** on the **Qualcomm Innovators Development Kit (QIDK)**, a Snapdragon 8 Gen 3 (SM8650) board, on its **Hexagon NPU (HTP v75)**.
* **How it got there:** the trained FP32 models were **not retrained**. They were **post-training quantized** with **Qualcomm AIMET** to **W8A16** (8-bit weights, 16-bit activations), converted with **Qualcomm QNN/QAIRT 2.47**, and compiled into Android model libraries.
* **How good it is (measured on the QIDK, held-out test period):**

| Model | What it predicts | NPU time | Fidelity to FP32 | R² FP32 → NPU | Same anomaly decisions |
|---|---|---|---|---|---|
| `all_sensors_1h` | next 1 h of campus Solar/Grid/Demand | **3.42 ms** | **48.4 dB** | 0.852 → 0.853 | 99.66 % |
| `all_sensors_24h` | next 24 h of campus Solar/Grid/Demand | **3.44 ms** | **46.4 dB** | 0.339 → 0.335 | 99.76 % |
| `single_node` | next 1 h of one solar plant (Palash) | **3.93 ms** | **52.1 dB** | 0.950 → 0.950 | 99.92 % |

* **How it's shown:** a **web dashboard served by the QIDK itself** (a native C++ server on the board). It replays the test period with the NPU's forecasts, lets you inject sensor faults and watch them get flagged, and runs **any input window live on the NPU** on request.

---

## 1. Problem and setting

### 1.1 The "net-zero" framing (from the original project, `model2/`)
The campus wants to know how close it is to **net-zero**: generating as much clean energy on-site as it consumes.

```
Net deficit (W)            = Demand − Solar
Solar self-sufficiency (%) = Solar / Demand × 100
Solar ≥ Demand  → surplus (charge batteries / export)
Solar <  Demand → deficit (import from the grid)
```
Forecasts serve two time scales:
* **1 h ahead (4 × 15 min):** real-time dispatch, battery ramp smoothing, cloud-transient buffering. **This is also the model used for anomaly checks.**
* **24 h ahead (96 × 15 min):** day-ahead grid procurement, battery charge scheduling, HVAC pre-cooling, peak-tariff shaving.

### 1.2 Why "edge" and why the QIDK
* Data arrives every **15 minutes** from devices on campus. Running the model on a local board avoids a cloud round-trip and keeps telemetry on-site.
* The course target is the **QIDK**: Snapdragon 8 Gen 3 (SM8650), Android 14, aarch64 CPU, **Hexagon Tensor Processor (HTP) v75** NPU. ADB serial of our board: `d42f06b`. **There is no Python on the device**, which shaped every deployment decision.
* NPUs run integer arithmetic fast and efficiently, so the FP32 models had to be **quantized** to run there.

---

## 2. Project history (who did what, in what order)

| Phase | When | What | Where |
|---|---|---|---|
| **A. Data + training** (team, original project) | up to 2026-09-30 | Cleaned the raw telemetry, built 58 features, trained 3 LSTMs in PyTorch, exported static ONNX (batch 1, opset 18), benchmarked vs naive baselines, wrote reports | [model2/](model2/), [model2/report.md](model2/report.md), [model2/progress.txt](model2/progress.txt) |
| **B. Audit + plan** | 2026-10-05 | Surveyed the code, identified the model type, audited the tools, found 4 blockers and 8 inconsistencies, wrote the plan | docs [00](aimet_quantization/00_README.md)–[09](aimet_quantization/09_action_plan_risks_next_steps.md) |
| **C. Quantization + deployment** (P0–P5) | 2026-10-05 | Built the AIMET environment, rebuilt the missing dataset, made the anomaly layer, quantized with AIMET, converted, ran on the QIDK, verified | docs [10](aimet_quantization/10_P0_environment_log.md)–[15](aimet_quantization/15_summary_lessons_and_next_steps.md), `aimet_quantization/scripts/00–12` |
| **D. Demo dashboard** | 2026-10-05 | Web dashboard served by the QIDK, with live NPU runs | docs [16](aimet_quantization/16_dashboard_plan.md)–[17](aimet_quantization/17_dashboard_build_log.md), [aimet_quantization/dashboard/](aimet_quantization/dashboard/) |

Ground rules that held throughout:
* **No retraining.** The original ONNX files in `model2/qidk_deployment/` are frozen inputs and were never modified. `model2/` is unchanged.
* On the device, everything lives in **`/data/local/tmp/nz_aimet/`**. Other folders on the board (another team project's `campus_energy/`, `qairt-2.37.*`) were not touched.
* Out of scope: the sibling `../project/` autoencoder anomaly detector (already on the QIDK as an SNPE INT8 DLC).

---

## 3. Data

### 3.1 The 11 sensor nodes

| Node | Family | Role | Channels used (pre-normalization) |
|---|---|---|---|
| SL-PL00-00 | Solar plant meter | **Palash** solar plant | active power, reactive power, avg voltage, neutral current |
| SL-VN95-00 | Solar plant meter | Vindhya solar plant | same 4 |
| SL-NI01-00 | Solar plant meter | Nilgiri solar plant | same 4 |
| SL-NI03-00 / SL-NI03-01 | Solar inverter | Nilgiri inverters | active power, AC voltage, DC PV-string voltage |
| SL-VN02-01 | Solar inverter | Vindhya inverter | same 3 |
| EM-CR-SB00-00 | Substation incomer | Main substation (grid) | grid power, transformer T1/T2 power, HV & LT1/LT2 voltages |
| EM-CR-KH95-00 | Energy meter | Control-room meter | total power, phase R & B power, voltage, power factor |
| SR-EM-KH04-00 | Room meter | Lab main incomer | power |
| SR-EM-KH04-01 / -02 | Room sub-meters | Lab branches | power, voltage, current, frequency |

### 3.2 The 58 input features
**58 = 41 electrical channels + 11 `was_imputed` flags (one per node) + 6 cyclic time features.**
* `was_imputed` (0/1) marks values that were filled in rather than measured, so the network can tell synthetic fills from real telemetry.
* Cyclic time features, all computed in **Indian Standard Time (UTC+5:30)**: `sin/cos(2π·hour/24)`, `sin/cos(2π·weekday/7)`, `sin/cos(2π·day-of-year/365.25)`.
* **Every feature except `was_imputed` is MinMax-scaled to [0, 1]** with its own node's min/max, **including the sin/cos features** (so in practice they're `(sin+1)/2`). This was discovered during the rebuild (doc 11, finding 1) and is the most common integration mistake: feeding raw sin/cos in [−1, 1] gives wrong forecasts.
* The single-node model uses **10 features**: Palash's 4 electrical channels + the 6 cyclic features (no `was_imputed`).

### 3.3 Targets (what the models predict)
| Domain | Definition | Capacity used for scaling |
|---|---|---|
| **Solar** | Σ `active_power_W` of the 6 SL nodes | 546,010 W (model) |
| **Grid** | SB00 `gd_power_W` + KH95 `power_total_W` | 887,251 W |
| **Demand** | Σ `power_W` of the 3 SR-EM room meters | 30,046 W |
| Solar (single node) | Palash `active_power_W` | 65,808 W |

Outputs are **fractions of capacity**. Multiply by `capacities_W` to get Watts. Output layout: `reshape(H, 3)`, index `[step, domain]`. Solar is clamped at ≥ 0 W for display (models can output slightly negative night-time solar).

### 3.4 Original preprocessing (team, `model2/generate_lstm_dataset.py` + a missing first stage)
Fixes for raw-data problems: kW/W unit mix → all Watts; negative power from reversed CT clamps → `abs()`; irregular cadence (10 s – 5 min) → **15-min mean resampling**; gaps → forward-fill (limited) + `was_imputed`; heterogeneous column names → unified schema; `fillna(0)`; MinMax to [0, 1]; scaler parameters saved to `scaler_params.json`.

Aligned timeline: **32,866 intervals, 2025-01-23 04:15 → 2025-12-31 12:30 (UTC index).** Split **chronologically 70 / 15 / 15** (train / validation / test). **Test period: 2025-11-10 → 2025-12-31.**

### 3.5 The data rebuild (why it was needed, what was found)
The normalized dataset (`lstm_normalized/`), its intermediate (`lstm_ready/`) and the script that built the intermediate were **not on this machine** (they're on a teammate's). Without data there's nothing to calibrate or evaluate with, so it was **rebuilt from raw** `../nodes/parquet_sorted/` ([doc 11](aimet_quantization/11_P1_data_rebuild_log.md)). Only the missing first stage was reconstructed. The second stage is the team's **original, unmodified** `generate_lstm_dataset.build_normalized_dataset()`, imported rather than copied.

Reverse-engineering findings, each checked against the data:
1. Cyclic features are MinMax-scaled too (see 3.2).
2. The test sample window starts **2025-11-10 04:00 UTC** (09:30 IST, Monday). This confirms a UTC index with IST cyclic features.
3. The parquet index is **server-ingest time**. For SB00 it was off by a median of 105 days (back-filled uploads) → use `Sensor_Timestamp`.
4. KH95 in `parquet_sorted` had the wrong (substation) schema → taken from `raw_data/EM-Kh95-00.csv`; `max(R+Y+B) = 1,739.4 W` matches exactly.
5. **SL-VN95-00 had active/reactive registers swapped** → swapped back (solar capacity 385.9 → 535.8 kW, target 546.0).
6. SL-NI01-00 reports Watts, not kW.
7. Inverters stop transmitting at dusk. The original carried the last AC reading through the night (with `was_imputed=1`) but not the DC string voltage.
8. One spurious leading 04:00 bin dropped → original split boundaries reproduced.
9. Feature order reordered to the deployed model's `feature_names`.

How close the rebuild is:
| Check | Result |
|---|---|
| Timeline (32,866 bins, same start/end), test start | **identical** |
| `grid_max_W`, `demand_max_W` | **exact** |
| `solar_max_W` | 535,813 vs 546,010 (−1.9 %) |
| Reference window features within 0.01 | 44 / 58 (differences are in auxiliary voltage/current channels, not power targets) |
| FP32 Solar R² vs originally reported | within **0.007–0.012** |

**Key decision (the "approximate rebuild"):** quantization loss is always measured **quantized vs FP32 on the same rebuilt windows**. That comparison needs no labels and isn't biased by small normalization differences, because they affect both models equally. Absolute R² on rebuilt data is reported but labelled "approximate data".

Windows produced:
| Model | train / val / test windows | calibration windows |
|---|---|---|
| all_sensors_1h | 22,907 / 4,830 / **4,832** | 1,024 |
| all_sensors_24h | 22,815 / 4,738 / **4,740** | 1,024 |
| single_node | 23,041 / 4,859 / **4,861** | 1,024 |

Calibration windows come from **train only**, stratified over the IST hour of the window's last step (night, dawn, noon, dusk equally represented), seed 0.

---

## 4. The models

### 4.1 Architecture (read from the ONNX graphs)
All three are **multi-step, multi-output regression forecasters**. They are **not** classifiers or anomaly detectors: anomaly detection is a separate post-processing layer (section 5).

```
input [1, 96, F]   (past 24 h, F features in [0,1])
  → LSTM(hidden H) → LSTM(hidden H)     2 stacked unidirectional layers, zero initial state, dropout 0.1 between (training only)
  → last time step → Linear(H→M) → ReLU → Linear(M→O)
output [1, O]      (fractions of capacity)
```

| Model | ONNX file | F | H | M | O | Params | Epochs (train) |
|---|---|---|---|---|---|---|---|
| `all_sensors_1h` | `all_sensors_lstm_qidk.onnx` | 58 | 96 | 48 | 12 = 4 steps × 3 | 139,749 | 12 |
| `all_sensors_24h` | `all_sensors_24h_lstm_qidk.onnx` | 58 | 128 | 128 | 288 = 96 × 3 | 282,153 | 10 |
| `single_node` | `SL-PL00-00_lstm_qidk.onnx` | 10 | 64 | 32 | 4 | 55,021 | 15 |

Training (team): MSE loss, Adam (lr 0.002), batch 64, ReduceLROnPlateau, best-validation checkpoint, chronological 70/15/15 split. Export: `torch.onnx.export`, opset 18, static batch 1, ONNX-Runtime parity < 2.4 × 10⁻⁷.

Why each model exists: **1 h** = real-time decisions and anomaly checks; **24 h** = day-ahead planning; **single node** = the only model that can point at *which* plant misbehaves (campus totals can't).

### 4.2 Accuracy (FP32)
| Model / domain | R² reported by team | R² on rebuilt data | MAE (rebuilt) |
|---|---|---|---|
| 1 h Solar | 0.955 | **0.943** | 23.3 kW (4.3 % of cap.) |
| 1 h Grid | 0.794 | **0.762** | 36.3 kW |
| 1 h Demand | −0.341 | −0.658 | 2.2 kW |
| 24 h Solar | 0.805 | **0.798** | 42.1 kW |
| 24 h Grid | 0.741 (*suspect*) | −0.120 | 73.1 kW |
| single node Solar | 0.957 | **0.950** | 1.96 kW |

* The team's 1 h LSTM beat three naive baselines (persistence, 4-step moving average, 24 h lag), e.g. Solar R² 0.955 vs persistence 0.908.
* **Headline R²** used throughout = mean(Solar R², Grid R²) for the campus models, Solar R² for the single node. **Demand is excluded**: its R² is negative even in FP32 (worse than predicting the mean).
* **24 h Grid "0.741" is almost certainly a reporting error:** it equals the *persistence baseline's* Grid R² in `report.md`, and it doesn't fit the model's own MAE (74.3 kW, which our 73.1 kW matches).

### 4.3 Inconsistencies found in the original project (doc 01)
| # | Finding | Consequence |
|---|---|---|
| 1 | Models are forecasters, not anomaly detectors | Anomaly detection built as a residual layer |
| 2 | Normalized data missing | Rebuilt from raw |
| 3 | `qidk_deploy.sh` runs `python3` on the device; the QIDK has no Python or ONNX Runtime | Deployment rewritten around native `qnn-net-run` |
| 4 | Report claimed "141,696 MACs, ≈0.092 ms" | That's one LSTM time step; the real graph is ≈12.8 M MACs. Measured latency is 3.4 ms |
| 5 | 24 h Grid MAE: report 34.9 kW vs metadata 74.3 kW | Metadata used |
| 6 | 1 h Demand R² = −0.34 | Demand excluded from objectives |
| 7 | `solar_max_W` differs slightly between 1 h and 24 h metadata | Models trained on slightly different data builds |
| 8 | Runner prints `net = grid + solar − demand`, reports define Net = Demand − Solar | Cosmetic, not copied |

---

## 5. Anomaly detection (no extra model)

### 5.1 The rule
The 1 h forecaster is the reference. 15 minutes before a reading arrives, the model has already predicted it.

```
r       = actual(t+15 min) − forecast(t+15 min)            per domain, in Watts
flag   ⇔  | r − median_h | > limit_h
limit_h = max( 4 · 1.4826 · MAD_h ,  2 % of capacity )     h = IST hour of the reading
```
* **median_h / MAD_h**: per hour of day, from **FP32 residuals on the validation split**. MAD (median absolute deviation) × 1.4826 is a **robust standard deviation**: outliers don't inflate it.
* **Centring on the median** removes systematic forecast bias (first version without it flagged 48 % of Demand points).
* **Per hour** because forecasting noon solar is much harder than midnight solar. One global threshold would be too loose at night and too tight at noon.
* **K = 4** (lowered from 5: dropout detection 32 % → 69 % at ≈2 % false flags).
* **2 % floor** stops near-zero night-time noise from raising alarms.
* The hour is **decoded from the window's own `hod_sin/cos` features**, so the detector needs no timestamps. Thresholds are stored in `release/<model>/anomaly_thresholds_W.npy` (shape `[2, 24, D]`: medians, limits).

### 5.2 What it catches
Synthetic faults injected into 300 daylight test readings (solar actual only):

| Fault | 1 h (campus total) | 24 h (campus total) | single node (one plant) |
|---|---|---|---|
| Sensor dropout (reading → 0) | 69 % | 8 % | **92 %** |
| Sensor spike (+50 % capacity) | **91 %** | 30 % | 83 % |
| Partial panel damage (−40 %) | 0 % | 0 % | 17 % |
| False flags on normal data (Solar / Grid) | 2.3 % / 0.9 % | 1.1 % / 0.2 % | 2.7 % |

Honest reading:
* **Sensor faults (dropout, spike) are detected well** by the 1 h and single-node models.
* **Partial panel damage is invisible at campus level**: −40 % of one plant is within normal forecast error of the campus total. The single-plant model does a bit better (17 %) because the signal isn't diluted. Reliable damage detection would need per-plant models and multi-day residual aggregation, which means training (out of scope).
* The 24 h model's first step is a poor short-term detector: it optimizes the whole day.
* **Demand can't be used** (49 % natural flag rate), so the dashboard shows Demand verdicts as **"n/a"**.

Role during quantization: a **behavioural check**. The quantized models must raise the same flags as FP32.

---

## 6. Quantization with AIMET

### 6.1 Concepts in one page
* **Quantization** stores and computes numbers as small integers plus a scale: `real ≈ scale × (int − offset)`. 8-bit = 256 levels, 16-bit = 65,536 levels. Values outside the chosen range are **clipped**; values inside are **rounded**.
* **W8A16** = weights 8-bit, activations (intermediate tensors) 16-bit. The usual sweet spot for LSTMs on Hexagon: weights shrink ≈4×, and the recurrent state, carried through 96 steps, keeps enough precision.
* **Post-training quantization (PTQ)** = quantize a trained model using only a small **calibration** set to pick ranges. No retraining. (The alternative, quantization-aware training, retrains.)
* **Per-tensor vs per-channel**: one scale for a whole weight matrix vs one per output row. Per-channel is usually more accurate.
* **SQNR (signal-to-quantization-noise ratio)** = `10·log10( Σ fp32² / Σ (fp32 − quant)² )` on identical inputs. Higher is better. **30 dB ≈ 3 % RMS error, 40 dB ≈ 1 %, 48 dB ≈ 0.4 %.** This was the project's main metric: it measures *what quantization changed*, independent of how good the model was to begin with.
* **AIMET** (AI Model Efficiency Toolkit, Qualcomm) simulates quantization on the host ("QuantSim"), picks ranges, and optimizes. Its exports (`model.onnx` + `model.encodings` = scales/offsets) are handed to Qualcomm's converter.
* **AdaRound** = instead of rounding each weight to the nearest integer, learn whether to round up or down so that the layer's *output* error is minimal.
* **AutoQuant** = AIMET's automatic pipeline: quant-scheme selection → batch-norm folding → cross-layer equalization (CLE) → AdaRound; keeps the best by your metric.

### 6.2 Environment (P0, [doc 10](aimet_quantization/10_P0_environment_log.md))
| Blocker found | Fix |
|---|---|
| **B1** The QAIRT Docker image's ONNX converter crashed (`protobuf 3.19` too old for `onnx 1.16`) | Derived image `qairt-fixed:2.47` with `protobuf==3.20.3` ([docker/Dockerfile.qairt-fixed](aimet_quantization/docker/Dockerfile.qairt-fixed)) |
| **B2** No AIMET; ONNX AutoQuant only exists up to `aimet-onnx 2.12.0` (Python 3.10 only) | `uv` venv `mideval/.venv-aimet`: aimet-onnx 2.12.0, onnxruntime 1.19.2, onnx 1.16.2, numpy < 2, protobuf < 5, CPU torch |
| **B3** No data | Rebuild (section 3.5) |
| **B4** No Python on device; device runtimes 2.37 vs host SDK 2.47 | Native `qnn-net-run` flow; QAIRT **2.47** runtime pushed to a new folder `/data/local/tmp/nz_aimet` |

Gotchas: AutoQuant in the 2.12 wheel must be imported from `aimet_onnx._deprecated.auto_quant_v2`; per-channel quantization crashed on a constant-folded LSTM weight, so a custom config [configs/htp_v75_lstm_per_tensor.json](aimet_quantization/configs/htp_v75_lstm_per_tensor.json) (stock HTP v75 + LSTM per-tensor) is used, which also matches what the converter does.

Sanity check **V1**: AIMET's rewritten graph with all quantizers off equals ONNX Runtime to **1.2 × 10⁻⁷**.

### 6.3 What happened, step by step (P3, [doc 13](aimet_quantization/13_P3_aimet_quantization_log.md))
1. **First grid, a surprise.** On the 1 h model, *16-bit activations alone* (weights still float) lost up to 0.037 of capacity (~20 kW). Enabling quantizers one at a time pinned it on the **model output** and the **LSTM hidden outputs**.
2. **Diagnosis: range clipping, not resolution.** Calibration windows produced outputs in [−0.022, 0.987], but validation forecasts reach −0.068 (negative night solar). LSTM hidden outputs were calibrated to ≈[−0.7, 0.7] but can reach ±1.
3. **Fix: range priors** (domain knowledge applied after calibration, then frozen):
   * **LSTM hidden outputs → [−1, 1]**, a mathematical bound: `h = o · tanh(c)` with `o ∈ (0,1)`, `tanh ∈ (−1,1)`.
   * **Model outputs → calibrated range ± 10 %** headroom.
   * Effect on the 1 h model: activation-only fidelity **43.8 → 77.5 dB**. With clipping gone, 16-bit activations are near-lossless and **8-bit weights become the bottleneck**, which is what AdaRound targets.
4. **LSTM weights weren't really 8-bit.** PyTorch's exporter reorders LSTM gates with Slice/Concat, so the weights reached the LSTM *through ops*, and AIMET treated them as 16-bit *activations*. Fixed by walking back from each LSTM's weight inputs and forcing those quantizers to 8-bit symmetric. The "W8A16" label is now honest (≈95 % of parameters are LSTM weights).
5. **Per-channel dense weights mis-executed on the NPU** in this deployment path (found in P4, see 7.2 #6), so AutoQuant was re-run with **per-tensor** scales, letting AdaRound optimize the scales actually deployed.
6. **AutoQuant run** (target 60 dB, deliberately above every plain 8-bit result so all stages run):
   * scheme selection → min-max ("tf");
   * batch-norm folding and CLE → **no-ops** (no batch norm, no conv chains in an LSTM + MLP);
   * **AdaRound** (10,000 iterations/layer, 1,024 windows) → the gain.
   * A 2.12 bug (`num_iterations=None`) was worked around by passing explicit AdaRound parameters. A subclass `AutoQuantWithPriors` injects the range priors into every simulation AutoQuant builds.

Final validation SQNR vs FP32 (600 validation windows):
| Config | 1 h | 24 h | single |
|---|---|---|---|
| W32A16 + priors (activation ceiling) | 77.5 | 80.7 | 63.0 |
| W8A16, calibration only | 42.4 | 43.3 | 49.9 |
| W8A16 + range priors | 47.6 | 43.3 | 49.9 |
| **+ AutoQuant / AdaRound (deployed)** | **50.3** | **47.1** | **51.1** |
| W8A8 (comparison) | 43.3 | 41.2 | 42.4 |

**Where the gains came from:** range priors (our addition) + AdaRound. Not from BN-folding or CLE.

Exports per model: `out/<model>/final/model.{onnx,encodings}` → copied to `release/<model>/<model>_aimet_w8a16.{onnx,encodings}`.

---

## 7. Conversion and deployment to the QIDK (P4, [doc 14](aimet_quantization/14_P4_qidk_deployment_log.md))

### 7.1 The working path
```
AIMET export (model.onnx + model.encodings)
  └─ 07_prepare_raw.py --encodings: keep only real activations (input, ReLU, output) + dense-weight encodings
qnn-onnx-converter  --quantization_overrides model.qairt.encodings   (AIMET's ranges win)
                    --input_list calib(512)                           (calibrates everything AIMET didn't cover)
                    --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32
                    --expand_lstm_op_structure                        (LSTM → MatMul/Add/Sigmoid/Tanh/Mul per step)
                    --preserve_io layout                              (keep [1, 96, F] input)
  └─ qnn-model-lib-generator -t aarch64-android → lib<model>_w8a16_htp.so  (~3.8–4.0 MB)
adb push → qnn-net-run --backend libQnnHtp.so --model lib….so --input_list … → adb pull → compare
```
Input: raw float32 `[1, 96, F]`. Output: raw float32. Quantize/dequantize happens inside the graph. On the device the final graph has 8-bit weights, **all activations at 16-bit** (3,853 tensors in the 1 h model), 16-bit LSTM biases, 32-bit dense biases.

### 7.2 The seven failures that shaped it
| # | Tried | Symptom | Cause → fix |
|---|---|---|---|
| 1 | Planned `qairt-converter` → `qairt-quantizer` | "bitwidth conversion 8 → 32 not supported"; LSTM steps fall back to float | Unrolled LSTM internals had no encodings → use one-step `qnn-onnx-converter` (overrides **+** calibration list); drop encodings on constant-folded tensors |
| 2 | Quantizer in container | can't create `./output/` | cwd not writable as non-root → `cd` to output dir |
| 3 | Fused quantized `Lstm` op | `MODEL_GRAPH_OP_VALIDATION_ERROR` on device; recurrent state 8-bit despite flags | → `--expand_lstm_op_structure` |
| 4 | Expanded graph | **4.9 dB** (garbage) | Converter re-laid input to `[1, 58, 96]` → `--preserve_io layout` |
| 5 | (same) | 14.6 dB; per-step hidden tensors exactly **×1.44** too large | Overriding LSTM-hidden ranges where per-step views alias the tensor → don't pass those priors to QNN (kept in the AIMET simulation only) |
| 6 | (same) | still 14.6 dB for AIMET W8A16 vs 44.9 dB converter-only | Only difference: **per-channel dense weights** → removing them gave 46.2 dB → re-run AutoQuant per-tensor |
| 7 | Long device runs | `adb: device not found` | Host adb drops (laptop suspended mid-build) → retry wrapper in all device scripts |

**Debugging method:** `qnn-net-run --debug` dumps every intermediate tensor. One window was run on HTP-INT and CPU-FP32 and compared tensor by tensor, per LSTM step. That showed the recurrence itself is stable (gate pre-activations 37–45 dB across all 96 steps) and located the broken tensors in #4–#6.

### 7.3 Results on the device (held-out test period, all windows)
| Model | Variant | SQNR vs FP32 | Max error (% cap.) | R² FP32 → device | Flag agreement natural / injected | NPU (ms) | QNN API (ms) | inf/s incl. I/O |
|---|---|---|---|---|---|---|---|---|
| 1 h | CPU FP32 | bit-exact | 0 | 0.8522 → 0.8522 | 100 / 100 % | – | – | 60 |
| 1 h | **HTP AIMET W8A16** | **48.4 dB** | **0.45 %** | 0.8522 → **0.8529** | **99.66 / 100 %** | **3.42** | 5.11 | 169 |
| 1 h | HTP converter-only W8A16 | 44.9 dB | 2.61 % | 0.8522 → 0.8548 | 99.69 / 100 % | 3.41 | 5.08 | 170 |
| 1 h | HTP AIMET W8A8 | 32.8 dB | 4.02 % | 0.8522 → 0.8500 | 98.37 / 99.75 % | 2.51 | 4.15 | 208 |
| 24 h | CPU FP32 | bit-exact | 0 | 0.3388 → 0.3388 | 100 / 100 % | – | – | 32 |
| 24 h | **HTP AIMET W8A16** | **46.4 dB** | **1.26 %** | 0.3388 → **0.3349** | **99.76 / 99.96 %** | **3.44** | 5.11 | 170 |
| 24 h | HTP converter-only W8A16 | 43.8 dB | 1.47 % | 0.3388 → 0.3344 | 99.77 / 99.98 % | 3.43 | 5.11 | 169 |
| 24 h | HTP AIMET W8A8 | **14.7 dB ✗** | 42 % | 0.3388 → 0.1670 | 89.27 / 94.89 % | 2.79 | 4.38 | 197 |
| single | CPU FP32 | bit-exact | 0 | 0.9499 → 0.9499 | 100 / 100 % | – | – | 68 |
| single | **HTP AIMET W8A16** | **52.1 dB** | **0.45 %** | 0.9499 → **0.9498** | **99.92 / 99.92 %** | **3.93** | 5.48 | 165 |
| single | HTP converter-only W8A16 | 46.3 dB | 0.65 % | 0.9499 → 0.9495 | 99.90 / 99.88 % | 3.89 | 5.44 | 165 |
| single | HTP AIMET W8A8 | 36.2 dB | 3.68 % | 0.9499 → 0.9482 | 99.88 / 99.84 % | 1.84 | 3.27 | 264 |

"NPU" = HTP accelerator execute time (mean of 300 runs, `--perf_profile burst`); "QNN API" includes the CPU↔DSP RPC.

What the table says:
* **W8A16 is the right deliverable.** 46–52 dB, |ΔR²| ≤ 0.004, ≥ 99.66 % identical anomaly decisions on all three models.
* **AIMET measurably beats the converter's own calibration**: same bitwidths, converter and calibration data, **+3.5 / +2.6 / +5.9 dB**; worst-case error on the 1 h model **6× lower** (0.45 % vs 2.6 % of capacity).
* **W8A8 isn't deployable for the 24 h model** (14.7 dB, R² halves) to save ≈0.7 ms. AIMET's simulation (40.6 dB) didn't predict this: AIMET-ONNX treats `LSTM` as one op and computes gates in float, while the NPU runs all 96 unrolled steps with 8-bit internals and the error compounds.
* **Latency is a non-issue**: 3.4–3.9 ms vs a 15-minute data cadence (>200,000× headroom). The original "0.092 ms" estimate was ≈40× too optimistic.
* **Size**: quantized weights 153 KB / 297 KB / 71 KB (≈3.8× smaller than FP32). The `.so` libraries are ~3.8–4 MB because they contain graph-construction code for 96 unrolled steps.

---

## 8. Verification ladder (how we know it's right)

Each rung isolates one possible source of error. All use the same held-out test windows.

| Rung | Question | Result |
|---|---|---|
| **V0** | Does the rebuilt data reproduce the team's FP32 accuracy? | ✅ Solar R² within 0.007–0.012; 1 h Grid within 0.03 |
| **V1** | Does AIMET's instrumented graph equal the original? | ✅ 1.2 × 10⁻⁷ |
| **V2** | Is the quantized *simulation* close to FP32 (≥ 30 dB)? | ✅ 45.8–51.8 dB on test |
| **V3** | Is the *conversion* lossless? (FP32 graph on the device CPU vs ONNX) | ✅ **bit-exact**, all 3 models |
| **V4** | Does the NPU match the AIMET simulation? | ✅ 45–55 dB agreement |
| **V5** | Is AIMET better than converter-only quantization on the NPU? | ✅ +2.6 to +5.9 dB |
| Demo | Fault injected → flagged on device | ✅ |

V3 is what makes the rest trustworthy: because conversion alone is bit-exact, any remaining difference on the NPU is quantization, not a conversion bug.

---

## 9. The dashboard (in depth)

### 9.1 What it is
A single-page web app **served by the QIDK itself**. A native C++ server on the board serves the page and data, and runs **live inference on the Hexagon NPU** when the page asks. The laptop browser only displays. Built to present the project to a technical audience on a projector.

### 9.2 Architecture
```
┌──────────────── QIDK (Android 14, SM8650) ─────────────────────────────────────┐
│ /data/local/tmp/nz_aimet/                                                      │
│   dashboard/nz_dashboard   native C++17 server (cpp-httplib 0.59), 1.9 MB      │
│     GET  /…               static files from dashboard/www (HTML, JS, CSS,      │
│                           uPlot charts, data pack: ~19 MB)                     │
│     GET  /api/status      device model, SoC, uptime, server version           │
│     POST /api/run         body = float32 window → spawns qnn-net-run →         │
│                           Hexagon HTP (W8A16) or CPU (FP32) → JSON out+timings │
│   bin/qnn-net-run, bin/qnn-profile-viewer, lib/libQnnHtp*.so …  (QAIRT 2.47)    │
│   <model>/models/libaimet_w8a16.so, libfp32.so      (compiled model libraries) │
└───────────────┬───────────────────────────────────────────────────────────────┘
                │ USB: adb forward tcp:8080 tcp:8080      (or Wi-Fi :8080, or the QIDK's own Chrome)
┌───────────────▼──────────────┐
│ Laptop browser → projector   │  http://localhost:8080
│ all charts + anomaly maths   │
│ run in JavaScript (app.js)   │
└──────────────────────────────┘
```

Why this design ([doc 16](aimet_quantization/16_dashboard_plan.md)):
* **No Python on the device** → the server had to be native. Cross-compiled with the Android NDK r26c (`aarch64-linux-android31`, `-static-libstdc++`) inside the QAIRT image: [dashboard/build.sh](aimet_quantization/dashboard/build.sh).
* **A web page instead of an Android app**: same visuals, works on a laptop, a projector and the QIDK's own touchscreen, and needs no JDK/Gradle/APK signing.
* **Fast build, honest trade-offs** (plan §7): the full plan (persistent QNN context in a custom runner, ~5 ms live) was replaced by spawning `qnn-net-run` per request (~1.5–2 s). The replay views use **recorded** NPU outputs. Both choices are shown openly in the UI.
* **Offline-safe**: all JS/CSS (including uPlot 1.6.32, ~50 KB) is bundled; no CDN or internet needed.
* **USB as the primary link**: immune to venue Wi-Fi client isolation.

### 9.3 Starting and stopping
```bash
cd mideval/aimet_quantization
scripts/14_dashboard_start.sh          # → http://localhost:8080
scripts/14_dashboard_start.sh --stop
```
The launcher ([scripts/14_dashboard_start.sh](aimet_quantization/scripts/14_dashboard_start.sh)): kills any running server (`pkill nz_dashboard`), pushes the runtime and model libraries only if missing, pushes the server binary and web files with `adb push --sync` (only changed files), starts the server with **`--daemon`** (double fork + `setsid`, so `adb shell` returns instead of hanging), sets up `adb forward`, waits for `/api/status`, prints the USB and Wi-Fi URLs, opens the browser. Every adb call retries 3× (restarting the adb server if the link dropped). Takes ≈1 s when nothing changed. Environment overrides: `SERIAL`, `PORT`, `NO_BROWSER=1`.

### 9.4 Header (always visible)
* **Title**, subtitle "LSTM forecasters · AIMET W8A16 · running on the Qualcomm QIDK Hexagon NPU".
* **KPI tiles** (change with the tab's model): `ms / inference (NPU)` (recorded mean; replaced by the live value after a live run, label "ms NPU (live)"), `dB fidelity vs FP32`, `R² FP32 → NPU`, `same anomaly verdicts` (natural-flag agreement), and **`anomalies in view`** (flags currently visible in the replay charts).
* **Device badge**: "● live Pineapple for arm64 · SM8650 · nz_dashboard 1.0 (cpp-httplib 0.59.0)", fetched from `/api/status` (Android `getprop`). Shows "server offline" in red if the board doesn't answer. This proves the page is served by the QIDK.

### 9.5 Tab 1: Campus replay (`all_sensors_1h`)
Scrolls through the **test period** (Nov 10 – Dec 31 2025) using the outputs **the QIDK's NPU recorded for every one of the 4,832 test windows** during P4.

**Controls:** Play/Pause, speed 1× / 4× / 16× / 64× steps per second, time slider, date-time jump (IST), **"Jump to moment…"** (the 20 strongest *natural* Solar anomalies in the test data, by score = |residual − median| / limit, listed chronologically), current time. Opens at **11:00 IST on day 4** so the first view shows daylight solar.

**Charts** (48 h visible = 192 steps):
1. **Solar: actual vs NPU forecast (+15 min) with typical band.**
2. **Grid:** same.
   * White line = actual reading. Dashed coloured line = NPU forecast made 15 min earlier. Hidden grey series = FP32 forecast (toggle in the legend to compare).
   * **Green band = forecast + median_h ± limit_h**: the range of "typical" readings for that hour. Exactly the thresholds of the anomaly rule.
   * **Red dot = anomaly** (reading outside the band). **Orange dot = injected fault.**
3. **Solar residual (actual − forecast − typical bias) vs alarm limit:** the same decision drawn as "residual inside ±limit or not".

**Side panel:**
* **Reading at <time>**: for Solar, Grid, Demand: actual, forecast, residual (bias-corrected), ±limit, verdict **ok / ANOMALY**. Demand shows grey **"n/a"** (its forecast head is unreliable; see 5.2).
* **Inject a fault** into the next 1 / 4 / 8 / 16 readings of a chosen domain:
  * *Sensor dropout* → reading = 0 W
  * *Spike* → reading + 50 % of capacity
  * *Stuck at current value* → holds the reading at the moment of injection
  * *Panel damage* → reading × 0.6
  * *Clear faults*. A counter shows how many injected readings are active.
* **Forecast from this window (recorded NPU)**: the 4-step table (+15/+30/+45/+60 min) in kW for each domain.
* **Run this window live on the NPU →**: jumps to tab 4 with this exact window loaded.

**How fault injection works (and its limit):** a fault changes only the **actual** reading. The verdict is recomputed in the browser against the recorded forecast with the same per-hour thresholds, i.e. the same maths as the offline evaluation (`anomaly.py`). The faulty value is **not** fed back into later input windows. That's realistic for the first faulty reading, which is the one the detector judges. To see how a faulty *input history* changes the forecast, use the channel editor in tab 4.

Example from the screenshot: at 2025-11-14 11:00 IST a +50 % spike makes Solar read 761.2 kW vs 416.4 kW forecast; residual 343.9 kW > limit 274.6 kW → **ANOMALY**; Grid stays ok.

### 9.6 Tab 2: Plant SL-PL00-00 (`single_node`)
Same template as tab 1 for the **Palash** plant model (one chart for its solar output + residual chart). This is the plant-level detector: it catches dropouts on one plant (92 %) that would be diluted in the campus total, and partially catches panel damage (17 %).

### 9.7 Tab 3: Day-ahead 24 h (`all_sensors_24h`)
* **Slider: "Forecast issued at"** any test time.
* **Main chart:** next 96 steps of Solar and Grid, forecast (dashed, recorded NPU) vs actual.
* **Second chart:** room Demand (sub-meters) forecast vs actual.
* **Next 24 h energy table:** forecast vs actual kWh for Solar, Grid, Demand (Σ W × 0.25 h), error %, and a **Net (demand − solar)** row. Note: "Demand" here is only the 3 room sub-meters (~30 kW capacity), not the whole campus, so "Net" is illustrative.
* **"What a typical day looks like":** P10–P90 band and median of actual campus solar by IST hour (validation + test period), with the forecasted and actual day overlaid (hourly averages).

This model's Grid accuracy is weak (R² ≈ −0.1 on our data); Solar is decent (R² 0.80). Present it as a planning view, not as an accuracy showcase.

### 9.8 Tab 4: Run on NPU (live inference)
The only tab that computes **new** results at the moment you click.

**Top bar:** model selector (1 h / 24 h / single node), window index, the IST time span of the 96 input rows.

**Card 1: Input window.**
* **Download as CSV template:** the current window in **physical units** (W, V, A, Hz…), ISO UTC timestamps, one column per feature named as in the model metadata, plus **row 97 = the reading that actually arrived next**.
* **Upload CSV**: your own sensor readings. Validation: `timestamp` column present; every sensor column present (missing ones are listed); 96 rows (+ optional 97th); numbers only; rows exactly 15 min apart. Values more than 5 % outside the training range trigger a warning (the model would be extrapolating).
  * Cyclic time features are **recomputed from the timestamps** (IST, MinMax-scaled like training). `was_imputed` columns default to 0.
  * Checkbox "values already normalized" skips the MinMax step.
* **Reset to test window.**

**Card 2: Edit a sensor channel** (physical units), over the **last N of 96 steps**: set to 0 (dropout), scale × k, freeze (stuck sensor), add spike (+k × channel max). A chart shows original vs edited. This changes the model's **input**, so you see how a faulty sensor history moves the forecast.

**Card 3: Run.**
* **▶ Run on NPU (HTP, AIMET W8A16)** or **Run FP32 on CPU (reference)** (keyboard `R` runs on the NPU).
* Chart: live result vs recorded NPU output vs actual, over the forecast horizon.

**Side panel:**
* **Last live run** timings:
  | Field | Meaning | Typical |
  |---|---|---|
  | NPU/CPU execute | Pure model execution on the accelerator, from QNN's profiler | **≈4.4–5.6 ms** (HTP) |
  | graph prepare (per launch) | `qnn-net-run` loading the library and preparing the HTP graph, paid on **every** launch | 0.9–1.6 s |
  | qnn-net-run total | Wall time of the whole process on the QIDK | ≈1.4–2.2 s |
  | browser round trip | Click → answer, as seen by the laptop | ≈ total + a few ms |
  | match vs recorded run | For an unedited test window: max difference to the output recorded in P4 | ≈ 4–6 × 10⁻⁷ (identical up to float printing) |
  The gap between ~5 ms and ~2 s is process start-up, not the model. A persistent QNN context inside the server would remove it (planned, not built).
* **Forecast** table (first 8 steps; all steps in the chart).
* **Anomaly check of the next reading:** if row 97 exists, the actual Solar/Grid/Demand targets are computed from its sensor values (same definitions as training: Σ solar `active_power_W`; SB00 `gd_power_W` + KH95 `power_total_W`; Σ room `power_W`) and judged with the same per-hour rule. Demand again "n/a".

**What `/api/run` does on the QIDK** ([nz_dashboard.cpp](aimet_quantization/dashboard/server/nz_dashboard.cpp)):
1. Validates `model` ∈ {all_sensors_1h, all_sensors_24h, single_node}, `variant` ∈ {aimet_w8a16, fp32}, and body size = 96 × F × 4 bytes.
2. Takes a mutex (one NPU job at a time).
3. Writes `<model>/live/window.raw` and an input list.
4. Runs `qnn-net-run --backend libQnnHtp.so` (or `libQnnCpu.so`) `--model <model>/models/lib<variant>.so --profiling_level basic` (+ `--perf_profile burst` on the HTP).
5. Parses `qnn-profile-viewer` output for execute and prepare times (backend-specific labels).
6. Returns `{ok, model, variant, backend, wall_ms, execute_us, finalize_us, out:[…]}`; on failure, HTTP 500 with the tail of the run log.

### 9.9 Tab 5: Under the hood
* **Pipeline diagram** (FP32 ONNX → AIMET → converter → model library → QIDK server → qnn-net-run → HTP).
* **Chart: max |NPU − FP32| per window over the whole test period** for the 1 h model, in % of capacity (stays below 0.45 %).
* **Verification table** for all three models: NPU vs FP32 SQNR, converter-only SQNR, R² FP32 → NPU, same anomaly verdicts, NPU execute ms, "FP32 graph on device CPU: bit-exact".
* The **anomaly rule** in one formula, and the **keyboard cheat-sheet**.

### 9.10 Keyboard shortcuts and URL hooks
| Key | Action |
|---|---|
| `1`–`5` | switch tab |
| `Space` | play / pause replay |
| `←` / `→` | step 15 min; with `Shift`: ±1 day |
| `D` / `S` / `F` / `P` | inject dropout / spike / stuck / panel damage |
| `C` | clear faults |
| `N` | jump to next natural anomaly (Solar/Grid) |
| `R` | in a replay tab: open this window in "Run on NPU"; in "Run on NPU": run on the NPU |

URL hashes (deep links + headless tests): `#replay`, `#plant`, `#dayahead`, `#live`, `#hood`; `#live-run` (fires one NPU run on load); `#inject-dropout|spike|stuck|panel`; `#csv-selftest` (template → real upload parser → compare → NPU run).

### 9.11 The data pack ([scripts/13_dashboard_pack.py](aimet_quantization/scripts/13_dashboard_pack.py))
Consecutive test windows overlap by 95 of 96 rows, so the pack stores **one timeline** of `n_test + 95` rows and slices `window i = timeline[i : i+96]`. That's 1.1 MB instead of 107 MB for a 58-feature model. Per model in `dashboard/www/data/<model>/`:

| File | Shape | Content |
|---|---|---|
| `timeline.f32` | [n+95, F] | normalized inputs, exactly what the model sees |
| `y.f32` | [n, H·D] | actual targets (normalized with the *data* capacities) |
| `npu.f32` | [n, H·D] | **recorded QIDK HTP W8A16 outputs** from P4 |
| `fp32.f32` | [n, H·D] | FP32 ONNX outputs on the same windows |
| `meta.json` | – | feature names with physical min/max, model vs data capacities, per-hour anomaly medians/limits, typical-day P10/P50/P90, device KPIs, timeline start (UTC) |

Total ≈ **19 MB** (1.85 + 17.5 + 0.44; the 24 h model's 288 outputs dominate). The script asserts that sliced windows equal the original test windows.

### 9.12 What's live and what's recorded
| Element | Source | Computed when |
|---|---|---|
| Replay / Plant / Day-ahead forecasts | **Real NPU outputs**, recorded by the QIDK in P4 | earlier, on the same board |
| Anomaly verdicts, bands, fault injection | Browser JavaScript, same rule and thresholds as `anomaly.py` | live, in the browser |
| "Run on NPU" forecasts | `qnn-net-run` on the Hexagon HTP | **live, on click** |
| KPI numbers (SQNR, R², agreement) | P4 measurement files | earlier |
| Device badge | Android `getprop` on the QIDK | live |

### 9.13 Design decisions taken while building
| Decision | Why |
|---|---|
| Demand verdicts shown as "n/a" | The first screenshot showed red Demand "ANOMALY" in normal conditions; its head has negative R² and a 49 % natural flag rate |
| Replay opens at 11:00 IST, day 4 | Opening at night shows flat zero solar |
| 14 px anomaly markers | The current reading sits on the chart's right edge; small dots were missed on a projector |
| Side panel 470 px, `nowrap` numbers | Values were clipped and wrapping |
| Dark, high-contrast theme; layout for 1600×1000 / 1920×1080; collapses below 1100 px | Projector presentation |
| URL test hooks | Headless verification of every path, and deep links while presenting |

### 9.14 How the dashboard was verified
Headless Chromium against the live server on the QIDK (screenshots in [dashboard/screenshots/](aimet_quantization/dashboard/screenshots/)):
| Check | Result |
|---|---|
| All 5 tabs render, no console errors | ✅ |
| Live API vs recorded P4 outputs (window 10) | ✅ HTP ≤ 5.0 × 10⁻⁷, CPU ≤ 6.1 × 10⁻⁷ for all 3 models |
| Live run from the browser | ✅ NPU 4.76 ms, max |Δ| 4.6 × 10⁻⁷ |
| Fault injection | ✅ dropout: −417 kW vs ±275 kW limit → ANOMALY; spike: +344 kW → ANOMALY; Grid ok; Demand n/a |
| CSV round trip | ✅ sensors 1.5 × 10⁻⁸, cyclic features recomputed from timestamps 2.9 × 10⁻¹⁸, next-reading actuals identical |

Bugs fixed during the build: the launcher hung on `adb shell` (→ `--daemon`); two server instances (a `pkill -f` matched its own shell → `pkill nz_dashboard`); `413 Payload Too Large` for form-encoded clients (limit raised to 4 MB); CPU execute time −1 (backend-specific profiler labels).

### 9.15 Dashboard limitations
* Live runs take ~2 s end-to-end (process start-up per request), although the NPU part is ~5 ms.
* Replay forecasts are recorded, not recomputed per frame.
* Injected faults in the replay don't propagate into later input windows (use tab 4 for that).
* The Net row in Day-ahead uses room sub-meter Demand only.
* One NPU job at a time (mutex). Fine for a demo.

---

## 10. Using the deployed models without the dashboard
```bash
cd mideval/aimet_quantization
../.venv-aimet/bin/python scripts/12_qidk_forecast.py --model all_sensors_1h --test-index 10
../.venv-aimet/bin/python scripts/12_qidk_forecast.py --model all_sensors_1h --test-index 10 --inject sensor_dropout_zero
../.venv-aimet/bin/python scripts/12_qidk_forecast.py --model single_node --input my_window.npy     # [1, 96, F] float32, normalized
```
Output: forecast in Watts per 15-min step and, if the actual reading is known, `ok` / `ANOMALY` per domain. Directly on the device: push the `.so` and a raw `[1, 96, F]` float32 window, run `qnn-net-run --backend lib/libQnnHtp.so --model lib….so`, pull `Result_0/forecast_all_pillars.raw` (see [README_DETAILED §8](aimet_quantization/README_DETAILED.md)).

Other Snapdragon chips: the model library is chip-independent; push the matching `libQnnHtpV<arch>Stub/Skel.so` and optionally use the matching AIMET HTP config.

---

## 11. Repository map
```
mideval/
  explanation.md, presentation_flow.md     this document, presentation plan
  model2/                                  ORIGINAL PROJECT (unchanged)
    generate_lstm_dataset.py               normalization stage
    train_and_export_qidk.py               single-node model
    train_all_sensors_lstm.py              1 h model
    train_24h_multisensor_lstm.py          24 h model
    run_comprehensive_benchmark.py         I/O + baselines benchmark → benchmark_results.json
    report.md, report2.md, progress.txt    team reports
    qidk_deployment/                       FP32 ONNX models, .pt checkpoints, metadata, sample inputs
  aimet_quantization/                      QUANTIZATION + DEPLOYMENT + DASHBOARD
    README.md / README_DETAILED.md / TECHNICAL_QA.md
    00–17_*.md                             plan (00–09), execution logs (10–15), dashboard (16–17)
    scripts/
      00_smoke_test_aimet.py               AIMET loads/simulates the models (V1)
      01_rebuild_dataset.py                raw telemetry → normalized dataset
      02_build_windows.py                  windows, calibration set, fidelity report
      03_fp32_baseline.py                  FP32 reference outputs + metrics (V0)
      04_anomaly_baseline.py, anomaly.py   thresholds, fault injection
      05_aimet_quantize.py                 grid + range priors + AutoQuant/AdaRound + export
      06_evaluate_sim.py                   test evaluation of exports (V2)
      07_prepare_raw.py                    raw inputs + filtered encodings for the converter
      08_qairt_build.sh                    converter + model-library build (Docker)
      09_run_on_qidk.sh                    push / run / pull
      10_compare_device.py                 V3/V4/V5
      11_benchmark_latency.sh              latency
      12_qidk_forecast.py                  end-user runner
      13_dashboard_pack.py                 dashboard data pack
      14_dashboard_start.sh                dashboard launcher
      common.py                            shared helpers
    configs/                               AIMET HTP v75 configs (LSTM per-tensor)
    docker/Dockerfile.qairt-fixed          protobuf-fixed QAIRT image
    release/<model>/                       NPU .so, AIMET ONNX + encodings, thresholds, metadata
    release/qidk_runtime_2.47/             device runtime (qnn-net-run, libQnnHtp*, …)
    dashboard/server/nz_dashboard.cpp      device web server; httplib.h
    dashboard/www/                         index.html, app.js, style.css, vendor/uPlot, data/
    dashboard/build.sh, bin/, screenshots/
    data/, out/                            generated datasets, results, device outputs
```

Reproduce everything: [README_DETAILED §10](aimet_quantization/README_DETAILED.md) (`00` → `14`, ≈7 min AIMET + ≈10 min build + device runs).

---

## 12. Limitations (stated honestly)
* Absolute accuracy figures use an **approximate rebuild** of the training data (44/58 reference features identical, solar capacity −1.9 %). Quantized-vs-FP32 comparisons are exact.
* **Demand** forecasts are poor in FP32 (negative R²) and aren't used for anomaly decisions.
* **Partial panel damage** isn't detectable from campus totals; per-plant models would need training.
* **24 h Grid** R² ≈ −0.1 on our data (the reported 0.741 looks like a copied baseline value).
* Dense layers use **per-tensor** weight scales on the NPU; per-channel simulated +1.5–3 dB better but mis-executed in this path.
* Dashboard live runs pay ~1.5 s process start-up per click.

## 13. Next steps (if continued)
1. Per-channel dense weights on the HTP via a newer converter path or a context binary (+1.5–3 dB in simulation).
2. Mixed precision (8-bit except the per-step cell state) to see whether W8A8 can be rescued for the 1 h / single models.
3. Persistent QNN context (or context binary) in the server → millisecond live runs; optionally an Android app.
4. Per-plant forecasters + multi-day residual aggregation for panel-degradation detection (needs training).
5. Compare against TCN / DLinear / GRU (suggested in `model2/report2.md`), which run more efficiently on NPUs than recurrences.

## 14. Lessons learned
1. **Calibration-only ranges clip.** Outputs overshoot calibration; LSTM hidden states have a known ±1 bound. Priors took 16-bit activation fidelity from 44 to 77 dB.
2. **Check what is actually quantized.** The exporter routed LSTM weights through ops, so "W8" was really 16-bit until fixed.
3. **Simulation ≠ hardware for recurrent models.** Only on-device measurement caught the W8A8 collapse.
4. **Isolate conversion from quantization.** The bit-exact CPU rung plus per-tensor debug dumps turned four "it's broken" moments into four specific causes.
5. **Fidelity to FP32 is the right metric** when the ground-truth data is approximate.

---

## 15. Glossary
| Term | Meaning |
|---|---|
| **QIDK** | Qualcomm Innovators Development Kit: the Snapdragon 8 Gen 3 (SM8650) dev board |
| **HTP / Hexagon NPU** | Hexagon Tensor Processor, the neural accelerator; v75 on SM8650 |
| **QNN / QAIRT** | Qualcomm AI Engine Direct / Qualcomm AI Runtime SDK (converter, model-library generator, `qnn-net-run`); version 2.47 used |
| **AIMET** | AI Model Efficiency Toolkit (Qualcomm), used here as `aimet-onnx 2.12` |
| **QuantSim** | AIMET's quantization simulation of a model on the host |
| **Encodings** | Per-tensor scale/offset/bitwidth JSON that AIMET exports and the converter consumes |
| **PTQ / QAT** | Post-training quantization / quantization-aware training |
| **W8A16 / W8A8** | 8-bit weights with 16-bit / 8-bit activations |
| **Per-tensor / per-channel** | One scale per weight tensor / one per output channel |
| **AdaRound** | Learned up/down rounding of weights to minimize layer output error |
| **CLE / BN folding** | Cross-layer equalization / batch-norm folding (no effect on this architecture) |
| **Range priors** | Our fixed ranges: LSTM hidden ∈ [−1, 1], outputs = calibrated ± 10 % |
| **SQNR** | Signal-to-quantization-noise ratio vs FP32, in dB; 48 dB ≈ 0.4 % RMS error |
| **R²** | Coefficient of determination; 1 = perfect, 0 = as good as predicting the mean, < 0 = worse |
| **MAD** | Median absolute deviation; × 1.4826 ≈ standard deviation, robust to outliers |
| **IST** | Indian Standard Time, UTC+5:30 |
| **Window** | 96 consecutive 15-min rows (24 h) = one model input |
| **Model library (.so)** | Compiled QNN graph for Android, executed by `qnn-net-run` or the QNN API |
| **adb forward** | Tunnels a laptop port to a device port over USB |

## 16. Numbers cheat-sheet
* 11 nodes · 58 features (41 + 11 + 6) · 96-step (24 h) input · 15-min cadence
* 32,866 aligned intervals · 70/15/15 split · test = 2025-11-10 → 12-31 · 4,832 / 4,740 / 4,861 test windows
* 1,024 calibration windows · 600 selection windows · 512 converter calibration windows
* Params 139,749 / 282,153 / 55,021 · quantized weights 153 / 297 / 71 KB (≈3.8× smaller)
* NPU 3.42 / 3.44 / 3.93 ms · 48.4 / 46.4 / 52.1 dB · ΔR² +0.0007 / −0.0039 / −0.0001
* AIMET vs converter-only: +3.5 / +2.6 / +5.9 dB · worst-case error 0.45 % vs 2.6 % (1 h)
* W8A8 24 h on NPU: 14.7 dB, R² 0.34 → 0.17 ✗
* Anomaly: K = 4, floor 2 %, per IST hour · dropout 69 % / 92 %, spike 91 % / 83 %, panel 0 % / 17 %, false flags ≈2–3 %
* Dashboard: 1.9 MB server · 19 MB data · live run ≈5 ms NPU / ≈2 s end-to-end · live = recorded to ≤ 6 × 10⁻⁷
