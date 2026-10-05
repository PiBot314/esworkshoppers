# Presentation Flow: Campus Net-Zero Forecasting & Anomaly Detection on the QIDK

**Format assumed:** technical evaluation, **~18 min talk + live demo, then Q&A**. A 10-minute cut is at the end.
**Setup:** laptop + projector, QIDK on USB, dashboard at `http://localhost:8080`.
**Story in one sentence:** *"We took three trained LSTM forecasters, quantized them with AIMET without retraining, proved on the QIDK's NPU that they still behave like the originals, and built a dashboard that the board itself serves and that runs our models live."*

Background for every number: [explanation.md](explanation.md).

---

## Pre-flight checklist (15 min before)

- [ ] Disable laptop suspend / screen lock (the laptop suspended mid-build once and adb dropped).
- [ ] `adb devices` lists `d42f06b`.
- [ ] `cd mideval/aimet_quantization && scripts/14_dashboard_start.sh`. The browser opens. Header badge shows **"● live … SM8650"**.
- [ ] Do one warm-up **Run on NPU** (tab 4) so you know it answers (~2 s).
- [ ] Browser zoom so the page fits the projector (layout designed for 1920×1080; collapses below 1100 px).
- [ ] Open in background tabs: the [screenshots](aimet_quantization/dashboard/screenshots/) (fallback), [README_DETAILED.md](aimet_quantization/README_DETAILED.md), [TECHNICAL_QA.md](aimet_quantization/TECHNICAL_QA.md).
- [ ] Optional backup terminal: `../.venv-aimet/bin/python scripts/12_qidk_forecast.py --model all_sensors_1h --test-index 10 --inject sensor_dropout_zero`.

---

## Running order

| # | Section | Time | Medium |
|---|---|---|---|
| 1 | Problem and goal | 1:30 | slide |
| 2 | Data and models (what we inherited) | 2:00 | slide |
| 3 | Anomaly detection without a new model | 1:30 | slide |
| 4 | Quantization with AIMET | 3:30 | slides |
| 5 | Getting it onto the NPU | 2:00 | slide |
| 6 | Results and verification | 2:00 | slide |
| 7 | **Live demo** | 5:00 | dashboard |
| 8 | Limitations, lessons, next steps | 1:00 | slide |
| | Q&A | open | dashboard tab 5 + backup slides |

---

## 1. Problem and goal (1:30)

**Slide content**
* Campus microgrid, **11 sensor nodes**: 6 solar, 2 grid/substation, 3 room meters; data every 15 min.
* Net-zero question: `Net = Demand − Solar`. Need forecasts at **1 h** (dispatch) and **24 h** (planning).
* Goal: run the forecasters **on the edge**, on the **Qualcomm QIDK's Hexagon NPU**, and use them to **flag faulty sensor readings**.
* Constraint: **no retraining**; the trained models stay as they are.

**Say:** "The models already existed in FP32. The task was to make them run on the NPU with minimal loss, and to prove the loss is minimal."

---

## 2. Data and models (2:00)

**Slide content**
* Input: last **24 h = 96 steps × 58 features** (41 electrical channels, 11 imputation flags, 6 cyclic time features), all MinMax-scaled to [0, 1].
* Three models, each **2-layer LSTM + 2-layer MLP head**:

| Model | Output | Params | FP32 headline R² |
|---|---|---|---|
| `all_sensors_1h` | next 1 h, Solar/Grid/Demand | 140 k | 0.852 |
| `all_sensors_24h` | next 24 h | 282 k | 0.339 |
| `single_node` | next 1 h, Palash plant | 55 k | 0.950 |

* What we found when we picked the project up: the normalized dataset was missing (→ **rebuilt from raw telemetry**: identical timeline and splits, Solar R² within 0.012 of the reported values); the deploy script assumed Python on the board (there is none); the latency estimate counted one LSTM step (≈40× too optimistic).

**Say:** "Because our data is a reconstruction, we never judge quantization against labels. We compare the quantized model with the FP32 model on identical inputs. That comparison is exact."

---

## 3. Anomaly detection without a new model (1:30)

**Slide content**
```
r = actual(t+15 min) − forecast(t+15 min)
flag if |r − median_hour| > max(4 · robust σ_hour, 2 % of capacity)
```
* Statistics per IST hour, from validation residuals. Robust σ = 1.4826 × MAD.
* Detection on injected faults: **dropout 69 % (campus) / 92 % (plant)**, **spike 91 % / 83 %**, false flags ≈ 2–3 %.
* Honest limit: **−40 % panel damage is invisible at campus level** (0 %), weak per plant (17 %). Demand forecast not reliable enough → not used.

**Say:** "The forecaster is the reference. If a reading is far from what the model expected for that hour of day, it's flagged."

---

## 4. Quantization with AIMET (3:30)

**Slide 4a: Target and method**
* **W8A16**: 8-bit weights, 16-bit activations. The LSTM state passes through 96 recurrent steps, so activations keep 16 bits.
* **AIMET 2.12** (last version with ONNX AutoQuant), HTP v75 config, 1,024 calibration windows stratified by hour of day.
* Metric: **SQNR vs FP32** (48 dB ≈ 0.4 % RMS error), plus R² and anomaly-decision agreement.

**Slide 4b: The surprise and the fix (the core technical contribution)**
* 16-bit activations alone lost ~20 kW. Diagnosis: **range clipping**. Calibration never saw negative night-time solar forecasts, and LSTM hidden states can reach ±1 but were calibrated to ±0.7.
* **Range priors**: LSTM hidden → **[−1, 1]** (`h = o·tanh(c)`, mathematically bounded); outputs → calibrated range **±10 %**.
* Activation-only fidelity **43.8 → 77.5 dB**. Now the 8-bit weights are the bottleneck → **AdaRound** (learned rounding).
* Also fixed: the exporter routed LSTM weights through Slice/Concat, so AIMET had quantized them as 16-bit activations. Forced to true 8-bit.

**Slide 4c: Validation SQNR (dB)**

| | 1 h | 24 h | plant |
|---|---|---|---|
| W8A16, calibration only | 42.4 | 43.3 | 49.9 |
| + range priors | 47.6 | 43.3 | 49.9 |
| **+ AutoQuant / AdaRound** | **50.3** | **47.1** | **51.1** |

**Say:** "AutoQuant's batch-norm folding and cross-layer equalization had nothing to do on an LSTM. The gains came from the priors we added and from AdaRound."

---

## 5. Getting it onto the NPU (2:00)

**Slide content**
```
AIMET model.onnx + encodings
 → qnn-onnx-converter  (AIMET ranges as overrides + calibration for the rest,
                        LSTM expanded per step, I/O layout preserved)
 → qnn-model-lib-generator → lib<model>_w8a16_htp.so
 → QIDK: qnn-net-run on the Hexagon HTP  (QAIRT 2.47 runtime in /data/local/tmp/nz_aimet)
```
* Seven failures on the way, each with a specific cause. Highlights: input layout silently transposed (4.9 dB → fixed with `--preserve_io layout`); per-channel dense weights mis-executing (14.6 dB → per-tensor + AdaRound re-run).
* Method: `qnn-net-run --debug` dumps every tensor; compare NPU vs CPU-FP32 tensor by tensor, per LSTM step.

**Say:** "The FP32 graph on the device CPU is bit-exact with ONNX, so every difference left on the NPU is quantization, not a conversion bug."

---

## 6. Results and verification (2:00)

**Slide 6a: On the QIDK, held-out test period (≈4,800 windows per model)**

| Model | NPU time | SQNR vs FP32 | R² FP32 → NPU | Same anomaly decisions | AIMET vs converter-only |
|---|---|---|---|---|---|
| 1 h | **3.42 ms** | **48.4 dB** | 0.852 → 0.853 | 99.66 % | +3.5 dB |
| 24 h | **3.44 ms** | **46.4 dB** | 0.339 → 0.335 | 99.76 % | +2.6 dB |
| plant | **3.93 ms** | **52.1 dB** | 0.950 → 0.950 | 99.92 % | +5.9 dB |

* Weights ≈3.8× smaller. Latency is >200,000× below the 15-min data interval.
* **W8A8 looked fine in simulation (40.6 dB) but collapsed on the NPU for the 24 h model (14.7 dB, R² 0.34 → 0.17)** because AIMET simulates the LSTM as one float op, while the NPU runs 96 steps with 8-bit internals. That's why on-device verification matters.

**Slide 6b: Verification ladder** (one line each): V0 data reproduces FP32 · V1 AIMET graph = ONNX (1e-7) · V2 sim ≥ 30 dB · V3 device CPU bit-exact · V4 NPU ≈ sim (45–55 dB) · V5 AIMET > converter-only.

**Transition:** "Let me show it running on the board."

---

## 7. Live demo (5:00)

Switch to the browser. Keys in **bold**.

| Time | Do | Say / point at |
|---|---|---|
| 0:00 | Tab **1** (Campus replay). Point at the header | "This page is served by the QIDK itself: the badge reads the board's model and SoC live. KPIs are the on-device measurements." |
| 0:20 | Point at the Solar chart | "White = actual, dashed = the NPU's forecast made 15 minutes earlier, green band = what's typical for this hour. These are the real NPU outputs for every test window." |
| 0:45 | **Space** (play), let it run a few seconds, **Space** | "Days scroll by; readings stay inside the band." |
| 1:00 | **S** (spike) | "A sensor spike on Solar: the reading leaves the band, the residual crosses the limit → **ANOMALY**. Grid stays ok. Demand shows n/a because its forecast isn't reliable enough to judge." |
| 1:25 | **C**, then **D** (dropout) | "A dead sensor reading 0 W → flagged." |
| 1:40 | **C**, then **P** (panel damage −40 %) | "Partial panel damage on the campus total: usually *not* flagged. We say so openly: it's within normal forecast error at campus level." |
| 2:00 | **C**, then **N** | "A real anomaly from the test data, nothing injected." |
| 2:15 | **R** (opens this window in Run on NPU), click **▶ Run on NPU** | "Now we send this exact 24-hour window to the board, and it runs on the Hexagon NPU right now." |
| 2:30 | Point at the *Last live run* table | "**~5 ms** of NPU compute. The ~1.5 s is `qnn-net-run` preparing the graph on each launch. And the live output matches the recorded run to ~10⁻⁷." |
| 3:00 | Card 2: feature `SL-PL00-00__active_power_W`, *set to 0*, last 16 steps → **Apply edit** → **Run on NPU** | "We simulate the Palash plant's sensor dying for the last 4 hours. The input changes, so the forecast changes." |
| 3:30 | (Optional) **Download as CSV template** | "Anyone can download a window in physical units, edit it, upload it back, and run it on the NPU." |
| 3:50 | Tab **2** | "Plant-level model: it can say *which* plant misbehaves." |
| 4:10 | Tab **3**, move the slider | "Day-ahead: 24 h forecast vs actual, energy totals, and a typical-day envelope." |
| 4:35 | Tab **5** | "Under the hood: the pipeline and the per-window NPU vs FP32 difference over the whole test period, always under 0.45 % of capacity." |
| 5:00 | Stay on tab 5 for Q&A | |

**If something fails during the demo**
| Problem | Fallback |
|---|---|
| Badge shows "server offline" | `scripts/14_dashboard_start.sh` again (≈1 s), refresh |
| Live run errors / hangs | Replay tabs still work (data is already in the browser). Show `screenshots/live-run.png` |
| adb lost | Replug USB, `adb devices`, re-run the launcher |
| Everything down | Screenshots `plant.png`, `inject-spike.png`, `live-run.png`, `dayahead.png`, `hood.png`, `csv-selftest.png` |

---

## 8. Limitations, lessons, next steps (1:00)

**Slide content**
* **Limits:** approximate data rebuild (comparisons vs FP32 are exact); Demand head unreliable; panel damage not detectable at campus level; dashboard live runs pay ~1.5 s graph preparation.
* **Lessons:** calibration ranges clip, so use physical/mathematical priors · check what is *really* quantized · simulation ≠ hardware for recurrent models · isolate conversion from quantization.
* **Next:** persistent QNN context (ms live runs), per-channel weights on the HTP (+1.5–3 dB sim), per-plant models for damage detection, NPU-friendlier architectures (TCN / DLinear).

**Closing line:** "Three models, quantized without retraining, running on the NPU in under 4 ms, matching FP32 to 46–52 dB with 99.7 %+ identical anomaly decisions, and verified at every step from the data to the device."

---

## Q&A preparation (likely questions → short answers)

| Question | Answer | Where to point |
|---|---|---|
| Why not full INT8? | W8A8 halved the 24 h model's R² on the NPU (14.7 dB) to save 0.7 ms. HTP v75 runs 16-bit activations natively; weights stay 8-bit. | Slide 6a |
| Did AIMET actually matter? | Same converter, bitwidths and calibration data: converter-only 44.9/43.8/46.3 dB vs AIMET 48.4/46.4/52.1. Worst-case error 6× lower on 1 h. | Tab 5 table |
| Why SQNR, not accuracy? | It measures what quantization changed, independent of data quality. R² and anomaly agreement are reported too. | Slide 4a |
| Why is the R² different from the original report? | Dataset rebuilt from raw; Solar R² within 0.012. The 24 h Grid "0.741" equals the persistence baseline's R²: likely a copy error. | explanation §4.2 |
| Is the replay really from the NPU? | Yes, recorded by the QIDK for every test window in P4. The "Run on NPU" button computes new results live; they match the recordings to ~10⁻⁷. | Tab 4 timing table |
| Why 2 s if the NPU takes 5 ms? | Each request starts `qnn-net-run`, which prepares the graph (~1.5 s). A persistent context would remove it. | Tab 4 |
| Why didn't AutoQuant's CLE/BN-folding help? | No batch norm and no conv chains in an LSTM + MLP. Gains came from range priors and AdaRound. | Slide 4c |
| Why per-tensor weights? | Per-channel dense weights produced wrong results on the HTP in this path (14.6 vs 46.2 dB). AdaRound was re-run for per-tensor. | Slide 5 |
| What is "expand LSTM"? | Converter flag that rewrites each of the 96 steps into MatMul/Add/Sigmoid/Tanh/Mul; the fused quantized LSTM op failed HTP validation. | Slide 5 |
| Can it detect panel damage? | Not at campus level (0 %); 17 % per plant. Needs per-plant models and multi-day aggregation, i.e. training. | Slide 3 |
| Why is Demand "n/a"? | Its FP32 R² is negative and it flags 49 % of normal readings. Showing its alarms would mislead. | Tab 1 side panel |
| How big is the model? | Quantized weights 71–297 KB (≈3.8× smaller). The `.so` is ~4 MB because it contains code for 96 unrolled steps. | – |
| Can it run in an app? | Yes, via the QNN C API / JNI with the same `.so` and `libQnnHtp.so`; a context binary would speed up start-up. | – |
| Other Snapdragon chips? | The model library is chip-independent; push the matching HTP Stub/Skel libraries. | – |

Full list: [TECHNICAL_QA.md](aimet_quantization/TECHNICAL_QA.md) (33 Q&As).

---

## 10-minute cut

| Section | Time |
|---|---|
| 1 + 2 merged: problem, models, "no retraining", data rebuild in one line | 1:30 |
| 3 Anomaly rule (formula only) | 0:45 |
| 4b Range priors + AdaRound (one slide, with the SQNR table) | 2:00 |
| 6a Results table + W8A8 collapse | 1:30 |
| 7 Demo: tab 1 → **S** → **C** → **R** → Run on NPU → tab 5 | 3:30 |
| Closing line | 0:15 |

Skip: conversion failures (slide 5), verification ladder (keep for Q&A), day-ahead and plant tabs.
