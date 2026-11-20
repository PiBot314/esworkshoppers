# QIDK benchmark: baseline vs guided LSTM (interim results, 11 Nov 2026)

> **Status (20 Nov 2026):** speed (§1) and power (§3) are still current for single windows (batch 1). Accuracy (§2)
> uses each model's own chronological test split; to compare models with each other, with baselines and with the
> 20 Nov models, use [forecast_2025/results/e1_compare.md](../forecast_2025/results/e1_compare.md) (identical rows).
> Not measured yet: fused and batched NPU calls, the case where the NPU should beat the CPU ([STATUS.md](../STATUS.md)
> items 4a/4b).

Board: Qualcomm QIDK, Snapdragon 8 Gen 3 (SM8650), Android 14, QAIRT 2.47. Models: the four LSTMs in `common/`
(baseline and guided, 1 h and 24 h), plus the CNN-LSTM and LightGBM for reference. One input window (96 × 58), batch 1.

**NPU precision.** Tables below use W8A16 (8-bit weights, 16-bit activations; the earlier deployment found W8A8 hurt
the 24 h LSTM). The NPU can't run FP32 at all. Speed at the other precisions (measured 11 Nov, ms per inference):

| Model | NPU W8A8 | NPU W8A16 | NPU FP16 | CPU ONNX Runtime FP32 |
|---|---|---|---|---|
| baseline / guided 1 h | 3.8 | 4.8–4.9 | 61 | 0.54–0.56 |
| baseline / guided 24 h | 4.1–4.2 | 4.9–5.1 | 97 | 0.89–0.91 |
| CNN-LSTM 24 h | 1.4 | 1.8 | 13 | 0.14 |

FP16 is 12–20× slower than W8A16 on this NPU (it is built for integer maths). W8A8 saves ~1 ms; its accuracy can be
checked live in the app (Benchmark → NPU W8A8 → Accuracy check). Power numbers are preliminary (see section 3).

**Update, 19 Nov:** a partial check on the live system (7 models per check, resident NPU runners, the assistant on the
CPU) is in [WORKLOG_2026-11-19.md](../archive/worklogs/WORKLOG_2026-11-19.md) §5, with scripts and raw output in
[live_eval_2026-11-19/](live_eval_2026-11-19/). It agrees with this page: per window, ONNX Runtime on the CPU is
~5–10× faster than the NPU.

## Bottom line
1. **All models run correctly on the board.** NPU outputs match the FP32 model to within 44–47 dB, and R² changes by
   at most ±0.03. Both CPU runtimes are exact.
2. **The NPU is not the fastest option for these models.** ONNX Runtime on the board's CPU is 5–12× faster than the NPU.
   The LSTMs are small and step through 96 time steps one after another, which the NPU can't parallelise, and each NPU
   call carries ~1.5 ms of fixed overhead.
3. **Baseline and guided cost the same to run** (same architecture, same parameter count). Guided is the more accurate
   of the two at 1 h; at 24 h they are within noise.
4. **Power:** running the NPU flat out does not register above idle on the only sensor the board has. CPU runs show a
   small but consistent rise and heat the chip more. No absolute figures yet; these need an inline power meter.
5. Every option is fast enough: the forecast is needed once every 15 minutes, and the slowest config takes 31 ms.

## 1. Time per inference (ms, measured on the QIDK)

| Model | NPU W8A16 (total) | NPU compute only | QNN CPU (FP32) | ONNX Runtime CPU, 1 thread | ONNX Runtime CPU, 4 threads |
|---|---|---|---|---|---|
| baseline 1 h | 4.86 | 3.30 | 16.1 | **0.59** | 0.53 |
| guided 1 h | 4.81 | 3.28 | 16.3 | **0.61** | 0.55 |
| baseline 24 h | 5.02 | 3.47 | 30.2 | **1.05** | 0.86 |
| guided 24 h | 5.05 | 3.49 | 30.9 | **1.06** | 0.86 |
| CNN-LSTM 24 h | 1.87 | 0.97 | 3.5 | **0.16** | 0.18 |
| LightGBM 24 h | n/a | n/a | n/a | **2.07** | 1.94 |

- "NPU total" is the time per call as the app sees it; "compute only" is the NPU's own work. The NPU's `power_saver` profile
  gave the same time as `burst`.
- LightGBM was converted to ONNX (matches LightGBM to 1e-6). On the board it takes 2 ms, not the 35–150 ms in the
  earlier PC report, which was Python call overhead.

## 2. Accuracy on the test split (R²)

| Model | Run on | Agreement with FP32 | Solar R² | Grid R² | Demand R² |
|---|---|---|---|---|---|
| baseline 1 h | FP32 (CPU) | exact | 0.909 | 0.607 | -0.004 |
| baseline 1 h | NPU W8A16 | 46.5 dB | 0.909 | 0.615 | 0.025 |
| guided 1 h | FP32 (CPU) | exact | **0.954** | **0.721** | -0.268 |
| guided 1 h | NPU W8A16 | 44.9 dB | 0.954 | 0.718 | -0.286 |
| baseline 24 h | FP32 (CPU) | exact | 0.778 | -0.060 | -2.918 |
| baseline 24 h | NPU W8A16 | 44.4 dB | 0.777 | -0.067 | -2.905 |
| guided 24 h | FP32 (CPU) | exact | **0.800** | -0.067 | -3.995 |
| guided 24 h | NPU W8A16 | 44.3 dB | 0.801 | -0.064 | -4.002 |
| CNN-LSTM 24 h | FP32 (CPU) | exact | 0.739 | -1.776 | -6.933 |
| CNN-LSTM 24 h | NPU W8A16 | 44.2 dB | 0.738 | -1.781 | -7.077 |
| LightGBM 24 h | FP32 (CPU) | – | 0.752 | **0.198** | -1.726 |

**Caveat on the absolute numbers:** these are lower than the figures in `common/*_meta_*.json` (e.g. baseline 1 h grid
0.61 here vs 0.76 reported). The `lstm_normalized` data on this machine is a slightly different version from the one the
models were trained on (solar scaling constant 535,813 W vs 546,010 W). Comparisons *between* rows are valid, since every
row uses the same inputs. For exact figures, the original `lstm_normalized` folder is needed.

What the table shows regardless:
- Guided beats baseline clearly at 1 h (solar 0.954 vs 0.909, grid 0.721 vs 0.607); at 24 h the gap is small.
- At 24 h no LSTM forecasts grid draw usefully (R² ≈ 0); LightGBM is the only model above zero.
- Demand R² is negative for every model. "Demand" here is the three KH04 room meters, not campus demand.

## 3. Power (preliminary)

The QIDK runs from its DC adapter and its battery gauge is simulated, so the board has no current sensor. As a stand-in,
the PMIC's main supply-rail voltage was sampled (it sags slightly as current rises) during 20 s of back-to-back inference,
compared with idle before and after. One pass so far; a second pass and a busy-CPU reference are still running.

| Model | Config | Inferences/s | Rail sag vs idle (mV) | Chip temp. at end (°C) |
|---|---|---|---|---|
| guided 1 h | NPU W8A16 | 177 | -0.4 (≈ idle) | 36.5 |
| guided 1 h | QNN CPU | 58 | 1.0 | 42.7 |
| guided 1 h | ONNX Runtime 1 thread | 1,634 | 1.2 | 41.7 |
| guided 1 h | ONNX Runtime 4 threads | 1,508 | 1.7 | 44.5 |
| guided 24 h | NPU W8A16 | 170 | 0.1 (≈ idle) | 39.5 |
| guided 24 h | QNN CPU | 30 | 1.7 | 45.0 |
| guided 24 h | ONNX Runtime 1 thread | 977 | 0.9 | 43.5 |
| guided 24 h | ONNX Runtime 4 threads | 1,027 | 2.5 | 46.1 |
| CNN-LSTM 24 h | NPU W8A16 | 410 | -0.4 (≈ idle) | 40.5 |
| CNN-LSTM 24 h | QNN CPU | 266 | 1.5 | 45.7 |
| CNN-LSTM 24 h | ONNX Runtime 1 thread | 6,175 | -0.1 | 43.8 |
| CNN-LSTM 24 h | ONNX Runtime 4 threads | 6,458 | 2.6 | 46.0 |
| LightGBM 24 h | ONNX Runtime 1 thread | 468 | 2.1 | 46.0 |
| LightGBM 24 h | ONNX Runtime 4 threads | 505 | 2.7 | 48.7 |

Noise is about ±0.5–1 mV per reading. Read this as a ranking, not a measurement:
- NPU runs are indistinguishable from idle and leave the chip coolest.
- CPU runs add a small load, more with 4 threads than 1, and heat the chip 4–10 °C more than NPU runs.
- **Per forecast, ONNX Runtime may still use less energy than the NPU**, because it finishes in ~1 ms (and the CPU
  goes back to sleep) while an NPU call takes ~5 ms. This can't be settled without a real power reading.
- In an earlier one-off test a single busy CPU core sagged the rail by ~54 mV, far more than any model here. The reference
  measurement at the end of this run will show whether the sensor is behaving consistently.

**To get real watts and mJ per inference:** put an inline power meter between the adapter and the board and run
`scripts/08_device_power.sh`-style 30 s loops (the loop mode is already built into `tools/ort_bench` and `qnn-net-run`);
about 15 minutes for all models.

## How this was run
Everything is in `benchmark_qidk/scripts/`, in order: `01` rebuild the test split with `common/common_data.py`
(feature order taken from `common/lgbm_feature_weights_24h.json`), `02` PC reference outputs, `03` LightGBM → ONNX,
`04` QNN builds (Docker, QAIRT 2.47; W8A16 calibrated by QNN on 512 training windows, not AIMET), `05` board tools
(ONNX Runtime 1.20 benchmark, rail sampler), `06` accuracy on the board (all 4,740–4,832 test windows), `07` latency
(1,000 runs NPU / 300 QNN CPU / 50 warm-up + 1,000 ONNX Runtime), `08` power, `09` summary.
Raw results: `benchmark_qidk/results/`.
