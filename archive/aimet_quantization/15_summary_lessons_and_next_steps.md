# 15 — Summary, Lessons Learned, and Next Steps (2026-10-05)

## 1. What was delivered
| Deliverable | Where |
|---|---|
| Three AIMET-quantized **W8A16** forecasters running on the QIDK Hexagon NPU (HTP v75) | `release/<model>/lib<model>_w8a16_htp.so` |
| AIMET exports (ONNX + encodings) for re-conversion with any QAIRT version | `release/<model>/*_aimet_w8a16.{onnx,encodings}` |
| Residual-based anomaly layer (thresholds per IST hour) | `scripts/anomaly.py`, `release/<model>/anomaly_thresholds_W.npy` |
| End-user runner (forecast + anomaly verdict on the device) | `scripts/12_qidk_forecast.py` |
| Fully scripted, re-runnable pipeline 00–12 | `scripts/` |
| Verification evidence (V0–V5) | `out/*.json`, docs 12–14 |
| Final READMEs and technical Q&A | `README.md`, `README_DETAILED.md`, `TECHNICAL_QA.md` |

Headline (held-out TEST, on device): fidelity to FP32 **48.4 / 46.4 / 52.1 dB**, R² change **+0.001 / −0.004 / −0.000**, anomaly-decision agreement **≥ 99.66 %**, **3.4–3.9 ms** per inference on the NPU, weights ≈3.8× smaller. No model was retrained. The original ONNX files are unchanged.

## 2. Plan vs reality
| Planned (docs 04–09) | What happened |
|---|---|
| aimet-onnx 2.12 AutoQuant | Worked, but only via `aimet_onnx._deprecated.auto_quant_v2`, and needed explicit AdaRound parameters (bug) |
| W8A16 via AutoQuant | AutoQuant's BN-fold/CLE are no-ops for LSTMs. The value came from **range priors** (our addition) + **AdaRound** on the head |
| "W8A16" in AIMET | Was initially 16-bit on the LSTM weights (exporter's Slice/Concat). Fixed so all weights really are 8-bit |
| `qairt-converter` → `qairt-quantizer` → DLC | Failed (float-fallback ops). The working path is `qnn-onnx-converter` + `--expand_lstm_op_structure` + `--preserve_io layout` → model `.so` |
| Per-channel Gemm (best in simulation) | Mis-executed on the HTP in this path → per-tensor |
| W8A8 as a comparison | Passed in simulation, **failed on the HTP for the 24 h model** (14.7 dB) |

## 3. Lessons (transferable)
1. **Calibration-only ranges clip.** Two tensor groups needed priors: outputs (over/undershoot beyond calibration) and LSTM hidden states (mathematically bounded ±1). That alone took 16-bit activation fidelity from 44 → 77 dB.
2. **Check what is actually quantized, and how.** Exported graphs can route weights through ops, so a "W8" label can be wrong. Inspect quantizer bitwidths and the compiled graph's tensor dtypes (`*_net.json`).
3. **Simulation ≠ hardware for recurrent models.** AIMET-ONNX treats `LSTM` as one op. The HTP runs 96 unrolled steps with quantized internals. Only on-device measurement (V4) caught the W8A8 collapse.
4. **Isolate conversion from quantization.** The bit-exact FP32-on-device-CPU rung (V3) plus per-tensor `--debug` dumps turned four "it's broken" moments into specific causes (layout, aliasing scale, per-channel FC, Lstm op validation).
5. **Fidelity to FP32 is the right quantization metric** when ground-truth data is approximate. It's insensitive to data mismatch and directly answers "did quantization change the model?".

## 4. Limitations (honest)
* Absolute accuracy figures are on an **approximate rebuild** of the training data (44/58 features identical on the reference window; solar capacity −1.9 %). INT-vs-FP32 comparisons are exact.
* The Demand heads are poor in FP32 already (negative R²), so they're not usable for anomaly detection.
* Partial panel damage (−40 %) isn't detectable from campus-total forecasts. Per-plant forecasting would need training, which was out of scope.
* The 24 h model's Grid R² is ≈ −0.1 on our data. The originally reported 0.741 appears to be a copy of the persistence-baseline value.

## 5. Next steps (if continued)
1. Per-channel Gemm on HTP: try `qairt-converter`'s newer path once the float-fallback issue is resolved, or a context binary (`qnn-context-binary-generator`). +1.5–3 dB in simulation.
2. Mixed precision: W8A16 everywhere except the per-step cell state, to see if W8A8 can be rescued for the 1 h/single models (2.5 ms → ~1.8 ms). Not needed for the 15-min cadence.
3. Context binary + an Android app (JNI) for a live demo instead of `qnn-net-run`.
4. Per-plant anomaly models (requires training) and multi-day residual aggregation for panel-degradation detection.
