# 08 — Verification Plan

Goal: prove that (a) the deployed model computes what the FP32 model computes, within a stated tolerance, (b) the remaining error comes from quantization and not from conversion or data bugs, and (c) the anomaly-detection use case still behaves the same.

## 1. Verification ladder (each rung isolates one source of error)
| Rung | Compare | Where | Pass criterion | If it fails |
|---|---|---|---|---|
| V0 | Rebuilt data → FP32 ONNX metrics vs `*_meta.json` | host ORT | R² within ±0.01 | Fix data rebuild (doc 05) |
| V1 | FP32 ONNX (ORT) vs AIMET pass-through sim | host | max \|Δ\| < 1e-5 | AIMET graph-handling issue (doc 04 §4.1) |
| V2 | FP32 ONNX vs AIMET quantized sim (E1–E5) | host | drop ≤ budget (0.01 R²) | QuantAnalyzer → bitwidth/AMP |
| V3 | FP32 ONNX vs QAIRT **FP32 DLC on device CPU** | QIDK | max \|Δ\| < 1e-4 | Converter issue (e.g. LSTM unroll, gate order) |
| V4 | AIMET sim (chosen) vs **HTP quantized** on device | QIDK | output SQNR ≥ 30 dB (W8A16) and metric drop within budget + 0.005 | Sim-to-target gap: inspect LSTM-internal encodings, try `--act_bitwidth 16` everywhere / float fallback for cell state, `qairt-accuracy-debugger` per-layer |
| V5 | HTP AIMET-quantized vs HTP QAIRT-only quantized | QIDK | AIMET ≥ QAIRT-only | If not better, report it honestly; keep the better one |

All comparisons use the **same TEST windows** (never used for calibration or AutoQuant scoring).

## 2. Metrics reported per rung
* Per domain (Solar/Grid/Demand): MAE, RMSE, nMAE (% of capacity), R², in **Watts**.
* Per horizon step (+15 … +60 min for 1 h; +15 min … +24 h for 24 h). Quantization error often grows with horizon.
* Output SQNR (dB) and max abs error vs FP32 in normalized units.
* Night-time solar leakage: fraction of night windows where predicted solar > 1 % capacity (physics sanity check).
* **Anomaly-flag agreement** (doc 02 §4): thresholds fixed from FP32 residuals on VAL. On TEST, compute precision/recall/F1 of INT flags against FP32 flags, plus agreement on injected faults:
  * *Sensor fault injection:* zero-out / flat-line / spike one solar node's channels in a test window.
  * *Panel-damage injection:* scale one plant's active power by 0.6–0.8 over a day.
  Expect FP32 and INT to flag the same injected events (target ≥ 95 % agreement).

## 3. On-device performance
| Measure | Tool | Report |
|---|---|---|
| Latency per inference (HTP INT, HTP FP16 if attempted, CPU FP32) | `qnn-net-run --profiling_level basic` + `qnn-profile-viewer`, `qnn-throughput-net-run` | median / p95 over ≥100 runs |
| Model size | file sizes of DLC / context binary | KB |
| Memory | `qairt-dlc-info` estimate + `dumpsys meminfo` of the process if feasible | |

These numbers **replace** the report's unverified "0.092 ms / 141,696 MACs" claims (doc 01 §5 #4).

## 4. Deliverables of verification
* `results/summary_table.md`: one row per model × configuration × rung.
* AutoQuant HTML diagnostics, QuantAnalyzer plots, per-horizon error plots.
* Device logs and raw outputs archived under `out/<model>/device/`.
