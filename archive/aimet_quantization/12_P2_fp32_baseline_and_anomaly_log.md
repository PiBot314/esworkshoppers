# 12 — P2 Log: FP32 Baseline (V0) and Anomaly Layer (executed 2026-10-05)

Scripts: `scripts/03_fp32_baseline.py`, `scripts/anomaly.py`, `scripts/04_anomaly_baseline.py`. Shared helpers: `scripts/common.py`.

## 1. FP32 baseline on rebuilt TEST windows (V0, informational)
Model outputs → Watts with the **model's** capacities (from its metadata). Rebuilt targets → Watts with the **rebuilt data's** capacities. FP32 outputs for calib/val/test are cached in `out/<model>/fp32/` as the reference for all quantization comparisons.

| Model | Domain | R² rebuilt | R² reported | MAE rebuilt | nMAE |
|---|---|---|---|---|---|
| 1 h | Solar | **0.943** | 0.955 | 23.3 kW | 4.26 % |
| 1 h | Grid | **0.762** | 0.794 | 36.3 kW | 4.09 % |
| 1 h | Demand | −0.658 | −0.341 | 2.2 kW | 7.28 % |
| 24 h | Solar | **0.798** | 0.805 | 42.1 kW | 7.71 % |
| 24 h | Grid | −0.120 | 0.741 (see below) | 73.1 kW | 8.24 % |
| 24 h | Demand | −6.25 | n/a | 4.8 kW | 15.98 % |
| single node | Solar (PL00) | **0.950** | 0.957 | 1.96 kW | 2.98 % |

Reasoning:
* Solar R² comes within 0.007–0.012 of the reported values, and the 1 h Grid R² within 0.03. The approximate rebuild is representative, which is what the quantization study needs.
* **24 h Grid:** our MAE (73.1 kW, nMAE 8.24 %) agrees with the 24 h metadata's own MAE (74.3 kW, 8.38 %), but its R² of 0.741 doesn't fit that MAE. **0.7410 is exactly the persistence-baseline Grid R² in `report.md`**, so it was most likely copied by mistake. We report the 24 h Grid head as weak (R² ≈ −0.1 at a 24 h horizon).
* Demand heads are poor in FP32 (already flagged in doc 01) and excluded from the headline score: **headline R² = mean(Solar R², Grid R²)**, or Solar R² for the single node.

## 2. Anomaly layer (residual-based, no retraining)
`r = actual(+15 min) − forecast(+15 min)` per domain, in Watts. Flag if `|r − median_h| > K·1.4826·MAD_h` (K = 4, floor 2 % of capacity), with median and MAD estimated per **IST hour** on FP32 **validation** residuals. The hour is decoded from the window's own (scaled) `hod_sin/cos`, so the detector needs no timestamps.

Design iterations:
1. First version (K = 5, uncentred) flagged 48 % of Demand points → the Demand head is biased → **centre on the per-hour median**.
2. K = 5 → 4: dropout detection on the 1 h model went 32 % → 69 % with a natural false-flag rate of ≈2 % (Solar).

Synthetic faults (300 daylight test windows per model; solar actual only): panel damage (×0.6), sensor dropout (→0), sensor spike (+50 % capacity).
| Model | Natural flag rate (Solar / Grid) | Dropout | Spike | Panel damage −40 % |
|---|---|---|---|---|
| 1 h (campus total) | 2.3 % / 0.9 % | 69 % | 91 % | **0 %** |
| 24 h (campus total) | 1.1 % / 0.2 % | 8 % | 30 % | 0 % |
| single node (PL00) | 2.7 % | **92 %** | 83 % | **17 %** |

Interpretation (honest limits):
* **Sensor faults** (dropout/spike) are detectable from the 1 h and single-node forecasts.
* **Partial panel damage is essentially invisible at campus-total level.** A 40 % loss on ~10–60 kW of daylight output is within normal forecast error. Per-plant models do somewhat better (17 %) because the signal isn't diluted. Reliable damage detection would need per-plant models and multi-day aggregation of residuals. That's out of scope (no retraining).
* The 24 h model's +15 min step isn't a good short-term detector (it optimizes the whole day).
* The Demand head (49 % natural flags on test) **can't be used for anomaly detection.**

For quantization, the detector's role is a **behavioural check**: the INT models must raise the same flags as FP32 (doc 13).
