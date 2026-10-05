# 05 — Rebuilding Calibration and Evaluation Data from Raw

Decision (yours): rebuild `lstm_normalized/` from `../nodes/parquet_sorted/` instead of fetching the teammate's copy.

> **Revision 2026-10-05: lighter "approximate rebuild" approach adopted.**
> * The FP32 accuracy we **quote** is the original one from `model2/benchmark_results.json` and the `*_meta.json` files. We don't try to reproduce it exactly.
> * The rebuild only needs to be **representative** (right features, right order, roughly right scaling, all seasons and day/night). The anchors in §3 become **sanity checks** (log the deviation), not pass/fail gates.
> * Quantization loss is measured **relative to FP32 on the same rebuilt windows** (INT vs FP32 outputs). That needs no labels and is insensitive to small normalization mismatches, because those affect both models equally.
> * R² vs ground truth on the rebuilt data is still reported, labelled **"approximate data"**.
>
> The rest of this document is kept as the original reasoning. §2 still describes what the rebuild contains, and §5 (calibration-set design) still applies in full.

## 1. Why the data is non-negotiable
* **Calibration:** AIMET computes activation ranges (and AdaRound optimizes rounding) from representative inputs. Two sample windows aren't enough: one day/night cycle with no seasonal spread would clip real inputs on device.
* **Evaluation:** AutoQuant's `eval_callback` and `allowed_accuracy_drop` need a held-out labelled set. Without one, "minimal loss of accuracy" can't be measured.
* **Fidelity:** the models are frozen. They only behave as reported if inputs are normalized **exactly** as in training (same per-column min/max, same column order, same fill rules). A scaler mismatch shows up as accuracy loss that looks like quantization error but isn't.

## 2. What must be reconstructed
The missing step is `by_version/ → lstm_ready/{node}.parquet + features.json`. `generate_lstm_dataset.py` (present) does the rest. From the code and `progress.txt` §3–§5, `lstm_ready` must contain, per node:

| Node | Columns expected in `lstm_ready` (pre-normalization) | Raw source columns (`parquet_sorted`) |
|---|---|---|
| SL-PL00-00, SL-VN95-00, SL-NI01-00 (V3.0.0 meters) | `active_power_kW`, `reactive_power_kVAr`, `voltage_avg_V`, `neutral_current_A`, `was_imputed`, 6×cyclic | `3_Active_power`, `3_Reactive_power`, `Voltage_avg`, `NPhaseLC` |
| SL-NI03-00, SL-NI03-01, SL-VN02-01 (V2.0.1 inverters) | `active_power_W`, `ac_voltage_avg_V`, `pv_voltage_avg_V`, `was_imputed`, cyclic | `ActivePower`, mean(`Voltage_RS/ST/TR`)? or mean(`Voltage1..3`)?, mean(`PV1..6Voltage`)? **(to infer)** |
| EM-CR-SB00-00 | `gd_power_W`, `t1_power_W`, `t2_power_W`, `hv_voltage_VLL`, `lt1_voltage_VLN`, `lt2_voltage_VLN`, `was_imputed` | `GDPower`, `T1Power`, `T2Power`, `GDVLL`?, `LT1VLN`, `LT2VLN` (units kW→W?) **(to infer)** |
| EM-CR-KH95-00 | `power_total_W`, `power_R_W`, `power_B_W`, `voltage_V`, `power_factor`, `was_imputed` | parquet has GD/T1 columns only. The R/Y/B-Power schema is in `../raw_data/EM-Kh95-00.csv` / `../esw/EM.csv` **(to verify)** |
| SR-EM-KH04-00 | `power_W` (+ `was_imputed`) | `Power` (kW per report → ×1000) |
| SR-EM-KH04-01/02 | `power_W`, `voltage_V` (raw "Current"), `freq_col_raw` (raw "Frequency"), (+ `frequency_Hz` from raw "Power Factor"), `was_imputed` | `Power`, `Current`, `Frequency`, `Power Factor` |

Processing rules to replicate (from `progress.txt`): convert epoch seconds → datetime, resample to **15 min mean**, `ffill(limit=4)`, `was_imputed` = 1 where the value was filled, cyclic features in **IST (UTC+5:30)**, then `generate_lstm_dataset.py` (abs, kW→W, renames, `fillna(0)`, MinMax).

Items marked "to infer" are resolved empirically by matching the anchors below. Each candidate mapping is accepted only if it reproduces the anchor values.

## 3. Verification anchors (acceptance criteria for the rebuild)
| Anchor | Source | Pass criterion |
|---|---|---|
| Per-node row counts (e.g. SL-PL00-00 = 32,982; SR-EM-KH04-00 = 35,040) | `progress.txt` §5 | exact or ±1 bin |
| Per-node target max (e.g. SL-PL00-00 65,808.2 W; EM-CR-SB00-00 886,356.1 W) | `progress.txt` §5, `qidk_model_meta.json` `scaler_max_W`=65808.15 | ≤0.1 % |
| Aligned intersection = **32,866** intervals, 2025-01-23 04:15 → 2025-12-31 12:30 | `progress.txt` §6 | exact |
| Target capacities `solar_max_W`≈546,010.26, `grid_max_W`≈887,250.77, `demand_max_W`≈30,045.70 | `all_sensors_qidk_meta.json` | ≤0.01 % |
| **Window equality**: rebuilt `X_test[0]` vs `sample_input_all_sensors_1x96x58.npy` | 1 h script saves `X_test[:1]` | max \|Δ\| < 1e-4 per feature (feature-wise diff report) |
| Same for the 24 h sample and the single-node `sample_input_1x96.npy` | respective scripts | same |
| **FP32 metric reproduction**: rebuilt test set through the frozen ONNX gives Solar R² ≈ 0.955, Grid R² ≈ 0.794 (1 h) | meta JSONs | within ±0.01 R² |

The last anchor is the decisive one. If FP32 metrics reproduce, the data is good enough for calibration and eval, even if some feature matches only approximately.

**If the anchors can't be met** (likely candidates: the KH95 schema, the inverter voltage aggregation), there are two options:
(a) accept an approximate rebuild and report the FP32 metrics on the *rebuilt* data as the new baseline. Quantization loss is always measured INT vs FP32 **on the same data**, so the comparison stays valid.
(b) ask the teammate for `lstm_normalized.zip` after all.
Option (a) keeps the study internally consistent. It just can't claim the original numbers.

## 4. Output layout (all new files; nothing in `model2/` is modified)
```
mideval/aimet_quantization/data/
  lstm_ready/            reconstructed intermediates
  lstm_normalized/       {node}.parquet, scaler_params.json, features.json
  windows/
    calib_1h.npy         calibration windows  (from TRAIN split)
    val_1h_X.npy/y.npy   threshold/AutoQuant eval (VAL split)
    test_1h_X.npy/y.npy  final reporting (TEST split, never used for tuning)
    (same for 24h and single-node)
  rebuild_report.md      anchor-by-anchor pass/fail table
```
The `generate_lstm_dataset.py` logic is **imported or copied, not edited**, with `INPUT_DIR`/`OUTPUT_DIR` redirected.

## 5. Calibration set design
* Source: **train split only**, which avoids leaking test data into encodings.
* Size: **1,000–2,000 windows**, batch 1 (the model is static B=1; AIMET forward passes are cheap at ~13 M MACs).
* **Stratified** across hour-of-day (night zeros, dawn ramps, noon peaks), weekday/weekend, and months (monsoon vs summer irradiance). Include windows with `was_imputed=1` stretches and zero-filled outages, because those happen in production and are where ranges get set.
* Also write the same windows as float32 `.raw` files + `input_list.txt` for the QAIRT quantizer, which needs them for the LSTM-internal tensors AIMET doesn't cover (doc 07).
