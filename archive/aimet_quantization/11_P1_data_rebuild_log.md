# 11 — P1 Log: Approximate Data Rebuild (executed 2026-10-05)

Scripts: `scripts/01_rebuild_dataset.py` (raw → `data/lstm_ready` → `data/lstm_normalized`) and `scripts/02_build_windows.py` (→ `data/windows/<model>/`).
Principle: reconstruct only the missing first stage. The second stage is model2's **original, unmodified** `generate_lstm_dataset.build_normalized_dataset()`, and windows come from model2's own `load_and_merge_all_sensors()`, `create_multisensor_windows()` and `create_sliding_windows()`, imported, not copied.

## 1. Reverse-engineering findings (each one checked against data)
| # | Finding | Evidence | Rule adopted |
|---|---|---|---|
| 1 | Cyclic time features were **also MinMax-scaled** by `generate_lstm_dataset` (it scales every column except `was_imputed`) | Decoding the sample's `hod_sin/cos` as plain sin/cos gave non-uniform 4-min steps. Decoding as `2x−1` gave exact 15-min steps | Nothing to change (stage B does it). Important for anyone feeding live data |
| 2 | The sample window `X_test[0]` starts **2025-11-10 04:00 UTC** (09:30 IST, Monday) | Decoded hod/dow, and `2025-01-23 04:15 + 27,935×15 min` (70/15/15 split of 32,866) gives the same instant | Confirms: UTC index, IST cyclic features, contiguous timeline |
| 3 | The parquet **index is server-ingest time**. For EM-CR-SB00-00 it only spans Aug–Nov (back-filled uploads, median 105 days off) | `Sensor_Timestamp` spans Jan 23–Dec 31 and is identical to the index for every other node | Use `Sensor_Timestamp` for all parquet nodes |
| 4 | EM-CR-KH95-00 in `parquet_sorted` has the wrong (substation) schema | `../raw_data/EM-Kh95-00.csv` has R/Y/B Power with the same 35,837 rows. `max(R+Y+B) = 1,739.4 W` = documented `power_total_W` max exactly | KH95 from the CSV: `power_total = R+Y+B` |
| 5 | **SL-VN95-00 active/reactive registers are swapped** | `3_Active_power` max 4.6, `3_Reactive_power` max 159 (documented active peak 156.3 kW). Solar capacity went 385.9 → 535.8 kW (target 546.0) | Swap them for VN95 |
| 6 | SL-NI01-00 reports Watts (not kW) in `3_Active_power` | \|min\| = 130,800 ≈ documented 130,781.6 W | Map directly to `active_power_W` |
| 7 | Inverters stop transmitting after dusk; the original carried the last AC reading through the night with `was_imputed=1`, but **not** the DC PV-string voltage (0 at night) | Step-by-step comparison with the sample window | Inverters: unlimited ffill except `pv_voltage_avg_V` (`ffill(limit=4)`). Meters: `ffill(limit=4)` |
| 8 | The original aligned timeline starts at 04:15. Our SB00 series had an extra 04:00 bin built from a single reading (04:14:09) | `progress.txt` §6 | Drop that one leading bin, which reproduces the original split boundaries |
| 9 | Feature column order inside a node differs from the original run (stage B appends kW→W columns) | names identical, order not | Reorder by the deployed model's `feature_names` |

## 2. Sanity anchors (informational under the approximate approach)
| Anchor | Original | Rebuilt | |
|---|---|---|---|
| Aligned timeline | 32,866 bins, 2025-01-23 04:15 → 12-31 12:30 | **identical** | ✅ |
| Test start (1 h/24 h) | 2025-11-10 04:00 UTC | **identical** | ✅ |
| `grid_max_W` | 887,250.77 | 887,250.8 | ✅ exact |
| `demand_max_W` | 30,045.70 | 30,045.7 | ✅ exact |
| `solar_max_W` | 546,010.3 | 535,813.4 | ≈ −1.9 % |
| Row counts SR-EM ×3 / VN95 / NI03-01 / VN02-01 / SB00 | 35,040 / 35,041 / 34,992 / 34,992 / 32,889 | identical | ✅ |
| Row counts PL00 / NI01 / KH95 / NI03-00 | 32,982 / 33,055 / 34,029 / 32,928 | 33,058 / 33,060 / 34,036 / 34,994 | small offsets in start/end trimming |
| Sample window feature match (\|Δ\| < 0.01 over all 96 steps) | 58 | **44 / 58** | see §3 |
| Single-node sample window | | not aligned (PL00 row count differs, so its split boundary shifts by 1.5 h) | expected |

## 3. Known residual differences (accepted)
`SL-NI03-01/VN02-01 ac_voltage` (right shape, different scale → a voltage spike in our max), `NI03-00` (3 bins with genuine zero readings in the original), `VN95 neutral_current` (scale ×~76), `PL00 reactive`, `NI01 voltage_avg` (outlier max 357 V), `SB00 lt1_voltage` (zero readings averaged in). They're single meter quirks (outliers or registers) in auxiliary channels, not in the power targets.
Why we stop here: under the revised approach (doc 05 banner), quantization is judged **INT vs FP32 on identical inputs**, so these differences don't bias the quantization result. They only shift the absolute R². §4 shows how much.

## 4. Outputs
`data/windows/<model>/{X_val,X_test,y_val,y_test,calib}.npy, info.json`, `data/windows/fidelity_report.json`.
| Model | train / val / test windows | calib |
|---|---|---|
| all_sensors_1h | 22,907 / 4,830 / 4,832 | 1,024 |
| all_sensors_24h | 22,815 / 4,738 / 4,740 | 1,024 |
| single_node | 23,041 / 4,859 / 4,861 | 1,024 |
Calibration windows are drawn from **train only**, stratified over the IST hour of each window's last step (night, dawn, noon, dusk equally represented), with seed 0.
