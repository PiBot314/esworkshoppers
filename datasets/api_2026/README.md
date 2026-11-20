# Mentors' API history (synthetic 2026)

Everything the campus API (`http://10.2.203.214:9999`) serves, from its first day (2026-01-01) to yesterday, fetched by
[fetch_api_history.py](fetch_api_history.py) ([plan_19112026.md](../../archive/plan_19112026.md) B1). It is synthetic (below), so it
calibrates the live system and is never used for training. **Fetch complete** (20 Nov 07:24): 9 node folders, 7,752
hourly files, ~255 MB; rerunning tops up new days only.

| Folder | Contents | Request size |
|---|---|---|
| `nodes/<node_id>/<date>.json.gz` | Phase 1: the 9 model nodes (EM-CR-SB00-00/01, EM-CR-KH95-00, SL-PL00-00, SL-NI01-00, SL-VN95-00, SR-EM-KH04-01/02, WE-VN04-00), a list of rows per IST day | one node-day |
| `all/<date>/<HH>.json.gz` | Phase 2: all 180 nodes, the raw response per IST hour (`{vertical: {table, data: [rows]}}`) | one hour, all nodes |

**About the data: synthetic, not driven by real weather.** Solar output has a realistic day shape, season and level
(94 % of real 2025 for SL-PL00-00), but its day-to-day variation doesn't follow the weather: daily output vs ERA5
irradiance r = 0.21 for 2026 (no month above 0.46), against r = 0.80 for the real 2025 data. It isn't a shifted copy of
2025 either. The weather station `WE-VN04-00` is synthetic too (and empty in April–May). So use it for the live
system's behaviour (shadow scores, API-fitted limits, replays), not for weather experiments; weather-aware models need
real data.
Readings every ~10 min; don't use energy counters as model inputs.

**Running it again** (adaptive: up to 6 requests in parallel while response times stay normal, halving on slowdowns or failures, pauses only if it's still slow at 1; stops after 5 consecutive failures; resumable. Details in the script's docstring):

| Task | Command |
|---|---|
| Start / resume | `systemd-run --user --unit=nz-api-fetch --working-directory=$PWD -p StandardOutput=append:$PWD/fetch.log -p StandardError=append:$PWD/fetch.log /usr/bin/python3 -u $PWD/fetch_api_history.py all` |
| Progress | `tail fetch.log`; `find nodes all -name '*.json.gz' \| wc -l` |
| Stop | `systemctl --user stop nz-api-fetch` |

History: started 19 Nov 18:33, finished 20 Nov 07:24. Observed: the API keeps normal response times up to ~3–5 requests in parallel and slows 2–3× at 6; the script settles around that. Phase 1 ≈ 2,900 node-days at ~2 per second; phase 2 ≈ 7,700 hourly requests of ~5.5 s each (~3–5 h at 3–4 in parallel).

## Calibration set and M4 (20 Nov)

[m4_flag_rates.py](m4_flag_rates.py) builds `api_calib_15min.parquet` (31,008 bins × 39: every 15-min bin from
1 Jan to 19 Nov 2026 for the 7 nodes the live loop reads, built exactly like the live loop's `features.ts`) and runs
check **M4** (natural grid flag rate ≤ 3 %) for all 6 registered models in FP32 over the whole history (31k windows),
with injected grid faults (actual × 1.2 and × 0.6) for catch rates. Results: `m4_results.json`.

| Model | Grid R² on API | Flag rate, 2025 limits | Catch +20 % / −40 % | Flag rate, API-fitted limits* | Catch +20 % / −40 % |
|---|---|---|---|---|---|
| all_sensors_1h (deployed) | 0.69 | 9.10 % ✗ | 49 / 73 % | 1.87 % ✓ | 11 / 45 % |
| baseline_h4 | 0.65 | 12.87 % ✗ | 58 / 38 % | 1.77 % ✓ | 9 / 33 % |
| **guided_h4** | **0.75** | 6.83 % ✗ | 58 / 68 % | **2.24 % ✓** | **19 / 70 %** |
| baseline_24h | 0.52 | 3.36 % ✗ | 14 / 38 % | 0.95 % ✓ | 3 / 3 % |
| guided_24h | 0.56 | 13.66 % ✗ | 46 / 28 % | 1.26 % ✓ | 6 / 9 % |
| cnn_lstm_24h | 0.38 | 7.25 % ✗ | 26 / 20 % | 0.83 % ✓ | 2 / 3 % |

\* Same rule as `anomaly.py` (per IST hour: median, max(4 × 1.4826 × MAD, 2 % of capacity)), fitted on API residuals,
scored out-of-sample (2-fold, alternate weeks).

* **The earlier M4 pass (2.3–3.0 %) was a November-only result.** By month the deployed model flags 2.7 % in November
  but 9–35 % in Feb–May, when the API's grid load is highest (April mean 620 kW vs 370 kW in January). The 2025 limits
  were fitted on Aug–Oct validation residuals and don't cover the hot season.
* With API-fitted limits every model passes M4, but the 24 h models only by going blind (≤ 9 % of −40 % faults).
  **`guided_h4` with API-fitted limits is the best detector** (2.2 % false flags, 70 % of −40 % caught).
* The API-fitted limits are now in the models' `meta.json` files: next section.

### API-fitted limits applied (20 Nov)

[apply_api_limits.py](apply_api_limits.py) refits the same rule on **all** API windows (not cross-fitted) and writes
the **grid column only** into each model's `meta.json` (the live loop only flags grid: registry `"signals": ["grid"]`).
The 2025 limits stay in the same file as `anomaly_2025`, with an `anomaly_source` note; `--undo` restores them.
Summary: `api_limits_applied.json`.

| Model | Mean grid limit, 2025 → API (kW) | In-sample flag rate | Out-of-sample (M4 table above) |
|---|---|---|---|
| all_sensors_1h | 132 → 222 | 1.85 % | 1.87 % |
| baseline_h4 | 161 → 241 | 1.78 % | 1.77 % |
| guided_h4 | 142 → 173 | 2.26 % | 2.24 % |
| baseline_24h | 234 → 324 | 0.93 % | 0.95 % |
| guided_24h | 188 → 297 | 1.22 % | 1.26 % |
| cnn_lstm_24h | 263 → 368 | 0.80 % | 0.83 % |

* The files changed are local copies (`nz_live/models/*/meta.json`,
  `aimet_quantization/dashboard/www/data/all_sensors_1h/meta.json`). The live loop reads them **from the board**
  (`MODELS_DIR` / the NPU server): **not pushed yet**. How to push them: [SYSTEM_DESIGN.md](../../SYSTEM_DESIGN.md) §9.4.
* Wider limits mean fewer false flags and fewer catches: the 24 h models catch ≤ 9 % of −40 % faults with them.
  For detection use `guided_h4` (70 % of −40 % faults, 2.2 % false flags).
* The API data are synthetic; these limits calibrate the live demo, not the real campus. Refit when real data
  replaces the feed.
