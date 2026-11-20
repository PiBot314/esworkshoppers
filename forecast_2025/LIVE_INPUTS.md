# What the 20 Nov models need live, and what the live loop has today

Checked 20 Nov against the live loop (`qidk_export/dashboard/src/lib/server/live/`: `features.ts`, `loop.ts`) and
with live requests to Open-Meteo. **The live loop fetches no weather and no calendar today**: it builds the old
models' 58 meter inputs from the campus API only. Every model from 20 Nov (LSTM and LightGBM, solar and grid) needs
weather forecasts, the clear-sky series, and for grid the academic calendar. This is the main blocker to deploying
them, more than the NPU.

| Input | Used by | Training source (2025) | Live source | Status |
|---|---|---|---|---|
| Meter history (SB00 grid; per-plant solar) | all | `data_15min.parquet` from the meter dump | campus API, as the live loop already reads | ✅ grid. ⚠ solar: the API serves PL00, NI01, VN95 only; NI03-00/01 and VN02-01 are neighbour proxies in the live loop, so the per-plant solar model would run on 3 real plants + 3 proxies (or on the 3 only) |
| Clear-sky GHI, sun zenith | all | `datasets/weather/clearsky.py` (computed; now covers 2015–2026) | computed, no network | ✅ port the formula (NOAA sun position + Haurwitz × monthly NASA POWER factor) or ship the 2026 table (`clearsky_15min.csv.gz`) |
| 15-min forecast: GHI, direct, diffuse, temperature (`fc15_*`) | 1 h models (+ "recent irradiance" input of all) | Open-Meteo historical forecast API, `minutely_15` | `api.open-meteo.com/v1/forecast`, `minutely_15=…`, `past_days=1` | ✅ tested 20 Nov: 288 steps (yesterday → tomorrow), no nulls, 3 s |
| Hourly forecast: cloud, low cloud, humidity, rain, wind (`fc_*`) | 1 h models | same, `hourly` | same request, `hourly=…` | ✅ tested: 72 h, no nulls |
| Forecast issued a day before (`d1_*`: GHI, cloud, temperature, humidity, rain) | 24 h models | Open-Meteo previous-runs API, `*_previous_day1` | `previous-runs-api.open-meteo.com/v1/forecast`, same variables, `forecast_days=2` | ✅ tested: values for every hour through tomorrow, 1.6 s |
| Anomalies vs normal (`anom_*_d1`) | 24 h LightGBM | d1 forecast − 2015–2024 ERA5 day-of-year normal | static normals table + the d1 values | ✅ ship the normals (366 rows) |
| Academic calendar: in session, holiday, exam, event | grid models | `datasets/training_2025/campus_calendar.py` (2025 almanacs) | — | ⚠ **2025 only.** The live feed is 2026: the 2026 almanacs must be transcribed the same way. Without it the grid models see "in session, no holiday" every day |
| Time features (hour, weekday, day of year) | all | computed | computed | ✅ |

## What the live loop would need

1. **A weather fetch every 15 min** (or hourly; the forecast only changes every 1–6 h): one `forecast` request
   (minutely_15 + hourly, `past_days=1`, `forecast_days=2`) and one `previous-runs` request, both ~2–3 s, cached.
   Align exactly as `datasets/training_2025/build_dataset.py` does (IST, value at *t* = mean over (*t* − 15 min, *t*]);
   a 15-min shift here is the kind of bug that silently costs R² (the old dataset was start-labelled, see E1).
   **Fallback when Open-Meteo is unreachable**: the board reaches the campus API through the hotspot; whether it reaches
   the internet is not confirmed. Without weather, use the `noweather` variants (on identical rows they lose
   ≤ 0.01 R² at 1 h and 0.01–0.06 at 24 h, `results/e1_compare.md`).
2. **The 2026 calendar** (`campus_calendar_2026.csv`), from the almanac.
3. **The clear-sky table** for 2026 on the board (or the formula in TypeScript).
4. **The new input layout**: `past [1, 96, F]`, `future [1, H, C]`, `node [1, n]` instead of the old single
   `[1, 96, 58]` tensor; names and order are in each model's `models/<model>.json`; for LightGBM in
   `models/grid_<h>_lgbm.json` (each feature with its live source).
5. **Anomaly limits for the new models**, fitted the same way on API residuals (`datasets/api_2026/apply_api_limits.py`).
   Note: the live rule checks the **+15 min** forecast (first step); E1 scores the value at the horizon (last step).
   For a 1 h model used as the live detector, its first-step accuracy is what matters (B3's "all steps" numbers
   bracket it).

The API data are synthetic: weather matched to the real 2026 sky will not explain synthetic load or solar. Expect the
weather inputs to help the live demo much less than they help on real 2025 data.
