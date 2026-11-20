# Weather and solar irradiance, IIIT-H campus

One point (17.4456 N, 78.3487 E, campus centre), 2025-01-01 → 19 Nov 2026, fetched 19 Nov by
[fetch_weather.py](fetch_weather.py) (rerun to extend; existing files are skipped). For the weather-aware models in
[plan_19112026.md](../../archive/plan_19112026.md) B1.

| File | Source | What | Period | Use |
|---|---|---|---|---|
| `archive_era5.json.gz` | Open-Meteo archive, ERA5 reanalysis (0.25°) | What the weather was: shortwave (GHI), direct, diffuse, DNI; cloud cover total/low/mid/high; temperature, humidity, rain, wind. Hourly, IST | → 2026-11-13 (~5 days behind) | Validation, upper bound ("perfect weather") |
| `hist_forecast.json.gz` | Open-Meteo historical forecast | What forecast models said at the time, same variables; plus **15-min** GHI/direct/diffuse/temperature (`minutely_15`) | → 2026-11-19 | Short-lead inputs |
| `previous_day1.json.gz` | Open-Meteo previous runs | The forecast issued **1 and 2 days before** each hour: GHI, cloud, temperature, humidity, rain | → 2026-11-19 | **24 h model inputs without train/serve skew** |
| `nasa_power.json.gz` | NASA POWER hourly (local solar time) | All-sky and **clear-sky** GHI, DNI, diffuse, cloud amount, temperature, humidity | 2025 complete; 2026 has −999 gaps in the last ~3 months | Cross-check; clear-sky series |
| `era5_2015_2024.json.gz` | Open-Meteo archive, ERA5 ([fetch_era5_history.py](fetch_era5_history.py)) | Same variables as `archive_era5`, hourly, IST | 2015-01-01 → 2024-12-31 | Climatology (normals, anomalies); simulated-plant pre-training |
| `clearsky_15min.csv.gz` | Computed ([clearsky.py](clearsky.py)) | Clear-sky GHI every 15 min (mean over the interval ending at the stamp): NOAA sun position + Haurwitz model, scaled per month to NASA POWER's clear-sky (factors 0.82 in July to 0.93 in November); zenith angle | 2015-01-01 → 2026-12-31 (extended back to 2015 on 20 Nov; 2025 values unchanged) | Clear-sky index (actual / clear-sky), solar features |

Notes:
* The Open-Meteo archive's default ("best_match") at this location is identical, hour for hour, to the historical
  forecast, so it isn't independent; `archive_era5` requests ERA5 explicitly.
* pvlib couldn't be installed on 19 Nov (PyPI downloads stalled on the network in use), so `clearsky.py` has no
  dependency on it. Checked: noon zenith = latitude at the equinox (17.5°), 6.1° at the June solstice, 40.9° in December.
* **Which source matches the campus:** the campus API's weather station `WE-VN04-00` (in `../api_2026/`) is synthetic
  and can't be used to check these. Against the **real 2025** Palash solar output, ERA5 gives r = 0.80 (daily) / 0.83 (hourly), the historical
  forecast 0.79 / 0.83: both usable, about equally good.
* **Real 2025 solar plants vs clear-sky** (centre of the average day; clear-sky 12:16 IST): NI01 −8 min, NI03-00 −8,
  NI03-01 −5, PL00 −13, VN02-01 −6, VN95 −3, so timestamps are right and panels face roughly south. Not usable: PL96
  (21 days, night readings 42 % of noon), VN02-00 and VN03-00 (timestamps only, no measurements). Units differ:
  `ActivePower` in W, `3_Active_power` in kW. VN95's active and reactive registers are swapped (handled in
  `aimet_quantization/scripts/01_rebuild_dataset.py`); an earlier note of a VN95 night offset came from the wrong
  register and was wrong.
* **Live use:** the same two forecast endpoints (15-min forecast, previous-day run) answer from the board over the
  hotspot (checked 20 Nov); what the live loop would need: [forecast_2025/LIVE_INPUTS.md](../../forecast_2025/LIVE_INPUTS.md).
