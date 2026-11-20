# Training dataset: real 2025 + weather (15 min)

Built by [build_dataset.py](build_dataset.py) for the next models ([plan_19112026.md](../../archive/plan_19112026.md) B1,
option C: real data for training and evaluation; the synthetic API history only calibrates the live system).

* `data_15min.parquet`: 35,040 rows (2025, every 15 min, IST) × 88 columns. `meta.json`: columns, capacities, status
  counts, split sizes, input hashes, build time.
* **Convention:** IST, naive timestamps; the value at *t* is the mean over (*t* − 15 min, *t*], the same as the weather
  files. Power in W, generation positive.

| Column group | What |
|---|---|
| `solar_<plant>_W` | Output of the 6 usable plants (PL00, VN95, NI01, NI03-00, NI03-01, VN02-01): sign fixed, 0 when the sun is down, daytime gaps ≤ 15 min filled (`_filled`), longer gaps NaN |
| `solar_<plant>_csi` | Clear-sky index = output ÷ (capacity × clear-sky GHI / 1000), when clear-sky GHI ≥ 50 W/m². Capacity = 99.5th percentile of daytime output (**estimated**, not nameplate). Values above 1 in winter are expected (tilted panels vs horizontal GHI) |
| `solar_<plant>_status`, `_peer_ratio` | Per day: the plant's clear-sky index ÷ the median of the other plants. `outage` < 0.15, `degraded` < 0.8, `unknown` if too little sunny data. **Train on `ok` days only.** Real faults: NI01 Jan–Apr (outage), partly May–Oct (degraded). Thresholds provisional (W4 tunes them) |
| `grid_W`, `kh95_W` | Substation `EM-CR-SB00-00` GDPower; `EM-CR-KH95-00` separately (a ~0.1–0.5 kW sub-circuit, not added to the grid) |
| `fc15_*` | 15-min historical forecast: GHI, direct, diffuse, temperature (short horizons) |
| `fc_*` | Hourly historical forecast: cloud cover (total, low), humidity, rain, wind |
| `d1_*`, `d2_*` | Forecasts issued 1 and 2 days earlier (**the 24 h model's inputs**) |
| `era5_*` | ERA5 actual weather: "perfect weather" upper bound only, not a live input |
| `clim_*_day`, `anom_*_d1` | 2015–2024 ERA5 normals for the day of year (±7-day smoothing); day-ahead forecast minus normal |
| `ghi_clearsky`, `zenith_deg` | Clear-sky GHI and sun angle |
| `hod/dow/doy_sin/cos`, `is_weekend` | Cyclic time features |
| `in_session`, `holiday`, `exam`, `event`, `term` | Academic calendar from the two official 2025 IIIT-H almanacs (`campus_calendar.py`; the PDFs are in `almanacs/`); `term`=`in_session` (teaching vs vacation). Holiday dates independently match the biggest real grid-load drops (Holi, Gandhi Jayanti, Independence Day, Diwali, Christmas…) |
| `split_blocked` | `train`/`val`/`test`/`gap`: 7-day blocks in 8 groups of ~6–7 weeks, one val and one test week per group (seed 0), 2-day gaps |
| `month` | For rolling-origin evaluation |

Checks (19 Nov): every plant aligns best with forecast irradiance at shift 0 (r 0.92–0.95, NI01 0.79 because of its
faults); midday clear-sky index median ≈ 1.0 (NI01 0.58); daily clear-sky index vs day-ahead cloud forecast r = −0.65;
no missing weather values; target coverage 84–100 %.

Calendar validation vs real grid load: in-session weekday median 512 kW; holidays 411 kW (80%, strong closures like Christmas 276 / Republic Day 305 down most); vacation weekdays 442 kW (86%); exam weekdays 547 kW (campus full).

## Simulated 2015–2024 pre-training set

Built by [simulate_plants.py](simulate_plants.py) (20 Nov). **Pre-training only; real 2025 stays the only train/test target.**

* `sim_2015_2024.parquet`: 350,688 rows (2015-01-01 00:15 → 2025-01-01 00:00, 15 min, same convention) × 27 columns:
  `sim_<plant>_W` for the 6 plants, `csi_era5`, `ghi_clearsky`, `zenith_deg`, all 12 `era5_*` weather variables,
  `hod/doy_sin/cos`, `year`, `month`. No grid load, no calendar (solar only).
* Model per plant: `gain(month, 15-min slot) × clear-sky GHI × ERA5 clear-sky index`. The clear-sky GHI gives the
  sub-hour sun shape; ERA5 (hourly, interpolated to 15 min) gives the clouds; the gain table (`sim_gain.csv`, 12 × 96
  per plant) is fitted on real 2025 `ok` days as Σ W / Σ(clear-sky × CSI), so it carries each plant's size, tilt,
  shading and seasonal angle effects and is energy-unbiased. NI01 is simulated as a **healthy** plant.
* Validation on 2025 (`sim_report.json`), 2-fold cross-fit by alternate weeks (gain fitted on one half, scored on the other):

| Plant | R² per 15 min | R² daily energy | Energy bias | Baseline (capacity × ERA5 GHI) R² 15 min / daily |
|---|---|---|---|---|
| PL00 | 0.77 | 0.57 | +0.7 % | 0.70 / 0.46 |
| VN95 | 0.81 | 0.57 | +0.3 % | 0.75 / 0.51 |
| NI01 | 0.58 | 0.36 | +0.8 % | 0.54 / 0.44 |
| NI03-00 | 0.79 | 0.55 | +0.1 % | 0.72 / 0.40 |
| NI03-01 | 0.80 | 0.56 | +0.2 % | 0.73 / 0.40 |
| VN02-01 | 0.81 | 0.59 | +0.1 % | 0.75 / 0.48 |

* **Limit:** day-to-day accuracy is capped by ERA5's clouds: ERA5 daily GHI correlates only r ≈ 0.80–0.82 with real
  daily output (R² ceiling ≈ 0.67), so the simulated years get the right shapes, seasons and variability statistics
  rather than the exact day. An earlier note of daily R² 0.70–0.75 was not out-of-sample; these numbers replace it.
  A median-over-clear-hours gain was tried and was ~6 % low on energy.
* Simulated annual energy 2015–2024 ranges ±5 % around the 2025 simulation (e.g. VN95 250–276 MWh vs 258 in 2025).
* For pre-training, the inputs are ERA5 (actual weather), not forecasts; fine-tuning on real 2025 with the
  forecast inputs (`d1_*`, `fc15_*`) removes that skew.

**Result of using it (20 Nov, [forecast_2025](../../forecast_2025/README.md)):** pre-training the solar LSTM on this set
and fine-tuning on real 2025 did **not** help (−0.01 to −0.02 R² at 1 h and 24 h; a lighter recipe chosen on validation
picked no pre-training). Kept for reference and climatology-style uses.
