# Energy forecasting on the ESW campus meter data

This folder holds everything we did with the ESW campus electricity data: cleaning it, forecasting it with six
different methods, and a few follow-on experiments (uncertainty, fault detection, federated training, a battery
simulation, and a deployment-readiness check). It is written so someone who has not seen the project can follow it.

**In one paragraph:** we took a year (2025) of readings from 11 campus meters (3 building-load meters, 5 solar
nodes, plus a substation and two room meters), put them on a common 15-minute grid, and compared how well six
methods predict the next 15 minutes, 1 hour and 24 hours of power. **LightGBM (a tree-based model) has the lowest
average error at all three horizons**, but the margins are small, and for solar 24 hours ahead, "same time yesterday"
is as good or better because we have no weather data.

**Read this first: what this is and is not.** This is a study on historical data and simulations. There is no
weather, HVAC, occupancy or carbon data, and we did not have the Qualcomm QIDK board. Nothing here is a field
result, and nothing here shows a building reaches "net zero". The test period is only two months (Nov-Dec 2025).

Contents: [1. The data](#1-the-data) - [2. The models](#2-the-models-what-each-one-is) - [3. How we tested](#3-how-we-tested-fairly) -
[4. Forecasting results](#4-forecasting-results) - [5. Follow-on experiments](#5-follow-on-experiments) -
[6. How to run it](#6-how-to-run-it) - [7. Folder map](#7-folder-map) - [8. Limitations](#8-limitations-and-assumptions)

---

## 1. The data

**Source.** The `ESW-Data` files: CSV exports from campus IoT energy meters, one row per sensor reading, with
`Node_id`, `Timestamp`, a firmware `Version` and the measurements. The columns mean different things in different firmware
versions, which is the main source of trouble in this data. There are three families of node:

| Family | Nodes we used | What it measures |
|---|---|---|
| **EM** (energy meters) | `EM-CR-SB00-00` (the substation: grid draw plus two transformers), `EM-CR-KH95-00` (a building load with R/Y/B phases) | Voltage, current, power, power factor, energy counters |
| **SL** (solar) | `SL-PL00-00`, `SL-VN95-00`, `SL-NI01-00` (3-phase meters); `SL-NI03-00`, `SL-NI03-01`, `SL-VN02-01` (inverters) | AC power, per-phase values, PV string voltage/current |
| **SR-EM** (room meters) | `SR-EM-KH04-00`, `-01`, `-02` | Power, energy, voltage/current |

`SB00-01`, `SL-VN02-00`, `SL-VN03-00` and `SL-PL96-00` were excluded (97% frozen readings, no measurements, or too
sparse). `metadata.xlsx` in the original data maps nodes to locations.

**How the raw data became model input** (`data/feature_selection.py`, notes in `data/FEATURE_SELECTION.md`):
1. Drop repeated ("frozen") frames and placeholder/sentinel values (-0.999, all-0.1 rows).
2. Fix known label problems: on `SL-VN95-00` active and reactive power are swapped; on `SR-EM-KH04-01/02` the
   "power factor" column actually holds frequency and "current" holds voltage. Units and signs are also unified
   (kW vs W, import vs export).
3. Drop columns that can be computed from others (for example power = voltage x current) so models do not see the same
   information several times.
4. Resample to a **15-minute grid**. Forward-fill gaps of up to 4 steps (1 hour). Longer gaps stay empty (NaN).
   Solar inverters send nothing at night, so missing night readings (20:00-05:00 IST) are filled with 0.
5. Add calendar features: hour of day, day of week and day of year as sine/cosine pairs (India time).

**What each node forecasts** (the "target"). All are converted to kW, non-negative, and solar generation is positive
(`SL-PL00-00` reports generation as negative in the source, so its sign is flipped).

| Node | Target | Share of time slots with data | Typical level |
|---|---|---|---|
| `EM-CR-SB00-00` | grid draw | 90% | ~464 kW average, up to 886 kW |
| `EM-CR-KH95-00` | total building power | 83% | ~0.46 kW (very small) |
| `SR-EM-KH04-00` | room power | 83% | ~7 kW |
| `SR-EM-KH04-01`, `-02` | room power | 83% | under 0.15 kW (tiny) |
| `SL-NI03-00`, `-01`, `SL-VN02-01` | solar AC power | 96-98% | ~13 kW average, up to ~70 kW |
| `SL-PL00-00`, `SL-VN95-00` | solar power | 97-98% | ~12-30 kW average |
| `SL-NI01-00` | solar power | 67% (low quality) | ~24 kW average |

`data/lstm_ready/` holds these tables (one Parquet file per node). `data/raw_by_version/` holds the raw per-node data
used for the data-quality checks.

---

## 2. The models: what each one is

We predict `y(T)`, the power at a target time `T`, from information available `h` steps earlier (`h` = 1, 4 or 96, i.e.
15 minutes, 1 hour, 24 hours). The first three are baselines: no learning at all. Any real model has to beat them.

| # | Model | How it works (plain language) |
|---|---|---|
| 1 | **Persistence** | "It will be the same as right now." Predicts `y(T) = y(now)`. Very hard to beat 15 minutes ahead, because power changes slowly. |
| 2 | **Same time yesterday** | Predicts `y(T) = y(T - 24 h)`. Captures the daily cycle. For solar 24 h ahead this is the same as persistence. |
| 3 | **Same time last week** | Predicts `y(T) = y(T - 7 days)`. Captures the weekly cycle (weekday vs weekend). |
| 4 | **Ridge regression** | A straight-line (linear) model: the forecast is a weighted sum of recent values, yesterday's and last week's values and the calendar features. "Ridge" adds a penalty on large weights (alpha = 10) so it does not overfit. Inputs are standardised and missing values filled with the median. |
| 5 | **LightGBM** | Gradient-boosted decision trees: hundreds of small trees, each one correcting the mistakes of the ones before it. It can capture non-linear rules like "if it is 2 pm on a weekday and yesterday was high, expect high". It handles missing values directly. We used up to 800 trees, learning rate 0.05, 31 leaves per tree, and stopped early when the validation error stopped improving. Trained to minimise absolute error (which matches the WAPE score). |
| 6 | **LSTM** | A neural network for sequences (Long Short-Term Memory). It reads the last 24 hours (96 steps) one step at a time, keeps an internal "memory", and at the end produces forecasts for all three horizons at once. One layer, 64 units, about 25,000 parameters. Trained with an absolute-error loss, AdamW optimiser, up to 20 epochs with early stopping. Missing values are set to 0 with a "was missing" flag. |

**Inputs given to Ridge and LightGBM** (built in `src/02_forecast.py`, function `features`):
- recent power at the forecast time: the last value and values 1, 2, 3 and 7 steps earlier; rolling averages over the last 1 hour, 4 hours and 24 hours;
- power at the same time yesterday, 2 days ago and a week ago, plus yesterday's daily average;
- extra measurements from that node (voltage, reactive power, transformer power...), as they were at forecast time;
- calendar features (hour, weekday, day of year) for the time being predicted.

Everything is built so a model never sees information from after the forecast time. The LSTM receives the same kinds of
inputs, but only as a 24-hour window, so it does **not** directly see last week's value.

---

## 3. How we tested fairly

- **Chronological split, no shuffling.** Train: before 1 Sep 2025. Validation (for stopping training / picking the best baseline): Sep-Oct. **Test: Nov-Dec 2025.** Random splits would leak the future into training.
- **One score: WAPE** = total absolute error divided by total actual power. 0.10 means the forecast is off by 10% of the energy on average. Lower is better. It stays well-defined for solar, where power is zero at night.
- **Same rows for every model.** A test row counts only if all methods could produce a forecast for it.
- The "best naive baseline" for each node is chosen on the validation months and then applied to the test months.

---

## 4. Forecasting results

Mean WAPE over 10 nodes (`SL-NI01-00` excluded because of poor data quality), Nov-Dec 2025. Lower is better.

| Horizon | Persistence | Yesterday | Last week | Ridge | **LightGBM** | LSTM |
|---|---|---|---|---|---|---|
| 15 min ahead | 0.163 | 0.353 | 0.416 | 0.186 | **0.135** | 0.161 |
| 1 hour ahead | 0.354 | 0.353 | 0.416 | 0.309 | **0.219** | 0.239 |
| 24 hours ahead | 0.353 | 0.353 | 0.417 | 0.391 | **0.297** | 0.303 |

Per-node tables: `results/forecast_model_comparison.md`. Chart: `results/figures/forecast_model_comparison.png`.
Raw numbers: `results/forecast_metrics.csv`.

**What the results say**
- **LightGBM has the lowest average error at every horizon, but the margins are modest.** At 15 minutes and 24 hours the LSTM is close behind. At 24 hours the LSTM wins at 2 of 10 nodes (`SL-VN95-00`, `SR-EM-KH04-02`).
- **15 minutes ahead, "same as now" is very hard to beat** on the big substation, on `EM-CR-KH95-00` and on `SR-EM-KH04-00`: persistence is as good or better. Models clearly help on solar and on the tiny room meters.
- **Solar 24 hours ahead: "same time yesterday" beats LightGBM at all 5 solar nodes** (and wins or ties against every model at 4 of them). Tomorrow's cloud cover is not predictable from past power alone, and we have no weather data. (We tried adding sun-position and neighbouring-node cloud signals in `src/08_pv_clearsky.py`. They did not beat "yesterday".)
- **Building loads 24 hours ahead benefit most from models**: `SR-EM-KH04-01` drops from 0.85 (yesterday) to 0.46, `EM-CR-KH95-00` from 0.54 to 0.34, the substation from 0.103 to 0.091.
- **Against the proposal's target** ("beat the best naive forecast by at least 10%"): LightGBM meets it at 6 of 11 nodes at 15 minutes, 8 of 11 at 1 hour and 3 of 11 at 24 hours (LSTM: 4, 6 and 2). Details: `results/forecast_h1a.csv`.

**Why the LSTM does not beat LightGBM.** We have not proved the reasons, but the likely ones are:
1. It sees less history: a 24-hour window, versus LightGBM's explicit values from 2 days and a week ago (loads here are strongly weekly). This is a confirmed difference in the setup.
2. It is small, trained once per node (one random seed) on about 8 months of data, so small gaps between it and LightGBM are within run-to-run noise.
3. It has to learn things LightGBM is handed directly, such as "the same time yesterday".
4. The problem is mostly "recent values + time of day + day of week", which suits tree models, and the test months (Nov-Dec) differ from the training months, which trees tolerate better than neural networks.
A fair test would add the week-ago value to the LSTM, run several seeds and try a larger network. We have not done this.

---

## 5. Follow-on experiments

These use the forecasts above. Each is in its own script; details and full tables are in `results/`.

**Data-quality rules** (`src/01_data_quality.py`). Deterministic checks on about 3 million raw rows: repeated frames,
placeholders, constant ("stuck") readings, out-of-range values, physics contradictions (e.g. power vs voltage x current),
swapped/shifted labels, generation at night, counters running backwards. They detect **10 of the 11 faults** we had documented
by hand. The miss is the `SL-NI01-00` March-April unit change (range checks flag only 2% of that window); it is visible
in a monthly check of the raw voltage (March is 1.8x the year's median).

**Prediction intervals** (`src/03_uncertainty.py`). Conformal prediction turns a point forecast into a range that should
contain the truth 90% of the time. Adaptive conformal inference hit 90.1% on average on test (89.9-90.3% at every node);
plain split conformal over-covered (96.9%). Intervals are narrow for the substation (about +/-11% at 15 min) and very wide for the tiny room meters.

**Drift and fault detection** (`src/04_drift_faults.py`). Monitoring forecast error to spot sensor faults and data drift.
On injected faults (25 events x 4 types x 6 nodes): stuck sensor 100% caught by a simple rule (33% by forecast error),
spikes 76%, dropouts 85%, **slow drift only 16%**, at about 1.5 false alarms per day, too noisy for real use. Our drift monitor
was **not clearly better than a naive distribution check**; this needs more work.

**Federated training** (`src/05_federated.py`). Each node trains locally and only model weights are averaged (no raw data leaves a node).
On 24-hour forecasts: plain federated averaging was **worse** (WAPE 0.55) than training locally (0.28) because the nodes are so different;
fine-tuning locally afterwards recovers local-model quality (0.28), so **no gain over training locally**. Communication was 66-99% smaller than sending raw data
(a rough proxy: file size on disk).

**Battery + solar simulation** (`src/06_mpc_carbon.py`). A model-predictive controller (re-solves a 24-hour optimisation every hour)
schedules a 400 kWh / 100 kW battery against the substation load minus solar, on the 49 days in Nov-Dec that had clean data.
Simulation only, with an assumed tariff (see Section 8):

| Controller | Change in monthly peak import | Change in cost |
|---|---|---|
| Simple rule (threshold peak-shaver) | -1.4% | -0.1% |
| MPC with our practical forecast (LightGBM load + "yesterday" solar) | -0.2% | -0.2% |
| MPC with a **perfect** forecast (upper bound) | -13.7% | -2.7% |

The takeaway: **with the forecasts we can build from this data, the MPC gained almost nothing; only perfect foresight gave a
large improvement.** Forecast quality (especially solar, without weather data) is the bottleneck. A carbon-aware variant using
a *synthetic* carbon-intensity curve we invented reduces carbon by ~1.7% (2.1% with perfect foresight): this shows the mechanism, not real emissions.
The battery does not reduce energy use; round-trip losses add about 0.2%.

**Deployment readiness** (`src/07_edge_benchmark.py`). We shrank the LSTM from 32-bit to 8-bit weights (INT8): file size 101 kB to 31 kB, accuracy change
+0.3% to +1.2% at short horizons. Forecast latency is under 1 ms on a desktop CPU (single thread). ONNX files are exported in
`results/edge/`. **These are not Qualcomm QIDK numbers**, and no energy per forecast was measured; that needs the device.

---

## 6. How to run it

Requires Python 3.10+ with `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `torch` (CPU is fine); the script installs the rest
(`pyarrow`, `lightgbm`, `onnx`, `onnxruntime`). About 45 minutes on 8 CPU cores, dominated by LSTM training.

```bash
bash run_all.sh                 # runs everything in order, results go to results/
# or step by step (from this folder):
.venv/bin/python src/01_data_quality.py     # data-quality rules
.venv/bin/python src/02_forecast.py         # THE forecasting comparison (writes results/preds/, forecast_metrics.csv)
.venv/bin/python src/02_forecast.py --no-lstm --nodes SL-NI03-00     # quick run: one node, no LSTM (much faster)
.venv/bin/python src/03_uncertainty.py      # prediction intervals (needs step 02)
.venv/bin/python src/04_drift_faults.py     # drift + fault injection (needs step 02)
.venv/bin/python src/05_federated.py        # federated training (needs step 02)
.venv/bin/python src/06_mpc_carbon.py       # battery simulation (needs step 02)
.venv/bin/python src/07_edge_benchmark.py   # INT8 / ONNX / latency
```

To rebuild the per-node 15-minute tables from raw data, run `data/feature_selection.py` (its header explains the inputs).
Every script uses fixed random seeds, but LSTM results can still vary slightly between machines.

---

## 7. Folder map

| Path | What is in it |
|---|---|
| `src/common.py` | Node list, unit/sign conventions, train/val/test dates, WAPE |
| `src/01_data_quality.py` ... `07_edge_benchmark.py`, `08_pv_clearsky.py` | The experiments above (numbered in run order) |
| `data/lstm_ready/` | Cleaned 15-minute table per node |
| `data/raw_by_version/` | Raw per-node data |
| `data/feature_selection.py`, `data/FEATURE_SELECTION.md` | How raw data became the cleaned tables, and every data issue found |
| `results/forecast_metrics.csv` | WAPE for every node x horizon x model |
| `results/forecast_model_comparison.md` | The comparison tables in readable form |
| `results/preds/` | Every model's forecasts per node (used by later steps) |
| `results/figures/` | All charts |
| `results/*.csv` | Numbers for each follow-on experiment |
| `results/edge/` | Exported ONNX models |

To share the folder, leave out `.venv/` (about 500 MB, can be rebuilt).

---

## 8. Limitations and assumptions

- **Test window is two months in one season.** Nothing here covers a full year.
- **No weather, HVAC, occupancy or carbon data.** Solar day-ahead forecasts cannot be good without weather. HVAC and comfort goals of the proposal cannot be tested.
- **The room meters are tiny** (`KH04-01/02` draw under 0.15 kW; `KH04-00` about 7 kW), so their high WAPE values are weak evidence.
- **`SL-NI01-00`** has low-quality data and is kept out of averages, drift, fault and federated experiments.
- **LSTM: one seed per node.** Small differences to LightGBM should not be over-read.
- **Battery simulation assumptions (all invented, replace before quoting):** tariff 8 INR/kWh base with time-of-day multipliers 0.9/1.0/1.25 and a 400 INR/kW-month demand charge;
  2 INR/kWh battery wear; 90% round-trip efficiency, 10-90% charge limits; average carbon factor 0.71 kg/kWh (approximate, please verify);
  the "synthetic diurnal" carbon curve is made up (plus or minus 25%, evening peak); 8 W assumed for the edge device.
  The substation (`SB00`) is treated as total campus load with the solar behind the same meter: **unconfirmed**. If the substation reading is already net of solar, solar is double counted.
  Days with missing data were excluded (49 replay days), and short solar gaps were filled from the other solar nodes.
- **Not done:** foundation models (Chronos, TimesFM, Moirai), running on the Qualcomm QIDK (latency or energy), and the safety guard
  we added to the battery controller had no effect in these runs.

**AI use disclosure.** The analysis code, experiments and this write-up were produced with Claude Code (Anthropic) working from the project's
reference paper and data. Results were generated by running the scripts in this folder; please review the assumptions above before relying on them.
