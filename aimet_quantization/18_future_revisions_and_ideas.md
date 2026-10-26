# 18 — Future Revisions and Ideas (living document, started 2026-10-06)

Purpose: collect ideas for the next iteration of the project in one place. Nothing in this document is implemented yet. Each idea gives the reasoning, a sketch of how it would work, and open questions. The short-term engineering items from [doc 15 §5](15_summary_lessons_and_next_steps.md) (per-channel Gemm, mixed precision, context binary) still apply and are not repeated here.

**Project direction (2026-10-15):** the spine of the project is *decisions, not metrics*. §7 sets the overall roadmap. §1–§5 are supporting ML work that should be justified by how much it improves a decision in §7.

## 0. Overview
| # | Idea | Problem it solves | Effort | Needs retraining |
|---|---|---|---|---|
| 1 | [Models that evolve with the system](#1-models-that-evolve-with-the-system) | Train-once models go stale (drift, new panels, scaler/encoding ranges) | M–L | yes (host side) |
| 2 | [Weather data as input](#2-weather-data-as-input) | Solar and 24 h forecasts only see the past and the clock | M | yes |
| 3 | [Campus almanac as input](#3-campus-almanac-as-input) | Demand/Grid depend on occupancy, which the models can't see | S–M | yes |
| 4 | [Quantization-aware training (QAT)](#4-quantization-aware-training-qat) | W8A8 collapses with PTQ, which blocks faster/cheaper models | M | yes |
| 5 | [Scalable per-node anomaly detection](#5-scalable-per-node-anomaly-detection) | Only one plant out of 11 nodes can be localised | M–L | yes |
| 7 | [Project direction: from forecasts to decisions](#7-project-direction-from-forecasts-to-decisions) | Forecasts and flags alone don't tell the campus what to do | — | depends on item |

S = days, M = 1–2 weeks, L = more.

---

## 1. Models that evolve with the system

### 1.1 Problem
The current models are trained once and frozen. Over time, two different things change:

* **Drift: same sensors, different behaviour.** Seasons, monsoon, semester timetables, a new HVAC unit, panels slowly degrading or getting dirty. The inputs stay the same; the patterns change. Retraining on recent data fixes this.
* **Structural change: new panels or nodes.** `all_sensors_1h/24h` take a fixed `[1, 96, 58]` input where every column belongs to a specific node ([model_meta.json](release/all_sensors_1h/model_meta.json)). A 7th solar plant has no column to go into. This changes the input shape, not just the weights, so it needs a new model.
* **Hidden range problem.** The `MinMaxScaler` is fit once ([model2/generate_lstm_dataset.py:92](../model2/generate_lstm_dataset.py#L92)) and the W8A16 activation encodings were calibrated on the same ranges. New capacity pushes inputs and targets above 1.0, and the NPU would clip exactly those values ([lesson 1 in doc 15](15_summary_lessons_and_next_steps.md): calibration-only ranges clip).

### 1.2 Where training happens
Not on the QIDK. It has no Python, and the HTP only runs inference (no backprop in QNN). The split:

* **QIDK:** inference, logging of incoming readings, drift monitoring.
* **Host:** retraining, then the existing AIMET → QNN pipeline (`scripts/01`–`10`), then push of a new `.so`.

The models are small (55 k–282 k parameters), so retraining takes minutes on a laptop. The slow, error-prone part, quantization and conversion, is already scripted.

### 1.3 Parts of a sound update loop
1. **Drift detection.** The residual layer flags single spikes. A *sustained* residual bias, or rolling MAE rising above the test-period baseline, means drift. Also track how often normalised inputs fall outside [0, 1], which measures range growth directly. Input-distribution checks (e.g. PSI per feature) are optional extras.
2. **Clean training data.** Readings flagged as anomalies, and imputed readings (`was_imputed`), must be kept out of retraining. Otherwise a dead sensor slowly becomes "normal" and stops being flagged.
3. **Data window.** Keep at least a full year so seasons aren't forgotten. Weight recent data more instead of training only on the last few weeks. Decide per retrain: fine-tune from the last checkpoint, or retrain from scratch (from scratch is safer when features change).
4. **Gating before promotion (champion/challenger).** The new model runs in shadow next to the current one. It is promoted only if it:
   * beats the current model on recent held-out data,
   * passes the same FP32-relative SQNR and anomaly-agreement checks as P4/P5 ([doc 08](08_verification_plan.md)),
   * comes with re-fitted residual thresholds (`anomaly_thresholds_W.npy` belongs to one specific model).

   The previous `.so` and thresholds are kept for rollback.
5. **Recalibrate quantization every time.** Even without retraining, re-running AIMET calibration on recent data is a cheap first fix when ranges grow.

### 1.4 Cheap adaptation on the board (no backprop)
* **Online bias correction:** the C++ server keeps an exponentially smoothed mean of recent residuals per domain and hour, and adds it to the forecast. A few lines of code; absorbs slow drift between retrains.
* **Re-fit the last layer on the CPU:** expose the `fc1` activations (48-dim for `all_sensors_1h`) as an extra graph output. The server re-fits `fc2` (48 → 12) by ridge regression in closed form on recent clean data. The LSTM stays frozen on the NPU, while the head adapts on the board.

### 1.5 Making new panels a non-event
This needs an input redesign. Either normalise solar by installed capacity (W per kWp, or campus totals plus a capacity feature), or use one shared per-node model (see [§5.3 option A](#53-options)). Both keep the NPU graph shape fixed as the campus grows.

### 1.6 Open questions
* How often do readings arrive without gaps, and how much data does a retrain need to beat the current model?
* Who approves a promotion? Is it automatic once the gates pass, or a manual step?
* Where does the host side run long-term? A laptop that suspends ([project gotcha](15_summary_lessons_and_next_steps.md)) is not a reliable retraining server.

---

## 2. Weather data as input

### 2.1 Problem
The models only see past sensor values and the clock (`hod/dow/doy` sin/cos). They can't know that tomorrow is cloudy. That's why the 24 h model is much weaker than the 1 h one (Solar R² 0.80 vs 0.94, 24 h Grid R² ≈ −0.12 on our data). The 1 h model mostly gets by on persistence.

### 2.2 Candidate features
| Feature | Why |
|---|---|
| Global horizontal irradiance (GHI), cloud cover | Main driver of solar output |
| Ambient temperature | Panel efficiency drops as panels heat up; cooling load drives Demand/Grid |
| Humidity, rainfall | Monsoon effects, cooling load, panel soiling (rain also cleans panels) |
| Clear-sky irradiance / solar elevation (computed, e.g. with `pvlib`) | Deterministic and free; lets the model predict the *clear-sky index* (actual / clear-sky) instead of raw power, which is easier to learn |

Already tried by the team (per-node benchmark, `src/08_pv_clearsky.py`): sun-position and neighbouring-node cloud signals, without weather data, did not beat "same time yesterday" for 24 h solar. So clear-sky features are worth adding only together with real cloud/irradiance forecasts, not instead of them.

Sources: an on-campus weather station if one exists, otherwise a public API such as Open-Meteo (historical data and forecasts). The QIDK has internet access ([doc 16 §1](16_dashboard_plan.md)), so the board could fetch forecasts itself. It needs an offline fallback (e.g. run the current weather-free model).

### 2.3 Train/serve skew
At inference time only weather **forecasts** for the next hours exist, not measurements. Training on measured weather makes the offline scores look better than what the deployed model can reach. Train on archived forecasts where possible, or at least report the skew.

### 2.4 Open questions
* Is there archived forecast data for the training period (2025)?
* How should weather enter the model? Past weather goes into the 96-step input window naturally. Future weather (the forecast for the horizon) needs a second input or extra per-step columns.

---

## 3. Campus almanac as input

### 3.1 Problem
Demand and Grid depend on how many people are on campus. The models only see weekday and season. Demand R² is negative even in FP32 (−0.34 reported, −0.66 on rebuilt data), and the dashboard shows Demand verdicts as "n/a" for that reason.

### 3.2 Candidate features
* Academic calendar: teaching weeks, exam weeks, vacations.
* Public holidays and festivals (IST, regional holidays included).
* Campus events (fests, conferences, admissions), hostel occupancy if available.
* Derived: "days to next holiday/break", "first week of semester".

All of these are known in advance, so there's no train/serve skew, unlike weather.

### 3.3 Caveat: data volume
The dataset has 32,866 aligned 15-min intervals, about 342 days, so the model sees each holiday and each semester break at most once. Almanac features risk overfitting until at least two years of data exist. Coarse flags (teaching / exams / break / holiday) are safer than fine-grained ones.

---

## 4. Quantization-aware training (QAT)

### 4.1 Why
PTQ to W8A16 already reaches 46–52 dB vs FP32, so QAT isn't needed for the current deliverable. QAT becomes interesting for:
* **W8A8**, which collapsed with PTQ on the NPU (24 h model: 14.7 dB, R² 0.34 → 0.17). 8-bit activations would cut latency and power.
* **Mixed precision** (doc 15 next step 2): QAT can make the network tolerate 8 bits in every layer except the sensitive one (e.g. the LSTM cell state).
* **Continual learning (§1)**: if we retrain regularly anyway, adding QAT to the training loop costs little.

### 4.2 How (sketch)
* QAT needs the PyTorch training code ([model2/train_*.py](../model2/)) with AIMET's PyTorch package (`aimet-torch` QuantSim + fine-tuning). Our current pipeline uses `aimet-onnx`, which we used only for PTQ. Check which QAT support the installed AIMET version has.
* Start from the trained FP32 weights, insert QuantSim with our range priors (LSTM hidden ∈ [−1, 1]), fine-tune for a few epochs at a low learning rate, export ONNX + encodings, then the existing conversion path (`08_qairt_build.sh`).

### 4.3 Experiment design
| Variant | Compare on the device |
|---|---|
| W8A16 PTQ (current baseline) | SQNR vs FP32, R², anomaly agreement, NPU ms |
| W8A8 PTQ (known to fail) | same |
| W8A8 QAT | same |
| Mixed precision QAT | same |

Measure on the QIDK, not only in simulation. Lesson 3 from doc 15: for recurrent models, simulation ≠ hardware, and only on-device measurement caught the W8A8 collapse.

---

## 5. Scalable per-node anomaly detection

### 5.1 Current state
* All three models run the residual check, but `all_sensors_1h/24h` check **campus totals** (Σ solar, grid). They can say "campus solar is off", not *which* plant.
* `single_node` is the only model that localises a fault, and it covers one node: `SL-PL00-00` (Palash), 1 of 6 solar plants. The 2 grid/substation nodes and 3 room meters have no per-node check at all, and Demand verdicts are "n/a".
* Partial panel damage (−40 %) is caught 0 % of the time from campus totals and 17 % with the single-node model ([explanation.md §5.2](../explanation.md)).

### 5.2 Why one LSTM per node doesn't scale
Compute is **not** the problem. 11 nodes × ≈4 ms ≈ 45 ms of NPU time every 15 minutes. What doesn't scale is everything around the model, multiplied by N:
* N training runs, N AIMET quantizations, N `.so` files, N graph prepares and HTP contexts in memory.
* N threshold sets and N drift monitors. With continual learning (§1), N retrain-and-gate cycles.
* A new node means running the whole pipeline again, and a new node has too little history to train on.
* Separate models ignore the strongest signal: plants under the same sky should behave alike.

### 5.3 Options
**A. One shared model per node *type*, run over all nodes (recommended).**
* `single_node`'s 10 inputs are already node-agnostic: no node ID, power as a fraction of capacity ([model_meta.json](release/single_node/model_meta.json)). Train it on the pooled data of all 6 solar plants instead of Palash only.
* Add static node features (capacity, inverter type, orientation) or a small node embedding so the model can tell plants apart.
* Inference: one model, N windows. Loop over nodes in a persistent HTP context (≈4 ms each), or export with a fixed batch of the maximum node count plus a mask.
* One model per node type (solar, grid, room meter) means **3 models no matter how many nodes**. A new panel is one more row in the batch, and it gets reasonable forecasts from day one.
* Complication: sensor schemas differ. `SL-NI03-00/01` and `SL-VN02-01` report `ac_voltage`/`pv_voltage` instead of reactive power and neutral current. Use the common subset (active power + time features), or masked optional features.

**B. Per-node outputs on the campus model (short-term stop-gap).**
Extend `all_sensors_1h` from 3 totals to one forecast per node (11 × 4 = 44 outputs). Still one model and one quantization, so it's cheap. But the input shape is still tied to the node list, so it fails the "new panels" test from §1.

**C. Peer comparison between plants (no forecaster needed).**
Normalise each plant's output by its capacity and compare it with the campus median of the other plants at the same moment. A plant that stays below its peers is flagged. This is very cheap, scales to any number of plants, and targets partial panel damage, which the LSTMs miss. Needs care when the whole campus is cloudy, and plants with different orientation need their own expected ratio per hour.

**D. Detectors that see all nodes at once.**
* Reconstruction-based: an autoencoder over all nodes; per-node reconstruction error localises the fault. The sibling `../project/` autoencoder is an existing starting point.
* Spatio-temporal graph neural networks treat nodes as graph vertices and handle a changing node count by design. More complex, and NPU support for their ops needs checking first.

**E. Temporal aggregation for slow faults.**
Run CUSUM or EWMA on each node's residuals over days. This finds degradation and soiling, which a single 15-min reading can never show (doc 15 next step 4).

### 5.4 Recommended combination
A (one shared model per node type) + C (peer comparison) + E (multi-day aggregation), with B only as a quick interim step. Together they cover all 11 nodes with 3 models, localise faults, and catch slow panel damage.

### 5.5 Open questions
* Are the 6 solar plants' data aligned well enough in time to pool for training and to compare as peers?
* Is a pooled model as good as the dedicated Palash model on Palash? This is the acceptance test for option A.
* Room meters have very different load profiles (negative Demand R² today). They may need the almanac features (§3) before any per-node detection is useful.

---

## 6. Suggested order
1. **Quick, no retraining:** online bias correction and drift monitoring in `nz_dashboard` (§1.4, §1.3.1); peer comparison between solar plants (§5.3 C).
2. **One retraining round** that bundles several ideas: pooled shared solar model (§5.3 A), almanac flags (§3), clear-sky features (§2.2); W8A16 PTQ through the existing pipeline.
3. **Host-side update loop** with gating and rollback (§1.3).
4. **Experiments:** QAT for W8A8 / mixed precision (§4), weather forecasts as inputs (§2).
5. **Before any of the above, check the order against §7:** each item should earn its place by improving a decision (solar sizing, load shifting), not only MAE.

---

## 7. Project direction: from forecasts to decisions
*Added 2026-10-15. This is the overall summary of what the project is about.*

### 7.1 Framing
**Spine: decisions, not metrics.** Reframe the project from "a forecaster with anomaly flags" to "a system that tells the campus what to do about energy, with the forecasts as support."

### 7.2 What exists now
* On-device forecasting on the QIDK (W8A16 on the Hexagon NPU): campus totals at 1 h (`all_sensors_1h`) and 24 h (`all_sensors_24h`), plus a 1 h model for one plant (`single_node`, Palash).
* An anomaly layer: a 15-min reading is flagged when its residual (actual − forecast) exceeds the accepted error. That limit is set per hour of day on the validation split. Solar and Grid only; Demand is "n/a".
* The `nz_dashboard` demo (replay, live NPU runs, fault injection).

### 7.3 Where it goes
1. **Solar sizing what-if (first prescriptive deliverable).** Load forecast plus irradiance (§2) feed a small linear program: how much of the load X kW of panels covers, how much is wasted or exported, and the payback period.
2. **Load shifting.** Use the 24 h forecast to move flexible loads into solar hours. Replay it on logged data: "this schedule would have saved Y kWh last month."
3. **Water treatment scheduling (only if the data is usable).** Treat the tank as a battery, use conformal prediction intervals, and add a tank-level safety constraint. First check the columns, the gaps and whether there is a daily rhythm.
4. **Supporting ML, in service of 1–3:**
   * Generalisation to new nodes without retraining: leave-one-node-out evaluation (ties to the shared model, §5.3 A).
   * Drift handling (§1).
   * The question that ties them to net zero: **does forecast error change the recommendation?**

### 7.4 Guardrails
* Measure success in kWh, rupees or CO₂, not just MAE.
* ConvLSTM and the RAG-SLM only if they clearly improve a decision, and only as a controlled comparison or an evaluated component.
* Eval hygiene: time-ordered splits, baselines, rolling-origin backtests, error analysis, ablations.

### 7.5 Honest pitch
> "We built the monitoring and forecasting layer. Waste is the cheapest cut, and the same forecasts drive solar sizing and load shifting next."

Net zero is mostly a systems problem. This project is the measurement layer that makes the systems argument concrete.
