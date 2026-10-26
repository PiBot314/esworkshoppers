# Project History and Map: Edge AI for Net-Zero

One page to get oriented: what was built, in what order, by whom, which slide comes from which piece of work, and which numbers belong together.

*Written 2026-10-15. Dates before October are approximate (from file dates).*

---

## TL;DR

* **Goal:** forecast campus energy (Solar, Grid, Demand) from the IIIT-H meter data, run it on the Qualcomm QIDK board, and use the forecasts for fault detection now and energy decisions next.
* **There are four separate pieces of work.** Most confusion comes from mixing them up:
  1. **Data cleaning + EDA** (team)
  2. **Per-node benchmark**: 6 methods compared per meter, scored with **WAPE** (team)
  3. **Deployed LSTMs** (`model2/`): 3 campus-level forecasters, scored with **R²** (team)
  4. **AIMET quantization + QIDK deployment + dashboard** (`aimet_quantization/`), scored with **SQNR** against the FP32 model (this workspace)
* **Separate and out of scope:** an autoencoder anomaly detector in `../project/`.
* **New direction (15 Oct):** "decisions, not metrics". Next comes solar sizing and load shifting, measured in kWh and rupees.

---

## 1. Timeline

| When | What happened | Output |
|---|---|---|
| Jan–Dec 2025 | Campus meters log data. This is the period all the models train and test on | raw CSVs (`../esw/`, `../nodes/`) |
| Aug 2026 | Course starts. QIDK labs | `../20260806_lab01/` |
| ~10 Sep 2026 | Team receives the data dumps | `eswData.zip`, `nodes.zip` |
| Sep 2026 | **Team:** cleaning, EDA, per-node benchmark | workspace `../README.md`, `../report.md` |
| Sep 2026 | **Team:** autoencoder anomaly detector, put on the QIDK as an SNPE INT8 model | `../project/` |
| Sep 2026 | **Team:** 3 multi-sensor LSTMs trained and exported to ONNX, with a deployment plan that turned out not to work (it needed Python on the board) | `model2/` |
| Sep 2026 | **Team:** crawler for the oneM2M API (1,647 resources) | slide 4 |
| 4–5 Oct 2026 | **AIMET quantization** of the 3 LSTMs (W8A16), **running on the QIDK NPU**, verified against FP32 | `aimet_quantization/` docs 00–15 |
| 5 Oct 2026 | **Dashboard** served by the QIDK itself (C++ server, live NPU runs, fault injection) | docs 16–17, `dashboard/` |
| 6 Oct 2026 | Team deck and technical deck merged into one revised deck. Future-ideas doc and demo recording script written | `deck_revision.md`, doc 18, `recording_flow.md` |
| 8 Oct 2026 | Viva question banks and reading list | `viva_questions*.md`, `reading_list.md` |
| 15 Oct 2026 | Models re-pushed to the QIDK. oneM2M portal 400 errors explained. **Project reframed: decisions, not metrics** | `onem2m_portal_notes.md`, doc 18 §7 |

---

## 2. The four pieces of work

### 2.1 Data cleaning and EDA (team) · slides 4–14
* **Source:** IIIT-H oneM2M testbed exports. 11 usable nodes: 6 solar, 2 grid/substation, 3 room meters.
* **Fixes:** firmware versions with different column meanings were split and realigned; 4 frozen nodes were dropped; SL-PL00-00's inverted wiring was sign-corrected; W vs kW mismatches were standardized; sentinel values (−0.999) were removed; missing night-time inverter readings were filled as 0 W.
* **Result:** a 15-minute grid with sin/cos time features (hour, day of week, day of year).

### 2.2 Per-node benchmark (team) · slides 15–19
**Question:** which forecasting method works best for each meter?

| | |
|---|---|
| Models | 6: persistence, same time yesterday, same time last week, Ridge, LightGBM, LSTM |
| One model per | node, forecasting **that node's own power** |
| Horizons | 15 min, 1 h, 24 h |
| Split | train before 1 Sep 2025 · validation Sep–Oct · test Nov–Dec |
| Metric | **WAPE** (lower is better), averaged over 10 nodes (SL-NI01-00 excluded) |
| Its LSTM | 1 layer, 64 units, ≈25 k parameters, absolute-error loss, AdamW |
| Result | **LightGBM is best on average** but by small margins. "Same time yesterday" wins for 24 h solar, because there's no weather data |
| Where | workspace `../README.md`. The `src/` code isn't on this machine (it's probably on a teammate's) |

### 2.3 Deployed LSTMs (team, `model2/`) · slide 20
**Question:** forecast the whole campus balance for net-zero planning.

| Model | Forecasts | Inputs | Parameters | FP32 R² (rebuilt data) |
|---|---|---|---|---|
| `all_sensors_1h` | campus Solar, Grid, Demand, next 1 h (4 × 15 min) | 58 features × 96 steps | 140 k | 0.852 |
| `all_sensors_24h` | the same, next 24 h (96 × 15 min) | 58 × 96 | 282 k | 0.339 |
| `single_node` | Palash plant (SL-PL00-00), next 1 h | 10 × 96 | 55 k | 0.950 |

* Architecture: 2 stacked LSTM layers plus a small dense head, squared-error loss, Adam.
* Split: **70 / 15 / 15 in time order**, test from 10 Nov to 31 Dec 2025.
* R² here is the **mean of Solar and Grid**. Demand is excluded because its R² is negative even in FP32.
* **These are not the benchmark LSTM.** They're bigger, multi-sensor models of campus totals.

### 2.4 AIMET quantization, QIDK deployment and dashboard (`aimet_quantization/`) · slides 21–25
**Question:** can the deployed LSTMs run on the QIDK's NPU without losing accuracy and without retraining?

1. **Data rebuild:** the team's normalized dataset wasn't on this machine, so it was rebuilt from raw data using the team's own script. That's why it's called "approximate data".
2. **Anomaly layer** (no new model): a 15-min reading is flagged if |actual − forecast − hourly median| > 4 × robust σ, with the limits set on validation data.
3. **AIMET PTQ, W8A16:** 8-bit weights, 16-bit activations. Uses AdaRound, plus range priors (LSTM states always lie in [−1, 1]). Calibrated on 1,024 training windows.
4. **Conversion and running on the device:** `qnn-onnx-converter` → model `.so` → `qnn-net-run` on the HTP (QAIRT 2.47).
5. **Dashboard:** a C++ server on the QIDK. It replays the test period, injects faults and runs windows live on the NPU.

| Model | NPU time | SQNR vs FP32 | R² FP32 → NPU | Same anomaly verdicts |
|---|---|---|---|---|
| 1 h | 3.42 ms | 48.4 dB | 0.852 → 0.853 | 99.66 % |
| 24 h | 3.44 ms | 46.4 dB | 0.339 → 0.335 | 99.76 % |
| single | 3.93 ms | 52.1 dB | 0.950 → 0.950 | 99.92 % |

**Out of scope:** the `../project/` autoencoder. It's a real anomaly detector, already on the QIDK, but you chose to work on the LSTMs instead.

---

## 3. The deck, slide by slide

Revised deck, 27 slides: `Edge AI for Net-Zero (revised).pptx`.

| Slide | Topic | Comes from | Key numbers |
|---|---|---|---|
| 1 | Title | – | – |
| 2 | Progress checklist | all | ⚠ "problem statement" is still unticked |
| 3 | Net-zero question: net deficit, self-sufficiency | `model2/` framing | ⚠ defined, but no value is ever shown |
| 4 | Data source: oneM2M crawler | team crawler | 1,647 resources, 9 device families |
| 5–7 | Data analysis, auditing, repeated readings | 2.1 cleaning | 97 % / 41 % / 17 % repeated frames |
| 8 | 11 nodes → 58 features | 2.3 deployed LSTMs | 41 readings + 11 filled-in flags + 6 time features · 70/15/15 split |
| 9 | What the sensors measure | 2.1 | – |
| 10–14 | Graphs: KH95 gap, smart room, solar, demand, seasons | 2.1 EDA | spring ≈53 kW vs summer ≈37 kW midday solar |
| 15–16 | The 6 models explained | **2.2 benchmark** | LSTM: 1 layer, 64 units, 25 k parameters |
| 17 | Forecasting results | **2.2 benchmark** | WAPE table, e.g. 1 h: LightGBM 0.219, LSTM 0.239 |
| 18 | What the results mean | **2.2 benchmark** | "beat naive by ≥10 %" at 6/8/3 of 11 nodes (LightGBM) |
| 19 | Research backing | literature | – |
| 20 | Three deployed LSTMs | **2.3 deployed** | 140 k / 282 k / 55 k parameters · R² 0.852 / 0.339 / 0.950 |
| 21 | Anomaly detection | 2.4 anomaly layer | dropout 69 %, spike 91 %, panel damage 0 % caught (campus) |
| 22 | Shrinking for the NPU | 2.4 AIMET | 44 → 78 dB (activations only, simulation) · ≈4× smaller |
| 23 | On the board | 2.4 device | 3.4 ms · 48 / 46 / 52 dB · 99.7 % same verdicts |
| 24 | Live demo | 2.4 dashboard | ≈5 ms compute, ≈2 s end to end |
| 25 | What we learned | 2.4 | W8A8: 41 dB in simulation vs 15 dB on the device |
| 26 | Done and next | all | – |
| 27 | Growing with the campus | doc 18 | – |

**The big boundary is between slides 19 and 20.** Before it, "LSTM" means the small benchmark model. After it, it means the deployed campus models.

---

## 4. Easily confused things

| Looks the same | But actually | Why it matters |
|---|---|---|
| "The LSTM" on slide 16 vs slide 20 | Two different model families: per-node 25 k parameters vs campus 140 k / 282 k / 55 k | An examiner may see a contradiction |
| WAPE (slide 17) vs R² (slide 20) | Different targets, metrics and splits | They **can't be compared** |
| Benchmark split vs 70/15/15 | Test from 1 Nov (benchmark) vs from 10 Nov (deployed) | – |
| "LSTM beat the baselines" | Deployed 1 h model: Solar R² 0.955 vs persistence 0.908. That's the team's original report figure | In the benchmark, the LSTM met the target at only 4/6/2 of 11 nodes |
| Solar R² 0.955 vs 0.943 | 0.955 is the team's figure; 0.943 is the same model on the rebuilt data | Slide 20 uses rebuilt values (0.94) |
| 24 h Grid R² 0.741 | A copy error: it's the persistence baseline's number. The real value is ≈ −0.12 | ⚠ if it still appears anywhere, remove it |
| 78 dB (slide 22) vs 48 dB (slide 23) | 78 = activations only, simulation. 48 = full W8A16 model on the device | Once activations were fixed, the 8-bit weights became the main error |
| SQNR vs R² | SQNR compares quantized vs FP32. R² compares the model vs real readings | SQNR isolates what quantization changed |
| "Anomaly detector" | The LSTMs are forecasters. Flags come from forecast error | Why: inherited forecasters, no retraining allowed, no labels |
| 3.42 ms vs 5.1 ms vs ~2 s | NPU only · plus CPU↔DSP transfer · plus graph preparation on each dashboard run | – |
| "≈4× smaller" | Measured 3.8× (biases and some tensors stay higher precision) | – |

---

## 5. Metrics cheat sheet

| Metric | Measures | Used in |
|---|---|---|
| **WAPE** = Σ\|error\| ÷ Σ\|actual\| | Forecast error as a share of energy | Benchmark (slides 17–18) |
| **R²** | 1 = perfect, 0 = no better than predicting the mean, < 0 = worse | Deployed models (slides 20, 23) |
| **MAE / RMSE / nMAE** | Typical error in kW, big misses weighted more, % of capacity | Project docs; "≈4 %" on slide 23 |
| **SQNR (dB)** | Quantized vs FP32 output. > 40 dB is very good, < 20 dB is broken | Slides 22, 23, 25 |
| **Flag agreement** | % of readings where the quantized and FP32 models give the same verdict | Slide 23 |
| **Detection rate** | % of injected faults caught | Slide 21 |
| **False-flag rate** | % of normal readings flagged (Solar ≈2–3 %, Grid ≈1 %) | Slide 21 |
| **Robust σ** = 1.4826 × MAD | Spread that ignores outliers; used for the thresholds | Slide 21 formula |

---

## 6. Where things live

| Path (relative to `mideval/`) | What |
|---|---|
| `model2/` | The team's deployed LSTMs (ONNX, training scripts, original report). **Don't modify** |
| `aimet_quantization/` | All quantization and deployment work. Start at `README.md` |
| `aimet_quantization/00–15_*.md` | Plan and logs, in order |
| `aimet_quantization/16–17_*.md` | Dashboard plan and build |
| `aimet_quantization/18_future_revisions_and_ideas.md` | Future ideas. **§7 = project direction** |
| `aimet_quantization/scripts/` | Pipeline scripts 00–14 |
| `explanation.md` | Full technical explanation of the deployment work |
| `deck_revision.md` | How the revised deck was assembled |
| `presentation_flow.md`, `recording_flow.md` | Talk script, demo recording script |
| `viva_questions.md` / `_deck.md` / `_choices.md` | Question banks: theory, slides, "why this number" |
| `reading_list.md` | Papers |
| `onem2m_portal_notes.md` | oneM2M web portal access and the 400 error fix |
| `../README.md` | The per-node benchmark write-up |
| `../project/` | The autoencoder (out of scope) |
| QIDK `/data/local/tmp/nz_aimet` | Runtime, models, test data, dashboard on the board |

**Run one forecast on the board** (from `aimet_quantization/`):
```bash
../.venv-aimet/bin/python scripts/12_qidk_forecast.py --model all_sensors_1h --test-index 10
```
**Start the dashboard:** `scripts/14_dashboard_start.sh`, then open http://localhost:8080.

---

## 7. Where it goes next

Full version: doc 18 §7.

1. **Solar sizing what-if:** load forecast plus irradiance feed a small LP that answers how much X kW of panels covers, what's wasted, and the payback. Needs an irradiance data source first.
2. **Load shifting:** move flexible loads into solar hours, replayed on logged data to give kWh saved.
3. **Water treatment scheduling:** only if that data turns out to be usable.
4. **Supporting ML:** leave-one-node-out generalisation, drift handling. The test for all of it: *does forecast error change the recommendation?*

**Pitch:** "We built the monitoring and forecasting layer. Waste is the cheapest cut, and the same forecasts drive solar sizing and load shifting next."
