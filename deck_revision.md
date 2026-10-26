# Deck revision: "Edge AI for Net-Zero" (mid-eval)

Base: the team deck (`Edge AI for Net-Zero.pdf`, 25 slides). The technical deck (`Edge_AI_Net_Zero_MidEval.pdf`) is used as a source only: its content is rewritten below in the team deck's voice, with the numbers kept and a plain-words "so what" for each one.

How to read this: each slide has an **action** (keep / edit / new / move / backup), the **on-slide text**, and an optional **say** line for the speaker. On-slide text is kept short on purpose; the context goes in what you say.

Tone rules used throughout:
* Every number comes with what it means ("3.4 ms, so the board is idle almost all of the time").
* Problems are told as "we found X, we fixed it with Y" or "next we'll do Z", never as bare admissions.
* Jargon (SQNR, AdaRound, range priors) appears at most once per slide, with a plain description next to it.

---

## Running order at a glance

| New # | Slide | Source | Action |
|---|---|---|---|
| 1 | Title | team 1 | keep |
| 2 | Progress since the proposal | team 2 | edit |
| 3 | The question: how close is campus to net-zero? | team 15 + tech 2 | new (merges the equations slide) |
| 4 | Where the data comes from | team 18 + tech 4 | move + edit |
| 5 | Data analysis | team 3 | keep |
| 6 | Data auditing | team 4 | keep |
| 7 | Data quality: repeated readings | team 22 | move |
| 8 | 11 nodes, 58 features | team 8 + tech 8 | replace duplicate slide |
| 9 | What the sensors measure | team 7 | keep |
| 10–14 | Graphs and insights (KH95 gap, smart room, solar, demand, seasons) | team 5, 6, 19, 20, 21 | keep, small fixes |
| 15–18 | Baseline models and results | team 9–12 | keep (small trims) |
| 19 | Research backing | team 13 | keep |
| 20 | From benchmark to deployment: the three LSTMs | team 14 + 23 + tech 9 | edit/merge |
| 21 | Spotting faulty sensors with the forecast | team 24 | edit caption |
| 22 | Fitting the models onto the NPU | tech 11 + 12 | new |
| 23 | Results on the real board | team 16 + tech 15 + 16 | replace |
| 24 | Live demo: the board serves its own dashboard | tech 19 + 20 | new → switch to demo |
| 25 | What we learned | tech 21 | new |
| 26 | Done since the proposal / what's next | team 17 + tech 22 | edit |
| 27 | Making the system grow with the campus | team 25 | edit |
| — | Backup: W8A8 on the NPU, verification ladder, conversion pipeline | tech 14, 17, 18 | backup (for Q&A) |

That's 27 slides including about 5 graph slides that go by quickly. If time is short, cut 9 (sensor features), merge 15–18 into two slides, and skip 25.

---

## Slide-by-slide

### 1 · Title (keep)
No change. Optional subtitle: *"Forecasting campus energy and spotting faulty sensors, on a Qualcomm edge board."*

### 2 · Progress since the proposal (edit team 2)
Tick the boxes that are now done and add the deployment phase. This is the first "positive" signal in the talk: the demo box was empty, and now it's full.

| Done | Phase | Item |
|---|---|---|
| ✓ | 1. Problem & research | Understand the domain · preliminary research |
| ☐ | | Final problem statement · follow-up research |
| ✓ | 2. Data | Clean and preprocess energy data · EDA and visualisation |
| ☐ | | Water-domain data (to be provided) · live ingestion once the API is available |
| ✓ | 3. Baseline models | Model per domain · train and evaluate |
| ✓ | | **Working demo** *(new: live dashboard on the QIDK)* |
| ✓ | **4. Edge deployment (new)** | **Quantized with Qualcomm AIMET · running on the QIDK's NPU · verified against the original models** |

**Say:** "Since the proposal we've closed the demo item and added a whole phase: the models now run on the Qualcomm board itself."

### 3 · The question: how close is campus to net-zero? (new, replaces team 15 "Equations used")
**On slide**
* **Net deficit = Demand − Solar.** Positive: we import from the grid. Negative: we have surplus to store or export.
* **Self-sufficiency = Solar ÷ Demand × 100 %.**
* Two time scales:
  * **1 hour ahead:** real-time decisions (battery ramping, cloud dips) and checking whether sensors are telling the truth.
  * **24 hours ahead:** planning (how much grid power to buy, when to charge batteries, pre-cooling).
* **Why on the edge:** the data stays on campus and nothing waits on a cloud round trip.

**Say:** "Everything that follows serves these two numbers, an hour ahead and a day ahead."

### 4 · Where the data comes from (move team 18 here, edit)
Retitle from "API: Cleaning" (the slide is about the source, not cleaning). Keep the 4-box diagram and add the numbers from the technical deck:
* **1,647 resources crawled** across **9 device families** (energy, solar, water, air quality and more).
* Today: a one-off crawl of the placeholder oneM2M server (schema matches the real one).
* **Next:** a live poller as soon as the main API is shared.

### 5–6 · Data analysis, data auditing (keep team 3, 4)
No change.

### 7 · Data quality: some nodes mostly repeat themselves (move team 22 here)
Moved up so all the data-cleaning story sits together. No text change.

### 8 · 11 sensor nodes feed 58 input features (replace team 8)
Team slide 8 currently repeats slide 7's text under a different title. Replace it with:
* **6 solar** (3 plants, 3 inverters) · **2 grid** (substation, incomer) · **3 room meters**
* **58 features:** 41 electrical readings + 11 "was this value filled in?" flags + 6 time-of-day / week / year signals
* Each model looks at the **last 24 hours** (96 readings, 15 minutes apart)
* Train / validate / test in time order: **70 / 15 / 15**; test period is **10 Nov – 31 Dec 2025**

### 9 · Major features of solar sensors (keep team 7)
Optional; cut if short on time.

### 10–14 · Graphs and insights (keep team 5, 6, 19, 20, 21, with fixes)
* **Slide 6 (smart room):** the chart titles say *Current*, but the values (232–246) look like mains **voltage** in volts. Check the column before presenting; if it is voltage, retitle to "Voltage – 1-day profile" and reword the caption (voltage sags when load rises).
* **Slides 19, 20:** the accent bar overlaps the title ("~~Solar~~ output", "~~October~~"). Move the bar or the title down.
* **Slide 21 (seasons):** remove the leftover "[Confirm against weather data.]". Replace with: *"We think monsoon cloud is the cause. Adding weather data is the next step, and it will tell us for sure."*

### 15–18 · Baseline models and results (keep team 9–12)
Keep as is; this is solid, honest benchmarking. Two small fixes:
* Slide 11 says "10 nodes" and slide 12 says "6 of 11 nodes". Use one count, or add "(of 11, SL-NI01-00 excluded from the averages)".
* Slide 12 is dense. If you trim it, keep these three points: LightGBM wins on average · "same time yesterday" is hard to beat for solar 24 h without weather data · building loads gain the most from models (KH04-01: 0.85 → 0.46 WAPE).

### 19 · Research backing (keep team 13)
No change.

### 20 · From benchmark to deployment: the three LSTMs (merge team 14 + 23 + tech 9)
This replaces team slide 14 ("LSTM Deployment") and the table half of team slide 23.

**On slide**

| Model | Forecasts | Size | Accuracy (R², 1 = perfect) |
|---|---|---|---|
| Campus, 1 h | Solar + grid, next hour | 140 k parameters | **0.85** (solar 0.94) |
| Campus, 24 h | Solar + grid, next day | 282 k parameters | **0.34** (solar 0.80) |
| One plant (Palash), 1 h | That plant's output | 55 k parameters | **0.95** |

* Each one reads the last 24 hours and forecasts the whole horizon in one go.
* The 1-hour model is the workhorse: it drives both dispatch and fault detection.
* The 24-hour model's weak point is grid demand, which depends on things it can't see yet (weather, campus events). That's exactly what the next iteration adds.

**Say:** "These are the models we took onto the board."

**Why the numbers changed from team slide 14:** slide 14's 24 h **Grid R² = 0.741** matches the persistence baseline's value in `model2/report.md`, and it doesn't fit the model's own error (74 kW MAE). On our rebuilt data it's about −0.1. Presenting 0.741 risks a hard question, so the table shows the average (0.34) with solar broken out. Solar numbers are consistent between the team report and our rebuild (within 0.012).

**Likely question: "LightGBM beat the LSTM in your benchmark. Why deploy LSTMs?"**
Suggested answer: "The neural networks are what run on the board's NPU through Qualcomm's toolchain. Tree models would run on the CPU. Our LSTMs are close behind LightGBM, and the benchmark itself lists the fixes: a week-ago input, more seeds, a bigger network. NPU-friendly networks like TCNs are on the roadmap too."

### 21 · Spotting faulty sensors with the forecast (edit caption of team 24)
Keep the formula and the table. Swap the caption for a forward-looking one:

> False alarms stay around 2–3 %. Dropouts and spikes are caught well. Gradual panel damage is hidden in campus totals, which is why per-plant detection is on our roadmap (slide 27). Demand isn't judged yet; its forecast needs the almanac features first.

Also update team slide 23's text (moved into this section, or dropped if slide 20 covers it). The current "This is a placeholder that we need to explore solutions for" becomes:

> Campus totals tell us *that* something is wrong. The single-plant model shows we can also tell *where*: it catches 92 % of dropouts at Palash. The next step is extending that to every plant without one model per node (slide 27).

**Say:** "We didn't train a separate detector. The forecaster already knows what to expect, so a reading that's far from the forecast for that hour gets flagged."

### 22 · Fitting the models onto the NPU (new, from tech 11 + 12)
**Title:** *Shrinking the models for the NPU, without retraining*

**On slide**
* **Why:** the board's NPU is fast at integer maths, so we convert the models from 32-bit floats to **8-bit weights** (files about 4× smaller: 153 / 297 / 71 KB).
* **Why we kept 16 bits inside:** an LSTM passes its memory through 96 steps, and small rounding errors pile up. 16-bit activations keep it stable.
* **The interesting fix:** at first we lost about 20 kW of accuracy. The cause: the tool sized its number ranges from sample data, which never contained some values the model really produces. We set those ranges from the maths instead (an LSTM's internal state is always between −1 and +1). Fidelity jumped from **44 to 78 dB**.
* **Tool:** Qualcomm AIMET, the same trained models, no retraining.

**Say:** "This was the main technical problem we solved. Once the ranges came from the maths instead of from samples, the 16-bit part was essentially lossless."

### 23 · Results on the real board (replace team 16 "QIDK Consideration", from tech 15 + 16)
Team slide 16's **0.286 ms** was measured with ONNX Runtime on a laptop CPU, not on the board. Replace it with the measured on-board numbers:

**Title:** *On the QIDK: under 4 ms, and practically identical to the originals*

| Model | Time on the NPU | Matches the original model | Accuracy before → after | Same fault verdicts |
|---|---|---|---|---|
| Campus 1 h | **3.4 ms** | **48 dB** | 0.852 → 0.853 | **99.7 %** |
| Campus 24 h | **3.4 ms** | **46 dB** | 0.339 → 0.335 | **99.8 %** |
| One plant | **3.9 ms** | **52 dB** | 0.950 → 0.950 | **99.9 %** |

* **What 48 dB means:** the NPU's answer is off from the original by at most **0.45 % of capacity**, about ten times smaller than the forecast error itself (≈4 % for 1 h solar).
* **What 3.4 ms means:** new data arrives every 15 minutes, so the board is more than **200,000× faster** than it needs to be.
* Qualcomm's AIMET beat the converter's own default by **2.6–5.9 dB** on every model.
* Measured on ~4,800 held-out test windows per model.

**Say:** "Accuracy didn't move, fault verdicts agree more than 99.6 % of the time, and it runs in under 4 milliseconds."

### 24 · Live demo: the board serves its own dashboard (new, from tech 19 + 20)
One screenshot, three bullets, then switch to the browser:
* A small C++ server **on the QIDK** serves the page and runs the models; no laptop-side compute.
* Replay the test period, inject a fault (dropout, spike, stuck sensor, panel damage) and watch it get flagged.
* Edit any input or upload a CSV and run it **live on the NPU**: about 5 ms of compute, 2 s end to end.

The 2 s is the graph being loaded on each run. Mention it only if asked, with "a persistent context removes it; it's on the list".

### 25 · What we learned (new, from tech 21 "Limitations and lessons")
Phrased as lessons, not limits:
* **Test on real hardware.** 8-bit everywhere looked fine in simulation (41 dB) but broke on the NPU (15 dB for the 24 h model). Only the on-board test caught it, which is why 16-bit activations ship.
* **Let physics set the ranges, not just sample data.**
* **Check what the tool really did.** The exporter had hidden the LSTM weights, so they weren't being shrunk to 8 bits until we fixed it.
* **Compare like with like.** We judge the board against the original model on identical inputs, so our results don't depend on the rebuilt dataset being perfect.

### 26 · Done since the proposal / what's next (edit team 17)
Team slide 17's first two items are now done. Showing them ticked is the strongest positive point in the deck.

**Done since the proposal**
* ✓ Quantized with Qualcomm AIMET: weights ~4× smaller (the slide promised 75 %)
* ✓ Running on the QIDK and measured: 3.4–3.9 ms on the NPU
* ✓ Working demo, served by the board

**Next**
* Weather and cloud-cover data for 24 h solar
* Campus almanac (semesters, exams, holidays) for demand
* NPU-friendly architectures (1-D CNN / TCN)
* Live ingestion from the node APIs; battery scheduling (MPC) on top of the forecasts
* Other domains (water) once the data arrives

### 27 · Making the system grow with the campus (edit team 25)
Keep the three existing paragraphs (they're good) and add a fourth. A tighter version:

* **Models that keep up.** Usage shifts and panels age. Short term: correct the forecasts on the board using recent errors (no retraining). Longer term: retrain regularly off the board, re-quantize with the same pipeline, and only swap in a new model if it beats the old one.
* **Weather and campus events** go into the next model iteration.
* **8-bit everywhere.** 16-bit activations work today. Training with quantization in mind (QAT) could make full 8-bit work too, for lower latency and power. Not tried yet.
* **Fault detection for every node, not one.** One model per node doesn't scale: 11 models to train, quantize, monitor and retrain. Instead: **one shared model per sensor type** (solar, grid, room) run across all nodes, so 3 models however big the campus gets. Plus comparing plants with each other, since panels under the same sky should agree. That's how we'd catch the panel damage that campus totals hide.

**Say:** "Everything here is written up with a plan in our repo, so the next iteration has a starting point."

---

## Backup slides (show only if asked)
* **W8A8 on the NPU** (tech 17): the chart with 48.4 / 46.4 / 52.1 vs 32.8 / 14.7 / 36.2 dB. Answers "why not 8-bit activations?".
* **Verification ladder** (tech 18): answers "how do you know it's right?".
* **Getting the models onto the NPU** (tech 14): the conversion pipeline and the seven failures. Answers "what went wrong along the way?".
* **AIMET gains per step** (tech 13): answers "what did AIMET actually do?".

---

## Checklist of fixes in the team deck
- [ ] Slide 2: tick "demo", add phase 4
- [ ] Slide 6: chart says *Current* but values look like volts. Check
- [ ] Slide 8: duplicate of slide 7. Replace
- [ ] Slide 11 vs 12: 10 vs 11 nodes. Make consistent
- [ ] Slide 14: 24 h Grid R² 0.741 is probably a copied baseline value. Replace
- [ ] Slide 16: 0.286 ms is laptop CPU. Replace with on-board 3.4 ms
- [ ] Slide 17: first two future steps are done. Move to "done"
- [ ] Slides 19, 20: accent bar overlaps title
- [ ] Slide 21: remove "[Confirm against weather data.]"
- [ ] Slide 23: reword "placeholder"
