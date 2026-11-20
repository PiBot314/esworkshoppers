# Recording flow: 10-minute mid-eval video

Deck: `Edge AI for Net-Zero (revised).pptx` (27 slides), used as is. Slide numbers below refer to that deck.
Target: **≈9:40 of content + 20 s slack**, of which **≈3:10 is the dashboard**.
Speaking pace assumed: ~150 words per minute, so a 30 s segment is about 75 words. The "say" lines are written to fit their time.

---

## At a glance

| Time | Segment | Slides | Length |
|---|---|---|---|
| 0:00 | Why we're doing this | 1, 2, 3 | 1:00 |
| 1:00 | The data | 4, 6, 7, 8, 12, 14 | 1:30 |
| 2:30 | Benchmarks | 17, 18 | 0:50 |
| 3:20 | What runs on the board | 20, 21, 22, 23 | 1:50 |
| 5:10 | **Dashboard demo** | 24 → browser | **3:10** |
| 8:20 | What we learned and what's next | 25, 26, 27 | 1:20 |
| 9:40 | End | | |

**Slides not shown: 5, 9, 10, 11, 13, 15, 16, 19.** Since this is a recording, step past them while recording and trim them out in the edit, or record the slide segments in pieces. The deck itself stays unchanged. If an evaluator asks later, the full deck covers them.

---

## Segment 1 · Why we're doing this (0:00–1:00)

**Slide 1 (5 s).** "We're esworkingshoppers, and this is Edge AI for Net-Zero."

**Slide 2 (20 s).** "Since the proposal we've finished data preparation, the baseline models and a working demo. We've also added a whole new phase: the models now run on a Qualcomm edge board."

**Slide 3 (35 s).** "The question is simple: how close is campus to generating what it uses? Net deficit is demand minus solar. We forecast it at two time scales: an hour ahead for real-time decisions and for spotting faulty sensors, and a day ahead for planning. We run it on the edge, so the data stays on campus."

## Segment 2 · The data (1:00–2:30)

**Slide 4 (20 s).** "Data comes from the campus oneM2M API: 1,647 resources across 9 device families. We crawl the placeholder server today, and a live poller comes next."

**Slide 6 (20 s).** "The raw data needed real work: dead nodes with frozen values, a meter wired backwards that logged 97 % negative solar, and mixed units."

**Slide 7 (15 s).** "One substation meter repeats the previous reading 97 % of the time, so we dropped it."

**Slide 8 (20 s).** "What's left: 11 nodes (6 solar, 2 grid, 3 room meters) giving 58 features. Each model reads the last 24 hours."

**Slide 12 (10 s).** "Solar has a clean daily cycle, and cloudy days show up as lower peaks…"

**Slide 14 (15 s).** "…and the seasons differ: spring middays average 53 kW, summer only 37. Weather explains a lot of this, which is why weather data is on our list."

## Segment 3 · Benchmarks (2:30–3:20)

**Slide 17 (30 s).** "We benchmarked six approaches, from 'same as now' to LightGBM and an LSTM. Lower is better here. LightGBM is best on average, and the LSTM is close behind: 0.161 vs 0.135 at 15 minutes, 0.303 vs 0.297 at a day ahead."

**Slide 18 (20 s).** "One honest finding: for solar a day ahead, 'same time yesterday' is hard to beat without weather data. Building loads gain the most from models; one room's error almost halves."

## Segment 4 · What runs on the board (3:20–5:10)

**Slide 20 (25 s).** "These three LSTMs are what we put on the Qualcomm board: campus 1 hour, campus 24 hours, and one solar plant. The 1-hour model is the workhorse. The 24-hour model's weak spot is grid demand, which needs weather and campus events as inputs."

**Slide 21 (30 s).** "We didn't train a separate fault detector. The forecast already knows what to expect, so a reading far outside the usual range for that hour gets flagged. Dropouts and spikes are caught well, with 2–3 % false alarms. Gradual panel damage hides in campus totals; that's a point we come back to at the end."

**Slide 22 (25 s).** "To run on the NPU we shrank the models to 8-bit weights, with no retraining. The main fix: the tool guessed number ranges from sample data and clipped real values. Setting the ranges from the maths took fidelity from 44 to 78 dB."

**Slide 23 (30 s).** "The result: under 4 milliseconds per forecast, when data arrives every 15 minutes. Accuracy is unchanged, 0.852 before and 0.853 after, and fault verdicts agree with the original more than 99.6 % of the time. Let's see it running."

## Segment 5 · Dashboard demo (5:10–8:20)

**Slide 24 (10 s)**, then cut to the browser. "The board serves this dashboard itself, and the models run on its NPU."

| Time | Do | Say |
|---|---|---|
| 0:00 | Tab **1** (Campus replay). Hover the header badge, then the KPI tiles | "This page comes from the QIDK. The badge shows the board, and these tiles are the measurements from the board." |
| 0:20 | Point along the Solar chart | "White is the actual reading, dashed is what the NPU forecast 15 minutes earlier, and green is what's normal for this hour." |
| 0:35 | Set **Speed** to 16× (it starts at 4×), **Space** to play for about 5 s, then **Space** | "Days go by, and readings stay inside the band." |
| 0:50 | **S** (spike) | "Now a sensor spikes by half the plant's capacity. The reading leaves the band, and it's flagged as an anomaly. Grid stays OK." |
| 1:10 | **C**, then **D** (dropout) | "A dead sensor reading zero: flagged too." |
| 1:25 | **C**, then **P** (panel damage −40 %) | "Here's the limit we mentioned. Losing 40 % of one plant is hidden in the campus total, so it isn't flagged. That's why per-plant detection is on our roadmap." |
| 1:50 | **C**, then **R** (opens this window in *Run on NPU*), click **▶ Run on NPU** | "Now we send this exact 24-hour window to the board and run it on the NPU." |
| 2:05 | Point at *Last live run* | "About 5 milliseconds on the NPU. The rest of the two seconds is loading the model for each run, which we'll remove next. And the output matches the recorded run to a millionth." |
| 2:30 | Card 2: `SL-PL00-00__active_power_W` → *set to 0* → last 16 steps → **Apply edit** → **Run on NPU** | "We can also change the input. Here the Palash plant's sensor dies for the last 4 hours, and the campus solar forecast for the next 15 minutes drops from 416 to 373 kW." |
| 2:55 | Tab **5** (Under the hood), scroll to the per-window chart | "Over the whole test period, the board never strays more than 0.45 % of capacity from the original model." |
| 3:00 | End of the demo clip | |

## Segment 6 · What we learned and what's next (8:20–9:40)

**Slide 25 (20 s).** "Four lessons. Test on real hardware: full 8-bit looked fine in simulation and broke on the board. Let the maths set the ranges. Check what the tools really did. And compare against the original model on identical inputs."

**Slide 26 (25 s).** "Since the proposal: models quantized, running on the board in under 4 ms, and a working demo. Next: weather and campus-calendar data, NPU-friendlier architectures, live data from the APIs, and battery scheduling on top of the forecasts."

**Slide 27 (35 s).** "And making it grow with the campus. Usage shifts and panels age, so the models need to keep up: correct them on the board first, then retrain on a schedule. Training with 8-bit in mind could make the models faster still. And fault detection should cover every node, using one shared model per sensor type rather than one per node, plus comparing plants with each other. That's how we'd catch the panel damage campus totals hide. Thank you."

---

## Recording the demo

**Setup**
- Turn off laptop suspend and the screen lock. The laptop suspended mid-run before, and adb dropped.
- `adb devices` should list `d42f06b`. Then run `cd aimet_quantization && scripts/14_dashboard_start.sh`.
- Do one warm-up **Run on NPU** before recording.
- Record at 1920×1080 in full-screen browser, with bookmarks hidden and the zoom at 100 %. **Press F11 right after the page loads, before touching any tab.** Resizing the window after visiting other tabs can trip a harmless script error that may leave a chart at the old width; a page reload fixes it.
- Turn on cursor highlighting in your recorder (e.g. OBS) so viewers can follow the pointing.
- Record the demo as its own clip. If a step goes wrong, redo just that step, then join the clips.

**Checked on the board (6 Oct 2026):** every step above works. Expected values: spike → Solar 761 kW vs 416 kW forecast, ANOMALY; dropout → 0 W, ANOMALY; panel damage → 293 kW, residual −124 kW inside ±275 kW, ok; live run ≈5 ms on the NPU, ≈2.1 s end to end, match to the recording 4×10⁻⁷.

**Editing**
- Keep the ~2 s live-run wait in. It's honest, and the narration covers it.
- Speed up the 5 s playback if needed, but don't cut it out; it shows the replay is continuous.
- Show slide 24 for a moment before the browser, so the cut doesn't feel abrupt.
- If the board misbehaves on the day, the fallback screenshots are in `aimet_quantization/dashboard/screenshots/` (`inject-spike.png`, `live-run.png`, `hood.png`). Live footage is much stronger, though.

**Voices.** If several of you want to speak, a natural split: segments 1–3 (the team's data and benchmark work), segment 4 + demo (deployment), segment 6 (wrap-up). Record each person's audio separately and lay it over the slides.

**If you're over time**, trim in this order:
1. Slide 12 (10 s)
2. Slide 7 (15 s)
3. The demo's channel-edit step (25 s)
4. Slide 18 (20 s)
