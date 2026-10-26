# Viva Question Bank, Part 2: The Mid-Eval Deck

Questions aimed at **"Edge AI for Net-Zero" (26 slides)**, ordered by slide. Part 1 ([viva_questions.md](viva_questions.md)) covers the theory and the AIMET/QIDK pipeline in depth. This part checks whether the deck holds up under a hostile reading.

★ marks a trap. ⚠ marks something **on the slide itself** that is wrong, inconsistent or unverified. Fix it before presenting, or have the answer ready.

Two sources are behind the deck. Know which slide comes from which:
- **The per-node benchmark** (workspace [README.md](../README.md), `src/`): 6 methods, WAPE per node. Train is before 1 Sep 2025, validation is Sep–Oct, test is Nov–Dec. The LSTM there is 1 layer, 64 units, about 25 k parameters, with MAE loss and AdamW.
- **The deployed models** (`model2/`, `aimet_quantization/`): 3 multi-sensor LSTMs, R² on campus totals, a 70/15/15 split with test from 10 Nov. The LSTMs are 2-layer, 140 k / 282 k / 55 k parameters, with MSE loss and Adam.

---

## 0. The deck as a whole

**Q1. ★ Slides 10–13 describe a 1-layer, 64-unit, 25 k-parameter LSTM trained with absolute error. Slide 24 says 140 k parameters. Which LSTM is on the QIDK?**
Neither one is wrong; they are two different model families.
- The **slide 10 LSTM** is the per-node benchmark model: one model per meter, forecasting that meter's own power.
- The **deployed LSTMs** (slide 24) are campus-level, multi-sensor models. `all_sensors_1h` is 2 × LSTM(96) plus an MLP head, uses 58 features, has 139,749 parameters, and was trained with MSE and Adam.
⚠ The deck never says this, so an examiner will assume a contradiction. Add one sentence: "The benchmark compares methods per node; the deployed models are separate campus-level LSTMs."

**Q2. ★ Can the WAPE table (slide 11) and the R² figures (slides 15 and 24) be compared?**
No:
- They measure different targets: each node's own power in the benchmark, versus the Solar/Grid/Demand sums in the deployed models.
- They use different metrics (WAPE vs R²).
- They use different splits (test from 1 Nov vs from 10 Nov) and different preprocessing (the benchmark keeps long gaps as NaN; `model2` uses `fillna(0)`).
Say so before you're asked.

**Q3. ⚠ Slide 15 gives 24 h R²: Solar 0.8045 and Grid 0.7410. Slide 24 gives 0.339 for the same model. Which is right?**
Both come from the same model, but the numbers mean different things.
- **0.339** is mean(Solar R², Grid R²) on the rebuilt data: (0.798 + (−0.120)) / 2.
- **0.741 for 24 h Grid is almost certainly a copy error.** It equals the *persistence baseline's* 1 h Grid R² in `model2/report.md`. It contradicts the model's own MAE (74.3 kW in the metadata; the rebuild measures 73.1 kW). The real 24 h Grid R² is about −0.1.
⚠ Remove 0.741 from slide 15 or mark it as superseded. An examiner who spots both numbers will doubt every other figure.

**Q4. ⚠ Slide 18 (Future steps) says "Quantize to INT8 with AIMET" and "Run on QIDK". Slide 26 says W8A16 PTQ works. Which is current?**
Both are done: W8A16, not INT8, and the models run on the QIDK. Slide 18 is stale. It also proposes INT8, which was *measured to fail* on the 24 h model (14.7 dB, R² 0.34 → 0.17). Update slide 18, or say in the talk that it's the plan from before quantization.

**Q5. ⚠ Slide 2's checklist leaves "Formulate a clear problem statement" unchecked. What is your problem statement?**
Have one sentence ready. For example: "Forecast campus solar, grid and demand 1 h and 24 h ahead from the campus's own meters. Use the forecast errors to flag faulty sensors. Run it on a Qualcomm edge NPU without retraining or losing accuracy."
"Basic working demo" is also unchecked, but the QIDK dashboard exists. Tick it or explain.

**Q6. Where does the work end and the plan begin?** ★
Done:
- data cleaning and EDA;
- the 6-method benchmark;
- the 3 deployed LSTMs;
- residual anomaly detection;
- W8A16 PTQ on the QIDK HTP;
- the live dashboard.
Not done: the real API ingestion, water-domain data, weather inputs, QAT, MPC battery control on live data, and online adaptation.

---

## 1. Data analysis and auditing (slides 3–4)

**Q7. Slide 3, step 1 says "separated based on version", and step 2 says "separated further based on version". What do the versions mean, and why separate twice?**
Firmware versions change the *meaning* of columns: the same column name holds different quantities in different versions. Step 1 splits each node's rows by version. Step 2 maps each version's columns onto one schema and joins them in time. ⚠ The repeated wording reads like a typo; reword step 2 as "aligned matching columns across versions".

**Q8. "Two timestamp columns: resolved by picking the correct one." Which is correct, and how did you know?**
`Sensor_Timestamp` (when the meter measured) is correct; the index was **server-ingest time**. For EM-CR-SB00-00 the ingest time was off by a median of 105 days (back-filled uploads), so its index spanned only Aug–Nov. The evidence: `Sensor_Timestamp` covered Jan 23 – Dec 31 and matched the index on every other node.

**Q9. Why Parquet for computation and CSV for reading?**
Parquet is columnar, compressed and typed, which makes reads fast and keeps dtypes. CSV is human-readable but slow and loses types. For roughly 3 million raw rows, the difference is minutes versus seconds.

**Q10. Slide 4 says 4 nodes were dropped: SB00-01, SL-VN02-00, and SL-VN03-00 V1.0.1 and V2.0.1. Is that 4 nodes?** ⚠ ★
It's 3 nodes, with VN03-00 counted twice for its two firmware versions. Slide 23 shows **SL-PL96-00** with 41 % repeated frames, and the README lists it as excluded, but slide 4 doesn't mention it. Make the list match: SB00-01, SL-VN02-00, SL-VN03-00, SL-PL96-00.

**Q11. What does "frozen values" mean, and how is it detected?**
The meter keeps repeating its last frame because it's stuck or offline while the gateway still posts. It shows up as consecutive rows identical in all measurement columns. Slide 23 gives the rates: SB00-01 97 %, PL96-00 41 %, NI01-00 17 %.

**Q12. ★ SL-NI01-00 repeats 17 % of its frames and is excluded from the benchmark averages (slide 11). Is it in the deployed models?**
Yes. NI01-00 is one of the 6 solar nodes summed into the Solar target and is an input to both campus models. Be ready to defend the inconsistency:
- The benchmark excluded it from *averages* because its own-node WAPE is unreliable.
- The campus models need it, because it is part of total solar generation.
- Its data issues include a March–April unit change, and it reports Watts where the others report kW.

**Q13. "Fixed inverted meter wiring on SL-PL00-00 using sign correction." How did you know it was wiring and not real export?**
97.5 % of its readings were negative, including the midday peaks, and the magnitude and shape match a solar curve. A plant can't *consume* a solar-shaped curve, so the CT clamp or meter is mounted reversed.

**Q14. ★ Is "sign correction" what the code does?**
Not exactly. The benchmark flips the sign for PL00. `model2`'s `generate_lstm_dataset` applies `abs()` to power columns, which differs at the margins:
- `abs()` also turns genuinely negative readings (a small night-time inverter draw, noise around 0) into positive generation.
- A sign flip makes them negative.
The effect is small, but know which one each pipeline uses.

**Q15. "Unit standardization." Give a concrete example and how you detected it.**
- SL-NI01-00 reports Watts where the others report kW: its |min| of 130,800 matches the documented 130,781.6 W.
- NI01 also has a unit change in March–April, visible as March voltage at 1.8× the year's median.
Detection: compare each column's range with the device's nameplate and documented peak.

**Q16. Why 15-minute bins and not the raw cadence?**
Raw cadence ranged from 10 s to 5 min and differed per device, so a common grid is needed to put nodes side by side. 15 minutes matches utility metering intervals and smooths noise. It costs sub-15-minute transients such as cloud edges.

---

## 2. EDA slides (5, 6, 20–23)

**Q17. Slide 5: what is the "clear issue" in EM-CR-KH95 from October to December?**
A data gap: the meter stopped reporting from about early October to early December. ⚠ The straight line across the gap is a **plotting artefact** (matplotlib joins the points either side), not data. Say so, or plot with gaps.

**Q18. ★ Your test period is 10 Nov – 31 Dec. KH95 is part of the Grid target. What does the model see during that gap?**
Meters are forward-filled for at most 4 steps; after that the value becomes 0 (`fillna(0)`) with `was_imputed = 1`. So for much of the test period, the KH95 component of "Grid" is 0. The effect on Grid is negligible because KH95 peaks at 1.7 kW against 886 kW for SB00. But you should know that part of your test target is imputed.

**Q19. ★ Why is KH95, a building load meter, added to "Grid" at all?**
That was the original team's target definition: Grid = SB00 `gd_power_W` + KH95 `power_total_W`. Physically it's odd to add a single building's load to the substation import. If KH95 sits downstream of SB00, it is double-counted (by about 0.2 % of capacity). Either justify it or call it a known quirk.

**Q20. ⚠ ★ Slide 6 says "Current – 1-Day Profile", but the y-axis runs from 232 to 246. What unit is that?**
Almost certainly **volts**, i.e. mains voltage. Room current on KH04-01/02 is under 1 A (their draw is below 0.15 kW). The README notes that on SR-EM-KH04-01/02 the "current" column **actually holds voltage** (and "power factor" holds frequency). The two plots are "almost the same" because both rooms sit on the same supply bus.
The interpretation also flips: the dip around 01:00–02:00 and the rise toward 21:00 are *voltage* changes. Voltage typically **sags when network load rises**. The slide can't be read as "usage dropping after midnight". Retitle and re-caption.

**Q21. Why a "10-day median" profile rather than a mean?**
A median is robust to outlier days (outages, spikes), so the profile shows a typical day. Same reasoning as the MAD in the anomaly rule.

**Q22. Slides 7 and 8: what is the difference?** ⚠
None. Slide 8 repeats slide 7's text under a different title. Slide 7 also lists VN95-00 (a power-quality meter) under "solar sensors".
Also expect: "you list per-string DC V/I/P for 6 arrays, energy counters, phase angles. Which of these does the model use?" Only active power, AC voltage and PV voltage average for the inverters, and active power, reactive power, voltage average and neutral current for the plant meters. The rest were dropped because they are redundant (P ≈ V·I) or poor quality.

**Q23. Why drop columns that can be computed from others (P = V × I)?**
Redundant features add no information, increase the model's input size and parameters, and can make linear models unstable (collinearity). The model can learn the relationship when it needs it.

**Q24. Slide 20 (SL-NI03-00): why do cloudy days show up as lower *peaks* rather than shifted curves?**
Clouds reduce irradiance, and so output, but don't move the sun. Sunrise, sunset and solar noon stay fixed by the date; only the amplitude changes. That is why calendar features plus recent power go a long way, and why day-ahead solar needs weather data.

**Q25. ⚠ Slide 21 is titled "Campus demand", but it plots the substation's *grid power*. Are they the same?**
Only if no solar sits behind the substation meter. If solar is behind it, grid import = demand − solar, so the plot understates demand by up to the solar output (hundreds of kW at noon). The README states this assumption is **unconfirmed**. Also, "Demand" in the deployed models means the 3 room meters (about 30 kW), not campus load. Use one name per quantity.

**Q26. ⚠ Slide 21's "whole record" runs only Aug–Dec, and the busiest month is October. Why not the full year?**
Check this before presenting. SB00's *ingest-time* index spans only Aug–Nov, because of back-filled uploads. If this plot used that index, the date range, and possibly the "busy October" pattern, could be an artefact of upload timing rather than real load. Re-plot with `Sensor_Timestamp`.

**Q27. Slide 22 says spring peaks are highest and summer lowest. Which node is it, and how are Indian seasons defined here?**
Know the node: the slide doesn't say, and about 53 kW at midday suggests one plant, not the campus. Know the month-to-season mapping you used. The monsoon (Jun–Sep) explains the low summer: cloud cover, not the sun's position, because day length is longest in June.
⚠ "[Confirm against weather data.]" is placeholder text left on the slide. Remove it or confirm.

**Q28. ★ Slide 22 says "A model needs season-aware inputs". Don't the models already have them?**
Partly. The day-of-year sin/cos features encode the season. But with about 8–10 months of training data, the model sees each season roughly once, so it can't separate "season" from "that particular month's weather". Weather and almanac data (clear-sky irradiance) would give a physical reason instead of a calendar proxy.

**Q29. Slide 23 mentions sentinel values like −0.999. What are they, and why are they dangerous?**
They are placeholders a device sends when it has no valid reading. Left in, they look like real (small negative) power. They distort scalers (min/max) and teach the model fake patterns. They are dropped before resampling.

**Q30. "Inverters send nothing at night, filled as 0 W." But the deployed pipeline forward-fills the last AC reading through the night. Which is it?** ⚠ ★
The two pipelines differ:
- The **benchmark** fills 20:00–05:00 IST with 0 W.
- **`model2`** (reconstructed during the rebuild) carries the last AC reading through the night with `was_imputed = 1`, but not the DC string voltage.
So the deployed model sees a frozen evening value with an "imputed" flag rather than zero. Filling 0 is physically correct. Forward-filling relies on the flag to tell the model to ignore the value.

---

## 3. The benchmark (slides 9–13)

**Q31. Define WAPE. Why WAPE and not MAPE?**
WAPE = Σ|y − ŷ| / Σ|y|, the total absolute error as a fraction of total energy. MAPE divides by each actual value, which is undefined or explodes when y = 0, and solar is 0 every night. WAPE stays defined and weights errors by energy.

**Q32. ★ In slide 11, "Yesterday" is 0.353 at all three horizons and "Last week" is about 0.416 at all three. Why don't they change with horizon?**
Their forecasts don't depend on the horizon. y(T − 24 h) is known as long as the horizon is ≤ 24 h, so the prediction for time T is the same whether you make it 15 minutes or 24 hours ahead. Only the forecast's *issue time* changes, not its value. The tiny difference at 24 h (0.416 vs 0.417) comes from slightly different evaluation rows at the edges.

**Q33. ★ Persistence at 24 h is 0.353, the same as "Yesterday". Coincidence?**
No. At a 24 h horizon, "the value now" *is* "the value 24 h before the target", so the two predictors are identical by definition. The README says so for solar. The 1 h persistence score (0.354) being close to 0.353 is a coincidence.

**Q34. At 15 minutes the LSTM scores 0.161 and persistence 0.163. Did the LSTM learn anything?**
Barely, on average. At 15 minutes, power changes slowly, so "same as now" is near-optimal for the large, smooth meters. Slide 13 says persistence is as good or better on the substation, KH95 and KH04-00. The LSTM's gains show at 1 h (0.239 vs 0.354) and on solar and small room meters. Don't claim a 15-minute win.

**Q35. ★ LightGBM wins at every horizon. Why deploy LSTMs?**
Have an honest answer:
1. The course goal is NPU deployment. Neural networks map to the Hexagon through AIMET/QNN; gradient-boosted trees don't, and would run on the CPU.
2. The gap is modest: 0.297 vs 0.303 at 24 h, and within single-seed noise per slide 13.
3. Slide 13 lists the fixes: a week-ago input, more seeds, a larger network.
Don't claim the LSTM is the better forecaster. Also be ready for: "a tree model on the CPU would also meet a 15-minute cadence". That's true. The NPU case rests on energy and on the course requirement, not on necessity.

**Q36. ★ Slide 14 cites papers saying LSTMs beat classical methods. Slide 11 shows them losing to LightGBM. Isn't the research backing contradicted by your own results?**
Yes, and say so. The cited papers compare LSTMs with ARIMA-type statistical models, which you didn't benchmark. Gradient-boosted trees with hand-built lag features are a much stronger baseline. On short, tabular-like energy data, tree models often match or beat deep sequence models. Your result is consistent with that literature, not with the slide's framing. Reword slide 14 to "why sequence models are worth testing", not "why LSTMs win".

**Q37. Slide 14 says CNN-LSTM and TCN track best. Did you try them?**
No. They are on slide 18 as future work. The argument for TCNs on Hexagon: convolutions run in parallel across time steps, whereas an LSTM's 96 steps must run in sequence.

**Q38. What do the benchmark's Ridge and LightGBM see that its LSTM doesn't?**
Explicit lags: the value 2 days ago, a week ago, and yesterday's daily mean. The LSTM sees only a 24 h window, so it can't use the weekly cycle. Slide 13 calls this the confirmed setup difference. The deployed LSTMs have the same 24 h limit.

**Q39. Ridge: "inputs standardised, missing values filled with the median". Median of what?** ★
It must be the **training** median, with the standardisation fitted on train only. Otherwise test information leaks in. Know which one your code uses (`src/02_forecast.py`).

**Q40. Why train LightGBM with absolute-error loss?**
The score is WAPE, an absolute-error metric. Minimising absolute error gives the conditional *median*, which is robust to spikes. MSE would aim for the mean and chase outliers. The deployed LSTMs use MSE, a mismatch worth knowing about.

**Q41. "Early stopping when validation error stopped improving". Isn't the test set touched?**
No. Early stopping and the choice of best naive baseline use validation (Sep–Oct) only. Test (Nov–Dec) is scored once.

**Q42. "Beat the best naive forecast by at least 10 %": LSTM meets it at 4/6/2 of 11 nodes. Is that a pass?**
Not for most nodes. Present it as a known shortfall. LightGBM does better (6/8/3) but also misses at most nodes 24 h ahead, because solar needs weather data.

**Q43. ★ "SR-EM-KH04-01 drops from 0.85 to 0.46 WAPE": the biggest gain in the deck. Should we be impressed?**
Be careful. KH04-01 draws under 0.15 kW. WAPE on a near-zero series is dominated by small absolute errors, so large relative gains can be tiny in Watts. The README calls these weak evidence.

**Q44. Slide 13 says the test months (Nov–Dec) differ from training months, "which trees tolerate better than neural networks". Why would they?**
Trees predict piecewise constants within the training range and never extrapolate beyond it. Neural networks can extrapolate wildly on unfamiliar inputs. Under distribution shift, trees fail more gracefully. This is a plausible claim, not one you proved.

**Q45. One seed per model: why does that matter?**
Neural-network training is stochastic (initialisation, batch order). Run-to-run variation can be comparable to the LSTM–LightGBM gap. Several seeds with mean ± std are needed before claiming either one is better.

---

## 4. The deployed models and their metrics (slides 15, 16, 24)

**Q46. Slide 15 quotes Solar R² 0.9554 and Grid R² 0.7940 for the 1 h model. Do you reproduce them?**
On the rebuilt data: Solar 0.943 and Grid 0.762, within 0.012 and 0.032. The original training data wasn't available, so it was reconstructed from raw. The differences come from the reconstruction (solar capacity is 1.9 % lower). They don't affect any quantization claim, which compares quantized vs FP32 on identical inputs.

**Q47. Slide 15: what is nMAE, and why is 24 h Grid nMAE 3.94 % suspicious?**
nMAE = MAE / capacity. The model's own metadata says 74.3 kW, which is **8.38 %**. 3.94 % pairs with the persistence baseline, the same row that 0.741 was copied from. ⚠ Both numbers on that line are suspect.

**Q48. ★ Slide 16: "Net Deficit = Campus Demand − Solar". Which Demand?**
The deployed models' "Demand" is the sum of 3 room meters, about 30 kW at most. Campus load is roughly 300–600 kW (slide 21). Plug the model's Demand into this formula and you get a large *surplus* at noon, which is nonsense. Either define Campus Demand ≈ Grid import + Solar (if solar is behind the substation meter, which is unconfirmed), or state that the formula needs a campus-load measurement you don't have.

**Q49. ★ Self-sufficiency = Solar / Demand × 100 %. Any problems?**
- It is undefined when Demand = 0, and it exceeds 100 % whenever there's surplus. Cap it or define it over energy (kWh over a day), not instantaneous power.
- With room-meter "Demand" it would read several hundred percent.
- The original runner printed `net = grid + solar − demand`, which contradicts this slide.

**Q50. Slide 24: "If sensor readings don't match the forecast, that indicates an anomaly." Is every mismatch a sensor fault?** ★
No. A residual can come from:
- a faulty sensor;
- a real physical event (cloud, a plant tripping, unusual load);
- the forecaster simply being wrong.
The rule can't tell these apart. It flags "unexpected", and a person or a cross-check against other sensors must diagnose the cause.

**Q51. Slide 24 calls the single-node forecaster a "placeholder" for localising anomalies. Why only Palash?**
Only one plant-level model was trained. Localising a fault to *any* plant needs a model, or at least a residual check, per node. That is the scalable path: per-node models, or one model with per-node outputs.

**Q52. Slide 24 says the 24 h model's FP32 R² is 0.339. Why deploy something that weak?**
Solar 24 h on its own is 0.798. The mean is dragged down by Grid at about −0.1: day-ahead grid draw isn't predictable from 24 h of history without schedule or weather data. It is deployed to demonstrate the day-ahead pipeline on the NPU, not as a finished planning tool. Say that explicitly.

---

## 5. QIDK slide (17)

**Q53. ⚠ "Bitwise numerical parity, Δ < 2.38 × 10⁻⁷". Correct?**
No. Bitwise means Δ = 0. 2.38 × 10⁻⁷ ≈ 2⁻²², the float32 rounding level near 1. Say "matches to float32 precision". True bit-exactness *was* achieved later, for the converted FP32 graph on the QIDK CPU against ONNX.

**Q54. ⚠ "0.286 ms (≈3,495 FPS) on ONNX Runtime CPU": on which CPU? Is that an edge number?**
It was measured on a host CPU, not on the QIDK, and FP32, not quantized. "FPS" is the wrong unit: there are no frames, so use inferences/s. On the QIDK NPU, W8A16 execution takes 3.4 ms. Expect "why is the NPU 12× slower than your laptop?" The answer: per-call RPC and data-transfer overhead dominates for a model this small, and in any case both are irrelevant against a 15-minute cadence.

**Q55. "B = 1 matching QNN/SNPE constraints". Is batch 1 a constraint or a choice?**
Both. The converter needs *static* shapes, and the HTP graph is compiled for fixed tensor sizes. A batch of 8 is possible but would also be fixed. Batch 1 fits the use case: one window every 15 minutes.

**Q56. "Effortlessly satisfying the 15-minute control loop". Is latency the right metric for this system?** ★
Hardly. Any device meets 15 minutes. The relevant edge metrics are **energy per inference**, memory, and running without the cloud. Energy wasn't measured, so don't imply it was.

---

## 6. Future work and "more things to think about" (slides 18, 19, 26)

**Q57. Slide 18: "INT8 shrinks size by 75 %". Do you get 75 %?**
In weight terms, about 3.8× (≈ 74 %), because biases stay 32/16-bit. The compiled model `.so` is *larger* than the FP32 ONNX file (3.8–4.0 MB), because it contains code for 96 unrolled steps. And full INT8 (W8A8) wasn't used: it breaks the 24 h model.

**Q58. Slide 18: "TCNs map onto Qualcomm vector extensions without recurrent bottlenecks". Explain.**
An LSTM step needs the previous step's output, so the 96 steps run strictly in sequence. That gives a long dependency chain and the 3,853-tensor unrolled graph. A TCN's dilated convolutions compute all time steps at once, with no state carried across steps, so they use HVX/HMX parallelism better. Quantization error also doesn't compound through a recurrence.

**Q59. Slide 18: "connect forecasts to battery MPC". Did you simulate it? What happened?**
Yes, in the benchmark study, for a 400 kWh / 100 kW battery over 49 days:
- MPC with the practical forecast: −0.2 % peak import.
- MPC with a perfect forecast: −13.7 %.
- A simple threshold rule: −1.4 %, better than the MPC.
Conclusion: forecast quality, especially solar without weather data, is the bottleneck, not the controller. The tariff and carbon numbers were assumed. Don't present this as a win.

**Q60. Slide 19: what is oneM2M?**
An IoT interoperability standard. Data sits in a resource tree (CSE → AE → container → contentInstance) and is read over HTTP or MQTT. The crawler walks that tree with 8 parallel workers and retries.

**Q61. ★ Slide 19 says the real API "has not been given to us". So how does the QIDK get live data?**
It doesn't yet. The dashboard replays recorded test data and accepts CSV uploads. Live ingestion is blocked on the API. Be precise: the inference is live on the NPU, the data is not.

**Q62. Slide 26: "online bias correction using exponentially smoothed residuals". How would that work, and what's the risk?**
Keep b_t = α·r_t + (1 − α)·b_{t−1} and add b_t to the next forecast. It corrects slow drift, such as panel soiling or a load change. **Risk:** it also absorbs real faults. A slowly dying sensor gets "corrected away", which defeats anomaly detection. You'd need to freeze the correction while a flag is raised, or cap it.

**Q63. Slide 26: "refit the last layer periodically". Where?**
Not on the HTP: QNN is inference-only, with no backpropagation. You'd refit on the CPU (the head is a small least-squares problem), re-quantize the head, and redeploy. Fine-tuning in quantized form also needs care to stay within the existing encodings.

**Q64. ⚠ Slide 26: "QAT might make W8A8 usable, which would slash latency and power". Would it?**
The measured W8A8 latency saving is about 0.7–2 ms out of 3.4–3.9 ms. That's real, but "slash" overstates it, and power wasn't measured. QAT might fix the 24 h model's W8A8 collapse, but this LSTM's error compounds through 8-bit internals over 96 steps, which QAT can only partly train around. Say "might help; untested".

**Q65. Slide 26: "weather and campus events affect readings". What specifically would you add, and from where?**
- **Weather:** cloud cover / irradiance forecasts (for day-ahead solar) and temperature (for HVAC load), from a weather API or IMD.
- **Clear-sky irradiance:** computed from solar geometry, no API needed.
- **Calendar:** holidays and exam or vacation periods (for load).
The benchmark tried sun position and neighbouring-node cloud signals (`src/08_pv_clearsky.py`), and they didn't beat "yesterday" on their own. Forecast weather is what's missing.

---

## 7. Rapid-fire on the deck

**Q66.** How many sensor nodes do the deployed models use? → 11: 6 solar, 2 grid/substation, 3 room meters. The benchmark scores 10 of them in its averages (NI01-00 excluded).

**Q67.** What is "Spike (+50 %)" on slide 25 relative to? → 50 % of **capacity**, not of the current reading.

**Q68.** ⚠ Slide 25's footer says "Edge AI for Net Zero · 10". → It was copied from another deck. Fix the footer.

**Q69.** Train and test dates? → Benchmark: train before 1 Sep, val Sep–Oct, test Nov–Dec. Deployed: 70/15/15 of 23 Jan – 31 Dec, test from 10 Nov. Both test on winter only.

**Q70.** Why IST for the cyclic features? → Solar and occupancy follow local time. The raw index is UTC.

**Q71.** Slide 13 says about 8 months of training data. True for the deployed models? → Train is the first 70 % of 23 Jan – 31 Dec, which ends in mid-September: about 7.7 months. Similar.

**Q72.** ⚠ Slide 20's title has a strike-through on "Solar", and the accent bar overlaps the title on slides 20–21. → Cosmetic, but examiners notice. Fix it.

---

## 8. If you only fix five things on the deck

1. **Slide 15:** remove or flag 24 h Grid R² 0.741 and nMAE 3.94 %. Reconcile with slide 24's 0.339.
2. **Slide 6:** "Current" is voltage. Retitle, and redo the interpretation.
3. **Slides 10/24:** add one line separating the per-node benchmark LSTM from the deployed campus LSTMs.
4. **Slides 16/21:** define "Demand" once. Room-meter Demand ≠ campus load ≠ grid import.
5. **Slides 17/18:** "bitwise" → "float32 precision". Label the 0.286 ms as host CPU and add the QIDK NPU figure (3.4 ms). Update the stale INT8 / "run on QIDK" future steps.
