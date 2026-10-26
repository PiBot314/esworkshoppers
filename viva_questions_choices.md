# Viva Question Bank, Part 3: "Why this number and not another?"

Every design constant in the project will draw the same question from an examiner: why 15 minutes, why K = 4, why 1,024 windows? This part lists each number with an honest answer.

Parts 1 and 2: [viva_questions.md](viva_questions.md) (theory and pipeline), [viva_questions_deck.md](viva_questions_deck.md) (the slides).

---

## How to answer any "why this number?" question

Use four moves, in this order:

1. **What it trades off.** Every constant sits between two failure modes. Name both.
2. **Where it actually came from**, honestly. Each number in this file is tagged:
   - **[measured]**: we compared alternatives and picked the best.
   - **[reasoned]**: derived from physics, maths or the use case, but not tested against alternatives.
   - **[inherited]**: set by the original training code, kept because the models were frozen.
   - **[arbitrary]**: a sensible default nobody tuned.
3. **What happens if you move it** up or down.
4. **What experiment would settle it**, if it hasn't been settled.

"It was inherited, and here's what I'd expect if we changed it" is a strong answer. A made-up justification is a weak one, because examiners probe it.

---

## 1. Data and time resolution

**Q1. Why 15-minute resolution?** [reasoned / inherited]
- **What forced a common grid:** raw cadence was irregular, from 10 s to 5 min, and differed per device. A shared grid was needed to put 11 nodes side by side. The coarsest regular sensors report every few minutes, so 15 minutes is the finest grid where nearly every bin contains real readings.
- **It matches how power is scheduled and billed.** Indian grid scheduling and settlement work in 15-minute time blocks (96 per day). A forecast at that resolution plugs straight into those decisions.
- **Finer (5 min):** a 24 h window becomes 288 steps, so the LSTM's sequential compute and its unrolled graph grow 3×, and gradients must travel 3× further. Inputs get noisier, with more empty bins and more imputation.
- **Coarser (1 h):** you lose intra-hour ramps (cloud transients, load switching), get only 24 points per day, and anomaly alarms can't fire sooner than an hour.
- **Cost of the choice:** events shorter than 15 minutes are averaged away. An anomaly can't be detected sooner than one bin.
- **Not done:** no comparison of 5/15/60 minutes was run.

**Q2. Why a 96-step (24 h) lookback?** [inherited, reasoned]
It covers exactly one daily cycle, so the window always contains "the same time yesterday", the strongest single predictor for solar and load.
- **Shorter (12 h):** loses yesterday's same hour.
- **Longer (7 days = 672 steps):** would capture the weekly cycle. The benchmark showed loads are strongly weekly, and LightGBM's week-ago lag is one reason it beat the LSTM. But it costs 7× the compute and a 7× longer recurrence.
- **Alternative:** feed the week-ago value as an *extra feature* instead of lengthening the window.

**Q3. Why horizons of 4 steps (1 h) and 96 steps (24 h)?** [reasoned]
They match two decision timescales:
- **1 h:** dispatch, battery ramping, and the anomaly check.
- **24 h:** day-ahead procurement and scheduling.

Nothing in between was built.

**Q4. Why a 70/15/15 split?** [inherited]
It is a convention. It gives about 7 weeks of test data (Nov 10 – Dec 31), and the test period sits entirely in one season, which is a known weakness. Better: rolling-origin evaluation across several months.

**Q5. Why forward-fill gaps of at most 4 steps (1 hour)?** [reasoned / arbitrary]
- A gap of up to an hour is plausibly smooth, so carrying the last value is a reasonable guess.
- Beyond that, you are inventing data. Longer gaps become 0 with `was_imputed = 1`.
- **Higher limit:** more fabricated stretches look real.
- **Lower limit:** more zeros, more holes.

**Q6. Why MinMax to [0, 1] and not z-score standardisation?** [inherited]
- **For it:** bounded inputs suit the LSTM's sigmoid and tanh gates. Outputs as fractions of capacity are interpretable.
- **Against it:** MinMax is fragile to outliers, because a single spike sets the max and squashes everything else. Example: SL-NI01-00's voltage has a 357 V outlier max, so normal voltages occupy a narrow band.
- **Alternatives:** z-scores, or MinMax with clipped percentiles (1st/99th).

**Q7. Why 365.25 in the day-of-year encoding?**
Leap years. The cycle repeats every 365.25 days on average, so the encoding doesn't jump at year boundaries.

---

## 2. Model hyperparameters

**Q8. Why 2 layers and hidden sizes 96 / 128 / 64?** [inherited]
No architecture search was done. What can be said:
- The hidden sizes scale with each task: largest for 288 outputs (24 h), smallest for a single plant.
- About 140 k parameters against about 23 k training windows sounds risky. Worse, the windows overlap by 95/96, so the *effective* number of independent samples is much smaller.
- Overfitting is controlled by best-validation checkpointing.
- A sweep (1–3 layers, hidden 32–256) with several seeds would settle it.

**Q9. Why dropout 0.1, lr 0.002, batch 64, 10–15 epochs, ReduceLROnPlateau (×0.5, patience 2)?** [inherited]
Common defaults.
- **Few epochs:** the best-validation checkpoint means training stops being useful early. Validation loss stops improving within about 10 epochs.
- **Dropout 0.1:** light regularisation between the LSTM layers.

Say "defaults, not tuned" and move on. Don't invent reasons.

**Q10. Why MSE loss?** [inherited]
- MSE targets the conditional mean and punishes big errors hardest.
- MAE targets the median and is robust to spikes. The benchmark used MAE for LightGBM because its score is WAPE.
- The deployed models are scored by R², which is MSE-based, so MSE is consistent with their metric.

---

## 3. Anomaly detection

**Q11. Why 4 robust standard deviations? Why not 3 or 5?** [measured, but only partly]

**Theory first.** If residuals were Gaussian, the share of normal readings falsely flagged at each K would be:

| K | P(\|Z\| > K) |
|---|---|
| 2 | 4.6 % |
| 3 | 0.27 % |
| 4 | 0.006 % |
| 5 | 0.00006 % |

**Reality.** We see about 2 % false flags even at K = 6. The residuals are heavy-tailed (cloud transients), and the test months (Nov–Dec) differ from the validation months the thresholds were fitted on. Gaussian theory doesn't apply, so K has to be chosen empirically.

**How K = 4 was originally chosen:** it was lowered from 5 because dropout detection rose from 32 % to 69 %. Only those two values were compared.

**Full sweep, run for this question:** FP32 outputs, test period, the same 300 injected faults (100 per type). Solar false-flag rate, with the 2 % floor:

| K | 1 h false flags | 1 h dropout / spike / damage | single-node false flags | single-node dropout / spike / damage |
|---|---|---|---|---|
| 2 | 4.7 % (4.5/day) | 100 / 99 / 30 | 5.3 % | 100 / 99 / 81 |
| **3** | **2.6 % (2.5/day)** | **89 / 98 / 0** | 3.3 % | 99 / 93 / 31 |
| 4 (used) | 2.3 % (2.2/day) | 69 / 91 / 0 | 2.7 % | 92 / 83 / 17 |
| 5 | 1.9 % (1.8/day) | 37 / 52 / 0 | 2.5 % | 79 / 56 / 9 |
| 6 | 1.8 % (1.8/day) | 10 / 29 / 0 | 2.4 % | 59 / 49 / 3 |

**What the sweep shows:**
- **K = 3 looks better than K = 4 for the 1 h model.** Dropout detection rises from 69 % to 89 % and spike detection from 91 % to 98 %, for only +0.3 false flags per day. Be honest about this if asked.
- **Raising K buys very little** on false alarms (2.3 % → 1.8 %) but costs most of the detection. That is the signature of heavy-tailed residuals: the false flags are genuinely large errors, not marginal ones.
- **Only K = 2 catches panel damage at campus level** (30 %), at about 4.5 false alarms per day.

**Caveat:** this sweep is on the *test* period. Choosing K from it is tuning on test. The proper procedure:
1. inject faults into *validation* data;
2. fix an alarm budget (e.g. ≤ 1 false alarm per day per domain);
3. pick the K that maximises detection within the budget;
4. report on test once.

The original K = 5 → 4 decision was also judged on test data, so say that too.

**How to answer at the viva:** "K trades false alarms against missed faults. Gaussian theory doesn't hold because the residuals are heavy-tailed, so we chose it empirically. K = 4 came from a two-point comparison. A later sweep suggests K = 3 is a better trade-off. Properly, K should be chosen on validation against an alarm budget."

**Q12. Why the 2 % floor? Why not 1 % or 5 %?** [reasoned, now measured]
At night and at dawn, MAD → 0, so the limit shrinks toward 0 W and tiny noise triggers alarms. The floor sets a minimum limit.

Sweep result for the 1 h model at K = 4:

| Floor | False flags | Detection |
|---|---|---|
| 1 % | 2.6 % | unchanged |
| 2 % (used) | 2.3 % | unchanged |
| 5 % | **0.9 %** | unchanged |

The **5 % floor more than halves false alarms** with no loss of detection. K = 3 with a 5 % floor gives about 0.9 false alarms per day, with 89 % dropout and 98 % spike detection.

**Caveats:**
- The injected faults are all daytime and large, so this test can't see what the floor costs at night.
- 5 % of capacity is about 27 kW for Solar and 44 kW for Grid. Faults smaller than that during low-output hours would be missed.
- Solar is 0 at night anyway, so the cost there is small.

What this tells us: most false flags come from low-output hours, where tiny absolute errors exceed tiny limits.

**Q13. Why 1.4826 × MAD?** [maths]
1.4826 = 1/Φ⁻¹(0.75). It makes MAD a consistent estimator of σ for Gaussian data, so "K robust σ" means the same thing as "K σ" would for clean Gaussian residuals.

**Q14. Why median and MAD rather than mean and standard deviation?** [reasoned]
They have a 50 % breakdown point. Outliers, including the anomalies themselves, can't inflate the threshold and hide themselves. The mean and SD have a 0 % breakdown point: one huge spike moves them.

**Q15. Why 24 hourly bins? Why not 96 quarter-hour bins, or hour × month?** [reasoned]
- **One global threshold:** far too tight at noon and too loose at night (the errors are heteroscedastic).
- **96 bins:** about 50 validation samples each instead of about 200, so MAD becomes noisy.
- **Hour × month:** validation covers only Sep–Oct, so there are no months to bin by.

There is a real weakness here. Sunrise and sunset shift between Sep–Oct (validation) and Nov–Dec (test), so the dawn and dusk bins are fitted to the wrong light conditions. Binning by **solar elevation** instead of clock hour would fix that.

**Q16. Why fit thresholds on validation residuals and not training residuals?** [reasoned]
The model fitted the training data, so training residuals are optimistically small and the thresholds would be too tight. Validation residuals reflect out-of-sample error, which is what live readings will show.

**Q17. Why check only the +15 min step and not all four?** [reasoned]
The first step is the most accurate, so it gives the tightest thresholds. It is also the earliest warning. Later steps have wider error and would only flag bigger faults, later.

**Q18. Why those fault types and magnitudes (dropout → 0, spike +50 % of capacity, damage −40 %)?** [arbitrary]
They are illustrative, not derived from field data:
- **+50 % of capacity** is a huge spike (about 273 kW at campus level), so a high detection rate is easy.
- **−40 % damage** is a severe loss for one plant but small against the campus total.

The honest, stronger answer is a **detection-vs-magnitude curve**: inject faults from 5 % to 100 % and report the smallest fault caught 90 % of the time. That curve wasn't produced.

**Q19. How precise are "69 %" and "92 %"?** ★
300 faults split into 3 types, so 100 per type. The binomial standard error at p = 0.69 is √(0.69 × 0.31 / 100) ≈ 4.6 points, so the 95 % interval is **69 % ± 9 points**. Differences under about 10 points between settings are not significant. Quote numbers like these to the nearest 5 %, not to one decimal place.

**Q20. Why inject only in daylight (08–16 IST, actual > 10 % of capacity)?** [reasoned]
A solar dropout at night (0 → 0) is not a fault and can't be seen. Restricting injection to daylight measures what matters. But it means night-time Grid and Demand faults were never tested.

---

## 4. Quantization

**Q21. Why W8A16? Why not W8A8, W16A16 or 4-bit weights?** [measured, except 4-bit]

| Option | Evidence |
|---|---|
| W8A8 | Measured on the NPU: the 24 h model drops to 14.7 dB (R² 0.34 → 0.17), to save about 0.7–2 ms |
| W16A16 | Would be more accurate. Saves no weight memory, and the HTP's 8-bit weight paths are faster |
| W8A16 | Already within 0.004 R² of FP32, so the extra precision of W16A16 buys nothing |
| 4-bit weights | Not tried |

**Q22. Why ±10 % output headroom? Why not 0 % or 25 %?** [reasoned, partly measured]
- **0 %:** measured to clip. Validation forecasts reached −0.068 against a calibrated minimum of −0.022.
- **More headroom:** coarser steps. ±10 % widens the range by 1.2×, so the step size grows 1.2× and noise power 1.44×, about 1.6 dB in theory. At 16 bit that's negligible, because there are tens of dB to spare. At 8 bit it showed up almost exactly as predicted: 1.8 dB (single node, W8A8, 44.0 → 42.2).
- 10 % covered the overshoot seen on validation. Values between 5 % and 25 % weren't compared.

**Q23. Why fix the LSTM hidden range at exactly [−1, 1]?** [maths]
h = o · tanh(c) lies strictly inside (−1, 1). It is the tightest range that can never clip, so no data is needed.

**Q24. Why 1,024 calibration windows? Why 512 for the converter? Why 600 for selection?** [arbitrary / practical]
- **Min-max calibration needs coverage, not count.** What matters is including the extremes, which is why the windows are stratified by hour.
- **1,024** is plenty for coverage and also feeds AdaRound.
- **512 for the converter** keeps the input list and file I/O manageable.
- **600 evenly spaced validation windows** keep AutoQuant runtime to minutes.
- **Evidence it doesn't matter much:** test SQNR ≈ validation SQNR, so the subsets were representative. No ablation over sizes was run.

**Q25. Why stratify the calibration windows by hour?** [reasoned]
A random draw over-represents common conditions and under-represents rare ones, such as dawn ramps and peak noon. Their ranges then get clipped in deployment. Equal shares per hour guarantee all regimes are seen.

**Q26. Why 10,000 AdaRound iterations?** [default]
AIMET's default. The optimisation converges well before that for layers this small. Not tuned.

**Q27. Why an AutoQuant target of 60 dB?** [reasoned]
It is deliberately unreachable. AutoQuant stops at the first stage that meets the target, and an earlier, lower target caused early exits. 60 dB forces every stage to run, and AutoQuant then returns the best result.

**Q28. Why 30 dB as the pass threshold for the simulation check?** [arbitrary sanity gate]
30 dB ≈ 3 % RMS error, a "nothing is badly broken" gate. The real acceptance criteria were behavioural: |ΔR²| ≤ 0.01 and anomaly-flag agreement. All models cleared 45 dB, so the exact gate never mattered.

**Q29. Why SQNR as the main metric and not R² or MAE?** [reasoned]
- It measures *what quantization changed*, needs no labels, and isn't masked by the model's own error.
- R² can even improve by chance after quantization (it did: 0.8522 → 0.8529).
- **Weakness:** SQNR is relative to signal energy, so low-energy outputs (Demand in winter) score lower for the same absolute error.

**Q30. Why per-tensor weights when per-channel simulated 1.5–3 dB better?** [measured]
Per-channel dense-layer weights gave 14.6 dB *on the device* in our conversion path, against 46.2 dB per-tensor. A gain in simulation that breaks on hardware is worthless.

**Q31. Why min-max ("tf") and not the SQNR-searching scheme ("tf-enhanced")?** [measured]
AutoQuant compared them: 52.15 vs 52.06 dB. They are effectively tied, and min-max is simpler.

---

## 5. Benchmarking and evaluation

**Q32. Why latency over 300 runs with `--perf_profile burst`?** [reasoned]
- 300 runs average out scheduling jitter.
- `burst` pins clocks high for stable, best-case numbers.
- A real 15-minute deployment would use a low-power profile: slower per inference, but less energy. The burst number is best case, not typical.

**Q33. Why is the headline R² the mean of Solar and Grid?** [reasoned]
Demand's R² is negative even in FP32. Including it would hide the quantization comparison behind a broken head. State the exclusion every time, so it doesn't look like cherry-picking.

**Q34. Why stride-1 windows when consecutive windows are 95/96 identical?** [inherited]
It maximises training samples and evaluates at every 15-minute point. But the samples are highly correlated, so "4,832 test windows" overstates the independent evidence: it is about 50 days of data. Error bars should account for that, for example with a block bootstrap by day.

---

## 6. Summary

| Number | Origin | Settled? |
|---|---|---|
| 15 min, 96-step lookback, 1 h / 24 h horizons | reasoned / inherited | no ablation |
| 70/15/15, forward-fill limit 4, MinMax | inherited / reasoned | no |
| Architecture, lr, dropout, epochs | inherited defaults | no |
| **K = 4** | two-point test on the test period | **sweep says K = 3 is better** |
| **2 % floor** | reasoned | **sweep says 5 % halves false alarms** (night-time cost untested) |
| 24 hourly bins | reasoned | dawn/dusk bins shift with season |
| Fault magnitudes, 300 faults | arbitrary | ± 9 pt uncertainty; no magnitude curve |
| W8A16, per-tensor, tf | measured on device | yes |
| ±10 % headroom, [−1, 1] prior | reasoned / maths | partly |
| 1,024 / 512 / 600 windows, 10,000 iterations, 60 dB target | practical / default | no ablation, but test ≈ validation |
