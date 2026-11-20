# Mid-Evaluation Viva: Hostile Question Bank

Questions an examiner could ask to check that the team understands **what** each part does, **how** the parts connect, and **why** each idea works. Answers are short so they can be said out loud. They use the project's own numbers (docs `aimet_quantization/10`–`17`, `explanation.md`).

★ marks a trap question: the obvious answer is wrong or incomplete.

---

## 1. The project in one breath

**Q1. What does the project do, in two sentences?**
Three LSTM forecasters read the last 24 h of campus sensor data (96 × 15 min). They predict Solar generation, Grid draw and room Demand for the next 1 h or 24 h. The models were quantized with AIMET to W8A16 and run on the QIDK's Hexagon NPU. A residual check on the 1 h forecast flags sensor readings that deviate too far from what the model expected.

**Q2. Is this an anomaly-detection model?** ★
No. All three models are regression forecasters. Anomaly detection is a statistical rule applied afterwards to the forecast error (the residual). No model was trained to detect anomalies.

**Q3. Why three models and not one?**
- `all_sensors_1h` serves real-time decisions and the anomaly check.
- `all_sensors_24h` serves day-ahead planning.
- `single_node` covers one plant (Palash, SL-PL00-00). A campus total can't tell you *which* plant misbehaves, and a single-plant fault is diluted in the total.

**Q4. What does "net-zero" mean here, and do your models compute it?**
Net deficit = Demand − Solar. Solar ≥ Demand means surplus. The models forecast the inputs to that equation (Solar, Grid, Demand); net-zero is derived from them. Note: the team's original runner printed `grid + solar − demand`, which contradicts the reports. It is cosmetic, but know which definition you use.

**Q5. Why run on an edge device at all, when data arrives only every 15 minutes?** ★
Latency is *not* the reason: 3.4 ms against a 15-minute cadence is 230,000× slack, and a laptop could also do it.
The real reasons are:
- telemetry stays on site (privacy, no cloud dependency, works offline);
- low energy per inference;
- the course target is on-device NPU deployment.
Don't claim the NPU was needed for speed.

**Q6. What was your team's contribution vs what already existed?**
The team trained and exported the FP32 models (`model2/`).
The quantization phase:
- rebuilt the missing dataset from raw telemetry;
- added the anomaly layer;
- quantized with AIMET without retraining;
- converted to QNN and deployed on the QIDK HTP;
- verified each stage;
- built a dashboard served by the board itself.
Be ready to say who did which part.

---

## 2. Data

**Q7. How many nodes and features, and how do 58 features break down?**
11 nodes: 6 solar, 2 grid/substation, 3 room meters. 58 = 41 electrical channels + 11 `was_imputed` flags + 6 cyclic time features. The single-node model uses 10 features (Palash's 4 electrical channels + 6 cyclic features).

**Q8. Define the three targets exactly.**
- **Solar** = Σ `active_power_W` over the 6 SL nodes.
- **Grid** = SB00 `gd_power_W` + KH95 `power_total_W`.
- **Demand** = Σ `power_W` over the 3 SR-EM room meters.

Each target is divided by its capacity (the max of that sum), so the outputs are fractions of capacity.

**Q9. Are the targets also inputs?** ★
Yes, in a sense. The components of each target (each plant's active power, the grid meters, the room meters) are input channels. The model sees the past of the quantity it forecasts. This is legitimate autoregressive forecasting, not leakage: the targets lie strictly *after* the input window. It is also why naive persistence is a strong baseline (Q31).

**Q10. Why 15-minute resolution?**
Raw cadence was irregular (10 s to 5 min, differing per device). Resampling to a 15-min mean gives one common, aligned timeline, matches typical utility metering intervals, and smooths sensor noise. The price: events shorter than 15 min (a brief cloud transient) are averaged away.

**Q11. What preprocessing was applied, and why each step?**
- **kW/W unification:** devices report different units.
- **`abs()`:** some CT clamps were installed reversed, producing negative power.
- **15-min resampling:** irregular cadence.
- **Forward-fill (limited) + `was_imputed`:** gaps.
- **`fillna(0)`:** remaining gaps.
- **MinMax to [0, 1]:** a common scale for the network.

**Q12. What is `was_imputed` and why feed it to the network?**
A 0/1 flag per node marking filled-in values. Without it, the network can't tell a flat forward-filled line from a genuinely constant reading. With it, the network can learn to trust imputed stretches less.

**Q13. Why encode time as sin and cos, and why both?** ★
Hour 23 and hour 0 are adjacent, but as raw numbers they're 23 apart. Mapping to a circle fixes that. You need *both* coordinates because sin alone is ambiguous: sin(6 h) = sin(18 h), i.e. morning and evening look identical. (sin, cos) is a unique point on the circle.

**Q14. In what form do the cyclic features enter the model?** ★
They are **also MinMax-scaled**, so effectively (sin + 1)/2 in [0, 1]. This was discovered during the rebuild: decoding the sample window as plain sin/cos gave uneven 4-minute steps, while decoding as 2x − 1 gave exact 15-minute steps. Feeding raw [−1, 1] sin/cos is the most likely integration bug.

**Q15. Which timezone, and why does it matter?**
The index is UTC, and the cyclic features are computed in IST (UTC+5:30). Solar follows local solar time, so IST hour is physically meaningful. Mixing these up shifts the model's sense of "noon" by 5.5 hours.

**Q16. Why did you have to rebuild the dataset, and how do you know the rebuild is faithful?**
The normalized dataset and its first-stage script were only on a teammate's machine. The first stage was rebuilt from raw telemetry. The team's original second stage (`generate_lstm_dataset`) was *imported, not copied*.

Evidence of fidelity:
- identical timeline (32,866 bins) and identical test start;
- exact grid and demand capacities;
- solar capacity within 1.9 %;
- FP32 Solar R² within 0.007–0.012 of the reported values;
- 44/58 features of the reference window match within 0.01; the rest are auxiliary voltage/current channels.

**Q17. Name three real data bugs found during the rebuild.**
1. The parquet index was *server-ingest time*, not sensor time. For SB00 it was off by a median of 105 days, so `Sensor_Timestamp` is used instead.
2. **SL-VN95-00 had its active and reactive power registers swapped.** Fixing it moved solar capacity from 386 kW to 536 kW.
3. SL-NI01-00 reports Watts, not kW.

Also: KH95 had the wrong schema in parquet, so it was taken from CSV, and the matching max of 1,739.4 W confirmed the choice.

**Q18. Inverters stop sending at night. How is that handled?**
Inverter AC readings are forward-filled through the night with `was_imputed = 1`. The DC PV-string voltage is not carried over (it really is 0 at night). This matched the original sample window step by step.

**Q19. How was the data split?**
Chronologically, 70/15/15 (train/val/test), with no shuffling. The test period is 2025-11-10 → 2025-12-31. Windows are built *inside* each split, so no window crosses a boundary.

**Q20. Why not random k-fold cross-validation?** ★
Consecutive windows overlap by 95 of 96 steps. A random split puts near-identical windows in both train and test, which inflates scores and is a form of leakage. Time series also have temporal order: you must predict the future from the past. The correct alternative to a single split is rolling-origin (walk-forward) evaluation.

**Q21. Is there any data leakage?** ★
The major sources are controlled:
- the split is chronological, with no cross-boundary windows;
- model selection, thresholds and K tuning used validation only;
- the calibration set came from train only.

There is **one known minor leak**: `MinMaxScaler` was fit on the *whole* series before splitting (`generate_lstm_dataset.py:92`). The min/max therefore include test-period values. Only two numbers per feature leak, not patterns, so the effect is small. The clean fix is to fit the scaler on train only. The original training scripts were not re-audited, since the models were frozen inputs.

**Q22. Test is Nov–Dec. Does the test period represent the year?** ★
No. It is winter-ish, with lower solar, and room demand is low and quiet in those months. That is part of why Demand R² collapses there and why the 1 h model's test SQNR (46.8 dB) is below its validation SQNR (50.3 dB): Demand has little signal energy in Nov–Dec. Five months of training data also cover less than one full seasonal cycle.

---

## 3. The model: LSTM theory

**Q23. Describe the architecture.**
Input [1, 96, F] → LSTM(H) → LSTM(H), with dropout 0.1 between the LSTM layers during training only. Then the **last time step's hidden state** → Linear(H→M) → ReLU → Linear(M→O).

| Model | F | H | M | O | Params |
|---|---|---|---|---|---|
| 1 h | 58 | 96 | 48 | 12 = 4 × 3 | 139,749 |
| 24 h | 58 | 128 | 128 | 288 = 96 × 3 | 282,153 |
| single node | 10 | 64 | 32 | 4 | 55,021 |

**Q24. Write the LSTM equations.**
- Forget gate: f = σ(W_f x + U_f h + b_f)
- Input gate: i = σ(W_i x + U_i h + b_i)
- Output gate: o = σ(W_o x + U_o h + b_o)
- Candidate: g = tanh(W_g x + U_g h + b_g)
- Cell state: c_t = f ⊙ c_{t−1} + i ⊙ g
- Hidden state: h_t = o ⊙ tanh(c_t)

**Q25. Why an LSTM and not a plain RNN?**
In a vanilla RNN, gradients through 96 steps are repeated products of Jacobians, so they vanish or explode. The LSTM cell state has an additive update path (c_t = f·c_{t−1} + …). When f ≈ 1, gradients flow back largely unattenuated (the "constant error carousel"). Gates learn what to keep and what to forget.

**Q26. What range can h take, and what range can c take?** ★
h = o · tanh(c), with o ∈ (0, 1) and tanh ∈ (−1, 1), so **h ∈ (−1, 1) by construction**. The quantization range prior uses exactly this fact. **c is not bounded**: it can grow over steps. That is why the cell state is the dangerous tensor to quantize coarsely.

**Q27. Why use only the last hidden state?**
After 96 steps, the last h is the network's learned summary of the whole window. The MLP head maps that summary to all horizon steps at once. Alternatives exist (attention over all steps, seq2seq decoders) but would be larger.

**Q28. Is the 24 h model autoregressive over its 96 output steps?** ★
No. It is a **direct multi-output** model: one forward pass emits all 288 values. That avoids feeding errors back into itself (no compounding over the horizon). Each step does get its own error, and the model tends to predict a smooth "average day" far out.

**Q29. Why a 96-step (24 h) lookback?**
One full diurnal cycle. The window always contains yesterday at the same hour, which is the strongest seasonal predictor for solar and demand. A longer lookback costs linearly more compute (96 unrolled steps already means ≈ 12.8 M MACs).

**Q30. What do the initial states look like? Is the model stateful across windows?**
Zero initial h and c for every window. It is stateless: each inference is independent, which suits a one-window-per-request device API.

**Q31. Did the LSTM beat simple baselines?** ★
For the 1 h model, Solar R² was 0.955 vs **0.908 for naive persistence** ("next value = last value"). Grid was 0.794 vs 0.741. The gain is real but modest: at 15-minute resolution, the next value is very close to the current one. Always quote the baseline alongside the model.

**Q32. What loss and optimiser, and what does MSE imply?**
MSE loss, Adam (lr 0.002), batch 64, ReduceLROnPlateau (factor 0.5, patience 2), best-validation checkpoint, 10–15 epochs. MSE makes the model predict the **conditional mean**. Under uncertainty (e.g. clouds) it hedges toward smooth curves and under-predicts peaks. It also lets outputs go slightly negative (night solar), because nothing constrains the sign.

**Q33. What is R², and how can it be negative?**
R² = 1 − SS_res / SS_tot. That is 1 minus (model error energy) over (error energy of always predicting the test mean). R² < 0 means the model is worse than that constant. R² depends on the test variance: a low-variance target makes even small errors look terrible.

**Q34. Why is Demand R² negative (−0.34 reported, −0.66 rebuilt)?**
Room demand is small (≈ 30 kW cap against hundreds of kW for solar and grid). It is noisy and behaviour-driven, and in Nov–Dec it has low variance. The Demand head shares one network with two much larger targets. MSE on capacity-normalized outputs weights them equally, but the demand signal is weak. So Demand is excluded from the headline R² and from anomaly verdicts.

**Q35. What is the "headline R²"?**
The mean of the Solar and Grid R² for the campus models, and Solar R² for the single node. Demand is excluded. Say this explicitly, otherwise it looks like cherry-picking.

**Q36. The original report says the 24 h Grid R² is 0.741. Do you believe it?** ★
No. 0.741 is exactly the **persistence baseline's** 1 h Grid R² in `report.md`. It also contradicts the model's own MAE (74.3 kW in the metadata; we measure 73.1 kW). The real 24 h Grid R² is about −0.1. It was most likely a copy error.

---

## 4. Anomaly detection

**Q37. State the rule.**
r = actual(t+15) − forecast(t+15), per domain, in Watts. A reading is flagged when |r − median_h| > max(4 · 1.4826 · MAD_h, 2 % of capacity). Here h is the IST hour, and median_h and MAD_h come from FP32 residuals on the validation set.

**Q38. Why the median and MAD instead of the mean and standard deviation?**
They are robust. Outliers, the very anomalies you want to find, inflate the mean and SD and mask themselves. The median and MAD have a 50 % breakdown point.

**Q39. Where does 1.4826 come from?**
1.4826 = 1 / Φ⁻¹(0.75). For Gaussian data, 1.4826 · MAD estimates σ consistently. So 4 · 1.4826 · MAD is "4 robust σ".

**Q40. With K = 4, Gaussian theory gives about 0.006 % false flags. You observe 2.3 %. Why?** ★
The residuals are **not Gaussian and not stationary**. They are heavy-tailed: cloud transients produce large but legitimate errors. The test period (Nov–Dec) also differs from validation, where the thresholds were fit. The empirical false-flag rate is the number that matters, and it was measured.

**Q41. Why thresholds per hour of day?**
The error is heteroscedastic. Noon solar is hard to forecast and midnight solar is trivially 0. One global threshold would be too loose at night and too tight at noon.

**Q42. Why the 2 % floor?**
At night MAD ≈ 0, so the limit would shrink to ≈ 0 W and any tiny noise would trigger an alarm.

**Q43. Why centre on the median residual?**
The forecasts are systematically biased in some hours (strongly for Demand). Without centring, the first version flagged 48 % of Demand points.

**Q44. How did you choose K = 4?**
On injected faults versus false flags: going from K = 5 to K = 4 raised 1 h dropout detection from 32 % to 69 %, at about 2 % false flags (Solar). It is a precision/recall trade-off. A full ROC sweep was not done.

**Q45. What does it detect, and what not?**

| Fault | 1 h campus | single node |
|---|---|---|
| Sensor dropout | 69 % | 92 % |
| Spike (+50 % of capacity) | 91 % | 83 % |
| Panel damage (−40 %) | 0 % | 17 % |
| False flags on normal data | 2.3 % | 2.7 % |

**Q46. Why is panel damage invisible?** ★
Detectability is a signal-to-noise problem. Losing 40 % of one 10–60 kW plant is a 4–24 kW drop. The campus forecast error at noon is tens of kW, and the noon limit is ±275 kW. The fault sits inside normal error. Per-plant models do better because the signal isn't diluted. Aggregating residuals over many days (CUSUM/EWMA) would detect persistent small losses.

**Q47. The limit at noon is ±275 kW, about 65 % of the forecast. Isn't that useless?** ★
It is 4 robust σ of the *actual* forecast error at that hour. It reflects how uncertain a 15-minute solar forecast is at noon, not a design choice. It catches gross sensor faults (dropouts, spikes) and misses subtle ones. To tighten it, you need a better forecaster, per-plant models, a lower K (more false alarms) or temporal aggregation.

**Q48. A sensor dies and stays dead. What happens to the detector over the next hours?** ★
The dead reading enters the next input windows. The model sees zeros in its history and starts forecasting low values, so the residuals shrink and later readings may no longer be flagged. The detector catches the *onset* best. Persistent-fault logic (alarm latching, or a rule for "stuck at 0 during daylight") would be needed.

**Q49. How many false alarms per day should operators expect?**
About 2.3 % × 96 readings ≈ 2 Solar false flags per day, plus about 1 for Grid. That is acceptable for a demo, too many for paging someone. Real deployment would need an alarm policy, e.g. 2 consecutive flags.

**Q50. Why can't the 24 h model serve as the detector?**
Its +15 min output is optimised as part of a whole-day forecast, not as a sharp short-term prediction. Dropout detection is 8 % and spike detection 30 %.

**Q51. In the dashboard, why doesn't editing the input window trigger alarms, while editing row 97 does?** ★
The check compares the **next reading** (row 97) with the forecast made from rows 1–96. If you corrupt the input, the *forecast* moves with it, and there is nothing to compare against. A fault must appear in the arriving reading to be judged.

**Q52. What are `k` and `K` in the dashboard?** ★
Lowercase `k` is the editor's fault strength (scale × k, or spike + k × max). Capital `K = 4` is the threshold multiplier. They are unrelated.

---

## 5. Quantization theory

**Q53. What is quantization?**
Represent a real value x as an integer q with real ≈ scale · (q − offset). For b bits there are 2^b levels. The scale is (max − min) / (2^b − 1). Values inside the range are **rounded** (error ≤ scale/2). Values outside are **clipped**.

**Q54. What are the two error sources, and how do they trade off?**
Rounding error (resolution) and clipping error (range). A wider range means less clipping but coarser steps. Range setting is choosing that trade-off. Min-max calibration has zero clipping on the calibration data and maximum step size.

**Q55. What is the quantization noise power, and what SQNR should you expect per bit?**
For a uniform step Δ, the error is roughly uniform with power Δ²/12. For a full-range signal, SQNR ≈ 6.02 · b + 1.76 dB: about 50 dB at 8 bit and 98 dB at 16 bit. Real signals don't fill the range, so actual SQNR is lower. That is why 48 dB at W8A16 is good.

**Q56. Define SQNR and translate the numbers.**
SQNR = 10 · log10(Σ fp32² / Σ (fp32 − q)²) on identical inputs.
- 30 dB ≈ 3 % RMS error
- 40 dB ≈ 1 %
- 48 dB ≈ 0.4 %

It is a ratio of energies, so it depends on signal energy. Low-energy outputs (Demand in winter) get lower SQNR from the same absolute error.

**Q57. Why measure fidelity (SQNR vs FP32) and not accuracy vs ground truth?** ★
Quantization should change the model as little as possible. Comparing against ground truth mixes quantization error with the model's own error. The quantized model can even score *higher* R² by chance (it did: 0.8522 → 0.8529). SQNR isolates what quantization changed and needs no labels. Accuracy and flag agreement are reported too.

**Q58. Symmetric vs asymmetric. Which for weights and why?**
- **Symmetric** (offset 0) for weights: weights are centred near 0. A zero offset avoids extra cross terms in the integer matmul: Σ(q_w − z_w)(q_x − z_x) expands into correction terms.
- **Asymmetric** for activations: e.g. post-ReLU tensors are all ≥ 0, and a symmetric range would waste half the levels.

**Q59. Per-tensor vs per-channel?**
One scale per weight matrix vs one scale per output channel (row). Per-channel handles rows with very different magnitudes and is usually more accurate (+1.5–3 dB in our simulation). We deploy **per-tensor**, because per-channel Gemm weights produced wrong results on the HTP in our conversion path (14.6 dB vs 46.2 dB).

**Q60. Why are biases 32-bit?**
In an integer matmul, the accumulator has scale s_w · s_x. The bias is added into that accumulator, so it is quantized to the same tiny scale, which needs 32 bits to cover its range. It costs almost nothing (few biases). The LSTM biases are 16-bit on device because the expanded LSTM turns them into Add operands.

**Q61. What does W8A16 mean, and why not W8A8?** ★
8-bit weights and 16-bit activations. The LSTM carries its state through 96 recurrent steps, so activation error re-enters the computation at every step and compounds. On the NPU, W8A8 dropped the 24 h model to **14.7 dB** (R² 0.34 → 0.17) and saved only about 0.7 ms. HTP v75 supports 16-bit activations natively. The 8-bit weights still give the ≈ 3.8× size reduction.

**Q62. Why is size reduction 3.8× and not 4×?** ★
Biases stay at 32 bit (16 for the LSTM), and quantization encodings add a little metadata. Also: the compiled `.so` files are 3.8–4.0 MB, larger than the FP32 ONNX file, because they contain graph-construction code for 96 unrolled steps. Weight size and binary size are different things.

**Q63. PTQ vs QAT. Why PTQ?**
PTQ needs only unlabeled calibration data, with no retraining. "No retraining" was a ground rule. QAT simulates quantization during training and lets weights adapt, which is usually better at very low bit widths. At ≥ 46 dB and |ΔR²| ≤ 0.004 there was no gap for QAT to close.

---

## 6. AIMET specifics

**Q64. What is AIMET, and what does it output?**
Qualcomm's AI Model Efficiency Toolkit. It wraps the model in **QuantSim**, a graph with quantize/dequantize nodes that simulate integer arithmetic in floating point. It calibrates ranges, runs optimisations (AdaRound etc.), and exports `model.onnx` (with adjusted weights) plus `model.encodings` (scale/offset/bitwidth per tensor).

**Q65. What is calibration, and how was the calibration set built?**
The model runs forward on representative data to observe each tensor's range. We used 1,024 windows from **train only**, stratified by IST hour of the last step (night, dawn, noon, dusk equally), with seed 0. Stratifying matters: a random draw under-represents rare regimes (dawn ramps), and those ranges get clipped later.

**Q66. What went wrong in the first quantization grid?** ★
16-bit activations *alone* (float weights) lost up to 0.037 of capacity (~20 kW). At 16 bit, rounding error is tiny, so resolution couldn't explain it. Enabling quantizers one at a time showed the cause was **range clipping**:
- Calibration outputs covered [−0.022, 0.987], but validation forecasts reached −0.068 (negative night solar).
- LSTM hidden outputs were calibrated to about ±0.7, but can reach ±1.

**Q67. What are the range priors, and how are they justified?**
1. LSTM hidden outputs are fixed to [−1, 1], a mathematical bound (Q26).
2. Model outputs get the calibrated range ± 10 % headroom, because forecasts over- and undershoot.

Effect on the 1 h model: activation-only SQNR went from 43.8 to 77.5 dB, and W8A16 from 42.4 to 47.6 dB. They are justified by maths and physics, not tuned on test.

**Q68. Is there any cost to the ±10 % headroom?**
Slightly coarser steps. At 16 bit it is negligible. At 8 bit it costs one step of resolution: the single-node W8A8 model went from 44.0 to 42.2 dB.

**Q69. What does AdaRound do, conceptually?** ★
Round-to-nearest minimises error *per weight*, but what matters is the *layer output* error ‖Wx − Ŵx‖². Errors of individual weights can cancel or add. AdaRound learns, per weight, whether to round **up or down**. It uses a continuous relaxation (a rectified sigmoid) plus a regulariser that pushes each choice to 0 or 1, and it is optimised layer by layer on unlabeled calibration data. It is justified by a second-order Taylor expansion of the loss, which shows that rounding errors interact through the Hessian. Here it ran 10,000 iterations per layer and gained +2.7 / +3.8 / +1.2 dB.

**Q70. What does AutoQuant do?**
AutoQuant runs quant-scheme selection → batch-norm folding → cross-layer equalization → AdaRound, evaluating your metric after each stage. It stops early if the target is met and returns the best model. We set the target to 60 dB, deliberately unreachable, so every stage ran.

**Q71. Why did BN folding and CLE do nothing?** ★
- **BN folding:** there is no BatchNorm in the model.
- **CLE:** it rescales weights between consecutive layers through a positively homogeneous activation (ReLU(s·x) = s·ReLU(x)), equalising per-channel ranges for per-tensor quantization. It is designed for conv→conv chains. Our only candidate is the Gemm→ReLU→Gemm head, and CLE left the SQNR unchanged (e.g. 52.15 dB before and after). The LSTM, with its sigmoid and tanh, is not homogeneous, so CLE can't apply there at all.

**Q72. What is "tf" vs "tf-enhanced"?**
These are range-setting schemes:
- **tf:** plain min-max.
- **tf-enhanced:** searches for a range that maximises SQNR, accepting some clipping.

Selection picked `tf` (52.15 vs 52.06 dB).

**Q73. What was wrong with the LSTM weights in the first export?** ★
PyTorch's ONNX export reorders the LSTM gate weights through Slice/Concat. The weight initialisers therefore reach the LSTM *through ops*, and AIMET saw them as **activations** and gave them 16-bit quantizers. The "W8A16" label was dishonest: about 95 % of parameters were 16-bit. The fix (`lstm_weights_as_params()`) walks back from each LSTM W/R input and forces 8-bit symmetric. The compiled device graph confirms 8-bit LSTM weights.

**Q74. Name two AIMET tooling gotchas.**
1. In aimet-onnx 2.12, AutoQuant lives only in `aimet_onnx._deprecated.auto_quant_v2`, and 2.12 is the last version with ONNX AutoQuant (Python 3.10 only).
2. It builds `AdaroundParameters` with `num_iterations=None`, which raises a TypeError. We fixed it by setting the parameters explicitly.

Also: QuantSim renames graph outputs, so prior tensors must be looked up by name in the *un-instrumented* graph.

**Q75. Does AIMET's simulation predict the device exactly?** ★
No. AIMET-ONNX simulates the LSTM as **one op**: it quantizes the op's inputs, weights and outputs, but computes gates and cell state in float. On the HTP, the LSTM is unrolled and every internal tensor is quantized. That is why W8A8 looked fine in simulation (40.6 dB for 24 h) and failed on device (14.7 dB). On-device verification is mandatory, not optional.

**Q76. Did AIMET actually help, or would the QNN converter alone suffice?**
Same bit widths, converter and calibration windows. On the NPU, AIMET gave +3.5 / +2.6 / +5.9 dB, and the worst-case error on the 1 h model dropped 6× (2.6 % → 0.45 % of capacity). The gain came from the range priors and AdaRound.

---

## 7. Conversion and the Hexagon NPU

**Q77. Walk through the deployment pipeline.**
1. FP32 ONNX → AIMET → `model.onnx` + `model.encodings`.
2. Filter the encodings (keep the input, ReLU and output activations plus the Gemm weights).
3. `qnn-onnx-converter --quantization_overrides … --input_list calib(512) --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32 --expand_lstm_op_structure --preserve_io layout`.
4. `qnn-model-lib-generator -t aarch64-android` → `lib<model>.so`.
5. On the QIDK: `qnn-net-run --backend libQnnHtp.so --model lib….so`.

**Q78. Why the classic `qnn-onnx-converter` and not the newer `qairt-converter` + `qairt-quantizer`?**
The two-step QAIRT flow tagged the unrolled LSTM internals (which have no overrides) as float fallback. The quantizer then failed ("bitwidth conversion from 8 to 32 not supported"). The one-step converter takes the overrides *and* calibrates the uncovered tensors from `--input_list`.

**Q79. Why drop some AIMET encodings before conversion?**
- Encodings on **constant subgraphs** (reordered LSTM weights, zero initial states): the converter folds these into static parameters and quantizes them itself.
- The **LSTM-hidden prior**: in the expanded graph, per-step tensors alias it, and overriding it scaled values by **×1.44** on device (1/0.6947, the ratio of calibrated range to the [−1, 1] prior).

**Q80. What does `--expand_lstm_op_structure` do, and why?**
It rewrites every LSTM step into MatMul/Add/Sigmoid/Tanh/Mul, standard HTP ops. The fused quantized LSTM op failed HTP graph validation (`MODEL_GRAPH_OP_VALIDATION_ERROR`). It also kept the recurrent state between steps at 8 bit despite `--act_bitwidth 16`. After expansion, all 3,853 activations are 16-bit.

**Q81. Why `--preserve_io layout`?** ★
By default the converter re-lays [1, 96, F] as channel-first [1, F, 96]. Our raw input files were [1, 96, F]. The model read transposed data and produced garbage (4.9 dB). Preserving the layout means apps feed the same array as for ONNX.

**Q82. What does "unrolling" mean, and how many MACs is the 1 h model?**
The 96 recurrent steps are laid out as 96 copies of the cell in a static graph (the HTP needs a static graph). Per step:
- layer 1: 4 · 96 · (58 + 96) ≈ 59 k MACs
- layer 2: 4 · 96 · 192 ≈ 74 k MACs
- total ≈ 133 k per step

× 96 steps ≈ **12.8 M MACs**.

**Q83. The original report claimed 141,696 MACs and 0.092 ms. Comment.** ★
141,696 ≈ one LSTM time step plus the head. The report counted one step, not 96. The measured NPU time is 3.4 ms, about 40× larger than claimed.

**Q84. What is the HTP, and why does it suit this model?**
The Hexagon Tensor Processor (v75 on SM8650) is Qualcomm's NPU. It has scalar, vector (HVX) and tensor (matrix) units, optimised for fixed-point arithmetic at low power. It natively supports 8- and 16-bit integer activations, which is why W8A16 runs without float fallback.

**Q85. What are `libQnnHtpV75Stub.so` and `libQnnHtpV75Skel.so`?**
The two halves of a FastRPC call. The **Stub** runs on the CPU (Android side) and marshals the call. The **Skel** runs on the Hexagon DSP and executes it. That is why `ADSP_LIBRARY_PATH` must point at the folder with the Skel.

**Q86. Explain the three latency numbers: 3.42 ms, 5.11 ms, and ~2 s in the dashboard.** ★
- **3.42 ms:** HTP execute time (the accelerator only).
- **5.11 ms:** through the QNN API, including the CPU↔DSP RPC.
- **~2 s:** the dashboard launches `qnn-net-run` per request, which loads the library and prepares the HTP graph (0.9–1.5 s) each time.

A persistent QNN context, or a serialized **context binary**, removes the prepare cost.

**Q87. "0.286 ms on ONNX Runtime" vs 3.4 ms on the NPU: why is the NPU slower?** ★
The 0.286 ms figure is FP32 on a host CPU, not on the board, and not on the NPU. Different hardware and different measurement. The NPU path also pays fixed per-call overhead (RPC, data transfer) that dominates for a model this small. On the board's own CPU, FP32 ran at about 15–31 ms per inference including file I/O. Both are irrelevant against 15 minutes. The NPU's real advantage is energy, not speed here.

**Q88. "Bitwise numerical parity, Δ < 2.38 × 10⁻⁷": is that wording correct?** ★
No. Bitwise parity means Δ = 0. 2.38 × 10⁻⁷ ≈ 2⁻²², about float32 rounding near 1. Say "matches to float32 precision". Note that *on the device* we did get true bit-exactness: the converted FP32 graph on the QIDK CPU matched ONNX exactly (V3).

**Q89. Why static batch size 1 and opset 18?**
Static shapes allow the QNN converter to build a fixed graph and the HTP to allocate memory up front. Batch 1 matches the use: one window every 15 minutes. Opset 18 is what the team exported with, and the converter supports it.

**Q90. Why is there no Python on the device, and why does that matter?** ★
The QIDK runs Android. There is no Python or ONNX Runtime. The team's original `qidk_deploy.sh` ran `python3` on the device and could never have worked. The deployment was rewritten around native `qnn-net-run`, and the dashboard server is native C++ (NDK).

**Q91. How would you move to a different Snapdragon?**
Rebuild with the same script (the model library is chip-independent). Push the matching `libQnnHtpV<arch>Stub/Skel.so`. Optionally use the matching AIMET HTP config (v73, v79, …).

---

## 8. Verification

**Q92. How do you know the deployed model is correct? Walk the ladder.**
Each rung isolates one error source:

| Rung | Check | Result |
|---|---|---|
| V0 | Rebuilt data reproduces reported FP32 accuracy | yes |
| V1 | AIMET graph with quantizers off = ONNX Runtime | 1.2 × 10⁻⁷ |
| V2 | AIMET simulation | ≥ 30 dB; got 45.8–51.8 dB |
| V3 | Converted FP32 graph on device CPU = ONNX | bit-exact |
| V4 | NPU W8A16 ≈ AIMET simulation | 45–55 dB |
| V5 | AIMET beats converter-only | yes, on all models |

All of these ran on thousands of held-out windows.

**Q93. Why is V3 (CPU FP32 on device) important?**
It proves the *conversion* (graph rewriting, unrolling, layout) is lossless. Any remaining error on the NPU is therefore pure quantization, not a conversion bug.

**Q94. How did you find the ×1.44 and layout bugs?**
With `qnn-net-run --debug`, which dumps every intermediate tensor. One window was run on HTP-INT and on CPU-FP32 and compared tensor by tensor and step by step. This showed the recurrence was stable (gate SQNR 37–45 dB over all 96 steps) and pinpointed exactly the mis-scaled tensors.

**Q95. The device SQNR (48.4 dB) is higher than the simulation's for the 1 h model. Isn't that suspicious?** ★
It is explainable. On device, the LSTM hidden outputs use their calibrated (narrower) range, which gives finer steps, because the [−1, 1] prior was not passed to the converter. The simulation used the conservative prior. These are different range choices, not an error.

**Q96. What does "99.66 % identical anomaly decisions" mean?**
For 99.66 % of (window, domain) readings, the quantized model raised exactly the same flag/no-flag decision as FP32. For injected faults it is ≥ 99.92 %. This is the behavioural test: does quantization change what an operator would see?

**Q97. Did you overfit the quantization choices to validation?**
The selection used 600 validation windows. The test SQNR matches the validation SQNR (except the Demand-energy effect), and re-loading the export reproduces validation exactly. The range priors are derived from maths, not tuned.

---

## 9. Dashboard and system integration

**Q98. What runs where?**
- **On the QIDK:** everything. A native C++ server (`nz_dashboard`, cpp-httplib, built with the NDK) serves the page and data. Its `POST /api/run` calls `qnn-net-run` on the HTP.
- **On the laptop:** the browser only displays, via `adb forward` over USB or via Wi-Fi.

**Q99. Which parts are live and which are recorded?**
The replay tabs show NPU outputs **recorded** on the device for every test window. "Run on NPU" computes a **new** forecast on the HTP at that moment. For an unedited window, the live output matches the recorded one to ≤ 6 × 10⁻⁷.

**Q100. How is an uploaded CSV turned into a model input?**
The browser:
1. parses 96 rows in physical units;
2. normalizes each column with the training min/max;
3. recomputes the cyclic features from the timestamps (in IST, then MinMax);
4. defaults `was_imputed` to 0;
5. checks the 15-minute spacing;
6. sends a float32 [96 × F] buffer.

Optional row 97 is the "actual next reading" for the anomaly check.

**Q101. Values outside the training min/max in a CSV: what happens?** ★
They normalize to values outside [0, 1]. The UI warns about them. The network never saw such inputs, and the quantized input range clips them on the NPU. The forecast becomes an extrapolation and can't be trusted, as in the k = 23.7 test, which gave negative Demand forecasts.

**Q102. Why does the window stride by 1 step, and how does the pack exploit it?**
Consecutive test windows overlap by 95 of 96 rows. The pack stores one timeline of n + 95 rows, and window i = timeline[i : i + 96]. That is 1.1 MB instead of 107 MB.

**Q103. Why does the dashboard server need `--daemon`?**
Without it, `adb shell` keeps the session attached and hangs. The server must detach from the shell.

---

## 10. Rapid-fire gotchas

**Q104.** Output units? → Fractions of capacity. Multiply by `capacities_W`; layout is reshape(H, 3) = [step, domain].

**Q105.** Model capacities vs data capacities? → Model outputs → Watts with the **model's** capacities (from metadata). Rebuilt targets use the **rebuilt data's** (solar differs by 1.9 %). Mixing them biases the residuals.

**Q106.** Why can the solar forecast be negative? → MSE regression has no sign constraint. It is clamped to ≥ 0 only for display. This is also why the output range needed headroom.

**Q107.** Why are dense weights symmetric 8-bit but LSTM weights asymmetric 8-bit on the device? → The converter quantizes the folded LSTM weights itself (asymmetric). The dense-layer weights come from AIMET's overrides (symmetric).

**Q108.** How many activation tensors in the expanded 1 h graph? → About 3,853, all 16-bit.

**Q109.** Why did the 1 h test SQNR drop below validation? → Low Demand energy in Nov–Dec. SQNR is relative to signal energy (Solar columns 52–54 dB, Demand 37–44 dB).

**Q110.** What is QIDK's SoC? → Snapdragon 8 Gen 3 (SM8650), HTP v75, Android 14.

**Q111.** Which runtime version and why does it matter? → QAIRT 2.47 on both host and device. The board originally had 2.37 libraries. A model library and runtime from different versions may not load, so 2.47 was pushed to its own folder (`/data/local/tmp/nz_aimet`).

**Q112.** What is `ADSP_LIBRARY_PATH` for? → It tells the DSP loader where to find the Skel library. Without it, the HTP backend fails to start.

**Q113.** Dropout in the deployed model? → None. Dropout is training-only, and the ONNX export is in eval mode.

**Q114.** Does the anomaly detector need timestamps? → No. The hour is decoded from the window's own `hod_sin/cos` features. (The dashboard CSV still needs timestamps to *compute* those features.)

**Q115.** What would you do with one more month? → Per-channel weights once the path supports them; a context binary for start-up time; per-plant forecasters and multi-day residual aggregation for panel damage; a train-only scaler fit; walk-forward evaluation; and an alarm policy (latching / consecutive flags).

---

## 11. Questions to fear most (prepare these out loud)

1. **"Your LSTM beats persistence by 0.05 R². Why not just ship persistence?"**
   The LSTM adds diurnal awareness and multi-step horizons, where persistence degrades fast (+60 min, 24 h). But be honest that the +15 min gain is modest. Quote the per-horizon comparison if you have it, and say so if you don't.
2. **"Show me that quantization didn't break anomaly detection."** → Flag agreement ≥ 99.66 % natural and ≥ 99.92 % injected, measured on device.
3. **"Why should I trust absolute R² if you rebuilt the data?"** → Don't, beyond ±0.03. Every quantization claim is relative (INT vs FP32 on identical inputs) and therefore unaffected.
4. **"W8A8 passed in simulation. How do you know W8A16 isn't also secretly broken?"** → It was measured on device against FP32 (48.4 dB), not just simulated. Tensor-level dumps confirmed the recurrence is stable over 96 steps.
5. **"What's the one thing you'd change about the original model?"** → Fit the scaler on train only. Give Demand its own model or loss weight. Evaluate walk-forward over seasons rather than on one Nov–Dec block.
