# Reading List: The Theory Behind the Project

Material for a formal understanding of every part of the project, grouped by topic. Each item names what in the project it explains.

References are cited from memory. Check titles, years and chapter numbers when you look them up.

★ = covers most of what an examiner would test. ★★ = read this first.

Companion files: [viva_questions.md](viva_questions.md), [viva_questions_deck.md](viva_questions_deck.md), [viva_questions_choices.md](viva_questions_choices.md).

---

## 0. The project's own map

Start with [explanation.md](explanation.md), then docs `10`–`17` in `aimet_quantization/`, then the three viva files. Use them as a checklist: for every claim, find the theory below that justifies it.

## 1. Mathematical prerequisites

| Reading | What it covers | Used for |
|---|---|---|
| Wasserman, *All of Statistics* | Estimators, consistency, the bootstrap, hypothesis tests | "Why 1.4826", "69 % ± 9" |
| Deisenroth, Faisal and Ong, *Mathematics for Machine Learning* (free PDF) | Linear algebra and calculus for ML | Everything below |
| Agresti and Coull (1998), "Approximate is better than 'exact' for interval estimation of binomial proportions" | Confidence intervals for rates | Uncertainty of the detection rates |

## 2. Time-series forecasting and evaluation

| Reading | What it covers | Used for |
|---|---|---|
| ★ Hyndman and Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. (free at otexts.com/fpp3). Read the chapters on simple benchmarks, residual diagnostics, accuracy measures and time-series cross-validation | Naive and seasonal-naive baselines, evaluation, WAPE/MAE, walk-forward evaluation | Slides 9–13; the benchmark-comparison questions |
| ★ Hewamalage, Bergmeir and Bandara (2023), "Forecast evaluation for data scientists: common pitfalls and best practices" | Leakage, split design, metric choice | Part 1, Q21 (leakage) |
| Cerqueira, Torgo and Mozetič (2020), "Evaluating time series forecasting models: an empirical study on performance estimation methods" | Walk-forward vs k-fold for time series | Why the split is chronological |
| Kaufman et al. (2012), "Leakage in data mining" | A formal definition of leakage | The scaler-fit leak |
| Makridakis et al. (2022), M5 competition results | Why LightGBM-style models dominate tabular forecasting | Slide 11 (LightGBM beats the LSTM) |
| Hong and Fan (2016), "Probabilistic electric load forecasting: a tutorial review" | Load-forecasting practice | Grid and Demand heads |
| Antonanzas et al. (2016), "Review of photovoltaic power forecasting" | PV forecasting, the role of weather | Solar heads; why day-ahead solar needs weather |

## 3. Neural networks and LSTMs

| Reading | What it covers | Used for |
|---|---|---|
| ★ Goodfellow, Bengio and Courville, *Deep Learning* (free online). Ch. 10 (sequence models, LSTM, back-propagation through time), Ch. 8 (optimisation, Adam), Ch. 7 (regularisation, early stopping) | The full theory of the model and its training | Model architecture and training |
| Olah, "Understanding LSTM Networks" (blog) | Intuition for the gates | LSTM equations; h ∈ (−1, 1) |
| Hochreiter and Schmidhuber (1997); Gers, Schmidhuber and Cummins (2000), "Learning to forget" | The original LSTM; the forget gate | Why an LSTM and not a plain RNN |
| Bengio, Simard and Frasconi (1994); Pascanu, Mikolov and Bengio (2013), "On the difficulty of training recurrent neural networks" | Vanishing and exploding gradients | Why an LSTM and not a plain RNN |
| Bai, Kolter and Koltun (2018), "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling" | Temporal convolutional networks (TCNs) vs RNNs | Slide 18 (TCNs) |
| Grinsztajn, Oyallon and Varoquaux (2022), "Why do tree-based models still outperform deep learning on tabular data?" | Tree models vs neural networks | Why LightGBM beat the LSTM |
| Ke et al. (2017), "LightGBM"; Friedman (2001), "Greedy function approximation: a gradient boosting machine" | Gradient-boosted trees | Slide 10 |

## 4. Anomaly detection and robust statistics

| Reading | What it covers | Used for |
|---|---|---|
| ★ Hundman et al. (2018), "Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding" (KDD) | The same design as ours: an LSTM forecast with a threshold on its residuals | The anomaly layer; ideas for improving it |
| Rousseeuw and Croux (1993), "Alternatives to the median absolute deviation"; Huber and Ronchetti, *Robust Statistics* | MAD, the 1.4826 constant, breakdown points | Median/MAD instead of mean/SD |
| Leys et al. (2013), "Detecting outliers: do not use standard deviation around the mean, use absolute deviation around the median" | A short case for median and MAD | Same |
| ★ Montgomery, *Introduction to Statistical Quality Control*, the chapters on Shewhart, CUSUM and EWMA charts | False-alarm rate, average run length, detecting slow drift | The K-σ rule; why panel damage needs CUSUM/EWMA |
| Chandola, Banerjee and Kumar (2009), "Anomaly detection: a survey" | A taxonomy of methods | Placing our method among alternatives |

## 5. Quantization (the core of the project)

| Reading | What it covers | Used for |
|---|---|---|
| ★★ Nagel et al. (2021), "A White Paper on Neural Network Quantization" (Qualcomm, arXiv 2106.08295) | Affine and symmetric quantization, range setting, post-training quantization, CLE, AdaRound, bias correction, quantization-aware training, in AIMET's own terms | Everything in docs 13–14 |
| Jacob et al. (2018), "Quantization and training of neural networks for efficient integer-arithmetic-only inference" | Integer matrix multiplies, zero points, 32-bit biases | Part 1, Q58 and Q60 |
| Nagel et al. (2020), "Up or down? Adaptive rounding for post-training quantization" | AdaRound, including the second-order Taylor argument | The AdaRound gain (+1.2 to +3.8 dB) |
| Nagel et al. (2019), "Data-free quantization through weight equalization and bias correction" | Cross-layer equalization (CLE) | Why CLE did nothing here |
| Siddegowda et al. (2022), "Neural network quantization with AI Model Efficiency Toolkit (AIMET)" (arXiv 2201.08442) | AIMET's design and workflows | AutoQuant, QuantSim |
| Gholami et al. (2021), "A survey of quantization methods for efficient neural network inference" | The broad landscape | Context |
| Krishnamoorthi (2018), "Quantizing deep convolutional networks for efficient inference: a whitepaper" | Per-channel vs per-tensor quantization | Part 1, Q59 |
| Widrow and Kollár, *Quantization Noise* (2008); or Gray and Neuhoff (1998), "Quantization" | The Δ²/12 noise power, 6.02 dB per bit | Reading SQNR figures |
| Ott et al. (2016), "Recurrent neural networks with limited numerical precision" | Why recurrence compounds quantization error | Why W8A8 failed on the 24 h model |

## 6. Hardware and deployment

| Reading | What it covers | Used for |
|---|---|---|
| Sze, Chen, Yang and Emer, *Efficient Processing of Deep Neural Networks* (2020) | Dataflows, why NPUs favour integers, energy per operation | "Why an NPU"; the energy argument |
| Hennessy and Patterson, *Computer Architecture: A Quantitative Approach*, 6th ed., chapter on domain-specific architectures | Accelerator design | What the HTP is |
| Qualcomm QAIRT/QNN SDK documentation | Converters, quantization overrides, HTP backend, context binaries | Doc 14, the conversion path |
| AIMET documentation | QuantSim, AutoQuant, encoding formats | Doc 13 |
| Hexagon SDK documentation on FastRPC | How CPU-side Stub libraries call DSP-side Skel libraries | `ADSP_LIBRARY_PATH`; the gap between 3.4 and 5.1 ms |
| Android NDK documentation | Native builds | The dashboard server |

## 7. The energy domain

| Reading | What it covers | Used for |
|---|---|---|
| Glover, Sarma and Overbye, *Power System Analysis and Design*, early chapters | Active and reactive power, power factor, three-phase systems | Slides 7–8; the input features |
| Kirschen and Strbac, *Fundamentals of Power System Economics* | Day-ahead and real-time scheduling | Why 1 h and 24 h horizons |
| India's CERC Grid Code | Scheduling time blocks | Verify the 15-minute time-block claim before quoting it |
| Duffie and Beckman, *Solar Engineering of Thermal Processes*, chapters on solar geometry and irradiance; pvlib documentation (clear-sky models) | Sun position, irradiance, clear-sky output | Why clouds change the amplitude of the solar curve, not its timing; binning thresholds by sun elevation |

## 8. The benchmark study's extras (slides 9–13, 18)

| Reading | What it covers | Used for |
|---|---|---|
| Angelopoulos and Bates, "A gentle introduction to conformal prediction"; Gibbs and Candès (2021), "Adaptive conformal inference under distribution shift" | Prediction intervals | The benchmark's uncertainty experiment |
| McMahan et al. (2017), "Communication-efficient learning of deep networks from decentralized data" | Federated averaging (FedAvg) | The federated experiment |
| Rawlings, Mayne and Diehl, *Model Predictive Control* (free PDF), first chapters | Model predictive control | Slide 18, the battery simulation |

---

## Suggested order (about two weeks)

1. Nagel's quantization white paper, alongside docs 13–14.
2. *Deep Learning* Ch. 10 and Olah's LSTM post, alongside the LSTM questions in [viva_questions.md](viva_questions.md).
3. *Forecasting: Principles and Practice* (benchmarks, evaluation) and Hewamalage et al.
4. Montgomery's control-chart chapters and Hundman et al., alongside the anomaly material and the K sweep in [viva_questions_choices.md](viva_questions_choices.md).
5. Jacob et al. and the AdaRound paper, for the deeper "why" behind quantization.
6. Everything else as reference, for the questions you can't yet answer.
