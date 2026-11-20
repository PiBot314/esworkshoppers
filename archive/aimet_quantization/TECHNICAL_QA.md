# Technical Q&A — AIMET Quantization and QIDK Deployment

Answers to technical questions about the process and the product. Numbers refer to the held-out test period measured on the QIDK unless stated otherwise. Step-by-step evidence is in docs `10`–`15`.

---

## A. The model

**Q1. What kind of model is this?**
A multi-step, multi-output time-series **regression** model (forecaster): a 2-layer unidirectional LSTM encoder plus a 2-layer MLP head. It takes the past 24 hours (96 × 15-min steps) of sensor data and predicts future power values. It isn't a classifier. Anomaly detection is a separate post-processing step on its forecast errors (Q4).

**Q2. Why three models?**
`all_sensors_1h` (1-hour horizon, campus totals) for real-time decisions and anomaly checks. `all_sensors_24h` (24-hour horizon) for day-ahead planning. `single_node` (one solar plant) because campus totals can't say *which* plant misbehaves.

**Q3. What do the inputs need to look like?**
Float32 `[1, 96, F]`. Each feature is MinMax-scaled to [0, 1] with the training min/max, in the exact order listed in the model metadata. The cyclic time features (sin/cos of hour, weekday, day-of-year in IST) are **also** MinMax-scaled, i.e. `(sin + 1) / 2` in practice. Feeding unscaled sin/cos values is a common integration mistake.

**Q4. How does it detect anomalous readings?**
`r = actual − forecast` at +15 min. A reading is flagged when `|r − median_h| > 4 × robust σ_h`, where median and σ are estimated per hour of day from validation residuals, with a floor of 2 % of capacity. Centring on the median removes systematic bias. The per-hour statistics reflect that forecasting noon solar is harder than night solar.

**Q5. What can and can't it detect?**
It detects sensor dropouts (69 % campus-level, 92 % plant-level) and spikes (83–91 %) with a ≈2–3 % false-flag rate. Partial panel damage (−40 %) is mostly invisible: 0 % at campus level, 17 % at plant level, because it's comparable to normal forecast error. The Demand head isn't usable (its FP32 R² is negative).

---

## B. Quantization choices

**Q6. Why post-training quantization and not quantization-aware training?**
The requirement was to keep the trained models unchanged. PTQ needs only representative unlabeled data for calibration. The result (≥ 46 dB fidelity, |ΔR²| ≤ 0.004) left no gap that QAT would need to close.

**Q7. Why W8A16 rather than full INT8?**
The LSTM state is carried through 96 recurrent steps, so activation quantization error compounds. Measured on the NPU, W8A8 drops the day-ahead model to 14.7 dB (R² 0.34 → 0.17) while saving only ~0.7 ms. HTP v75 executes 16-bit activations natively. Weights stay 8-bit, so the size reduction (≈3.8×) is kept.

**Q8. What exactly does AutoQuant do here, and did it help?**
AutoQuant (aimet-onnx 2.12) tries quant-scheme selection → batch-norm folding → cross-layer equalization → AdaRound and keeps the best by our metric. BN-folding and CLE have nothing to act on (no BN, no conv chains). Scheme selection picked min-max ("tf"). **AdaRound** (learned rounding of the dense-layer weights) added **+2.7 / +3.8 / +1.2 dB** over plain W8A16.

**Q9. What are the "range priors" and why were they needed?**
Calibration sets each tensor's range from the data it sees, and real data later exceeded those ranges, so values were clipped. Two fixes with physical or mathematical justification:
(a) LSTM hidden outputs are bounded by construction, h = o·tanh(c) ∈ (−1, 1), so the range is fixed to [−1, 1].
(b) Model outputs get the calibrated range ± 10 % (forecasts under- and overshoot, e.g. slightly negative solar at night).
For the 1 h model this lifted activation-only fidelity from 43.8 to 77.5 dB, and W8A16 from 42.4 to 47.6 dB.

**Q10. Why measure fidelity (SQNR vs FP32) instead of accuracy?**
Quantization should change the model as little as possible. SQNR of quantized vs FP32 outputs on identical inputs measures exactly that. It's independent of ground-truth quality and of the inherent model error. Accuracy (R², MAE in Watts) and anomaly-decision agreement are reported alongside it.

**Q11. What was used for calibration?**
1,024 training-period windows, stratified over hour of day, so night zeros, dawn ramps and noon peaks are all represented. Validation (selection) and test (reporting) windows are never used for calibration.

**Q12. Were the LSTM weights really 8-bit?**
Yes, but it had to be enforced. The exported graph reorders the LSTM gate weights through Slice/Concat, so AIMET initially treated them as *activations* (16-bit). They're explicitly configured as 8-bit symmetric parameters. The compiled device graph confirms it: LSTM input/recurrent weights are 8-bit (asymmetric, quantized by the converter after it folds the reordering), dense weights are 8-bit symmetric, all 3,853 activations are 16-bit, LSTM biases are 16-bit (they become Add operands when the LSTM is expanded), and dense biases are 32-bit.

**Q13. Per-channel or per-tensor weights?**
Per-tensor. Per-channel dense-layer weights simulated +1.5–3 dB better, but produced wrong results on the NPU in this deployment path (14.6 dB vs 46.2 dB for the same model). AdaRound was therefore run for per-tensor scales, so its learned rounding matches what's deployed.

---

## C. Conversion and deployment

**Q14. How does the AIMET result get onto the NPU?**
AIMET exports `model.onnx` (AdaRound-adjusted weights) plus `model.encodings` (scales/offsets). `qnn-onnx-converter` takes them as `--quantization_overrides`, calibrates any tensors without an override from an input list, and emits a QNN model. `qnn-model-lib-generator` compiles that into an Android shared library, which `qnn-net-run` (or an app through the QNN API) executes with the HTP backend.

**Q15. Which encodings are passed to the converter, and why not all of them?**
Only the input, ReLU and output activations, plus the dense-layer weight encodings. Encodings on constant subgraphs (gate-reordered LSTM weights, zero initial states) are dropped, because the converter folds those into static parameters and quantizes them itself (8-bit weights, 32-bit bias). The LSTM-hidden priors are also dropped for the converter: in the expanded graph, per-step views alias that tensor, and overriding it caused a scale mismatch (values ×1.44 on device).

**Q16. What does `--expand_lstm_op_structure` do and why is it used?**
It rewrites each LSTM time step into MatMul, Add, Sigmoid, Tanh and Mul ops, standard HTP operations that all receive 16-bit activation encodings. The fused quantized LSTM op failed HTP graph validation for this configuration. Cost: a larger graph (≈3,850 activation tensors), still 3.4 ms.

**Q17. Why `--preserve_io layout`?**
By default the converter re-lays the `[1, 96, F]` sequence input as `[1, F, 96]`. Keeping the ONNX layout means apps can pass the same array they would give the ONNX model.

**Q18. How fast is it, and is that fast enough?**
3.4 ms (1 h, 24 h) and 3.9 ms (single node) average NPU execution, ≈5.1–5.5 ms through the QNN API, ≈165–170 inferences/s including file I/O. Data arrives every 15 minutes, so latency is irrelevant for this application. Energy per inference is what benefits.

**Q19. How big is it?**
Quantized weights: 153 KB (1 h), 297 KB (24 h), 71 KB (single), ≈3.8× smaller than FP32. The compiled libraries are ~3.8–4.0 MB because they contain the graph-construction code for 96 unrolled steps.

---

## D. Verification

**Q20. How do you know the deployed model is correct?**
A ladder of checks, each isolating one possible error source:
(1) the rebuilt data reproduces the reported FP32 accuracy;
(2) AIMET's instrumented graph equals the original (1.2e-7);
(3) quantized simulation ≥ 30 dB;
(4) the converted FP32 graph on the device CPU is **bit-exact** to ONNX;
(5) quantized NPU outputs agree with the AIMET simulation (45–55 dB);
(6) AIMET beats the converter's own calibration on the NPU.
All on thousands of held-out windows.

**Q21. How much accuracy is lost?**
On the NPU vs FP32: R² +0.0007 (1 h), −0.0039 (24 h), −0.0001 (single). Worst-case output error 0.45 % / 1.26 % / 0.45 % of capacity. Anomaly decisions identical for ≥ 99.66 % of readings and ≥ 99.92 % of injected faults.

**Q22. Did AIMET actually matter, or would the converter alone do?**
Same bitwidths, converter and calibration windows. Converter-only W8A16 on the NPU: 44.9 / 43.8 / 46.3 dB. AIMET: 48.4 / 46.4 / 52.1 dB. The worst-case error on the 1 h model drops from 2.6 % to 0.45 % of capacity.

**Q23. Why did W8A8 look fine in simulation but fail on the NPU?**
AIMET-ONNX simulates the LSTM as a single op: it quantizes its inputs, weights and outputs but computes the gates and cell state in float. On the NPU, all 96 unrolled steps run with 8-bit internals, and the error compounds. This is why on-device verification is part of the process and not optional.

**Q24. The reported accuracy differs slightly from the original project report. Why?**
The original normalized dataset wasn't available, so it was reconstructed from raw telemetry. The reconstruction reproduces the timeline, splits and grid/demand capacities exactly, and solar capacity within 1.9 %. FP32 R² is within 0.007–0.03 of the reported values. One reported figure (24 h Grid R² 0.741) is inconsistent with its own MAE and equals a baseline's R², so it's likely a reporting error.

---

## E. Practical

**Q25. What's needed to run it?**
A QIDK (or another SM8650 device) over ADB, the QAIRT 2.47 runtime files in `/data/local/tmp/nz_aimet/{bin,lib}` (provided in `release/qidk_runtime_2.47`), and the model library. The host-side runner `scripts/12_qidk_forecast.py` needs only numpy and adb.

**Q26. Can it run in an Android app instead of `qnn-net-run`?**
Yes. Load the model library and `libQnnHtp.so` through the QNN C API (or wrap them in JNI), create the graph, and execute with float32 input buffers. The QIDK repository's QNN sample apps show the pattern. For faster start-up, the graph can be serialized once into a context binary.

**Q27. How would you rebuild for a different Snapdragon chip?**
Re-run `08_qairt_build.sh` unchanged (the model library is chip-independent) and push the runtime's matching `libQnnHtpV<arch>Stub/Skel.so`. Optionally use the matching AIMET HTP config (`htp_v73`, `htp_v79`, …) in `05_aimet_quantize.py`.

**Q28. What would improve it further?**
Per-channel dense-layer weights once supported in the deployment path (+1.5–3 dB in simulation), a context binary for start-up time, and plant-level forecasters for localizing and detecting panel degradation (requires training).

---

## F. Demo dashboard

**Q29. What runs where in the dashboard?**
The QIDK runs everything: a native C++ web server (`nz_dashboard`) serves the page and data, and executes live inference requests with `qnn-net-run` on the Hexagon NPU. The laptop browser only displays, connected over USB via `adb forward` (or Wi-Fi). The replay views use the NPU outputs the QIDK recorded for every test window. "Run on NPU" computes a new forecast on the device at that moment.

**Q30. Why does a live run take ~2 s when the NPU needs ~5 ms?**
Each request starts `qnn-net-run`, which loads the model library and prepares the HTP graph (0.9–1.5 s) before executing it (≈5 ms). Both are shown separately in the UI. A persistent QNN context inside the server would bring the end-to-end time to milliseconds. For a demo that wasn't worth the extra integration work.

**Q31. Can I test it with my own sensor data?**
Yes. Download a test window as a CSV template, edit it, or prepare your own file: 96 rows at 15-minute steps with a `timestamp` column and one column per sensor feature (names as in the template), in physical units. Optionally add a 97th row with the reading that actually arrived, which is then checked for anomalies. The browser normalizes the values with the training min/max, recomputes the time features from the timestamps, and sends the window to the NPU.

**Q32. Is the anomaly logic in the browser the same as in the evaluation?**
Yes: same per-hour median and limits (exported from `anomaly.py`), same residual definition, same target definitions for uploaded data. Demand verdicts are shown as "n/a" because that forecast head isn't reliable enough for anomaly detection.

**Q33. How was the dashboard verified?**
Headless Chromium against the live server on the QIDK: all tabs render without console errors. A live run from the browser matches the recorded device output (≤ 6×10⁻⁷). Injected faults are flagged. A CSV template → upload → NPU round trip reproduces the model input to 1.5×10⁻⁸.
