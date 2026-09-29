# Campus Energy Monitoring Pipeline on QIDK

## 1. Problem framing

The campus building energy system is a high-value, low-latency operational environment where small power anomalies can quickly escalate into avoidable energy waste. The dataset (`EM-Kh95-00.csv`) captures electrical meter telemetry from a campus node and includes voltage, phase currents, power draw, frequency, power factor and cumulative energy.

The project uses an edge-native anomaly detection pipeline to flag abnormal windows before they result in sustained waste. In practice, a small increase in phase imbalance, a power-factor drop, or a sudden current spike can indicate equipment running unnecessarily, a breaker problem, poor maintenance, or thermostat/lighting control drift. For facilities management, the important step is not only detection, but explainability: every alert should translate into a human-readable operational recommendation.

## 2. Ingestion and quantization architecture

The implemented workflow is intentionally lightweight and suitable for QIDK class devices:

1. CSV ingestion and field cleanup
   - Raw data is read from the CSV using pandas.
   - The duplicated `Timestamp` header is resolved by selecting the ISO-8601 timestamp column and converting it to a proper datetime index.
   - Numeric sensor columns are coerced to floats and sorted by time.

2. Time-series resampling and feature engineering
   - Using a 15-minute cadence, the raw meter time series is aggregated with both mean and max statistics per feature.
   - This preserves spikes that matter for energy wastage analysis while reducing input-rate overhead for the inference model.
   - The processed samples contain the relevant 15-minute energy features such as `Voltage_mean`, `Voltage_max`, `R Power_mean`, `B Power_max`, `Power Factor_mean`, and `Total Energy_max`.

3. Normalization and autoencoder training
   - The aggregated features are standardized with `StandardScaler`.
   - A compact PyTorch autoencoder is trained on the normalized input using MSE reconstruction loss.
   - An anomaly is considered any 15-minute bucket whose reconstruction error exceeds the 99.5th percentile threshold learned on the training data.

4. ONNX export and calibration
   - The trained model is exported to ONNX for hardware-agnostic deployment.
   - A representative calibration batch is extracted from the same data distribution for quantization tuning.

5. QNN / SNPE conversion path
   - The ONNX model is converted to a Qualcomm DLC through the SNPE toolchain.
   - The result is intended for execution on the Hexagon NPU/DSP as `.dlc` with a quantized inference graph.

6. Human-readable diagnostics
   - A lightweight prompt generator maps anomaly features such as high phase power, voltage spikes, or poor power factor to operator-good instructions.
   - This behaves like a retrieval-style contextual assistant: it retrieves the most relevant facility guidance template for the observed anomaly signature.

## 3. Observed model behavior

The current implementation generated a clean 15-minute energy feature set and trained an autoencoder successfully on the dataset. The observed result on this local run was:

- Aggregated rows: 34,036
- Anomaly threshold: 14.794133
- Detected anomalies: 170

This is a useful starting point for campus monitoring because the model is sensitive to sharp non-stationary events while remaining compact enough for an edge deployment.

## 4. QNN / SNPE deployment plan and expected benchmarks

The repository includes the command path for QNN/SNPE conversion in `campus_energy_pipeline/qnn_flow.py`:

```bash
snpe-onnx-to-dlc --input_network artifacts/models/energy_autoencoder.onnx \
  --output_path artifacts/models/energy_autoencoder.dlc \
  --input_shape "input[1,20]"
```

Then, on-device benchmarking is performed with ADB and `snpe-net-run` using the calibration set:

```bash
adb shell
snpe-net-run --container /data/local/tmp/energy_autoencoder.dlc \
  --input_list /data/local/tmp/energy_calibration.npy \
  --output_dir /data/local/tmp
```

Expected benchmark summary for this class of model:

| Metric | FP32 target | INT8 target | Notes |
| --- | --- | --- | --- |
| Inference latency | ~1-3 ms | <1 ms | Best-case on Hexagon NPU with a compact 20-feature input |
| Memory footprint | low | lower | INT8 usually reduces activation and weight memory |
| Reconstruction accuracy | baseline | slight degradation | Small accuracy loss acceptable for anomaly thresholding |

Important note: this environment does not include the physical QIDK or the QAIRT SNPE runtime, so the actual device-side latency and memory numbers remain target metrics rather than measured values. The workflow is structured so the same scripts can be run directly on the target device when QAIRT is available.

## 5. Observation, limitations, and expansion options

### Observations

- The dataset is not a clean synthetic benchmark; it includes normal operations, long gaps, and occasional outliers that are useful for anomaly detection.
- The 15-minute aggregation strategy is sufficient for campus facilities monitoring because it reduces data volume while preserving edge conditions like voltage spikes and load surges.
- The autoencoder is compact and easy to export; it is a good fit for low-power inference on edge hardware.

### Limitations

- The baseline workflow uses post-training quantization (PTQ), which is easy to adopt but can be suboptimal for highly sensitive threshold-based anomaly scoring.
- A single-node model does not yet capture cross-building or cross-campus correlations.
- The engineering prompt generator is retrieval-based rather than a full generative RAG stack; it is adequate for a midterm demo but not a full facilities assistant.

### Expansions

- Multi-node monitoring: build one model per building or meter and aggregate their anomaly scores in a campus-level dashboard.
- Drift-aware thresholds: update the anomaly threshold after seasonal load changes or holidays.
- Explainability: attach root-cause labels to recurrent patterns (e.g., HVAC compressor drift, phase imbalance, lighting schedule leak).
- LLM/RAG layer: use a proper vector store with building manuals and maintenance SOPs for richer operational recommendations.

## 6. Is the current workflow sufficient?

For a midterm evaluation and a lightweight field deployment, yes. The current pipeline is sufficient for a proof-of-concept campus energy anomaly monitor because it:

- ingests sensor data cleanly,
- reduces storage and compute with 15-minute aggregation,
- trains a compact reconstruction model,
- exports the model to ONNX,
- provides a QNN/SNPE conversion path for QIDK deployment, and
- translates anomalies into action-oriented messages for operators.

However, for production-grade quantized inference and more aggressive NPU efficiency, the recommended next step is quantization-aware training (QAT) or AIMET-based optimization. These methods usually preserve model accuracy better than plain PTQ when thresholds are tight and small sensor deviations matter. In short:

- PTQ is ideal for the first deployment and faster turnaround.
- QAT / AIMET is better when the model is close to the operating boundary and the system must preserve anomaly sensitivity after int8 conversion.

## 7. Future campus-node extension

A scalable campus rollout would treat each building meter as a node with its own autoencoder model. Each node generates a local anomaly score, and a campus coordinator combines node-level information with building occupancy and weather data. This preserves privacy and latency while enabling a broader campus-level energy optimization strategy. The same dashboard can then summarize the whole site while drilling into individual nodes for action.

## 8. Code and artifacts

The generated project artifacts are:

- `artifacts/energy_15m.csv`
- `artifacts/energy_anomalies.csv`
- `artifacts/energy_calibration.npy`
- `artifacts/models/energy_autoencoder.onnx`
- `artifacts/models/energy_autoencoder_threshold.json`

These support the development and QIDK handoff workflow for the campus energy monitoring system.
