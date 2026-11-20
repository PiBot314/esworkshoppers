<<<<<<< HEAD
# esworkingshoppers
=======
# Campus Energy Monitoring on QIDK

This project implements a lightweight edge-native anomaly detection pipeline for campus electrical load monitoring using the provided `EM-Kh95-00.csv` dataset. The workflow is designed for deployment on a Qualcomm Innovator Development Kit (QIDK) and follows the midterm evaluation flow:

- ingest and clean raw time-series data from CSV
- standardize sampling with 15-minute mean/max aggregation
- train a PyTorch autoencoder to detect abnormal demand spikes
- export the model to ONNX and prepare calibration samples for quantization
- describe the SNPE/QNN `.dlc` compilation path for Hexagon NPU execution
- expose a lightweight Streamlit dashboard and anomaly narrative generator

## Quick start

1. Install dependencies with uv:
   ```bash
   uv sync
   ```
2. Generate the processed 15-minute dataset:
   ```bash
   uv run python campus_energy_pipeline/data_pipeline.py
   ```
3. Train the autoencoder and export ONNX:
   ```bash
   uv run python campus_energy_pipeline/train_autoencoder.py
   ```
4. Launch the dashboard from the project root:
   ```bash
   uv run streamlit run campus_energy_pipeline/dashboard.py
   ```
5. Open the dashboard and inspect anomaly score trends and remediation prompts.

## Output artifacts

- `artifacts/energy_15m.csv` — resampled time-series dataset
- `artifacts/energy_15m_summary.csv` — per-bucket summary stats
- `artifacts/energy_calibration.npy` — representative inputs for PTQ calibration
- `artifacts/models/energy_autoencoder.onnx` — exported ONNX model
- `artifacts/models/energy_autoencoder_threshold.json` — anomaly threshold and metadata
- `artifacts/energy_anomalies.csv` — flagged anomaly events

## QIDK deployment notes

The QNN/SNPE deployment path is prepared in `campus_energy_pipeline/qnn_flow.py`. On a QIDK machine with QAIRT/SNPE installed, use the generated `.dlc` and `snpe-net-run` commands described in that script and in the report.

## Report

See [report.md](report.md) for the structured analysis and hardware deployment plan.
>>>>>>> 1e78acd (evil big commit)
