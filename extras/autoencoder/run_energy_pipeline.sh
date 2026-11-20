#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 campus_energy_pipeline/data_pipeline.py
python3 campus_energy_pipeline/train_autoencoder.py
python3 campus_energy_pipeline/qnn_flow.py

echo "Pipeline complete. Launch dashboard with: streamlit run campus_energy_pipeline/dashboard.py"
