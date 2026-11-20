#!/usr/bin/env python3
"""
qidk_edge_inference.py
Runs real-time inference on the Qualcomm QIDK board using ONNX Runtime / QNN Execution Provider.
"""
import numpy as np
import onnxruntime as ort
import json

# Load metadata
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "qidk_model_meta.json")) as f:
    meta = json.load(f)

# Initialize Session (Falls back to CPU if QNN Execution Provider is unconfigured)
available_providers = ort.get_available_providers()
print(f"Available Execution Providers: {available_providers}")

# On QIDK: use QNNExecutionProvider for Hexagon NPU acceleration
providers = ["QNNExecutionProvider", "CPUExecutionProvider"] if "QNNExecutionProvider" in available_providers else ["CPUExecutionProvider"]
session = ort.InferenceSession(os.path.join(SCRIPT_DIR, "SL-PL00-00_lstm_qidk.onnx"), providers=providers)

# Load sample 24-hour normalized window
input_data = np.load(os.path.join(SCRIPT_DIR, "sample_input_1x96.npy")).astype(np.float32)

# Run Inference
outputs = session.run([meta["output_tensor_name"]], {meta["input_tensor_name"]: input_data})[0]

# Unscale to real Watts
preds_watts = outputs[0] * (meta["scaler_max_W"] - meta["scaler_min_W"]) + meta["scaler_min_W"]

print("\n=== INFERENCE RESULT ===")
print(f"Predicted Power for next 4 intervals (+15m, +30m, +45m, +60m):")
for i, w in enumerate(preds_watts, 1):
    print(f"  T + {i*15} min: {w:.2f} Watts")
