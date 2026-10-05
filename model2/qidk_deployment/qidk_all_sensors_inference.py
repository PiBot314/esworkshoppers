#!/usr/bin/env python3
"""
qidk_all_sensors_inference.py
Runs campus-wide Net-Zero inference on Qualcomm QIDK using ONNX Runtime / QNN Execution Provider.
"""
import os
import json
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, "all_sensors_qidk_meta.json")) as f:
    meta = json.load(f)

# Initialize Session
available_providers = ort.get_available_providers()
providers = ["QNNExecutionProvider", "CPUExecutionProvider"] if "QNNExecutionProvider" in available_providers else ["CPUExecutionProvider"]
model_path = os.path.join(SCRIPT_DIR, "all_sensors_lstm_qidk.onnx")
session = ort.InferenceSession(model_path, providers=providers)

# Load sample 24-hour multi-sensor payload [1, 96, 58]
input_path = os.path.join(SCRIPT_DIR, "sample_input_all_sensors_1x96x58.npy")
input_data = np.load(input_path).astype(np.float32)

# Run Inference
raw_output = session.run([meta["output_tensor_name"]], {meta["input_tensor_name"]: input_data})[0]
# Shape: [1, 12] -> Reshape to [4 time intervals, 3 domains: Solar, Grid, Demand]
preds = raw_output[0].reshape(4, 3)

caps = meta["capacities_W"]
solar_cap = caps["solar_max_W"]
grid_cap = caps["grid_max_W"]
demand_cap = caps["demand_max_W"]

import datetime

# Reference timestamp: 09:30 AM IST (Forecast steps: +15m -> 09:45 AM, +30m -> 10:00 AM, etc.)
base_time = datetime.datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)

print("=" * 100)
print("  SHORT-TERM (1-HOUR) REAL-TIME DISPATCH")
print("=" * 100)
print(f" Timestep  | Clock Time (IST) |  Solar Yield (W) |  Campus Grid (W) |  Room Demand (W) | Net Balance (W) | Net-Zero Status")
print("-" * 100)
for i in range(4):
    s_w = float(max(0.0, preds[i, 0] * solar_cap))
    g_w = float(max(0.0, preds[i, 1] * grid_cap))
    d_w = float(max(0.0, preds[i, 2] * demand_cap))
    net_deficit = d_w - s_w
    if net_deficit <= 0:
        status = f"SURPLUS (+{-net_deficit:,.0f} W export)"
    else:
        status = f"DEFICIT ({net_deficit:,.0f} W grid)"
    step_min = (i + 1) * 15
    t_slot = base_time + datetime.timedelta(minutes=step_min)
    time_str = t_slot.strftime("%I:%M %p")
    print(f"  +{step_min:02d} min |     {time_str:<8}    | {s_w:14,.1f} W | {g_w:14,.1f} W | {d_w:14,.1f} W | {net_deficit:13,.1f} W | {status}")
print("=" * 100)

