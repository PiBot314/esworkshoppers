#!/usr/bin/env python3
"""
qidk_24h_netzero_inference.py
-----------------------------
Runs Day-Ahead (+24 Hours) Net-Zero Building Inference on Qualcomm QIDK.

Net-Zero Energy Balance Physics:
  Net Deficit (W) = Campus Demand (W) - Solar Generation (W)
  Solar Self-Sufficiency (%) = (Solar Generation / Campus Demand) * 100%

  * When Solar >= Demand: NET-ZERO SURPLUS (Excess clean energy exports to grid / charges batteries).
  * When Solar < Demand : NET DEFICIT (Campus must import residual power from the grid).
"""

import os
import json
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Load Model Metadata & Calibration
with open(os.path.join(SCRIPT_DIR, "all_sensors_24h_qidk_meta.json")) as f:
    meta = json.load(f)

# 2. Initialize ONNX Runtime Session (with Qualcomm QNN Execution Provider fallback)
available_providers = ort.get_available_providers()
providers = ["QNNExecutionProvider", "CPUExecutionProvider"] if "QNNExecutionProvider" in available_providers else ["CPUExecutionProvider"]
model_path = os.path.join(SCRIPT_DIR, "all_sensors_24h_lstm_qidk.onnx")
session = ort.InferenceSession(model_path, providers=providers)

# 3. Load Sample 24-hour Multi-Sensor Payload [1, 96, 58]
# If specific payload file exists, load it, otherwise use sample_input_all_sensors_1x96x58.npy
sample_file = os.path.join(SCRIPT_DIR, "sample_input_all_sensors_1x96x58.npy")
input_data = np.load(sample_file).astype(np.float32)

# 4. Run Inference on Edge Hardware
raw_output = session.run([meta["output_tensor_name"]], {meta["input_tensor_name"]: input_data})[0]

# Shape: [1, 288] -> Reshape to [96 time intervals of 15-min, 3 domains: Solar, Grid, Demand]
preds = raw_output[0].reshape(96, 3)

caps = meta["capacities_W"]
solar_cap = caps["solar_max_W"]
grid_cap = caps["grid_max_W"]
demand_cap = caps["demand_max_W"]

import datetime

# Reference timestamp for this sample sequence: 08:30 AM IST (Forecast steps: 09:30 AM to 08:30 AM next day)
base_time = datetime.datetime.now().replace(hour=8, minute=30, second=0, microsecond=0)

print("=" * 100)
print("   DAY-AHEAD (+24 HOURS) NET-ZERO FORECAST")
print("=" * 100)
print(f" Hour  | Clock Time (IST) |  Solar Yield (W) |  Campus Grid (W) |  Room Demand (W) | Net Balance (W) | Net-Zero Status")
print("-" * 100)

hourly_solar = []
hourly_grid = []
hourly_demand = []

# Summarize hourly (4 steps of 15 min = 1 hour)
for h in range(24):
    start_idx = h * 4
    end_idx = start_idx + 4
    t_slot = base_time + datetime.timedelta(hours=h+1)
    time_str = t_slot.strftime("%I:%M %p")

    # Physics rule: At night (sunset ~6:30 PM to sunrise ~6:00 AM), solar irradiance is strictly 0.0 W
    if t_slot.hour >= 19 or t_slot.hour < 6:
        s_w = 0.0
    else:
        s_w = float(max(0.0, np.mean(preds[start_idx:end_idx, 0]) * solar_cap))

    g_w = float(max(0.0, np.mean(preds[start_idx:end_idx, 1]) * grid_cap))
    d_w = float(max(0.0, np.mean(preds[start_idx:end_idx, 2]) * demand_cap))

    hourly_solar.append(s_w)
    hourly_grid.append(g_w)
    hourly_demand.append(d_w)

    # Net-Zero Physics: Demand minus Solar
    net_deficit = d_w - s_w
    if net_deficit <= 0:
        surplus_w = -net_deficit
        status = f"SURPLUS (+{surplus_w:,.0f} W export)"
    else:
        status = f"DEFICIT ({net_deficit:,.0f} W grid)"

    print(f" +{h+1:02d}h  |     {time_str:<8}    | {s_w:14,.1f} W | {g_w:14,.1f} W | {d_w:14,.1f} W | {net_deficit:13,.1f} W | {status}")

print("=" * 100)

# Compute 24-Hour Projected Cumulative Energy
total_solar_kwh = float(np.sum(np.array(hourly_solar) * 1.0) / 1000.0)
total_grid_kwh = float(np.sum(np.array(hourly_grid) * 1.0) / 1000.0)
total_demand_kwh = float(np.sum(np.array(hourly_demand) * 1.0) / 1000.0)
net_day_deficit_kwh = total_demand_kwh - total_solar_kwh

print(f"\n24-Hour Projected Energy Balance:")
print(f"  * Total Solar Generation : {total_solar_kwh:,.1f} kWh")
print(f"  * Total Grid Electricity : {total_grid_kwh:,.1f} kWh")
print(f"  * Total Monitored Demand : {total_demand_kwh:,.1f} kWh")
print(f"  * 24h Net-Zero Status    : {'SURPLUS (-%.1f kWh)' % abs(net_day_deficit_kwh) if net_day_deficit_kwh <= 0 else 'DEFICIT (+%.1f kWh required from grid)' % net_day_deficit_kwh}")
print("=" * 100)
