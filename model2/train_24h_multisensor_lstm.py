"""
train_24h_multisensor_lstm.py
-----------------------------
Campus-Wide Net-Zero Day-Ahead (+24 Hours) Forecaster for Qualcomm QIDK.

Key Concept & Physics:
1. 24-Hour Ahead Forecasting (Horizon H = 96 steps):
   Essential for day-ahead battery scheduling, HVAC pre-cooling, and campus grid dispatch.
   Input: Past 24 hours (L = 96 steps, 58 multi-sensor features).
   Output: Full next 24 hours (H = 96 steps x 3 domains: Solar, Grid, Demand = 288 predictions).

2. Correct Net-Zero Energy Balance Physics:
   Supply = Solar Generation + Grid Import
   Total Load = Campus Demand
   Net-Zero Balance (W) = Campus Demand (W) - Solar Generation (W)
     * Net Balance > 0 : Energy Deficit (Requires Grid import / Battery discharge)
     * Net Balance = 0 : True Net-Zero Equilibrium (100% self-sufficient)
     * Net Balance < 0 : Green Energy Surplus (Excess solar exported to grid or charged to battery)
   Solar Self-Sufficiency (%) = (Solar Generation / Campus Demand) * 100%
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import onnx
import onnxruntime as ort

DATA_DIR = "lstm_normalized"
OUTPUT_DIR = "qidk_deployment"
LOOKBACK = 96       # 24 hours history (96 x 15-min)
HORIZON = 96        # Full next 24 hours forecast (96 x 15-min)
DOMAINS = 3         # Solar, Grid, Demand
BATCH_SIZE = 64
EPOCHS = 10
HIDDEN_DIM = 128
NUM_LAYERS = 2
LR = 0.002
DEVICE = torch.device("cpu")


def load_and_merge_all_sensors():
    print("[1] Loading & merging all 11 sensor nodes across the campus...")
    with open(os.path.join(DATA_DIR, "features.json")) as f:
        meta = json.load(f)["nodes"]
    with open(os.path.join(DATA_DIR, "scaler_params.json")) as f:
        scalers = json.load(f)

    dfs = {}
    for node_id in meta.keys():
        dfs[node_id] = pd.read_parquet(os.path.join(DATA_DIR, f"{node_id}.parquet"))

    # Intersection of indices
    common_idx = dfs["SL-PL00-00"].index
    for node_id, df in dfs.items():
        common_idx = common_idx.intersection(df.index)

    feature_dict = {}
    for node_id, df in dfs.items():
        sub_df = df.loc[common_idx]
        for c in sub_df.columns:
            if c in ["hod_sin", "hod_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"]:
                continue
            feature_dict[f"{node_id}__{c}"] = sub_df[c].values

    pl_sub = dfs["SL-PL00-00"].loc[common_idx]
    for c in ["hod_sin", "hod_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"]:
        feature_dict[c] = pl_sub[c].values

    X_all_df = pd.DataFrame(feature_dict, index=common_idx)
    num_features = X_all_df.shape[1]

    # Compute Physical Totals in Watts
    solar_nodes = ["SL-PL00-00", "SL-VN95-00", "SL-NI01-00", "SL-NI03-00", "SL-NI03-01", "SL-VN02-01"]
    grid_nodes = ["EM-CR-SB00-00", "EM-CR-KH95-00"]
    room_nodes = ["SR-EM-KH04-00", "SR-EM-KH04-01", "SR-EM-KH04-02"]

    solar_watts = np.zeros(len(common_idx))
    for sn in solar_nodes:
        s_min = scalers[sn]["active_power_W"]["min"]
        s_max = scalers[sn]["active_power_W"]["max"]
        solar_watts += dfs[sn].loc[common_idx, "active_power_W"].values * (s_max - s_min) + s_min

    grid_watts = np.zeros(len(common_idx))
    g_sb_min = scalers["EM-CR-SB00-00"]["gd_power_W"]["min"]
    g_sb_max = scalers["EM-CR-SB00-00"]["gd_power_W"]["max"]
    grid_watts += dfs["EM-CR-SB00-00"].loc[common_idx, "gd_power_W"].values * (g_sb_max - g_sb_min) + g_sb_min

    g_kh_min = scalers["EM-CR-KH95-00"]["power_total_W"]["min"]
    g_kh_max = scalers["EM-CR-KH95-00"]["power_total_W"]["max"]
    grid_watts += dfs["EM-CR-KH95-00"].loc[common_idx, "power_total_W"].values * (g_kh_max - g_kh_min) + g_kh_min

    demand_watts = np.zeros(len(common_idx))
    for rn in room_nodes:
        r_min = scalers[rn]["power_W"]["min"]
        r_max = scalers[rn]["power_W"]["max"]
        demand_watts += dfs[rn].loc[common_idx, "power_W"].values * (r_max - r_min) + r_min

    solar_max = float(solar_watts.max())
    grid_max = float(grid_watts.max())
    demand_max = float(demand_watts.max())

    targets_norm = np.stack([
        solar_watts / solar_max,
        grid_watts / grid_max,
        demand_watts / demand_max
    ], axis=1) # shape (N, 3)

    target_scalers = {
        "solar_max_W": solar_max,
        "grid_max_W": grid_max,
        "demand_max_W": demand_max
    }

    print(f"    -> Timestamps: {len(common_idx):,} | Features: {num_features}")
    print(f"    -> Solar Peak:  {solar_max:,.1f} W ({solar_max/1000:,.1f} kW)")
    print(f"    -> Grid Peak:   {grid_max:,.1f} W ({grid_max/1000:,.1f} kW)")
    print(f"    -> Demand Peak: {demand_max:,.1f} W ({demand_max/1000:,.1f} kW)")

    return X_all_df, targets_norm, target_scalers


def create_24h_windows(X_data, y_data, lookback=96, horizon=96):
    X, y = [], []
    for i in range(len(X_data) - lookback - horizon + 1):
        X.append(X_data[i : i + lookback])
        # Flatten horizon: 96 steps * 3 domains = 288 outputs
        y.append(y_data[i + lookback : i + lookback + horizon, :].flatten())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class NetZero24hLSTM(nn.Module):
    """
    Day-Ahead 24h Multi-Sensor LSTM for Qualcomm QIDK.
    Input:  [Batch, 96, 58] (24h past multi-sensor context)
    Output: [Batch, 288] (96 timesteps x 3 pillars: Solar, Grid, Demand)
    """
    def __init__(self, input_dim=58, hidden_dim=128, num_layers=2, output_dim=288):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = out[:, -1, :] # [Batch, hidden_dim]
        h = self.relu(self.fc1(last_step))
        preds = self.fc2(h)      # [Batch, 288]
        return preds


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print(" DAY-AHEAD (+24 HOURS) MULTI-SENSOR LSTM FOR NET-ZERO BUILDINGS")
    print("=" * 70)

    # 1. Load Data
    X_df, y_targets, target_scalers = load_and_merge_all_sensors()
    feature_names = list(X_df.columns)
    input_dim = len(feature_names)

    # 2. Chronological Split (70% Train, 15% Val, 15% Test)
    n = len(X_df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    X_train_df = X_df.iloc[:n_train]
    y_train_arr = y_targets[:n_train]

    X_val_df = X_df.iloc[n_train : n_train + n_val]
    y_val_arr = y_targets[n_train : n_train + n_val]

    X_test_df = X_df.iloc[n_train + n_val :]
    y_test_arr = y_targets[n_train + n_val :]

    X_train, y_train = create_24h_windows(X_train_df.values, y_train_arr, LOOKBACK, HORIZON)
    X_val, y_val = create_24h_windows(X_val_df.values, y_val_arr, LOOKBACK, HORIZON)
    X_test, y_test = create_24h_windows(X_test_df.values, y_test_arr, LOOKBACK, HORIZON)

    print(f"\n[2] 24-Hour Day-Ahead Windows Created:")
    print(f"    Train: {X_train.shape} -> Targets: {y_train.shape}")
    print(f"    Val:   {X_val.shape}   -> Targets: {y_val.shape}")
    print(f"    Test:  {X_test.shape}  -> Targets: {y_test.shape}")

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE, shuffle=False)

    # 3. Model Training
    model = NetZero24hLSTM(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=HORIZON * DOMAINS).to(DEVICE)
    best_weights_path = os.path.join(OUTPUT_DIR, "best_24h_all_sensors_lstm.pt")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    print(f"\n[3] Training 24-Hour Forecaster on CPU ({EPOCHS} Epochs)...")
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                val_loss += criterion(model(bx), by).item() * len(bx)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_weights_path)
            mark = "(*) BEST"
        else:
            mark = ""

        print(f"    Epoch [{epoch:02d}/{EPOCHS}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} {mark}")

    # 4. Evaluation in Real-World Watts over the 24-Hour Horizon
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()

    with torch.no_grad():
        test_preds_norm = model(torch.tensor(X_test).to(DEVICE)).numpy()

    # Reshape: (N, 96 steps, 3 domains)
    preds_reshaped = test_preds_norm.reshape(-1, HORIZON, DOMAINS)
    y_test_reshaped = y_test.reshape(-1, HORIZON, DOMAINS)

    domains = ["Solar", "Grid", "Demand"]
    capacities = [target_scalers["solar_max_W"], target_scalers["grid_max_W"], target_scalers["demand_max_W"]]

    print(f"\n[4] Full 24-Hour Horizon Evaluation Metrics (Test Set):")
    print(" " + "-"*75)
    print(f" {'Domain':<12s} | {'Capacity':<14s} | {'MAE (Watts)':<12s} | {'RMSE (Watts)':<12s} | {'nMAE (%)':<9s} | {'R² Score':<8s}")
    print(" " + "-"*75)

    evaluation_summary = {}
    for idx, (dom, cap) in enumerate(zip(domains, capacities)):
        pred_w = preds_reshaped[:, :, idx] * cap
        true_w = y_test_reshaped[:, :, idx] * cap

        mae = float(np.mean(np.abs(pred_w - true_w)))
        rmse = float(np.sqrt(np.mean((pred_w - true_w) ** 2)))
        nmae = float(mae / cap * 100.0)
        ss_tot = np.sum((true_w - np.mean(true_w)) ** 2)
        ss_res = np.sum((true_w - pred_w) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot))

        evaluation_summary[dom] = {"MAE_W": mae, "RMSE_W": rmse, "nMAE_pct": nmae, "R2": r2}
        print(f" {dom:<12s} | {cap:11,.1f} W | {mae:10.1f} W | {rmse:10.1f} W | {nmae:7.2f}% | {r2:7.4f}")

    print(" " + "-"*75)

    # 5. Export to Static-Shape ONNX for Qualcomm QIDK
    onnx_path = os.path.join(OUTPUT_DIR, "all_sensors_24h_lstm_qidk.onnx")
    dummy_input = torch.randn(1, LOOKBACK, input_dim, dtype=torch.float32, device=DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_sequence_all_sensors_24h"],
        output_names=["forecast_all_pillars_24h"],
    )

    onnx_model = onnx.load(onnx_path, load_external_data=True)
    onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)
    data_ext = onnx_path + ".data"
    if os.path.exists(data_ext):
        os.remove(data_ext)

    print(f"\n[5] Exported Static-Shape 24h ONNX Model -> {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")

    # 6. Verify Numerical Parity via ONNX Runtime
    ort_session = ort.InferenceSession(onnx_path)
    sample_input = X_test[:1] # shape: (1, 96, 58)

    with torch.no_grad():
        pt_out = model(torch.tensor(sample_input).to(DEVICE)).numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: sample_input}
    ort_out = ort_session.run(None, ort_inputs)[0]

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    print(f"[6] ONNX vs PyTorch Numerical Parity: Max diff = {max_diff:.8e}")
    assert max_diff < 1e-4, "Numerical divergence between PyTorch and ONNX!"
    print("    -> Numerical parity confirmed!")

    # 7. Package Multi-Sensor Calibration & Metadata
    sample_path = os.path.join(OUTPUT_DIR, "sample_input_all_sensors_24h_1x96x58.npy")
    np.save(sample_path, sample_input)

    deploy_meta = {
        "model_type": "Day_Ahead_24h_All_Sensors_Campus_Forecaster",
        "input_tensor_name": "input_sequence_all_sensors_24h",
        "input_shape": [1, LOOKBACK, input_dim],
        "output_tensor_name": "forecast_all_pillars_24h",
        "output_shape": [1, HORIZON * DOMAINS],
        "lookback_hours": 24,
        "horizon_hours": 24,
        "step_interval_min": 15,
        "feature_count": input_dim,
        "feature_names": feature_names,
        "domains": domains,
        "capacities_W": target_scalers,
        "evaluation_summary": evaluation_summary,
        "physics_equations": {
            "net_energy_balance_W": "Campus_Demand_W - Solar_Generation_W",
            "solar_self_sufficiency_pct": "(Solar_Generation_W / Campus_Demand_W) * 100",
            "net_zero_condition": "Solar_Generation_W >= Campus_Demand_W (Surplus or 0 grid draw)"
        }
    }
    with open(os.path.join(OUTPUT_DIR, "all_sensors_24h_qidk_meta.json"), "w") as f:
        json.dump(deploy_meta, f, indent=2)

    # 8. Create Standalone Edge Inference Script for QIDK
    runner_script = f'''#!/usr/bin/env python3
"""
qidk_24h_netzero_inference.py
Runs Day-Ahead (+24 Hours) Net-Zero Building Inference on Qualcomm QIDK.
Implements the true Net-Zero Energy Balance Physics:
  Net Deficit (W) = Demand - Solar
  Solar Sufficiency (%) = (Solar / Demand) * 100%
"""
import os
import json
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, "all_sensors_24h_qidk_meta.json")) as f:
    meta = json.load(f)

# Initialize Session with QNN NPU Fallback
available_providers = ort.get_available_providers()
providers = ["QNNExecutionProvider", "CPUExecutionProvider"] if "QNNExecutionProvider" in available_providers else ["CPUExecutionProvider"]
model_path = os.path.join(SCRIPT_DIR, "all_sensors_24h_lstm_qidk.onnx")
session = ort.InferenceSession(model_path, providers=providers)

# Load sample 24-hour multi-sensor payload [1, 96, 58]
input_path = os.path.join(SCRIPT_DIR, "sample_input_all_sensors_24h_1x96x58.npy")
input_data = np.load(input_path).astype(np.float32)

# Run Inference
raw_output = session.run([meta["output_tensor_name"]], {{meta["input_tensor_name"]: input_data}})[0]
# Shape: [1, 288] -> Reshape to [96 time intervals, 3 domains: Solar, Grid, Demand]
preds = raw_output[0].reshape(96, 3)

caps = meta["capacities_W"]
solar_cap = caps["solar_max_W"]
grid_cap = caps["grid_max_W"]
demand_cap = caps["demand_max_W"]

print("=" * 82)
print("  QUALCOMM QIDK ON-DEVICE INFERENCE: DAY-AHEAD (+24 HOURS) NET-ZERO FORECAST")
print("=" * 82)
print(f" Hour  |  Solar Yield (W) |  Campus Grid (W) |  Room Demand (W) | Net Balance (W) | Net-Zero Status")
print("-" * 82)

# Summarize hourly (every 4 steps of 15 min = 1 hour)
for h in range(24):
    start_idx = h * 4
    end_idx = start_idx + 4
    s_w = float(np.mean(preds[start_idx:end_idx, 0]) * solar_cap)
    g_w = float(np.mean(preds[start_idx:end_idx, 1]) * grid_cap)
    d_w = float(np.mean(preds[start_idx:end_idx, 2]) * demand_cap)
    
    # Net-Zero Physics: Demand minus Solar
    # If Demand > Solar: Net Deficit (requires grid power)
    # If Solar >= Demand: Net Surplus (100% self-sufficient / exporting)
    net_deficit = d_w - s_w
    if net_deficit <= 0:
        status = f"SURPLUS ({-net_deficit:,.0f} W export)"
    else:
        status = f"DEFICIT ({net_deficit:,.0f} W import)"

    print(f" +{{h+1:02d}}h  | {{s_w:14,.1f}} W | {{g_w:14,.1f}} W | {{d_w:14,.1f}} W | {{net_deficit:13,.1f}} W | {{status}}")

print("=" * 82)
total_solar_kwh = float(np.sum(preds[:, 0] * solar_cap * 0.25) / 1000.0) # 15 min = 0.25 h
total_grid_kwh = float(np.sum(preds[:, 1] * grid_cap * 0.25) / 1000.0)
total_demand_kwh = float(np.sum(preds[:, 2] * demand_cap * 0.25) / 1000.0)

print(f"\\n24-Hour Projected Energy Totals:")
print(f"  * Total Solar Generation : {{total_solar_kwh:,.1f}} kWh")
print(f"  * Total Grid Electricity : {{total_grid_kwh:,.1f}} kWh")
print(f"  * Total Monitored Demand : {{total_demand_kwh:,.1f}} kWh")
print("=" * 82)
'''
    with open(os.path.join(OUTPUT_DIR, "qidk_24h_netzero_inference.py"), "w") as f:
        f.write(runner_script)

    print(f"\n[7] Day-Ahead 24h QIDK Deployment Package created in '{OUTPUT_DIR}/':")
    print(f"    - Model:            {onnx_path}")
    print(f"    - Sample Payload:   {sample_path}")
    print(f"    - Deployment Meta:  {os.path.join(OUTPUT_DIR, 'all_sensors_24h_qidk_meta.json')}")
    print(f"    - On-Device Runner: {os.path.join(OUTPUT_DIR, 'qidk_24h_netzero_inference.py')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
