"""
train_all_sensors_lstm.py
-------------------------
Unified Multi-Sensor LSTM Architecture for Campus-Wide Net-Zero Management.
Trains on the synchronized, continuous telemetry of ALL 11 SENSORS simultaneously:
  - 6 Solar Plants / Inverters (SL-PL00-00, SL-VN95-00, SL-NI01-00, SL-NI03-00, SL-NI03-01, SL-VN02-01)
  - 2 Substation / Grid Incomers (EM-CR-SB00-00, EM-CR-KH95-00)
  - 3 Smart Room / Sub-Meters (SR-EM-KH04-00, SR-EM-KH04-01, SR-EM-KH04-02)

Total Synchronized Input Features: 58 electrical dimensions + IST cyclic time encodings.
Forecasts 4-step ahead (1 hour) Net-Zero pillars simultaneously:
  1. Campus Total Solar Generation (Watts)
  2. Campus Total Grid Draw (Watts)
  3. Campus Total Room Demand (Watts)

Exports a static-shape ONNX model for Qualcomm QIDK / Hexagon NPU deployment.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import onnx
import onnxruntime as ort

DATA_DIR = "lstm_normalized"
OUTPUT_DIR = "qidk_deployment"
LOOKBACK = 96     # 24 hours of 15-min history
HORIZON = 4       # 1 hour forecast (+15m, +30m, +45m, +60m)
BATCH_SIZE = 64
EPOCHS = 12
HIDDEN_DIM = 96
NUM_LAYERS = 2
LR = 0.002
DEVICE = torch.device("cpu")


def load_and_merge_all_sensors():
    print("[1] Merging all 11 sensor nodes across the campus...")
    with open(os.path.join(DATA_DIR, "features.json")) as f:
        meta = json.load(f)["nodes"]
    with open(os.path.join(DATA_DIR, "scaler_params.json")) as f:
        scalers = json.load(f)

    dfs = {}
    for node_id in meta.keys():
        dfs[node_id] = pd.read_parquet(os.path.join(DATA_DIR, f"{node_id}.parquet"))

    # Find common overlapping index
    common_idx = dfs["SL-PL00-00"].index
    for node_id, df in dfs.items():
        common_idx = common_idx.intersection(df.index)

    print(f"    -> Aligned continuous timeline: {len(common_idx):,} intervals (from {common_idx.min()} to {common_idx.max()})")

    # Build multi-sensor unified feature table
    feature_dict = {}
    for node_id, df in dfs.items():
        sub_df = df.loc[common_idx]
        for c in sub_df.columns:
            if c in ["hod_sin", "hod_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"]:
                continue
            feature_dict[f"{node_id}__{c}"] = sub_df[c].values

    # Add single set of IST cyclical features
    pl_sub = dfs["SL-PL00-00"].loc[common_idx]
    for c in ["hod_sin", "hod_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"]:
        feature_dict[c] = pl_sub[c].values

    X_all_df = pd.DataFrame(feature_dict, index=common_idx)
    num_features = X_all_df.shape[1]
    print(f"    -> Unified Input Matrix: {X_all_df.shape[0]:,} timesteps x {num_features} multi-sensor features")

    # Compute Real-World Physical Totals (Watts)
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

    # Normalize targets to [0, 1]
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

    print(f"    -> Total Solar Capacity : {solar_max:,.1f} W ({solar_max/1000:,.1f} kW)")
    print(f"    -> Total Grid Capacity  : {grid_max:,.1f} W ({grid_max/1000:,.1f} kW)")
    print(f"    -> Total Demand Capacity: {demand_max:,.1f} W ({demand_max/1000:,.1f} kW)")

    return X_all_df, targets_norm, target_scalers


def create_multisensor_windows(X_data, y_data, lookback=96, horizon=4):
    X, y = [], []
    for i in range(len(X_data) - lookback - horizon + 1):
        X.append(X_data[i : i + lookback])
        # Flatten horizon for 3 targets: 4 steps * 3 targets = 12 outputs
        y.append(y_data[i + lookback : i + lookback + horizon, :].flatten())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class AllSensorsQIDKLSTM(nn.Module):
    """
    Multi-Sensor Recurrent Architecture for QIDK Edge Deployment.
    Accepts all 58 synchronized campus sensor features.
    Outputs 12 values: 4 timesteps ahead for (Solar, Grid, Demand).
    """
    def __init__(self, input_dim=58, hidden_dim=96, num_layers=2, output_dim=12):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc1 = nn.Linear(hidden_dim, 48)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(48, output_dim)

    def forward(self, x):
        # x: [batch, 96, 58]
        out, _ = self.lstm(x)
        last_step = out[:, -1, :] # [batch, hidden_dim]
        h = self.relu(self.fc1(last_step))
        preds = self.fc2(h)      # [batch, 12]
        return preds


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print(" CAMPUS-WIDE ALL-SENSORS LSTM TRAINING & QIDK DEPLOYMENT")
    print("=" * 70)

    # 1. Load & Merge Data
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

    X_train, y_train = create_multisensor_windows(X_train_df.values, y_train_arr, LOOKBACK, HORIZON)
    X_val, y_val = create_multisensor_windows(X_val_df.values, y_val_arr, LOOKBACK, HORIZON)
    X_test, y_test = create_multisensor_windows(X_test_df.values, y_test_arr, LOOKBACK, HORIZON)

    print(f"\n[2] Sequence Tensors Created:")
    print(f"    Train: {X_train.shape} -> Targets: {y_train.shape}")
    print(f"    Val:   {X_val.shape}   -> Targets: {y_val.shape}")
    print(f"    Test:  {X_test.shape}  -> Targets: {y_test.shape}")

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE, shuffle=False)

    # 3. Model Training
    model = AllSensorsQIDKLSTM(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=HORIZON * 3).to(DEVICE)
    best_weights_path = os.path.join(OUTPUT_DIR, "best_all_sensors_lstm.pt")
    
    if os.path.exists(best_weights_path):
        print(f"\n[3] Found existing trained weights at '{best_weights_path}'. Loading...")
        model.load_state_dict(torch.load(best_weights_path))
    else:
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        print(f"\n[3] Training Multi-Sensor PyTorch LSTM on CPU ({EPOCHS} Epochs)...")
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

    # 4. Evaluation in Real-World Watts
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()

    with torch.no_grad():
        test_preds_norm = model(torch.tensor(X_test).to(DEVICE)).numpy()

    # Reshape predictions and targets: (N, 4 steps, 3 targets)
    preds_reshaped = test_preds_norm.reshape(-1, HORIZON, 3)
    y_test_reshaped = y_test.reshape(-1, HORIZON, 3)

    domains = ["Solar", "Grid", "Demand"]
    capacities = [target_scalers["solar_max_W"], target_scalers["grid_max_W"], target_scalers["demand_max_W"]]

    print(f"\n[4] Multi-Sensor Evaluation Results in Real-World Watts (Test Set):")
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
    onnx_path = os.path.join(OUTPUT_DIR, "all_sensors_lstm_qidk.onnx")
    dummy_input = torch.randn(1, LOOKBACK, input_dim, dtype=torch.float32, device=DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_sequence_all_sensors"],
        output_names=["forecast_all_pillars"],
    )

    # Ensure self-contained ONNX
    onnx_model = onnx.load(onnx_path, load_external_data=True)
    onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)
    data_ext = onnx_path + ".data"
    if os.path.exists(data_ext):
        os.remove(data_ext)

    print(f"\n[5] Exported Static-Shape ONNX Model -> {onnx_path} ({os.path.getsize(onnx_path)/1024:.1f} KB)")

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
    sample_path = os.path.join(OUTPUT_DIR, "sample_input_all_sensors_1x96x58.npy")
    np.save(sample_path, sample_input)

    deploy_meta = {
        "model_type": "All_Sensors_Campus_Forecaster",
        "input_tensor_name": "input_sequence_all_sensors",
        "input_shape": [1, LOOKBACK, input_dim],
        "output_tensor_name": "forecast_all_pillars",
        "output_shape": [1, HORIZON * 3],
        "feature_count": input_dim,
        "feature_names": feature_names,
        "domains": domains,
        "capacities_W": target_scalers,
        "evaluation_summary": evaluation_summary
    }
    with open(os.path.join(OUTPUT_DIR, "all_sensors_qidk_meta.json"), "w") as f:
        json.dump(deploy_meta, f, indent=2)

    # 8. Create Standalone Edge Inference Script for QIDK
    runner_script = f'''#!/usr/bin/env python3
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
raw_output = session.run([meta["output_tensor_name"]], {{meta["input_tensor_name"]: input_data}})[0]
# Shape: [1, 12] -> Reshape to [4 time intervals, 3 domains: Solar, Grid, Demand]
preds = raw_output[0].reshape(4, 3)

caps = meta["capacities_W"]
solar_cap = caps["solar_max_W"]
grid_cap = caps["grid_max_W"]
demand_cap = caps["demand_max_W"]

print("=" * 65)
print("  QUALCOMM QIDK ON-DEVICE INFERENCE: CAMPUS NET-ZERO FORECAST")
print("=" * 65)
print(f" Timestep  |  Solar Yield (W) |  Grid Draw (W)  | Room Demand (W) | Net Balance (W)")
print("-" * 65)
for i in range(4):
    s_w = preds[i, 0] * solar_cap
    g_w = preds[i, 1] * grid_cap
    d_w = preds[i, 2] * demand_cap
    net_w = g_w + s_w - d_w
    step_min = (i + 1) * 15
    print(f"  +{{step_min:02d}} min | {{s_w:14,.1f}} W | {{g_w:13,.1f}} W | {{d_w:13,.1f}} W | {{net_w:13,.1f}} W")
print("=" * 65)
'''
    with open(os.path.join(OUTPUT_DIR, "qidk_all_sensors_inference.py"), "w") as f:
        f.write(runner_script)

    print(f"\n[7] All-Sensors QIDK Deployment Package created in '{OUTPUT_DIR}/':")
    print(f"    - Model:            {onnx_path}")
    print(f"    - Sample Payload:   {sample_path}")
    print(f"    - Deployment Meta:  {os.path.join(OUTPUT_DIR, 'all_sensors_qidk_meta.json')}")
    print(f"    - On-Device Runner: {os.path.join(OUTPUT_DIR, 'qidk_all_sensors_inference.py')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
