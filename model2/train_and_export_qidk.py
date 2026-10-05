"""
train_and_export_qidk.py
-------------------------
Complete pipeline to train an LSTM forecaster on normalized campus IoT data
and export the model to ONNX format optimized for deployment on the
Qualcomm Innovators Development Kit (QIDK) / Hexagon NPU.

Workflow:
1. Loads normalized 15-minute time series from lstm_normalized/.
2. Constructs 3D sequence tensors (Lookback = 96 steps [24h], Horizon = 4 steps [1h]).
3. Trains a PyTorch LSTM model with early stopping.
4. Evaluates real-world error metrics (RMSE, MAE, R²) in actual Watts.
5. Exports to static-shape ONNX (1, 96, D) required by Qualcomm QNN/SNPE.
6. Verifies numerical parity between PyTorch and ONNX Runtime.
7. Packages on-device inference runner for QIDK.
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

# Configuration
NODE_NAME = "SL-PL00-00"  # Target Node (Solar Generation)
DATA_DIR = "lstm_normalized"
LOOKBACK = 96              # 24 hours of 15-min history
HORIZON = 4                # Predict next 1 hour (4 x 15-min intervals)
BATCH_SIZE = 64
EPOCHS = 15
HIDDEN_DIM = 64
NUM_LAYERS = 2
LR = 0.002
DEVICE = torch.device("cpu")
OUTPUT_DIR = "qidk_deployment"


class QIDK_LSTM(nn.Module):
    """
    Qualcomm QNN / Hexagon NPU-compatible LSTM Architecture.
    Uses standard operators supported by Qualcomm QNN ONNX Converter.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(QIDK_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Select last time-step representation
        last_step = lstm_out[:, -1, :]
        h = self.relu(self.fc1(last_step))
        out = self.fc2(h)
        return out


def create_sliding_windows(X_data, y_data, lookback, horizon):
    X, y = [], []
    for i in range(len(X_data) - lookback - horizon + 1):
        X.append(X_data[i : i + lookback])
        y.append(y_data[i + lookback : i + lookback + horizon, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print(f"  TRAINING & QIDK DEPLOYMENT PIPELINE: Node [{NODE_NAME}]")
    print("=" * 65)

    # 1. Load Data & Scalers
    meta_path = os.path.join(DATA_DIR, "features.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)["nodes"][NODE_NAME]

    scaler_path = os.path.join(DATA_DIR, "scaler_params.json")
    with open(scaler_path, "r") as f:
        scalers = json.load(f)[NODE_NAME]

    target_col = meta["target"]
    feature_cols = [c for c in meta["features"] if c != "was_imputed"]
    input_dim = len(feature_cols)

    data_file = os.path.join(DATA_DIR, f"{NODE_NAME}.parquet")
    df = pd.read_parquet(data_file)
    print(f"[1] Loaded dataset: {len(df)} steps | Target: {target_col} | Features: {input_dim}")

    # 2. Chronological Split (70% Train, 15% Val, 15% Test)
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    X_train, y_train = create_sliding_windows(train_df[feature_cols].values, train_df[[target_col]].values, LOOKBACK, HORIZON)
    X_val, y_val = create_sliding_windows(val_df[feature_cols].values, val_df[[target_col]].values, LOOKBACK, HORIZON)
    X_test, y_test = create_sliding_windows(test_df[feature_cols].values, test_df[[target_col]].values, LOOKBACK, HORIZON)

    print(f"[2] Sequence Tensors Created:")
    print(f"    Train: {X_train.shape} -> Targets: {y_train.shape}")
    print(f"    Val:   {X_val.shape}   -> Targets: {y_val.shape}")
    print(f"    Test:  {X_test.shape}  -> Targets: {y_test.shape}")

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE, shuffle=False)

    # 3. Model Training
    model = QIDK_LSTM(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=HORIZON).to(DEVICE)
    best_weights_path = os.path.join(OUTPUT_DIR, "best_lstm.pt")

    if os.path.exists(best_weights_path):
        print(f"\n[3] Found existing trained weights at '{best_weights_path}'. Loading checkpoint...")
        model.load_state_dict(torch.load(best_weights_path))
    else:
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        print(f"\n[3] Training PyTorch LSTM on CPU ({EPOCHS} Epochs)...")
        best_val_loss = float("inf")

        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
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

    # Inverse Transform to Watts
    t_min = scalers[target_col]["min"]
    t_max = scalers[target_col]["max"]

    test_preds_watts = test_preds_norm * (t_max - t_min) + t_min
    y_test_watts = y_test * (t_max - t_min) + t_min

    rmse = np.sqrt(np.mean((test_preds_watts - y_test_watts) ** 2))
    mae = np.mean(np.abs(test_preds_watts - y_test_watts))
    ss_tot = np.sum((y_test_watts - np.mean(y_test_watts)) ** 2)
    ss_res = np.sum((y_test_watts - test_preds_watts) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)

    print(f"\n[4] Evaluation Metrics in Real-World Watts (Test Set):")
    print(f"    -> Target Range: {t_min:.1f} W to {t_max:.1f} W")
    print(f"    -> Test RMSE:    {rmse:.2f} W")
    print(f"    -> Test MAE:     {mae:.2f} W")
    print(f"    -> R² Score:     {r2:.4f}")

    # 5. Export to ONNX with Static Shapes for Qualcomm QIDK
    # Qualcomm Hexagon NPU requires fixed batch size 1 and fixed sequence length
    onnx_path = os.path.join(OUTPUT_DIR, f"{NODE_NAME}_lstm_qidk.onnx")
    dummy_input = torch.randn(1, LOOKBACK, input_dim, dtype=torch.float32, device=DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_sequence"],
        output_names=["forecast_horizon"],
    )
    print(f"\n[5] Exported Static-Shape ONNX Model -> {onnx_path}")

    # Check ONNX validity
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("    -> ONNX structural verification passed!")

    # 6. Verify Numerical Parity via ONNX Runtime
    ort_session = ort.InferenceSession(onnx_path)
    sample_input = X_test[:1]  # shape: (1, 96, input_dim)

    with torch.no_grad():
        pt_out = model(torch.tensor(sample_input).to(DEVICE)).numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: sample_input}
    ort_out = ort_session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(pt_out - ort_out))
    print(f"[6] ONNX vs PyTorch Numerical Parity: Max diff = {max_diff:.8e}")
    assert max_diff < 1e-4, "Numerical divergence between PyTorch and ONNX!"
    print("    -> Numerical parity confirmed (identical predictions)!")

    # 7. Package Calibration & Deployment Files
    sample_path = os.path.join(OUTPUT_DIR, "sample_input_1x96.npy")
    np.save(sample_path, sample_input)

    deploy_meta = {
        "node_id": NODE_NAME,
        "input_tensor_name": "input_sequence",
        "input_shape": [1, LOOKBACK, input_dim],
        "output_tensor_name": "forecast_horizon",
        "output_shape": [1, HORIZON],
        "feature_order": feature_cols,
        "target_col": target_col,
        "scaler_min_W": t_min,
        "scaler_max_W": t_max,
        "test_rmse_W": float(rmse),
        "test_mae_W": float(mae),
        "test_r2": float(r2)
    }
    with open(os.path.join(OUTPUT_DIR, "qidk_model_meta.json"), "w") as f:
        json.dump(deploy_meta, f, indent=2)

    # 8. Create Standalone Edge Inference Script for QIDK
    edge_script_content = f'''#!/usr/bin/env python3
"""
qidk_edge_inference.py
Runs real-time inference on the Qualcomm QIDK board using ONNX Runtime / QNN Execution Provider.
"""
import numpy as np
import onnxruntime as ort
import json

# Load metadata
with open("qidk_model_meta.json") as f:
    meta = json.load(f)

# Initialize Session (Falls back to CPU if QNN Execution Provider is unconfigured)
available_providers = ort.get_available_providers()
print(f"Available Execution Providers: {{available_providers}}")

# On QIDK: use QNNExecutionProvider for Hexagon NPU acceleration
providers = ["QNNExecutionProvider", "CPUExecutionProvider"] if "QNNExecutionProvider" in available_providers else ["CPUExecutionProvider"]
session = ort.InferenceSession("{NODE_NAME}_lstm_qidk.onnx", providers=providers)

# Load sample 24-hour normalized window
input_data = np.load("sample_input_1x96.npy").astype(np.float32)

# Run Inference
outputs = session.run([meta["output_tensor_name"]], {{meta["input_tensor_name"]: input_data}})[0]

# Unscale to real Watts
preds_watts = outputs[0] * (meta["scaler_max_W"] - meta["scaler_min_W"]) + meta["scaler_min_W"]

print("\\n=== ON-DEVICE INFERENCE RESULT (QIDK) ===")
print(f"Predicted Power for next 4 intervals (+15m, +30m, +45m, +60m):")
for i, w in enumerate(preds_watts, 1):
    print(f"  T + {{i*15}} min: {{w:.2f}} Watts")
'''
    with open(os.path.join(OUTPUT_DIR, "qidk_edge_inference.py"), "w") as f:
        f.write(edge_script_content)

    print(f"\n[7] QIDK Deployment Package created in '{OUTPUT_DIR}/':")
    print(f"    - Model:              {onnx_path}")
    print(f"    - Sample Input:       {sample_path}")
    print(f"    - Deployment Meta:    {os.path.join(OUTPUT_DIR, 'qidk_model_meta.json')}")
    print(f"    - On-Device Runner:   {os.path.join(OUTPUT_DIR, 'qidk_edge_inference.py')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
