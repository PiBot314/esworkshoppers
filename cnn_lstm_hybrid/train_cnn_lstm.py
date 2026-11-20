#!/usr/bin/env python3
"""
train_cnn_lstm.py
-----------------
Trains a 1D-CNN + LSTM hybrid network on Net Zero sensor data
and exports a clean, static ONNX model for AIMET quantization.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import common_data as cd


class CNNLSTMNetZero(nn.Module):
    def __init__(self, in_features=58, horizon=96, out_channels_per_step=3):
        super().__init__()
        self.horizon = horizon
        self.out_channels = out_channels_per_step

        # 1D-CNN feature extractor over temporal length 96
        # Input shape: (Batch, Channels/Features, Time) -> (B, 58, 96)
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Compresses 96 timesteps down to 48
        )

        # LSTM processes the compressed sequence: (Batch, Time=48, Channels=64)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        # Output projection head to flattened horizon (H * 3)
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, horizon * out_channels_per_step)
        )

    def forward(self, x):
        # x is (B, 96, 58) -> permute to (B, 58, 96) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)       # -> (B, 64, 48)

        # Permute to (B, 48, 64) for LSTM
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)    # -> (B, 48, 64)

        # Take last time step
        last_step = lstm_out[:, -1, :]  # -> (B, 64)
        return self.fc(last_step)       # -> (B, H * 3)


def make_dataset(X, y, start_idx, end_idx, horizon):
    """Constructs rolling lookback windows and flattened targets."""
    n_win = (end_idx - start_idx) - cd.LOOKBACK - horizon + 1
    xs = np.stack([X[start_idx + i: start_idx + i + cd.LOOKBACK] for i in range(n_win)]).astype(np.float32)
    ys = cd.build_targets(y[start_idx:end_idx], cd.LOOKBACK, horizon).astype(np.float32)
    return TensorDataset(torch.from_numpy(xs), torch.from_numpy(ys))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=96, help="Forecast horizon (default: 96 steps / 24h)")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--onnx_out", default="cnn_lstm_qidk_24h.onnx", help="Output ONNX filename")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data via common_data
    print("Loading and preprocessing dataset...")
    X_df, y, scal = cd.load_and_merge_all_sensors()
    X = X_df.values.astype(np.float32)
    caps = cd.capacities(scal)
    n_train, n_val = cd.split_bounds(len(X))

    print(f"Building window splits (Lookback: {cd.LOOKBACK}, Horizon: {args.horizon})...")
    train_ds = make_dataset(X, y, 0, n_train, args.horizon)
    val_ds = make_dataset(X, y, n_train, n_train + n_val, args.horizon)
    test_ds = make_dataset(X, y, n_train + n_val, len(X), args.horizon)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Model, Loss, Optimizer
    model = CNNLSTMNetZero(in_features=X.shape[1], horizon=args.horizon, out_channels_per_step=3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 3. Training Loop
    best_val_loss = float("inf")
    best_weights_path = "best_cnn_lstm.pt"

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx)
                val_loss += criterion(preds, by).item() * len(bx)
        val_loss /= len(val_ds)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_weights_path)

    # 4. Evaluation on Test Set in Real Watts
    print(f"\nLoading best checkpoint ({best_weights_path}) for test evaluation...")
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    preds_all = []
    with torch.no_grad():
        for bx, _ in test_loader:
            preds_all.append(model(bx.to(device)).cpu().numpy())
    preds_all = np.concatenate(preds_all, axis=0)

    test_y = test_ds.tensors[1].numpy()
    results = cd.evaluate(preds_all, test_y, caps, args.horizon)

    print("\n=== Test Split Performance (Watts) ===")
    for domain in cd.DOMAINS:
        res = results[domain]
        print(f"{domain:8s} | nMAE: {res['nMAE_pct']:6.2f}% | R2: {res['R2']:6.3f} | RMSE: {res['RMSE_W']:8.2f} W")

    # 5. Export to static ONNX for Qualcomm QAIRT / AIMET
    print(f"\nExporting static ONNX model -> {args.onnx_out} ...")
    model.to("cpu")
    model.eval()
    dummy_input = torch.randn(1, cd.LOOKBACK, X.shape[1], dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        args.onnx_out,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input_sequence_all_sensors"],
        output_names=["forecast_output"],
    )
    print(f"Successfully exported {args.onnx_out}!")


if __name__ == "__main__":
    main()