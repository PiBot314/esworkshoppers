#!/usr/bin/env python3
"""
cnn_benchmark.py
----------------
Evaluates the CNN-LSTM Hybrid model for:
  - Accuracy: Solar, Grid, Demand (MAE, RMSE, nMAE, R2) for 24h (96 steps) and 60m (4 steps)
  - Hardware: Total parameters, host CPU latency (Mean & P95)
"""

import argparse
import os
import time
import numpy as np
import torch

from train_cnn_lstm import CNNLSTMNetZero, make_dataset
import common_data as cd


def compute_metrics(preds, targets, capacities, horizon_steps):
    """
    Computes MAE, RMSE, nMAE (%), and R2 across Solar, Grid, Demand
    for a given number of horizon steps.
    """
    # Handle list, tuple, np.ndarray, or dict for capacities
    if isinstance(capacities, (list, tuple, np.ndarray)):
        caps_dict = {d: float(capacities[i]) for i, d in enumerate(cd.DOMAINS)}
    else:
        caps_dict = {d: float(capacities[d]) for d in cd.DOMAINS}

    # Reshape predictions and targets: (N, horizon * 3) -> (N, horizon, 3)
    p_3d = preds.reshape(preds.shape[0], -1, 3)[:, :horizon_steps, :]
    t_3d = targets.reshape(targets.shape[0], -1, 3)[:, :horizon_steps, :]

    results = {}
    for idx, domain in enumerate(cd.DOMAINS):
        p = p_3d[:, :, idx].flatten()
        t = t_3d[:, :, idx].flatten()

        mae = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        cap = caps_dict[domain]
        nmae = (mae / cap) * 100.0

        ss_tot = np.sum((t - np.mean(t)) ** 2)
        ss_res = np.sum((t - p) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else -999.0

        results[domain] = {
            "MAE": mae,
            "RMSE": rmse,
            "nMAE": nmae,
            "R2": r2
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="best_cnn_lstm.pt", help="Path to model weights")
    parser.add_argument("--horizon", type=int, default=96, help="Full horizon steps (default: 96)")
    parser.add_argument("--runs", type=int, default=300, help="Inference benchmark iterations")
    args = parser.parse_args()

    # 1. Load Data
    print("Loading test data...")
    X_df, y, scal = cd.load_and_merge_all_sensors()
    X = X_df.values.astype(np.float32)
    caps = cd.capacities(scal)
    n_train, n_val = cd.split_bounds(len(X))

    test_ds = make_dataset(X, y, n_train + n_val, len(X), args.horizon)
    test_x = test_ds.tensors[0]
    test_y = test_ds.tensors[1].numpy()

    # 2. Load Model
    print(f"Loading model checkpoint from {args.weights}...")
    model = CNNLSTMNetZero(in_features=X.shape[1], horizon=args.horizon, out_channels_per_step=3)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    # Total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 3. Predict & Accuracy Evaluation
    print("Running evaluation on test split...")
    with torch.no_grad():
        preds_all = model(test_x).numpy()

    acc_24h = compute_metrics(preds_all, test_y, caps, horizon_steps=96)
    acc_60m = compute_metrics(preds_all, test_y, caps, horizon_steps=4)

    # 4. Latency Benchmark (Host CPU)
    print(f"\nRunning {args.runs} benchmark inferences for latency...")
    dummy_input = torch.randn(1, cd.LOOKBACK, X.shape[1], dtype=torch.float32)

    # Warmup
    for _ in range(30):
        with torch.no_grad():
            _ = model(dummy_input)

    latencies = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)

    # 5. Output Results Formatted for Markdown Tables
    print("\n" + "=" * 60)
    print("COPY-PASTE BENCHMARK RESULTS")
    print("=" * 60)

    print("\n### Horizon: 60 Minutes (4 Steps)")
    print("**Accuracy Metrics**")
    print("| Domain & Metric | CNN-LSTM Hybrid |")
    print("| :--- | :--- |")
    for d in cd.DOMAINS:
        print(f"| **{d} - MAE (W)** | {acc_60m[d]['MAE']:,.1f} |")
        print(f"| **{d} - RMSE (W)** | {acc_60m[d]['RMSE']:,.1f} |")
        print(f"| **{d} - nMAE (%)** | {acc_60m[d]['nMAE']:.2f} |")
        print(f"| **{d} - R²** | {acc_60m[d]['R2']:.4f} |")

    print("\n### Horizon: 24 Hours (96 Steps)")
    print("**Accuracy Metrics**")
    print("| Domain & Metric | CNN-LSTM Hybrid |")
    print("| :--- | :--- |")
    for d in cd.DOMAINS:
        print(f"| **{d} - MAE (W)** | {acc_24h[d]['MAE']:,.1f} |")
        print(f"| **{d} - RMSE (W)** | {acc_24h[d]['RMSE']:,.1f} |")
        print(f"| **{d} - nMAE (%)** | {acc_24h[d]['nMAE']:.2f} |")
        print(f"| **{d} - R²** | {acc_24h[d]['R2']:.4f} |")

    print("\n**Hardware & Performance**")
    print("| Metric | CNN-LSTM Hybrid |")
    print("| :--- | :--- |")
    print(f"| **Parameter Count** | {total_params:,} |")
    print(f"| **Host Latency (Mean / P95)** | {mean_lat:.2f} / {p95_lat:.2f} ms |")
    print("(Run on QIDK board using `qnn-net-run` for official NPU Latency & mJ metrics)")


if __name__ == "__main__":
    main()