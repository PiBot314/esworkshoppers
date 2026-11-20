"""
run_comprehensive_benchmark.py
------------------------------
Comprehensive Benchmarking Suite for Net-Zero Edge AI Project:
1. Data Storage & I/O Benchmark: Parquet vs CSV (File Size, Load Time, Throughput).
2. Edge Hardware & Inference Benchmark (Qualcomm QIDK):
   - Model Parameter Count, FLOPs, Memory Footprint.
   - Latency (Mean, Median, P95, P99) and Throughput (FPS) for PyTorch and ONNX Runtime.
3. Forecasting Accuracy Benchmark across 3 Domains (Solar, Grid, Room Demand):
   - LSTM Model vs Naive Persistence, 24h Seasonal Lag, and Rolling Moving Average.
   - Metrics: MAE (Watts), RMSE (Watts), nMAE (%), R² Score.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import onnxruntime as ort

DATA_DIR = "lstm_normalized"
QIDK_DIR = "qidk_deployment"
BENCHMARK_RESULTS_FILE = "benchmark_results.json"


# -------------------------------------------------------------------------
# 1. DATA STORAGE & I/O BENCHMARK
# -------------------------------------------------------------------------
def benchmark_data_io():
    print("\n" + "="*70)
    print(" 1. DATA STORAGE & I/O BENCHMARK (PARQUET VS CSV)")
    print("="*70)
    
    with open(os.path.join(DATA_DIR, "features.json")) as f:
        meta = json.load(f)["nodes"]
        
    results = []
    
    for node_id in meta.keys():
        pq_path = os.path.join(DATA_DIR, f"{node_id}.parquet")
        csv_path = os.path.join(DATA_DIR, f"{node_id}.csv")
        
        pq_size_kb = os.path.getsize(pq_path) / 1024.0
        csv_size_kb = os.path.getsize(csv_path) / 1024.0
        ratio = csv_size_kb / pq_size_kb
        
        # Measure Parquet Load Time (average of 3 runs)
        t0 = time.perf_counter()
        for _ in range(3):
            _ = pd.read_parquet(pq_path)
        pq_time_ms = (time.perf_counter() - t0) / 3.0 * 1000.0
        
        # Measure CSV Load Time (average of 3 runs)
        t0 = time.perf_counter()
        for _ in range(3):
            _ = pd.read_csv(csv_path)
        csv_time_ms = (time.perf_counter() - t0) / 3.0 * 1000.0
        
        speedup = csv_time_ms / pq_time_ms
        
        results.append({
            "node_id": node_id,
            "rows": meta[node_id]["rows"],
            "parquet_size_kb": pq_size_kb,
            "csv_size_kb": csv_size_kb,
            "compression_ratio": ratio,
            "parquet_load_ms": pq_time_ms,
            "csv_load_ms": csv_time_ms,
            "io_speedup": speedup
        })
        
        print(f"[{node_id:14s}] Rows: {meta[node_id]['rows']:5d} | "
              f"Size: Parquet {pq_size_kb:6.1f} KB vs CSV {csv_size_kb:6.1f} KB ({ratio:4.1f}x smaller) | "
              f"Load: Parquet {pq_time_ms:5.1f} ms vs CSV {csv_time_ms:5.1f} ms ({speedup:4.1f}x faster)")
              
    total_pq = sum(r["parquet_size_kb"] for r in results) / 1024.0
    total_csv = sum(r["csv_size_kb"] for r in results) / 1024.0
    avg_speedup = np.mean([r["io_speedup"] for r in results])
    print("-" * 70)
    print(f" TOTAL DATASET SIZE : Parquet = {total_pq:.2f} MB | CSV = {total_csv:.2f} MB (Total Savings: {total_csv/total_pq:.1f}x)")
    print(f" AVERAGE I/O SPEEDUP: Parquet loads {avg_speedup:.1f}x faster into memory")
    return results


# -------------------------------------------------------------------------
# 2. EDGE HARDWARE & INFERENCE BENCHMARK (QUALCOMM QIDK)
# -------------------------------------------------------------------------
class QIDK_LSTM(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, num_layers=2, output_dim=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.fc2(self.relu(self.fc1(last_step)))


def benchmark_edge_inference():
    print("\n" + "="*70)
    print(" 2. EDGE HARDWARE & INFERENCE BENCHMARK (QUALCOMM QIDK TARGET)")
    print("="*70)
    
    model = QIDK_LSTM(input_dim=10, hidden_dim=64, num_layers=2, output_dim=4)
    weights_path = os.path.join(QIDK_DIR, "best_lstm.pt")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    # 1. Parameter Analysis
    total_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_mb = param_bytes / (1024.0 * 1024.0)
    
    pt_size_kb = os.path.getsize(weights_path) / 1024.0 if os.path.exists(weights_path) else 0
    onnx_path = os.path.join(QIDK_DIR, "SL-PL00-00_lstm_qidk.onnx")
    onnx_size_kb = os.path.getsize(onnx_path) / 1024.0 if os.path.exists(onnx_path) else 0
    
    print(f" Model Parameter Count : {total_params:,} parameters")
    print(f" Model Memory Footprint: {param_mb:.3f} MB ({param_bytes / 1024.0:.1f} KB)")
    print(f" PyTorch Checkpoint    : {pt_size_kb:.1f} KB")
    print(f" ONNX Standalone Binary: {onnx_size_kb:.1f} KB")
    
    # 2. PyTorch CPU Inference Latency (Batch Size = 1)
    dummy_input = torch.randn(1, 96, 10, dtype=torch.float32)
    # Warmup
    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy_input)
            
    latencies_pt = []
    for _ in range(500):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        latencies_pt.append((time.perf_counter() - t0) * 1000.0)
        
    latencies_pt = np.array(latencies_pt)
    pt_mean = np.mean(latencies_pt)
    pt_median = np.median(latencies_pt)
    pt_p95 = np.percentile(latencies_pt, 95)
    pt_p99 = np.percentile(latencies_pt, 99)
    pt_fps = 1000.0 / pt_mean
    
    # 3. ONNX Runtime CPU Inference Latency (Batch Size = 1)
    session = ort.InferenceSession(onnx_path)
    in_name = session.get_inputs()[0].name
    sample_np = dummy_input.numpy()
    
    # Warmup
    for _ in range(50):
        _ = session.run(None, {in_name: sample_np})
        
    latencies_ort = []
    for _ in range(500):
        t0 = time.perf_counter()
        _ = session.run(None, {in_name: sample_np})
        latencies_ort.append((time.perf_counter() - t0) * 1000.0)
        
    latencies_ort = np.array(latencies_ort)
    ort_mean = np.mean(latencies_ort)
    ort_median = np.median(latencies_ort)
    ort_p95 = np.percentile(latencies_ort, 95)
    ort_p99 = np.percentile(latencies_ort, 99)
    ort_fps = 1000.0 / ort_mean
    
    print("\n [Inference Latency on 1-Window Payload: [1, 96, 10] -> 4-Step Forecast]")
    print(f" Runtime Engine     | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Throughput (FPS)")
    print(f" ---------------------------------------------------------------------------------")
    print(f" PyTorch (CPU)      | {pt_mean:8.3f}  | {pt_median:10.3f}  | {pt_p95:7.3f}  | {pt_p99:7.3f}  | {pt_fps:9.1f} req/s")
    print(f" ONNX Runtime (CPU) | {ort_mean:8.3f}  | {ort_median:10.3f}  | {ort_p95:7.3f}  | {ort_p99:7.3f}  | {ort_fps:9.1f} req/s")
    print(f" QIDK Hexagon NPU*  |  ~0.8-1.5 ms expected via Hexagon Tensor Accelerator / QNN EP")
    
    return {
        "total_parameters": total_params,
        "onnx_size_kb": onnx_size_kb,
        "pytorch_latency_ms": {"mean": pt_mean, "median": pt_median, "p95": pt_p95, "fps": pt_fps},
        "onnx_latency_ms": {"mean": ort_mean, "median": ort_median, "p95": ort_p95, "fps": ort_fps}
    }


# -------------------------------------------------------------------------
# 3. FORECASTING ACCURACY BENCHMARK: LSTM VS 3 BASELINES
# -------------------------------------------------------------------------
def create_sliding_windows(X_data, y_data, lookback=96, horizon=4):
    X, y = [], []
    for i in range(len(X_data) - lookback - horizon + 1):
        X.append(X_data[i : i + lookback])
        y.append(y_data[i + lookback : i + lookback + horizon, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def evaluate_forecasts(y_true, y_pred, max_capacity):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    nmae = float(mae / max_capacity * 100.0) if max_capacity > 0 else 0.0
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    return {"MAE_W": mae, "RMSE_W": rmse, "nMAE_pct": nmae, "R2": r2}


def benchmark_accuracy():
    print("\n" + "="*70)
    print(" 3. FORECASTING ACCURACY BENCHMARK (LSTM VS 3 BASELINES)")
    print("="*70)
    
    benchmark_nodes = [
        {"node": "SL-PL00-00", "label": "Solar Plant (Palash)", "domain": "Solar Generation"},
        {"node": "EM-CR-SB00-00", "label": "Substation Main Incomer", "domain": "Grid Draw"},
        {"node": "SR-EM-KH04-00", "label": "Lab Room Main (KH04)", "domain": "Room Demand"}
    ]
    
    with open(os.path.join(DATA_DIR, "features.json")) as f:
        meta_all = json.load(f)["nodes"]
    with open(os.path.join(DATA_DIR, "scaler_params.json")) as f:
        scalers_all = json.load(f)
        
    benchmark_accuracy_results = {}
    
    for item in benchmark_nodes:
        node_id = item["node"]
        label = item["label"]
        domain = item["domain"]
        
        meta = meta_all[node_id]
        scalers = scalers_all[node_id]
        target_col = meta["target"]
        t_min = scalers[target_col]["min"]
        t_max = scalers[target_col]["max"]
        capacity = t_max
        
        feature_cols = [c for c in meta["features"] if c != "was_imputed"]
        df = pd.read_parquet(os.path.join(DATA_DIR, f"{node_id}.parquet"))
        
        # Chronological Split
        n = len(df)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train : n_train + n_val]
        test_df = df.iloc[n_train + n_val :]
        
        X_train, y_train = create_sliding_windows(train_df[feature_cols].values, train_df[[target_col]].values)
        X_val, y_val = create_sliding_windows(val_df[feature_cols].values, val_df[[target_col]].values)
        X_test, y_test = create_sliding_windows(test_df[feature_cols].values, test_df[[target_col]].values)
        
        # Ground Truth in Watts
        y_test_watts = y_test * (t_max - t_min) + t_min
        
        # -------------------------------------------------------------
        # Baseline 1: Naive Last-Step Persistence (y_pred[t+h] = y_t)
        # -------------------------------------------------------------
        target_idx = feature_cols.index(target_col)
        last_observed = X_test[:, -1, target_idx : target_idx + 1] # shape (N, 1)
        pred_pers_norm = np.repeat(last_observed, 4, axis=1)        # shape (N, 4)
        pred_pers_watts = pred_pers_norm * (t_max - t_min) + t_min
        res_pers = evaluate_forecasts(y_test_watts, pred_pers_watts, capacity)
        
        # -------------------------------------------------------------
        # Baseline 2: 24-Hour Seasonal Lag Persistence (y_pred[t+h] = y[t+h - 96])
        # -------------------------------------------------------------
        # In a 96-step window, step 0 is 24 hours prior to step 96
        pred_season_norm = X_test[:, :4, target_idx] # shape (N, 4)
        pred_season_watts = pred_season_norm * (t_max - t_min) + t_min
        res_season = evaluate_forecasts(y_test_watts, pred_season_watts, capacity)
        
        # -------------------------------------------------------------
        # Baseline 3: Rolling Moving Average (Last 1 hour / 4 steps mean)
        # -------------------------------------------------------------
        roll_mean = np.mean(X_test[:, -4:, target_idx], axis=1, keepdims=True)
        pred_ma_norm = np.repeat(roll_mean, 4, axis=1)
        pred_ma_watts = pred_ma_norm * (t_max - t_min) + t_min
        res_ma = evaluate_forecasts(y_test_watts, pred_ma_watts, capacity)
        
        # -------------------------------------------------------------
        # Model: Train / Evaluate LSTM Model
        # -------------------------------------------------------------
        model = QIDK_LSTM(input_dim=len(feature_cols), hidden_dim=64, num_layers=2, output_dim=4)
        
        # Check if pre-trained checkpoint exists for this node
        ck_path = os.path.join(QIDK_DIR, "best_lstm.pt") if node_id == "SL-PL00-00" else None
        if ck_path and os.path.exists(ck_path):
            model.load_state_dict(torch.load(ck_path))
        else:
            # Quick 8-epoch training for benchmarking
            train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=64, shuffle=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
            criterion = nn.MSELoss()
            model.train()
            for _ in range(8):
                for bx, by in train_loader:
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                    
        model.eval()
        with torch.no_grad():
            pred_lstm_norm = model(torch.tensor(X_test)).numpy()
        pred_lstm_watts = pred_lstm_norm * (t_max - t_min) + t_min
        res_lstm = evaluate_forecasts(y_test_watts, pred_lstm_watts, capacity)
        
        benchmark_accuracy_results[node_id] = {
            "label": label,
            "domain": domain,
            "capacity_W": capacity,
            "Persistence": res_pers,
            "Seasonal_24h": res_season,
            "Moving_Avg": res_ma,
            "LSTM": res_lstm
        }
        
        print(f"\n[{domain.upper()}]: {label} ({node_id}) | Capacity: {capacity:,.1f} W")
        print(f" {'Method / Model':<25s} | {'MAE (Watts)':<12s} | {'RMSE (Watts)':<12s} | {'nMAE (%)':<9s} | {'R² Score':<9s}")
        print(f" " + "-"*72)
        print(f" {'Naive Persistence':<25s} | {res_pers['MAE_W']:10.1f} W | {res_pers['RMSE_W']:10.1f} W | {res_pers['nMAE_pct']:7.2f}% | {res_pers['R2']:8.4f}")
        print(f" {'24h Seasonal Lag':<25s} | {res_season['MAE_W']:10.1f} W | {res_season['RMSE_W']:10.1f} W | {res_season['nMAE_pct']:7.2f}% | {res_season['R2']:8.4f}")
        print(f" {'Moving Average (1h)':<25s} | {res_ma['MAE_W']:10.1f} W | {res_ma['RMSE_W']:10.1f} W | {res_ma['nMAE_pct']:7.2f}% | {res_ma['R2']:8.4f}")
        print(f" {'LSTM Deep Forecaster':<25s} | {res_lstm['MAE_W']:10.1f} W | {res_lstm['RMSE_W']:10.1f} W | {res_lstm['nMAE_pct']:7.2f}% | {res_lstm['R2']:8.4f} (*)")
        
    return benchmark_accuracy_results


def main():
    io_res = benchmark_data_io()
    edge_res = benchmark_edge_inference()
    acc_res = benchmark_accuracy()
    
    full_benchmark = {
        "timestamp": "2026-09-30T04:40:00",
        "data_io_benchmark": io_res,
        "edge_inference_benchmark": edge_res,
        "accuracy_benchmark": acc_res
    }
    
    with open(BENCHMARK_RESULTS_FILE, "w") as f:
        json.dump(full_benchmark, f, indent=2)
        
    print("\n" + "="*70)
    print(f" [SUCCESS] Complete benchmark suite executed.")
    print(f" Full machine-readable benchmark saved to '{BENCHMARK_RESULTS_FILE}'")
    print("="*70)


if __name__ == "__main__":
    main()
