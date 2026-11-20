#!/usr/bin/env python3
"""
benchmark.py
------------
Comprehensive Dual-Horizon 3-Way Benchmark for 60-Minute & 24-Hour models.
Automatically searches your specific directory structure (common, guided_lstm, 
lightgbm_baseline, lstm_baseline) to find and evaluate the models.
"""

import os
import json
import time
import joblib
import psutil
import numpy as np
import torch
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

SEARCH_DIRS = [
    SCRIPT_DIR,                                        # model/common
    os.path.join(SCRIPT_DIR, "metadata"),
    os.path.join(SCRIPT_DIR, "onnx"),
    os.path.join(SCRIPT_DIR, "pt"),
    os.path.join(MODEL_ROOT, "guided_lstm"),           # model/guided_lstm
    os.path.join(MODEL_ROOT, "lightgbm_baseline"),     # model/lightgbm_baseline
    os.path.join(MODEL_ROOT, "lstm_baseline"),         # model/lstm_baseline
    os.path.join(MODEL_ROOT, "lstm_baseline", "metadata"),
    os.path.join(MODEL_ROOT, "lstm_baseline", "onnx"),
    os.path.join(MODEL_ROOT, "lstm_baseline", "pt"),
]


def find_file(filenames):
    for d in SEARCH_DIRS:
        if not os.path.exists(d):
            continue
        for f in filenames:
            p = os.path.join(d, f)
            if os.path.exists(p):
                return p
    return None


def read_power_watts():
    sysfs_paths = [
        "/sys/class/power_supply/battery/power_now",
        "/sys/class/power_supply/qcom-battery/power_now",
        "/sys/class/hwmon/hwmon0/power1_input"
    ]
    for path in sysfs_paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return float(f.read().strip()) / 1e6
            except Exception:
                pass
    cpu_pct = max(psutil.cpu_percent(interval=None), 5.0)
    return 1.80 + 4.50 * (cpu_pct / 100.0)


def get_ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)


def profile_callable(run_fn, warmup=30, runs=300):
    for _ in range(warmup):
        run_fn()

    psutil.cpu_percent(interval=None)
    ram_before = get_ram_mb()
    latencies_ms = []
    power_samples_w = []

    t_total_start = time.perf_counter()
    for i in range(runs):
        t0 = time.perf_counter()
        run_fn()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if i % 20 == 0:
            power_samples_w.append(read_power_watts())
    total_time_ms = (time.perf_counter() - t_total_start) * 1000.0

    ram_after = get_ram_mb()
    lat_arr = np.array(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(lat_arr))
    p95_ms = float(np.percentile(lat_arr, 95))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    avg_power_w = float(np.mean(power_samples_w)) if power_samples_w else 2.50
    energy_mj = avg_power_w * mean_ms

    return {
        "mean_ms": mean_ms, "p95_ms": p95_ms, "fps": fps, 
        "ram_mb": max(ram_before, ram_after), "power_w": avg_power_w, "energy_mj": energy_mj
    }


def count_lstm_params(pt_path, fallback_hidden=128, fallback_fc=128, output_dim=12):
    if pt_path and os.path.exists(pt_path):
        state = torch.load(pt_path, map_location="cpu", weights_only=True)
        return sum(v.numel() for k, v in state.items() if "prior_anchor" not in k and "feature_gate" not in k)
    
    input_dim = 58
    l1 = 4 * fallback_hidden * (input_dim + fallback_hidden + 2)
    l2 = 4 * fallback_hidden * (fallback_hidden + fallback_hidden + 2)
    fc = (fallback_hidden * fallback_fc + fallback_fc) + (fallback_fc * output_dim + output_dim)
    return l1 + l2 + fc


def load_metrics(path, default_key="evaluation_summary"):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            d = json.load(f)
            return d.get(default_key, d)
    return {}


def evaluate_horizon(title, orig_meta, lgb_meta, guid_meta, orig_onnx, guid_onnx, lgb_pkl, orig_pt, guid_pt, horizon_steps):
    lines = []
    lines.append("\n" + "=" * 96)
    lines.append(f" HORIZON: {title.upper()}")
    lines.append("=" * 96)
    
    m_orig = load_metrics(orig_meta)
    m_guid = load_metrics(guid_meta)
    
    m_lgb = {}
    if lgb_meta and os.path.exists(lgb_meta):
        with open(lgb_meta, "r") as f:
            d = json.load(f)
            # Handle standard format or flat dict
            m_lgb = d.get("evaluation_summary", d.get("accuracy", {}).get("lightgbm_2d", d))

    lines.append(f" {'Domain & Model':<34s} | {'MAE (Watts)':<13s} | {'RMSE (Watts)':<13s} | {'nMAE (%)':<10s} | {'R² Score':<10s}")
    lines.append("-" * 96)

    for dom in ["Solar", "Grid", "Demand"]:
        o = m_orig.get(dom, {})
        l = m_lgb.get(dom, {})
        g = m_guid.get(dom, {})
        
        lines.append(f" {dom + ' - 1. Baseline LSTM':<34s} | {o.get('MAE_W', 0):11,.1f} W | {o.get('RMSE_W', 0):11,.1f} W | {o.get('nMAE_pct', 0):8.2f}% | {o.get('R2', 0):9.4f}")
        if l:
            lines.append(f" {dom + ' - 2. LightGBM (2D)':<34s} | {l.get('MAE_W', 0):11,.1f} W | {l.get('RMSE_W', 0):11,.1f} W | {l.get('nMAE_pct', 0):8.2f}% | {l.get('R2', 0):9.4f}")
        else:
            lines.append(f" {dom + ' - 2. LightGBM (2D)':<34s} | {'--':>13s} | {'--':>13s} | {'--':>9s} | {'--':>9s}")
        lines.append(f" {dom + ' - 3. Guided LSTM':<34s} | {g.get('MAE_W', 0):11,.1f} W | {g.get('RMSE_W', 0):11,.1f} W | {g.get('nMAE_pct', 0):8.2f}% | {g.get('R2', 0):9.4f}")
        lines.append("-" * 96)

    hw_o = {"mean_ms": 0.0, "p95_ms": 0.0, "energy_mj": 0.0}
    hw_g = {"mean_ms": 0.0, "p95_ms": 0.0, "energy_mj": 0.0}
    hw_l = {"mean_ms": 0.0, "p95_ms": 0.0, "energy_mj": 0.0}
    
    dummy_3d = np.random.randn(1, 96, 58).astype(np.float32)
    dummy_2d = np.random.randn(1, 58).astype(np.float32)

    if orig_onnx and os.path.exists(orig_onnx):
        sess = ort.InferenceSession(orig_onnx, providers=["CPUExecutionProvider"])
        hw_o = profile_callable(lambda: sess.run(None, {sess.get_inputs()[0].name: dummy_3d}))
        
    if guid_onnx and os.path.exists(guid_onnx):
        sess = ort.InferenceSession(guid_onnx, providers=["CPUExecutionProvider"])
        hw_g = profile_callable(lambda: sess.run(None, {sess.get_inputs()[0].name: dummy_3d}))
        
    if lgb_pkl and os.path.exists(lgb_pkl):
        lgb_mod = joblib.load(lgb_pkl)
        hw_l = profile_callable(lambda: lgb_mod.predict(dummy_2d))

    p_o = count_lstm_params(orig_pt, output_dim=horizon_steps*3)
    p_g = count_lstm_params(guid_pt, output_dim=horizon_steps*3)

    lines.append(f" {'Latency (Mean / P95)':<34s} | {'Baseline LSTM':<17s} | {'LightGBM (2D)':<17s} | {'Guided LSTM':<17s}")
    lines.append(f" {' ':<34s} | {hw_o['mean_ms']:5.2f} / {hw_o['p95_ms']:5.2f} ms | {hw_l['mean_ms']:5.2f} / {hw_l['p95_ms']:5.2f} ms | {hw_g['mean_ms']:5.2f} / {hw_g['p95_ms']:5.2f} ms")
    lines.append(f" {'Parameter Count':<34s} | {p_o:,} params{'':>5s} | {'Tree ensemble':<17s} | {p_g:,} params")
    lines.append(f" {'Energy per Inference':<34s} | {hw_o['energy_mj']:.3f} mJ{'':>8s} | {hw_l['energy_mj']:.3f} mJ{'':>8s} | {hw_g['energy_mj']:.3f} mJ")
    lines.append("=" * 96)

    return lines


def main():
    print("=" * 96)
    print(" DUAL-HORIZON BENCHMARK SUITE: 60-MINUTE & 24-HOUR FORECASTS")
    print("=" * 96)
    print(f" Generated Timestamp : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report = []

    # --- 60 MINUTE (4 Steps) Searches ---
    orig_m_h4 = find_file(["baseline_lstm_meta_h4.json", "all_sensors_qidk_meta.json"])
    guid_m_h4 = find_file(["guided_lstm_meta_h4.json", "lgbm_guided_lstm_meta.json"])
    lgb_m_h4  = find_file(["lightgbm_metrics_24h_first60min.json", "lightgbm_metrics_h4.json", "lgbm_guided_lstm_benchmark_report.json"]) 
    orig_o_h4 = find_file(["baseline_lstm_qidk_h4.onnx", "all_sensors_lstm_qidk.onnx"])
    guid_o_h4 = find_file(["guided_lstm_qidk_h4.onnx", "lgbm_guided_all_sensors_lstm_qidk.onnx"])
    lgb_p_h4  = find_file(["lightgbm_all_sensors.pkl"]) 
    orig_p_h4 = find_file(["best_baseline_lstm_h4.pt", "best_all_sensors_lstm.pt"])
    guid_p_h4 = find_file(["best_guided_lstm_h4.pt", "best_lgbm_guided_lstm.pt"])
    
    if orig_m_h4 and guid_m_h4:
        report.extend(evaluate_horizon(
            "60 Minutes (4 Steps)", 
            orig_m_h4, lgb_m_h4, guid_m_h4, 
            orig_o_h4, guid_o_h4, lgb_p_h4, 
            orig_p_h4, guid_p_h4, 4
        ))
    else:
        print("[WARNING] Could not find metadata for 60-Minute horizon.")

    # --- 24 HOUR (96 Steps) Searches ---
    orig_m_24h = find_file(["baseline_lstm_meta_24h.json"])
    guid_m_24h = find_file(["guided_lstm_meta_24h.json"])
    lgb_m_24h  = find_file(["lightgbm_metrics_24h.json"])
    orig_o_24h = find_file(["baseline_lstm_qidk_24h.onnx"])
    guid_o_24h = find_file(["guided_lstm_qidk_24h.onnx"])
    lgb_p_24h  = find_file(["lightgbm_all_sensors_24h.pkl"])
    orig_p_24h = find_file(["best_baseline_lstm_24h.pt"])
    guid_p_24h = find_file(["best_guided_lstm_24h.pt"])
    
    if orig_m_24h and guid_m_24h:
        report.extend(evaluate_horizon(
            "24 Hours (96 Steps)", 
            orig_m_24h, lgb_m_24h, guid_m_24h, 
            orig_o_24h, guid_o_24h, lgb_p_24h, 
            orig_p_24h, guid_p_24h, 96
        ))
    else:
        print("[WARNING] Could not find metadata for 24-Hour horizon.")

    if not report:
        print("[ERROR] No models evaluated. Check paths and ensure training scripts have completed.")
        return

    report_str = "\n".join(report)
    print(report_str)

    out_txt = os.path.join(SCRIPT_DIR, "dual_horizon_benchmark_report.txt")
    with open(out_txt, "w") as f:
        f.write(report_str)
    print(f"\n[SUCCESS] Saved comprehensive dual-horizon report to '{out_txt}'")


if __name__ == "__main__":
    main()