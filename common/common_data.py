#!/usr/bin/env python3
"""
common_data.py
--------------
Shared by train_lightgbm_24h.py and train_fair_lstm.py.
Saves all outputs (.onnx, .pt, .json, .pkl) directly inside the same folder (model/common).
"""

import os
import json
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- path resolution
if os.path.exists(os.path.join(SCRIPT_DIR, "normalized")) or os.path.exists(os.path.join(SCRIPT_DIR, "lstm_normalized")):
    MODEL_ROOT = SCRIPT_DIR
elif os.path.exists(os.path.join(SCRIPT_DIR, "..", "normalized")) or os.path.exists(os.path.join(SCRIPT_DIR, "..", "lstm_normalized")):
    MODEL_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
elif os.path.exists(os.path.join(SCRIPT_DIR, "..", "Data", "normalized")) or os.path.exists(os.path.join(SCRIPT_DIR, "..", "Data", "lstm_normalized")):
    MODEL_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Data"))
elif os.path.exists(os.path.join(SCRIPT_DIR, "../..", "normalized")) or os.path.exists(os.path.join(SCRIPT_DIR, "../..", "lstm_normalized")):
    MODEL_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
elif os.path.exists("normalized") or os.path.exists("lstm_normalized"):
    MODEL_ROOT = os.path.abspath(".")
else:
    MODEL_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if os.path.exists(os.path.join(MODEL_ROOT, "normalized")):
    DATA_DIR = os.path.join(MODEL_ROOT, "normalized")
elif os.path.exists(os.path.join(MODEL_ROOT, "lstm_normalized")):
    DATA_DIR = os.path.join(MODEL_ROOT, "lstm_normalized")
elif os.path.exists(os.path.join(MODEL_ROOT, "Data", "normalized")):
    DATA_DIR = os.path.join(MODEL_ROOT, "Data", "normalized")
elif os.path.exists(os.path.join(MODEL_ROOT, "Data", "lstm_normalized")):
    DATA_DIR = os.path.join(MODEL_ROOT, "Data", "lstm_normalized")
else:
    DATA_DIR = os.path.join(MODEL_ROOT, "normalized")

# Save everything directly inside this folder (model/common)
DEPLOY_DIR = SCRIPT_DIR
ONNX_DIR = SCRIPT_DIR
PT_DIR = SCRIPT_DIR
META_DIR = SCRIPT_DIR

LOOKBACK = 96
DOMAINS = ["Solar", "Grid", "Demand"]
TIME_COLS = ["hod_sin", "hod_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"]
METRICS = ("MAE_W", "RMSE_W", "nMAE_pct", "R2")


# ---------------------------------------------------------------- data
def load_and_merge_all_sensors():
    print("[1] Loading & aligning all 11 sensor Parquet files (58 features)...")
    with open(os.path.join(DATA_DIR, "features.json")) as f:
        meta = json.load(f)["nodes"]
    with open(os.path.join(DATA_DIR, "scaler_params.json")) as f:
        scalers = json.load(f)

    dfs = {nid: pd.read_parquet(os.path.join(DATA_DIR, f"{nid}.parquet")) for nid in meta.keys()}
    common_idx = dfs["SL-PL00-00"].index
    for df in dfs.values():
        common_idx = common_idx.intersection(df.index)

    feature_dict = {}
    for node_id, df in dfs.items():
        sub_df = df.loc[common_idx]
        for c in sub_df.columns:
            if c in TIME_COLS:
                continue
            feature_dict[f"{node_id}__{c}"] = sub_df[c].values

    pl_sub = dfs["SL-PL00-00"].loc[common_idx]
    for c in TIME_COLS:
        feature_dict[c] = pl_sub[c].values

    X_all_df = pd.DataFrame(feature_dict, index=common_idx)

    solar_nodes = ["SL-PL00-00", "SL-VN95-00", "SL-NI01-00", "SL-NI03-00", "SL-NI03-01", "SL-VN02-01"]
    room_nodes = ["SR-EM-KH04-00", "SR-EM-KH04-01", "SR-EM-KH04-02"]

    solar_watts = np.zeros(len(common_idx))
    for sn in solar_nodes:
        s_min, s_max = scalers[sn]["active_power_W"]["min"], scalers[sn]["active_power_W"]["max"]
        solar_watts += dfs[sn].loc[common_idx, "active_power_W"].values * (s_max - s_min) + s_min

    grid_watts = np.zeros(len(common_idx))
    g_sb_min, g_sb_max = scalers["EM-CR-SB00-00"]["gd_power_W"]["min"], scalers["EM-CR-SB00-00"]["gd_power_W"]["max"]
    grid_watts += dfs["EM-CR-SB00-00"].loc[common_idx, "gd_power_W"].values * (g_sb_max - g_sb_min) + g_sb_min
    g_kh_min, g_kh_max = scalers["EM-CR-KH95-00"]["power_total_W"]["min"], scalers["EM-CR-KH95-00"]["power_total_W"]["max"]
    grid_watts += dfs["EM-CR-KH95-00"].loc[common_idx, "power_total_W"].values * (g_kh_max - g_kh_min) + g_kh_min

    demand_watts = np.zeros(len(common_idx))
    for rn in room_nodes:
        r_min, r_max = scalers[rn]["power_W"]["min"], scalers[rn]["power_W"]["max"]
        demand_watts += dfs[rn].loc[common_idx, "power_W"].values * (r_max - r_min) + r_min

    solar_max, grid_max, demand_max = float(solar_watts.max()), float(grid_watts.max()), float(demand_watts.max())
    targets_norm = np.stack([solar_watts / solar_max, grid_watts / grid_max, demand_watts / demand_max], axis=1)
    target_scalers = {"solar_max_W": solar_max, "grid_max_W": grid_max, "demand_max_W": demand_max}

    print(f"    -> Timestamps: {len(common_idx):,} | Features: {X_all_df.shape[1]}")
    return X_all_df, targets_norm.astype(np.float32), target_scalers


def capacities(target_scalers):
    return [target_scalers["solar_max_W"], target_scalers["grid_max_W"], target_scalers["demand_max_W"]]


def split_bounds(n):
    """Chronological 70 / 15 / 15 split -> (n_train, n_val)."""
    return int(n * 0.70), int(n * 0.15)


def build_targets(y, lookback, horizon):
    """
    y: (T, 3) normalized targets.
    Returns (n_windows, horizon*3); window i covers y[i+lookback : i+lookback+horizon],
    flattened step-major (same layout as the original training scripts).
    """
    n_win = len(y) - lookback - horizon + 1
    if n_win <= 0:
        raise ValueError("Split too short for this lookback/horizon.")
    sw = np.lib.stride_tricks.sliding_window_view(y, horizon, axis=0)      # (T-H+1, 3, H)
    out = sw[lookback: lookback + n_win].transpose(0, 2, 1)                # (n_win, H, 3)
    return np.ascontiguousarray(out).reshape(n_win, -1).astype(np.float32)


# ---------------------------------------------------------------- metrics
def evaluate(pred_norm, true_norm, caps, horizon, steps=None):
    """
    pred_norm/true_norm: (N, horizon*3) normalized. steps: use only the first `steps` steps
    (e.g. steps=4 -> the 60-minute slice of a 96-step forecast).
    Returns {domain: {MAE_W, RMSE_W, nMAE_pct, R2}} in Watts.
    """
    p = pred_norm.reshape(-1, horizon, 3)
    t = true_norm.reshape(-1, horizon, 3)
    if steps is not None:
        p, t = p[:, :steps], t[:, :steps]

    out = {}
    for i, (dom, cap) in enumerate(zip(DOMAINS, caps)):
        pw, tw = p[:, :, i] * cap, t[:, :, i] * cap
        mae = float(np.mean(np.abs(pw - tw)))
        rmse = float(np.sqrt(np.mean((pw - tw) ** 2)))
        nmae = float(mae / cap * 100.0)
        ss_tot = float(np.sum((tw - np.mean(tw)) ** 2))
        ss_res = float(np.sum((tw - pw) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        out[dom] = {"MAE_W": round(mae, 1), "RMSE_W": round(rmse, 1),
                    "nMAE_pct": round(nmae, 2), "R2": round(r2, 4)}
    return out