#!/usr/bin/env python3
"""
train_lightgbm_24h.py
---------------------
Step 2: 24-hour (96-step) multi-output LightGBM (288 targets = 96 steps x 3 domains).

Same hyper-parameters as the 60-min LightGBM (200 trees, 31 leaves, lr 0.05) so the only
change is the horizon. Unlike the 60-min script, this one also EVALUATES on the held-out
test split, and saves:

  lightgbm_all_sensors_24h.pkl                  model            (next to this script)
  lightgbm_metrics_24h.json                     24 hr test metrics   {"Solar": {...}, ...}
  lightgbm_metrics_24h_first60min.json          metrics on first 4 steps only
  lgbm_feature_weights_24h.json                 gain-importance weights (NEW file; the
                                                existing lgbm_feature_weights.json is NOT touched)

NOTE: 288 estimators x 200 trees -> expect ~24x the training time and file size of the
60-min model (roughly 150-200 MB pickle).
"""

import os
import json
import time
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

import common_data as cd

HORIZON = 96
OUTPUT_DIR = cd.SCRIPT_DIR


def main():
    print("=" * 70)
    print(" TRAINING 24-HOUR (96-STEP) LIGHTGBM ON 58-SENSOR CAMPUS TELEMETRY")
    print("=" * 70)

    X_df, y, target_scalers = cd.load_and_merge_all_sensors()
    feature_names = list(X_df.columns)
    X = X_df.values.astype(np.float32)
    caps = cd.capacities(target_scalers)

    n_train, n_val = cd.split_bounds(len(X))
    bounds = {"train": (0, n_train), "test": (n_train + n_val, len(X))}

    def make(lo, hi):
        Xs, ys = X[lo:hi], y[lo:hi]
        n_win = len(Xs) - cd.LOOKBACK - HORIZON + 1
        # features of the LAST observed step of each window; targets = next 96 steps
        return (Xs[cd.LOOKBACK - 1: cd.LOOKBACK - 1 + n_win],
                cd.build_targets(ys, cd.LOOKBACK, HORIZON))

    X_train, y_train = make(*bounds["train"])
    X_test, y_test = make(*bounds["test"])
    print(f"[2] Train: {X_train.shape} -> {y_train.shape} | Test: {X_test.shape} -> {y_test.shape}")

    print("\n[3] Fitting MultiOutputRegressor(LGBMRegressor) with 288 outputs (this takes a while)...")
    t0 = time.perf_counter()
    base = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31,
                             n_jobs=-1, random_state=42, verbose=-1)
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    print(f"    -> Done in {time.perf_counter() - t0:.1f} s")

    model_path = os.path.join(OUTPUT_DIR, "lightgbm_all_sensors_24h.pkl")
    joblib.dump(model, model_path)
    print(f"[4] Saved model -> '{model_path}' ({os.path.getsize(model_path) / 1024 / 1024:.1f} MB)")

    # ------------------------------------------------------------ test evaluation
    preds = model.predict(X_test).astype(np.float32)
    full = cd.evaluate(preds, y_test, caps, HORIZON)
    first60 = cd.evaluate(preds, y_test, caps, HORIZON, steps=4)

    print("\n[5] TEST METRICS (24 hr, all 96 steps)")
    for dom, m in full.items():
        print(f"    {dom:<7s} MAE {m['MAE_W']:>10,.1f} W | RMSE {m['RMSE_W']:>10,.1f} W | "
              f"nMAE {m['nMAE_pct']:5.2f}% | R2 {m['R2']:.4f}")

    with open(os.path.join(OUTPUT_DIR, "lightgbm_metrics_24h.json"), "w") as f:
        json.dump(full, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "lightgbm_metrics_24h_first60min.json"), "w") as f:
        json.dump(first60, f, indent=2)

    # ------------------------------------------------------------ feature gain weights
    # Output layout is step-major (step*3 + domain), so domain = out_idx % 3.
    domain_gains = np.zeros((3, len(feature_names)), dtype=np.float64)
    for out_idx, est in enumerate(model.estimators_):
        g = est.booster_.feature_importance(importance_type="gain")
        domain_gains[out_idx % 3] += g / (np.sum(g) + 1e-12)
    combined = np.mean(domain_gains, axis=0)
    weights = combined / (np.mean(combined) + 1e-12)

    payload = {
        "feature_names": feature_names,
        "feature_weight_map": {feature_names[i]: round(float(weights[i]), 5) for i in range(len(feature_names))},
    }
    for d in {cd.META_DIR, OUTPUT_DIR}:
        with open(os.path.join(d, "lgbm_feature_weights_24h.json"), "w") as f:
            json.dump(payload, f, indent=2)
    print(f"[6] Saved 24h feature weights -> '{cd.META_DIR}' and '{OUTPUT_DIR}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
