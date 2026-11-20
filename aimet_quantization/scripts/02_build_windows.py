"""P1: turn data/lstm_normalized into model-ready windows, using model2's own merge/window code.

Outputs per model in data/windows/<model>/:
  X_{train,val,test}.npy, y_{test,val}.npy (normalized targets), calib.npy (stratified subset of train),
  info.json (sizes, capacities, time range)
And a fidelity report comparing the rebuilt first test window with model2's saved sample window.
"""
import json
import os
import sys
sys.dont_write_bytecode = True   # importing model2 modules must not leave __pycache__ in model2/ (read-only input)

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
DEPLOY = os.path.join(ROOT, "model2", "qidk_deployment")
sys.path.insert(0, os.path.join(ROOT, "model2"))

N_CALIB = 1024
SEED = 0


def split(n):
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def stratified_calib(X, hours, n, seed=SEED):
    """Pick ~n windows evenly across the hour-of-day of each window's last step (covers night/dawn/noon)."""
    rng = np.random.default_rng(seed)
    idx = []
    for h in range(24):
        pool = np.flatnonzero(hours == h)
        if len(pool):
            idx.extend(rng.choice(pool, size=min(len(pool), n // 24 + 1), replace=False))
    idx = np.sort(rng.choice(np.array(idx), size=min(n, len(idx)), replace=False))
    return X[idx]


def last_step_hour_ist(index, lookback, n_windows, offset):
    ist = index.tz_convert("Asia/Kolkata")
    return np.asarray(ist.hour[offset + lookback - 1: offset + lookback - 1 + n_windows])


def save(name, X_tr, X_va, X_te, y_va, y_te, hours_tr, extra):
    out = os.path.join(DATA, "windows", name)
    os.makedirs(out, exist_ok=True)
    calib = stratified_calib(X_tr, hours_tr, N_CALIB)
    for k, v in dict(X_val=X_va, X_test=X_te, y_val=y_va, y_test=y_te, calib=calib).items():
        np.save(os.path.join(out, f"{k}.npy"), v.astype(np.float32))
    info = dict(n_train=len(X_tr), n_val=len(X_va), n_test=len(X_te), n_calib=len(calib), **extra)
    with open(os.path.join(out, "info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    return X_te


def compare(name, rebuilt, sample_file, feature_names):
    ref = np.load(os.path.join(DEPLOY, sample_file))[0]
    d = np.abs(rebuilt - ref).max(axis=0)
    rows = sorted(zip(feature_names, d), key=lambda t: -t[1])
    print(f"\n[{name}] rebuilt X_test[0] vs {sample_file}: overall max|d|={d.max():.4f}, "
          f"features with max|d|<0.01: {(d < 0.01).sum()}/{len(d)}")
    for fname, v in rows[:15]:
        print(f"    {fname:36s} {v:.4f}")
    return {f: float(v) for f, v in zip(feature_names, d)}


def main():
    os.chdir(DATA)                               # model2 code reads ./lstm_normalized
    import train_all_sensors_lstm as t1h
    X_df, y, caps = t1h.load_and_merge_all_sensors()
    report = {"common_index": [str(X_df.index.min()), str(X_df.index.max()), len(X_df)], "capacities": caps}
    tr, va, te = split(len(X_df))
    # generate_lstm_dataset appends kW->W columns at the end, so per-node column order can differ from the
    # original run; the model only cares about the order recorded in its metadata.
    fn = json.load(open(os.path.join(DEPLOY, "all_sensors_qidk_meta.json")))["feature_names"]
    assert sorted(fn) == sorted(X_df.columns), "feature set differs from the deployed model's metadata"
    X_df = X_df[fn]
    X = X_df.values
    for name, horizon, sample in (("all_sensors_1h", 4, "sample_input_all_sensors_1x96x58.npy"),
                                  ("all_sensors_24h", 96, "sample_input_all_sensors_24h_1x96x58.npy")):
        parts = [t1h.create_multisensor_windows(X[s], y[s], 96, horizon) for s in (tr, va, te)]
        hours = last_step_hour_ist(X_df.index, 96, len(parts[0][0]), 0)
        X_te = save(name, parts[0][0], parts[1][0], parts[2][0], parts[1][1], parts[2][1], hours,
                    dict(horizon=horizon, capacities_W=caps, test_start=str(X_df.index[te.start])))
        report[name] = compare(name, X_te[0], sample, fn)

    # single node (train_and_export_qidk.py logic)
    import pandas as pd
    import train_and_export_qidk as ts
    node = "SL-PL00-00"
    fmeta = json.load(open("lstm_normalized/features.json"))["nodes"][node]
    cols = json.load(open(os.path.join(DEPLOY, "qidk_model_meta.json")))["feature_order"]
    df = pd.read_parquet(f"lstm_normalized/{node}.parquet")
    tr, va, te = split(len(df))
    parts = [ts.create_sliding_windows(df[cols].values[s], df[[fmeta["target"]]].values[s], 96, 4) for s in (tr, va, te)]
    hours = last_step_hour_ist(df.index, 96, len(parts[0][0]), 0)
    scal = json.load(open("lstm_normalized/scaler_params.json"))[node][fmeta["target"]]
    X_te = save("single_node", parts[0][0], parts[1][0], parts[2][0], parts[1][1], parts[2][1], hours,
                dict(horizon=4, feature_order=cols, scaler_min_W=scal["min"], scaler_max_W=scal["max"],
                     test_start=str(df.index[te.start])))
    report["single_node"] = compare("single_node", X_te[0], "sample_input_1x96.npy", cols)
    with open(os.path.join(DATA, "windows", "fidelity_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    main()
