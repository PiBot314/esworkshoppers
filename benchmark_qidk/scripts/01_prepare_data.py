#!/usr/bin/env python3
"""Build the test and calibration windows for the teammate models in ../common, using their own data code.

Uses common/common_data.py unchanged, pointed at aimet_quantization/data/lstm_normalized (the same normalized
data the models were trained on). Writes:
  data/test/NNNNN.raw     float32 [96, 58] windows of the test split (window i starts at test row i)
  data/calib/NNNNN.raw    512 windows spread evenly over the training split (QNN calibration)
  data/{test,calib}_list.txt, data/test_list_h96.txt (the 96-step models have fewer test windows)
  data/y_test_h4.npy, data/y_test_h96.npy, data/X_test_last.npy (last row of each window, LightGBM's input)
  data/info.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.dirname(HERE)
FINAL = os.path.dirname(B)
sys.path.insert(0, os.path.join(FINAL, "common"))
import common_data as cd  # noqa: E402

cd.DATA_DIR = os.path.join(FINAL, "aimet_quantization", "data", "lstm_normalized")
OUT = os.path.join(B, "data")
N_CALIB = 512


def windows(X, start, n):
    return np.stack([X[start + i: start + i + cd.LOOKBACK] for i in range(n)]).astype(np.float32)


def write_raw(arr, sub):
    d = os.path.join(OUT, sub)
    os.makedirs(d, exist_ok=True)
    paths = []
    for i, w in enumerate(arr):
        p = os.path.join(d, f"{i:05d}.raw")
        w.tofile(p)
        paths.append(f"{sub}/{i:05d}.raw")
    return paths


def main():
    X_df, y, scal = cd.load_and_merge_all_sensors()
    # The teammate's parquet files list the SL-PL00-00 / SL-VN95-00 columns in a different order than ours.
    # Use their feature order (saved next to the LightGBM model), which is what all their models expect.
    order = json.load(open(os.path.join(FINAL, "common", "lgbm_feature_weights_24h.json")))["feature_names"]
    assert set(order) == set(X_df.columns)
    X_df = X_df[order]
    X = X_df.values.astype(np.float32)
    n_train, n_val = cd.split_bounds(len(X))
    t0 = n_train + n_val
    Xt, yt = X[t0:], y[t0:]

    n4 = len(Xt) - cd.LOOKBACK - 4 + 1
    n96 = len(Xt) - cd.LOOKBACK - 96 + 1
    test = windows(Xt, 0, n4)
    y4 = cd.build_targets(yt, cd.LOOKBACK, 4)
    y96 = cd.build_targets(yt, cd.LOOKBACK, 96)
    assert len(y4) == n4 and len(y96) == n96

    n_tr_win = n_train - cd.LOOKBACK - 96 + 1
    starts = np.linspace(0, n_tr_win - 1, N_CALIB).astype(int)
    calib = np.stack([X[s: s + cd.LOOKBACK] for s in starts]).astype(np.float32)

    os.makedirs(OUT, exist_ok=True)
    tp = write_raw(test, "test")
    cp = write_raw(calib, "calib")
    open(os.path.join(OUT, "test_list.txt"), "w").write("\n".join(tp) + "\n")
    open(os.path.join(OUT, "test_list_h96.txt"), "w").write("\n".join(tp[:n96]) + "\n")
    open(os.path.join(OUT, "calib_list.txt"), "w").write("\n".join(os.path.join(OUT, p) for p in cp) + "\n")
    np.save(os.path.join(OUT, "y_test_h4.npy"), y4)
    np.save(os.path.join(OUT, "y_test_h96.npy"), y96)
    np.save(os.path.join(OUT, "X_test_last.npy"), test[:, -1, :])
    np.save(os.path.join(OUT, "X_test.npy"), test)
    info = {"features": list(X_df.columns), "n_test_h4": n4, "n_test_h96": n96, "n_calib": N_CALIB,
            "capacities": cd.capacities(scal), "test_start": str(X_df.index[t0]), "data_dir": cd.DATA_DIR}
    json.dump(info, open(os.path.join(OUT, "info.json"), "w"), indent=1)
    print(f"test windows: {n4} (h4) / {n96} (h96), calib {N_CALIB}, features {X.shape[1]}, test from {info['test_start']}")
    print("capacities W:", [round(c) for c in info["capacities"]])


if __name__ == "__main__":
    main()
