#!/usr/bin/env python3
"""Run each teammate ONNX model (FP32, onnxruntime on the PC) on our rebuilt test split and compare
the metrics with the ones the teammate reported in common/*_meta_*.json. Also writes the FP32 reference
outputs (data/ref_<model>.npy) that the device results are compared against."""
import json
import os
import sys

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.dirname(HERE)
FINAL = os.path.dirname(B)
sys.path.insert(0, os.path.join(FINAL, "common"))
import common_data as cd  # noqa: E402

D = os.path.join(B, "data")
C = os.path.join(FINAL, "common")
MODELS = json.load(open(os.path.join(HERE, "models.json")))


def run(onnx_path, X):
    s = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    return np.concatenate([s.run(None, {name: X[i:i + 1]})[0] for i in range(len(X))])


def main():
    info = json.load(open(os.path.join(D, "info.json")))
    caps = info["capacities"]
    X = np.load(os.path.join(D, "X_test.npy"))
    rows = []
    for name, m in MODELS.items():
        H = m["horizon"]
        y = np.load(os.path.join(D, f"y_test_h{H}.npy"))
        pred = run(os.path.join(FINAL, m["onnx"]), X[:len(y)])
        np.save(os.path.join(D, f"ref_{name}.npy"), pred)
        pred = np.clip(pred, 0, None)            # the teammate's predict() clips at 0
        ours = cd.evaluate(pred, y, caps, H)
        theirs = {}
        if m.get("meta"):
            theirs = json.load(open(os.path.join(FINAL, m["meta"])))["evaluation_summary"]
        for dom in cd.DOMAINS:
            rows.append((name, dom, ours[dom]["R2"], theirs.get(dom, {}).get("R2"), ours[dom]["nMAE_pct"],
                         theirs.get(dom, {}).get("nMAE_pct")))
    print(f"{'model':<14s}{'domain':<8s}{'R2 ours':>9s}{'R2 theirs':>11s}{'nMAE ours':>11s}{'nMAE theirs':>13s}")
    for r in rows:
        f = lambda v, p: "-" if v is None else f"{v:.{p}f}"
        print(f"{r[0]:<14s}{r[1]:<8s}{f(r[2],4):>9s}{f(r[3],4):>11s}{f(r[4],2):>11s}{f(r[5],2):>13s}")


if __name__ == "__main__":
    main()
