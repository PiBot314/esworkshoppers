"""Offline results for the Results tab: every campus model AND the two simple baselines on the 2025 test split,
at +15 min, 1 h and 24 h, with skill against persistence; plus LightGBM's feature importance, grouped.

  ../project/.venv/bin/python nz_live/scripts/offline_results.py        (from final/)
Output: nz_live/models/offline_results.json (pushed next to registry.json on the board).

Skill = 1 - MSE(model) / MSE(persistence) on the same targets: > 0 means better than "next reading = current
reading" (for every step of the horizon). The baselines need no training, so they are what each model must beat.
"""
import json, os, sys
from collections import defaultdict
import numpy as np
import onnxruntime as ort

FINAL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(FINAL, "common"))
import common_data as cd  # noqa: E402

cd.DATA_DIR = os.path.join(FINAL, "aimet_quantization", "data", "lstm_normalized")
OUT = os.path.join(FINAL, "nz_live", "models", "offline_results.json")
DEPLOYED = json.load(open(os.path.join(FINAL, "aimet_quantization", "dashboard", "www", "data", "all_sensors_1h", "meta.json")))
L = cd.LOOKBACK
DOMS = ["Solar", "Grid"]
MODELS = [  # id, onnx, horizon, feature order ("lgbm" = teammate order, "deployed" = all_sensors_1h meta), output caps
    ("all_sensors_1h", "model2/qidk_deployment/all_sensors_lstm_qidk.onnx", 4, "deployed", "model"),
    ("baseline_h4", "common/baseline_lstm_qidk_h4.onnx", 4, "lgbm", "data"),
    ("guided_h4", "common/guided_lstm_qidk_h4.onnx", 4, "lgbm", "data"),
    ("baseline_24h", "common/baseline_lstm_qidk_24h.onnx", 96, "lgbm", "data"),
    ("guided_24h", "common/guided_lstm_qidk_24h.onnx", 96, "lgbm", "data"),
    ("cnn_lstm_24h", "cnn_lstm_hybrid/cnn_lstm_qidk_24h.onnx", 96, "lgbm", "data"),
    ("lightgbm_24h", "benchmark_qidk/models/lightgbm_24h.onnx", 96, "lgbm", "data"),
]


def r2(t, p):
    return float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))


def main():
    X_df, y, scal = cd.load_and_merge_all_sensors()
    data_caps = np.asarray(cd.capacities(scal))
    orders = {"lgbm": json.load(open(os.path.join(FINAL, "common", "lgbm_feature_weights_24h.json")))["feature_names"],
              "deployed": [f["name"] for f in DEPLOYED["features"]]}
    n_tr, n_val = cd.split_bounds(len(y))
    t0 = n_tr + n_val
    yw = y[t0:] * data_caps  # actual W, test split
    N = len(yw) - L - 96 + 1  # same windows for every horizon and model (the 24 h ones need 96 steps ahead)
    i0 = np.arange(N) + L     # row of the +15 min target of window i

    def targets(H):
        return np.stack([yw[i0 + h] for h in range(H)], 1)  # [N, H, 3]

    def score(P, H):
        T = targets(H)
        pers = np.repeat(yw[i0 - 1][:, None, :], H, 1)
        out = {}
        for d, dom in enumerate(DOMS):
            mse_p = np.mean((T[..., d] - pers[..., d]) ** 2)
            out[dom] = {"R2": round(r2(T[..., d], P[..., d]), 4),
                        "skill_vs_persistence": round(float(1 - np.mean((T[..., d] - P[..., d]) ** 2) / mse_p), 3)}
        return out

    horizons = {"15min": 1, "1h": 4, "24h": 96}
    res = {"split": f"2025 test split, {N} windows from {X_df.index[t0 + L]} UTC", "horizons": list(horizons), "rows": []}
    for name, f in (("persistence", lambda H: np.repeat(yw[i0 - 1][:, None, :], H, 1)),
                    ("yesterday", lambda H: np.stack([yw[i0 + h - 96] for h in range(H)], 1))):
        res["rows"].append({"id": name, "kind": "baseline", **{k: score(f(H), H) for k, H in horizons.items()}})

    for mid, onnx, H, order, capkind in MODELS:
        X = X_df[orders[order]].values.astype(np.float32)[t0:]
        sess = ort.InferenceSession(os.path.join(FINAL, onnx))
        inp = sess.get_inputs()[0]
        cap = np.asarray(DEPLOYED["model_capacities_W"]) if capkind == "model" else data_caps
        P = np.zeros((N, H, 3), np.float32)
        for i in range(N):
            x = X[i:i + L][None] if len(inp.shape) == 3 else X[i + L - 1][None]
            P[i] = sess.run(None, {inp.name: x})[0].reshape(H, 3)
        P = P * cap
        row = {"id": mid, "kind": "model", "horizon_steps": H}
        for k, Hk in horizons.items():
            if Hk <= H:
                row[k] = score(P[:, :Hk], Hk)
        res["rows"].append(row)
        print(mid, {k: row[k] for k in horizons if k in row})

    # LightGBM gain importance (also the gate of the guided LSTMs), grouped by node and by kind of quantity
    w = json.load(open(os.path.join(FINAL, "common", "lgbm_feature_weights_24h.json")))["feature_weight_map"]
    tot = sum(w.values())
    by_node, by_qty = defaultdict(float), defaultdict(float)
    for k, v in w.items():
        node, col = (k.split("__") + [""])[:2] if "__" in k else ("time of day/week/year", k)
        by_node[node] += v / tot
        q = ("time features" if "__" not in k else "imputed flag" if col == "was_imputed" else
             "power" if "power" in col else "voltage" if "voltage" in col else "current" if "current" in col else col)
        by_qty[q] += v / tot
    res["importance"] = {
        "source": "LightGBM 24 h, gain importance (normalised; the same weights gate the guided LSTMs' inputs). Not SHAP.",
        "by_node": sorted(([k, round(v, 4)] for k, v in by_node.items()), key=lambda x: -x[1]),
        "by_quantity": sorted(([k, round(v, 4)] for k, v in by_qty.items()), key=lambda x: -x[1]),
        "top_features": sorted(([k, round(v / tot, 4)] for k, v in w.items()), key=lambda x: -x[1])[:12],
    }
    json.dump(res, open(OUT, "w"), indent=1)
    print("baselines:", res["rows"][:2])


if __name__ == "__main__":
    main()
