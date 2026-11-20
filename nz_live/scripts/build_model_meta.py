"""Build meta.json (features, capacities, per-hour anomaly limits) for the common/ campus models, so the live loop
can switch to them (plan_model_switcher.md §3.3, archive/plan_integrated.md W2).

  ../project/.venv/bin/python nz_live/scripts/build_model_meta.py        (from final/)

For each model:
  * features: the model's own training order (common_data.py: features.json node order, parquet column order,
    then the 6 time features), with min/max from scaler_params.json (identical to the deployed models' meta);
  * anomaly: per IST hour, median residual and K * 1.4826 * MAD (floor 2 % of capacity) of the +15-min residuals on
    the VALIDATION split, the same rule as aimet_quantization/scripts/anomaly.py fit_thresholds();
  * metrics: R2 / nMAE of the first step on validation, and the test-split numbers from the training meta;
  * check: one validation window + its FP32 output, to compare the board's NPU against (SQNR).
Output: nz_live/models/<id>/{meta.json, check_window.raw, check_fp32.json}, nz_live/models/registry.json
"""
import json, os, sys
import numpy as np
import onnxruntime as ort

FINAL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(FINAL, "common"))
import common_data as cd  # noqa: E402

cd.DATA_DIR = os.path.join(FINAL, "aimet_quantization", "data", "lstm_normalized")
OUT = os.path.join(FINAL, "nz_live", "models")
K, FLOOR = 4.0, 0.02
DEPLOYED = json.load(open(os.path.join(FINAL, "aimet_quantization", "dashboard", "www", "data", "all_sensors_1h", "meta.json")))
BYNAME = {f["name"]: f for f in DEPLOYED["features"]}

MODELS = {
    "baseline_h4": ("common/baseline_lstm_qidk_h4.onnx", "common/baseline_lstm_meta_h4.json", 4, "lstm", "LSTM (common setup)"),
    "guided_h4": ("common/guided_lstm_qidk_h4.onnx", "common/guided_lstm_meta_h4.json", 4, "lstm-guided", "Guided LSTM (LightGBM-gated)"),
    "baseline_24h": ("common/baseline_lstm_qidk_24h.onnx", "common/baseline_lstm_meta_24h.json", 96, "lstm", "LSTM (common setup)"),
    "guided_24h": ("common/guided_lstm_qidk_24h.onnx", "common/guided_lstm_meta_24h.json", 96, "lstm-guided", "Guided LSTM (LightGBM-gated)"),
    "cnn_lstm_24h": ("cnn_lstm_hybrid/cnn_lstm_qidk_24h.onnx", None, 96, "cnn-lstm", "CNN-LSTM hybrid"),
}


def main():
    X_df, y, scal = cd.load_and_merge_all_sensors()
    data_caps = np.asarray(cd.capacities(scal))  # what the normalised targets were divided by (rebuilt data)
    assert np.allclose(data_caps, DEPLOYED["data_capacities_W"]), "data capacities differ from the deployed models'"
    # The teammate's models expect their own feature order (saved with the LightGBM model), not the parquet order;
    # benchmark_qidk/scripts/01_prepare_data.py does the same.
    order = json.load(open(os.path.join(FINAL, "common", "lgbm_feature_weights_24h.json")))["feature_names"]
    assert set(order) == set(X_df.columns)
    X_df_all = X_df
    X_df = X_df[order]
    names = list(X_df.columns)
    features = [BYNAME[n] for n in names]  # KeyError here = a feature the live loop can't build
    X = X_df.values.astype(np.float32)
    n_tr, n_val = cd.split_bounds(len(X))
    L = cd.LOOKBACK
    idx = X_df.index
    registry = []
    for mid, (onnx_path, meta_path, H, family, label) in MODELS.items():
        sess = ort.InferenceSession(os.path.join(FINAL, onnx_path))
        inp = sess.get_inputs()[0].name
        T, P = predict(sess, inp, X, y, n_tr, n_val, H)
        Tt, Pt = predict(sess, inp, X, y, n_tr + n_val, len(X) - n_tr - n_val, H)
        Xv = X[n_tr:n_tr + n_val]
        N = len(T)
        tmeta = json.load(open(os.path.join(FINAL, meta_path))) if meta_path else None
        test = tmeta["evaluation_summary"] if tmeta else None
        # outputs in W with the data capacities, as benchmark_qidk does (its R2 reproduce the training meta's)
        cap = data_caps
        res = T[:, 0] * data_caps - P[:, 0] * cap  # actual - forecast, W, +15 min
        tgt = idx[n_tr + L: n_tr + L + N]
        hours = ((tgt.tz_localize(None) if tgt.tz is not None else tgt) + np.timedelta64(330, "m")).hour.values
        med = np.zeros((24, 3)); lim = np.zeros((24, 3))
        for h in range(24):
            r = res[hours == h]
            med[h] = np.median(r, axis=0)
            lim[h] = np.maximum(K * 1.4826 * np.median(np.abs(r - med[h]), axis=0), FLOOR * cap)
        flag_rate = float(np.mean(np.abs(res - med[hours]) > lim[hours], axis=0)[1])
        r2 = first_step_r2(T, P, data_caps, cap)
        r2_test = first_step_r2(Tt, Pt, data_caps, cap)
        d = os.path.join(OUT, mid); os.makedirs(d, exist_ok=True)
        meta = {
            "model": mid, "label": label, "family": family, "domains": cd.DOMAINS, "horizon": H, "lookback": L,
            "features": features, "input_name": inp, "model_capacities_W": list(map(float, cap)),
            "anomaly": {"K": K, "floor_frac": FLOOR, "median_W": med.tolist(), "limit_W": lim.tolist(),
                        "fitted_on": "2025 validation split, +15 min residuals (FP32)"},
            "metrics": {"val_first_step_R2": r2, "test_first_step_R2": r2_test, "val_grid_flag_rate": round(flag_rate, 4), "test_all_steps_training_meta": test},
            "source_onnx": onnx_path,
        }
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        k = N // 2  # a midday-ish validation window for the NPU check
        Xv[k:k + L].astype(np.float32).tofile(os.path.join(d, "check_window.raw"))
        json.dump({"index": int(k), "out": P[k].reshape(-1).tolist()}, open(os.path.join(d, "check_fp32.json"), "w"))
        extra = {"comparison_only": True, "note": "Grid R² below zero on the 2025 test split: shadow and comparison only"} if r2_test["Grid"] < 0 else {}
        registry.append({**extra, "id": mid, "label": f"{label}, {'1 h' if H <= 4 else '24 h'}", "family": family, "horizon": H, "signals": ["grid"], "runtime": "htp",
                         "features": len(features), "val_first_step_R2": r2, "test_first_step_R2": r2_test,
                         "val_grid_flag_rate": round(flag_rate, 4)})
        print(f"{mid:13s} H={H:2d} val R2 {r2} test R2 {r2_test} grid flag rate {flag_rate:.3%}")

    # the deployed 1 h model, scored the same way (its own feature order and capacities, as anomaly.py does)
    X2 = X_df_all[[f["name"] for f in DEPLOYED["features"]]].values.astype(np.float32)
    sess = ort.InferenceSession(os.path.join(FINAL, "model2", "qidk_deployment", "all_sensors_lstm_qidk.onnx"))
    inp = sess.get_inputs()[0].name
    mcap = np.asarray(DEPLOYED["model_capacities_W"])
    T, P = predict(sess, inp, X2, y, n_tr, n_val, 4)
    Tt, Pt = predict(sess, inp, X2, y, n_tr + n_val, len(X2) - n_tr - n_val, 4)
    r2, r2_test = first_step_r2(T, P, data_caps, mcap), first_step_r2(Tt, Pt, data_caps, mcap)
    registry.insert(0, {"id": "all_sensors_1h", "label": "Team LSTM, 1 h (deployed)", "family": "lstm", "horizon": 4,
                        "signals": ["grid"], "runtime": "htp", "features": 58, "val_first_step_R2": r2, "test_first_step_R2": r2_test})
    print(f"all_sensors_1h H= 4 val R2 {r2} test R2 {r2_test}")
    old = {m["id"]: m for m in json.load(open(os.path.join(OUT, "registry.json")))} if os.path.exists(os.path.join(OUT, "registry.json")) else {}
    for m in registry:  # keep on-device results from check_models_on_board.py
        for k in ("sqnr_db", "npu_ms"):
            if k in old.get(m["id"], {}):
                m[k] = old[m["id"]][k]
    json.dump(registry, open(os.path.join(OUT, "registry.json"), "w"), indent=1)


def predict(sess, inp, X, y, start, n, H):
    """Windows over X[start:start+n] -> (targets, predictions), both normalised, [N, H, 3]."""
    Xs, ys = X[start:start + n], y[start:start + n]
    T = cd.build_targets(ys, cd.LOOKBACK, H).reshape(-1, H, 3)
    P = np.stack([sess.run(None, {inp: Xs[i:i + cd.LOOKBACK][None]})[0].reshape(H, 3) for i in range(len(T))])
    return T, P


def first_step_r2(T, P, data_caps, cap):
    out = {}
    for d, dom in enumerate(cd.DOMAINS):
        t, p = T[:, 0, d] * data_caps[d], P[:, 0, d] * cap[d]
        out[dom] = round(float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)), 4)
    return out


if __name__ == "__main__":
    main()
