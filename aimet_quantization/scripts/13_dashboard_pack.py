"""F1: data pack for the QIDK demo dashboard -> dashboard/www/data/<model>/

Test windows are consecutive slices of one timeline, so the pack stores the timeline once
(n_test + 95 rows) instead of n_test overlapping windows:  window i = timeline[i : i + 96].

Files per model (float32 little-endian, row-major):
  timeline.f32   [n_test + 95, F]  normalized inputs (exactly what the model sees)
  y.f32          [n_test, H*D]     actual targets, normalized with the *data* capacities
  npu.f32        [n_test, H*D]     recorded QIDK HTP outputs (AIMET W8A16), normalized with the *model* capacities
  fp32.f32       [n_test, H*D]     FP32 ONNX outputs on the same windows
  meta.json      shapes, feature names, scalers (physical units), capacities, thresholds, timestamps, stats
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import AQ, DEPLOY, MODELS, OUT, WINDOWS, data_capacities, model_capacities  # noqa: E402

WWW = os.path.join(AQ, "dashboard", "www", "data")
SCALERS = json.load(open(os.path.join(AQ, "data", "lstm_normalized", "scaler_params.json")))
CYCLIC = {"hod_sin", "hod_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"}


def feature_info(name):
    if name == "single_node":
        cols = json.load(open(os.path.join(DEPLOY, "qidk_model_meta.json")))["feature_order"]
        nodes = ["SL-PL00-00"] * len(cols)
        names = cols
    else:
        names = json.load(open(os.path.join(DEPLOY, "all_sensors_qidk_meta.json")))["feature_names"]
        nodes, cols = [], []
        for f in names:
            n, c = (f.split("__") + [None])[:2] if "__" in f else ("SL-PL00-00", f)   # cyclic taken from SL-PL00-00
            nodes.append(n)
            cols.append(c)
    out = []
    for f, n, c in zip(names, nodes, cols):
        if c == "was_imputed":
            lo, hi = 0.0, 1.0                                   # kept unscaled by generate_lstm_dataset
        else:
            s = SCALERS[n][c]
            lo, hi = s["min"], s["max"]
        out.append(dict(name=f, node=n, col=c, min=lo, max=hi, cyclic=c in CYCLIC))
    return out


def main():
    summary = {}
    for name, spec in MODELS.items():
        d = os.path.join(WWW, name)
        os.makedirs(d, exist_ok=True)
        X = np.load(os.path.join(WINDOWS, name, "X_test.npy"))
        y = np.load(os.path.join(WINDOWS, name, "y_test.npy"))
        Xv = np.load(os.path.join(WINDOWS, name, "X_val.npy"))
        yv = np.load(os.path.join(WINDOWS, name, "y_val.npy"))
        info = json.load(open(os.path.join(WINDOWS, name, "info.json")))
        timeline = np.concatenate([X[0], X[1:, -1]], axis=0).astype(np.float32)
        npu = np.load(os.path.join(OUT, name, "device", "htp_aimet_w8a16_out_test.npy")).astype(np.float32)
        fp32 = np.load(os.path.join(OUT, name, "fp32", "out_test.npy")).astype(np.float32)
        assert np.array_equal(timeline[5:5 + 96], X[5]) and npu.shape == fp32.shape == y.shape
        for fn, arr in (("timeline", timeline), ("y", y.astype(np.float32)), ("npu", npu), ("fp32", fp32)):
            arr.tofile(os.path.join(d, f"{fn}.f32"))

        D, H = len(spec["domains"]), spec["horizon"]
        mcaps, dcaps = model_capacities(name), data_capacities(name)
        thr = np.load(os.path.join(OUT, name, "anomaly_thresholds_W.npy"))

        # typical day: actual first-step value per IST hour over val + test (Watts)
        import anomaly
        hrs = np.concatenate([anomaly.target_hour_ist(Xv, name), anomaly.target_hour_ist(X, name)])
        act = np.concatenate([yv, y]).reshape(-1, H, D)[:, 0, :] * np.asarray(dcaps)
        typical = {q: [[float(np.percentile(act[hrs == h, j], q)) if np.any(hrs == h) else 0.0 for h in range(24)]
                       for j in range(D)] for q in (10, 50, 90)}

        dev = json.load(open(os.path.join(OUT, name, "device_results.json")))
        lat = json.load(open(os.path.join(OUT, "latency_summary.json")))[name]
        meta = dict(
            model=name, domains=spec["domains"], horizon=H, lookback=96, n=len(X), features=feature_info(name),
            input_name=("input_sequence" if name == "single_node" else
                        "input_sequence_all_sensors" + ("_24h" if name == "all_sensors_24h" else "")),
            model_capacities_W=mcaps, data_capacities_W=dcaps,
            timeline_start_utc=info["test_start"], step_minutes=15,
            anomaly=dict(K=anomaly.K, floor_frac=anomaly.FLOOR_FRAC, median_W=thr[0].tolist(), limit_W=thr[1].tolist()),
            typical_W=typical,
            device=dict(sqnr_db=dev["htp_aimet_w8a16"]["vs_fp32"]["SQNR_dB"],
                        qairt_only_sqnr_db=dev["htp_qairt_w8a16"]["vs_fp32"]["SQNR_dB"],
                        r2_fp32=dev["htp_aimet_w8a16"]["fp32_headline_R2"], r2_npu=dev["htp_aimet_w8a16"]["headline_R2"],
                        flag_agreement=dev["htp_aimet_w8a16"]["flag_agreement_natural"],
                        npu_ms=lat["htp_aimet_w8a16"]["accel_ms"], qnn_ms=lat["htp_aimet_w8a16"]["qnn_execute_ms"]),
            files={k: f"{k}.f32" for k in ("timeline", "y", "npu", "fp32")},
        )
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
        size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
        summary[name] = dict(rows=len(timeline), features=timeline.shape[1], n=len(X), MB=round(size / 1e6, 2))
        print(name, summary[name])
    json.dump(summary, open(os.path.join(WWW, "index.json"), "w"))


if __name__ == "__main__":
    main()
