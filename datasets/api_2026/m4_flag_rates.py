"""API calibration set + check M4 (natural grid flag rate on API data) for every model in the live registry.
plan_20112026.md "B1 leftovers"; archive/plan_model_switcher.md §4 (M4: grid flag rate <= 3 %).

1. Calibration set: every 15-min bin of the API history (nodes/<node>/<date>.json.gz, 1 Jan 2026 -> last fetched day)
   for the nodes the live loop reads, built EXACTLY like the live loop (qidk_export/dashboard/src/lib/server/live/
   features.ts, mirrored by verify_api_lstm/verify.py): signed values -> 15-min UTC mean -> ffill(limit=4) +
   was_imputed -> abs() -> kW to W; inputs the API doesn't serve estimated from a neighbour as live ("proxy").
   -> api_calib_15min.parquet  (frames as node__column, UTC index) — the data the live system actually sees.
2. M4: every registered model (FP32 ONNX; the NPU builds match FP32 to 44-52 dB) forecasts +15 min for every window
   whose current and next substation bin exist; flag <=> |actual - forecast - median[h]| > limit[h] with the model's
   own per-IST-hour limits (fitted on 2025 validation data). Grid actual = SB00 gd_power + KH95 total, as live.
   -> m4_results.json (overall, daytime, per month, grid R2 on API data, pass/fail at 3 %)

The API data are synthetic (see README): these numbers calibrate the LIVE system's behaviour, not model accuracy.
Run from anywhere:  python3 datasets/api_2026/m4_flag_rates.py
"""
import glob, gzip, json, sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent.parent
log = lambda *a: print(*a, flush=True)

num = lambda d, c: pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)
V3 = lambda d: {"active_power_kW": num(d, "active_power_3"), "reactive_power_kVAr": num(d, "reactive_power_3"),
                "voltage_avg_V": num(d, "voltage_avg"), "neutral_current_A": num(d, "nphaselc")}
SREM = lambda d: {"power_W": num(d, "power"), "voltage_V": num(d, "voltage"), "current_A": num(d, "current"),
                  "frequency_Hz": num(d, "frequency")}
MAP = {  # = features.ts MAP (first column = the node's target)
    "EM-CR-SB00-00": lambda d: {"gd_power_W": num(d, "gdpower"), "t1_power_W": num(d, "t1power"),
                                "t2_power_W": num(d, "t2power"), "hv_voltage_VLL": num(d, "gdvll"),
                                "lt1_voltage_VLN": num(d, "lt1vln"), "lt2_voltage_VLN": num(d, "lt2vln")},
    "EM-CR-KH95-00": lambda d: {"power_total_W": num(d, "r_power") + num(d, "y_power") + num(d, "b_power"),
                                "power_R_W": num(d, "r_power"), "power_B_W": num(d, "b_power"),
                                "voltage_V": num(d, "voltage"), "power_factor": num(d, "power_factor")},
    "SR-EM-KH04-01": SREM, "SR-EM-KH04-02": SREM,
    "SL-PL00-00": V3,
    "SL-NI01-00": lambda d: {"active_power_W": num(d, "active_power_3"), "reactive_power_VAr": num(d, "reactive_power_3"),
                             "voltage_avg_V": num(d, "voltage_avg"), "neutral_current_A": num(d, "nphaselc")},
    "SL-VN95-00": lambda d: dict(V3(d), active_power_kW=num(d, "reactive_power_3"), reactive_power_kVAr=num(d, "active_power_3")),
}
NEIGHBOUR = {"SL-NI03-00": "SL-NI01-00", "SL-NI03-01": "SL-NI01-00", "SL-VN02-01": "SL-VN95-00"}
MODELS = {  # registry id -> (FP32 ONNX, meta.json)
    "all_sensors_1h": ("model2/qidk_deployment/all_sensors_lstm_qidk.onnx",
                       "aimet_quantization/dashboard/www/data/all_sensors_1h/meta.json"),
    "baseline_h4": ("common/baseline_lstm_qidk_h4.onnx", "nz_live/models/baseline_h4/meta.json"),
    "guided_h4": ("common/guided_lstm_qidk_h4.onnx", "nz_live/models/guided_h4/meta.json"),
    "baseline_24h": ("common/baseline_lstm_qidk_24h.onnx", "nz_live/models/baseline_24h/meta.json"),
    "guided_24h": ("common/guided_lstm_qidk_24h.onnx", "nz_live/models/guided_24h/meta.json"),
    "cnn_lstm_24h": ("cnn_lstm_hybrid/cnn_lstm_qidk_24h.onnx", "nz_live/models/cnn_lstm_24h/meta.json"),
}
M4_MAX = 0.03


def load_node(node):
    rows = []
    for f in sorted(glob.glob(str(HERE / "nodes" / node / "*.json.gz"))):
        rows += json.load(gzip.open(f, "rt"))
    d = pd.DataFrame(rows)
    d.index = pd.to_datetime(d["created_at"], utc=True)
    return d.sort_index()


def build_frames():
    raws = {n: load_node(n) for n in MAP}
    lo = min(r.index.min() for r in raws.values()).floor("15min")
    hi = max(r.index.max() for r in raws.values()).floor("15min")
    idx = pd.date_range(lo, hi, freq="15min")
    frames = {}
    for node, fn in MAP.items():
        raw = raws[node]
        f = pd.DataFrame(fn(raw), index=raw.index).resample("15min").mean().reindex(idx)
        target = f.columns[0]
        miss = f[target].isna()
        f = f.ffill(limit=4)
        f["was_imputed"] = (miss & f[target].notna()).astype(int)
        for c in f.columns:
            if c != "was_imputed":
                f[c] = f[c].abs()
        for kw, w in (("active_power_kW", "active_power_W"), ("reactive_power_kVAr", "reactive_power_VAr")):
            if kw in f.columns:
                f[w] = f.pop(kw) * 1000.0
        frames[node] = f
        log(f"  {node:14s} {len(raw):6d} readings, target missing after ffill {f[f.columns[0]].isna().mean():6.1%}")
    return idx, frames


def features(meta, idx, frames):
    ist = idx.tz_convert("Asia/Kolkata")
    hod = ist.hour + ist.minute / 60.0
    cyc = {}
    for name, val, p in (("hod", hod, 24.0), ("dow", ist.dayofweek, 7.0), ("doy", ist.dayofyear, 365.25)):
        cyc[f"{name}_sin"] = np.sin(2 * np.pi * np.asarray(val) / p)
        cyc[f"{name}_cos"] = np.cos(2 * np.pi * np.asarray(val) / p)
    cap = lambda n: next(x["max"] for x in meta["features"] if x["name"] == n)
    cols = []
    for f in meta["features"]:
        node, col = f["node"], f["col"]
        if f["cyclic"]:
            s = pd.Series(cyc[col], index=idx)
        elif node in frames and col in frames[node]:
            s = frames[node][col]
        elif col == "active_power_W" and node in NEIGHBOUR:
            peer = NEIGHBOUR[node]
            s = frames[peer]["active_power_W"] * cap(f"{node}__active_power_W") / cap(f"{peer}__active_power_W")
        elif col == "ac_voltage_avg_V":
            s = frames["SL-NI01-00"]["voltage_avg_V"]
        else:
            s = pd.Series(0.0, index=idx)
        cols.append(((s.fillna(0.0) - f["min"]) / (f["max"] - f["min"])).to_numpy(np.float32))
    return np.stack(cols, axis=1)


def r2(a, p):
    return float(1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2))


if __name__ == "__main__":
    log("building the API calibration set ...")
    idx, frames = build_frames()
    calib = pd.concat({n: f for n, f in frames.items()}, axis=1)
    calib.columns = [f"{n}__{c}" for n, c in calib.columns]
    grid = frames["EM-CR-SB00-00"]["gd_power_W"] + frames["EM-CR-KH95-00"]["power_total_W"].fillna(0)
    calib["grid_actual_W"] = grid
    calib.index.name = "time_utc"
    calib.to_parquet(HERE / "api_calib_15min.parquet")
    log(f"saved api_calib_15min.parquet: {calib.shape}, {idx[0]} .. {idx[-1]}")

    ok = grid.notna().to_numpy()
    ends = np.array([t for t in range(95, len(idx) - 1) if ok[t] and ok[t + 1]])
    nxt = ends + 1
    hours = idx[nxt].tz_convert("Asia/Kolkata").hour.to_numpy()
    months = idx[nxt].tz_convert("Asia/Kolkata").month.to_numpy()
    actual = grid.to_numpy()[nxt]
    day = (hours >= 8) & (hours <= 16)
    results = {"windows": int(len(ends)), "span_utc": [str(idx[0]), str(idx[-1])], "threshold": M4_MAX, "models": {}}
    for mid, (onnx, metap) in MODELS.items():
        meta = json.load(open(FINAL / metap))
        X = features(meta, idx, frames)
        sess = ort.InferenceSession(str(FINAL / onnx), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        D = len(meta["domains"])
        g = meta["domains"].index("Grid")
        capg = meta["model_capacities_W"][g]
        fc = np.empty(len(ends))
        for i, t in enumerate(ends):
            fc[i] = sess.run(None, {name: X[None, t - 95:t + 1]})[0].reshape(-1, D)[0, g] * capg
        med = np.asarray(meta["anomaly"]["median_W"])[hours, g]
        lim = np.asarray(meta["anomaly"]["limit_W"])[hours, g]
        flag = np.abs(actual - fc - med) > lim
        res = {"flag_rate": float(flag.mean()), "flag_rate_daytime_8_16": float(flag[day].mean()),
               "flag_rate_by_month": {int(m): round(float(flag[months == m].mean()), 4) for m in np.unique(months)},
               "grid_R2_api": r2(actual, fc), "grid_MAE_kW_api": float(np.mean(np.abs(actual - fc)) / 1000),
               "pass_M4": bool(flag.mean() <= M4_MAX)}
        # API-fitted limits: same rule as anomaly.py fit_thresholds (per IST hour: median, max(K*1.4826*MAD, floor)),
        # fitted on API residuals, 2-fold cross-fit by alternate weeks (fit on one half, flag the other)
        K, FLOOR = meta["anomaly"]["K"], meta["anomaly"]["floor_frac"]
        resid = actual - fc
        week = ((idx[nxt] - idx[0]).days // 7) % 2 == 0
        flag_api = np.zeros(len(resid), bool)
        catch = {"orig": {}, "api": {}}
        for tr in (week, ~week):
            m_h, l_h = np.zeros(24), np.zeros(24)
            for h in range(24):
                r = resid[tr & (hours == h)]
                m_h[h] = np.median(r)
                l_h[h] = max(K * 1.4826 * np.median(np.abs(r - m_h[h])), FLOOR * capg)
            te = ~tr
            flag_api[te] = np.abs(resid[te] - m_h[hours[te]]) > l_h[hours[te]]
            for kind, mult in (("+20%", 1.2), ("-40%", 0.6)):     # injected grid faults on the actual reading
                rf = actual[te] * mult - fc[te]
                catch["api"].setdefault(kind, []).append(np.abs(rf - m_h[hours[te]]) > l_h[hours[te]])
                catch["orig"].setdefault(kind, []).append(np.abs(rf - med[te]) > lim[te])
        res["api_fitted_limits"] = {
            "flag_rate": float(flag_api.mean()), "flag_rate_daytime_8_16": float(flag_api[day].mean()),
            "flag_rate_by_month": {int(m): round(float(flag_api[months == m].mean()), 4) for m in np.unique(months)},
            "pass_M4": bool(flag_api.mean() <= M4_MAX)}
        res["catch_rate_injected"] = {lim_set: {k: float(np.concatenate(v).mean()) for k, v in d.items()}
                                      for lim_set, d in catch.items()}
        results["models"][mid] = res
        a, c = res["api_fitted_limits"], res["catch_rate_injected"]
        log(f"{mid:15s} 2025 limits: flag {res['flag_rate']:6.2%} (day {res['flag_rate_daytime_8_16']:6.2%}) "
            f"M4 {'PASS' if res['pass_M4'] else 'FAIL'} catch +20% {c['orig']['+20%']:.0%} -40% {c['orig']['-40%']:.0%} | "
            f"API limits: flag {a['flag_rate']:6.2%} M4 {'PASS' if a['pass_M4'] else 'FAIL'} "
            f"catch +20% {c['api']['+20%']:.0%} -40% {c['api']['-40%']:.0%} | grid R2 {res['grid_R2_api']:.3f}")
    json.dump(results, open(HERE / "m4_results.json", "w"), indent=1)
    log("saved m4_results.json")
