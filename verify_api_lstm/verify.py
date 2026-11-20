"""Feed API data (synthetic 2026 replay) to the EXISTING deployed LSTMs (FP32 ONNX; the NPU builds match these to
44-52 dB) and apply the EXISTING anomaly thresholds. Question: do they work on API data without retraining?

Preprocessing mirrors aimet_quantization/scripts/01_rebuild_dataset.py + model2/generate_lstm_dataset.py:
15-min UTC mean -> ffill(limit=4) + was_imputed -> abs() -> kW->W -> fillna(0) -> min-max with the features' own
physical min/max -> IST cyclic features (min-max scaled too). Inputs not served by the API are filled per MODE.
"""
import glob, json, os, sys
import numpy as np, pandas as pd, onnxruntime as ort

FINAL = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "zeros"          # zeros | proxy
HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(FINAL, "aimet_quantization/dashboard/www/data")
DEPLOY = os.path.join(FINAL, "model2/qidk_deployment")

def load(node):
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "raw", f"{node}_*.json"))):
        rows += json.load(open(f))
    d = pd.DataFrame(rows)
    d.index = pd.to_datetime(d["created_at"], utc=True)
    return d.sort_index()

num = lambda d, c: pd.to_numeric(d[c], errors="coerce")
# Signed raw values, units as in the 2025 raw data (kW where the meter reports kW). Exactly like training,
# abs() and kW->W are applied AFTER the 15-min mean and ffill (model2/generate_lstm_dataset.py steps 1-2).
V3 = lambda d: {"active_power_kW": num(d, "active_power_3"), "reactive_power_kVAr": num(d, "reactive_power_3"),
                "voltage_avg_V": num(d, "voltage_avg"), "neutral_current_A": num(d, "nphaselc")}
MAP = {
    # API names the SR-EM fields correctly: the 2025 firmware transposition must NOT be re-applied.
    "SR-EM-KH04-01": lambda d: {"power_W": num(d, "power"), "voltage_V": num(d, "voltage"),
                                "current_A": num(d, "current"), "frequency_Hz": num(d, "frequency")},
    "EM-CR-KH95-00": lambda d: {"power_total_W": num(d, "r_power") + num(d, "y_power") + num(d, "b_power"),
                                "power_R_W": num(d, "r_power"), "power_B_W": num(d, "b_power"),
                                "voltage_V": num(d, "voltage"), "power_factor": num(d, "power_factor")},
    "EM-CR-SB00-00": lambda d: {"gd_power_W": num(d, "gdpower"), "t1_power_W": num(d, "t1power"),
                                "t2_power_W": num(d, "t2power"), "hv_voltage_VLL": num(d, "gdvll"),
                                "lt1_voltage_VLN": num(d, "lt1vln"), "lt2_voltage_VLN": num(d, "lt2vln")},
    "SL-PL00-00": V3,
    # this meter reports W, not kW (same as 2025)
    "SL-NI01-00": lambda d: {"active_power_W": num(d, "active_power_3"), "reactive_power_VAr": num(d, "reactive_power_3"),
                             "voltage_avg_V": num(d, "voltage_avg"), "neutral_current_A": num(d, "nphaselc")},
    # registers swapped on this meter (same rule as 2025)
    "SL-VN95-00": lambda d: dict(V3(d), active_power_kW=num(d, "reactive_power_3"), reactive_power_kVAr=num(d, "active_power_3")),
}
MAP["SR-EM-KH04-02"] = MAP["SR-EM-KH04-01"]

raws = {node: load(node) for node in MAP}
# One shared 15-min grid (UTC, left-labelled) from the first to the last reading of any node.
t_lo = min(r.index.min() for r in raws.values()).floor("15min")
t_hi = max(r.index.max() for r in raws.values()).floor("15min")
idx = pd.date_range(t_lo, t_hi, freq="15min")

frames = {}
for node, fn in MAP.items():
    raw = raws[node]
    f = pd.DataFrame(fn(raw), index=raw.index).resample("15min").mean().reindex(idx)
    target = list(f.columns)[0]
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

# Inputs the API doesn't serve
CAP = lambda m, n: next(x["max"] for x in m["features"] if x["name"] == n)
def missing_value(meta, node, col):
    if MODE == "proxy" and col == "active_power_W" and node in ("SL-NI03-00", "SL-NI03-01", "SL-VN02-01"):
        peer = "SL-NI01-00" if node.startswith("SL-NI") else "SL-VN95-00"
        return frames[peer]["active_power_W"] * CAP(meta, f"{node}__active_power_W") / CAP(meta, f"{peer}__active_power_W")
    if MODE == "proxy" and col in ("ac_voltage_avg_V",):
        return frames["SL-NI01-00"]["voltage_avg_V"]
    return pd.Series(0.0, index=idx)

def cyclic(ix):
    ist = ix.tz_convert("Asia/Kolkata")
    hod = ist.hour + ist.minute / 60.0
    out = {}
    for name, val, p in (("hod", hod, 24.0), ("dow", ist.dayofweek, 7.0), ("doy", ist.dayofyear, 365.25)):
        out[f"{name}_sin"], out[f"{name}_cos"] = np.sin(2 * np.pi * np.asarray(val) / p), np.cos(2 * np.pi * np.asarray(val) / p)
    return pd.DataFrame(out, index=ix)

cyc = cyclic(idx)

def build(meta, single=False):
    cols, used_missing = [], set()
    for f in meta["features"]:
        node, col = (f["node"], f["col"])
        if f["cyclic"]:
            s = cyc[col]
        elif node in frames:
            s = frames[node][col]
        else:
            used_missing.add(node)
            s = 0.0 if col == "was_imputed" else missing_value(meta, node, col)
        s = pd.Series(s, index=idx).fillna(0.0)
        cols.append(((s - f["min"]) / (f["max"] - f["min"])).to_numpy(np.float32))
    return np.stack(cols, axis=1), sorted(used_missing)

def run(model, onnx):
    meta = json.load(open(os.path.join(W, model, "meta.json")))
    X, miss = build(meta)
    if os.environ.get("DUMP"):   # for the TypeScript port's self-test (live_selftest.ts)
        np.save(os.path.join(HERE, f"X_{model}_{MODE}.npy"), X)
        json.dump({"t0": int(idx[0].timestamp()), "n": len(idx)}, open(os.path.join(HERE, "grid.json"), "w"))
    sess = ort.InferenceSession(os.path.join(DEPLOY, onnx), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    cap = np.asarray(meta["model_capacities_W"])
    D = len(meta["domains"])
    # actuals (W) per 15-min bin, same definitions as train_all_sensors_lstm.py / the single-node model
    if D == 1:
        act = frames["SL-PL00-00"]["active_power_W"].fillna(0).to_numpy()[:, None]
    else:
        solar_served = sum(frames[n]["active_power_W"].fillna(0) for n in ("SL-PL00-00", "SL-VN95-00", "SL-NI01-00"))
        grid = frames["EM-CR-SB00-00"]["gd_power_W"].fillna(0) + frames["EM-CR-KH95-00"]["power_total_W"].fillna(0)
        dem = frames["SR-EM-KH04-01"]["power_W"].fillna(0) + frames["SR-EM-KH04-02"]["power_W"].fillna(0)
        act = np.stack([solar_served.to_numpy(), grid.to_numpy(), dem.to_numpy()], axis=1)
    med, lim = np.asarray(meta["anomaly"]["median_W"]), np.asarray(meta["anomaly"]["limit_W"])
    ok = ~frames["EM-CR-SB00-00"]["gd_power_W"].isna().to_numpy()
    rows = []
    for t in range(95, len(idx) - 1):
        if not (ok[t] and ok[t + 1]):
            continue
        out = sess.run(None, {name: X[None, t - 95:t + 1]})[0].reshape(-1, D)[0] * cap   # +15 min, W
        hr = (idx[t + 1].tz_convert("Asia/Kolkata")).hour
        r = act[t + 1] - out
        rows.append((idx[t + 1], hr, out, act[t + 1], np.abs(r - med[hr]) > lim[hr]))
    return meta, miss, rows

def summarise(model, onnx):
    meta, miss, rows = run(model, onnx)
    P = np.array([r[2] for r in rows]); A = np.array([r[3] for r in rows]); F = np.array([r[4] for r in rows])
    hrs = np.array([r[1] for r in rows])
    print(f"\n== {model} ({MODE}); windows scored: {len(rows)}; inputs not served by API: {miss or 'none'}")
    for d, dom in enumerate(meta["domains"]):
        p, a = P[:, d], A[:, d]
        r2 = 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)
        day = (hrs >= 8) & (hrs <= 16)
        print(f"  {dom:10s} MAE {np.mean(np.abs(a-p))/1000:8.1f} kW  R2 {r2:6.3f}  mean actual {a.mean()/1000:7.1f} kW  "
              f"mean forecast {p.mean()/1000:7.1f} kW  flag rate {100*F[:, d].mean():5.1f} % (daytime {100*F[day, d].mean():5.1f} %)")
    return rows

if __name__ == "__main__":
    print("bins:", len(idx), idx[0], "..", idx[-1])
    for n, f in frames.items():
        print(f"  {n:14s} missing bins after ffill: {int(f.iloc[:, 0].isna().sum()):4d}  imputed: {int(f['was_imputed'].sum()):3d}")
    r1 = summarise("all_sensors_1h", "all_sensors_lstm_qidk.onnx")
    r2 = summarise("single_node", "SL-PL00-00_lstm_qidk.onnx")
    json.dump({"all_sensors_1h": [(str(t), h, p.tolist(), a.tolist(), f.tolist()) for t, h, p, a, f in r1],
               "single_node": [(str(t), h, p.tolist(), a.tolist(), f.tolist()) for t, h, p, a, f in r2]},
              open(os.path.join(HERE, f"result_{MODE}.json"), "w"))
