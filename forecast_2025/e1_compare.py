"""E1: one fair comparison on identical rows, with confidence intervals and breakdowns (20 Nov).

Fixes the issues found after B2/B3 (reports/WRITEUP_2026-11-20.md):
  * B2 (LightGBM) and B3 (LSTM) were scored on different test rows -> here every method is scored on the SAME rows:
    the B3 test windows (origin and target in the same test week), last step (the value at +1 h / +24 h).
  * The deployed models were never scored on the 2025 test weeks -> they are, here, on (a) all 8 test weeks (they
    trained on most of them: favours them) and (b) the 3 test weeks their weights never saw (8-15 Oct, 29 Oct-5 Nov =
    their validation period, 24-31 Dec = their test period). Solar is compared on the campus total (what they
    forecast), grid on SB00.
  * Differences of 0.01 R2 were reported without uncertainty -> day-block bootstrap (2000 resamples of test days),
    95 % intervals for every method and for the paired differences that the conclusions rest on.
  * One average hid where weather matters -> breakdown by season, by sky (solar: daily clearness tercile of the
    target day) and by load regime (grid: weekday working hours, calendar days).

  python3 forecast_2025/e1_compare.py            -> results/e1_compare.{json,md}, results/e1_rows_<family>_<h>.parquet
"""
import json, sys, time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
sys.path.insert(0, str(HERE))
import b3_guided_lstm as B          # noqa: E402
import b2_lgbm_feature_test as B2   # noqa: E402

OUT = HERE / "results"
log = lambda *a: print(*a, flush=True)
HZ = {"1h": 4, "24h": 96}
OLD_DATA = FINAL / "aimet_quantization" / "data" / "lstm_normalized"
DEPLOYED = {   # registry id -> (FP32 ONNX, meta.json); the same files the live system and M4 use
    "1h": {"all_sensors_1h (live default)": ("model2/qidk_deployment/all_sensors_lstm_qidk.onnx",
                                             "aimet_quantization/dashboard/www/data/all_sensors_1h/meta.json"),
           "baseline_h4": ("common/baseline_lstm_qidk_h4.onnx", "nz_live/models/baseline_h4/meta.json"),
           "guided_h4": ("common/guided_lstm_qidk_h4.onnx", "nz_live/models/guided_h4/meta.json")},
    "24h": {"baseline_24h": ("common/baseline_lstm_qidk_24h.onnx", "nz_live/models/baseline_24h/meta.json"),
            "guided_24h": ("common/guided_lstm_qidk_24h.onnx", "nz_live/models/guided_24h/meta.json"),
            "cnn_lstm_24h": ("cnn_lstm_hybrid/cnn_lstm_qidk_24h.onnx", "nz_live/models/cnn_lstm_24h/meta.json")},
}
CLEAN_WEEKS = [("2025-10-08", "2025-10-15"), ("2025-10-29", "2025-11-05"), ("2025-12-24", "2025-12-31")]
WEATHER_P, WEATHER_C = {"ghi_obs", "temp_obs"}, {"ghi_fc", "cloud_fc", "temp_fc", "rh_fc"}
SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "summer", 4: "summer", 5: "summer", 6: "monsoon",
          7: "monsoon", 8: "monsoon", 9: "monsoon", 10: "post-monsoon", 11: "post-monsoon"}


# ----------------------------------------------------------------------------------------------- inputs
def series(df, family, H):
    return B.series_solar(df, H) if family == "solar" else B.series_grid(df, H)


def windows_for_model(ser, names_p, names_c, mj, H, n_nodes):
    """Windows with the columns (and zeroed weather) the model was trained with, by input name."""
    ip = [names_p.index(n) for n in mj["inputs"]["past"]]
    ic = [names_c.index(n) for n in mj["inputs"]["future"]]
    out = []
    for s in ser:
        P, C = s["P"][:, ip].copy(), s["C"][:, ic].copy()
        if mj["variant"].startswith("noweather"):
            P[:, [i for i, n in enumerate(mj["inputs"]["past"]) if n in WEATHER_P]] = 0
            C[:, [i for i, n in enumerate(mj["inputs"]["future"]) if n in WEATHER_C]] = 0
        out.append(dict(s, P=P, C=C))
    return B.Windows(out, H, n_nodes)


def run_onnx(path, W, t, bs=4096):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    o = []
    for i in range(0, len(t), bs):
        past, fut, oh, _, _ = W.batch(t[i:i + bs])
        o.append(sess.run(None, {"past": past.cpu().numpy(), "future": fut.cpu().numpy(),
                                 "node": oh.cpu().numpy()})[0][:, -1])
    return np.concatenate(o)


def lstm_models(family, hname):
    """(label, onnx path, meta) for every B3 model of this family/horizon (+ the archived v1 grid models)."""
    out = []
    for d, pre in ((HERE / "models", ""), (HERE / "models" / "v1_grid_noweek", "v1 (no week) ")):
        for j in sorted(d.glob(f"{family}_{hname}_*.json")):
            out.append((f"LSTM {pre}{json.load(open(j))['variant']}", j.with_suffix(".onnx"), json.load(open(j))))
    return out


def lgbm(df, family, H):
    """LightGBM 'no weather' (lags + time [+ calendar] + clear-sky) and 'forecast' configs, as in B2 (blocked)."""
    f, groups = B2.build(df, family, H)
    cat = ["plant"] if family == "solar" else None
    tr, va = (f["split"] == "train"), (f["split"] == "val")
    base = groups["base"] + (groups.get("calendar", [])) + groups["clearsky"]
    preds = {}
    for name, feats in (("LightGBM noweather", base), ("LightGBM forecast", base + groups["forecast"])):
        m = B2.fit(f.loc[tr, feats], f.loc[tr, "target"], f.loc[va, feats], f.loc[va, "target"], cat=cat)
        p = m.predict(f[feats], num_iteration=m.best_iteration)
        preds[name] = np.clip(p, 0, None) if family == "solar" else p
        log(f"    {name}: {m.best_iteration} rounds")
    key = pd.MultiIndex.from_arrays([f.index, f["plant"].to_numpy() if family == "solar" else np.zeros(len(f), int)])
    return pd.DataFrame(preds, index=key)


def old_inputs(meta, index):
    """The deployed models' 58 inputs from the dataset they were trained on, on our time index (bin END, IST)."""
    nodes = {}
    for f in meta["features"]:
        if f["node"] and f["node"] not in nodes:
            nodes[f["node"]] = pd.read_parquet(OLD_DATA / f"{f['node']}.parquet")
    if "SL-PL00-00" not in nodes:
        nodes["SL-PL00-00"] = pd.read_parquet(OLD_DATA / "SL-PL00-00.parquet")
    common = None
    for d in nodes.values():
        common = d.index if common is None else common.intersection(d.index)
    cols = []
    for f in meta["features"]:
        src = nodes["SL-PL00-00"] if f["cyclic"] or not f["node"] else nodes[f["node"]]
        cols.append(src.loc[common, f["col"]].to_numpy(np.float32))
    X = pd.DataFrame(np.stack(cols, 1), index=common)
    # old label = START of the bin (UTC); ours = END (IST naive): checked, SB00 identical after +15 min (r = 0.99999)
    X.index = X.index.tz_convert("Asia/Kolkata").tz_localize(None) + pd.Timedelta(minutes=15)
    return X.reindex(index).to_numpy(np.float32)


def deployed_preds(hname, H, index, origins):
    """{model: [n_origins, 3] (Solar, Grid, Demand) in W at the last step} for origin positions in `index`."""
    out = {}
    for name, (onnx, metap) in DEPLOYED[hname].items():
        meta = json.load(open(FINAL / metap))
        X = old_inputs(meta, index)
        sess = ort.InferenceSession(str(FINAL / onnx), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        caps = np.asarray(meta["model_capacities_W"])
        P = np.full((len(origins), 3), np.nan)
        t0 = time.time()
        for i, t in enumerate(origins):
            x = X[t - 95:t + 1]
            if t >= 95 and np.isfinite(x).all():
                P[i] = sess.run(None, {inp: x[None]})[0].reshape(H, 3)[-1] * caps
        out[name] = P
        log(f"    {name}: {np.isfinite(P[:, 0]).sum():,} origins ({time.time() - t0:.0f}s)")
    return out


# ----------------------------------------------------------------------------------------------- scoring
def r2(y, p):
    return float(1 - np.mean((y - p) ** 2) / np.var(y))


def boot(rows, methods, pairs, n=2000, seed=0):
    """Day-block bootstrap (resample test days by the TARGET's date) of R2 per method and of paired differences."""
    day = rows["T"].dt.normalize()
    codes, uniq = pd.factorize(day)
    D = len(uniq)
    y = rows["y"].to_numpy()
    agg = lambda v: np.bincount(codes, v, minlength=D)
    cnt, sy, syy = agg(np.ones_like(y)), agg(y), agg(y * y)
    sse = {m: agg((y - rows[m].to_numpy()) ** 2) for m in methods}
    rng = np.random.default_rng(seed)
    Wt = np.stack([np.bincount(rng.integers(0, D, D), minlength=D) for _ in range(n)]).astype(float)
    n_, s_, ss_ = Wt @ cnt, Wt @ sy, Wt @ syy
    sst = ss_ - s_ ** 2 / n_
    R = {m: 1 - (Wt @ sse[m]) / sst for m in methods}
    ci = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    res = {"days": int(D), "rows": int(len(rows)),
           "R2": {m: {"point": round(r2(y, rows[m].to_numpy()), 4), "ci95": ci(R[m])} for m in methods},
           "diff": {}}
    for a, b in pairs:
        if a in R and b in R:
            d = R[a] - R[b]
            pt = r2(y, rows[a].to_numpy()) - r2(y, rows[b].to_numpy())
            res["diff"][f"{a}  −  {b}"] = {"point": round(pt, 4), "ci95": ci(d),
                                           "p_le_0": round(float(np.mean(d <= 0)), 3)}
    return res


def breakdown(rows, methods, ref, by):
    """Per bucket: n, MSE skill vs `ref` (1 - MSE/MSE_ref) and nMAE for each method."""
    out = {}
    for k, g in rows.groupby(by):
        y = g["y"].to_numpy()
        mref = np.mean((y - g[ref].to_numpy()) ** 2)
        out[str(k)] = {"n": int(len(g)), "days": int(g["T"].dt.normalize().nunique()),
                       **{m: {"skill": round(float(1 - np.mean((y - g[m].to_numpy()) ** 2) / mref), 3),
                              "nMAE": round(float(np.mean(np.abs(y - g[m].to_numpy())) / np.mean(np.abs(y))), 3)}
                          for m in methods}}
    return out


def in_clean(T):
    m = np.zeros(len(T), bool)
    for a, b in CLEAN_WEEKS:
        m |= (T > pd.Timestamp(a)) & (T <= pd.Timestamp(b))
    return m


# ----------------------------------------------------------------------------------------------- per family
def family_rows(df, family, hname):
    """Main table: B3 test windows, last step, all methods on identical rows (solar per kWp on ok days)."""
    H = HZ[hname]
    ser, names_p, names_c, _, _ = series(df, family, H)
    n_nodes = len(B.PLANTS) if family == "solar" else 1
    W = B.Windows(ser, H, n_nodes)
    t = W.idx["test"]
    N = len(df)
    tn = t.cpu().numpy()
    node, loc = tn // N, tn % N
    rows = pd.DataFrame({"origin": df.index[loc], "T": df.index[loc + H], "node": node,
                         "y": W.Y[t + H].cpu().numpy(), "day": W.D[t + H].cpu().numpy()})
    for b, p in B.baseline_preds(W, t, H, family).items():
        rows[f"baseline {b}"] = p[:, -1]
    for label, path, mj in lstm_models(family, hname):
        Wm = windows_for_model(ser, names_p, names_c, mj, H, n_nodes)
        rows[label] = run_onnx(path, Wm, t)
        log(f"    {label}")
    lg = lgbm(df, family, H)
    key = pd.MultiIndex.from_arrays([rows["origin"], rows["node"]])
    for c in lg.columns:
        rows[c] = lg[c].reindex(key).to_numpy()
    if family == "grid":                                    # deployed models' grid output (SB00 + KH95 ~ 0.1-0.5 kW)
        dep = deployed_preds(hname, H, df.index, loc)
        for name, P in dep.items():
            rows[f"deployed {name}"] = P[:, 1] / 1e6
    rows = rows[np.isfinite(rows["y"]) & rows["day"]].drop(columns="day")
    return rows


def solar_total_rows(df, hname):
    """Campus solar total (sum of 6 plants, W): what the deployed models forecast. All plant days (any status),
    since the deployed models saw faults too; our per-plant models summed with their capacities."""
    H = HZ[hname]
    meta = json.load(open(B.DATA / "meta.json"))
    caps = np.array([meta["solar"][f"SL-{p.replace('_', '-')}"]["est_capacity_W"] for p in B.PLANTS])
    d2 = df.copy()
    for p in B.PLANTS:
        d2[f"solar_{p}_status"] = "ok"                     # use every day as input; the target is the real total
    ser, names_p, names_c, _, _ = series(d2, "solar", H)
    N = len(df)
    Y = np.stack([df[f"solar_{p}_W"].to_numpy() for p in B.PLANTS], 1)
    tot = Y.sum(1)                                          # NaN if any plant missing
    split = df["split_blocked"].to_numpy()
    cs = df["ghi_clearsky"].to_numpy()
    t = np.arange(95, N - H)
    keep = (split[t] == "test") & (split[t + H] == "test") & np.isfinite(tot[t]) & np.isfinite(tot[t + H]) & (cs[t + H] > 0)
    t = t[keep]
    rows = pd.DataFrame({"origin": df.index[t], "T": df.index[t + H], "y": tot[t + H] / 1e3})   # kW
    rows["baseline persistence"] = tot[t] / 1e3
    yd = tot[t + H - 96]
    rows["baseline same_time_yesterday"] = np.where(np.isfinite(yd), yd, tot[t]) / 1e3
    W = B.Windows(ser, H, len(B.PLANTS))
    tg = torch.tensor(np.concatenate([t + k * N for k in range(len(B.PLANTS))]), device=B.DEV)
    bp = B.baseline_preds(W, tg, H, "solar")["csi_persistence"][:, -1].reshape(len(B.PLANTS), -1)
    rows["baseline csi_persistence"] = (bp.T * caps).sum(1) / 1e3
    for label, path, mj in lstm_models("solar", hname):
        if "+pre" in mj["variant"]:
            continue
        Wm = windows_for_model(ser, names_p, names_c, mj, H, len(B.PLANTS))
        p = run_onnx(path, Wm, tg).reshape(len(B.PLANTS), -1)
        rows[label] = (p.T * caps).sum(1) / 1e3
    dep = deployed_preds(hname, H, df.index, t)
    for name, P in dep.items():
        rows[f"deployed {name}"] = np.clip(P[:, 0], 0, None) / 1e3
    return rows


# ----------------------------------------------------------------------------------------------- main
def summarize(rows, family, hname, kind):
    methods = [c for c in rows.columns if c.startswith(("baseline", "LSTM", "LightGBM", "deployed"))]
    rows = rows.dropna(subset=methods)
    bases = [m for m in methods if m.startswith("baseline")]
    ref = max(bases, key=lambda m: r2(rows["y"].to_numpy(), rows[m].to_numpy()))
    dep = [m for m in methods if m.startswith("deployed")]
    best_dep = max(dep, key=lambda m: r2(rows["y"].to_numpy(), rows[m].to_numpy())) if dep else None
    L = lambda v: f"LSTM {v}"
    pairs = [(L("forecast"), L("noweather")), ("LightGBM forecast", "LightGBM noweather"),
             (L("forecast"), "LightGBM forecast"), (L("forecast"), L("forecast_unguided")),
             (L("forecast"), L("forecast+pre")), (L("forecast+pre"), L("forecast+pre_unguided")),
             (L("forecast_skip"), L("forecast")), (L("forecast"), L("v1 (no week) forecast")),
             ("LightGBM forecast", ref), (L("forecast"), ref)]
    if best_dep:
        pairs += [(L("forecast"), best_dep), ("LightGBM forecast", best_dep), (best_dep, ref)]
    res = {"kind": kind, "best_baseline": ref, "best_deployed": best_dep, "all_test_weeks": boot(rows, methods, pairs)}
    if dep:
        clean = rows[in_clean(rows["T"])]
        res["clean_weeks"] = boot(clean, methods, pairs)
    rows = rows.assign(season=rows["T"].dt.month.map(SEASON))
    res["by_season"] = breakdown(rows, methods, ref, "season")
    return res, rows, methods, ref


def main():
    df = pd.read_parquet(B.DATA / "data_15min.parquet")
    res = {}
    for hname in HZ:
        for family in ("solar", "grid"):
            log(f"[{family} {hname}] per-plant / SB00 table")
            rows = family_rows(df, family, hname)
            r, rows, methods, ref = summarize(rows, family, hname, "solar per kWp, ok days" if family == "solar" else "grid SB00, MW")
            if family == "solar":                       # sky: clearness of the target day for that plant
                meta = json.load(open(B.DATA / "meta.json"))
                clr = {}
                for i, p in enumerate(B.PLANTS):
                    cap = meta["solar"][f"SL-{p.replace('_', '-')}"]["est_capacity_W"]
                    e = df[f"solar_{p}_W"].groupby(df.index.normalize()).sum(min_count=40)
                    c = (df["ghi_clearsky"] * cap / 1000).groupby(df.index.normalize()).sum()
                    clr[i] = e / c
                k = np.array([clr[n].get(d, np.nan) for n, d in zip(rows["node"], rows["T"].dt.normalize())])
                q = np.nanquantile(k, [1 / 3, 2 / 3])
                rows["sky"] = np.select([k < q[0], k < q[1]], ["1 cloudy", "2 mixed"], "3 clear")
                r["sky_thresholds_clearness"] = [round(float(x), 3) for x in q]
                r["by_sky"] = breakdown(rows, methods, ref, "sky")
            else:
                ist_h = (rows["T"] - pd.Timedelta(minutes=15)).dt.hour
                wd = rows["T"].dt.dayofweek < 5
                cal = df[["holiday", "exam", "event"]].reindex(rows["T"]).to_numpy().any(1)
                rows["regime"] = np.where(cal, "calendar day (holiday/exam/event)",
                                          np.where(wd & (ist_h >= 9) & (ist_h < 18), "weekday 09-18", "other hours"))
                r["by_regime"] = breakdown(rows, methods, ref, "regime")
                r["MAE_kW"] = {m: round(float(np.mean(np.abs(rows["y"] - rows[m])) * 1000), 1) for m in methods}
            res[f"{family}_{hname}"] = r
            rows.to_parquet(OUT / f"e1_rows_{family}_{hname}.parquet")
            json.dump(res, open(OUT / "e1_compare.json", "w"), indent=1)
        log(f"[solar total {hname}] vs deployed")
        rows = solar_total_rows(df, hname)
        r, rows, *_ = summarize(rows, "solar_total", hname, "campus solar total, kW, all days")
        res[f"solar_total_{hname}"] = r
        rows.to_parquet(OUT / f"e1_rows_solar_total_{hname}.parquet")
        json.dump(res, open(OUT / "e1_compare.json", "w"), indent=1)
    report(res)


def report(res):
    f3 = lambda x: f"{x:.3f}"
    lines = ["# E1: all methods on identical test rows (20 Nov)", "",
             "Built by [e1_compare.py](../e1_compare.py). R² at the horizon (last step). 95 % intervals: day-block "
             "bootstrap over test days (2000 resamples). *Clean weeks* = the 3 test weeks the deployed models' weights "
             "never saw (8–15 Oct, 29 Oct–5 Nov, 24–31 Dec). Seeds: the ONNX models are seed 0; seed spread is in "
             "b3_results.md.", ""]
    for key, r in res.items():
        a = r["all_test_weeks"]
        lines += [f"## {key} ({r['kind']}; {a['rows']:,} rows, {a['days']} days)", "",
                  "| Method | R² all test weeks [95 % CI] |" + (" R² clean weeks [95 % CI] |" if "clean_weeks" in r else ""),
                  "|---|---|" + ("---|" if "clean_weeks" in r else "")]
        order = sorted(a["R2"], key=lambda m: -a["R2"][m]["point"])
        for m in order:
            v = a["R2"][m]
            s = f"| {m} | {f3(v['point'])} [{f3(v['ci95'][0])}, {f3(v['ci95'][1])}] |"
            if "clean_weeks" in r:
                c = r["clean_weeks"]["R2"][m]
                s += f" {f3(c['point'])} [{f3(c['ci95'][0])}, {f3(c['ci95'][1])}] |"
            lines.append(s)
        lines += ["", "Paired differences (A − B, R²; P = share of resamples with A ≤ B):", "",
                  "| A − B | all weeks [95 % CI] | P(A ≤ B) |" + (" clean weeks [95 % CI] |" if "clean_weeks" in r else ""),
                  "|---|---|---|" + ("---|" if "clean_weeks" in r else "")]
        for k, v in a["diff"].items():
            s = f"| {k} | {v['point']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {v['p_le_0']:.2f} |"
            if "clean_weeks" in r:
                c = r["clean_weeks"]["diff"][k]
                s += f" {c['point']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] |"
            lines.append(s)
        for bk in ("by_season", "by_sky", "by_regime"):
            if bk not in r:
                continue
            buckets = list(r[bk])
            ms = [m for m in order if not m.startswith("baseline") and "unguided" not in m and "v1" not in m][:6]
            lines += ["", f"{bk.replace('_', ' ')}: MSE skill vs {r['best_baseline']} (nMAE)", "",
                      "| Method | " + " | ".join(f"{b} (n {r[bk][b]['n']:,}, {r[bk][b]['days']} d)" for b in buckets) + " |",
                      "|---|" + "---|" * len(buckets)]
            for m in ms:
                lines.append(f"| {m} | " + " | ".join(f"{r[bk][b][m]['skill']:+.2f} ({r[bk][b][m]['nMAE']:.2f})"
                                                       for b in buckets) + " |")
        if "MAE_kW" in r:
            lines += ["", "MAE (kW): " + ", ".join(f"{m} {v:.0f}" for m, v in sorted(r["MAE_kW"].items(), key=lambda kv: kv[1])[:8])]
        lines.append("")
    (OUT / "e1_compare.md").write_text("\n".join(lines))
    log("saved results/e1_compare.{json,md}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report(json.load(open(OUT / "e1_compare.json")))
    else:
        main()
