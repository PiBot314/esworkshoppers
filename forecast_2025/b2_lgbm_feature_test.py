"""B2: LightGBM feature test on REAL 2025 data (plan_19112026.md §3 B2, plan_20112026.md).

Question: do weather forecasts, clear-sky and the campus calendar help, at which horizon, and for which family?

Direct forecasting: from origin t, predict the value at T = t + h for h = 4 (1 h), 24 (6 h), 96 (24 h) bins.
Families
  solar  one model pooled over the 6 plants (plant id as a categorical), target = output per kWp (W / est. capacity);
         rows only on 'ok' days of that plant at t and T; scored on DAYTIME targets (clear-sky GHI > 0), since nights
         are 0 for every method and would inflate R2.
  grid   substation SB00 (W), all hours.
Feature groups (added cumulatively, each run = the previous + one group)
  base      lags known at t (value at t, t-1, t-2, t-4, mean of the last 1 h / 24 h, same time yesterday T-96,
            same time last week T-672 for grid) + time of T (hod, dow, doy sin/cos, weekend)
  calendar  in_session / holiday / exam / event at T (grid only; solar doesn't care)
  clearsky  clear-sky GHI and zenith at T, clear-sky index at t (solar), clear-sky-index persistence forecast
  forecast  weather FORECAST valued at T as the board would have it: h <= 6 h -> 15-min historical forecast
            (GHI, direct, diffuse, temperature) + hourly (cloud, low cloud, humidity, rain, wind);
            h = 24 h -> forecasts issued 1 day earlier (d1_*: GHI, cloud, temp, humidity, rain) + d1 anomalies vs the
            2015-2024 normal for the date
  era5      ERA5 actual weather at T ("perfect weather" upper bound; NOT available live)
Baselines: persistence (value at t), same time yesterday (T-96), and for solar clear-sky-index persistence.
Every method is scored on the same rows: target known, and the origin value and same-time-yesterday value known
(otherwise the baselines would be scored on a filled 0 -- that was a bug in the first run, grid 24 h baseline -0.40).
Splits: split_blocked (train / early-stop on val / report test; by the TARGET's week) and rolling-origin (train on
months < m minus a 2-day gap, test month m, m = 3..12, rounds fixed from the blocked run).
Outputs: forecast_2025/results/b2_results.json, b2_results.md, b2_importance_<family>_<h>.json (for the guided gate).
"""
import json, sys, time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
DATA = FINAL / "datasets" / "training_2025"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
log = lambda *a: print(*a, flush=True)
PLANTS = ["PL00_00", "VN95_00", "NI01_00", "NI03_00", "NI03_01", "VN02_01"]
HORIZONS = {"1h": 4, "6h": 24, "24h": 96}
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=0, seed=0)
SHORT_FC = ["fc15_shortwave_radiation", "fc15_direct_radiation", "fc15_diffuse_radiation", "fc15_temperature_2m",
            "fc_cloud_cover", "fc_cloud_cover_low", "fc_relative_humidity_2m", "fc_precipitation", "fc_wind_speed_10m"]
DAY_FC = ["d1_shortwave_radiation", "d1_cloud_cover", "d1_temperature_2m", "d1_relative_humidity_2m",
          "d1_precipitation", "anom_ghi_d1", "anom_temp_d1", "anom_cloud_d1"]
ERA5 = ["era5_shortwave_radiation", "era5_direct_radiation", "era5_diffuse_radiation", "era5_cloud_cover",
        "era5_cloud_cover_low", "era5_temperature_2m", "era5_relative_humidity_2m", "era5_precipitation"]
CAL = ["in_session", "holiday", "exam", "event"]
TIME = ["hod_sin", "hod_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend"]


def at(s, k):
    """Value k bins ahead of each row (k > 0 = future)."""
    return s.shift(-k)


def frame_for(df, y, h, family, extra=None):
    """Feature/target table for one series y (normalised), horizon h. Index = origin t."""
    f = pd.DataFrame(index=df.index)
    f["y_t"], f["y_t1"], f["y_t2"], f["y_t4"] = y, y.shift(1), y.shift(2), y.shift(4)
    f["y_mean1h"] = y.rolling(4, min_periods=1).mean()
    f["y_mean24h"] = y.rolling(96, min_periods=24).mean()
    f["y_yday"] = at(y, h - 96)                                # same time yesterday relative to T (known at t)
    if family == "grid":
        f["y_week"] = at(y, h - 672)
    for c in TIME:
        f[c] = at(df[c], h)
    groups = {"base": list(f.columns)}
    if family == "grid":
        for c in CAL:
            f[c] = at(df[c], h)
        groups["calendar"] = CAL
    f["cs_T"] = at(df["ghi_clearsky"], h)
    f["zen_T"] = at(df["zenith_deg"], h)
    cs_cols = ["cs_T", "zen_T"]
    if family == "solar":
        f["csi_t"] = extra["csi"].ffill(limit=16)
        f["csi_persist"] = f["csi_t"] * f["cs_T"] / 1000
        cs_cols += ["csi_t", "csi_persist"]
    groups["clearsky"] = cs_cols
    fc = SHORT_FC if h <= 24 else DAY_FC
    for c in fc:
        f[c] = at(df[c], h)
    groups["forecast"] = fc
    for c in ERA5:
        f[c] = at(df[c], h)
    groups["era5"] = ERA5
    f["target"] = at(y, h)
    f["T"] = pd.Series(df.index, index=df.index).shift(-h)
    f["split"] = at(df["split_blocked"], h)
    f["month_T"] = at(df["month"], h)
    return f, groups


def build(df, family, h):
    meta = json.load(open(DATA / "meta.json"))
    if family == "grid":
        y = df["grid_W"] / 1e6                                  # MW
        f, groups = frame_for(df, y, h, "grid")
        f["unit"] = 1e6
        f = f[f["target"].notna()]
        f["eval"] = f["y_t"].notna() & f["y_yday"].notna()     # score only where the baselines' inputs exist
        return f, groups
    parts = []
    for i, p in enumerate(PLANTS):
        cap = meta["solar"][f"SL-{p.replace('_', '-')}"]["est_capacity_W"]
        y = df[f"solar_{p}_W"] / cap
        ok = df[f"solar_{p}_status"] == "ok"
        f, groups = frame_for(df, y, h, "solar", {"csi": df[f"solar_{p}_csi"]})
        f["plant"] = i
        f["unit"] = cap
        keep = f["target"].notna() & ok & at(ok, h).fillna(False).astype(bool)
        f = f[keep]
        f["eval"] = (f["cs_T"] > 0) & f["y_t"].notna() & f["y_yday"].notna()
        parts.append(f)
    groups["base"] = groups["base"] + ["plant"]
    return pd.concat(parts), groups


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mse = np.mean((y - p) ** 2)
    return {"R2": float(1 - mse / np.var(y)), "nMAE": float(np.mean(np.abs(y - p)) / np.mean(np.abs(y))),
            "MSE": float(mse), "n": int(len(y))}


def baselines(f, family):
    b = {"persistence": f["y_t"].fillna(0), "same_time_yesterday": f["y_yday"].fillna(f["y_t"]).fillna(0)}
    if family == "solar":
        b["csi_persistence"] = f["csi_persist"].fillna(f["y_yday"]).fillna(0)
    return b


def fit(Xtr, ytr, Xva=None, yva=None, rounds=None, cat=None):
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat or "auto", free_raw_data=False)
    if rounds is None:
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        m = lgb.train(PARAMS, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        m = lgb.train(PARAMS, dtr, rounds)
    return m


def run(df, family, hname, h):
    f, groups = build(df, family, h)
    order = ["base", "calendar", "clearsky", "forecast", "era5"] if family == "grid" else ["base", "clearsky", "forecast", "era5"]
    cat = ["plant"] if family == "solar" else None
    res = {"rows": int(len(f)), "configs": {}, "baselines": {}}
    tr, va, te = (f["split"] == s for s in ("train", "val", "test"))
    ev = f["eval"].to_numpy(bool)
    te_e = te.to_numpy() & ev
    for name, pred in baselines(f, family).items():
        res["baselines"][name] = {"blocked": metrics(f["target"][te_e], pred[te_e])}
    # rolling-origin folds (target month m; train on targets in months < m that are >= 2 days before month m)
    T = pd.to_datetime(f["T"])
    folds = []
    for m in range(3, 13):
        start = pd.Timestamp(2025, m, 1)
        trm = (T < start - pd.Timedelta(days=2)).to_numpy()
        tem = (f["month_T"] == m).to_numpy() & ev
        folds.append((m, trm, tem))
    for name, pred in baselines(f, family).items():
        sel = np.concatenate([np.flatnonzero(tem) for _, _, tem in folds])
        res["baselines"][name]["rolling"] = metrics(f["target"].to_numpy()[sel], pred.to_numpy()[sel])
    feats = []
    for g in order:
        feats = feats + groups[g]
        t0 = time.time()
        m = fit(f.loc[tr, feats], f.loc[tr, "target"], f.loc[va, feats], f.loc[va, "target"], cat=cat)
        p = m.predict(f.loc[te_e, feats], num_iteration=m.best_iteration)
        if family == "solar":
            p = np.clip(p, 0, None)
        r = {"features": len(feats), "best_iter": int(m.best_iteration), "blocked": metrics(f["target"][te_e], p)}
        # rolling-origin
        ys, ps, per_month = [], [], {}
        for mo, trm, tem in folds:
            mm = fit(f.loc[trm, feats], f.loc[trm, "target"], rounds=max(50, m.best_iteration), cat=cat)
            pp = mm.predict(f.loc[tem, feats])
            if family == "solar":
                pp = np.clip(pp, 0, None)
            ys.append(f["target"].to_numpy()[tem]); ps.append(pp)
            per_month[mo] = round(metrics(ys[-1], pp)["R2"], 4)
        r["rolling"] = metrics(np.concatenate(ys), np.concatenate(ps))
        r["rolling_by_month_R2"] = per_month
        imp = pd.Series(m.feature_importance("gain"), index=feats)
        r["importance_gain_share"] = (imp / imp.sum()).round(4).sort_values(ascending=False).to_dict()
        res["configs"]["+".join(order[:order.index(g) + 1])] = r
        log(f"  {family} {hname} {g:9s}: blocked R2 {r['blocked']['R2']:.4f}  rolling R2 {r['rolling']['R2']:.4f}  "
            f"({len(feats)} feats, {m.best_iteration} it, {time.time() - t0:.0f}s)")
    bb = max(res["baselines"].items(), key=lambda kv: kv[1]["blocked"]["R2"])
    res["best_baseline"] = bb[0]
    for k, r in res["configs"].items():
        for split in ("blocked", "rolling"):
            r[f"skill_vs_best_baseline_{split}"] = float(1 - r[split]["MSE"] / res["baselines"][bb[0]][split]["MSE"])
    json.dump(res["configs"][list(res["configs"])[-2]]["importance_gain_share"],
              open(OUT / f"b2_importance_{family}_{hname}.json", "w"), indent=1)   # without ERA5: live-usable
    return res


def report(all_res):
    lines = ["# B2: LightGBM feature test on real 2025", "",
             "Built by [b2_lgbm_feature_test.py](../b2_lgbm_feature_test.py). R² on the test rows (solar: daytime, per kWp, "
             "pooled over plants on ok days; grid: SB00, all hours). Feature groups are cumulative. "
             "Skill = 1 − MSE / MSE of the best baseline on the same rows. `era5` is the perfect-weather upper bound.", ""]
    for fam, hs in all_res.items():
        for hname, r in hs.items():
            lines += [f"## {fam}, {hname} ({r['rows']:,} rows)", "",
                      "| Method | Blocked R² | Rolling-origin R² | Skill (blocked) | Skill (rolling) |", "|---|---|---|---|---|"]
            for b, v in r["baselines"].items():
                lines.append(f"| baseline: {b} | {v['blocked']['R2']:.3f} | {v['rolling']['R2']:.3f} | – | – |")
            for k, v in r["configs"].items():
                lines.append(f"| LightGBM {k} | {v['blocked']['R2']:.3f} | {v['rolling']['R2']:.3f} | "
                             f"{v['skill_vs_best_baseline_blocked']:+.3f} | {v['skill_vs_best_baseline_rolling']:+.3f} |")
            top = list(r["configs"].values())[-2]["importance_gain_share"]
            lines += ["", "Top features (without ERA5, gain share): " +
                      ", ".join(f"`{k}` {v:.0%}" for k, v in list(top.items())[:8]), ""]
    (OUT / "b2_results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    df = pd.read_parquet(DATA / "data_15min.parquet")
    fams = sys.argv[1:] or ["solar", "grid"]
    all_res = {}
    for fam in fams:
        all_res[fam] = {}
        for hname, h in HORIZONS.items():
            all_res[fam][hname] = run(df, fam, hname, h)
            json.dump(all_res, open(OUT / "b2_results.json", "w"), indent=1)
    report(all_res)
    log("saved results/b2_results.{json,md}")
