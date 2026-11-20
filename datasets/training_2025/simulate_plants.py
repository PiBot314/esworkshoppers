"""Simulated 15-min output of the 6 campus solar plants for 2015-2024 -- a PRE-TRAINING set (plan_20112026.md).

Model (per plant p):   sim_p(t) = gain_p(month, slot) x ghi_clearsky(t) x csi_era5(t)
  * ghi_clearsky(t): deterministic 15-min clear-sky GHI (../weather/clearsky_15min.csv.gz) -> sub-hour sun shape
  * csi_era5(t):     ERA5 hourly shortwave / hourly mean clear-sky GHI (cloud attenuation, hourly), clipped to [0, 1.3],
                     linearly interpolated from hour centres to 15-min bin centres
  * gain_p(month, slot): the plant's power per W/m2 of (clear-sky GHI x csi_era5), fitted on REAL 2025 data as a
                     ratio of sums, sum(W) / sum(ghi_clearsky x csi_era5), over all 'ok'-status bins, per calendar month
                     and 15-min time-of-day slot (energy-unbiased; a median over clear hours only was ~6 % low). Carries the plant's size, tilt/orientation, shading
                     and seasonal sun-angle effects. Sparse cells borrow from neighbouring months/slots.
Validated on 2025 with a 2-fold cross-fit (gain fitted on even weeks, scored on odd weeks and vice versa), against a
plain "capacity x ERA5 GHI / 1000" baseline. Real 2025 stays the ONLY train/test target; this set is for pre-training.

Outputs (this folder): sim_2015_2024.parquet, sim_gain.csv (the fitted gain table), sim_report.json
"""
import gzip, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE.parent / "weather"
sys.dont_write_bytecode = True
PLANTS = ["PL00_00", "VN95_00", "NI01_00", "NI03_00", "NI03_01", "VN02_01"]
CS_MIN, MIN_N = 20.0, 3
log = lambda *a: print(*a, flush=True)


def load_era5(name):
    d = json.load(gzip.open(W / f"{name}.json.gz", "rt"))["hourly"]
    df = pd.DataFrame(d)
    df.index = pd.to_datetime(df.pop("time"))         # IST; value = mean over the hour ENDING at the stamp
    return df


def csi_15min(ghi_hourly, cs15):
    """ERA5 hourly GHI + 15-min clear-sky -> 15-min clear-sky index (hourly cloud, interpolated)."""
    cs_h = cs15.resample("1h", label="right", closed="right").mean().reindex(ghi_hourly.index)
    k = (ghi_hourly / cs_h).where(cs_h >= CS_MIN).clip(0, 1.3)
    # hours with too little sun: carry the day's nearest valid value (low weight anyway, sim ~ gain x small cs)
    day = (k.index - pd.Timedelta(minutes=30)).normalize()
    k = k.groupby(day).transform(lambda s: s.ffill().bfill()).fillna(1.0)
    x_h = (k.index - pd.Timedelta(minutes=30)).asi8.astype(float)           # hour centres
    x_q = (cs15.index - pd.Timedelta(minutes=7, seconds=30)).asi8.astype(float)  # 15-min bin centres
    return pd.Series(np.interp(x_q, x_h, k.to_numpy()), index=cs15.index)


def slot_of(idx):
    m = idx - pd.Timedelta(minutes=15)                 # bin (t-15, t] -> slot of its start
    return (m.hour * 4 + m.minute // 15).to_numpy(), m.month.to_numpy()


def fit_gain(df, plant, mask):
    """gain(month, slot) table (12 x 96) from real bins selected by mask."""
    y = df[f"solar_{plant}_W"]
    sel = mask & (df[f"solar_{plant}_status"] == "ok") & y.notna() & (df["ghi_clearsky"] >= CS_MIN)
    x = (df["ghi_clearsky"] * df["csi_era5"])[sel]
    slot, month = slot_of(x.index)
    num, den = y[sel].groupby([month, slot]).sum(), x.groupby([month, slot]).sum()
    cnt = x.groupby([month, slot]).count()
    tab = (num / den)[cnt >= MIN_N].unstack().reindex(index=range(1, 13), columns=range(96))
    # sparse cells: average of the neighbouring months, then interpolate along the day, then edge-fill
    for _ in range(3):
        a = tab.to_numpy()
        nb = np.nanmean(np.stack([np.roll(a, 1, axis=0), np.roll(a, -1, axis=0)]), axis=0)   # Dec <-> Jan wrap
        tab = tab.fillna(pd.DataFrame(nb, index=tab.index, columns=tab.columns))
    tab = tab.interpolate(axis=1, limit_area="inside").bfill(axis=1).ffill(axis=1)
    tab = tab.T.rolling(3, center=True, min_periods=1).median().T          # de-spike along the day
    return tab


def apply_gain(tab, idx, cs, csi):
    slot, month = slot_of(idx)
    g = tab.to_numpy()[month - 1, slot]
    return pd.Series(g * cs.to_numpy() * csi.to_numpy(), index=idx)


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def score(df, plant, pred):
    y = df[f"solar_{plant}_W"]
    ok = (df[f"solar_{plant}_status"] == "ok") & y.notna() & (df["ghi_clearsky"] > 0)
    out = {"r2_bin": r2(y[ok], pred[ok]), "n_bins": int(ok.sum())}
    day = (df.index - pd.Timedelta(minutes=15)).normalize()
    up = df["ghi_clearsky"] > 0
    cov = (y.notna() & up).groupby(day).sum() / up.groupby(day).sum()
    st = df[f"solar_{plant}_status"].groupby(day).first()
    good = (cov >= 0.9) & (st == "ok")
    ed = (y.where(up) / 4000).groupby(day).sum()[good]                     # kWh per day
    pd_ = (pred.where(up & y.notna()) / 4000).groupby(day).sum()[good]
    out |= {"r2_daily": r2(ed, pd_), "n_days": int(good.sum()),
            "bias_daily_pct": float(100 * (pd_.sum() - ed.sum()) / ed.sum())}
    return out


if __name__ == "__main__":
    cs_all = pd.read_csv(W / "clearsky_15min.csv.gz", index_col=0, parse_dates=True)
    df = pd.read_parquet(HERE / "data_15min.parquet")
    meta = json.load(open(HERE / "meta.json"))
    caps = {v["column"].replace("solar_", "").replace("_W", ""): v["est_capacity_W"] for v in meta["solar"].values()}

    # ---- 2025: validation of the simulator against real output
    e25 = load_era5("archive_era5")["shortwave_radiation"]
    df["csi_era5"] = csi_15min(e25, cs_all["ghi_clearsky"]).reindex(df.index)
    week = ((df.index - pd.Timedelta(minutes=15)).dayofyear - 1) // 7
    even = pd.Series(week % 2 == 0, index=df.index)
    report, tables = {"model": __doc__.split("Validated")[0].strip(), "plants": {}}, {}
    for p in PLANTS:
        cv = pd.Series(np.nan, index=df.index)
        for train in (even, ~even):
            tab = fit_gain(df, p, train)
            pr = apply_gain(tab, df.index, df["ghi_clearsky"], df["csi_era5"])
            cv[~train] = pr[~train]
        full = fit_gain(df, p, pd.Series(True, index=df.index))
        tables[p] = full
        base = caps[p] * df["era5_shortwave_radiation"] / 1000
        r = {"crossfit": score(df, p, cv),
             "insample": score(df, p, apply_gain(full, df.index, df["ghi_clearsky"], df["csi_era5"])),
             "baseline_cap_x_era5ghi": score(df, p, base),
             "gain_cells_from_data": int(full.notna().sum().sum()), "est_capacity_W": caps[p]}
        report["plants"][p] = r
        c, b = r["crossfit"], r["baseline_cap_x_era5ghi"]
        log(f"{p}: cross-fit R2 bin {c['r2_bin']:.3f} daily {c['r2_daily']:.3f} (bias {c['bias_daily_pct']:+.1f}%) | "
            f"baseline bin {b['r2_bin']:.3f} daily {b['r2_daily']:.3f} | {c['n_days']} days")

    g = pd.concat({p: t.stack() for p, t in tables.items()}, axis=1)
    g.index.names = ["month", "slot"]
    g.round(3).to_csv(HERE / "sim_gain.csv")

    # ---- 2015-2024 simulation
    h = load_era5("era5_2015_2024")
    idx = pd.date_range("2015-01-01 00:15", "2025-01-01 00:00", freq="15min")
    cs = cs_all.reindex(idx)
    csi = csi_15min(h["shortwave_radiation"], cs_all["ghi_clearsky"]["2015-01-01 00:15":"2025-01-01 00:00"]).reindex(idx)
    sim = pd.DataFrame(index=idx)
    sim.index.name = "time"
    sim["ghi_clearsky"], sim["zenith_deg"], sim["csi_era5"] = cs["ghi_clearsky"], cs["zenith_deg"], csi.round(4)
    for p in PLANTS:
        sim[f"sim_{p}_W"] = apply_gain(tables[p], idx, cs["ghi_clearsky"], csi).where(cs["ghi_clearsky"] > 0, 0.0) \
            .astype("float32")
    hb = h.reindex(h.index.union(idx)).sort_index().bfill(limit=3).reindex(idx).ffill(limit=4)  # hour (H-1,H] -> its 4 bins; last hour of 2024 carried
    for c in h.columns:
        sim[f"era5_{c}"] = hb[c].astype("float32")
    ist = idx - pd.Timedelta(minutes=7, seconds=30)
    hod = ist.hour + ist.minute / 60
    for name, val, period in (("hod", hod, 24.0), ("doy", ist.dayofyear, 365.25)):
        sim[f"{name}_sin"] = np.sin(2 * np.pi * np.asarray(val) / period).astype("float32")
        sim[f"{name}_cos"] = np.cos(2 * np.pi * np.asarray(val) / period).astype("float32")
    sim["year"], sim["month"] = ist.year.astype("int16"), ist.month.astype("int8")
    assert not sim.isna().any().any(), sim.isna().sum()[lambda s: s > 0]
    sim.to_parquet(HERE / "sim_2015_2024.parquet")

    # sanity: simulated annual energy per year vs real 2025 (in-sample sim on 2025 for comparison)
    yearly = (sim[[f"sim_{p}_W" for p in PLANTS]].groupby(sim["year"]).sum() / 4e6).round(1)   # MWh
    yearly.columns = PLANTS
    sim25 = pd.DataFrame({p: apply_gain(tables[p], df.index, df["ghi_clearsky"], df["csi_era5"]) for p in PLANTS})
    yearly.loc["2025_sim"] = (sim25.sum() / 4e6).round(1)
    report["annual_MWh"] = yearly.astype(float).to_dict(orient="index")
    report["rows"], report["span"] = len(sim), [str(idx[0]), str(idx[-1])]
    report["columns"] = list(sim.columns)
    json.dump(report, open(HERE / "sim_report.json", "w"), indent=1, default=str)
    log(yearly.to_string())
    log(f"saved sim_2015_2024.parquet: {sim.shape}, sim_gain.csv, sim_report.json")
