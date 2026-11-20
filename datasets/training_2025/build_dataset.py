"""Training dataset for the next models (plan_19112026.md B1, option C): REAL 2025 meter data + weather, 15-min, IST.

Inputs (all in datasets/ or aimet_quantization/):
  meters     real 2025 dump via aimet_quantization/scripts/01_rebuild_dataset.py (MAPPING + load_raw: same column
             names/units as the deployed models; VN95's swapped registers handled there)
  weather    weather/hist_forecast (15-min irradiance + hourly), weather/previous_day1 (1- and 2-day-ahead forecasts),
             weather/archive_era5 (actual weather, "perfect weather" upper bound), weather/era5_2015_2024 (climatology),
             weather/clearsky_15min
Conventions
  * every value at timestamp t (IST, naive) is the mean over (t-15 min, t]  (interval ENDING at t, as the weather files)
  * power in W, generation positive
  * solar: 0 when the sun is down (clear-sky GHI = 0); daytime gaps <= 15 min filled (flag *_filled), longer left NaN
  * grid = EM-CR-SB00-00 GDPower (the substation), gaps <= 1 h filled (flag), longer NaN. EM-CR-KH95-00 is kept as
    its own column (kh95_W): it is a ~0.1-0.5 kW sub-circuit (W; 0.54 A x 232 V ~ 125 W), so the deployed models'
    "SB00 + KH95" adds ~0.1 % and would blank the grid whenever KH95 alone is missing (81 % coverage)
  * hourly weather values (mean over the hour ending at H) are given to the four 15-min bins of that hour
Targets
  * solar_<plant>_W, grid_W; solar_<plant>_csi = output / (capacity x clear-sky GHI / 1000), only when clear-sky
    GHI >= 50 W/m2 (low sun is unstable). Capacity = 99.5th percentile of daytime output (ESTIMATED, not nameplate).
Splits (both kept, plan_19112026.md "Splits")
  * split_blocked: 7-day blocks from 1 Jan, in 8 consecutive groups of ~6-7 weeks; in each group one random week is
    val and one test (seed 0), the rest train (~70/15/15, every season in every set); training rows within 2 days of a
    val/test week are 'gap'
  * month: for rolling-origin (train on months < m, test month m)
Outputs: data_15min.parquet, meta.json (columns, units, sources, capacities, build info)
"""
import gzip, hashlib, json, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                                   # final/
W = HERE.parent / "weather"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "aimet_quantization" / "scripts"))
import importlib
rebuild = importlib.import_module("01_rebuild_dataset")     # MAPPING, load_raw (main guard: nothing runs)

SOLAR = ["SL-PL00-00", "SL-VN95-00", "SL-NI01-00", "SL-NI03-00", "SL-NI03-01", "SL-VN02-01"]
GRID = ["EM-CR-SB00-00", "EM-CR-KH95-00"]
START, END = pd.Timestamp("2025-01-01 00:15"), pd.Timestamp("2026-01-01 00:00")
IDX = pd.date_range(START, END, freq="15min")
log = lambda *a: print(*a, flush=True)


def to_ist_15min(s_utc):
    """UTC-indexed readings -> mean over (t-15, t], IST naive, on IDX; plus the raw-sample count per bin."""
    s = s_utc.dropna()
    s.index = s.index.tz_convert("Asia/Kolkata").tz_localize(None)
    r = s.resample("15min", label="right", closed="right")
    return r.mean().reindex(IDX), r.count().reindex(IDX).fillna(0).astype(int)


def meter(node):
    family, target, cols = rebuild.MAPPING[node]
    raw = rebuild.load_raw(node, family)
    v = cols[target](raw).astype(float)
    if target.endswith("_kW"):
        v = v * 1000.0
    return to_ist_15min(v)


def fill_short(v, max_bins, where=None):
    """Fill NaN runs of at most max_bins bins (forward fill); longer runs stay NaN. Returns (series, filled mask)."""
    isna = v.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    short = isna & (run_len <= max_bins) & v.ffill().notna()
    if where is not None:
        short &= where
    return v.where(~short, v.ffill()), short


def load_om(name, key="hourly", prefix=""):
    d = json.load(gzip.open(W / f"{name}.json.gz", "rt"))[key]
    df = pd.DataFrame(d)
    df.index = pd.to_datetime(df.pop("time"))
    return df.add_prefix(prefix)


def hourly_to_15(df):
    """Hourly mean over (H-1, H] -> the 15-min bins H-45, H-30, H-15, H."""
    out = df.reindex(df.index.union(IDX)).sort_index().bfill(limit=3)
    return out.reindex(IDX)


def build():
    cs = pd.read_csv(W / "clearsky_15min.csv.gz", index_col=0, parse_dates=True).reindex(IDX)
    out = pd.DataFrame(index=IDX)
    out.index.name = "time"
    out["ghi_clearsky"] = cs["ghi_clearsky"]
    out["zenith_deg"] = cs["zenith_deg"]
    sun_up = out["ghi_clearsky"] > 0
    meta = {"solar": {}, "grid": {}}

    # --- solar plants
    for n in SOLAR:
        v, cnt = meter(n)
        sign = np.sign(v[sun_up & (v.abs() > 0)].median())
        v = v * sign
        night = v[~sun_up].dropna()
        offset = float(night.median()) if len(night) else 0.0
        day_peak = float(v[sun_up].quantile(0.995))
        if abs(offset) > 0.01 * day_peak:                   # a standby draw / sensor offset
            v = v - offset
        v[~sun_up] = 0.0
        v, filled = fill_short(v, 1, where=sun_up)
        v = v.clip(lower=0)
        cap = float(v[sun_up].quantile(0.995))
        key = n.replace("SL-", "").replace("-", "_")
        out[f"solar_{key}_W"] = v
        out[f"solar_{key}_filled"] = filled.astype("int8")
        ok = out["ghi_clearsky"] >= 50
        out[f"solar_{key}_csi"] = (v / (cap * out["ghi_clearsky"] / 1000)).where(ok).clip(0, 1.5)
        meta["solar"][n] = {"column": f"solar_{key}_W", "sign_applied": int(sign), "night_offset_removed_W":
                            offset if abs(offset) > 0.01 * day_peak else 0.0, "est_capacity_W": cap,
                            "daytime_coverage": float(v[sun_up].notna().mean()), "filled_bins": int(filled.sum())}
        log(f"{n}: sign {int(sign):+d}, est. capacity {cap / 1000:.1f} kW, daytime coverage "
            f"{meta['solar'][n]['daytime_coverage']:.0%}, night offset {offset:.0f} W "
            f"({'removed' if abs(offset) > 0.01 * day_peak else 'kept'})")

    # --- per-plant daily status from a peer comparison (the plan's L2 fault check): daily mean clear-sky index of the
    #     plant / median of the other plants that day. < 0.15 outage, < 0.8 degraded (e.g. NI01 Jan-Apr / May-Oct 2025).
    #     Days without enough data stay 'unknown'. Training should use only 'ok' days per plant.
    keys = [n.replace("SL-", "").replace("-", "_") for n in SOLAR]
    day_of = (out.index - pd.Timedelta(minutes=15)).normalize()
    mid = (out["ghi_clearsky"] >= 300)
    dcsi = pd.DataFrame({k: out[f"solar_{k}_csi"].where(mid).groupby(day_of).mean() for k in keys})
    dcnt = pd.DataFrame({k: out[f"solar_{k}_csi"].where(mid).groupby(day_of).count() for k in keys})
    meta["solar_status_counts"] = {}
    for k in keys:
        peers = dcsi.drop(columns=k).median(axis=1)
        ratio = dcsi[k] / peers
        status = pd.Series("ok", index=dcsi.index)
        status[ratio < 0.8] = "degraded"
        status[ratio < 0.15] = "outage"
        status[(dcnt[k] < 8) | peers.isna() | (peers < 0.2)] = "unknown"   # < 2 h of good-sun data, or all cloudy
        out[f"solar_{k}_status"] = status.reindex(day_of).to_numpy()
        out[f"solar_{k}_peer_ratio"] = ratio.reindex(day_of).to_numpy()
        meta["solar_status_counts"][k] = status.value_counts().to_dict()
        log(f"  status {k}: {status.value_counts().to_dict()}")

    # --- grid
    parts = {}
    for n in GRID:
        v, cnt = meter(n)
        cov = float(v.notna().mean())
        v, filled = fill_short(v, 4)
        parts[n] = (v, filled)
        meta["grid"][n] = {"coverage": cov, "filled_bins": int(filled.sum())}
        log(f"{n}: coverage {v.notna().mean():.0%}, median {v.median():.0f} W")
    out["grid_W"], out["grid_filled"] = parts["EM-CR-SB00-00"][0], parts["EM-CR-SB00-00"][1].astype("int8")
    out["kh95_W"], out["kh95_filled"] = parts["EM-CR-KH95-00"][0], parts["EM-CR-KH95-00"][1].astype("int8")

    # --- weather: forecasts as the board would have them, ERA5 as the "perfect weather" upper bound
    hf15 = load_om("hist_forecast", "minutely_15")
    hf15 = hf15.reindex(IDX).add_prefix("fc15_")
    hf_h = hourly_to_15(load_om("hist_forecast")[["cloud_cover", "cloud_cover_low", "relative_humidity_2m",
                                                    "precipitation", "wind_speed_10m"]]).add_prefix("fc_")
    pd1 = hourly_to_15(load_om("previous_day1")).rename(columns=lambda c: "d1_" + c.replace("_previous_day1", "")
                                                       if c.endswith("day1") else "d2_" + c.replace("_previous_day2", ""))
    era = hourly_to_15(load_om("archive_era5")).add_prefix("era5_")
    out = out.join([hf15, hf_h, pd1, era])

    # --- climatology from ERA5 2015-2024: daily normals by day of year (circular +-7-day smoothing)
    h = load_om("era5_2015_2024")
    daily = h[["shortwave_radiation", "temperature_2m", "cloud_cover", "relative_humidity_2m"]].resample("1D").mean()
    doy = daily.groupby(np.minimum(daily.index.dayofyear, 365)).mean()
    doy = pd.concat([doy.iloc[-7:], doy, doy.iloc[:7]]).rolling(15, center=True).mean().iloc[7:-7]
    days = (out.index - pd.Timedelta(minutes=15)).normalize()  # the day a (t-15, t] bin belongs to
    dd = np.minimum(pd.DatetimeIndex(days).dayofyear, 365)
    for c, name in (("shortwave_radiation", "ghi"), ("temperature_2m", "temp"), ("cloud_cover", "cloud"),
                    ("relative_humidity_2m", "rh")):
        out[f"clim_{name}_day"] = doy[c].reindex(dd).to_numpy()
    d1_daily = out[["d1_shortwave_radiation", "d1_temperature_2m", "d1_cloud_cover"]].groupby(days).transform("mean")
    out["anom_ghi_d1"] = d1_daily["d1_shortwave_radiation"] - out["clim_ghi_day"]
    out["anom_temp_d1"] = d1_daily["d1_temperature_2m"] - out["clim_temp_day"]
    out["anom_cloud_d1"] = d1_daily["d1_cloud_cover"] - out["clim_cloud_day"]

    # --- time features (calendar: slot for term/exam/holiday flags, filled later)
    ist = out.index - pd.Timedelta(minutes=7, seconds=30)  # interval midpoint
    hod = ist.hour + ist.minute / 60
    for name, val, period in (("hod", hod, 24.0), ("dow", ist.dayofweek, 7.0), ("doy", ist.dayofyear, 365.25)):
        out[f"{name}_sin"] = np.sin(2 * np.pi * np.asarray(val) / period)
        out[f"{name}_cos"] = np.cos(2 * np.pi * np.asarray(val) / period)
    out["is_weekend"] = (ist.dayofweek >= 5).astype("int8")
    # campus academic calendar from the two official 2025 almanacs (see campus_calendar.py)
    import campus_calendar as cal
    flags = cal.daily_flags(ist)                               # one row per calendar date
    day_key = pd.DatetimeIndex(ist).normalize()
    for c in ("in_session", "holiday", "exam", "event"):
        out[c] = flags[c].reindex(day_key).to_numpy().astype("int8")
    out["term"] = out["in_session"]                            # back-compat alias (1 = teaching term, 0 = vacation)

    # --- splits
    week = pd.Series((ist.dayofyear.to_numpy() - 1) // 7, index=out.index)   # 7-day blocks from 1 Jan
    weeks = np.sort(week.unique())
    # season-balanced: 8 consecutive groups of ~6-7 weeks; in each, one week at random for val and one for test
    rng = np.random.default_rng(0)
    assign = {}
    for g in np.array_split(weeks, 8):
        pick = rng.choice(g, size=2, replace=False)
        assign |= {w: "train" for w in g} | {pick[0]: "val", pick[1]: "test"}
    split = week.map(assign)
    held = split.isin(["val", "test"])
    near = held.astype(int).rolling(2 * 96 * 2 + 1, center=True, min_periods=1).max().astype(bool)
    split[(split == "train") & near] = "gap"
    out["split_blocked"] = split
    out["month"] = ist.month
    meta["splits"] = {"seed": 0, "weeks": {k: int(sum(1 for v in assign.values() if v == k)) for k in
                                           ("train", "val", "test")},
                      "rows": out["split_blocked"].value_counts().to_dict(), "gap_days": 2}
    return out, meta


if __name__ == "__main__":
    df, meta = build()
    df.to_parquet(HERE / "data_15min.parquet")
    meta.update({
        "built": datetime.now().isoformat(timespec="seconds"), "rows": len(df), "span": [str(df.index[0]), str(df.index[-1])],
        "convention": "IST naive; value at t = mean over (t-15 min, t]; power W, generation positive",
        "columns": list(df.columns),
        "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(W.glob("*.gz"))},
    })
    json.dump(meta, open(HERE / "meta.json", "w"), indent=1, default=str)
    log(f"saved data_15min.parquet: {df.shape[0]} rows x {df.shape[1]} columns; splits {meta['splits']['rows']}")
