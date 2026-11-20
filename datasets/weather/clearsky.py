"""Clear-sky irradiance (GHI on a cloudless day) for the IIIT-H campus, every 15 min, 2015-01-01 .. 2026-12-31 (IST).

No pvlib (it couldn't be installed on 19 Nov): sun position with the NOAA solar-calculator formulas, clear-sky GHI with
the Haurwitz model, then scaled month by month to NASA POWER's clear-sky GHI (CLRSKY_SFC_SW_DWN, in nasa_power.json.gz),
which accounts for the local atmosphere. Each 15-min value is the mean over the interval ENDING at its timestamp (the
convention of Open-Meteo's values), from 1-min samples.

Output: clearsky_15min.csv.gz  columns: time (IST), zenith_deg (at the interval midpoint), ghi_haurwitz, ghi_clearsky
"""
import gzip, json
from pathlib import Path
import numpy as np, pandas as pd

LAT, LON = 17.4456, 78.3487
HERE = Path(__file__).parent


def solar_zenith(t_ist):
    """NOAA solar calculator (accurate to ~0.01 deg for our purposes). t_ist: DatetimeIndex, naive IST."""
    t = t_ist - pd.Timedelta(hours=5, minutes=30)                      # UTC
    jd = t.to_julian_date().to_numpy()
    jc = (jd - 2451545) / 36525
    l0 = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360)
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    mr = np.radians(m)
    c = np.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) + np.sin(2 * mr) * (0.019993 - 0.000101 * jc) \
        + np.sin(3 * mr) * 0.000289
    app = l0 + c - 0.00569 - 0.00478 * np.sin(np.radians(125.04 - 1934.136 * jc))
    obl = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    oblc = np.radians(obl + 0.00256 * np.cos(np.radians(125.04 - 1934.136 * jc)))
    dec = np.arcsin(np.sin(oblc) * np.sin(np.radians(app)))
    y = np.tan(oblc / 2) ** 2
    l0r = np.radians(l0)
    eqt = 4 * np.degrees(y * np.sin(2 * l0r) - 2 * e * np.sin(mr) + 4 * e * y * np.sin(mr) * np.cos(2 * l0r)
                         - 0.5 * y * y * np.sin(4 * l0r) - 1.25 * e * e * np.sin(2 * mr))
    minutes_utc = (t.hour * 60 + t.minute + t.second / 60).to_numpy()
    tst = np.mod(minutes_utc + eqt + 4 * LON, 1440)                      # true solar time, minutes
    ha = np.radians(np.where(tst / 4 < 0, tst / 4 + 180, tst / 4 - 180))
    lat = np.radians(LAT)
    cosz = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    return np.degrees(np.arccos(np.clip(cosz, -1, 1)))


def haurwitz(zen):
    cz = np.cos(np.radians(zen))
    return np.where(cz > 0, 1098 * cz * np.exp(-0.057 / np.maximum(cz, 1e-6)), 0.0)


if __name__ == "__main__":
    t1 = pd.date_range("2014-12-31 23:46", "2026-12-31 23:45", freq="1min")
    z1 = solar_zenith(t1)
    g1 = pd.Series(haurwitz(z1), index=t1)
    g15 = g1.resample("15min", label="right", closed="right").mean()
    g15 = g15["2015-01-01 00:00":"2026-12-31 23:45"]
    zmid = pd.Series(solar_zenith(g15.index - pd.Timedelta(minutes=7, seconds=30)), index=g15.index)

    # monthly scale to NASA POWER clear-sky (local solar time hours; compare monthly daily totals, so the clock
    # convention doesn't matter)
    years = json.load(gzip.open(HERE / "nasa_power.json.gz", "rt"))["years"]
    cs = pd.concat([pd.Series(y["CLRSKY_SFC_SW_DWN"]) for y in years])
    cs.index = pd.to_datetime(cs.index, format="%Y%m%d%H")
    cs = cs.where(cs >= 0)
    nasa_m = cs.resample("MS").mean()
    ours_m = g15.resample("MS").mean()
    k = (nasa_m / ours_m).dropna()
    k = k[cs.resample("MS").count() > 24 * 20]                           # months with >= 20 days of NASA data
    by_month = k.groupby(k.index.month).mean()                            # month-of-year factor (fills 2026 gaps)
    print("scale to NASA clear-sky by month:", by_month.round(3).to_dict())
    factor = pd.Series(g15.index.month.map(by_month).to_numpy(), index=g15.index)
    out = pd.DataFrame({"zenith_deg": zmid.round(3), "ghi_haurwitz": g15.round(2), "ghi_clearsky": (g15 * factor).round(2)})
    out.index.name = "time"
    out.to_csv(HERE / "clearsky_15min.csv.gz")
    print(f"saved clearsky_15min.csv.gz: {len(out)} rows, {out.index[0]} .. {out.index[-1]}, "
          f"peak {out.ghi_clearsky.max():.0f} W/m2")
