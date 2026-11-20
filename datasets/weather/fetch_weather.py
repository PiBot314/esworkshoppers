"""Weather and solar irradiance for the IIIT-H campus (Gachibowli), for training the next models (plan_19112026.md B1).
Free, keyless sources, one point (campus centre), 2025-01-01 -> today. Writes <source>.json.gz next to this file.

  archive_era5     Open-Meteo archive, ERA5 reanalysis (0.25 deg): what the weather was. Hourly. ~5 days behind.
                   (Not the archive's default "best_match": at this location it equals hist_forecast hour for hour.)
  hist_forecast    Open-Meteo historical forecast: what forecast models said at the time (hourly + 15-min irradiance).
  previous_day1    Open-Meteo previous runs: the forecast issued ~1 day before each hour (a 24 h model's live input).
  nasa_power       NASA POWER hourly: all-sky and clear-sky surface irradiance, cloud amount. Weeks behind.

Usage: python fetch_weather.py [start YYYY-MM-DD] [end YYYY-MM-DD]
"""
import gzip, json, sys, time, urllib.parse, urllib.request
from datetime import date, timedelta
from pathlib import Path

LAT, LON = 17.4456, 78.3487  # IIIT-H campus centre (map database)
HERE = Path(__file__).parent
START = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2025, 1, 1)
END = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date.today()
UA = {"User-Agent": "campus-energy-forecasting-research/1.0"}

VARS = ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance", "cloud_cover",
        "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "temperature_2m", "relative_humidity_2m",
        "precipitation", "wind_speed_10m"]
PREV = ["shortwave_radiation", "cloud_cover", "temperature_2m", "relative_humidity_2m", "precipitation"]


def get(url, params, tries=3):
    q = url + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(q, headers=UA), timeout=180) as r:
                return json.loads(r.read())
        except Exception as e:
            print(f"  retry {i + 1}: {e}", flush=True)
            time.sleep(10 * (i + 1))
    raise RuntimeError(f"failed: {url}")


def months(a, b):
    """Calendar-month chunks, so each request stays small."""
    while a <= b:
        nxt = (a.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield a, min(b, nxt - timedelta(days=1))
        a = nxt


def merge(parts, key):
    out = {"meta": {k: v for k, v in parts[0].items() if k not in ("hourly", "minutely_15")}, key: {}}
    for p in parts:
        for k, v in p.get(key, {}).items():
            out[key].setdefault(k, []).extend(v)
    return out


def save(name, obj):
    p = HERE / f"{name}.json.gz"
    with gzip.open(p, "wt") as f:
        json.dump(obj, f)
    print(f"saved {p.name}", flush=True)


def done(name):
    if (HERE / f"{name}.json.gz").exists():
        print(f"{name}: already saved, skipping", flush=True)
        return True
    return False


def open_meteo(name, url, hourly, extra=None, end=END):
    if done(name):
        return
    parts, parts15 = [], []
    for a, b in months(START, end):
        params = {"latitude": LAT, "longitude": LON, "start_date": a, "end_date": b, "timezone": "Asia/Kolkata",
                  "hourly": ",".join(hourly)}
        params.update(extra or {})
        d = get(url, params)
        parts.append(d)
        if "minutely_15" in d:
            parts15.append(d)
        print(f"{name} {a}..{b}: {len(d.get('hourly', {}).get('time', []))} hours", flush=True)
        time.sleep(1)
    obj = merge(parts, "hourly")
    if parts15:
        obj["minutely_15"] = merge(parts15, "minutely_15")["minutely_15"]
    save(name, obj)


if __name__ == "__main__":
    open_meteo("archive_era5", "https://archive-api.open-meteo.com/v1/archive", VARS, {"models": "era5"},
               end=min(END, date.today() - timedelta(days=6)))
    open_meteo("hist_forecast", "https://historical-forecast-api.open-meteo.com/v1/forecast", VARS,
               {"minutely_15": "shortwave_radiation,direct_radiation,diffuse_radiation,temperature_2m"})
    open_meteo("previous_day1", "https://previous-runs-api.open-meteo.com/v1/forecast",
               [f"{v}_previous_day1" for v in PREV] + [f"{v}_previous_day2" for v in PREV])
    # NASA POWER: one request per year, hourly, IST-aligned
    if done("nasa_power"):
        sys.exit(0)
    parts = []
    for y in range(START.year, END.year + 1):
        a, b = max(START, date(y, 1, 1)), min(END, date(y, 12, 31))
        d = get("https://power.larc.nasa.gov/api/temporal/hourly/point",
                {"parameters": "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,CLOUD_AMT,T2M,RH2M",
                 "community": "RE", "latitude": LAT, "longitude": LON, "start": a.strftime("%Y%m%d"),
                 "end": b.strftime("%Y%m%d"), "format": "JSON", "time-standard": "LST"})
        parts.append(d)
        print(f"nasa_power {y}: {len(next(iter(d['properties']['parameter'].values())))} hours", flush=True)
    save("nasa_power", {"meta": parts[0].get("header", {}), "years": [p["properties"]["parameter"] for p in parts]})
