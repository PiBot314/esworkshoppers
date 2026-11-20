"""ERA5 reanalysis for the IIIT-H campus, 2015-2024 (hourly, IST), for climatology features and simulated-plant
pre-training (plan_19112026.md B1). Same variables and point as fetch_weather.py. Writes era5_2015_2024.json.gz."""
import gzip, json, time, urllib.parse, urllib.request
from pathlib import Path
from fetch_weather import LAT, LON, VARS, UA

HERE = Path(__file__).parent
parts = []
for y in range(2015, 2025):
    q = {"latitude": LAT, "longitude": LON, "start_date": f"{y}-01-01", "end_date": f"{y}-12-31", "timezone": "Asia/Kolkata",
         "hourly": ",".join(VARS), "models": "era5"}
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request("https://archive-api.open-meteo.com/v1/archive?" +
                                        urllib.parse.urlencode(q), headers=UA), timeout=300) as r:
                d = json.loads(r.read()); break
        except Exception as e:
            print("retry", y, e, flush=True); time.sleep(20 * (i + 1))
    parts.append(d)
    print(y, len(d["hourly"]["time"]), "hours", flush=True)
    time.sleep(2)
out = {"meta": {k: v for k, v in parts[0].items() if k != "hourly"}, "hourly": {}}
for p in parts:
    for k, v in p["hourly"].items():
        out["hourly"].setdefault(k, []).extend(v)
with gzip.open(HERE / "era5_2015_2024.json.gz", "wt") as f:
    json.dump(out, f)
print("saved era5_2015_2024.json.gz")
