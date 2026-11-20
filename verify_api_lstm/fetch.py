"""Fetch raw API readings for the forecasters' served nodes, one day per call, cached as JSON."""
import json, os, sys, time, urllib.request, urllib.parse
from datetime import date, timedelta

B = "http://10.2.203.214:9999/verticals/all/temporal"
NODES = ["EM-CR-SB00-00", "EM-CR-KH95-00", "SR-EM-KH04-01", "SR-EM-KH04-02", "SL-PL00-00", "SL-VN95-00", "SL-NI01-00"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
os.makedirs(OUT, exist_ok=True)
start, end = date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2])
stats = []
d = start
while d <= end:
    for n in NODES:
        f = os.path.join(OUT, f"{n}_{d}.json")
        if os.path.exists(f):
            continue
        q = urllib.parse.urlencode({"from": f"{d}T00:00:00+05:30", "to": f"{d}T23:59:59+05:30", "node_id": n})
        t0 = time.time()
        body = urllib.request.urlopen(f"{B}?{q}", timeout=60).read()
        dt = time.time() - t0
        rows = list(json.loads(body).values())[0]["data"]
        json.dump(rows, open(f, "w"))
        stats.append((n, str(d), len(rows), len(body), dt))
    d += timedelta(days=1)
for s in stats:
    print("%-14s %s rows=%4d bytes=%8d %.2fs" % s)
