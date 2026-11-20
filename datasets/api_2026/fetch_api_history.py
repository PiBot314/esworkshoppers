"""History of the mentors' campus API (synthetic 2026, served from 2026-01-01) for training (plan_19112026.md B1).

Phase 1: the model nodes, one node-day per request  -> nodes/<node_id>/<YYYY-MM-DD>.json.gz
Phase 2: all nodes, one hour per request             -> all/<YYYY-MM-DD>/<HH>.json.gz  (raw response, all verticals)

Adaptive speed (since 19 Nov 19:10): as fast as the server stays healthy, backing off when it slows.
  * Starts with 1 request in flight and no pause.
  * Every 10 responses: median of (response time / that request type's usual time). The usual time is the median of
    the last 30 responses for that node (phase 1) or for hourly chunks (phase 2) taken while only 1 request was in
    flight, so it can't creep up with load; each type's first 5 responses only build it.
      median < 1.3x usual -> one more request in flight (up to MAX_INFLIGHT)
      median > 2x usual   -> halve the requests in flight; at 1 already, add a pause (doubling, up to 30 s)
      median < 1.5x usual -> drop the pause
  * Any failure halves the requests in flight and adds a pause; 5 consecutive failures stop the run.
Resumable: existing files are skipped. Today is never fetched (still filling); the last complete day is yesterday.

Run as a systemd user service (see README.md here):  python fetch_api_history.py [phase1|phase2|all]
"""
import gzip, json, statistics, sys, time, urllib.error, urllib.parse, urllib.request
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, datetime, timedelta
from pathlib import Path

API = "http://10.2.203.214:9999/verticals/all/temporal"
HERE = Path(__file__).parent
FIRST = date(2026, 1, 1)
LAST = date.today() - timedelta(days=1)
MODEL_NODES = ["EM-CR-SB00-00", "EM-CR-SB00-01", "EM-CR-KH95-00", "SL-PL00-00", "SL-NI01-00", "SL-VN95-00",
               "SR-EM-KH04-01", "SR-EM-KH04-02", "WE-VN04-00"]
MAX_INFLIGHT = 6
WINDOW = 10           # responses per adjustment
UP, DOWN, CALM = 1.3, 2.0, 1.5
MAX_PAUSE = 30.0
STOP_AFTER_FAILS = 5


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(obj, f)
    tmp.rename(path)


def days():
    d = FIRST
    while d <= LAST:
        yield d
        d += timedelta(days=1)


def tasks(what):
    """(key for the usual-time baseline, output path, query params, kind) for every file not yet saved."""
    if what in ("phase1", "all"):
        for node in MODEL_NODES:
            for d in days():
                p = HERE / "nodes" / node / f"{d}.json.gz"
                if not p.exists():
                    yield node, p, {"from": f"{d}T00:00:00+05:30", "to": f"{d}T23:59:59+05:30", "node_id": node}, "node"
    if what in ("phase2", "all"):
        for d in days():
            for h in range(24):
                p = HERE / "all" / str(d) / f"{h:02d}.json.gz"
                if not p.exists():
                    yield "hour", p, {"from": f"{d}T{h:02d}:00:00+05:30", "to": f"{d}T{h:02d}:59:59+05:30"}, "hour"


def fetch(params, timeout=180):
    t0 = time.time()
    with urllib.request.urlopen(API + "?" + urllib.parse.urlencode(params), timeout=timeout) as r:
        body = json.loads(r.read())
    return body, time.time() - t0


def run(what):
    todo = list(tasks(what))
    log(f"API history {FIRST}..{LAST}, {what}: {len(todo)} requests to go")
    history = defaultdict(lambda: deque(maxlen=30))  # key -> recent response times
    ratios, inflight, pause, fails, done = [], 1, 0.0, 0, 0
    started = time.time()
    pending, inflight_at_submit = {}, {}
    queue = deque(todo)
    with ThreadPoolExecutor(MAX_INFLIGHT) as pool:
        while queue or pending:
            while queue and len(pending) < inflight:
                key, path, params, kind = queue.popleft()
                fut = pool.submit(fetch, params)
                pending[fut] = (key, path, params, kind)
                inflight_at_submit[fut] = len(pending)
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                key, path, params, kind = pending.pop(fut)
                at_submit = inflight_at_submit.pop(fut)  # drop every reference to the future: it holds the response
                try:
                    body, dt = fut.result()
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
                    fails += 1
                    inflight = max(1, inflight // 2)
                    pause = min(MAX_PAUSE, max(2.0, pause * 2))
                    log(f"failed ({fails} in a row): {params} {e} -> in flight {inflight}, pause {pause:.0f} s")
                    queue.appendleft((key, path, params, kind))  # retry later
                    if fails >= STOP_AFTER_FAILS:
                        log("STOP: too many consecutive failures; rerun later to resume")
                        for f in pending:
                            f.cancel()
                        sys.exit(2)
                    continue
                fails = 0
                save(path, (next(iter(body.values()))["data"] if body else []) if kind == "node" else body)
                done += 1
                # baseline = median time for this key, learned only at 1 in flight (so it can't creep up with load);
                # the first 5 responses of a key only build the baseline
                h = history[key]
                if len(h) < 5 or at_submit == 1:
                    warm = len(h) < 5
                    h.append(dt)
                    if warm:
                        continue
                ratios.append(dt / max(statistics.median(h), 0.05))
                if len(ratios) >= WINDOW:
                    r = statistics.median(ratios)
                    ratios.clear()
                    before = (inflight, pause)
                    if r > DOWN:
                        if inflight > 1:
                            inflight = max(1, inflight // 2)
                        else:
                            pause = min(MAX_PAUSE, max(1.0, pause * 2))
                    elif r < UP:
                        inflight = min(MAX_INFLIGHT, inflight + 1)
                    if r < CALM:
                        pause = 0.0
                    if (inflight, pause) != before:
                        log(f"slowdown x{r:.2f} -> in flight {inflight}, pause {pause:.0f} s")
                if done % 100 == 0:
                    rate = done / (time.time() - started)
                    log(f"{done}/{len(todo)} done ({rate:.1f}/s, ~{(len(todo) - done) / rate / 3600:.1f} h left), "
                        f"in flight {inflight}, now at {path.relative_to(HERE)}")
            if pause:
                time.sleep(pause)
    log("all done")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "all")
