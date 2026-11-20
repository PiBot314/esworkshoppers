"""Recompute the baselines in results/b3_<family>_<h>.json on the identical test windows (no retraining), after the
fix in b3_guided_lstm.baseline_preds (a missing same-time-yesterday value was scored as 0; now falls back to
persistence, as in B2). Then rewrites b3_results.md.
"""
import json, sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b3_guided_lstm as B  # noqa: E402

df = pd.read_parquet(B.DATA / "data_15min.parquet")
for f in sorted((HERE / "results").glob("b3_*_*.json")):
    r = json.load(open(f))
    if "family" not in r:
        continue
    fam, H = r["family"], r["H"]
    ser, *_ = B.series_solar(df, H) if fam == "solar" else B.series_grid(df, H)
    W = B.Windows(ser, H, len(B.PLANTS) if fam == "solar" else 1)
    t = W.idx["test"]
    _, _, _, y, day = W.batch(t)
    y, day = y.cpu().numpy(), day.cpu().numpy()
    r["baselines"] = {n: B.scores(p, y, day) for n, p in B.baseline_preds(W, t, H, fam).items()}
    B.add_skill(r)
    json.dump(r, open(f, "w"), indent=1)
    print(f.name, {n: round(v["last_step"]["R2"], 3) for n, v in r["baselines"].items()})
B.report()
