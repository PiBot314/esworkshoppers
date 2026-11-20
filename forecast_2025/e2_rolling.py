"""E2: rolling-origin evaluation of the LSTM (never run before) next to LightGBM, on identical rows (20 Nov).

Fold m = 3..12: train on targets before (1st of month m - 16 days), early-stop on the 14 days before the 2-day gap,
test on month m. Same windows rule as B3 (origin and target in the same part). Per fold: LSTM forecast and noweather
(B3 template, seed 0, guided), LightGBM forecast and noweather (B2 features, early stopping on the same val days),
baselines. Scores pooled over folds (last step), plus per month.
Note: the guided gate's prior comes from B2's importances on the blocked split (a prior only; it isn't a fitted
parameter of the test month), and the grid LSTM is the v2 template (day/week-ago covariates).

  python3 forecast_2025/e2_rolling.py   -> results/e2_rolling.{json,md}
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b3_guided_lstm as B          # noqa: E402
import b2_lgbm_feature_test as B2   # noqa: E402
import e1_compare as E              # noqa: E402

OUT = HERE / "results"
log = lambda *a: print(*a, flush=True)


def fold_df(df, m):
    start = pd.Timestamp(2025, m, 1)
    T = df.index
    sp = np.full(len(df), "gap", dtype=object)
    sp[T < start - pd.Timedelta(days=16)] = "train"
    sp[(T >= start - pd.Timedelta(days=16)) & (T < start - pd.Timedelta(days=2))] = "val"
    sp[df["month"].to_numpy() == m] = "test"
    d = df.copy()
    d["split_blocked"] = sp
    return d


def lstm_fold(d, family, hname, H, noweather):
    ser, names_p, names_c, wp, wc = E.series(d, family, H)
    if noweather:
        for s in ser:
            s["P"][:, wp] = 0; s["C"][:, wc] = 0
    n_nodes = len(B.PLANTS) if family == "solar" else 1
    W = B.Windows(ser, H, n_nodes)
    gp, gc = B.gate_prior(family, hname, names_p, names_c)
    torch.manual_seed(0); np.random.seed(0)
    net = B.Net(len(names_p), len(names_c), H, n_nodes, gp, gc, guided=True, solar=family == "solar").to(B.DEV)
    B.log = lambda *a: None                                  # quiet per-epoch lines
    B.train(net, W, 30, 2e-3)
    o, y, dd, t = B.predict(net, W, "test")
    return W, t, o[:, -1]


def run(df, family, hname):
    H = E.HZ[hname]
    allrows, per_month = [], {}
    for m in range(3, 13):
        t0 = time.time()
        d = fold_df(df, m)
        W, t, o_fc = lstm_fold(d, family, hname, H, False)
        _, t2, o_nw = lstm_fold(d, family, hname, H, True)
        assert torch.equal(t, t2)
        N = len(df)
        tn = t.cpu().numpy()
        node, loc = tn // N, tn % N
        rows = pd.DataFrame({"origin": df.index[loc], "T": df.index[loc + H], "node": node,
                             "y": W.Y[t + H].cpu().numpy(), "day": W.D[t + H].cpu().numpy(),
                             "LSTM forecast": o_fc, "LSTM noweather": o_nw})
        for b, p in B.baseline_preds(W, t, H, family).items():
            rows[f"baseline {b}"] = p[:, -1]
        lg = E.lgbm(d, family, H)
        key = pd.MultiIndex.from_arrays([rows["origin"], rows["node"]])
        for c in lg.columns:
            rows[c] = lg[c].reindex(key).to_numpy()
        rows = rows[np.isfinite(rows["y"]) & rows["day"]].drop(columns="day").dropna()
        rows["fold"] = m
        y = rows["y"].to_numpy()
        per_month[m] = {c: round(E.r2(y, rows[c].to_numpy()), 3) for c in rows.columns
                        if c.startswith(("LSTM", "LightGBM", "baseline"))} | {"n": int(len(rows))}
        log(f"  {family} {hname} month {m:2d}: n {len(rows):5d}  " +
            "  ".join(f"{k} {v:.3f}" for k, v in per_month[m].items() if k != "n" and not k.startswith("baseline p"))
            + f"  ({time.time() - t0:.0f}s)")
        allrows.append(rows)
    rows = pd.concat(allrows)
    methods = [c for c in rows.columns if c.startswith(("LSTM", "LightGBM", "baseline"))]
    pairs = [("LSTM forecast", "LSTM noweather"), ("LightGBM forecast", "LightGBM noweather"),
             ("LSTM forecast", "LightGBM forecast")]
    res = E.boot(rows, methods, pairs)
    res["per_month"] = per_month
    rows.to_parquet(OUT / f"e2_rows_{family}_{hname}.parquet")
    return res


def report(res):
    lines = ["# E2: rolling-origin, LSTM vs LightGBM on identical rows (20 Nov)", "",
             "Built by [e2_rolling.py](../e2_rolling.py). Folds: test month m = Mar..Dec, train on targets before "
             "(1st of m − 16 d), early-stop on the next 14 d, 2-day gap. R² at the horizon pooled over folds; 95 % "
             "day-block bootstrap. Compare with the blocked split (e1_compare.md): the drop is what unseen seasons cost.", ""]
    for key, r in res.items():
        lines += [f"## {key} ({r['rows']:,} rows, {r['days']} days)", "", "| Method | R² [95 % CI] |", "|---|---|"]
        for m, v in sorted(r["R2"].items(), key=lambda kv: -kv[1]["point"]):
            lines.append(f"| {m} | {v['point']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] |")
        lines += ["", "| A − B | R² diff [95 % CI] | P(A ≤ B) |", "|---|---|---|"]
        for k, v in r["diff"].items():
            lines.append(f"| {k} | {v['point']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {v['p_le_0']:.2f} |")
        cols = ["LSTM forecast", "LSTM noweather", "LightGBM forecast", "LightGBM noweather"]
        lines += ["", "Per test month (R²):", "", "| Month | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        for m, v in r["per_month"].items():
            lines.append(f"| {m} | {v['n']:,} | " + " | ".join(f"{v[c]:.3f}" for c in cols) + " |")
        lines.append("")
    (OUT / "e2_rolling.md").write_text("\n".join(lines))


if __name__ == "__main__":
    df = pd.read_parquet(B.DATA / "data_15min.parquet")
    res = {}
    for family in ("grid", "solar"):
        for hname in ("1h", "24h"):
            res[f"{family}_{hname}"] = run(df, family, hname)
            json.dump(res, open(OUT / "e2_rolling.json", "w"), indent=1)
            report(res)
    log("saved results/e2_rolling.{json,md}")
