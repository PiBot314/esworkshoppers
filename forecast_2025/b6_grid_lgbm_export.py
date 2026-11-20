"""Grid LightGBM for the board CPU: export + single-prediction timing (20 Nov; decision 2 in archive/worklogs/WORKLOG_2026-11-20.md §8).

E1/E2 found LightGBM the better grid forecaster at 1 h and 24 h. This trains exactly the E1 'LightGBM forecast'
config (B2 features: lags + time + calendar + clear-sky + weather forecast; train split, early stop on val), checks it
reproduces E1's test R2, and writes for each horizon:
  models/grid_<h>_lgbm.txt         LightGBM text model (loads with lightgbm.Booster(model_file=...); C API on the board)
  models/grid_<h>_lgbm.json        feature list in order, with where each comes from live; rounds; test R2; timing
Timing: one row, num_threads=1, on this laptop (x86). The board CPU timing is from benchmark_qidk (see README).

  python3 forecast_2025/b6_grid_lgbm_export.py
"""
import json, sys, time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b2_lgbm_feature_test as B2   # noqa: E402

SOURCE = {   # where a live system would get each input
    "y_": "SB00 gd_power history (campus API), MW",
    "hod_": "clock", "dow_": "clock", "doy_": "clock", "is_weekend": "clock",
    "in_session": "academic calendar (campus_calendar.py, from the almanac)", "holiday": "academic calendar",
    "exam": "academic calendar", "event": "academic calendar",
    "cs_T": "clear-sky model (clearsky.py, computed)", "zen_T": "sun position (computed)",
    "fc15_": "Open-Meteo 15-min forecast (minutely_15), latest run", "fc_": "Open-Meteo hourly forecast, latest run",
    "d1_": "Open-Meteo forecast issued the day before (previous-day run)",
    "anom_": "d1 forecast minus the 2015-2024 ERA5 normal for the day of year (normals file, static)",
}


def source(f):
    return next(v for k, v in SOURCE.items() if f.startswith(k))


def main():
    df = pd.read_parquet(B2.DATA / "data_15min.parquet")
    out = {}
    for hname, H in (("1h", 4), ("24h", 96)):
        f, groups = B2.build(df, "grid", H)
        feats = groups["base"] + groups["calendar"] + groups["clearsky"] + groups["forecast"]
        tr, va, te = (f["split"] == s for s in ("train", "val", "test"))
        te = te & f["eval"]
        m = B2.fit(f.loc[tr, feats], f.loc[tr, "target"], f.loc[va, feats], f.loc[va, "target"])
        p = m.predict(f.loc[te, feats], num_iteration=m.best_iteration)
        r2 = B2.metrics(f.loc[te, "target"], p)["R2"]
        path = HERE / "models" / f"grid_{hname}_lgbm.txt"
        m.save_model(str(path), num_iteration=m.best_iteration)
        b = lgb.Booster(model_file=str(path), params={"num_threads": 1})
        x = f.loc[te, feats].to_numpy()[:1]
        for _ in range(20):
            b.predict(x)
        n = 500
        t0 = time.perf_counter()
        for i in range(n):
            b.predict(x)
        ms = (time.perf_counter() - t0) / n * 1e3
        same = float(np.max(np.abs(b.predict(f.loc[te, feats].to_numpy()) - p)))
        info = {"horizon": hname, "H": H, "rounds": int(m.best_iteration), "num_leaves": B2.PARAMS["num_leaves"],
                "test_R2_blocked": round(r2, 4), "reload_max_abs_diff": same,
                "predict_ms_1row_1thread_x86": round(ms, 3), "file_kB": round(path.stat().st_size / 1024, 1),
                "units": "MW (SB00 gd_power)",
                "features": [{"name": c, "live_source": source(c)} for c in feats]}
        json.dump(info, open(path.with_suffix(".json"), "w"), indent=1)
        out[hname] = {k: v for k, v in info.items() if k != "features"}
        print(hname, out[hname], flush=True)
    return out


if __name__ == "__main__":
    main()
