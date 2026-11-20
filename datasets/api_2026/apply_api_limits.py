"""Write API-fitted grid anomaly limits into every live model's meta.json (20 Nov; fixes M4, see README "Calibration
set and M4" and reports/WRITEUP_2026-11-20.md).

Why: with the limits fitted on 2025 validation residuals every model flags 3.4-14 % of normal API readings (M4 needs
<= 3 %); with limits fitted on API residuals (same rule) all pass, out-of-sample by alternate weeks (m4_results.json).
What: same rule as m4_flag_rates.py / anomaly.py fit_thresholds -- per IST hour, median residual and
max(K * 1.4826 * MAD, floor * capacity) -- now fitted on ALL API windows (1 Jan 2026 -> last fetched day), GRID
column only (the live loop only flags grid: registry "signals": ["grid"]). Solar/demand columns are left as they were.
The 2025 limits are kept in the same file as "anomaly_2025" so the change can be undone with --undo.
These are local copies; the live loop reads them from the board (MODELS_DIR / the NPU server), so they take effect
after the next push to the board.

  python3 datasets/api_2026/apply_api_limits.py [--undo]
"""
import json, sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m4_flag_rates as M   # noqa: E402

FINAL = M.FINAL


def main(undo=False):
    if undo:
        for mid, (_, metap) in M.MODELS.items():
            p = FINAL / metap
            meta = json.load(open(p))
            if "anomaly_2025" in meta:
                meta["anomaly"] = meta.pop("anomaly_2025")
                meta.pop("anomaly_source", None)
                json.dump(meta, open(p, "w"), indent=1)
                print(f"{mid}: restored the 2025 limits")
        return
    idx, frames = M.build_frames()
    grid = frames["EM-CR-SB00-00"]["gd_power_W"] + frames["EM-CR-KH95-00"]["power_total_W"].fillna(0)
    ok = grid.notna().to_numpy()
    ends = np.array([t for t in range(95, len(idx) - 1) if ok[t] and ok[t + 1]])
    nxt = ends + 1
    hours = idx[nxt].tz_convert("Asia/Kolkata").hour.to_numpy()
    actual = grid.to_numpy()[nxt]
    summary = {}
    for mid, (onnx, metap) in M.MODELS.items():
        p = FINAL / metap
        meta = json.load(open(p))
        base = meta.get("anomaly_2025", meta["anomaly"])          # idempotent: always start from the 2025 limits
        X = M.features(meta, idx, frames)
        sess = ort.InferenceSession(str(FINAL / onnx), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        D = len(meta["domains"])
        g = meta["domains"].index("Grid")
        capg = meta["model_capacities_W"][g]
        fc = np.array([sess.run(None, {name: X[None, t - 95:t + 1]})[0].reshape(-1, D)[0, g] * capg for t in ends])
        resid = actual - fc
        K, FLOOR = base["K"], base["floor_frac"]
        med = [list(r) for r in base["median_W"]]
        lim = [list(r) for r in base["limit_W"]]
        for h in range(24):
            r = resid[hours == h]
            med[h][g] = float(np.median(r))
            lim[h][g] = float(max(K * 1.4826 * np.median(np.abs(r - med[h][g])), FLOOR * capg))
        flag = np.abs(resid - np.asarray(med)[hours, g]) > np.asarray(lim)[hours, g]
        meta["anomaly_2025"] = base
        meta["anomaly"] = dict(base, median_W=med, limit_W=lim)
        meta["anomaly_source"] = (f"grid column: API residuals {idx[0].date()}..{idx[-1].date()} ({len(ends):,} windows), "
                                  "apply_api_limits.py (20 Nov 2026); other columns and anomaly_2025: 2025 validation")
        json.dump(meta, open(p, "w"), indent=1)
        old_lim = np.asarray(base["limit_W"])[:, g]
        summary[mid] = {"in_sample_flag_rate": round(float(flag.mean()), 4),
                        "limit_kW_mean_2025": round(float(old_lim.mean()) / 1e3, 1),
                        "limit_kW_mean_api": round(float(np.mean(np.asarray(lim)[:, g])) / 1e3, 1)}
        print(mid, summary[mid], flush=True)
    json.dump(summary, open(HERE / "api_limits_applied.json", "w"), indent=1)


if __name__ == "__main__":
    main(undo="--undo" in sys.argv)
