"""Flag clustering, fault-injection detection and threshold recalibration on the API run (result_<mode>.json)."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = sys.argv[1]
mode = sys.argv[2]
R = json.load(open(os.path.join(HERE, f"result_{mode}.json")))
K, FLOOR = 4.0, 0.02

def arr(model, d):
    rows = R[model]
    t = np.array([r[0] for r in rows]); h = np.array([r[1] for r in rows])
    P = np.array([r[2][d] for r in rows]); A = np.array([r[3][d] for r in rows])
    return t, h, P, A

def thresholds(model, d):
    m = json.load(open(os.path.join(FINAL, "aimet_quantization/dashboard/www/data", model, "meta.json")))
    return np.asarray(m["anomaly"]["median_W"])[:, d], np.asarray(m["anomaly"]["limit_W"])[:, d], m["model_capacities_W"][d]

def fit(res, h, cap):
    med, lim = np.zeros(24), np.zeros(24)
    for k in range(24):
        r = res[h == k]
        med[k] = np.median(r) if len(r) else 0
        mad = np.median(np.abs(r - med[k])) if len(r) else 0
        lim[k] = max(K * 1.4826 * mad, FLOOR * cap)
    return med, lim

def detect(h, P, A, med, lim, cap, kind, day_only=True, seed=0, n=150):
    rng = np.random.default_rng(seed)
    cand = np.flatnonzero((h >= 8) & (h <= 16) & (A > 0.10 * cap)) if day_only else np.arange(len(A))
    pick = rng.choice(cand, size=min(n, len(cand)), replace=False)
    A2 = A.copy()
    if kind == "dropout": A2[pick] = 0
    elif kind == "spike": A2[pick] += 0.5 * cap
    elif kind == "drop40": A2[pick] *= 0.6
    elif kind == "spike20": A2[pick] += 0.2 * cap
    f = np.abs((A2 - P) - med[h]) > lim[h]
    return 100 * f[pick].mean()

for model, d, name in (("single_node", 0, "Solar PL00 (single_node)"), ("all_sensors_1h", 1, "Grid (all_sensors_1h)")):
    t, h, P, A = arr(model, d)
    med, lim, cap = thresholds(model, d)
    res = A - P
    f = np.abs(res - med[h]) > lim[h]
    print(f"\n== {name}, {mode}: {len(t)} readings, {t[0]} .. {t[-1]}")
    by_h = [f[h == k].mean() * 100 if (h == k).any() else 0 for k in range(24)]
    print("  flag % by IST hour, 2025 thresholds:", " ".join(f"{k}:{v:.0f}" for k, v in enumerate(by_h) if v >= 5) or "none >= 5 %")
    print("  mean residual by IST hour (kW):", " ".join(f"{k}:{np.mean(res[h==k])/1000:+.0f}" for k in range(0, 24, 3)))
    # recalibrate on the first half, test on the second half
    half = len(t) // 2
    m2, l2 = fit(res[:half], h[:half], cap)
    f2 = np.abs(res[half:] - m2[h[half:]]) > l2[h[half:]]
    f1 = f[half:]
    print(f"  natural flag rate, 2nd half: 2025 thresholds {100*f1.mean():.1f} %   recalibrated on 1st half {100*f2.mean():.1f} %")
    kinds = ("dropout", "spike", "drop40") if d == 0 and model == "single_node" else ("dropout", "spike20", "spike")
    for k in kinds:
        a = detect(h[half:], P[half:], A[half:], med, lim, cap, k, day_only=(model == "single_node"))
        b = detect(h[half:], P[half:], A[half:], m2, l2, cap, k, day_only=(model == "single_node"))
        print(f"  injected {k:8s} caught: 2025 thresholds {a:5.1f} %   recalibrated {b:5.1f} %")
