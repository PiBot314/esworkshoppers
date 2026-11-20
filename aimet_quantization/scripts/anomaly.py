"""Residual-based anomaly detection on top of the frozen forecasters (no retraining).

For each window, the model's +15-min forecast is compared with the reading that actually arrives:
    r = actual - forecast   (Watts, per domain)
    anomaly  <=>  |r - median_r[hour]| > threshold[hour_of_day_IST, domain]
Centring on the per-hour median removes systematic forecast bias (e.g. the Demand head). Thresholds = K * robust sigma (1.4826 * MAD) of validation residuals in that IST hour, with a floor of
FLOOR_FRAC * capacity so near-zero night-time noise does not trigger alarms.
"""
import numpy as np

from common import MODELS, data_capacities, model_capacities, to_watts

K = 4.0
FLOOR_FRAC = 0.02


def hod_index(name):
    # position of hod_sin / hod_cos in the input features
    return (4, 5) if name == "single_node" else (52, 53)


def target_hour_ist(X, name):
    """IST hour of the +15 min target, decoded from the (MinMax-scaled) cyclic features of the last input step."""
    i_s, i_c = hod_index(name)
    s, c = 2 * X[:, -1, i_s] - 1, 2 * X[:, -1, i_c] - 1
    hour = (np.arctan2(s, c) / (2 * np.pi) * 24 + 0.25) % 24
    return np.floor(hour + 1e-6).astype(int) % 24


def residuals(pred_n, y_n, name):
    """[N, D] Watts residuals at the first horizon step."""
    P = to_watts(pred_n, name, model_capacities(name))[:, 0, :]
    T = to_watts(y_n, name, data_capacities(name))[:, 0, :]
    return T - P, T, P


def fit_thresholds(res, hours, name):
    """Returns array [2, 24, D]: [0] = per-hour median residual, [1] = per-hour threshold (Watts)."""
    caps = np.asarray(model_capacities(name))
    out = np.zeros((2, 24, res.shape[1]))
    for h in range(24):
        r = res[hours == h]
        med = np.median(r, axis=0) if len(r) else np.zeros(res.shape[1])
        mad = np.median(np.abs(r - med), axis=0) if len(r) else np.zeros(res.shape[1])
        out[0, h], out[1, h] = med, np.maximum(K * 1.4826 * mad, FLOOR_FRAC * caps)
    return out


def flags(res, hours, thr):
    return np.abs(res - thr[0][hours]) > thr[1][hours]


def inject(T, hours, name, seed=0, n=300):
    """Synthetic faults on the *actual* readings of the first domain (solar), daylight windows only.
    Returns modified actuals, boolean mask of injected rows, and the fault type per injected row."""
    rng = np.random.default_rng(seed)
    cap = model_capacities(name)[0]
    day = np.flatnonzero((T[:, 0] > 0.10 * cap) & (hours >= 8) & (hours <= 16))
    pick = rng.choice(day, size=min(n, len(day)), replace=False)
    kinds = np.array(["panel_damage_40pct", "sensor_dropout_zero", "sensor_spike"])[np.arange(len(pick)) % 3]
    T2 = T.copy()
    for i, k in zip(pick, kinds):
        if k == "panel_damage_40pct":
            T2[i, 0] *= 0.6
        elif k == "sensor_dropout_zero":
            T2[i, 0] = 0.0
        else:
            T2[i, 0] += 0.5 * cap
    mask = np.zeros(len(T), bool)
    mask[pick] = True
    kind_of = dict(zip(pick, kinds))
    return T2, mask, kind_of


def evaluate(pred_n, y_n, X, name, thr):
    """Natural flag rate + detection rate of injected faults for one model output."""
    res, T, P = residuals(pred_n, y_n, name)
    hours = target_hour_ist(X, name)
    f = flags(res, hours, thr)
    T2, mask, kind_of = inject(T, hours, name)
    f_inj = flags(T2 - P, hours, thr)[:, 0]
    # detection of an injected fault is only meaningful where the clean reading was not already flagged
    mask = mask & ~f[:, 0]
    by_kind = {}
    for k in set(kind_of.values()):
        idx = [i for i, kk in kind_of.items() if kk == k]
        by_kind[k] = float(np.mean(f_inj[idx]))
    return dict(natural_flag_rate={d: float(f[:, j].mean()) for j, d in enumerate(MODELS[name]["domains"])},
                injected_detection_rate=float(f_inj[mask].mean()), by_kind=by_kind,
                flags=f, flags_injected=f_inj)
