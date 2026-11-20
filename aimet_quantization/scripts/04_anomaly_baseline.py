"""P2: fit residual-anomaly thresholds on FP32 validation residuals and report FP32 detection behaviour on TEST."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anomaly  # noqa: E402
from common import MODELS, OUT, load  # noqa: E402


def main(names):
    summary = {}
    for name in names:
        fp = os.path.join(OUT, name, "fp32")
        Xv, yv = load(name, "val")
        res, _, _ = anomaly.residuals(np.load(os.path.join(fp, "out_val.npy")), yv, name)
        thr = anomaly.fit_thresholds(res, anomaly.target_hour_ist(Xv, name), name)
        np.save(os.path.join(OUT, name, "anomaly_thresholds_W.npy"), thr)
        Xt, yt = load(name, "test")
        ev = anomaly.evaluate(np.load(os.path.join(fp, "out_test.npy")), yt, Xt, name, thr)
        np.save(os.path.join(fp, "flags_test.npy"), ev.pop("flags"))
        np.save(os.path.join(fp, "flags_injected_test.npy"), ev.pop("flags_injected"))
        summary[name] = ev
        print(f"[{name}] natural flag rate {ev['natural_flag_rate']} | injected detection "
              f"{ev['injected_detection_rate']:.3f} {ev['by_kind']}")
    with open(os.path.join(OUT, "anomaly_fp32.json"), "w") as f:
        json.dump(dict(K=anomaly.K, FLOOR_FRAC=anomaly.FLOOR_FRAC, models=summary), f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
