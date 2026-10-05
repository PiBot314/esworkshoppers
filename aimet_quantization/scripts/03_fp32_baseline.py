"""P2: FP32 baseline on the rebuilt data (V0) + cached FP32 outputs used as the quantization reference."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, OUT, DEPLOY, domain_metrics, fp32_session, headline_r2, load, run_session  # noqa: E402


def main(names):
    summary = {}
    for name in names:
        out = os.path.join(OUT, name, "fp32")
        os.makedirs(out, exist_ok=True)
        sess = fp32_session(name)
        for split in ("calib", "val", "test"):
            X, _ = load(name, split)
            np.save(os.path.join(out, f"out_{split}.npy"), run_session(sess, X))
        X, y = load(name, "test")
        m = domain_metrics(np.load(os.path.join(out, "out_test.npy")), y, name)
        meta = json.load(open(os.path.join(DEPLOY, MODELS[name]["meta"])))
        reported = meta.get("evaluation_summary") or meta.get("test_metrics_24h") or {"Solar_PL00": {"R2": meta.get("test_r2")}}
        summary[name] = dict(rebuilt_test=m, headline_R2=headline_r2(m, name), reported=reported, n_test=len(X))
        print(f"[{name}] n_test={len(X)} headline R2 (rebuilt) = {summary[name]['headline_R2']:.4f}")
        for d, v in m.items():
            rep = reported.get(d, {}).get("R2")
            print(f"    {d:11s} R2={v['R2']:.4f} (reported {rep})  MAE={v['MAE_W']:.0f} W  nMAE={v['nMAE_pct']:.2f}%")
    with open(os.path.join(OUT, "fp32_baseline.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
