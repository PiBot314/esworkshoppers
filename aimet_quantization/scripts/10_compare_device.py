"""P4 (V3/V4/V5): compare outputs pulled from the QIDK with FP32 ONNX and with the AIMET simulation.

Usage: 10_compare_device.py <model> <variant> [<variant> ...]
Reads out/<model>/device/<variant>/Result_<i>/<output>.raw (float32, written by qnn-net-run).
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anomaly  # noqa: E402
from common import OUT, domain_metrics, fidelity, headline_r2, load  # noqa: E402


def read_device(d, n, width):
    files = [glob.glob(os.path.join(d, f"Result_{i}", "*.raw"))[0] for i in range(n)]
    return np.stack([np.fromfile(f, dtype=np.float32)[:width] for f in files])


def main(name, variants):
    Xt, yt = load(name, "test")
    ref = np.load(os.path.join(OUT, name, "fp32", "out_test.npy"))
    sim = np.load(os.path.join(OUT, name, "sim_W8A16_final", "out_test.npy"))
    thr = np.load(os.path.join(OUT, name, "anomaly_thresholds_W.npy"))
    f_fp = np.load(os.path.join(OUT, name, "fp32", "flags_test.npy"))
    fi_fp = np.load(os.path.join(OUT, name, "fp32", "flags_injected_test.npy"))
    path = os.path.join(OUT, name, "device_results.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    fp_r2 = headline_r2(domain_metrics(ref, yt, name), name)
    for v in variants:
        dev = read_device(os.path.join(OUT, name, "device", v), len(Xt), ref.shape[1])
        np.save(os.path.join(OUT, name, "device", f"{v}_out_test.npy"), dev)
        m = domain_metrics(dev, yt, name)
        ev = anomaly.evaluate(dev, yt, Xt, name, thr)
        results[v] = dict(vs_fp32=fidelity(ref, dev), vs_aimet_sim=fidelity(sim, dev), metrics=m,
                          headline_R2=headline_r2(m, name), fp32_headline_R2=fp_r2,
                          flag_agreement_natural=float(np.mean(ev["flags"] == f_fp)),
                          flag_agreement_injected=float(np.mean(ev["flags_injected"] == fi_fp)),
                          injected_detection_rate=ev["injected_detection_rate"])
        r = results[v]
        print(f"[{name}] {v:22s} vs FP32 SQNR {r['vs_fp32']['SQNR_dB']:6.2f} dB max|d| {r['vs_fp32']['max_abs']:.4f} | "
              f"vs AIMET sim {r['vs_aimet_sim']['SQNR_dB']:6.2f} dB | R2 {r['headline_R2']:.4f} (FP32 {fp_r2:.4f}) | "
              f"flags {r['flag_agreement_natural']:.4f}/{r['flag_agreement_injected']:.4f}")
    json.dump(results, open(path, "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
