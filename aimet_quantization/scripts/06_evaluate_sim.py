"""P3 (V2): evaluate the exported AIMET models (model.onnx + model.encodings) on the held-out TEST windows.

Rebuilds a QuantSim from each export, loads its encodings, and reports
  fidelity vs FP32 (SQNR, max/mean abs error), accuracy in Watts vs rebuilt ground truth,
  and anomaly-flag agreement with the FP32 model (natural flags + injected faults).
"""
import json
import os
import sys

import numpy as np
import onnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anomaly  # noqa: E402
from common import MODELS, OUT, domain_metrics, fidelity, headline_r2, input_name, load  # noqa: E402
from aimet_onnx.quantsim import QuantizationSimModel, load_encodings_to_sim  # noqa: E402
import importlib  # noqa: E402

q = importlib.import_module("05_aimet_quantize")

EXPORTS = {"W8A16_final": ("final", "int8", "int16"), "W8A8": ("w8a8", "int8", "int8")}


def sim_from_export(d, param, act, calib):
    model = onnx.load(os.path.join(d, "model.onnx"))
    n = input_name(model)
    sim = QuantizationSimModel(model, param_type=param, activation_type=act, config_file=q.CONFIG,
                               dummy_input={n: calib[:1]})
    load_encodings_to_sim(sim, os.path.join(d, "model.encodings"), strict=False)
    return sim


def agreement(a, b):
    return dict(agree=float(np.mean(a == b)), fp32_flags=int(a.sum()), int_flags=int(b.sum()),
                both=int((a & b).sum()))


def main(names):
    allres = {}
    for name in names:
        out = os.path.join(OUT, name)
        calib, _ = load(name, "calib")
        Xt, yt = load(name, "test")
        ref = np.load(os.path.join(out, "fp32", "out_test.npy"))
        thr = np.load(os.path.join(out, "anomaly_thresholds_W.npy"))
        f_fp = np.load(os.path.join(out, "fp32", "flags_test.npy"))
        fi_fp = np.load(os.path.join(out, "fp32", "flags_injected_test.npy"))
        res = {"FP32": dict(metrics=domain_metrics(ref, yt, name), headline_R2=headline_r2(domain_metrics(ref, yt, name), name))}
        for tag, (sub, p, a) in EXPORTS.items():
            sim = sim_from_export(os.path.join(out, sub), p, a, calib)
            pred = q.run(sim.session, Xt)
            os.makedirs(os.path.join(out, "sim_" + tag), exist_ok=True)
            np.save(os.path.join(out, "sim_" + tag, "out_test.npy"), pred)
            m = domain_metrics(pred, yt, name)
            ev = anomaly.evaluate(pred, yt, Xt, name, thr)
            res[tag] = dict(fidelity=fidelity(ref, pred), metrics=m, headline_R2=headline_r2(m, name),
                            delta_headline_R2=headline_r2(m, name) - res["FP32"]["headline_R2"],
                            anomaly_natural=agreement(f_fp, ev["flags"]),
                            anomaly_injected=agreement(fi_fp, ev["flags_injected"]),
                            injected_detection_rate=ev["injected_detection_rate"])
            r = res[tag]
            print(f"[{name}] {tag:12s} TEST SQNR {r['fidelity']['SQNR_dB']:.2f} dB  max|d| {r['fidelity']['max_abs']:.4f}  "
                  f"R2 {r['headline_R2']:.4f} (FP32 {res['FP32']['headline_R2']:.4f}, d={r['delta_headline_R2']:+.4f})  "
                  f"flag agree {r['anomaly_natural']['agree']:.4f} / injected {r['anomaly_injected']['agree']:.4f}", flush=True)
        allres[name] = res
        with open(os.path.join(out, "sim_test_results.json"), "w") as f:
            json.dump(res, f, indent=2)
    return allres


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
