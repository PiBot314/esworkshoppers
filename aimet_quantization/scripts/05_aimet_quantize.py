"""P3: AIMET post-training quantization of the frozen ONNX models (no retraining).

Per model:
  1. Manual QuantSim grid on a validation subset (fidelity = SQNR of outputs vs FP32 on identical windows):
       W32A16 (activations only), W8A32 (weights only), W8A8, W8A16, W8A16 + per-channel Gemm
  2. AutoQuant (aimet-onnx 2.12, W8A16, HTP v75 config) with target = SQNR(W32A16) - 1 dB
  3. Best W8A16 candidate -> export model.onnx + model.encodings to out/<model>/final/
Results -> out/<model>/aimet_results.json
"""
import json
import os
import shutil
import sys
import time

import numpy as np
import onnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import AQ, CONFIG, MODELS, OUT, input_name, load, onnx_path, sqnr_db  # noqa: E402

from aimet_onnx.quantsim import QuantizationSimModel  # noqa: E402
from aimet_onnx._deprecated.auto_quant_v2 import AutoQuant  # noqa: E402
from aimet_onnx.adaround.adaround_weight import AdaroundParameters  # noqa: E402
from aimet_common.defs import QuantScheme  # noqa: E402
from aimet_common import libpymo  # noqa: E402

AQ_TARGET_DB = 60.0
ADAROUND_ITERS = 10000           # AIMET's documented default; 2.12's deprecated AutoQuant leaves it as None (crash)
N_VAL = 600                      # validation subset used for selection (evenly spaced over the split)
PC_CONFIG = os.path.join(AQ, "configs", "htp_v75_per_channel_gemm_lstm_per_tensor.json")


def make_pc_config():
    import aimet_common.quantsim_config as qc
    src = os.path.join(list(qc.__path__)[0], "htp_quantsim_config_v75_per_channel_linear.json")
    cfg = json.load(open(src))
    cfg.setdefault("op_type", {})["LSTM"] = {"per_channel_quantization": "False"}
    json.dump(cfg, open(PC_CONFIG, "w"), indent=2)


# ---- range priors ---------------------------------------------------------------------------------------
# Calibration-only min/max under-covers two tensor groups (found by enabling quantizers one at a time):
#   * LSTM hidden outputs: h = o * tanh(c) is mathematically bounded to (-1, 1)  -> fix encoding to [-1, 1]
#   * model outputs (fractions of capacity): val/test forecasts go beyond the calib range (clipping of the
#     negative night-time forecasts)                                             -> calib range + 10 % headroom
LAYOUT_OPS = {"Transpose", "Reshape", "Squeeze", "Unsqueeze", "Gather", "Slice", "Identity"}
OUTPUT_HEADROOM = 0.10


# ---- LSTM weights are parameters ----------------------------------------------------------------------------
# PyTorch's export reorders the LSTM gate weights with Slice/Concat, so the raw initializers reach the LSTM node
# through ops and AIMET treats them as *activations* (16-bit in W8A16). On the HTP the converter folds that
# constant subgraph into the LSTM's static weights, which are quantized as params. For an honest W8A16 and a
# simulation that matches the target, every quantizer on that constant W/R subgraph is set to 8-bit symmetric.
CONST_OPS = {"Unsqueeze", "Concat", "Slice", "Reshape", "Transpose", "Squeeze", "Identity"}


def lstm_weight_tensors(model):
    producer = {o: n for n in model.graph.node for o in n.output}
    found, stack = set(), [n.input[i] for n in model.graph.node if n.op_type == "LSTM" for i in (1, 2)]
    while stack:
        t = stack.pop()
        if t in found:
            continue
        found.add(t)
        p = producer.get(t)
        if p is not None and p.op_type in CONST_OPS:
            stack.extend(i for i in p.input if i)
    return found


def lstm_weights_as_params(sim, model, bw=8):
    keys = lstm_weight_tensors(model) & set(sim.qc_quantize_op_dict)
    for k in keys:
        qz = sim.qc_quantize_op_dict[k]
        if qz.enabled and qz.data_type.name != "float":
            qz.set_bitwidth(bw)
            qz.use_symmetric_encodings = True
    return sorted(keys)


def lstm_hidden_tensors(model):
    consumers = {}
    for n in model.graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    found, stack = set(), [n.output[0] for n in model.graph.node if n.op_type == "LSTM"]
    while stack:
        t = stack.pop()
        found.add(t)
        for c in consumers.get(t, []):
            if c.op_type in LAYOUT_OPS:
                stack.extend(c.output)
    return found


def set_range(qz, lo, hi):
    bw = qz.bitwidth
    delta = (hi - lo) / (2 ** bw - 1)
    offset = round(lo / delta)
    enc = libpymo.TfEncoding()
    enc.bw, enc.delta, enc.offset = bw, delta, offset
    enc.min, enc.max = offset * delta, offset * delta + delta * (2 ** bw - 1)
    qz.load_encodings([enc])
    qz.freeze_encodings()


def apply_range_priors(sim, model):
    changed = {}
    hidden = lstm_hidden_tensors(model)
    outputs = {o.name for o in model.graph.output}
    for name, qz in sim.qc_quantize_op_dict.items():
        if not qz.enabled or qz.data_type.name == "float" or not qz.get_encodings():
            continue
        if name in hidden:
            set_range(qz, -1.0, 1.0)
            changed[name] = (-1.0, 1.0)
        elif name in outputs:
            e = qz.get_encodings()[0]
            pad = OUTPUT_HEADROOM * (e.max - e.min)
            set_range(qz, e.min - pad, e.max + pad)
            changed[name] = (round(e.min - pad, 4), round(e.max + pad, 4))
    return changed


class AutoQuantWithPriors(AutoQuant):
    """AutoQuant 2.12 with the same range priors applied to every QuantSim it creates."""

    def _create_quantsim_and_encodings(self, model, *args, **kwargs):
        sim = super()._create_quantsim_and_encodings(model, *args, **kwargs)
        param_bw = kwargs.get("param_bw") or self._quantsim_params["param_bw"]
        if param_bw <= 8 and lstm_weights_as_params(sim, self.fp32_model.model, param_bw):
            sim.compute_encodings(self.forward_pass_callback, None)   # re-calibrate the re-typed quantizers
        # tensor names from the un-instrumented FP32 graph (QuantSim renames graph outputs)
        apply_range_priors(sim, self.fp32_model.model)
        return sim


def run(session, X):
    n = input_name(session)
    return np.concatenate([session.run(None, {n: X[i:i + 1]})[0] for i in range(len(X))])


def build_sim(name, param, act, calib, config=CONFIG, scheme="min_max", priors=True):
    model = onnx.load(onnx_path(name))
    n = input_name(model)
    sim = QuantizationSimModel(model, param_type=param, activation_type=act, quant_scheme=scheme,
                               config_file=config, dummy_input={n: calib[:1]})
    if param == "int8":
        lstm_weights_as_params(sim, onnx.load(onnx_path(name)), 8)
    sim.compute_encodings([{n: calib[i:i + 1]} for i in range(len(calib))])
    if priors:
        sim.range_priors = apply_range_priors(sim, onnx.load(onnx_path(name)))
    return sim


def main(names):
    if not os.path.exists(PC_CONFIG):
        make_pc_config()
    for name in names:
        out = os.path.join(OUT, name)
        calib, _ = load(name, "calib")
        Xv, _ = load(name, "val")
        sel = np.linspace(0, len(Xv) - 1, N_VAL).astype(int)
        Xs = Xv[sel]
        ref = np.load(os.path.join(out, "fp32", "out_val.npy"))[sel]
        results = {}

        grid = {
            "W32A16_noPriors": ("float32", "int16", CONFIG, False),
            "W8A8_noPriors": ("int8", "int8", CONFIG, False),
            "W8A16_noPriors": ("int8", "int16", CONFIG, False),
            "W32A16": ("float32", "int16", CONFIG, True),
            "W8A32": ("int8", "float32", CONFIG, True),
            "W8A8": ("int8", "int8", CONFIG, True),
            "W8A16": ("int8", "int16", CONFIG, True),
            "W8A16_pcGemm": ("int8", "int16", PC_CONFIG, True),
        }
        sims = {}
        for tag, (p, a, cfg, pri) in grid.items():
            t = time.time()
            sim = build_sim(name, p, a, calib, cfg, priors=pri)
            pred = run(sim.session, Xs)
            results[tag] = dict(val_SQNR_dB=sqnr_db(ref, pred), val_max_abs=float(np.abs(ref - pred).max()),
                                seconds=round(time.time() - t, 1))
            sims[tag] = sim
            if pri and getattr(sim, "range_priors", None):
                results[tag]["range_priors"] = sim.range_priors
            print(f"[{name}] {tag:13s} val SQNR {results[tag]['val_SQNR_dB']:6.2f} dB  "
                  f"max|d| {results[tag]['val_max_abs']:.4f}  ({results[tag]['seconds']} s)", flush=True)

        # ---- AutoQuant -------------------------------------------------------------------------------
        # Target 60 dB: below the activations-only ceiling (W32A16, so AutoQuant's W32 early-exit check passes)
        # but above every plain 8-bit-weight result, so BN-fold -> CLE -> AdaRound are all tried and the best kept.
        target = min(AQ_TARGET_DB, results["W32A16"]["val_SQNR_dB"] - 1.0)
        aq_dir = os.path.join(out, "autoquant")
        shutil.rmtree(aq_dir, ignore_errors=True)
        os.makedirs(aq_dir)
        n_in = input_name(onnx.load(onnx_path(name)))

        def eval_callback(session, _num_samples):
            return sqnr_db(ref, run(session, Xs))

        t = time.time()
        aq = AutoQuantWithPriors(onnx.load(onnx_path(name)), dummy_input={n_in: calib[:1]},
                       data_loader=[calib[i:i + 1] for i in range(len(calib))],
                       eval_callback=eval_callback, param_bw=8, output_bw=16,
                       quant_scheme=QuantScheme.post_training_tf_enhanced, config_file=CONFIG,
                       results_dir=aq_dir)
        loader = [calib[i:i + 1] for i in range(len(calib))]
        aq.set_adaround_params(AdaroundParameters(loader, num_batches=len(loader),
                                                  default_num_iterations=ADAROUND_ITERS))
        _, aq_score, aq_enc = aq.optimize(allowed_accuracy_drop=100.0 - target)
        results["AutoQuant_W8A16"] = dict(val_SQNR_dB=aq_score, target_dB=target, encoding_path=aq_enc,
                                          seconds=round(time.time() - t, 1))
        print(f"[{name}] AutoQuant     val SQNR {aq_score} dB (target {target:.2f}) -> {aq_enc}", flush=True)

        # ---- pick final W8A16 deliverable ------------------------------------------------------------
        # Per-channel Gemm weights simulate well but mis-execute on the HTP in the QNN path used here (doc 14),
        # so the deliverable is chosen among per-tensor-weight candidates only.
        cands = {"W8A16": results["W8A16"]["val_SQNR_dB"]}
        if aq_score is not None:
            cands["AutoQuant_W8A16"] = aq_score
        best = max(cands, key=cands.get)
        final = os.path.join(out, "final")
        shutil.rmtree(final, ignore_errors=True)
        os.makedirs(final)
        if best == "AutoQuant_W8A16":
            base = aq_enc[:-len(".encodings")]
            shutil.copy(base + ".onnx", os.path.join(final, "model.onnx"))
            shutil.copy(aq_enc, os.path.join(final, "model.encodings"))
        else:
            sims[best].export(final, "model")
        os.makedirs(os.path.join(out, "w8a8"), exist_ok=True)
        sims["W8A8"].export(os.path.join(out, "w8a8"), "model")
        results["final"] = dict(choice=best, candidates=cands)
        print(f"[{name}] final W8A16 deliverable: {best}", flush=True)
        with open(os.path.join(out, "aimet_results.json"), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
