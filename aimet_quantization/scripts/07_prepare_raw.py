"""P4: write float32 .raw inputs + input lists for QAIRT (host quantizer) and qnn-net-run (device).

data/raw/<model>/calib/*.raw + calib_list.txt   (paths as seen inside the qairt container: /local/...)
data/raw/<model>/test/*.raw  + test_list.txt    (paths relative to the device workdir)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import AQ, MODELS, load  # noqa: E402

N_CALIB_QAIRT = 512          # subset of the AIMET calibration windows, for LSTM-internal encodings
CONTAINER_ROOT = "/local/aimet_quantization"


def main(names):
    for name in names:
        base = os.path.join(AQ, "data", "raw", name)
        calib, _ = load(name, "calib")
        Xt, _ = load(name, "test")
        for split, X in (("calib", calib[np.linspace(0, len(calib) - 1, N_CALIB_QAIRT).astype(int)]), ("test", Xt)):
            d = os.path.join(base, split)
            os.makedirs(d, exist_ok=True)
            lines = []
            for i, x in enumerate(X):
                fn = f"{i:05d}.raw"
                x.astype(np.float32).tofile(os.path.join(d, fn))
                lines.append(f"{CONTAINER_ROOT}/data/raw/{name}/calib/{fn}" if split == "calib" else f"{name}/test/{fn}")
            with open(os.path.join(base, f"{split}_list.txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
            print(f"[{name}] {split}: {len(X)} raw files")


if __name__ == "__main__" and "--encodings" not in sys.argv:
    main(sys.argv[1:] or list(MODELS))


# ---- AIMET encodings -> QAIRT overrides ---------------------------------------------------------------------
def constant_tensors(model):
    """Tensors computable from initializers only (the converter folds them into static params)."""
    const = {i.name for i in model.graph.initializer}
    changed = True
    while changed:
        changed = False
        for n in model.graph.node:
            if all((i in const) or i == "" for i in n.input) and not set(n.output) <= const:
                const.update(n.output)
                changed = True
    return const


def encodings_for_qairt(export_dir):
    """Drop activation encodings on constant-folded tensors (raw LSTM weights, gate-reordered W/R/B, zero
    initial states). QAIRT folds them into the LSTM's static weights/bias and quantizes those itself
    (8-bit weights, 32-bit bias); overriding them as 'activations' makes qairt-quantizer fail
    ("Activation bitwidth conversion from 8 to 32"). Gemm weight (param) encodings are kept."""
    import json
    import onnx
    model = onnx.load(os.path.join(export_dir, "model.onnx"))
    const = constant_tensors(model)
    enc = json.load(open(os.path.join(export_dir, "model.encodings")))
    act = enc["activation_encodings"]
    key = (lambda e: e["name"]) if isinstance(act, list) else None        # format 1.0.0 (list) vs 0.6.1 (dict)
    names = [key(e) for e in act] if key else list(act)
    dropped = sorted(n for n in names if n in const)
    enc["activation_encodings"] = ([e for e in act if key(e) not in const] if key
                                   else {k: v for k, v in act.items() if k not in const})
    # LSTM hidden outputs: the [-1, 1] prior is right for the AIMET simulation, but in the QNN graph (LSTM expanded
    # into per-step ops) these tensors are aliased by per-step split/reshape views that keep the *calibrated* scale;
    # overriding them made the HTP read values with the wrong scale (x1.44). Let QNN calibrate them (16-bit).
    import importlib
    q5 = importlib.import_module("05_aimet_quantize")
    hidden = q5.lstm_hidden_tensors(model)
    if key:
        enc["activation_encodings"] = [e for e in enc["activation_encodings"] if key(e) not in hidden]
    else:
        enc["activation_encodings"] = {k: v for k, v in enc["activation_encodings"].items() if k not in hidden}
    dropped += sorted(hidden & set(names))
    with open(os.path.join(export_dir, "model.qairt.encodings"), "w") as f:
        json.dump(enc, f, indent=2)
    params = enc["param_encodings"]
    names = [key(e) for e in enc["activation_encodings"]] if key else list(enc["activation_encodings"])
    return dropped, sorted(names), sorted(key(e) for e in params) if key else sorted(params)


if __name__ == "__main__" and "--encodings" in sys.argv:
    from common import OUT
    for name in MODELS:
        for sub in ("final", "w8a8"):
            d, kept, params = encodings_for_qairt(os.path.join(OUT, name, sub))
            print(f"[{name}/{sub}] dropped {d}\n    kept activations {kept}\n    params {params}")
