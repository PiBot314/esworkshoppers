"""C1: fuse several B3 models (same 96-step window, any input width / horizon) into ONE graph for the NPU.
plan_19112026.md §4 C1, plan_20112026.md Part C.

  python3 forecast_2025/c1_fuse.py OUT.onnx MODEL_A.onnx MODEL_B.onnx [...]

How: each model is 2-layer LSTM (hidden h_k) + dense head. The fused graph has ONE 2-layer LSTM with hidden sum(h_k):
  layer-1 input = the models' past windows concatenated;  W_ih, W_hh, W_ih(l2), W_hh(l2) block-diagonal per gate,
  biases concatenated per gate;  then each model's head reads its own slice of the last hidden state.
So one call per check and one wider multiply per LSTM step instead of K small ones. The off-diagonal blocks are exact
zeros, so the math is identical in FP32. With per-row weight quantization (--use_per_row_quantization
--enable_per_row_quantized_bias; plan_19112026.md Part A + A5) every row keeps its own scale, i.e. each model keeps its
own weight range after fusion.
Checks: FP32 outputs of the fused graph vs each original ONNX on random + real-scale inputs (max abs diff).
Writes OUT.onnx and OUT.json (inputs per model, slices, check results).
"""
import json, sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnx import numpy_helper

L = 96


def load(path):
    m = onnx.load(str(path))
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    lstms = [n for n in m.graph.node if n.op_type == "LSTM"]
    assert len(lstms) == 2, f"{path}: expected 2 LSTM nodes"
    layers = []
    for n in lstms:
        W, R, B = (init[x] for x in n.input[1:4])            # ONNX gate order i, o, f, c
        layers.append((W[0], R[0], B[0]))
    meta = json.load(open(Path(path).with_suffix(".json")))
    head = {k: torch.tensor(init[k]) for k in ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")}
    return dict(layers=layers, head=head, meta=meta, hidden=layers[0][1].shape[1], F=layers[0][0].shape[1],
                solar=meta.get("family") == "solar")


def onnx_to_torch_gates(M, H):
    """[4H, ...] in ONNX order (i, o, f, c) -> list of 4 blocks in torch order (i, f, g=c, o)."""
    i, o, f, c = (M[k * H:(k + 1) * H] for k in range(4))
    return [i, f, c, o]


def block_diag_gates(mats, hs, in_dims):
    """Per gate, block-diagonal over models; returns [4*sum(h), sum(in)] in torch gate order."""
    Hs, Is = sum(hs), sum(in_dims)
    out = np.zeros((4 * Hs, Is), np.float32)
    r0 = np.cumsum([0] + hs); c0 = np.cumsum([0] + in_dims)
    for k, (M, h) in enumerate(zip(mats, hs)):
        for g, blk in enumerate(onnx_to_torch_gates(M, h)):
            out[g * Hs + r0[k]: g * Hs + r0[k] + h, c0[k]: c0[k] + in_dims[k]] = blk
    return out


def cat_gates(vecs, hs):
    Hs = sum(hs)
    out = np.zeros(4 * Hs, np.float32)
    r0 = np.cumsum([0] + hs)
    for k, (v, h) in enumerate(zip(vecs, hs)):
        for g, blk in enumerate(onnx_to_torch_gates(v, h)):
            out[g * Hs + r0[k]: g * Hs + r0[k] + h] = blk
    return out


class Fused(nn.Module):
    def __init__(self, models):
        super().__init__()
        hs = [m["hidden"] for m in models]
        Fs = [m["F"] for m in models]
        self.hs, self.Fs = hs, Fs
        self.lstm = nn.LSTM(sum(Fs), sum(hs), num_layers=2, batch_first=True)
        sd = {}
        for l in range(2):
            ins = Fs if l == 0 else hs
            sd[f"weight_ih_l{l}"] = block_diag_gates([m["layers"][l][0] for m in models], hs, ins)
            sd[f"weight_hh_l{l}"] = block_diag_gates([m["layers"][l][1] for m in models], hs, hs)
            sd[f"bias_ih_l{l}"] = cat_gates([m["layers"][l][2][:4 * h] for m, h in zip(models, hs)], hs)
            sd[f"bias_hh_l{l}"] = cat_gates([m["layers"][l][2][4 * h:] for m, h in zip(models, hs)], hs)
        self.lstm.load_state_dict({k: torch.tensor(v) for k, v in sd.items()})
        self.heads = nn.ModuleList()
        for m in models:
            fc1 = nn.Linear(m["head"]["fc1.weight"].shape[1], m["head"]["fc1.weight"].shape[0])
            fc2 = nn.Linear(m["head"]["fc2.weight"].shape[1], m["head"]["fc2.weight"].shape[0])
            fc1.weight.data, fc1.bias.data = m["head"]["fc1.weight"], m["head"]["fc1.bias"]
            fc2.weight.data, fc2.bias.data = m["head"]["fc2.weight"], m["head"]["fc2.bias"]
            self.heads.append(nn.Sequential(fc1, nn.ReLU(), fc2))
        self.solar = [m["solar"] for m in models]

    def forward(self, *xs):
        K = len(self.heads)
        pasts, futs, nodes = xs[:K], xs[K:2 * K], xs[2 * K:]
        out, _ = self.lstm(torch.cat(pasts, -1))
        last = out[:, -1]
        outs, h0 = [], 0
        for k in range(K):
            h = last[:, h0:h0 + self.hs[k]]; h0 += self.hs[k]
            o = self.heads[k](torch.cat([h, futs[k].flatten(1), nodes[k]], 1))
            if self.solar[k]:
                o = torch.relu(o) * (futs[k][..., 0] > 0)
            outs.append(o)
        return tuple(outs)


def main(out_path, paths):
    models = [load(p) for p in paths]
    net = Fused(models).eval()
    K = len(models)
    names = [f"past_{k}" for k in range(K)] + [f"future_{k}" for k in range(K)] + [f"node_{k}" for k in range(K)]
    shapes = ([(L, m["F"]) for m in models] + [(m["meta"]["horizon"], len(m["meta"]["inputs"]["future"])) for m in models]
              + [(m["meta"]["inputs"]["node"],) for m in models])
    dummy = tuple(torch.zeros(1, *s) for s in shapes)
    torch.onnx.export(net, dummy, out_path, input_names=names, output_names=[f"forecast_{k}" for k in range(K)],
                      dynamic_axes={n: {0: "b"} for n in names + [f"forecast_{k}" for k in range(K)]},
                      opset_version=17, dynamo=False)
    # check vs the originals (FP32, ONNX Runtime)
    fused = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    B = 64
    feeds = {n: rng.uniform(0, 1, (B, *s)).astype(np.float32) for n, s in zip(names, shapes)}
    for k in range(K):                                       # node one-hot, clear-sky > 0 for half the steps
        nn_ = shapes[2 * K + k][0]
        feeds[f"node_{k}"] = np.eye(nn_, dtype=np.float32)[rng.integers(0, nn_, B)]
        feeds[f"future_{k}"][:, ::2, 0] = 0
    fo = fused.run(None, feeds)
    diffs = []
    for k, p in enumerate(paths):
        s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        ref = s.run(None, {"past": feeds[f"past_{k}"], "future": feeds[f"future_{k}"], "node": feeds[f"node_{k}"]})[0]
        diffs.append(float(np.abs(ref - fo[k]).max()))
    params_sep = sum(sum(x.size for x in (W, R, Bb)) for m in models for (W, R, Bb) in m["layers"])
    info = {"models": [str(p) for p in paths], "hidden": net.hs, "input_width": net.Fs, "fused_hidden": sum(net.hs),
            "max_abs_diff_vs_original": diffs, "lstm_params_separate": int(params_sep),
            "lstm_params_fused_incl_zero_blocks": int(sum(p.numel() for p in net.lstm.parameters())),
            "inputs": names}
    json.dump(info, open(Path(out_path).with_suffix(".json"), "w"), indent=1)
    print(json.dumps(info, indent=1))
    assert max(diffs) < 1e-4, "fused graph differs from the originals"


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
