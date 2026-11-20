"""Per-channel experiment (plan_19112026.md part A): the dense layers of all_sensors_1h rewritten so their per-channel
weight encodings go through ops other than QNN FullyConnected, which mis-executes them in QAIRT 2.47 (HTP).

  perchan_mm    Gemm -> MatMul(x, W^T) + Add(b); per-channel scales on the output axis (axis 1 of W^T)
  perchan_conv  Gemm -> Reshape [1,K,1,1] -> Conv 1x1 (W as [N,K,1,1]) -> Reshape [1,N]; per-channel on axis 0

Writes out/all_sensors_1h/<variant>/model.onnx and model.qairt.encodings (copy of the deployed encodings with the two
dense weights replaced by one symmetric 8-bit scale per output channel). Same FP32 weights as the deployed model.
"""
import copy, json, sys
from pathlib import Path
import numpy as np, onnx
from onnx import helper, numpy_helper

BASE = Path(__file__).resolve().parent.parent / "out" / "all_sensors_1h"
src = onnx.load(BASE / "final" / "model.onnx")
enc0 = json.load(open(BASE / "final" / "model.qairt.encodings"))


def per_channel(w_out_rows):
    ents = []
    for row in w_out_rows:
        s = float(np.abs(row).max()) / 128
        ents.append({"bitwidth": 8, "dtype": "int", "is_symmetric": "True", "max": 127 * s, "min": -128 * s,
                     "offset": -128, "scale": s})
    return ents


def build(kind):
    m = copy.deepcopy(src)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    enc = copy.deepcopy(enc0)
    nodes = []
    for n in g.node:
        if n.op_type != "Gemm":
            nodes.append(n)
            continue
        x, wname, bname = n.input
        y = n.output[0]
        w = numpy_helper.to_array(inits[wname])  # [N, K] (transB=1)
        N, K = w.shape
        del enc["param_encodings"][wname]
        if kind == "mm":
            wt = f"{wname}_T"
            g.initializer.append(numpy_helper.from_array(w.T.copy(), wt))  # [K, N]
            mm = f"{y}_mm"
            nodes += [helper.make_node("MatMul", [x, wt], [mm], name=f"{n.name}_matmul"),
                      helper.make_node("Add", [mm, bname], [y], name=f"{n.name}_add")]
            enc["param_encodings"][wt] = per_channel(w)  # one per output column of W^T
        else:
            wc = f"{wname}_conv"
            g.initializer.append(numpy_helper.from_array(w.reshape(N, K, 1, 1).copy(), wc))
            s_in = f"{n.name}_shape_in"; s_out = f"{n.name}_shape_out"
            g.initializer.append(numpy_helper.from_array(np.array([1, K, 1, 1], dtype=np.int64), s_in))
            g.initializer.append(numpy_helper.from_array(np.array([1, N], dtype=np.int64), s_out))
            x4, y4 = f"{y}_x4", f"{y}_y4"
            nodes += [helper.make_node("Reshape", [x, s_in], [x4], name=f"{n.name}_r_in"),
                      helper.make_node("Conv", [x4, wc, bname], [y4], name=f"{n.name}_conv", kernel_shape=[1, 1]),
                      helper.make_node("Reshape", [y4, s_out], [y], name=f"{n.name}_r_out")]
            enc["param_encodings"][wc] = per_channel(w)
        g.initializer.remove(inits[wname])
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(m)
    out = BASE / f"perchan_{kind}"
    out.mkdir(exist_ok=True)
    onnx.save(m, out / "model.onnx")
    json.dump(enc, open(out / "model.qairt.encodings", "w"), indent=1)
    print("wrote", out)


for k in sys.argv[1:] or ["mm", "conv"]:
    build(k)
