"""B5: score a board run made with fused/<STEM>_board/run_on_board.sh (20 Nov).

  python3 forecast_2025/b5_score_board.py STEM

SQNR of each model's output on the board vs FP32 ONNX Runtime and vs the x86 HTP emulator (same windows, same
conversion), plus mean execute time from qnn-net-run's basic profiling if present. -> fused/<STEM>_board/board.json
Gate (plan_19112026 B5): board SQNR vs FP32 >= 44 dB per model.
"""
import json, sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent


def sqnr(ref, x):
    return float(10 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - x) ** 2), 1e-30)))


def main(stem):
    F = HERE / "fused"
    B = F / f"{stem}_board"
    info = json.load(open(F / f"{stem}.json"))
    K = len(info["models"])
    lines = (B / "test_list.txt").read_text().split("\n")
    lines = [l for l in lines if l.strip()]
    sess = ort.InferenceSession(str(F / f"{stem}.onnx"), providers=["CPUExecutionProvider"])
    shapes = {i.name: [1] + list(i.shape[1:]) for i in sess.get_inputs()}
    feeds = {n: [] for n in shapes}
    for l in lines:
        for part in l.split():
            name, path = part.split(":=")
            feeds[name].append(np.fromfile(B / "test" / Path(path).name, np.float32).reshape(shapes[name]))
    feeds = {n: np.concatenate(v) for n, v in feeds.items()}
    ref = sess.run(None, feeds)
    res = {"windows": len(lines), "models": info["models"], "sqnr_vs_fp32_db": [], "sqnr_vs_emulator_db": []}
    for k in range(K):
        board = np.stack([np.fromfile(B / "device_out" / f"Result_{i}" / f"forecast_{k}.raw", np.float32)
                          .reshape(ref[k].shape[1:]) for i in range(len(lines))])
        emu = np.stack([np.fromfile(F / f"{stem}_emu" / "out_perrow" / f"Result_{i}" / f"forecast_{k}.raw", np.float32)
                        .reshape(ref[k].shape[1:]) for i in range(len(lines))])
        res["sqnr_vs_fp32_db"].append(round(sqnr(ref[k], board), 2))
        res["sqnr_vs_emulator_db"].append(round(sqnr(emu, board), 2))
    res["pass_44dB"] = all(s >= 44 for s in res["sqnr_vs_fp32_db"])
    log = B / "run.log"
    if log.exists():
        res["run_log_tail"] = log.read_text()[-1500:]
    json.dump(res, open(B / "board.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "run_log_tail"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
