"""C1 accuracy check without the board: fused model -> QNN W8A16 (per-tensor vs per-row) -> x86 HTP emulator -> SQNR.
plan_19112026.md Part A (the x86 HTP emulator reproduces the board: 50.1 vs 48.4 dB) and §4 C1.

  python3 forecast_2025/c1_emulator.py FUSED.onnx [N_TEST] [N_CAL] [CAL_SPLIT]
  (20 Nov: N_CAL / CAL_SPLIT added; 100 'val' windows held no holiday and few gaps, which clipped grid activations)

1. Real test windows (b3_guided_lstm.Windows, split 'test') for each model in FUSED.json -> .raw files + input list
   (calibration = the val split, as for the deployed models).
2. In the qairt-fixed:2.47 container: qnn-onnx-converter (--expand_lstm_op_structure --preserve_io layout, W8A16,
   bias 32) twice: 'pertensor' and 'perrow' (--use_per_row_quantization --enable_per_row_quantized_bias);
   qnn-model-lib-generator -t x86_64-linux-clang; qnn-net-run with the HTP backend library for x86 (emulator).
3. SQNR of each model's output vs FP32 ONNX Runtime on the same windows. Results -> FUSED.emu.json
"""
import json, os, subprocess, sys, zlib
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
import pandas as pd

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
sys.path.insert(0, str(HERE))
import b3_guided_lstm as B   # noqa: E402

IMAGE = os.environ.get("IMAGE", "qairt-fixed:2.47")


def windows_for(model_json, split, n, seed):
    m = json.load(open(model_json))
    fam, H = m["family"], m["horizon"]
    df = pd.read_parquet(B.DATA / "data_15min.parquet")
    if fam == "solar":
        ser, *_ , wp, wc = B.series_solar(df, H)
        nn_ = len(B.PLANTS)
    else:
        ser, *_ , wp, wc = B.series_grid(df, H)
        nn_ = 1
    if m["variant"].startswith("noweather"):
        for s in ser:
            s["P"][:, wp] = 0; s["C"][:, wc] = 0
    W = B.Windows(ser, H, nn_)
    ix = torch.cat([W.idx[s] for s in split.split("+")])
    pick = ix[np.random.default_rng(seed).choice(len(ix), n, replace=False)]
    past, fut, oh, y, day = W.batch(pick)
    return past.cpu().numpy(), fut.cpu().numpy(), oh.cpu().numpy()


def sqnr(ref, x):
    return float(10 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - x) ** 2), 1e-30)))


def main(fused, n=50, n_cal=100, cal_split="val"):
    fused = Path(fused).resolve()
    info = json.load(open(fused.with_suffix(".json")))
    K = len(info["models"])
    work = fused.parent / (fused.stem + "_emu")
    for sub in ("raw_test", "raw_cal"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    feeds = {}
    lists = {}
    for sub in ("raw_test", "raw_cal"):                 # stale .raw files from an earlier run would be listed otherwise
        for f in (work / sub).glob("*.raw"):
            f.unlink()
    for split, sub, nn_ in (("test", "raw_test", n), (cal_split, "raw_cal", n_cal)):
        per_in = {}
        for k, mp in enumerate(info["models"]):
            past, fut, oh = windows_for(Path(mp).with_suffix(".json"), split, nn_,
                                        seed=zlib.crc32(Path(mp).name.encode()))   # same windows fused or alone
            per_in[f"past_{k}"], per_in[f"future_{k}"], per_in[f"node_{k}"] = past, fut, oh
        lines = []
        for i in range(nn_):
            parts = []
            for name, arr in per_in.items():
                p = work / sub / f"{name}_{i:04d}.raw"
                arr[i:i + 1].astype(np.float32).tofile(p)
                parts.append(f"{name}:={'/local/' + str(p.relative_to(FINAL))}")
            lines.append(" ".join(parts))
        (work / f"{sub}.txt").write_text("\n".join(lines) + "\n")
        if split == "test":
            feeds = per_in
    dims = " ".join(f"--input_dim {nm} {','.join(str(d) for d in (1,) + feeds[nm].shape[1:])}" for nm in feeds)
    rel = lambda p: "/local/" + str(Path(p).relative_to(FINAL))
    script = f"""
source /usr/venv/bin/activate; source $QAIRT_SDK_ROOT/bin/envsetup.sh >/dev/null
set -eo pipefail
cd {rel(work)}
COMMON="--expand_lstm_op_structure --preserve_io layout {dims} --input_list {rel(work / 'raw_cal.txt')} --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32"
for V in pertensor perrow; do
  EXTRA=""; [ $V = perrow ] && EXTRA="--use_per_row_quantization --enable_per_row_quantized_bias"
  qnn-onnx-converter --input_network {rel(fused)} $COMMON $EXTRA --output_path $V.cpp > conv_$V.log 2>&1
  qnn-model-lib-generator -c $V.cpp -b $V.bin -o lib_$V -t x86_64-linux-clang > libgen_$V.log 2>&1
  rm -rf out_$V
  qnn-net-run --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so --model lib_$V/x86_64-linux-clang/lib$V.so \\
      --input_list {rel(work / 'raw_test.txt')} --output_dir out_$V > run_$V.log 2>&1
  echo built and ran $V
done
"""
    r = subprocess.run(["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
                        "-v", f"{FINAL}:/local:z", IMAGE, "bash", "-c", script], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:])
    if r.returncode:
        raise SystemExit("container step failed; see the logs in " + str(work))
    sess = ort.InferenceSession(str(fused), providers=["CPUExecutionProvider"])
    ref = sess.run(None, feeds)
    res = {"windows": n, "calibration": {"split": cal_split, "windows": n_cal}, "models": info["models"], "sqnr_db": {},
           "sqnr_db_per_window_median": {}}
    for V in ("pertensor", "perrow"):
        res["sqnr_db"][V] = []
        for k in range(K):
            outs = []
            for i in range(n):
                f = work / f"out_{V}" / f"Result_{i}" / f"forecast_{k}.raw"
                outs.append(np.fromfile(f, np.float32).reshape(ref[k].shape[1:]))
            res["sqnr_db"][V].append(round(sqnr(ref[k], np.stack(outs)), 2))
            res["sqnr_db_per_window_median"].setdefault(V, []).append(
                round(float(np.median([sqnr(ref[k][i], outs[i]) for i in range(n)])), 2))
    json.dump(res, open(fused.with_suffix(".emu.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    # python3 c1_emulator.py FUSED.onnx [N_TEST] [N_CAL] [CAL_SPLIT, e.g. val or train+val]
    a = sys.argv
    main(a[1], int(a[2]) if len(a) > 2 else 50, int(a[3]) if len(a) > 3 else 100, a[4] if len(a) > 4 else "val")
