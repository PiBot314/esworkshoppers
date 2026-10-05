"""Run the deployed (AIMET-quantized W8A16) forecaster on the QIDK's Hexagon NPU for one input window,
print the forecast in Watts and, when the actual next reading is known, the residual-anomaly verdict.

Examples
  python scripts/12_qidk_forecast.py --model all_sensors_1h --test-index 120
  python scripts/12_qidk_forecast.py --model single_node --input my_window_1x96x10.npy
  python scripts/12_qidk_forecast.py --model all_sensors_1h --test-index 120 --inject sensor_dropout_zero

Inputs must be prepared exactly like the training data (per-feature MinMax to [0, 1], 15-min steps, the
feature order in model2/qidk_deployment/*_meta.json); --test-index takes a ready window from data/windows.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anomaly  # noqa: E402
from common import MODELS, OUT, load, model_capacities  # noqa: E402

D = "/data/local/tmp/nz_aimet"


def adb(serial, *args, capture=False):
    cmd = ["adb", "-s", serial, *args]
    for _ in range(3):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout
        subprocess.run(["adb", "kill-server"], capture_output=True)
        subprocess.run(["adb", "start-server"], capture_output=True)
        subprocess.run(["timeout", "60", "adb", "wait-for-device"], capture_output=True)
    raise RuntimeError(f"adb failed: {' '.join(cmd)}\n{r.stderr}")


def run_on_qidk(name, x, serial, variant="aimet_w8a16", backend="htp"):
    lib = "libQnnHtp.so" if backend == "htp" else "libQnnCpu.so"
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "window.raw")
        x.astype(np.float32).tofile(raw)
        adb(serial, "shell", f"mkdir -p {D}/{name}/models {D}/{name}/live")
        adb(serial, "push", raw, f"{D}/{name}/live/window.raw")
        adb(serial, "push", os.path.join(OUT, name, "qnn", "lib", "aarch64-android", f"lib{variant}.so"),
            f"{D}/{name}/models/")
        adb(serial, "shell", f"cd {D} && echo {name}/live/window.raw > {name}/live/list.txt && rm -rf {name}/live/out && "
                             f"LD_LIBRARY_PATH={D}/lib ADSP_LIBRARY_PATH={D}/lib ./bin/qnn-net-run --backend lib/{lib} "
                             f"--model {name}/models/lib{variant}.so --input_list {name}/live/list.txt "
                             f"--output_dir {name}/live/out > {name}/live/run.log 2>&1")
        out_name = adb(serial, "shell", f"ls {D}/{name}/live/out/Result_0/").split()[0]
        local = os.path.join(tmp, "out.raw")
        adb(serial, "pull", f"{D}/{name}/live/out/Result_0/{out_name}", local)
        return np.fromfile(local, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), default="all_sensors_1h")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--test-index", type=int, help="use window i of the rebuilt TEST split (actuals known)")
    g.add_argument("--input", help=".npy window of shape [1, 96, F] (or [96, F]), already normalized")
    ap.add_argument("--inject", choices=["panel_damage_40pct", "sensor_dropout_zero", "sensor_spike"],
                    help="demo: corrupt the actual first-domain reading before the anomaly check")
    ap.add_argument("--serial", default=os.environ.get("SERIAL", "d42f06b"))
    ap.add_argument("--variant", default="aimet_w8a16", help="aimet_w8a16 | qairt_w8a16 | aimet_w8a8 | fp32")
    a = ap.parse_args()

    name, spec = a.model, MODELS[a.model]
    if a.test_index is not None:
        X, Y = load(name, "test")
        x, y = X[a.test_index:a.test_index + 1], Y[a.test_index:a.test_index + 1]
    else:
        x = np.load(a.input).astype(np.float32).reshape(1, 96, -1)
        y = None
    backend = "cpu" if a.variant == "fp32" else "htp"
    out = run_on_qidk(name, x, a.serial, a.variant, backend)
    D_ = len(spec["domains"])
    caps = np.asarray(model_capacities(name))
    watts = out.reshape(-1, D_) * caps
    if "Solar" in spec["domains"][0]:
        watts[:, 0] = np.maximum(watts[:, 0], 0.0)        # physical clamp, as in the original runners

    print(f"\nQIDK {backend.upper()} forecast  model={name}  variant={a.variant}")
    print("  step   " + "".join(f"{d:>16s}" for d in spec["domains"]))
    for t in range(min(len(watts), 8) if spec["horizon"] > 8 else len(watts)):
        print(f"  +{15 * (t + 1):4d}m " + "".join(f"{v:14,.0f} W" for v in watts[t]))
    if spec["horizon"] > 8:
        print(f"  ... ({spec['horizon']} steps total; hourly energy below)")
        e = watts.reshape(-1, 4, D_).mean(1)                # kWh per hour = mean W over 4 steps / 1000
        print("  next-24h energy: " + ", ".join(f"{d} {v.sum() / 1000:,.1f} kWh" for d, v in zip(spec["domains"], e.T)))

    if y is not None:
        thr = np.load(os.path.join(OUT, name, "anomaly_thresholds_W.npy"))
        res, T, P = anomaly.residuals(out[None, :], y, name)
        if a.inject:
            cap0 = caps[0]
            T[0, 0] = {"panel_damage_40pct": T[0, 0] * 0.6, "sensor_dropout_zero": 0.0,
                       "sensor_spike": T[0, 0] + 0.5 * cap0}[a.inject]
            res = T - P
        h = anomaly.target_hour_ist(x, name)
        f = anomaly.flags(res, h, thr)[0]
        print(f"\nAnomaly check at +15 min (IST hour {h[0]:02d}, K={anomaly.K}):")
        for j, d in enumerate(spec["domains"]):
            print(f"  {d:11s} actual {T[0, j]:12,.0f} W  forecast {P[0, j]:12,.0f} W  residual {res[0, j]:+12,.0f} W  "
                  f"limit ±{thr[1][h[0], j]:,.0f} W  -> {'ANOMALY' if f[j] else 'ok'}")


if __name__ == "__main__":
    main()
