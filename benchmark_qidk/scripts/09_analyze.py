#!/usr/bin/env python3
"""Collect accuracy, latency and power results into results/summary.json and results/summary.md."""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.dirname(HERE)
FINAL = os.path.dirname(B)
sys.path.insert(0, os.path.join(FINAL, "common"))
import common_data as cd  # noqa: E402

D, R = os.path.join(B, "data"), os.path.join(B, "results")
MODELS = json.load(open(os.path.join(HERE, "models.json")))
info = json.load(open(os.path.join(D, "info.json")))
CAPS = info["capacities"]
LABEL = {"htp_w8a16": "NPU W8A16 (QNN HTP)", "cpu_fp32": "CPU FP32 (QNN CPU)", "ort_fp32": "CPU FP32 (ONNX Runtime)",
         "ort_t1": "CPU FP32 (ONNX Runtime, 1 thread)", "ort_t4": "CPU FP32 (ONNX Runtime, 4 threads)",
         "htp_w8a16_powersaver": "NPU W8A16, power_saver profile",
         "htp_w8a8": "NPU W8A8 (QNN HTP)", "htp_fp16": "NPU FP16 (QNN HTP)"}


def sqnr(ref, x):
    return float(10 * np.log10(np.sum(ref ** 2) / max(np.sum((ref - x) ** 2), 1e-30)))


def accuracy():
    out = {}
    for name, m in MODELS.items():
        H = m["horizon"]
        y = np.load(os.path.join(D, f"y_test_h{H}.npy"))
        ref = np.load(os.path.join(D, f"ref_{name}.npy")).reshape(len(y), -1)
        res = {"pc_onnx_fp32": {"metrics": cd.evaluate(np.clip(ref, 0, None), y, CAPS, H)}}
        if H == 96:
            res["pc_onnx_fp32"]["metrics_first60"] = cd.evaluate(np.clip(ref, 0, None), y, CAPS, H, steps=4)
        for v in ("htp_w8a16", "htp_w8a8", "htp_fp16", "cpu_fp32", "ort_fp32"):
            p = os.path.join(R, "outputs", f"{name}_{v}.bin")
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                continue
            x = np.fromfile(p, dtype=np.float32)
            if x.size != ref.size:
                res[v] = {"error": f"size {x.size} != {ref.size}"}
                continue
            x = x.reshape(ref.shape)
            r = {"sqnr_db_vs_fp32": round(sqnr(ref, x), 1), "max_abs_diff": float(np.abs(ref - x).max()),
                 "metrics": cd.evaluate(np.clip(x, 0, None), y, CAPS, H)}
            if H == 96:
                r["metrics_first60"] = cd.evaluate(np.clip(x, 0, None), y, CAPS, H, steps=4)
            res[v] = r
        out[name] = res
    # LightGBM (24 h): its ONNX conversion matches LightGBM to 1e-6, so the PC result stands for the device one
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(os.path.join(B, "models", "lightgbm_24h.onnx"), providers=["CPUExecutionProvider"])
        X = np.load(os.path.join(D, "X_test_last.npy"))
        y = np.load(os.path.join(D, "y_test_h96.npy"))
        p = np.concatenate([s.run(None, {"input": X[i:i + 1]})[0] for i in range(len(y))])
        out["lightgbm_24h"] = {"pc_onnx_fp32": {"metrics": cd.evaluate(np.clip(p, 0, None), y, CAPS, 96),
                                                "metrics_first60": cd.evaluate(np.clip(p, 0, None), y, CAPS, 96, steps=4)}}
    except Exception as e:  # noqa: BLE001
        out["lightgbm_24h"] = {"error": str(e)}
    return out


def parse_qnn(txt):
    def avg(key):
        sec = txt.split("Execute Stats (Average):")[1].split("Execute Stats (Min)")[0]
        m = re.search(re.escape(key) + r"\s*:\s*([\d.]+)\s*us", sec)
        return float(m.group(1)) / 1000 if m else None
    ips = re.search(r"NetRun IPS[^:]*:\s*([\d.]+)", txt)
    return {"netrun_ms": avg("NetRun"), "accel_exec_ms": avg("Backend (Accelerator (execute) time)"),
            "qnn_exec_ms": avg("Backend (QNN (execute) time)"), "ips": float(ips.group(1)) if ips else None}


def latency():
    out = {}
    d = os.path.join(R, "latency")
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        name, cfg = re.match(r"(.+?)_(htp_.+|cpu_fp32|ort_t\d)\.txt", f).groups()
        txt = open(os.path.join(d, f)).read()
        if cfg.startswith("ort"):
            r = {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", txt)}
            r["per_inference_ms"] = r.get("mean_ms")
        else:
            r = parse_qnn(txt)
            r["per_inference_ms"] = r["netrun_ms"]
        out.setdefault(name, {})[cfg] = r
    return out


def power():
    p = os.path.join(R, "power", "power_raw.txt")
    if not os.path.exists(p):
        return {}
    rows = {}
    for line in open(p):
        m = re.match(r"rep=(\d+) model=(\S+) cfg=(\S+) rate=([\d.]*) \|", line)
        if not m:
            continue
        v = [float(x) for x in re.findall(r"mean_uV=([\d.]+)", line)]
        sem = [float(x) for x in re.findall(r"sem_uV=([\d.]+)", line)]
        temps = [int(x) for x in re.findall(r"msm_mC_end=(-?\d+)", line)]
        drop_mv = ((v[0] + v[2]) / 2 - v[1]) / 1000
        key = (m.group(2), m.group(3))
        rows.setdefault(key, []).append({"rep": int(m.group(1)), "rate": float(m.group(4) or 0), "drop_mV": drop_mv,
                                         "idle_drift_mV": (v[0] - v[2]) / 1000, "sem_mV": max(sem) / 1000,
                                         "temp_C_end_load": temps[1] / 1000 if len(temps) > 1 else None})
    out = {}
    busy1 = np.mean([r["drop_mV"] for r in rows.get(("ref", "busy1"), [])]) if ("ref", "busy1") in rows else None
    for (mname, cfg), rs in rows.items():
        drop = float(np.mean([r["drop_mV"] for r in rs]))
        rate = float(np.mean([r["rate"] for r in rs])) if rs[0]["rate"] else None
        e = {"drop_mV_mean": round(drop, 1), "drop_mV_reps": [round(r["drop_mV"], 1) for r in rs],
             "idle_drift_mV": [round(r["idle_drift_mV"], 1) for r in rs], "rate_per_s": rate,
             "temp_C_end_load": [r["temp_C_end_load"] for r in rs]}
        if busy1:
            e["busy_core_equiv"] = round(drop / busy1, 2)
        if rate:
            e["rel_energy_per_inf"] = drop / rate        # mV*s per inference, proportional to energy
        out.setdefault(mname, {})[cfg] = e
    return out


def fmt(x, p=2):
    return "-" if x is None else f"{x:.{p}f}"


def report(acc, lat, pw):
    L = ["# QIDK benchmark: baseline vs guided LSTM (and CNN-LSTM, LightGBM)", ""]
    L += ["## Accuracy on the test split (R², per domain)", "",
          "Same inputs for every row. `SQNR` = agreement with the FP32 ONNX model on the PC (higher is closer; >40 dB is"
          " effectively identical).", "",
          "| Model | Runs on | SQNR dB | Solar R² | Grid R² | Demand R² | Solar nMAE % | Grid nMAE % |",
          "|---|---|---|---|---|---|---|---|"]
    for name, res in acc.items():
        for v, r in res.items():
            if not isinstance(r, dict) or "metrics" not in r:
                continue
            m = r["metrics"]
            L.append(f"| {name} | {LABEL.get(v, 'PC ONNX FP32 (reference)')} | {fmt(r.get('sqnr_db_vs_fp32'), 1)} | "
                     f"{m['Solar']['R2']:.3f} | {m['Grid']['R2']:.3f} | {m['Demand']['R2']:.3f} | "
                     f"{m['Solar']['nMAE_pct']:.2f} | {m['Grid']['nMAE_pct']:.2f} |")
    L += ["", "## Latency per inference (one window, batch 1)", "",
          "| Model | Config | Time per inference ms | NPU compute ms | p95 ms |", "|---|---|---|---|---|"]
    for name, cf in lat.items():
        for cfg, r in cf.items():
            L.append(f"| {name} | {LABEL.get(cfg, cfg)} | {fmt(r.get('per_inference_ms'), 3)} | "
                     f"{fmt(r.get('accel_exec_ms'), 3)} | {fmt(r.get('p95_ms'), 3)} |")
    if pw:
        ref = pw.get("ref", {})
        L += ["", "## Power (relative)", "",
              "Extra load = how far the main supply rail sags while the model runs back-to-back (mV, idle subtracted)."
              " `busy-core equiv.` expresses it as a number of busy-looping CPU cores. Energy per inference is"
              " relative to the NPU run of the same model (= 1.00).", "",
              "| Model | Config | Inferences/s | Extra load mV | Busy-core equiv. | Rel. energy / inference |",
              "|---|---|---|---|---|---|"]
        for name, cf in pw.items():
            if name == "ref":
                continue
            base = cf.get("htp_w8a16", {}).get("rel_energy_per_inf")
            for cfg, r in cf.items():
                rel = r.get("rel_energy_per_inf")
                L.append(f"| {name} | {LABEL.get(cfg, cfg)} | {fmt(r.get('rate_per_s'), 0)} | {r['drop_mV_mean']:.1f} "
                         f"({', '.join(fmt(x, 1) for x in r['drop_mV_reps'])}) | {fmt(r.get('busy_core_equiv'))} | "
                         f"{fmt(rel / base if (rel and base) else None)} |")
        for cfg, r in ref.items():
            L.append(f"| reference | {cfg} | - | {r['drop_mV_mean']:.1f} | {fmt(r.get('busy_core_equiv'))} | - |")
    return "\n".join(L) + "\n"


def main():
    acc, lat, pw = accuracy(), latency(), power()
    json.dump({"accuracy": acc, "latency": lat, "power": pw, "data": info}, open(os.path.join(R, "summary.json"), "w"),
              indent=1, default=float)
    md = report(acc, lat, pw)
    open(os.path.join(R, "summary.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
