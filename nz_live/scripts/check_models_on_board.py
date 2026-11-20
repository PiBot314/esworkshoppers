"""On-device check (plan_model_switcher.md M2): run each registry model's saved validation window on the board's NPU
(nz_dashboard /api/run, W8A16) and compare with the laptop's FP32 output. Writes sqnr_db and npu_ms into
nz_live/models/registry.json.      python3 nz_live/scripts/check_models_on_board.py [board_url]"""
import json, math, os, sys, urllib.request

FINAL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(FINAL, "nz_live", "models")
URL = sys.argv[1] if len(sys.argv) > 1 else "http://10.42.0.237:8080"
reg = json.load(open(os.path.join(D, "registry.json")))
for m in reg:
    w = os.path.join(D, m["id"], "check_window.raw")
    if not os.path.exists(w):
        continue
    req = urllib.request.Request(f"{URL}/api/run?model={m['id']}&variant=aimet_w8a16", data=open(w, "rb").read(),
                                 headers={"content-type": "application/octet-stream"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    f = json.load(open(os.path.join(D, m["id"], "check_fp32.json")))["out"]
    sig = sum(x * x for x in f)
    err = sum((a - b) ** 2 for a, b in zip(r["out"], f))
    m["sqnr_db"], m["npu_ms"] = round(10 * math.log10(sig / err), 1), round(r["execute_us"] / 1000, 2)
    print(f"{m['id']:13s} SQNR {m['sqnr_db']} dB, NPU {m['npu_ms']} ms")
json.dump(reg, open(os.path.join(D, "registry.json"), "w"), indent=1)
