"""NPU vs CPU for the live system as it runs now (19 Nov): 7 models per 15-min check, resident NPU runners,
Qwen3-4B assistant (llama-server, 6 threads) sharing the CPU.

E1  one check's forecasts, idle board:   NPU (warm 7 runners + 7 /api/run through nz_dashboard, as loop.ts does)
                                          vs CPU (ONNX Runtime FP32, 1 thread: load + 1 run per model, and steady state)
E2  the same while the assistant generates text
E3  assistant tokens/s: alone, next to a continuous NPU loop (qnn-net-run --num_inferences), next to a continuous
    CPU loop (ort_bench loop, 1 thread)

Talks to the board over adb (forwards: 18080 -> nz_dashboard :8080, 18090 -> llama-server :8090).
Avoids the live loop's checks (every 15 min, at :02/:17/:32/:47 + a few s, runners warmed 20 s before).
"""
import json, statistics as st, subprocess, sys, threading, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ADB = ["adb", "-s", "d42f06b"]
NPU = "http://127.0.0.1:18080"
LLM = "http://127.0.0.1:18090"
HERE = Path(__file__).parent
MODELS = ["all_sensors_1h", "single_node", "baseline_h4", "guided_h4", "baseline_24h", "guided_24h", "cnn_lstm_24h"]
BOARD_WIN = {m: (f"/data/local/tmp/nz_aimet/{m}/live/window.raw" if m in ("all_sensors_1h", "single_node")
                 else f"/data/local/tmp/nz_live/models/{m}/live/window.raw") for m in MODELS}
WIN = {m: (HERE / "win" / f"{m}.raw").read_bytes() for m in MODELS}
BENCH = "/data/local/tmp/nz_bench"
RT = "/data/local/tmp/nz_aimet"
out = {}


def adb(cmd, timeout=600):
    return subprocess.run(ADB + ["shell", cmd], capture_output=True, text=True, timeout=timeout).stdout


def post(url, body=b"", timeout=600, ctype="application/octet-stream"):
    req = urllib.request.Request(url, data=body, method="POST", headers={"content-type": ctype})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def clear_of_live_check(need_s):
    """Wait until the next need_s seconds don't overlap a live check (minute 2/17/32/47, -40 s .. +60 s)."""
    while True:
        now = datetime.now(timezone.utc)
        m = (now.minute * 60 + now.second) % 900  # seconds into the 15-min period
        chk = 2 * 60  # the check fires ~2 min after each quarter hour (bin end + grace)
        lo, hi = chk - 40, chk + 60
        bad = any(lo <= ((m + s) % 900) <= hi for s in range(0, need_s + 1, 5))
        if not bad:
            return
        time.sleep(5)


def npu_check():
    """One check, NPU path: warm all 7 runners, then one window each through nz_dashboard."""
    t0 = time.perf_counter()
    w = post(f"{NPU}/api/warm?models={','.join(MODELS)}&ttl=30")
    t_warm = time.perf_counter() - t0
    runs = []
    for m in MODELS:
        t1 = time.perf_counter()
        r = post(f"{NPU}/api/run?model={m}&variant=aimet_w8a16", WIN[m])
        assert r["ok"], r
        runs.append({"model": m, "exec_ms": r["execute_us"] / 1000, "server_ms": r["wall_ms"],
                     "client_ms": (time.perf_counter() - t1) * 1000, "resident": r.get("runner") == "resident"})
    return {"warm_s": t_warm, "warm": w["started"], "runs": runs, "total_s": time.perf_counter() - t0}


def stop_runners():
    # let them idle out (ttl 30 s) so every NPU check starts cold, as in the live loop
    while json.loads(urllib.request.urlopen(f"{NPU}/api/warm", timeout=10).read())["running"]:
        time.sleep(3)


def cpu_check():
    """One check, CPU path: ONNX Runtime, 1 thread, a fresh process per model (load + 1 run), as a one-shot would."""
    cmd = " ; ".join(f"LD_LIBRARY_PATH={BENCH}/ortlib {BENCH}/bin/ort_bench {BENCH}/onnx/{m}.onnx 1 latency {BOARD_WIN[m]} 0 1"
                     for m in MODELS)
    t0 = time.perf_counter()
    txt = adb(f"cd {BENCH}/data && {cmd}")
    lines = [l for l in txt.splitlines() if l.startswith("load_ms")]
    assert len(lines) == len(MODELS), txt
    runs = []
    for m, l in zip(MODELS, lines):
        kv = dict(x.split("=") for x in l.split())
        runs.append({"model": m, "load_ms": float(kv["load_ms"]), "run_ms": float(kv["mean_ms"])})
    return {"runs": runs, "total_s_incl_adb": time.perf_counter() - t0}


def cpu_steady():
    res = {}
    for m in MODELS:
        l = adb(f"cd {BENCH}/data && LD_LIBRARY_PATH={BENCH}/ortlib {BENCH}/bin/ort_bench {BENCH}/onnx/{m}.onnx 1 latency {BOARD_WIN[m]} 20 300")
        res[m] = {k: float(v) for k, v in (x.split("=") for x in l.split()) if k != "runs"}
    return res


class Generate(threading.Thread):
    """The assistant writing a fixed-length answer on the CPU (6 threads)."""
    def __init__(self, n=400):
        super().__init__(daemon=True)
        self.n, self.res = n, None

    def run(self):
        body = json.dumps({"prompt": "Write a detailed technical report on energy use at a university campus, "
                           "covering solar generation, grid draw, water pumping and air conditioning.",
                           "n_predict": self.n, "ignore_eos": True, "temperature": 0, "cache_prompt": False}).encode()
        self.res = post(f"{LLM}/completion", body, timeout=900, ctype="application/json")["timings"]


def gen_alone(n=300):
    g = Generate(n); g.start(); g.join()
    return g.res["predicted_per_second"]


def with_generation(fn, warmup=4):
    g = Generate(500); g.start(); time.sleep(warmup)  # past prompt processing, into generation
    r = fn()
    alive = g.is_alive()
    g.join()
    return r, alive, g.res["predicted_per_second"]


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print("E1 idle board", flush=True)
    out["cpu_steady_idle"] = cpu_steady()
    e1n, e1c = [], []
    for i in range(reps):
        clear_of_live_check(60); stop_runners(); clear_of_live_check(30)
        e1n.append(npu_check()); e1c.append(cpu_check())
        print(f"  {i}: NPU check {e1n[-1]['total_s']:.2f} s, CPU check {sum(r['load_ms'] + r['run_ms'] for r in e1c[-1]['runs']):.0f} ms", flush=True)
    out["E1"] = {"npu": e1n, "cpu": e1c}

    print("E2 while the assistant generates", flush=True)
    e2n, e2c = [], []
    for i in range(max(3, reps // 2)):
        clear_of_live_check(120); stop_runners(); clear_of_live_check(120)
        r, alive, _ = with_generation(npu_check); r["overlapped"] = alive; e2n.append(r)
        r, alive, _ = with_generation(cpu_check); r["overlapped"] = alive; e2c.append(r)
        print(f"  {i}: NPU {e2n[-1]['total_s']:.2f} s ({alive=}), CPU {sum(x['load_ms'] + x['run_ms'] for x in e2c[-1]['runs']):.0f} ms", flush=True)
    out["E2"] = {"npu": e2n, "cpu": e2c}
    clear_of_live_check(120)
    out["cpu_steady_busy"] = with_generation(cpu_steady)[0]

    print("E3 assistant speed", flush=True)
    e3 = {"alone": [], "npu_loop": [], "cpu_loop": []}
    npu_loop = (f"cd {BENCH}/data && export LD_LIBRARY_PATH={RT}/lib ADSP_LIBRARY_PATH={RT}/lib && rm -rf e3tmp && "
                f"echo {BOARD_WIN['guided_h4']} > e3list.txt && "
                f"{RT}/bin/qnn-net-run --backend {RT}/lib/libQnnHtp.so --model /data/local/tmp/nz_live/models/guided_h4/libw8a16.so "
                f"--input_list e3list.txt --num_inferences 20000 --keep_num_outputs 1 --perf_profile burst --output_dir e3tmp > /dev/null 2>&1; rm -rf e3tmp")
    cpu_loop = f"cd {BENCH}/data && LD_LIBRARY_PATH={BENCH}/ortlib {BENCH}/bin/ort_bench {BENCH}/onnx/guided_h4.onnx 1 loop {BOARD_WIN['guided_h4']} 120"
    for i in range(3):
        clear_of_live_check(150)
        e3["alone"].append(gen_alone())
        for key, cmd in (("npu_loop", npu_loop), ("cpu_loop", cpu_loop)):
            clear_of_live_check(150)
            p = subprocess.Popen(ADB + ["shell", cmd], stdout=subprocess.PIPE, text=True)
            time.sleep(3)
            e3[key].append(gen_alone())
            subprocess.run(ADB + ["shell", "pkill -f qnn-net-run.*e3list; pkill -f ort_bench"], capture_output=True)
            txt = p.communicate(timeout=60)[0]
            if key == "cpu_loop" and txt: e3.setdefault("cpu_loop_rate", []).append(txt.strip())
        print(f"  {i}: alone {e3['alone'][-1]:.2f} t/s, NPU loop {e3['npu_loop'][-1]:.2f}, CPU loop {e3['cpu_loop'][-1]:.2f}", flush=True)
    out["E3"] = e3
    (HERE / "results.json").write_text(json.dumps(out, indent=1))
    print("saved", HERE / "results.json")


if __name__ == "__main__":
    main()
