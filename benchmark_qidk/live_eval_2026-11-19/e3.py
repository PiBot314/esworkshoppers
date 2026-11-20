"""E3: does forecasting on the NPU leave the CPU to the assistant? Assistant speed (prompt reading and answer writing)
alone, next to continuous NPU forecasting, next to continuous CPU forecasting (same model, guided_h4).
Saves after every measurement to e3_results.json."""
import json, subprocess, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ADB = ["adb", "-s", "d42f06b"]
LLM = "http://127.0.0.1:18090"
HERE = Path(__file__).parent
BENCH, RT = "/data/local/tmp/nz_bench", "/data/local/tmp/nz_aimet"
WIN = "/data/local/tmp/nz_live/models/guided_h4/live/window.raw"
LOOPS = {
    "npu": (f"cd {BENCH}/data && export LD_LIBRARY_PATH={RT}/lib ADSP_LIBRARY_PATH={RT}/lib && rm -rf e3tmp && echo {WIN} > e3list.txt && "
            f"{RT}/bin/qnn-net-run --backend {RT}/lib/libQnnHtp.so --model /data/local/tmp/nz_live/models/guided_h4/libw8a16.so "
            f"--input_list e3list.txt --num_inferences 100000 --keep_num_outputs 1 --perf_profile burst --output_dir e3tmp > /dev/null 2>&1"),
    "cpu": f"cd {BENCH}/data && LD_LIBRARY_PATH={BENCH}/ortlib {BENCH}/bin/ort_bench {BENCH}/onnx/guided_h4.onnx 1 loop {WIN} 300",
}
# ~600-token prompt: what the assistant reads when a tool returns data
PROMPT = ("You are the campus energy assistant. Tool output follows.\n" +
          "\n".join(f"{h:02d}:{m:02d} grid {300 + (h * 7 + m) % 200} kW forecast {310 + (h * 5 + m) % 190} kW limit 120 kW"
                    for h in range(0, 24) for m in (0, 30)) +
          "\nExplain in detail what happened today and whether anything looks unusual.")


def clear_of_live_check(need_s):
    while True:
        now = datetime.now(timezone.utc)
        m = (now.minute * 60 + now.second) % 900
        if not any(80 <= ((m + s) % 900) <= 180 for s in range(0, need_s + 1, 5)):
            return
        time.sleep(5)


def ask():
    body = json.dumps({"prompt": PROMPT, "n_predict": 200, "ignore_eos": True, "temperature": 0, "cache_prompt": False}).encode()
    req = urllib.request.Request(f"{LLM}/completion", data=body, headers={"content-type": "application/json"})
    t = json.loads(urllib.request.urlopen(req, timeout=900).read())["timings"]
    return {"prompt_tokens": t["prompt_n"], "prompt_tps": t["prompt_per_second"], "gen_tps": t["predicted_per_second"],
            "total_s": (t["prompt_ms"] + t["predicted_ms"]) / 1000}


def stop_loops():
    subprocess.run(ADB + ["shell", "pkill -f e3list.txt; pkill -x ort_bench; rm -rf " + BENCH + "/data/e3tmp"], capture_output=True)


res = {"alone": [], "npu": [], "cpu": []}
out = HERE / "e3_results.json"
for rnd in range(3):
    for cond in ("alone", "npu", "cpu"):
        clear_of_live_check(120)
        p = None
        if cond != "alone":
            p = subprocess.Popen(ADB + ["shell", LOOPS[cond]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(4)
        r = ask()
        if p:
            stop_loops()
            try: p.wait(timeout=15)
            except subprocess.TimeoutExpired: p.kill()
        res[cond].append(r)
        out.write_text(json.dumps(res, indent=1))
        print(rnd, cond, r, flush=True)
        time.sleep(10)  # let the SoC cool a little
print("done")
