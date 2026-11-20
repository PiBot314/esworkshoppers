# Running the demo on the QIDK

> **Status (20 Nov 2026):** the offline benchmark demo (app + :8080, 11 Nov). Since 16 Nov the board starts
> `nz_dashboard` and the live dashboard by itself at boot ([SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) §3), and the live
> dashboard (http://10.42.0.237:5173) is the main demo. `demo_setup.sh` calls `14_dashboard_start.sh`, which kills
> the supervisor's `nz_dashboard` and starts its own (not tested alongside the live system): on the live board, use
> only `demo_setup.sh --check` and install the app by hand if needed.

Everything runs on the board. The PC is only needed once after each reboot of the board, to start the server.

## Before the eval (2 minutes)
1. Plug the QIDK into the laptop with USB. Check it shows up: `adb devices` → `d42f06b  device`.
2. From the `final/` folder run:
   ```bash
   benchmark_qidk/scripts/demo_setup.sh
   ```
   This copies anything missing to the board, installs the app, starts the `nz_dashboard` server, keeps the screen
   on and opens the app. It ends with `Ready: …`.
3. Unplug the cable if you like. The server keeps running until the board reboots (then repeat step 2).

`benchmark_qidk/scripts/demo_setup.sh --check` only checks that all the files are on the board.

## On the touchscreen
App **Net-Zero NPU** → **Benchmark** (last button in the bottom bar). The header should say **● NPU live**.

| Button | What it does on the board | Time |
|---|---|---|
| **▶ Forecast** | Runs one real test window on the NPU (precision you pick), on the CPU with QNN, and on the CPU with ONNX Runtime. Shows time per forecast and plots forecast vs actual for solar and grid. | ~3 s (1 h) / ~5 s (24 h) |
| **⏱ Speed test** | Times every model (and LightGBM) on NPU, QNN CPU and ONNX Runtime CPU. | ~40 s |
| **✓ Accuracy check** | Runs the first 200 / 500 / 1000 test windows on the NPU; shows R² per domain next to the FP32 model, and the NPU-vs-FP32 agreement in dB. | ~30 s – 2 min |

Pick the model (baseline / guided, 1 h / 24 h, CNN-LSTM) and the NPU precision (W8A16 / FP16 / W8A8) at the top.
"Random" picks a random test window. The "Board printout" card shows the raw text the board produced.

The older tabs (Campus, Plant, Day-ahead, Run on NPU, Pipeline) are the earlier AIMET models and still work as before.

## Same thing from a terminal (fallback)
```bash
adb shell sh /data/local/tmp/nz_bench/demo.sh                          # help
adb shell sh /data/local/tmp/nz_bench/demo.sh forecast guided_24h 1000 w8a16
adb shell sh /data/local/tmp/nz_bench/demo.sh speed w8a8
adb shell sh /data/local/tmp/nz_bench/demo.sh accuracy guided_h4 w8a16 500
```

## If something goes wrong
- **"NPU server offline" in the app header**: the server isn't running (board rebooted). Run `demo_setup.sh` again.
- **Lock screen**: swipe up (no PIN). `demo_setup.sh` keeps the screen on while powered.
- **adb not finding the board**: replug the cable; `adb kill-server && adb devices`.
- The board clock is set to 2023; this only affects file dates.

## What is where
- `benchmark_qidk/board/demo.sh`: the on-board engine (also used by the app).
- `aimet_quantization/dashboard/server/nz_dashboard.cpp`: `GET /api/demo` runs `demo.sh`.
- `aimet_quantization/dashboard/android/web/bench.js`: the Benchmark tab.
- On the board: `/data/local/tmp/nz_bench` (models, test data, tools), `/data/local/tmp/nz_aimet` (QNN runtime, server).
