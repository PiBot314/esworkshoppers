# Campus Energy Forecaster + Anomaly Check on the Qualcomm QIDK (AIMET-quantized)

Three LSTM forecasters (campus solar / grid / demand) quantized with **Qualcomm AIMET** (post-training, no retraining) and running on the **QIDK's Hexagon NPU**. A residual check on the forecasts flags anomalous sensor readings.

| Model | Input → Output | On-NPU latency | Fidelity to FP32 | R² (FP32 → INT8/16) |
|---|---|---|---|---|
| `all_sensors_1h` | 24 h × 58 features → next 1 h (Solar, Grid, Demand) | 3.4 ms | 48.4 dB | 0.852 → 0.853 |
| `all_sensors_24h` | 24 h × 58 features → next 24 h | 3.4 ms | 46.4 dB | 0.339 → 0.335 |
| `single_node` | 24 h × 10 features → next 1 h (Palash plant) | 3.9 ms | 52.1 dB | 0.950 → 0.950 |

## Run it
Requirements: QIDK connected over USB (`adb devices` lists it), Python env `../.venv-aimet`, the runtime already on the device (`/data/local/tmp/nz_aimet`; to re-install: `adb push release/qidk_runtime_2.47/{bin,lib} /data/local/tmp/nz_aimet/`).

```bash
cd mideval/aimet_quantization
# forecast for a test window, on the NPU, with the anomaly verdict
../.venv-aimet/bin/python scripts/12_qidk_forecast.py --model all_sensors_1h --test-index 10
# same window with a simulated faulty solar sensor (reads 0 W) -> ANOMALY
../.venv-aimet/bin/python scripts/12_qidk_forecast.py --model all_sensors_1h --test-index 10 --inject sensor_dropout_zero
# your own window: [1, 96, F] float32, normalized like the training data
../.venv-aimet/bin/python scripts/12_qidk_forecast.py --model single_node --input my_window.npy
```
Output: forecast in Watts per 15-min step and, if the actual reading is known, `ok` / `ANOMALY` per domain.

## Live demo dashboard (served by the QIDK)
```bash
cd mideval/aimet_quantization
scripts/14_dashboard_start.sh        # QIDK on USB: starts the server on the QIDK, forwards port 8080, opens the browser
# -> http://localhost:8080   (or http://<QIDK-Wi-Fi-IP>:8080 on the same network; also works in the QIDK's own Chrome)
scripts/14_dashboard_start.sh --stop
```
Tabs: campus replay with typical band and anomaly flags · plant model · day-ahead 24 h · **Run on NPU** (upload/edit a CSV window of sensor readings and run it live on the Hexagon NPU) · under the hood. Keys: `1`–`5` tabs, `Space` play, `←/→` step, `D/S/F/P` inject dropout/spike/stuck/panel fault, `C` clear, `N` next anomaly, `R` run on NPU. Screenshots: `dashboard/screenshots/`.

## How the product was made (main steps)
1. **Data:** rebuilt calibration/test windows from the raw telemetry (`scripts/01–02`).
2. **Baseline:** FP32 outputs and anomaly thresholds (`03–04`).
3. **Quantize with AIMET** (`05–06`): HTP v75 config, range priors (LSTM hidden ∈ [−1, 1], output headroom), true 8-bit LSTM weights, **AutoQuant → AdaRound**, W8A16.
4. **Convert** (`07–08`): `qnn-onnx-converter` with the AIMET encodings, expanded LSTM, kept I/O layout → Android model library.
5. **Deploy and verify** (`09–11`): run all test windows on the QIDK, compare with FP32, measure latency.
6. **Demo dashboard** (`13–14`, `dashboard/`): native web server on the QIDK, live NPU runs on request.

Release files: `release/<model>/` (NPU library, AIMET ONNX + encodings, anomaly thresholds, metadata).
Details: `README_DETAILED.md` · Q&A: `TECHNICAL_QA.md` · Step-by-step logs: `00`–`17_*.md` · Ideas: `18_future_revisions_and_ideas.md`.
