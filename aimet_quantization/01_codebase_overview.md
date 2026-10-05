# 01 — Codebase Overview

*Survey date: 2026-10-05. Scope: `mideval/` (this folder's parent).*

## 1. Top-level layout

| Path | What it is | Relevant to AIMET work? |
|---|---|---|
| `model2/` | The project: data prep, LSTM training, ONNX export, benchmarks, reports, deployment bundle | **Yes, this is the subject** |
| `model2/qidk_deployment/` | The three exported ONNX models, `.pt` checkpoints, metadata JSONs, sample inputs, Python runners, `qidk_deploy.sh` | **Yes, these are the inputs to quantization** |
| `qidk/` | Clone of Qualcomm's public QIDK repo (examples, solutions, `Tools/qairt_docker`) | Yes, for the QAIRT Docker tooling and as reference |
| `.venv/` | Python **3.14** venv that holds only Sphinx and its deps (used to build `qidk/docs`) | No. It can't host AIMET (see doc 03) |

## 2. `model2/` scripts and how they chain together

```
(raw telemetry: by_version/*  → [MISSING step] → lstm_ready/*.parquet + features.json)
        │
        ▼
generate_lstm_dataset.py      abs(), kW→W, SR-EM register-swap fix, fillna(0), MinMax[0,1]
        │                     → lstm_normalized/{node}.parquet/.csv, scaler_params.json, features.json
        ▼
┌───────────────────────────┬──────────────────────────────┬────────────────────────────────┐
│ train_and_export_qidk.py  │ train_all_sensors_lstm.py    │ train_24h_multisensor_lstm.py  │
│ single node SL-PL00-00    │ 11 nodes, 58 feats, H=4      │ 11 nodes, 58 feats, H=96       │
│ → SL-PL00-00_lstm_qidk    │ → all_sensors_lstm_qidk      │ → all_sensors_24h_lstm_qidk    │
│   .onnx  [1,96,10]→[1,4]  │   .onnx [1,96,58]→[1,12]     │   .onnx [1,96,58]→[1,288]      │
└───────────────────────────┴──────────────────────────────┴────────────────────────────────┘
        │  each script: chronological 70/15/15 split, MSE+Adam, best-val checkpoint,
        │  metrics in Watts, torch.onnx.export(opset 18, static B=1), ORT parity check,
        │  writes *_meta.json, a sample .npy (= first test window) and a Python runner
        ▼
run_comprehensive_benchmark.py   Parquet vs CSV I/O, MAC estimates, LSTM vs 3 naive baselines
build_jupyter_notebook.py        builds live_demo_netzero_lstm.ipynb (demo notebook)
```

Notes:
* All training scripts **skip training if a `.pt` exists** and just reload it. Re-running them would not retrain, but it **would overwrite the ONNX files and `*_meta.json`**. Per your constraint, **don't re-run them.** Treat the existing ONNX files as frozen inputs.
* `generate_lstm_dataset.py` reads `lstm_ready/` and `by_version/`. **Neither exists on this machine**, and neither does its output `lstm_normalized/`. The reports point to a teammate's machine (`/home/claryfray29/esw/model/`). Doc 05 covers the rebuild.

## 3. `model2/qidk_deployment/` inventory

| File | Size | Notes |
|---|---|---|
| `all_sensors_lstm_qidk.onnx` | 586 KB | 1 h horizon, 139,749 params. **Primary quantization target** |
| `all_sensors_24h_lstm_qidk.onnx` | 1.14 MB | 24 h horizon, 282,153 params |
| `SL-PL00-00_lstm_qidk.onnx` | ~250 KB | single plant, 55,021 params |
| `best_*.pt` | | PyTorch state dicts (for reference only, since we won't retrain) |
| `all_sensors_qidk_meta.json` | | I/O names, 58 ordered feature names, target capacities, test metrics |
| `all_sensors_24h_qidk_meta.json` | | same for the 24 h model (no feature list) |
| `qidk_model_meta.json` | | single-node: feature order, scaler min/max of target |
| `sample_input_all_sensors_1x96x58.npy` | | `X_test[:1]` of the 1 h script, which makes it a **data-rebuild fidelity anchor** |
| `sample_input_all_sensors_24h_1x96x58.npy` | | test window for 24 h model |
| `sample_input_1x96.npy` | | actually shape **[1,96,10]** despite the name |
| `sample_ground_truth_24h_kw.npy` | | shape [3,24], hourly kW ground truth for the demo |
| `qidk_*_inference.py` | | ONNX Runtime runners (CPU / QNN EP) |
| `qidk_deploy.sh` | | ADB push + `adb shell python3 …` (**won't work on this QIDK**, see §5) |

## 4. Reports (`report.md`, `report2.md`, `progress.txt`)
These describe the data-cleaning rationale, sensor dictionary, the net-zero physics, evaluation talking points, and a suggestion to also try TCN/DLinear. They're the best source for reconstructing the data pipeline. `progress.txt` §5 has per-node row counts and target ranges, and §6 has the aligned timeline (32,866 intervals, 2025-01-23 04:15 → 2025-12-31 12:30).

## 5. Inconsistencies and defects found (keep these in mind)

| # | Finding | Impact |
|---|---|---|
| 1 | **The models are forecasters, not anomaly detectors** (see doc 02). | Anomaly detection has to be a post-processing layer on forecast residuals |
| 2 | `lstm_normalized/`, `lstm_ready/`, `by_version/` are absent | No calibration or eval data, so a rebuild is needed (doc 05) |
| 3 | `qidk_deploy.sh` runs `python3` on the device, but **the QIDK has no Python**, and the runners need `onnxruntime` + QNN EP, which isn't on the device either | The deployment path has to be rewritten around `qnn-net-run`/`snpe-net-run` (doc 07) |
| 4 | Report: "141,696 MACs, ≈0.092 ms". QAIRT converter: **≈12.8 M MACs** for the 1 h model | The report figure looks like one LSTM timestep. With 96 steps unrolled it's about 90× more. Latency claims must be re-measured on device |
| 5 | 24 h Grid MAE: `report.md` says 34.9 kW (3.94 %), `all_sensors_24h_qidk_meta.json` says **74.3 kW (8.38 %)** | Use the meta JSON (machine-written) as the FP32 baseline. Re-measure anyway |
| 6 | 1 h model **Demand R² = −0.34** (worse than predicting the mean) | Exclude Demand R² from the AutoQuant accuracy target, or use nMAE for it |
| 7 | `solar_max_W` differs between the 1 h meta (546010.259) and 24 h meta (546010.302) | The two models were trained on slightly different builds of `lstm_normalized/`. A rebuild can at best approximate both |
| 8 | Runner prints `net_w = g_w + s_w - d_w`, while the reports define Net = Demand − Solar | Cosmetic, but don't copy it into new runners |
