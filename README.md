# Edge AI for Net-Zero

Forecasting IIIT-H campus energy (solar plants, grid draw) from the campus meters and running the forecasters on the
Qualcomm QIDK's NPU. The board runs on its own: every 15 minutes it fetches the campus API, forecasts on the NPU,
flags readings far from the forecast, and explains them through an on-board assistant. It measures and recommends; it
doesn't control anything.

## Start here

| Read | For |
|---|---|
| [STATUS.md](STATUS.md) | **What is true now**: what runs, deployed vs candidate models and their scores, decisions, open items |
| [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) | How the live system works, and how to set it up and operate it (§9) |
| [PROJECT_HISTORY.md](PROJECT_HISTORY.md) | How we got here, which slide comes from which work, which numbers belong together |
| [reports/WRITEUP_2026-11-20.md](reports/WRITEUP_2026-11-20.md) | The latest results: weather, calendar, LightGBM vs LSTM, NPU fusion, live anomaly limits |

**Live dashboard:** from the laptop on the hotspot `QIDK_DevNet`, http://10.42.0.237:5173. Its Results tab compares
every model with the persistence and same-time-yesterday baselines: check it before quoting an accuracy number.

## Folders

| Folder | What |
|---|---|
| `forecast_2025/` | The current forecasters (20 Nov): LightGBM and guided LSTM on real 2025 data with weather and calendar, NPU fusion and emulator checks, board packages. Results in its README |
| `datasets/` | `esw/` original 2025 meter dump (+ `nodes/` per-node parquet), `raw_data/` KH95 export, `Water/` water meters, `weather/` weather and clear-sky, `training_2025/` the training dataset, `api_2026/` mentors' API history (synthetic), `esw-netzero/` the synthetic 2026 generator |
| `nz_live/` | The live board setup: boot service and supervisor (`board/`), switchable models and their limits (`models/`), laptop USB relay (`laptop/`), scripts and test runs |
| `qidk_export/` | The dashboard (SvelteKit on Bun, served from the board) with the live loop (`dashboard/src/lib/server/live/`), and the Qwen3-4B assistant |
| `aimet_quantization/` | The October AIMET W8A16 quantization and QIDK deployment, the :8080 C++ dashboard and the Android app; final NPU builds in `release/` |
| `model2/` | The team's deployed LSTMs (training scripts, ONNX exports, reports) |
| `common/`, `cnn_lstm_hybrid/` | The baseline / guided LSTMs, LightGBM and CNN-LSTM used by the benchmark and the switcher |
| `benchmark_qidk/` | Board benchmark: NPU vs CPU speed, accuracy and power; the 19 Nov live check (`live_eval_2026-11-19/`) |
| `benchmark/` | Data cleaning and the per-node benchmark write-up (6 methods, WAPE) |
| `verify_api_lstm/` | Live-loop verification scripts and `replay_server.py`, a fake API for replaying cached days |
| `mideval_md/` | `explanation.md` (the quantization and deployment work in full) and viva prep on the slides |
| `extras/autoencoder/` | Earlier autoencoder anomaly detector (out of scope) |
| `reports/` | Write-ups |
| `archive/` | Finished plans, day-by-day worklogs (`worklogs/`), the October quantization logs and mid-eval prep. Not current: for the reasoning behind decisions. Code comments that cite a plan by name (e.g. `plan_19112026.md` §3) mean the copy here |
| `env/` | Pinned Python packages and the QAIRT Docker image fix |
| `qidk/` | Qualcomm's QIDK repository (builds the QAIRT Docker image); not in git |

Top-level files: the presentation (`Edge AI for Net-Zero (revised).pptx`, 27 slides), `anomaly_test_window_294.csv`
(the sample window for the fault-injection demo), `MODELS_SHA256.txt` with `make_model_manifest.sh`, and `LICENSE.txt`
(MIT, from the team's first commit).

## What git tracks

Code, docs, configs and small results. **Not** data, builds, vendor SDKs and runtimes, or model binaries (see
`.gitignore`). Model binaries are fingerprinted in `MODELS_SHA256.txt`: rerun `./make_model_manifest.sh` after
training or building, and `sha256sum -c MODELS_SHA256.txt` checks the files on disk. The emulator and board-build
intermediates are in `archive/forecast_2025_emulator_work/` (regenerable). Python environments aren't in the folder:
recreate them from `env/` (SYSTEM_DESIGN.md §9.1).
