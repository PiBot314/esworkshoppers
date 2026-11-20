# Edge AI for Net-Zero: final submission

Forecasting campus energy (Solar, Grid, KH04 room load) from the IIIT-H meter data and running the forecasters on the
Qualcomm QIDK's NPU, with forecast-based fault detection and a dashboard/app served from the board. The board now runs
**on its own**:
* every 15 minutes it forecasts the live campus API on its NPU and flags unusual readings
* it explains what it found through an on-board assistant
* its interface leads with live NPU forecasting
* a switcher chooses among 7 forecasting models

It measures and recommends; it doesn't control anything.

**Start here:**
* [STATUS.md](STATUS.md): **what is true now**: what runs, the models and their scores, decisions, open items
* [PROJECT_HISTORY.md](PROJECT_HISTORY.md): the pieces of work, which slide comes from which, which numbers belong together
* [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md): the live system as built
* [WRITEUP_2026-11-20.md](reports/WRITEUP_2026-11-20.md): the latest results (weather, calendar, LightGBM vs LSTM, NPU fusion) written up
* [plan_integrated.md](archive/plan_integrated.md): the roadmap and its reasoning (finished plans are in `archive/`)

**Open the live system:** on the hotspot `QIDK_DevNet`, http://10.42.0.237:5173 (Live · Map · Net zero · Results · NPU lab). The **Results** tab compares every model with the persistence and same-time-yesterday baselines; read it before quoting any accuracy number.

## What's in this folder

| Path | What |
|---|---|
| `Edge AI for Net-Zero (revised).pptx` | The presentation (27 slides) |
| `STATUS.md` | **Current state**: what runs, deployed vs candidate models, decisions, open items with dependencies |
| `PROJECT_HISTORY.md` | One-page map of the whole project |
| `reports/` | Write-ups: `WRITEUP_2026-11-20.md` (20 Nov results: weather, calendar, LightGBM vs LSTM, NPU fusion, live anomaly limits) |
| `archive/worklogs/` | Day-by-day worklogs (16–17, 19 and 20 Nov): what was done, in order, and what was open at the time |
| `MODELS_SHA256.txt`, `make_model_manifest.sh` | Checksums of the model binaries (which git doesn't track); rerun the script after training or building |
| `.gitignore` | What git tracks: code, docs, configs, small results. Not data, builds, vendor runtimes, model binaries |
| `SYSTEM_DESIGN.md` | **The live system as built**: topology, boot and supervision, live loop, models and the switcher, dashboard routes and APIs, assistant, tests, operations |
| `forecast_2025/` | **The 20 Nov forecasters on real 2025 data:** LightGBM (B2), guided LSTM (B3/B4), fusion + emulator (C1), fair comparison (E1) and rolling-origin (E2). See its README for the results |
| `verify_api_lstm/` | Scripts behind the live-loop verification (API fetch, window building, flag and fault-injection scoring), plus `replay_server.py`, a fake API for replaying cached days quickly |
| `nz_live/` | The live board setup: `board/` (init service `nz_live.rc` and supervisor `start.sh`), `models/` (switchable models: `registry.txt`/`registry.json`, per-model `meta.json` with limits; grid limits refitted on API data on 20 Nov, the 2025 ones kept as `anomaly_2025`), `scripts/` (llama.cpp build, live-run and assistant test drivers, `build_model_meta.py`, `check_models_on_board.py`), `runs/` (16 Nov test logs and write-up), `STATUS_boot_network.md` (boot start and hotspot setup) |
| `benchmark_qidk/` | Board benchmark of baseline vs guided LSTMs (1 h and 24 h), CNN-LSTM and LightGBM: NPU vs CPU speed, accuracy and power. `RESULTS.md`, demo steps in `DEMO.md`; `live_eval_2026-11-19/` has the scripts and raw output of the 19 Nov check on the live system |
| `common/`, `cnn_lstm_hybrid/` | The models and training code that benchmark uses |
| `mideval_md/explanation.md` | Full technical explanation of the quantization and deployment work |
| `datasets/esw/` | Original meter data dump (`EM.csv`, `SL.csv`, `SR-EM.csv`, `metadata.xlsx` mapping nodes to locations) |
| `datasets/esw/nodes/` | `nodesort.py` (splits the dump per node and firmware version) and its output `parquet_sorted/`, which the pipeline reads |
| `datasets/weather/` | Weather and irradiance for the campus, 2025 → Nov 2026: ERA5 reanalysis, historical forecasts (incl. 15-min irradiance), 1-day-ahead forecasts, NASA POWER clear-sky. See its README |
| `datasets/training_2025/` | **The next models' dataset:** real 2025 meters + weather + climatology, 15 min, with plant fault status and season-balanced splits. See its README |
| `datasets/api_2026/` | History of everything the mentors' API serves (from 2026-01-01), fetched gently by a resumable user service (`nz-api-fetch`). See its README |
| `datasets/raw_data/` | `EM-Kh95-00.csv`, a separate export for the KH95 building meter |
| `benchmark/` | Data cleaning + per-node benchmark write-up (6 methods, WAPE) |
| `model2/` | The team's three deployed LSTMs: training scripts, ONNX exports, original reports |
| `aimet_quantization/` | AIMET W8A16 quantization, QIDK NPU deployment, dashboard and Android app, with all data and intermediate outputs. Start at its `README.md`; its step-by-step docs `00`–`18` are in `archive/aimet_quantization/` |
| `aimet_quantization/release/` | Final quantized models (`.so` for the HTP, ONNX + encodings) and the QAIRT 2.47 runtime for the board. Checksums in `SHA256SUMS` |
| `aimet_quantization/dashboard/` | C++ dashboard server, web UI, Android app source; built binaries in `bin/` (`nz_dashboard`, `nz_demo.apk`). The app (v1.1) opens the live :5173 dashboard when the board serves it, else its bundled offline demo |
| `qidk_export/` | Campus dashboard (SvelteKit, run by Bun on the board) + Qwen3-4B assistant (llama.cpp, board CPU). Includes the live loop (`dashboard/src/lib/server/live/`): every 15 min it fetches the campus API, forecasts on the NPU and flags readings far from the forecast. Original wiring plan: `archive/qidk_export/INTEGRATION_PLAN.md` |
| `qidk/` | Qualcomm's QIDK repo. `qidk/Tools/qairt_docker` builds the QAIRT SDK Docker image used to convert models for the NPU |
| `env/` | Pinned Python packages for the two environments, and the QAIRT image fix |
| `mideval_md/` | `explanation.md` (above) and `viva_questions_deck.md` (viva prep on the slides); the rest of the mid-eval prep is in `archive/mideval_md/` |
| `anomaly_test_window_294.csv` | Sample window used for the fault-injection demo |
| `archive/` | **Superseded or historical docs, moved here on 19 Nov at their original paths** (20 Nov: plus the finished plans `plan_integrated.md`, `plan_19112026.md`, `plan_20112026.md`; `worklogs/`; and `forecast_2025_emulator_work/`, the emulator/board-build intermediates, not in git): the earlier plans (`plan.md`, `plan_reduced_shortterm.md`, `plan_model_switcher.md`, the post-mideval direction doc, `qidk_export/INTEGRATION_PLAN.md`), the October quantization logs (`aimet_quantization/00`–`18`, `README_DETAILED.md`, `TECHNICAL_QA.md`) and the mid-eval prep (`mideval_md/`: talk and recording scripts, deck notes, viva banks, reading list, oneM2M notes). Code comments that cite these plans by name refer to the copies here |
| `extras/autoencoder/` | Earlier autoencoder anomaly detector (team, out of scope for the main line). Code and small artifacts only |

## Setting up from scratch

1. **Python environments** (from this folder; scripts expect them here):
   ```bash
   uv venv --python 3.10 .venv-aimet
   uv pip install --python .venv-aimet --index-strategy unsafe-best-match \
     --extra-index-url https://download.pytorch.org/whl/cpu -r env/requirements-aimet.txt
   uv venv .venv && uv pip install --python .venv -r env/requirements-main.txt
   ```
2. **QAIRT Docker image** (only needed to rebuild the NPU model libraries): follow `qidk/Tools/qairt_docker/README.md`
   to build `qairt:latest`, then
   `docker build -f env/Dockerfile.qairt-fixed -t qairt-fixed:2.47 .` (see `archive/aimet_quantization/04_environment_remediation.md`).
3. **Pipeline:** `aimet_quantization/scripts/01_…` to `14_…` in order. Script 01 reads `datasets/`. To regenerate
   `parquet_sorted/` from the dump, run `python nodesort.py` from inside `datasets/esw/nodes/`.
4. **Board:** an Android device with ADB access (the QIDK). `scripts/14_dashboard_start.sh` pushes the runtime and starts the dashboard.

## Running the live board (QIDK on the campus API)

The board starts everything by itself at boot: init runs `nz_live/board/start.sh` (from `nz_live.rc`), which keeps
three services alive: `nz_dashboard` (:8080, NPU forecasts), `llama-server` (:8090, the assistant) and the Bun
dashboard (:5173, with the live loop). The laptop is only the Wi-Fi uplink: it runs the hotspot `QIDK_DevNet`
(10.42.0.1/24, NetworkManager `ipv4.method shared`), and the board joins it as 10.42.0.237. Open
`http://10.42.0.237:5173` from the laptop. Full setup and undo steps are in `nz_live/STATUS_boot_network.md`.

**USB fallback (19 Nov):** if the board can't reach the API over Wi-Fi, the live loop switches to the USB cable by
itself: `adb reverse` plus a relay on the laptop (`nz_live/laptop/usb_uplink.sh`, run by the systemd user service
`nz-usb-uplink.service`), and back to Wi-Fi when that works again. The Live page shows which route is in use. Details:
`SYSTEM_DESIGN.md` §9a. It needs the cable plugged in and the laptop awake.

### Temporary fix: letting the board reach the outside world

The hotspot alone isn't enough. Docker (version 29 here) sets the kernel's `FORWARD` policy to `DROP`, so the laptop
drops every packet the board sends on to the campus network, and the live loop shows "no data". Until the
permanent fix below is applied, run this **after every laptop reboot or firewalld reload**:

```bash
sudo iptables -I DOCKER-USER -i wlp0s20f3 -j ACCEPT
sudo iptables -I DOCKER-USER -o wlp0s20f3 -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
```

`wlp0s20f3` is this laptop's Wi-Fi card (the one running the hotspot); on another machine check `nmcli d`. Test it
from the board with `adb shell curl -sI <API URL>` (expect HTTP 200).

**Permanent fix (not applied yet):** tell Docker not to drop forwarded traffic, then restart it:

```bash
echo '{"ip-forward-no-drop": true}' | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker
```

(If `/etc/docker/daemon.json` already exists, add the key to it instead of overwriting it.)

Two more things the board needs from the laptop:
* **The laptop must stay awake** while it is the router: closing the lid must not suspend it on power.
* On the board, `settings put global captive_portal_mode 0` has already been set. Without it, Android marks the
  intranet-only hotspot "no internet" and won't reconnect to it after a reboot.

## Not included

* `datasets/esw/nodes/csv_sorted/`: CSV copy of `parquet_sorted/` (same data); `nodesort.py` recreates it
* `dashboard/android/build/`: build intermediates; `dashboard/android/build.sh` recreates them, the finished app is `dashboard/bin/nz_demo.apk`
* The Python virtual environments themselves (they don't survive being moved; recreate from `env/`)
* The built QAIRT Docker images (~18 GB each); recreate with step 2
* `../project/` SNPE runtime tarball and chunks (the autoencoder's runtime)
* `../20260806_lab01*` (course lab exercise) and `../projectPapers/` (emission-factor references)
