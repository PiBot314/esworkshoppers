# Status (start here)

*Last updated 20 Nov 2026.* What is true **now**: what runs, which models exist and how good they are, what's
decided, and what's open. History and reasoning live elsewhere (links at the end). Update this file when one of
those changes.

## What runs now

| Part | State |
|---|---|
| Board (QIDK, `d42f06b`) | Connected over USB (through the hub) and on the laptop hotspot `QIDK_DevNet` (10.42.0.237). Starts its services on boot (`nz_live.rc`). Clock set by NTP over the hotspot (it falls back to 2023 after a reboot until NTP gets through, so HTTPS waits for that) |
| Board → campus API | Two routes, both working: hotspot (`10.2.203.214:9999`, needs the laptop's `DOCKER-USER` iptables rules, which a reboot clears) and USB fallback (`adb reverse` + `nz-usb-uplink.service`) |
| Board → internet | Works over the hotspot (Open-Meteo HTTPS 200 from the board, 20 Nov). No proxy over USB (not needed for now) |
| Live loop | Every 15 min: API → NPU forecast → anomaly flags. Grid: `all_sensors_1h` (AIMET W8A16 on the HTP, ~5 ms); PL00: `single_node`. Dashboard :5173, assistant :8090, NPU server :8080 |
| Anomaly limits | Refitted on API data for all 6 registered models **in the local `meta.json` files only**: not yet pushed to the board, so the board still over-flags (2025 limits) |

## Models

**Deployed / registered (old, 58 raw sensor inputs).** Scored on the 2025 test weeks for the first time on 20 Nov:
the live 1 h grid model is no better than persistence (R² 0.862 vs 0.866).

| Model | Where | Live? |
|---|---|---|
| `all_sensors_1h` (team LSTM, 1 h) | `model2/qidk_deployment/`; NPU build in `aimet_quantization/release/` | ✅ live grid model |
| `single_node` (PL00 only) | `model2/qidk_deployment/` (SL-PL00-00); NPU build in `aimet_quantization/` | ✅ live PL00 model |
| `baseline_h4`, `guided_h4`, `baseline_24h`, `guided_24h` | `common/` | registered, switchable |
| `cnn_lstm_24h` | `cnn_lstm_hybrid/` | registered, switchable |

**Candidates (20 Nov, real 2025 + weather + calendar, `forecast_2025/`).** R² at the horizon, identical rows,
blocked split / rolling-origin:

| Target | Best baseline | LightGBM | LSTM | Status |
|---|---|---|---|---|
| Solar 1 h | 0.772 / 0.744 | 0.846 / **0.802** | 0.841 / 0.783 | LSTM: fused solar NPU graph 52.9 dB (emulator), board package built |
| Solar 24 h | 0.679 / 0.598 | 0.818 / 0.744 | 0.813 / 0.751 | same fused graph, 46.5 dB |
| Grid 1 h | 0.866 / 0.845 | **0.931 / 0.888** | 0.914 / 0.784 | LightGBM exported (`models/grid_1h_lgbm.txt`); LSTM v2 51.3 dB, board package built |
| Grid 24 h | 0.654 / 0.626 | **0.873** / 0.713 | 0.774 / 0.705 | LightGBM exported (`models/grid_24h_lgbm.txt`) |

None can run live yet: they need weather, clear-sky and calendar inputs the live loop doesn't have
([forecast_2025/LIVE_INPUTS.md](forecast_2025/LIVE_INPUTS.md)). Model binaries are not in git; their checksums are in
[MODELS_SHA256.txt](MODELS_SHA256.txt) (`./make_model_manifest.sh` to refresh).

## Decided

* Real 2025 data is the only training and test target. The mentors' API data (synthetic, 2026) only calibrates the
  live system. Private oneM2M material stays out of the project (kept in `../ONEM2M_PRIVATE.md`, outside this folder and git).
* Compare methods only on identical rows, with 95 % intervals; differences below ~0.02 R² are not findings.
* Fuse NPU models only within a family (solar 1 h + 24 h). Calibrate per family: solar 100 validation windows, grid
  500 train + validation windows.
* Pre-training on the simulated 2015–2024 plants: dropped (no gain).
* Git tracks code, docs and small results; data, builds, vendor runtimes and model binaries are ignored (checksums
  instead). No Git LFS until there is a remote.

## Waiting on a decision

| Item | Who |
|---|---|
| Deployment design: grid on the board CPU with LightGBM, solar on the NPU (recommended), pending item 8 below | user |
| 2026 academic calendar (almanac PDFs), needed for the grid models' calendar inputs | user |
| Report framing: LightGBM as the accuracy reference, the NPU model matched to it under a hardware constraint | user |

## Open items

Numbers as used in the 20 Nov discussion. ★ = recommended next.

| # | Item | Needs |
|---|---|---|
| **Board** | | |
| 1 ★ | Run the board packages (`forecast_2025/fused/{solar_1h24h,grid_1h_v2_cal500}_board/run_on_board.sh`), score with `b5_score_board.py`, gate ≥ 44 dB | – |
| 2 ★ | Push the refitted anomaly limits to the board | – |
| 3 | Per-row weights confirmed on the device (A5) | 1 |
| 4a / 4b | Fused vs separate call timing (C1); batching plants into one call (C2) | 1; 4b also a batch > 1 build |
| 5 | Resident runner (C3) | 4a/4b |
| 6 | E3 live evaluation rerun | 2 (current models); 7 (new) |
| 7 | Shadow mode for the new models, then promote (B5) | 1, 16, 17 |
| **Models** | | |
| 8 ★ | LightGBM trees on the NPU (Hummingbird → ONNX → QNN): op support and branch flips | – |
| 9 | Dense NPU network distilled from LightGBM | 8 failing |
| 10 | Small network correcting a cheap baseline | optional, with 9 |
| 11 ★ | Score the +15 min step (the step the live anomaly rule uses) for every model | – |
| 12 | LightGBM over every step of the horizon | – |
| 13 | More seeds; 6 h LSTM; holiday / calendar-transition features | – |
| **Live loop** | | |
| 14 | Weather fetch (Open-Meteo, both endpoints verified from the board) | – |
| 15 | 2026 clear-sky table on the board | – |
| 16 | New input layout (`past` / `future` / `node`) in `features.ts` | 23, 14, 15, 18, 22 |
| 17 | Anomaly limits for the new models (API residuals) | 16, 11 |
| 18 | Handling of the 3 solar plants that are neighbour proxies in the feed | decision |
| **NPU tooling** | | |
| 19 | Per-family calibration as the default in `c1_emulator.py` / `b5_board_package.sh` | – |
| 20 | Fix or disable the converter's simplifier (`onnxsim` fails to import in the container on every graph) | – |
| 21 | AdaRound for margin on solar 24 h | 1, if the device margin is thin |
| **Decisions** | | |
| 22 | 2026 academic calendar | user |
| 23 | Deployment design | 1, 8 (or 9), 11, 12 |
| 24 | Report framing | 23, 13 |

**Critical path to deploying the new models:** 8 → 23 → 16 → 17 → 7, with 14, 15, 18 and 22 ready before 16.
Separate short track: 2 → 6 fixes the current models' over-flagging.

## Where things are

| What | Where |
|---|---|
| Overview of the folders | [README.md](README.md) |
| How the live system works | [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) |
| History of the project | [PROJECT_HISTORY.md](PROJECT_HISTORY.md) |
| 20 Nov results, written up | [WRITEUP_2026-11-20.md](reports/WRITEUP_2026-11-20.md); detail in [forecast_2025/README.md](forecast_2025/README.md) |
| Day-by-day logs | `archive/worklogs/` |
| Plans (finished; for the reasoning) | [plan_integrated.md](archive/plan_integrated.md), `archive/plan_19112026.md`, `archive/plan_20112026.md` |
| Emulator / board build intermediates | `archive/forecast_2025_emulator_work/` (not in git) |
| Private oneM2M notes | `../ONEM2M_PRIVATE.md` (outside this folder; never in git) |
