# NZ-Edge: system design (as built)

*Last updated 2026-11-20. Describes what runs on the QIDK today. For current status and what comes next, see [STATUS.md](STATUS.md). For how the dashboard's map, Net zero page and assistant work in detail, see [qidk_export/dashboard/HOW_IT_WORKS.md](qidk_export/dashboard/HOW_IT_WORKS.md).*

---

## 1. What it is

One Qualcomm QIDK board (Snapdragon 8 Gen 3, SM8650, Android 14) that, unattended:

1. **Fetches** campus readings from the mentors' API every 15 minutes (synthetic 2026 data, replayed in real time).
2. **Forecasts** campus grid draw and the Palash solar plant on the **Hexagon NPU** (LSTMs, AIMET W8A16, ~5 ms).
3. **Flags** readings far from the forecast (per-hour limits fitted on 2025 validation data; limits refitted on API
   data are prepared but not yet on the board: §9.4).
4. **Shows** it all in a dashboard served from the board (Live · Map · Net zero · Results · NPU lab), and **explains** it through an assistant (Qwen3-4B on the board's CPU).

It only measures and explains; it doesn't control any equipment ([plan_integrated.md](archive/plan_integrated.md) §2.1). Live flags on synthetic data are false alarms by construction, so demos use fault injection.

## 2. Topology

```mermaid
flowchart LR
    API["Mentors' API<br/>10.2.203.214:9999"]
    LAP["Laptop: Wi-Fi hotspot QIDK_DevNet<br/>NAT to IIIT-LAN (API + internet)"]
    USB["Laptop: USB fallback<br/>adb reverse :19999 → socat relay<br/>(nz-usb-uplink.service)"]
    subgraph QIDK["QIDK 10.42.0.237 · started by init (nz_live.rc), user shell"]
        SUP["start.sh supervisor"]
        NZ["nz_dashboard :8080 (C++)<br/>runs models on the NPU via qnn-net-run<br/>NPU lab page"]
        BUN["Bun + SvelteKit :5173<br/>live loop · UI · assistant tools"]
        LLM["llama-server :8090<br/>Qwen3-4B Q4_0, localhost only"]
        HTP["Hexagon NPU"]
    end
    VIEW["Browser / phone on the hotspot<br/>board's own screen (app v1.1: live dashboard,<br/>bundled offline demo as fallback)"]
    API --- LAP -- "route 1: Wi-Fi" --- BUN
    API --- USB -. "route 2: USB cable" .- BUN
    BUN -- "POST /api/run" --> NZ --> HTP
    BUN <--> LLM
    SUP -.-> NZ & BUN & LLM
    VIEW --> BUN
```

| Process | Port | Binds | Role |
|---|---|---|---|
| `nz_dashboard` | 8080 | 0.0.0.0 | Runs any registered model on the NPU: through a **resident runner** (`nz_runner`, ~3–5 ms per window) when warmed, otherwise one-shot `qnn-net-run` (~1.8 s, also FP32 on the CPU); serves the NPU lab page and the Android app's backend |
| Bun + SvelteKit (`build/index.js`) | 5173 | 0.0.0.0 | Live loop, dashboard UI, chat endpoint, relay `/npu-app/*` → :8080 |
| `llama-server` | 8090 | 127.0.0.1 | Assistant model, OpenAI-compatible API with tool calling |

## 3. Start-up and supervision

* `/system/etc/init/nz_live.rc` (source: [nz_live/board/nz_live.rc](nz_live/board/nz_live.rc)) starts `/data/local/tmp/nz_live/start.sh` on `sys.boot_completed=1` as `shell` (`u:r:shell:s0`). Init restarts the supervisor if it dies.
* [start.sh](nz_live/board/start.sh) starts the three processes in the foreground, restarts any that exit (back-off 2–60 s), and logs to `nz_live/logs/`.
* Power cycle → everything back, with the API and NPU OK, within ~5.5 min (test A5).
* Network, setup changes and how to undo them: §9.2–9.3.

## 4. The live loop (`qidk_export/dashboard/src/lib/server/live/`)

Runs inside the Bun server when `LIVE=1`. One cycle per 15-minute bin, 2 minutes after the bin closes (`GRACE`):

| Step | What | Code |
|---|---|---|
| Clock | "Now" = the API's `replay_now`, tracked against a monotonic clock (the board's wall clock starts at 2023 after a reboot until NTP sets it, §9.2) | `loop.ts` `syncClock` |
| Ingest | Last 50 h of raw readings for the 7 served nodes; same mapping and cleaning as training (signs, units, swapped fields) | `features.ts` |
| Features | 15-min UTC means → ffill(4) + `was_imputed` → min-max per feature from the model's `meta.json` → IST cyclic features. **3 unserved solar nodes are neighbour copies, KH04-00 = 0** | `features.ts` |
| Warm-up | 20 s before each check: `POST :8080/api/warm?models=<active + shadow>&ttl=45`; the runners exit 45 s after their last request | `loop.ts` `warm` |
| Inference | `POST :8080/api/run?model=<active model>&variant=aimet_w8a16`, FP32 on the CPU if the NPU fails | `loop.ts` `runNpu` |
| Shadow mode | Every other Grid model forecasts the same reading (own limits, never alerted); **persistence** and **same time yesterday** are scored for both signals → `shadow.jsonl` | `loop.ts` `shadowModels`, `baselines` |
| Detection | Flag ⇔ \|actual − forecast − median[hour]\| > limit[hour] (per model, from its `meta.json`). Grid and PL00 only | `detect.ts` |
| Outputs | Latest status; **events** (flags and injected faults) → `events.jsonl`; **history** (every checked reading: actual, forecast, normal range, model) → `history.jsonl` | `loop.ts` |
| Backfill | After start-up or a model switch: re-forecast the last 24 h from past windows (history only, never events) | `loop.ts` `backfill` |
| Catch-up | After an outage or restart (last checked bin saved in `last_bin.json`): every missed bin, up to 24 h, is checked as the cycle would have; flags become events marked **late** (no banner, shown as "found after an outage"). It warms the NPU runners first (since 19 Nov), so 24 h of catch-up takes ~1 min | `loop.ts` `catchUp` |
| Fault injection | `POST /api/live/inject {signal, kind, steps}` alters the next 1–4 actual readings (UI only) | `loop.ts` `requestInject` |

Files on the board (`/data/local/tmp/nz_live/live/`): `events.jsonl`, `history.jsonl`, `shadow.jsonl`, `active_models.json`.

**Resident runner** ([nz_live/runner/nz_runner](nz_live/runner/nz_runner)): Qualcomm's SDK SampleApp, with a `serveOnce` method that executes an input list without freeing the finalized graphs, and a stdin loop (`NZ_SERVE=1`): `<input_list> <output_dir>` → `OK <µs>`. Built with the SDK image's NDK; binary at `nz_live/bin/nz_runner` on the board. `nz_dashboard` forks one per model (`/api/warm`), feeds it requests, and a reaper thread stops idle ones. Start-up ~1.8 s per model, ~3–5 ms per window, output identical to `qnn-net-run` (47.5 dB vs FP32).

## 5. Models and the switcher

**Registry.** `/data/local/tmp/nz_live/models/`:

| File | Read by | Format |
|---|---|---|
| `registry.txt` | `nz_dashboard` (at start) | `<id> <n_features> <dir>` per line; `<dir>/libw8a16.so` (HTP), `<dir>/libfp32.so` (CPU) |
| `registry.json` | live loop, UI, assistant | id, label, family, horizon, signals, +15-min R² (val/test), SQNR, NPU ms |
| `<id>/meta.json` | live loop | features (order, min/max), lookback, capacities, per-hour anomaly limits, metrics |

The built-in models (`all_sensors_1h`, `all_sensors_24h`, `single_node`) keep their old paths under `/data/local/tmp/nz_aimet/`.

**Models available for the Grid signal** (scored the same way: R² of the +15-minute forecast, 2025 test split; NPU vs FP32 on one validation window):

| id | Family | Horizon | Solar R² | Grid R² | SQNR | NPU ms |
|---|---|---|---|---|---|---|
| `all_sensors_1h` (default) | LSTM (team, deployed) | 1 h | 0.966 | **0.824** | 46–48 dB* | ~5 |
| `guided_h4` | LSTM, LightGBM-gated | 1 h | 0.964 | 0.778 | 47.5 dB | 4.6 |
| `baseline_h4` | LSTM | 1 h | 0.928 | 0.666 | 47.7 dB | 4.8 |
| `guided_24h` | LSTM, LightGBM-gated | 24 h | 0.905 | 0.743 | 45.4 dB | 4.5 |
| `baseline_24h` | LSTM | 24 h | 0.917 | 0.680 | 44.5 dB | 5.0 |
| `cnn_lstm_24h` | CNN-LSTM | 24 h | 0.896 | −0.876 | 44.1 dB | 3.1 |

\* From earlier on-device checks (46.0 dB on API windows). PL00 has one model (`single_node`). LightGBM (24 h) needs a CPU runtime and isn't switchable yet.

**How the newer models' files were made:** [nz_live/scripts/build_model_meta.py](nz_live/scripts/build_model_meta.py) builds each `meta.json` (features in the model's own training order; same scaler as the deployed models; limits fitted with the same rule as `anomaly.py`). [check_models_on_board.py](nz_live/scripts/check_models_on_board.py) runs the on-device SQNR check. The QNN libraries are the benchmark's builds (`benchmark_qidk`), copied into `nz_live/models/<id>/`.

**Limits in `meta.json`.** Since 20 Nov the local copies of all six registered models' `meta.json` carry grid limits
refitted on API data (`anomaly`), with the 2025 ones kept as `anomaly_2025` (`datasets/api_2026/apply_api_limits.py`,
`--undo` restores). The board still has the 2025 limits until they are pushed (§9.4); with those, every model flags
3–14 % of normal API readings instead of ≤ 3 % ([datasets/api_2026/README.md](datasets/api_2026/README.md)).

**How good these models are.** The R² above is each model's own +15-minute score. Scored on the identical 2025 test
weeks as the new models (20 Nov), the default `all_sensors_1h` is no better than persistence at +1 h on grid (0.862 vs
0.866); see [STATUS.md](STATUS.md).

**New models (20 Nov, `forecast_2025/`) are not in the registry.** They take different inputs (`past` / `future` /
`node`, with weather, clear-sky and calendar) that the live loop doesn't build yet:
[forecast_2025/LIVE_INPUTS.md](forecast_2025/LIVE_INPUTS.md).

**Switching.** On the Live page's Grid card, or `POST /api/live/models {signal, model}` (UI only; the assistant can list models but never switch them). The loop loads the meta, saves the choice in `active_models.json` (it survives restarts), re-forecasts the last 24 h with the new model, and uses it from the next cycle.

## 6. Dashboard (`qidk_export/dashboard/`, SvelteKit on Bun)

| Route | What |
|---|---|
| `/` **Live** | NPU forecasting front page: status tiles; Grid and Palash cards (latest reading, forecast, allowed deviation, next hour, 24 h actual-vs-forecast chart with normal range and flags, fault-test buttons, model selector); recent flags with "Ask"; campus solar next hour; models table |
| `/map` | All 365 nodes on the campus map, node details from `campus.sqlite` (2025 history) |
| `/whatif` **Net zero** | Carbon balance and extra-solar what-if (2025 data). A **net-metering** switch decides whether surplus solar is exported and credited or wasted. The page states the solar needed for zero (+1,225 %, only with net metering or storage) and the floor without export (night load: net can't drop below ~1,118 t). "Today's solar" counts only the 3 meters with a clean 2025, so its multiples are overestimates |
| `/results` **Results** | Are the forecasters worth it? Verdict per horizon (best model vs best baseline), all models × persistence/yesterday on 2025 data with skill, the live shadow scoreboard (24 h / 3 d / 7 d), LightGBM gain importance by meter and input kind, per-hour allowed deviation |
| `/npu` **NPU lab** | The :8080 dashboard (replays, day-ahead, single NPU runs, FP32 vs NPU), through the `/npu-app/` relay; opens on "Run on NPU" |

| API | Use |
|---|---|
| `GET /api/live/status`, `/api/live/history?hours=`, `/api/live/events?since=` | Live state, chart data, flags |
| `GET/POST /api/live/models` | List models / switch (UI) |
| `GET /api/live/results?hours=` | Live shadow scores + offline 2025 results + per-hour limits (Results tab) |
| `POST /api/live/inject` | Fault injection (UI) |
| `POST /api/chat` | Assistant (streams; tool calls) |
| `GET /api/node/[id]`, `/api/node/[id]/series`, `GET /api/whatif` | Map node details and history; Net zero calculator |
| `/npu-app/*` | Relay to `nz_dashboard` |

## 7. Assistant

Qwen3-4B (Q4_0, llama.cpp, 6 threads on the board's CPU; ~6 tok/s; 13–39 s per answer). It answers only by calling tools:
* **6 database tools:** `data_overview`, `search_nodes`, `get_node`, `get_stats`, `compare_nodes`, `show_node` (on the map)
* **`net_zero`:** net-zero calculator
* **5 live tools:** `live_status` (says which model forecasts), `recent_anomalies`, `explain_anomaly`, `list_models`, `model_results` (accuracy vs the baselines, live and 2025; input importance)

The prompt routes by keywords and requires a tool call. It never calls a flag a confirmed fault, and it can't switch models or inject faults.

## 8. Testing

| Test | What | Latest |
|---|---|---|
| A1 | TS windows = Python windows on the same raw files | 1.2e-7 |
| A2 | Fast replay of 16 cached API days through the real `loop.ts` (`scripts/live_replay.ts` + `verify_api_lstm/replay_server.py`) | Grid 41/41, PL00 12/12, injected 20/20 (re-run 17 Nov after each change) |
| A3 | Live run | 24 h: Grid 1.0 % natural flags, PL00 0 % |
| A5 | Power cycle, no USB | Back in ~5.5 min |
| A7 | Assistant question set | Right tool 10/10 |
| M2 | NPU vs FP32 per switchable model | 44.1–47.7 dB |
| Switcher | Switch / reject / persist across restart / 24 h model | Passed on the stand-in server, 17 Nov |
| Persistence | Every model vs persistence and same-time-yesterday (`nz_live/scripts/offline_results.py`) | 2025 +15 min: persistence wins (Grid 0.937 vs best model 0.936); models win for 1 h solar (+0.45) and 24 h grid; yesterday wins 24 h solar |
| Runner | Resident vs one-shot | 3–5 ms vs ~1.8 s per window; same output |
| M4 | Natural grid flag rate on the full API history (1 Jan–19 Nov 2026), every registered model (`datasets/api_2026/m4_flag_rates.py`, 20 Nov) | **All fail with 2025 limits** (3.4–13.7 %; default 9.1 %); all pass with API-fitted limits (0.8–2.2 %), not yet on the board |
| Identical-row comparison | Registered models vs the 20 Nov models and baselines on the 2025 test weeks (`forecast_2025/e1_compare.py`) | Default no better than persistence at +1 h grid (0.862 vs 0.866) |

## 9. Operations

*Checked against the laptop and the board on 20 Nov 2026.*

### 9.1 Setting up from scratch

1. **Python environments.** The scripts expect `.venv` and `.venv-aimet` in this folder. **Neither is here now**
   (copies exist in sibling folders, e.g. `../mideval/.venv-aimet`); the 20 Nov work in `forecast_2025/` and
   `datasets/` ran on the system `python3` (PyTorch 2.14 + CUDA 13, LightGBM, ONNX Runtime). To recreate them:
   ```bash
   uv venv --python 3.10 .venv-aimet
   uv pip install --python .venv-aimet --index-strategy unsafe-best-match \
     --extra-index-url https://download.pytorch.org/whl/cpu -r env/requirements-aimet.txt
   uv venv .venv && uv pip install --python .venv -r env/requirements-main.txt
   ```
2. **QAIRT Docker image** (only to convert models for the NPU or run the x86 HTP emulator): build `qairt:latest` with
   `qidk/Tools/qairt_docker/README.md`, then `docker build -f env/Dockerfile.qairt-fixed -t qairt-fixed:2.47 .`
   (why: `archive/aimet_quantization/04_environment_remediation.md`). Both images are on this laptop (~18 GB each).
3. **October pipeline:** `aimet_quantization/scripts/01_…` to `14_…` in order; script 01 reads `datasets/`. To
   regenerate `parquet_sorted/` from the dump, run `python nodesort.py` inside `datasets/esw/nodes/`.
4. **Board:** the QIDK with adb access. `aimet_quantization/scripts/14_dashboard_start.sh` pushes the runtime and starts
   the :8080 dashboard; the live system's own files are in `/data/local/tmp/nz_live` (§3).

### 9.2 Network: how the board reaches the API and the internet

The laptop is the board's only way out. Two routes to the campus API; the live loop picks one per cycle.

**Route 1: the laptop hotspot (Wi-Fi).** The laptop runs the NetworkManager profile `Hotspot` (SSID `QIDK_DevNet`,
`ipv4.method shared`, 10.42.0.1/24, autoconnect) on its Wi-Fi card `wlp0s20f3`; the board joins as 10.42.0.237 and
uses the laptop's own uplink (wired `IIIT-LAN`). This route also gives the board the internet (Open-Meteo HTTPS works
from the board, 20 Nov). Open the live dashboard from the laptop at `http://10.42.0.237:5173`.

Docker 29 sets the kernel's `FORWARD` policy to `DROP`, so the laptop drops everything the board sends on. **After
every laptop reboot or firewalld reload**, run:
```bash
sudo iptables -I DOCKER-USER -i wlp0s20f3 -j ACCEPT
sudo iptables -I DOCKER-USER -o wlp0s20f3 -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
```
(`wlp0s20f3` is this laptop's Wi-Fi card; on another machine check `nmcli d`.) Check from the board:
`adb shell curl -sI http://10.2.203.214:9999/` (expect 200).

The **permanent fix is still not applied** (no `/etc/docker/daemon.json` on 20 Nov): tell Docker not to drop forwarded
traffic, then restart it (if the file exists by then, add the key instead of overwriting):
```bash
echo '{"ip-forward-no-drop": true}' | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker
```

**Route 2: USB fallback (since 19 Nov).** On the board `127.0.0.1:19999` → `adb reverse` over the cable → laptop
`127.0.0.1:19999` → `socat` → API. [nz_live/laptop/usb_uplink.sh](nz_live/laptop/usb_uplink.sh) runs the relay and
re-adds the reverse after a replug (every 10 s), as the systemd **user** service `nz-usb-uplink.service`
([unit](nz_live/laptop/nz-usb-uplink.service), installed in `~/.config/systemd/user/`, enabled). The laptop opens the
API connection itself, so Docker's `FORWARD` drop doesn't affect it. It carries the API only, not the internet.
Needs the cable, adb authorized, and the laptop awake on a campus network.

**How the loop chooses.** `LIVE_API` (comma-separated; `start.sh` sets `http://10.2.203.214:9999,http://127.0.0.1:19999`,
overridable by `live_api.url` on the board) is the ordered route list, and `loop.ts` always appends the USB route
unless `LIVE_USB=0`. At the start of every cycle the clock sync probes them in order (6 s each) and the cycle uses the
first that answers: it moves to USB within one cycle and back to Wi-Fi as soon as that works. Every switch is logged
(`API route: …`); the Live page's data-feed tile and `/api/live/status` (`api.route`) show the route in use. After a
board reboot the loop can stay on USB while the hotspot also works; that is harmless.

Tested 19 Nov: A2 replay unchanged (Grid 41/41, PL00 12/12, injected 20/20); direct route refused → USB in < 1 s,
silent → USB after ~7 s, working → direct; and for real when the hotspot went down: the loop picked USB, caught up the 2
missed checks and kept running.

**The board's clock.** It starts at 20 Sep 2023 after every reboot and is set by NTP (`time.android.com`, automatic
time on) once the hotspot route reaches the internet: on 20 Nov this took ~7 min after boot. The live loop doesn't
depend on it (it takes the time from the API), but HTTPS does: until NTP succeeds every certificate is "not yet
valid", so anything fetching over HTTPS (e.g. a weather fetch) must retry rather than give up.

**Also needed:** the laptop must stay awake while it is the router (lid closed must not suspend on power).
adb over Wi-Fi isn't enabled (`adb tcpip 5555` was denied on 16 Nov), so board-side checks need the USB cable.

### 9.3 Setup changes and how to undo them

| Where | Change | Undo |
|---|---|---|
| Laptop | NetworkManager profile `Hotspot` (above), priority 100 | `nmcli c down Hotspot; nmcli c modify Hotspot connection.autoconnect no` |
| Laptop | `DOCKER-USER` iptables rules (above) | gone after a reboot / firewalld reload |
| Laptop | `nz-usb-uplink.service` (systemd user service) | `systemctl --user disable --now nz-usb-uplink; rm ~/.config/systemd/user/nz-usb-uplink.service` |
| Board | Joined `QIDK_DevNet` (saved network) | `adb root; adb shell cmd wifi forget-network <id>` |
| Board | `settings put global captive_portal_mode 0` (else Android marks the intranet-only hotspot "no internet" and won't rejoin after a reboot; checked 20 Nov: `0`) | `adb shell settings put global captive_portal_mode 1` |
| Board | Boot service `/system/etc/init/nz_live.rc` ([nz_live/board/nz_live.rc](nz_live/board/nz_live.rc)): runs `/data/local/tmp/nz_live/start.sh` as `shell` on `sys.boot_completed=1` | `adb root && adb remount && adb shell rm /system/etc/init/nz_live.rc && adb reboot` |
| Board | `start.sh` lists both API routes (old copy `start.sh.bak-19nov`) | copy the backup back |

### 9.4 Routine tasks

| Task | How |
|---|---|
| Is everything up? | `adb shell curl -s http://127.0.0.1:5173/api/live/status` (`api.ok`, `api.route`, `npu.ok`, `next_check`); ports :8080 / :8090 `/health` / :5173 |
| Restart everything | `adb root; adb shell stop nz_live; adb shell start nz_live; adb unroot` |
| Logs (board) | `/data/local/tmp/nz_live/logs/{supervisor,nz_dashboard,llama-server,dashboard}.log` |
| USB route status / logs | `systemctl --user status nz-usb-uplink`; `adb reverse --list` shows `tcp:19999`; `journalctl --user -u nz-usb-uplink` |
| Deploy a dashboard build | `bun --bun vite build`; push `build/` to `nz_live/dashboard/build.new`, swap, `kill $(cat nz_live/run/dashboard.child.pid)` (the supervisor restarts it; history is backfilled) |
| Add a switchable model | its QNN `libw8a16.so` / `libfp32.so` + `meta.json` in `nz_live/models/<id>/`; a line in `registry.txt`; an entry in `registry.json`; restart `nz_dashboard` (kill its child pid) |
| Push changed anomaly limits (e.g. the API-fitted ones from `datasets/api_2026/apply_api_limits.py`) | `adb push nz_live/models/<id>/meta.json /data/local/tmp/nz_live/models/<id>/` for the 5 switchable models; `adb push aimet_quantization/dashboard/www/data/all_sensors_1h/meta.json /data/local/tmp/nz_aimet/dashboard/www/data/all_sensors_1h/`; then restart the Bun dashboard (`kill $(cat /data/local/tmp/nz_live/run/dashboard.child.pid)`): the live loop caches each model's `meta.json` in memory |
| Undo the boot service | see §9.3 |

## 10. Known limits

* The laptop is still the only way to the API: over the hotspot (whose forwarding rule is lost on a laptop reboot until the `daemon.json` fix is applied) or, since 19 Nov, over the USB fallback. Both stop if the laptop sleeps.
* 4 of 11 campus-model inputs aren't served (neighbour copies / zero); campus solar and KH04 room load ("Demand") can't be checked live.
* Grid −40 % faults at morning load fall inside the limit.
* The one-shot `qnn-net-run` (used when no runner is warm) often crashes with SIGSEGV while exiting, after writing its output: harmless to results, but it leaves a crash report each time. The live loop avoids it by warming runners before each cycle and before catch-up (`archive/worklogs/WORKLOG_2026-11-19.md` §3).
* Per check, starting the 7 runners takes ~10.8 s of CPU work (more while the assistant is busy); per window the NPU is 3–5 ms vs 0.3–0.5 ms for ONNX Runtime on the CPU (`archive/worklogs/WORKLOG_2026-11-19.md` §5).
* The limits on the board are the 2025 ones, and with them every model over-flags normal API readings (M4 above); the API-fitted limits are ready locally and need pushing (§9.4).
* Synthetic data: no real events or drift; every live flag on normal data is a false alarm.
* After a board reboot, HTTPS from the board fails until NTP sets the clock (§9.2); the current live loop uses plain HTTP and isn't affected.
* NPU is slower than the CPU at batch 1 for these models ([benchmark_qidk/RESULTS.md](benchmark_qidk/RESULTS.md)); fused and batched NPU calls are where the NPU should pay off; their timing on the board is open ([STATUS.md](STATUS.md) items 4a/4b).
