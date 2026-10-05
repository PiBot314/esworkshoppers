# 17 — Dashboard Build Log (executed 2026-10-05)

Plan: doc 16 §7 (fast build with on-demand live NPU inference). Files: `scripts/13_dashboard_pack.py`, `scripts/14_dashboard_start.sh`, `dashboard/server/nz_dashboard.cpp`, `dashboard/build.sh`, `dashboard/www/`.

## F1 — Data pack (`13_dashboard_pack.py`)
* **Observation that shaped the format:** consecutive test windows overlap by 95 of 96 rows (verified `X[i+1][:-1] == X[i][1:]` for all three models). So the pack stores one **timeline** of `n_test + 95` rows, and window *i* = `timeline[i : i+96]`. That's 1.1 MB instead of 107 MB for the 58-feature models.
* Contents per model (`dashboard/www/data/<model>/`): `timeline.f32` (normalized inputs), `y.f32` (actuals), `npu.f32` (**recorded QIDK HTP W8A16 outputs** from P4), `fp32.f32`, and `meta.json` (feature names with physical-unit min/max from the rebuilt scalers, model vs data capacities, per-hour anomaly medians/limits, typical-day P10/P50/P90 per IST hour, device KPIs from doc 14, UTC start of the timeline).
* Physical units for upload/download: `x_phys = x_norm · (max − min) + min` per feature. `was_imputed` stays 0/1. The cyclic features are scaled too (doc 11 finding 1), and the browser recomputes them from timestamps.
* Size: 1.85 MB (1 h) + 17.5 MB (24 h: 288 outputs × 4,740 windows × 3 arrays) + 0.44 MB (single) = **19 MB**, served over USB.
* Check: `timeline[i:i+96] == X_test[i]` asserted in the script. Typical-day solar median by IST hour runs from ≈25 W at night to 373 kW at 11:00, which looks physically right.

## F2 — Device server (`nz_dashboard`)
* **C++17 + cpp-httplib 0.59 (header-only)**, cross-compiled with the NDK r26c clang for `aarch64-linux-android31`, `-static-libstdc++` → a single 1.9 MB binary. Static files from `dashboard/www`, `GET /api/status`, `POST /api/run?model=&variant=` (body = float32 window).
* `/api/run` writes the window as `.raw`, runs `qnn-net-run` with `lib<variant>.so` on the **HTP** (`aimet_w8a16`) or **CPU** (`fp32`) with `--profiling_level basic`, parses `qnn-profile-viewer` for execute/finalize times, and returns outputs + timings. A mutex serializes NPU jobs. The body size is validated (`96 × F × 4` bytes).
* Issues and fixes:
  | Issue | Cause | Fix |
  |---|---|---|
  | Launcher hung at the start step (server *was* running) | `adb shell` waits for any process holding its pty, even with `setsid nohup … &` or `adb shell -T -n` | Server got a `--daemon` mode (double fork, `setsid`, stdio → log), so the parent exits and adb returns. Launcher now takes ≈1 s |
  | Two server instances | My `pkill -f` cleanup matched its own shell (third time this pattern bit; see memory). The second instance couldn't bind | Use `pkill nz_dashboard` (name match) on the device |
  | `413 Payload Too Large` from a Python client | urllib defaults to `application/x-www-form-urlencoded`; httplib caps form bodies at 8 KB | Raised `CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH` to 4 MB. The browser sends `application/octet-stream` anyway |
  | CPU execute time = −1 | CPU backend profile labels are `GRAPH_EXECUTE` / `GRAPH_FINALIZE` | Backend-specific labels |
* **Verification:** live API outputs vs the recorded P4 device outputs for the same window (test index 10):

  | Model | HTP W8A16 max\|live − recorded\| | CPU FP32 max\|live − recorded\| | HTP wall time | of which graph finalize | NPU execute |
  |---|---|---|---|---|---|
  | 1 h | 4.9e-7 | 4.0e-7 | 2.08 s | 1.51 s | 4.9 ms |
  | 24 h | 5.0e-7 | 6.1e-7 | 1.81 s | 1.23 s | 4.7 ms |
  | single | 3.5e-7 | 4.7e-7 | 1.44 s | 0.86 s | 5.6 ms |

  Differences are at JSON float-printing precision, so **live = recorded**. As expected (doc 16 §7), a live click costs ~1.5–2 s, dominated by `qnn-net-run`'s per-launch graph preparation, while the NPU computation itself is ~5 ms. Both numbers are shown in the UI so the audience sees the distinction.

## F3 — Frontend (`dashboard/www/`: `index.html`, `style.css`, `app.js`, `vendor/uPlot` 1.6.32)
* **Why uPlot:** ~50 KB, fast on thousands of points, supports bands (fill between series) for the "typical" envelope. Bundled locally because the demo must work without internet (served from the QIDK).
* **Single page, five tabs** (keys `1`–`5`): Campus replay (1 h model), Plant SL-PL00-00 (single-node model), Day-ahead 24 h, Run on NPU, Under the hood. Dark, high-contrast theme for projectors, designed for 1600×1000 / 1920×1080 (grid collapses below 1100 px).
* **Typical vs anomalous** (the core visual): each signal is drawn with its NPU +15-min forecast and a shaded band = forecast + per-hour median ± per-hour limit (exactly the thresholds of `anomaly.py`). Readings outside the band get a red marker. A residual chart shows the same decision as "residual vs ±limit". The Day-ahead tab adds a typical-day P10–P90 envelope by IST hour.
* **Fault injection in the browser:** dropout (→ 0 W), spike (+50 % of capacity), stuck (holds the current value), panel damage (−40 %), for 1–16 steps from the current reading. Faults change only the *actual*, so the verdict is recomputed against the recorded NPU forecast, the same maths as the offline evaluation.
* **Run on NPU:** model and window selector; CSV template download (physical units, timestamps in ISO UTC, 97th row = next reading); CSV upload (validation of columns, 96/97 rows, numeric values, 15-min spacing; warning for values outside the training range); in-page channel editor (zero / scale / freeze / spike over the last N steps, before/after preview); buttons for HTP W8A16 and CPU FP32. Results show NPU execute time, graph-prepare time, total and round trip, a "match vs recorded run" check, the forecast table, a chart against the recorded NPU output and actuals, and the anomaly verdict for the next reading.
* **Design decisions made while building:**
  | Decision | Reason |
  |---|---|
  | Demand verdicts shown as **"n/a"** (grey), not ANOMALY | First screenshot showed a red Demand "ANOMALY" in normal conditions. The Demand head has negative R² in FP32 and a 49 % natural flag rate (doc 12), so showing its alarms would mislead the audience |
  | Replay starts at 11:00 IST on day 4 instead of an arbitrary index | The first screenshot opened at 19:30 (solar flat at 0). The demo should open on an informative daytime view |
  | Anomaly markers enlarged (14 px) | The current reading sits on the right edge of the chart, and small markers were easy to miss on a projector |
  | Side panel widened to 470 px, numbers `nowrap` | Verdict column was clipped and values wrapped ("574.0 / kW") |
  | URL test hooks `#live-run`, `#inject-<fault>`, `#csv-selftest`, plus `#<tab>` | Allow headless verification of every path, and deep links for presenting |
  | Targets for the anomaly check of uploaded windows computed from the 97th row's sensors (Σ solar `active_power_W`; SB00 `gd_power_W` + KH95 `power_total_W`; Σ room `power_W`) | Same definition as `train_all_sensors_lstm.py`, so uploaded data is judged exactly like the dataset |
* **Verification (headless Chromium against the live QIDK server, screenshots in `dashboard/screenshots/`):**
  | Check | Result |
  |---|---|
  | All five tabs render, no console errors | ✅ Console captured per tab with `--enable-logging` (method checked against a page with a deliberate error, which it reported) |
  | Live run from the browser (`#live-run`) | ✅ NPU execute 4.76 ms. Output vs recorded P4 run max \|Δ\| = 4.6e-7 |
  | Fault injection (`#inject-dropout`, `#inject-spike`) | ✅ Solar → ANOMALY (dropout: −417 kW vs ±275 kW limit; spike: +344 kW). Grid stays ok, Demand n/a |
  | CSV round trip (`#csv-selftest`): template → real upload parser → NPU | ✅ sensors max \|Δ\| 1.5e-8, **cyclic features recomputed from timestamps 2.9e-18**, next-reading actuals identical (Solar 488.2 kW, Grid 397.1 kW). NPU 4.40 ms |

## F4 — Launcher and operation
`scripts/14_dashboard_start.sh`: restarts the server on the QIDK (`--daemon`), pushes only changed files (`adb push --sync`), ensures runtime + model libraries are present, sets up `adb forward tcp:8080`, prints the USB and Wi-Fi URLs, opens the browser. Run time ≈1 s when nothing changed. `--stop` stops the server. Rebuild the server with `dashboard/build.sh` (≈12 s).

## Time spent
Plan revision → working, verified dashboard: roughly 45–60 minutes of execution, in line with the estimate given before starting (not precisely timed).
