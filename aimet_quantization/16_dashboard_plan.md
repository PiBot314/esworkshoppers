# 16 — Plan: Live Demo Dashboard Served by the QIDK (2026-10-05)

Goal: a presentation-grade GUI that **runs the quantized models live on the QIDK's NPU**, visualises typical vs anomalous behaviour, and can be opened in a laptop browser *or* on the QIDK's own screen.

## 1. Facts checked before planning
| Fact | Consequence |
|---|---|
| QIDK has a 1080×2400 touchscreen (144 Hz) and **Chrome + WebView** installed | A web dashboard can also be the "on-device app": Chrome full-screen at `http://127.0.0.1:8080`. No APK needed |
| QIDK on Wi-Fi `10.2.135.201/20`; laptop `10.2.137.163/20`, **same subnet**; device has internet | Laptop browser can open `http://10.2.135.201:8080` directly. Fallback over USB: `adb forward tcp:8080 tcp:8080` → `http://localhost:8080` |
| **No Python on the device**; Android NDK r26c + QNN headers + **QNN SampleApp source** in our QAIRT image | Server must be a **native C++ binary**. The SampleApp shows how to load our model `.so` and keep a persistent HTP context |
| `qnn-net-run` spends ≈1.5 s preparing the graph per launch, then ≈5 ms per inference | Spawning it per request would feel sluggish. A **persistent in-process QNN context** gives ~5 ms live inference |
| No JDK/Gradle/Android SDK on the host (only a Java runtime) | A native Android app would need extra installs (several GB) plus JNI glue. Kept as option B |

## 2. Recommended architecture (option A: web dashboard served by the QIDK)
```
QIDK (Android 14, SM8650)
 └─ nz_dashboard  (single native aarch64 binary, started via adb, runs in background)
     ├─ HTTP server (cpp-httplib, header-only)       :8080
     │    GET  /                    → static dashboard (HTML/CSS/JS, chart lib bundled; works offline)
     │    GET  /api/meta            → models, features, capacities, test period, KPIs from our reports
     │    GET  /api/step?model&t    → runs the model on the NPU for time t, returns forecast,
     │                                 actual, residual, threshold band, flag, NPU latency
     │    POST /api/inject          → apply a fault (dropout / spike / panel damage / stuck sensor)
     │    GET  /api/day?t           → 24 h day-ahead forecast (24 h model) vs actual
     │    GET  /api/typical         → typical-day profiles (median & band per IST hour)
     └─ QNN runner (adapted from SDK SampleApp): dlopen libQnnHtp.so + lib<model>_w8a16_htp.so,
          compose + finalize graphs ONCE at start-up, graphExecute per request
 └─ data pack (~20 MB): normalized feature timeline + actual Watts + timestamps + thresholds + FP32 outputs
     (windows are sliced on the fly from the timeline instead of storing 300 MB of overlapping windows)

Viewers: laptop browser (Wi-Fi or adb forward) │ projector │ QIDK screen via Chrome full-screen
```

### Dashboard content (presentation-oriented)
| Panel | What it shows | Why it matters for the demo |
|---|---|---|
| KPI header | Live NPU latency (ms), model, quantization (W8A16), fidelity to FP32 (dB), anomalies in view | "It really runs on the NPU, fast and accurate" |
| **Live replay timeline** | Actual vs NPU forecast for Solar / Grid / Demand, scrolling through the test period with play/pause/speed/jump-to-date | Core story: forecasting works |
| **Typical band** | Shaded "normal" envelope (forecast ± hour-specific threshold) around the actual signal; points outside turn red | Shows *typical vs anomalous* at a glance |
| Residual chart | Residual with threshold band per hour, anomaly markers | Explains how detection works |
| **Fault-injection buttons** | Sensor dropout · spike · stuck value · 40 % panel loss on the next reading → the model's verdict appears live | Live, interactive "it catches faults" moment |
| Day-ahead view | 24 h forecast (24 h model) vs actual, net-zero balance (Demand − Solar), energy totals | Planning use case |
| Typical-day profiles | Median and inter-quantile band of solar/grid by IST hour (whole dataset), with the current day overlaid | What "normal" looks like |
| Plant view | Single-plant model (SL-PL00-00) forecast vs actual + its anomaly flags | Plant-level detection |
| "Under the hood" tab | FP32 vs NPU output overlay, our verification table (V0–V5), pipeline diagram | For technical questions |

Design: dark/light, high contrast for projectors, responsive for the QIDK portrait screen (1080×2400) and a landscape laptop or projector.

## 3. Option B (alternative): native Android app on the QIDK
Kotlin app with a WebView or native charts + JNI to the same C++ QNN runner. **Pros:** app icon, launches like a product. **Cons:** install JDK 17 + Android SDK/Gradle (~3–5 GB), signing, JNI packaging of the HTP libs. Roughly 1.5–2× the effort of A, for the same visuals. Option A already gives a full-screen on-device UI via Chrome. **Recommendation: A**, with B only if an installable app is a requirement.

Option C (fallback, not "served by the QIDK"): Python server on the laptop calling the QIDK via adb per request. Fast to build, but the device isn't the server, so it doesn't meet the brief.

## 4. Execution phases
| Phase | Work | Exit check |
|---|---|---|
| D0 | Decisions (questions in §6) | Answers recorded |
| D1 | **Data pack exporter** (host, Python): timelines for the 3 models, actual Watts, timestamps, thresholds, FP32 outputs, typical-day stats → compact binary + JSON | Windows sliced from the pack == our test windows (bit-exact) |
| D2 | **Native QNN runner** (C++ from SampleApp): persistent HTP context for the 3 model libs, `run(window) → outputs` | Outputs == `qnn-net-run` outputs (bit-exact) on 100 windows. Latency logged |
| D3 | **HTTP server + API** (cpp-httplib) around the runner + anomaly logic (port of `anomaly.py`) | `curl` through `adb forward` returns correct JSON. Flags == host `anomaly.py` |
| D4 | **Frontend** (single page, bundled chart library, no internet needed): panels from §2 | Works in laptop Chrome/Firefox and on the QIDK's Chrome |
| D5 | **Demo mode**: start/stop script, autoplay "story" (normal day → injected fault → detection → day-ahead), keyboard shortcuts for presenting | Rehearsal run end-to-end |
| D6 | Verification + docs (`17_…` logs, README section, presenter cheat-sheet) | All checks pass. Docs written |

Estimated effort: D1 ½ day · D2 1 day (the riskiest: QNN API integration) · D3 ½ day · D4 1 day · D5–D6 ½ day.

## 5. Risks
| Risk | Mitigation |
|---|---|
| QNN API integration (composeGraphs/finalize/execute, tensor memory) | Start from the SDK SampleApp, which already loads model `.so` files. Fallback: server spawns `qnn-net-run` per request (≈1.5 s latency, still demo-able) |
| Wi-Fi at the venue blocks device-to-device traffic | `adb forward` over USB, or the QIDK's own screen |
| Background process killed when adb disconnects | Start with `setsid nohup`. Provide a one-line restart script |
| Several models open on the HTP at once | One context per model (small models). Or load on demand |
| Presentation without internet | All JS/CSS bundled into the binary or the data folder. No CDN |

## 6. Decisions (answered 2026-10-05)
| Question | Decision | Effect on the plan |
|---|---|---|
| Delivery form | **Option A: web dashboard served by the QIDK** | Native C++ server on the device. No APK |
| Presentation setup | **Laptop + projector over USB** | Primary access: `adb forward tcp:8080 tcp:8080` → `http://localhost:8080`, which is immune to venue Wi-Fi isolation. Layout optimized for a landscape projector (1920×1080 / 1280×720). Wi-Fi URL and QIDK screen still work, but aren't the design target |
| Models | **All three** | Main view: 1 h campus model + anomaly panels. Tab "Plant": single-node model. Tab "Day-ahead": 24 h model |
| Demo flow | **Manual only** | No scripted story. Presenter controls: play/pause, speed, jump to date/"interesting moments" list, fault-injection buttons, keyboard shortcuts. D5 becomes "presenter controls + launch script" |

A one-command launcher (`scripts/13_dashboard_start.sh`) will push the files if needed, start the server on the QIDK, set up `adb forward`, and open the browser.

---
## 7. Revised plan (adopted 2026-10-05): fast build with on-demand live NPU inference
**Why revised:** the full plan (persistent QNN context in a custom C++ runner, D2) was estimated at 3–4 h of execution, most of it D2. The audience is technical and this is a demonstration, not a product. The user asked for ~30 min and then added live inference on user-supplied sensor inputs. So:

| Piece | Full plan (§2/§4) | **Adopted** | Trade-off accepted |
|---|---|---|---|
| Replay timeline | NPU run per step, live | **Recorded on-device NPU outputs** (`out/<model>/device/htp_aimet_w8a16_out_test.npy`, produced by the QIDK in P4) | Not computed at the moment of display, but they *are* real NPU results |
| Fault injection | server-side | **In the browser**: a fault changes only the actual reading; the residual is recomputed against the recorded forecast with the same per-hour thresholds | Identical maths to `anomaly.py`, no inference needed |
| Live inference | persistent context, ~5 ms | **On demand: "Run on NPU now"** → server writes the window as `.raw`, runs `qnn-net-run` with the W8A16 model library on the HTP, returns outputs + timings (~1.5 s incl. graph preparation) | Seconds instead of milliseconds per run. NPU execute time is still measured and shown |
| Inputs for live runs | n/a | **(a)** CSV upload: 96 rows × (timestamp + sensor columns named as in the model metadata), physical units or already normalized, optional 97th row = actual reading for the anomaly check. **(b)** Template download of any test window. **(c)** In-page editor: zero/scale/spike/freeze one sensor channel over a time range | Normalization uses the rebuilt dataset's scaler min/max (doc 11) |
| Server | C++ + QNN API | **C++ static file server + one `/api/run` endpoint** (cpp-httplib, NDK-built, spawns `qnn-net-run`) | No QNN API integration needed |
| Access | Wi-Fi / USB / QIDK screen | USB `adb forward` (primary), others still work | — |

### Revised phases
| Phase | Work | Check |
|---|---|---|
| F1 | Data pack (host): test-period feature timelines, actuals and recorded NPU/FP32 forecasts in Watts, timestamps, thresholds, scalers, typical-day stats → `dashboard/www/data/` | Windows sliced from the pack == `data/windows` test windows |
| F2 | Device server `nz_dashboard` (C++, cpp-httplib) + run endpoint | `curl` via adb forward. Live output == recorded device output for the same window |
| F3 | Frontend (HTML/JS, uPlot bundled): Replay (1 h), Plant (single), Day-ahead (24 h), **Live NPU input**, Under the hood | Opens in the laptop browser via USB |
| F4 | Launcher `scripts/13_dashboard_start.sh`, docs (`17_…`), README section | End-to-end run-through |

**Status (2026-10-05):** F1–F4 complete and verified, see doc 17.
