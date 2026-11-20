# Plan: switchable forecasting models, and a system that adapts to new nodes and capacity

*Written 2026-11-17. **Status (17 Nov, evening): built for the 5 NPU models + the 2 deployed ones**:
* registry (§3.1)
* `nz_dashboard` running any registered model (§3.2, HTP and CPU FP32; the ONNX Runtime path for LightGBM isn't done yet)
* per-model limits (§3.3)
* switching with a 24 h re-forecast, persisted across restarts (§3.4)
* selector and models table on the Live page (§3.5, the models table is on the Live page, not in the NPU lab)
* `list_models` tool (§3.6)

Checks passed: M2 (44.1–47.7 dB) and A2 with the default models. **Not yet done:** shadow mode, M3/M4 on API data for the new models, M6, LightGBM. As built: [SYSTEM_DESIGN.md](../SYSTEM_DESIGN.md) §5.* Builds on [plan_reduced_shortterm.md](plan_reduced_shortterm.md) (live loop, A1–A8) and the QIDK benchmark in [benchmark_qidk/](../benchmark_qidk/).*
*Out of scope: the energy autoencoder (`extras/autoencoder/`).*

---

## TL;DR

* **9 forecasting models, 3 essentially different approaches:** stacked LSTM (7 models, including the 2 LightGBM-guided ones), CNN-LSTM (1) and LightGBM (1).
* **Switching is cheap for 8 of the 9.** All the campus models take the same input as the live `all_sensors_1h`: 11 nodes, 58 features, a 96-step window, the same `scaler_params.json` min-max and the same step-major output. The live loop's window builder doesn't change. 5 of them already have NPU builds on the board (`/data/local/tmp/nz_bench/qnn/`).
* **What each model still needs:** an entry in a model registry, its own per-hour anomaly limits (the flag rule is meaningless without them), and the same API-data checks the current models passed (A1/A2, flag rates, injected-fault catch rates).
* **Palash solar (PL00) has only one model** (`single_node`). The switcher applies to the campus signals (Grid, campus Solar).
* **Best candidate (corrected 17 Nov, after scoring every model the same way):** R² of the **+15-minute** forecast on the 2025 test split, which is the step the flag rule uses. **The deployed `all_sensors_1h` is still the best Grid model** (0.824), ahead of `guided_h4` (0.778), `guided_24h` (0.743), `baseline_24h` (0.680) and `baseline_h4` (0.666); `cnn_lstm_24h` is unusable for Grid (−0.876). `guided_h4` is the best Solar model (0.964 vs 0.966 for the deployed one; effectively tied). The earlier recommendation of `guided_h4` rested on all-step, 1-hour benchmark numbers that the deployed model wasn't part of. **The default stays `all_sensors_1h`.**
* **Effort:** about 3–4 days AI-assisted (§6).
* **Part 2 (§7):** making the system adapt to new nodes and added solar is worth building, even on synthetic data. Most of it is configuration plus a model-free per-node detector, not retraining.

---

## 1. The models

| id | Family | Horizon | Inputs | Test R²: Solar / Grid | Runs on | NPU build on the board |
|---|---|---|---|---|---|---|
| `all_sensors_1h` | 2-layer LSTM, 140k params (team, `model2/`) | 1 h (4 × 15 min) | 11 nodes / 58 | 0.852 overall*; Grid ≈ 0.76* | NPU W8A16 | ✅ `nz_aimet` (**live**) |
| `all_sensors_24h` | 2-layer LSTM, 282k params (team) | 24 h (96) | 11 / 58 | 0.339 overall* | NPU W8A16 | ✅ `nz_aimet` (replay only) |
| `baseline_h4` | LSTM, retrained in the common setup (`common/`) | 1 h | 11 / 58 | 0.909 / 0.607 | NPU W8A16 | ✅ `nz_bench` |
| `guided_h4` | Same LSTM, LightGBM-gated inputs + teacher | 1 h | 11 / 58 | **0.954 / 0.721** | NPU W8A16 | ✅ `nz_bench` |
| `baseline_24h` | LSTM | 24 h | 11 / 58 | 0.778 / −0.060 | NPU W8A16 | ✅ `nz_bench` |
| `guided_24h` | Guided LSTM | 24 h | 11 / 58 | 0.800 / −0.067 | NPU W8A16 | ✅ `nz_bench` |
| `cnn_lstm_24h` | Conv1d ×2 → LSTM → dense | 24 h | 11 / 58 | 0.739 / −1.776 | NPU W8A16 | ✅ `nz_bench` |
| `lightgbm_24h` | Gradient-boosted trees | 24 h | 11 / 58 | 0.752 / 0.198 | **CPU only** (ONNX Runtime) | – (ONNX in `nz_bench/onnx`) |
| `single_node` | LSTM, 55k params (team) | 1 h | PL00 / 10 | 0.957 (PL00) | NPU W8A16 | ✅ `nz_aimet` (**live**) |

\* Different evaluation from the `common/` rows (team report and rebuilt data, not the common split). Re-score all models on one split and on the API data before comparing them in the UI (§4).

Demand R² is negative for every model; Demand stays hidden, as it is today.

## 2. What stays the same

* **The live loop's ingest and windows** (`features.ts`): same nodes, same features, same scaling. A1 has to be re-run once per model to prove this, not assumed.
* **The detection rule** (`detect.ts`): |actual − forecast − median[hour]| > limit[hour]. Only the numbers come from the active model.
* **The event log, history, banner and assistant tools.** Every point and event already records `model` and `variant`.

## 3. Design

### 3.1 Model registry (one JSON file on the board, read by both servers)
`/data/local/tmp/nz_live/models.json`, one entry per model:
```json
{ "id": "guided_h4", "family": "lstm-guided", "horizon": 4, "domains": ["Solar", "Grid", "Demand"],
  "signals": ["grid"], "input": "campus58", "runtime": "htp",
  "artifacts": { "aimet_w8a16": "/data/local/tmp/nz_bench/qnn/guided_h4/lib….so", "fp32": "…" },
  "meta": "/data/local/tmp/nz_live/models/guided_h4/meta.json" }
```
`meta.json` has the same shape as the deployed ones: features (with min/max), `model_capacities_W`, `lookback`, and an `anomaly` block (per-hour median and limit).

### 3.2 `nz_dashboard` (C++): run any registered model
* Replace the hard-coded `FEATURES` map with the registry. `/api/run?model=<id>` builds the `qnn-net-run` command from the entry's artifact paths.
* Add `runtime: "ort"` for LightGBM. Run it with ONNX Runtime on the CPU through the existing `nz_bench/bin/ort_bench`, or a ~50-line ORT runner built the same way. Report `backend: "CPU (ONNX Runtime)"` so the UI never claims NPU.
* Keep `/api/run`'s current behaviour for the three existing ids, so the NPU lab and the Android app don't change.

### 3.3 Anomaly limits for each model
* Fit with the existing `fit_thresholds` (K = 4 robust σ per IST hour, floor fraction as today) on each model's **validation-split residuals**, on the first step (+15 min) for the 24 h models.
* Write the result into the model's `meta.json` `anomaly` block. Keep the 2025 thresholds approach; refitting on a week of API data was shown to raise false flags (plan_reduced_shortterm §2.4).

### 3.4 Live loop
* The active model per signal comes from `LIVE_DIR/active_models.json` (default: today's choice). `SIGNALS[grid].model` becomes state, not a constant.
* **Switching** (POST `/api/live/model {signal, model}`, UI only; the assistant can't switch):
  1. load the model's meta
  2. clear the forecast cache for that signal
  3. log a "model switched" marker in the history, so the chart shows a vertical line
  4. if possible, backfill the last 24 h with the new model, as the start-up backfill does, so the chart compares like with like
* **Shadow mode** (optional, cheap at ~5 ms per run): also run one candidate model each cycle and store its forecast next to the active one, without flagging. That gives a live, like-for-like comparison of any two models on the API data.

### 3.5 UI
* **Live page:** a model selector in each signal card, showing family, horizon, runtime (NPU/CPU) and test R². PL00 shows "single_node (only model)". Marker lines on the chart where the model changed.
* **NPU lab:** a "Models" table with every model's test R², SQNR (NPU vs FP32), latency (NPU/CPU) and parameter count, from `benchmark_qidk/results`. Add a "run this window on all models" button.

### 3.6 Assistant
* `live_status` and `explain_anomaly` already return `model`. Add the family and horizon so answers can say "flagged by the guided LSTM (1 h)".
* One new read-only tool, `list_models`: the registry with each model's accuracy and latency. Switching models is never a tool.

## 4. Verification (per model, before it can be selected)

| # | Check | Pass |
|---|---|---|
| M1 | Same TS windows → model on PC (ONNX FP32) vs Python `common/benchmark.py` on the same windows | Outputs equal to 1e-4 |
| M2 | NPU W8A16 vs FP32 on API windows (as in §2.5 of the short-term plan) | SQNR ≥ 40 dB |
| M3 | Fast replay of the 16 cached API days (`scripts/live_replay.ts`) with the model active | Flags identical to an offline replay using the same model, thresholds and FP32 |
| M4 | Natural flag rate on API data | Grid ≤ 3 % (today's model: 2.3–3.0 %) |
| M5 | Injected faults (dropout, +20 %, +50 %, −40 %), 150 each | Catch rates reported in the UI table; no minimum |
| M6 | Switch back and forth 10 times while live | No missed bins; events and history keep the right `model` |

Models that fail M4 can still be shown in shadow mode and in the NPU lab, but can't be chosen for flagging.

## 5. Risks

| Risk | Mitigation |
|---|---|
| A `common/` model's scaling turns out not to match the deployed meta | M1 catches it; generate that model's `meta.json` features from `scaler_params.json` directly |
| 24 h models flag a lot at +15 min | Expected (Grid R² < 0.2); they are comparison-only unless M4 passes |
| LightGBM on the CPU slows the cycle | ~2 ms per run on the board's CPU (benchmark); negligible |
| `nz_dashboard` runs one NPU job at a time | Shadow mode adds one ~2 s job per cycle; backfill on switch takes ~3 min. Both fit well inside the 15-minute cycle |
| `nz_bench` builds were made for the benchmark, not deployment | Copy them into `nz_live/models/` with their meta; never point the live system at `nz_bench` |

## 6. Work order and effort (AI-assisted)

| # | Task | Days |
|---|---|---|
| 1 | Registry + `nz_dashboard` generalised (HTP and ORT runtimes) | 0.75 |
| 2 | Fit per-hour limits for the 7 campus models; write their `meta.json` | 0.5 |
| 3 | M1–M5 for `guided_h4` and `baseline_h4` first, then the 24 h models | 1 |
| 4 | Live loop: active model per signal, switch endpoint, backfill on switch, shadow mode | 0.75 |
| 5 | UI selector + NPU-lab models table + `list_models` tool | 0.5 |
| 6 | M6, then deploy | 0.25 |
| | **Total** | **≈ 3.75** |

---

## 7. Making the system adapt (new nodes, more solar, …)

**Yes, it's worth building, and synthetic data is fine for it.** What has to adapt is the *contract*: which nodes exist, their units and their capacities. That doesn't depend on whether the numbers are real. The synthetic API is a good test bench, because nodes and scenarios can be added on demand.

The key split is between what can adapt immediately and what needs retraining:

| Change | Adapts with no retraining | Needs retraining |
|---|---|---|
| **A new node starts reporting** | Discovered from the API's node list; shown on the map with live values and data age; a **generic per-node detector** flags its unusual readings (below) | Including it as an input to the campus LSTMs (the 58-input shape is fixed) |
| **More solar on an existing plant** (capacity raised) | A **capacity registry** (kWp per node and date). PL00's forecast scales with capacity: PV output is close to linear in installed capacity, so scale the input by old/new capacity and the output back. Net-zero what-if uses the real capacity | A model trained in per-unit (kW/kWp) form handles it without the adapter |
| **A new solar plant** | Generic detector + capacity registry. Its output is added to "campus solar actual" in Net zero | Campus models that use it as an input |
| **A node dies or goes stale** | Already marked stale. Make the neighbour-copy rules configurable per node instead of hard-coded | – |
| **A node is replaced** (same place, new ID or units) | Alias in the registry (old ID → new ID, unit conversion) | – |

### 7.1 Generic per-node detector (no training, works for any node on day one)
* For each node and value, keep a rolling profile from its own last 2–4 weeks: median and MAD (median absolute deviation) per IST hour, split into weekday and weekend once there's enough data.
* Flag a reading when it's more than k·MAD from that hour's median, plus simple rules: stuck value (identical for > 1 h), dropout to 0 in daytime for solar, physically impossible values.
* Label it clearly as **"statistical"**, separate from the NPU forecasters' flags, and show it on the map and in the assistant.
* This is the right first answer to "a new node appears": the system watches it immediately. The NPU models only cover nodes they were trained on.

### 7.2 Registries (configuration, not code)
* `nodes.json`: ID, type, location, units, capacity (with valid-from dates), aliases, neighbour-copy rule, which models use it. Generated from the API's node list plus `campus.sqlite`, then edited by hand.
* The live loop and the assistant read it at start-up. Adding a node means a config change and a restart.

### 7.3 Retraining pipeline (when the inputs change)
* One script: config → dataset (existing cleaning rules) → train → AIMET W8A16 → QNN `.so` → thresholds → registry entry. Most of these steps already exist as separate scripts in `common/` and `aimet_quantization/scripts/`.
* The output is a new registry entry, so a retrained model appears in the switcher (§3) with no other change.

### 7.4 Suggested order
1. Generic per-node detector + node registry (~1.5 days). This gives the biggest gain: every reporting node is covered, and new nodes are covered immediately.
2. Capacity registry + scaling adapter for solar (~0.5 day).
3. Model switcher (§3–6).
4. Retraining pipeline as one script (~1–2 days), once the inputs actually change.
