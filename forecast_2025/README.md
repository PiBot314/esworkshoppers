# Forecasters on real 2025 data

LightGBM and a guided LSTM forecasting solar (6 plants, per kWp) and grid draw (substation SB00) 1 h and 24 h ahead,
from real 2025 meters with weather forecasts, clear-sky irradiance and the academic calendar; plus fusing the models
for the NPU and checking them in the x86 emulator. Run on 20 Nov.

Data: [datasets/training_2025/](../datasets/training_2025/). Plans: [plan_19112026.md](../archive/plan_19112026.md)
§3–4, [plan_20112026.md](../archive/plan_20112026.md). Step by step:
[WORKLOG_2026-11-20.md](../archive/worklogs/WORKLOG_2026-11-20.md). Write-up:
[WRITEUP_2026-11-20.md](../reports/WRITEUP_2026-11-20.md).

| File | What |
|---|---|
| [b2_lgbm_feature_test.py](b2_lgbm_feature_test.py) | B2: LightGBM, feature groups added one at a time, 1 h / 6 h / 24 h, blocked + rolling-origin → `results/b2_results.{md,json}`, `results/b2_importance_*.json` (the guided gates' prior) |
| [b3_guided_lstm.py](b3_guided_lstm.py) | B3/B4: shared guided LSTM with future covariates in the head; variants no-weather / forecast / pre-trained / unguided → `results/b3_*.json`, `results/b3_results.md`, `models/<family>_<h>_<variant>.onnx` + `.json`. Grid v2: day/week-ago covariates; `--skip --tag _skip` adds a learnable day/week-ago skip connection |
| [b3_rescore_baselines.py](b3_rescore_baselines.py) | Recomputes B3's baselines on the identical test windows (after a fix, see below) |
| [b3_pretrain_check.py](b3_pretrain_check.py) | Lighter pre-training recipe for solar 24 h, chosen on validation → `results/b3_pretrain_check.json` |
| [c1_fuse.py](c1_fuse.py) | C1: K models → one ONNX graph (block-diagonal LSTM), FP32-identical |
| [c1_emulator.py](c1_emulator.py) | C1: graph → QNN W8A16 per-tensor vs per-row → x86 HTP emulator → SQNR vs FP32 (overall and per-window median) → `fused/*.emu.json`. Arguments `[N_TEST] [N_CAL] [CAL_SPLIT]`: use 100 `val` for solar, 500 `train+val` for grid |
| [e1_compare.py](e1_compare.py) | **E1**: all methods (baselines, LSTM variants, LightGBM, the 6 deployed models) on identical test rows; day-block bootstrap 95 % CIs and paired differences; by season / sky / load regime → `results/e1_compare.{md,json}`, `results/e1_rows_*.parquet` |
| [e2_rolling.py](e2_rolling.py) | **E2**: rolling-origin (test month Mar–Dec) LSTM vs LightGBM on identical rows → `results/e2_rolling.{md,json}` |
| [b5_board_package.sh](b5_board_package.sh), [b5_score_board.py](b5_score_board.py) | B5 prep: a graph that passed in the emulator → aarch64-android HTP library + test windows + `run_on_board.sh` in `fused/<stem>_board/`; then board SQNR vs FP32 and vs the emulator |
| [b6_grid_lgbm_export.py](b6_grid_lgbm_export.py) | Grid LightGBM (E1 config) → `models/grid_<h>_lgbm.txt` + `.json` (features with their live source, test R², single-row timing) |
| [LIVE_INPUTS.md](LIVE_INPUTS.md) | What the new models need live vs what the live loop has (weather, calendar, clear-sky; Open-Meteo endpoints tested 20 Nov) |
| `models/v1_grid_noweek/`, `results/v1_grid_noweek/` | The first grid models and results (before the day/week-ago inputs), kept for comparison |
| `fused/` | Fused / single-model graphs (`.onnx`, `.json`), emulator results (`.emu.json`, `*_emu/out_*`) and board packages (`*_board/`). Build intermediates: `archive/forecast_2025_emulator_work/` |
| `results_*.log` | Console logs of each run (not in git) |

## Model template (LSTM)

* Past window 96 × F per node. Solar (F = 8): output per kWp, missing flag, clear-sky, recent irradiance (15-min
  forecast as the board sees it), hour/day-of-year sin/cos. Grid (F = 15): adds temperature, day of week and the four
  calendar flags.
* 2-layer LSTM, hidden 64 (the NPU body). Head: [last hidden state | future covariates for the H target steps | node
  one-hot] → 128 → H. Future covariates: clear-sky, forecast GHI, cloud, temperature (grid: + humidity, time, calendar,
  and since v2 the load one day and one week before each target step); 1 h uses the 15-min historical forecast, 24 h
  the forecast issued a day before.
* Guided gate: one multiplier per input, initialised from B2's LightGBM gain shares (damped as in
  `common/train_fair_lstm.py`), L2-anchored, folded into `weight_ih_l0` / `fc1` at export.
* Solar: one model for all 6 plants (plants as the batch), trained on ok days; output × (clear-sky > 0).
* ONNX inputs `past [B,96,F]`, `future [B,H,C]`, `node [B,n]` (C = 4 solar, 14 grid); batch dimension dynamic.
  Params: solar 1 h 64k, 24 h 123k; grid 1 h 70k, 24 h 247k (v1 grid without the day/week-ago inputs: 222k).

## Results

**How everything is scored.** Every method on identical rows: the value at the horizon (+1 h or +24 h); solar per
kWp on daytime targets of ok days, grid in MW. Two test designs: **blocked** (8 test weeks spread over 2025, so the
model has seen every season) and **rolling-origin** (each month Mar–Dec predicted by a model trained only on data up
to 16 days before it). Brackets are 95 % intervals from resampling test days. With 44–55 test days per comparison,
differences below ~0.02 R² are not findings. Scripts: [e1_compare.py](e1_compare.py) (blocked, plus the deployed
models) → [results/e1_compare.md](results/e1_compare.md); [e2_rolling.py](e2_rolling.py) →
[results/e2_rolling.md](results/e2_rolling.md).

### Accuracy (R², blocked / rolling-origin)

| | Best baseline | LightGBM | LSTM | LSTM without weather |
|---|---|---|---|---|
| Solar 1 h | 0.772 / 0.744 (clear-sky-index persistence) | 0.846 [0.81, 0.88] / **0.802** [0.78, 0.82] | 0.841 [0.80, 0.88] / 0.783 [0.76, 0.80] | 0.847 / 0.770 |
| Solar 24 h | 0.679 / 0.598 | 0.818 [0.77, 0.86] / 0.744 [0.72, 0.77] | 0.813 [0.76, 0.86] / 0.751 [0.73, 0.77] | 0.804 / 0.702 |
| Grid 1 h | 0.866 / 0.845 (persistence) | **0.931** [0.91, 0.94] / **0.888** [0.88, 0.90] | 0.914 [0.89, 0.93] / 0.784 [0.75, 0.81] | 0.903 / 0.788 |
| Grid 24 h | 0.654 / 0.626 (same time last week) | **0.873** [0.83, 0.90] / 0.713 [0.67, 0.75] | 0.774 [0.68, 0.84] / 0.705 [0.68, 0.73] | 0.713 / 0.701 |

### Against the deployed models (blocked)

The six registered models, run on the same weeks from the dataset they were trained on. Solar is compared on the
campus total (what they forecast). *Clean* = only the 3 test weeks their weights never saw (8–15 Oct, 29 Oct–5 Nov,
24–31 Dec).

| | Best baseline | Best deployed | Best new | New − deployed |
|---|---|---|---|---|
| Grid 1 h | 0.866 | 0.862 (`all_sensors_1h`, live) | 0.931 (LightGBM) | +0.069 [+0.050, +0.098]; clean +0.071 |
| Grid 24 h | 0.654 | 0.660 (`guided_24h`) | 0.873 (LightGBM) | +0.213 [+0.140, +0.313]; clean +0.141 [+0.063, +0.273] |
| Campus solar 1 h (kW) | 0.722 | 0.748 (`all_sensors_1h`) | 0.786 (LSTM) | +0.038 [+0.007, +0.065] |
| Campus solar 24 h (kW) | 0.432 | 0.677 (`guided_24h`) | 0.724 (LSTM) | +0.047 [−0.003, +0.094] |

The live 1 h grid model is no better than persistence (−0.004 [−0.040, +0.024]). The deployed solar models lose to
clear-sky-index persistence in winter.

### What the comparisons answer (A − B in R², blocked unless marked)

| Question | Solar 1 h | Solar 24 h | Grid 1 h | Grid 24 h |
|---|---|---|---|---|
| Weather: LightGBM with − without | +0.004 [−0.007, +0.015] | **+0.055 [+0.016, +0.107]** | +0.003 [+0.001, +0.006] | **+0.029 [+0.012, +0.052]** |
| Weather: LSTM with − without | −0.006 [−0.017, +0.004] | +0.009 [−0.007, +0.024] | +0.011 [+0.004, +0.021] | +0.062 [+0.019, +0.119] |
| Weather, rolling-origin: LightGBM / LSTM | +0.009 / +0.013 | **+0.062 / +0.048** | +0.001 / −0.004 | +0.017 / +0.004 |
| LSTM − LightGBM | −0.005 [−0.012, +0.003] | −0.006 [−0.028, +0.013] | **−0.017 [−0.025, −0.011]** | **−0.099 [−0.183, −0.040]** |
| LSTM − LightGBM, rolling-origin | **−0.019 [−0.030, −0.009]** | +0.007 [−0.007, +0.020] | **−0.104 [−0.135, −0.077]** | −0.008 [−0.041, +0.027] |
| Guided − unguided gate | +0.010 [−0.002, +0.022] (pre-trained pair) | +0.001 [−0.015, +0.022] | **+0.019 [+0.007, +0.034]** | +0.005 [−0.029, +0.048] |
| Without − with pre-training on simulated years | +0.005 [−0.003, +0.013] | +0.015 [−0.005, +0.036] | – | – |

* **Weather helps a day ahead, not an hour ahead.** At 1 h there was no room: in B2, perfect (observed) weather
  improves solar 1 h by only 0.006 over no weather. At 24 h the solar gain is on clear days (LightGBM skill over the
  baseline on clear target days +0.52 with forecasts vs +0.22 without; cloudy +0.41 vs +0.22). It matters more on
  unseen months.
* **LightGBM is the most accurate:** significantly better than the LSTM in 4 of the 8 comparisons, tied in 4, never
  worse. The LSTM needs data: on grid 1 h with 1–3 months of history it scores 0.45–0.54 (LightGBM 0.75–0.83) and
  catches up only from July.
* **Unseen seasons cost 0.04–0.16 R²**, most at 24 h.
* **Guided gate:** helps grid 1 h only. **Pre-training** on the simulated 2015–2024 plants: no gain (dropped).

### Grid LSTM v2: day- and week-ago inputs (hypothesis rejected)

The grid LSTM's 96-step window can't see the same time last week, LightGBM's top grid feature (37 % of gain at
24 h). v2 feeds the load one day and one week before each target step to the head. Grid 1 h +0.006 [−0.000, +0.013]
(kept); **grid 24 h −0.024 [−0.094, +0.030]** (validation loss better, test not); a skip connection
(a·day-ago + b·week-ago + LSTM correction) didn't help either (−0.009). So the missing weekly lag is not why the LSTM
trails LightGBM; data size is (rolling-origin above). The v1 grid models are kept in `models/v1_grid_noweek/`.

### Blending LightGBM and LSTM

Averaging the two helps only where they are about equally good and their errors differ: grid 24 h rolling-origin
+0.034 [+0.017, +0.052] (50/50), solar 1 h rolling-origin +0.006 [+0.003, +0.008] (70 % LightGBM). Elsewhere it is
within noise or worse (grid 1 h rolling-origin 50/50 −0.022). Solar errors correlate 0.87–0.96 between the two models,
grid 0.67–0.82.

## NPU: fusion and quantization (x86 HTP emulator, W8A16, SQNR vs FP32)

Target: ≥ 44 dB per model. All numbers are from the emulator, which matched the board within 2 dB on 19 Nov; none of
these graphs has run on the board yet. Results: `fused/<graph>.emu.json`.

| Graph | Calibration | Per-tensor | **Per-row** | Per-row, median window |
|---|---|---|---|---|
| Solar 1 h + 24 h fused | 100 val windows | 43.5 / 42.0 | **52.9 / 46.5** | – |
| Solar 1 h / 24 h alone | 100 val windows | 43.4 / 42.0 | 49.4 / 46.4 | – |
| Solar 1 h + 24 h fused | 500 train + val | 37.5 / 43.0 | 40.8 / 49.5 | – |
| Grid 1 h alone | 100 val windows | 33.8 | 34.7 | 52.8 |
| Grid 1 h alone | 500 train + val | 46.6 | **51.3** | 54.8 |
| Grid 24 h alone | 100 val windows | 37.1 | 37.6 | 48.5 |
| Grid 24 h alone | 500 train + val | 43.8 | **48.6** | 48.4 |
| All 4 fused (solar 1 h / 24 h / grid 1 h / 24 h) | 500 train + val | 36.9 / 42.7 / 19.3 / 26.1 | 40.7 / 49.3 / 19.0 / 26.0 | – |

(Solar's per-window median isn't meaningful: night windows are all 0.)

* **Per-row weight quantization is what passes** (`--use_per_row_quantization --enable_per_row_quantized_bias`);
  per-tensor fails solar 24 h.
* **Fuse within a family.** Solar 1 h + 24 h in one graph (one LSTM with block-diagonal weights, [c1_fuse.py](c1_fuse.py))
  costs nothing. Grid with solar fails (19–26 dB, error in every window): per-row scales cover the weights, but the
  shared activation tensors keep one scale each, and the families' activation ranges differ.
* **Calibrate per family.** Grid's low first scores came from 5 of 50 test windows (98 % of the error), holding a
  holiday or a long data gap; none of the 100 val calibration windows held a holiday, so those activations were
  clipped. 500 train + val windows fix grid without AdaRound, but make solar worse. Use 100 val for solar, 500
  train + val for grid.
* **Board packages** ([b5_board_package.sh](b5_board_package.sh); score with [b5_score_board.py](b5_score_board.py)):
  `fused/solar_1h24h_board/` (52.9 / 46.5 dB) and `fused/grid_1h_v2_cal500_board/` (51.3 dB): an aarch64-android HTP
  library, 50 test windows and `run_on_board.sh` each. `fused/solar_1h24h_cal500_board/` was built with the wrong
  calibration and is marked NOT_FOR_DEPLOYMENT.
* **QNN's ONNX simplifier fails on every graph** (`onnxsim` can't import with the container's numpy, `_ARRAY_API not
  found`); conversion continues unsimplified. It isn't related to the grid results.
* **Still needs the board:** device SQNR, and timing of one fused call vs separate calls and of batching over plants.

## Fixed during the run (20 Nov)

* Solar output ReLU during training killed whole output units (step 1 predicted 0) → linear while training, ReLU in use.
* B2 and B3 scored baselines on a filled 0 where the origin or day-ago value was missing (grid 24 h same time yesterday
  came out −0.40 instead of +0.49). B2: every method now scored on rows where both exist; B3: missing day-ago value
  falls back to persistence; re-scored without retraining.
* B3 "skill, all steps" first used the baseline that was best at the last step; now the best per column.
* B3 had no same-time-last-week baseline for grid (it is the best grid 24 h baseline, 0.654); added.
* B2 and B3 on different rows, no intervals, deployed models never scored on the 2025 test weeks, no LSTM
  rolling-origin → E1 / E2 (the Results above).
* Emulator calibration set too narrow for grid → `c1_emulator.py` takes the calibration size and split.


## Appendix: the first pass (20 Nov morning)

B2 ([b2_lgbm_feature_test.py](b2_lgbm_feature_test.py), [results/b2_results.md](results/b2_results.md)) and B3
([b3_guided_lstm.py](b3_guided_lstm.py), [results/b3_results.md](results/b3_results.md)) were each scored on their
own test rows, without intervals, and without the deployed models. B2 is still the source for the feature-group
results (calendar +0.03–0.04 for grid at 6–24 h; forecast inputs within 0.01–0.02 of perfect weather; the 6 h horizon)
and for the gate priors. Where its numbers differ from the Results above, quote the Results. The older gates in
[plan_19112026.md](../archive/plan_19112026.md) (e.g. 24 h solar "above 0.915") were on the campus total including
nights and a chronological split: not comparable with either.
