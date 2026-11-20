# Weather-aware forecasters on real 2025 data (B2–B4, C1, E1–E2)

Plan: [plan_19112026.md](../archive/plan_19112026.md) §3–4, [plan_20112026.md](../archive/plan_20112026.md). Data:
[datasets/training_2025/](../datasets/training_2025/) (real 2025 + weather + calendar; simulated 2015–2024 plants).
Worklog: [WORKLOG_2026-11-20.md](../archive/worklogs/WORKLOG_2026-11-20.md).

Two passes on 20 Nov:
* **First pass** (B2, B3/B4, C1): each experiment scored on its own test rows, no uncertainty, deployed models not
  compared.
* **Second pass** (E1, E2, grid v2, C1 calibration fix): every method scored on identical rows, with 95 % intervals,
  per-season/sky breakdowns, the deployed models included, and the LSTM rolling-origin run. **Where the passes
  disagree, the second pass is the one to quote.**

| File | What |
|---|---|
| [b2_lgbm_feature_test.py](b2_lgbm_feature_test.py) | B2: LightGBM, feature groups added one at a time, 1 h / 6 h / 24 h, blocked + rolling-origin → `results/b2_results.{md,json}`, `results/b2_importance_*.json` (the guided gates' prior) |
| [b3_guided_lstm.py](b3_guided_lstm.py) | B3/B4: shared guided LSTM with future covariates in the head; variants no-weather / forecast / pre-trained / unguided → `results/b3_*.json`, `results/b3_results.md`, `models/<family>_<h>_<variant>.onnx` + `.json`. Grid v2: day/week-ago covariates; `--skip --tag _skip` adds a learnable day/week-ago skip connection |
| [b3_rescore_baselines.py](b3_rescore_baselines.py) | Recomputes B3's baselines on the identical test windows (after a fix, see below) |
| [b3_pretrain_check.py](b3_pretrain_check.py) | Lighter pre-training recipe for solar 24 h, chosen on validation → `results/b3_pretrain_check.json` |
| [c1_fuse.py](c1_fuse.py) | C1: K models → one ONNX graph (block-diagonal LSTM), FP32-identical |
| [c1_emulator.py](c1_emulator.py) | C1: graph → QNN W8A16 per-tensor vs per-row → x86 HTP emulator → SQNR vs FP32 → `fused/*.emu.json`. Now takes the calibration size and split (`[N_TEST] [N_CAL] [CAL_SPLIT]`) and reports the per-window median SQNR too |
| [e1_compare.py](e1_compare.py) | **E1**: all methods (baselines, LSTM variants, LightGBM, the 6 deployed models) on identical test rows; day-block bootstrap 95 % CIs and paired differences; by season / sky / load regime → `results/e1_compare.{md,json}`, `results/e1_rows_*.parquet` |
| [e2_rolling.py](e2_rolling.py) | **E2**: rolling-origin (test month Mar–Dec) LSTM vs LightGBM on identical rows → `results/e2_rolling.{md,json}` |
| [b5_board_package.sh](b5_board_package.sh), [b5_score_board.py](b5_score_board.py) | B5 prep: a graph that passed in the emulator → aarch64-android HTP library + test windows + `run_on_board.sh` in `fused/<stem>_board/`; then board SQNR vs FP32 and vs the emulator |
| [b6_grid_lgbm_export.py](b6_grid_lgbm_export.py) | Grid LightGBM (E1 config) → `models/grid_<h>_lgbm.txt` + `.json` (features with their live source, test R², single-row timing) |
| [LIVE_INPUTS.md](LIVE_INPUTS.md) | What the new models need live vs what the live loop has (weather, calendar, clear-sky; Open-Meteo endpoints tested 20 Nov) |
| `models/v1_grid_noweek/`, `results/v1_grid_noweek/` | The first-pass grid models and results (before the day/week-ago covariates), kept for comparison |

## Model template (B3)

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
* ONNX inputs `past [B,96,F]`, `future [B,H,C]`, `node [B,n]`; batch dimension dynamic. Params: solar 24 h 123k,
  grid 24 h 222k.

## Second pass: the results to quote

### E1: identical rows, 95 % intervals ([results/e1_compare.md](results/e1_compare.md))

R² at the horizon on the 8 blocked test weeks; brackets = day-block bootstrap 95 % CI. Solar per kWp on ok days
(daytime); grid = SB00.

| | Best baseline | Best deployed model | LightGBM forecast | LSTM forecast | LSTM no weather |
|---|---|---|---|---|---|
| Solar 1 h | 0.772 (CSI pers.) | – (see campus total) | 0.846 [0.81, 0.88] | 0.841 [0.80, 0.88] | 0.847 [0.80, 0.89] |
| Solar 24 h | 0.679 | – | 0.818 [0.77, 0.86] | 0.813 [0.76, 0.86] | 0.804 [0.75, 0.85] |
| Grid 1 h | 0.866 (persistence) | 0.862 (all_sensors_1h, live) | **0.931** [0.91, 0.94] | 0.914 [0.89, 0.93] | 0.903 |
| Grid 24 h | 0.654 (last week) | 0.660 (guided_24h) | **0.873** [0.83, 0.90] | 0.774 (v2) / 0.799 (v1) | 0.713 (v2) / 0.797 (v1) |
| Campus solar total 1 h (kW) | 0.722 | 0.748 (all_sensors_1h) | – | **0.786** [0.73, 0.83] | 0.783 |
| Campus solar total 24 h (kW) | 0.432 | 0.677 (guided_24h) | – | **0.724** [0.64, 0.79] | 0.701 |

Paired differences that the conclusions rest on (A − B in R², 95 % CI; *clean* = only the 3 test weeks the deployed
models' weights never saw: 8–15 Oct, 29 Oct–5 Nov, 24–31 Dec):

| Question | Solar 1 h | Solar 24 h | Grid 1 h | Grid 24 h |
|---|---|---|---|---|
| New vs deployed (best new − best deployed) | campus total: LSTM +0.038 [+0.007, +0.065] | campus total: LSTM +0.047 [−0.003, +0.094] | LightGBM +0.069 [+0.050, +0.098]; clean +0.071 | LightGBM +0.213 [+0.140, +0.313]; clean +0.141 [+0.063, +0.273] |
| Weather (LightGBM forecast − no weather) | +0.004 [−0.007, +0.015] | **+0.055 [+0.016, +0.107]** | +0.003 [+0.001, +0.006] | **+0.029 [+0.012, +0.052]** |
| Weather (LSTM forecast − no weather) | −0.006 [−0.017, +0.004] | +0.009 [−0.007, +0.024] | +0.011 [+0.004, +0.021] | +0.062 [+0.019, +0.119] (v2) |
| LSTM − LightGBM (both forecast) | −0.005 [−0.012, +0.003] | −0.006 [−0.028, +0.013] | **−0.017 [−0.025, −0.011]** | **−0.099 [−0.183, −0.040]** |
| Guided − unguided gate | +0.010 [−0.002, +0.022] (pre-trained pair) | +0.001 [−0.015, +0.022] | **+0.019 [+0.007, +0.034]** | +0.005 [−0.029, +0.048] |
| Pre-training on simulated years (forecast − forecast+pre) | +0.005 [−0.003, +0.013] | +0.015 [−0.005, +0.036] | – | – |

What this says:

* **The new models beat what is deployed**, clearly so for grid. The live 1 h model is no better than persistence on
  grid (0.862 vs 0.866, difference −0.004 [−0.040, +0.024]). The deployed solar models lose to clear-sky-index
  persistence in winter.
* **Weather forecasts matter at 24 h, not at 1 h.** The ERA5 "perfect weather" runs in B2 show that at 1 h there was
  almost no room for weather to help (best live-usable within 0.002 of perfect weather). At 24 h the solar gain comes
  from clear days (LightGBM skill vs baseline on clear target days +0.52 with forecasts vs +0.22 without; on cloudy
  days +0.41 vs +0.22).
* **LightGBM is the better grid forecaster** at both horizons (significant), and ties the LSTM on solar.
* **Guided gate:** helps grid 1 h (+0.019, significant); not distinguishable elsewhere. The first pass's "guided ≥
  unguided in all 4" is true as point estimates but 3 of 4 are within noise.
* **Pre-training on the simulated years:** no benefit (both point estimates negative, CIs include 0).
* Intervals are wide (≈ ±0.04 R²): 44–55 test days. Single-number differences below ~0.02 should not be quoted as
  findings.

### Grid v2: day/week-ago covariates (hypothesis rejected)

The grid LSTM's 96-step window can't see the same time last week, LightGBM's top grid feature (37 % of gain at 24 h).
v2 feeds the load one day and one week before each target step to the head (known at the origin). Result on identical
rows: grid 1 h +0.006 [−0.000, +0.013] (kept); **grid 24 h −0.024 [−0.094, +0.030]**, validation loss better
(0.00275 vs 0.00289) but test not; a skip connection (forecast = a·day-ago + b·week-ago + LSTM correction) didn't help
either (−0.009). So the missing weekly lag is **not** why the LSTM trails LightGBM at 24 h. E2 points at data size
instead (below). The v1 grid models are kept in `models/v1_grid_noweek/`.

### E2: rolling-origin ([results/e2_rolling.md](results/e2_rolling.md))

Test month m = Mar–Dec, train on everything up to 16 days before it (pooled over folds, R² [95 % CI]):

| | Best baseline | LightGBM forecast | LSTM forecast | LSTM − LightGBM | Weather gain LightGBM / LSTM |
|---|---|---|---|---|---|
| Solar 1 h | 0.744 (CSI pers.) | 0.802 [0.784, 0.820] | 0.783 [0.761, 0.802] | **−0.019 [−0.030, −0.009]** | +0.009 / +0.013 (both significant) |
| Solar 24 h | 0.598 | 0.744 [0.719, 0.768] | 0.751 [0.725, 0.774] | +0.007 [−0.007, +0.020] | **+0.062 / +0.048** |
| Grid 1 h | 0.845 (persistence) | 0.888 [0.877, 0.898] | 0.784 [0.749, 0.815] | **−0.104 [−0.135, −0.077]** | +0.001 / −0.004 |
| Grid 24 h | 0.626 (last week) | 0.713 [0.671, 0.748] | 0.705 [0.675, 0.731] | −0.008 [−0.041, +0.027] | +0.017 / +0.004 |

* With 1–3 months of history the grid 1 h LSTM is far behind (Mar 0.45, Apr 0.54 vs LightGBM 0.75, 0.83) and it
  catches up only from July (0.86–0.89 vs 0.87–0.90). It is data-hungry; LightGBM is not.
* At 24 h both lose ~0.08–0.16 vs the blocked split: unseen seasons cost the most at long horizons.
* Solar: LightGBM is ahead at 1 h (−0.019, significant; the blocked tie becomes a small win on unseen months), tied at
  24 h. Weather forecasts matter **more** on unseen months (solar 24 h +0.05–0.06 vs +0.01–0.06 blocked).
* Overall, 8 comparisons (blocked + rolling × 4): LightGBM significantly better in 4, tied in 4, never worse.

### Blending LightGBM and LSTM (on the E1/E2 rows)

Averaging the two forecasts helps only where they are about equally good and their errors differ: grid 24 h
rolling-origin +0.034 [+0.017, +0.052] (50/50) and solar 1 h rolling-origin +0.006 [+0.003, +0.008] (70/30 LightGBM).
Elsewhere it is within noise or worse (grid 1 h rolling 50/50 −0.022). Solar errors are 0.87–0.96 correlated between
the two models; grid 0.67–0.82.

## First pass (kept for the record)

| Family, horizon | Best baseline | LightGBM (B2), best live-usable | LSTM forecast (B3) | Weather helps? |
|---|---|---|---|---|
| Solar 1 h | CSI persistence 0.77 | 0.846 | 0.842 | barely (+0.004 LightGBM; LSTM no-weather 0.844) |
| Solar 24 h | same time yesterday 0.68 | 0.815 | 0.814 | yes, +0.05 (LightGBM) / +0.016 (LSTM) |
| Grid 1 h | persistence 0.87 | 0.930 | 0.911 | slightly |
| Grid 24 h | same time yesterday 0.45–0.49 | 0.870 | 0.804 | yes (+0.02); calendar +0.04 |

B2 and B3 were scored on different rows (B2: every origin with the target in a test week; B3: windows whose origin and
target are in the same test week), so their comparison was only good to ~0.01. E1 replaces it; the numbers agree to
within 0.005 for LightGBM and solar. The older models' gates in plan_19112026 (24 h solar "above 0.915") were on the
campus total including nights and on a chronological split, so they aren't comparable with either pass.

## C1: fusion and quantization (x86 HTP emulator, W8A16, SQNR vs FP32)

**Solar** (first pass, unchanged): solar 1 h + 24 h fused, per-row: **52.9 / 46.5 dB** (alone 49.4 / 46.4;
per-tensor 43.5 / 42.0 fails the 44 dB gate). Fusion costs nothing and per-row is what passes.

**Grid** (first pass: 34.5 / 37.2 dB alone, 24–30 dB fused, "needs the October recipe"). **Second pass, cause found and
fixed:**

* The error was not spread out: in grid 1 h the 5 worst of 50 test windows carried 98 % of it (median window
  52.6 dB). The worst ones held a holiday or an 87-bin data gap, and **none of the 100 calibration windows (val split)
  held a holiday**, so activations outside the calibrated range were clipped.
* Calibrating on 500 windows from train + val instead:

| Graph (v2 grid models) | Calibration | per-tensor | per-row | per-row, median window |
|---|---|---|---|---|
| grid 1 h | 100 val windows | 33.8 | 34.7 | 52.8 |
| grid 1 h | **500 train + val** | 46.6 | **51.3** | 54.8 |
| grid 24 h | 100 val windows | 37.1 | 37.6 | 48.5 |
| grid 24 h | **500 train + val** | 43.8 | **48.6** | 48.4 |

* So both grid models pass the 44 dB gate (per-row) with a calibration set that covers holidays and gaps; no AdaRound needed
  for that.
* **Grid still can't join the solar graph.** All 4 fused (v2 models, 500 train + val calibration), per-row: solar 1 h
  40.7, solar 24 h 49.3, **grid 1 h 19.0, grid 24 h 26.0 dB** (per-tensor 36.9 / 42.7 / 19.3 / 26.1). Here the error is
  in every window (grid median per window 19–28 dB), not a few clipped ones. Per-row quantization gives each model its
  own *weight* scales, but *activations* (the shared 256-wide gate and hidden tensors) still have one scale per tensor,
  so grid's activations are squeezed by sharing tensors with solar. **Rule confirmed: fuse within a family.**
  (Solar's 40.7 here is the calibration, not grid: see the next bullet. Solar's per-window median is not meaningful:
  night windows are all 0.)
* **Calibration is per family.** The wide calibration that fixed grid makes solar *worse*: solar 1 h + 24 h fused,
  500 train + val windows: per-row **40.8 / 49.5 dB** vs **52.9 / 46.5** with 100 val windows (solar 1 h fails 44 with
  the wide set). Use 100 val windows for solar, 500 train + val for grid.
* **Board packages** ([b5_board_package.sh](b5_board_package.sh)): `fused/solar_1h24h_board/` (100 val windows,
  52.9 / 46.5 dB) and `fused/grid_1h_v2_cal500_board/` (51.3 dB). `fused/solar_1h24h_cal500_board/` was built first
  by mistake with the wide calibration; it is marked NOT_FOR_DEPLOYMENT.
* **Correction:** the first pass said QNN's ONNX simplifier "segfaulted on the grid graphs only". It failed on **all**
  graphs (solar too): `onnxsim` can't import with the container's numpy (`_ARRAY_API not found`), and conversion
  continued unsimplified every time. It does not explain the grid result.

Timing (one fused call vs separate calls; batching over plants, C2) still needs the board.

## Fixed during the run (20 Nov)

* Solar output ReLU during training killed whole output units (step 1 predicted 0) → linear while training, ReLU in use.
* B2 and B3 scored baselines on a filled 0 where the origin or day-ago value was missing (grid 24 h same time yesterday
  came out −0.40 instead of +0.49). B2: every method now scored on rows where both exist; B3: missing day-ago value
  falls back to persistence; re-scored without retraining.
* B3 "skill, all steps" first used the baseline that was best at the last step; now the best per column.
* B3 had no same-time-last-week baseline for grid (it is the best grid 24 h baseline, 0.654); added.
* B2 vs B3 on different rows, no intervals, deployed models never scored on 2025 test weeks, no LSTM rolling-origin →
  E1 / E2.
* Emulator calibration set too narrow for grid → `c1_emulator.py` takes size and split.
