# 06 — Quantization Strategy with AIMET AutoQuant

## 1. Order of models
1. `all_sensors_lstm_qidk.onnx` (1 h): best FP32 metrics, mid size. **Pilot.**
2. `all_sensors_24h_lstm_qidk.onnx`: 288 outputs and a long horizon. Expect it to be most sensitive.
3. `SL-PL00-00_lstm_qidk.onnx`: the only model that localizes an anomaly to one plant.
Same pipeline for each. Lessons from (1) set the defaults for (2) and (3).

## 2. What AutoQuant can and can't do for *this* architecture
| AutoQuant 2.12 stage | Effect on LSTM + MLP-head model |
|---|---|
| Quant-scheme selection (tf vs tf_enhanced, W & A) | **Useful.** Picks range-setting per tensor class |
| W32 early-exit check | **Useful diagnostic.** If W32/A_bw already misses the target, activations (not weights) are the problem → raise A bitwidth |
| BatchNorm folding | No-op (no BN) |
| Cross-Layer Equalization | Little to none (designed for Conv/ReLU/Conv chains; no Conv here) |
| AdaRound | Applies to the 2 `Gemm` weights (the head). **Doesn't touch LSTM weights** (≈95 % of params) |
| AMP (`AutoQuantWithAutoMixedPrecision`) | **Useful.** Lets sensitive quantizers (LSTM outputs, input, final Gemm) go to 16 bit while the rest stay 8 bit |

So for this model, accuracy is governed mainly by **activation bitwidth and calibration coverage**, with AutoQuant mostly doing systematic search and bookkeeping. That's still the right tool, but the experiment grid (§4) has to include bitwidth choices explicitly.

## 3. Defining "accuracy" for AutoQuant (`eval_callback`)

> **Revision 2026-10-05 (approximate-data approach, see doc 05):** the AutoQuant objective is **fidelity to FP32**, not R² vs labels:
> ```python
> def eval_callback(session, num_samples):
>     P = run(session, X_val[:n]);  R = fp32_outputs_val[:n]      # same windows
>     return sqnr_db(R, P)        # 10·log10(Σ R² / Σ (R−P)²), higher is better
> ```
> Budget: `allowed_accuracy_drop` is expressed in SQNR dB from the FP32 score (FP32 vs itself = +∞, so it's capped at 100 dB). In practice the target is **SQNR ≥ 30 dB** (≈3 % RMS relative error), stated directly. R²/MAE in Watts on approximate data are still reported as secondary metrics. The original R²-based design below is kept for reference.

AutoQuant maximizes a scalar where **higher is better**, with a drop budget relative to FP32.

```python
def eval_callback(session, num_samples):         # signature required by AIMET 2.12
    X, Y = val_windows[:num_samples or None]     # VAL split, labels in normalized units
    P = run(session, X)                          # [N, H*3]
    P, Y = to_watts(P), to_watts(Y)              # × capacities_W per domain
    return mean(R2(P_solar, Y_solar), R2(P_grid, Y_grid))   # Demand excluded (FP32 R² < 0)
```
* Exclude Demand R² (FP32 is −0.34) from the objective, but **report** its nMAE.
* Proposed budget: **`allowed_accuracy_drop = 0.01`** (≤ 0.01 absolute R²). Confirm before execution.
* Secondary metrics in every run: per-domain MAE/RMSE/nMAE in Watts, max abs error, SQNR of outputs vs FP32, and **anomaly-flag agreement** (doc 02 §4) between INT and FP32 at fixed thresholds.

## 4. Experiment grid (all on AIMET simulation, host CPU)
| Run | Weights | Activations | Config | Expected |
|---|---|---|---|---|
| E0 | FP32 | FP32 | n/a | Baseline. Must reproduce meta metrics (doc 05 anchor) |
| E1 | INT8 | INT8 | htp v75 | Likely noticeable drop (recurrent error accumulation) |
| E2 | **INT8** | **INT16** | htp v75 | **Expected sweet spot**, standard for LSTMs on HTP |
| E3 | INT8 per-channel | INT16 | htp v75 per-channel-linear | Small gain on Gemm |
| E4 | AutoQuant (best of E1–E3 as base) + `optimize(0.01)` | | | AdaRound on head; report applied techniques |
| E5 | AutoQuantWithAMP, candidates {W8A8, W8A16, W16A16} | | | Only if E4 misses the budget |

Per run, save: AutoQuant HTML diagnostics, `*.encodings`, the exported ONNX, and a metrics JSON. If a run misses the budget, run **`QuantAnalyzer`** (per-layer sensitivity + range histograms) to find the guilty tensor (likely the LSTM output `Y` / cell outputs or the [−1,1]/[0,1] mixed input) and pin it to 16 bit.

## 5. Recommended target
**W8A16 on HTP v75** unless E1 (W8A8) stays within budget. Rationale: HTP v75 runs 16-bit activations natively, the model is tiny (latency is dominated by the 96-step recurrence, not MAC width), and 16-bit activations protect the cell state over 96 steps. W8A8 only matters for a "smallest/fastest" headline, and the on-device benchmark (doc 08) will show whether it buys anything measurable.

## 6. Hand-off artefacts to QAIRT
From AIMET: `model.onnx` (AdaRound-adjusted weights if AdaRound won, otherwise the original weights) + `model.encodings` (JSON).
These go to `qairt-converter --quantization_overrides model.encodings`, then `qairt-quantizer` with calibration `.raw` inputs, which fills encodings for the LSTM-internal tensors AIMET doesn't simulate (doc 03 §3). Doc 07 has the exact flow. Doc 08 measures the sim-to-target gap that this split creates.
