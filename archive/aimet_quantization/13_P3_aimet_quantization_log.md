# 13 — P3 Log: AIMET Quantization (executed 2026-10-05)

Scripts: `scripts/05_aimet_quantize.py` (grid + AutoQuant + export), `scripts/06_evaluate_sim.py` (TEST evaluation of the exports).
Config: `configs/htp_v75_lstm_per_tensor.json` (doc 10) and `configs/htp_v75_per_channel_gemm_lstm_per_tensor.json` (stock `htp_quantsim_config_v75_per_channel_linear.json` + LSTM per-tensor). Calibration: 1,024 stratified train windows. Selection: 600 evenly spaced validation windows. Metric: output SQNR vs FP32.

## 1. First grid: a surprise
| 1 h model (val) | SQNR | max\|Δ\| |
|---|---|---|
| W32A16 (activations only) | 43.8 dB | 0.037 |
| W8A32 (weights only) | 49.2 dB | 0.004 |
| W8A8 | 41.0 dB | 0.036 |
| W8A16 | 42.9 dB | 0.037 |

16-bit activations alone shouldn't lose 0.037 of capacity (~20 kW). **Diagnosis:** enable quantizers one at a time.
| Quantizer (16-bit) | max\|Δ\| alone |
|---|---|
| input, weights, biases, initial states, ReLU | ≤ 1e-5 |
| LSTM-1 / LSTM-2 hidden output (`val_65`, `val_128`) | 0.006 / 0.004 |
| **model output `forecast_all_pillars`** | **0.038** |

Cause: **range clipping, not resolution.** Calibration (train windows) saw outputs in [−0.022, 0.987], but validation forecasts reach −0.068 (negative night-time solar forecasts). Similarly, the LSTM hidden outputs were calibrated to ≈[−0.7, 0.7] while they can reach ±1.

## 2. Fix: range priors (domain knowledge, no retraining)
Applied after `compute_encodings`, then frozen:
* **LSTM hidden outputs → [−1, 1].** Mathematically guaranteed bounds: h = o·tanh(c), with o ∈ (0,1) and tanh ∈ (−1,1). The tensors are found by walking from each LSTM's `Y` output through layout-only ops.
* **Model outputs → calibrated range ± 10 % of its span.** Capacity fractions can over/undershoot. The headroom is cheap at 16 bit.
* AutoQuant had to be taught the same priors: `AutoQuantWithPriors` overrides `_create_quantsim_and_encodings`. Gotcha: QuantSim renames graph outputs, so tensor names must come from the **un-instrumented** FP32 graph.

## 3. Results on validation (SQNR vs FP32, dB)
| Config | 1 h | 24 h | single node |
|---|---|---|---|
| W32A16, no priors | 43.8 | 80.7 | 62.8 |
| **W32A16, priors** (activation ceiling) | **77.5** | 80.7 | 63.0 |
| W8A32 (weight-only) | 49.2 | 45.0 | 48.8 |
| W8A8, no priors | 41.0 | 41.4 | **44.0** |
| W8A8, priors | **42.8** | 40.9 | 42.2 |
| W8A16, no priors | 42.9 | 45.0 | 48.6 |
| W8A16, priors | 49.2 | 45.0 | 48.6 |
| W8A16, priors + per-channel Gemm | 52.2 | 50.8 | 52.6 |
| **AutoQuant W8A16** (priors, per-channel Gemm) | **58.5** | **57.2** | **53.7** |

Observations:
* With priors, 16-bit activations are nearly lossless (63–81 dB), so **8-bit weights become the bottleneck.** That's exactly what AdaRound addresses.
* Priors only matter where clipping existed (1 h). For the single node at **8-bit** activations, the 10 % output headroom costs one step of resolution (44.0 → 42.2 dB), a real trade-off. The W8A16 deliverable is unaffected.

## 4. AutoQuant run (aimet-onnx 2.12)
| Stage | 1 h | 24 h | single |
|---|---|---|---|
| Quant-scheme selection | tf (52.15) vs tf-enhanced (52.06) → **tf** | | |
| W32A16 check | 92.6 dB ≥ 60 target ✓ | | |
| BatchNorm folding | no-op (no BN) | no-op | no-op |
| Cross-layer equalization | no-op (52.15) | no-op (50.75) | no-op (52.57) |
| **AdaRound** (10,000 iters/layer, 1,024 windows) | **58.48** | **57.24** | **53.67** |

* Target = **60 dB**, deliberately above every plain 8-bit result and below the activation ceiling. AutoQuant therefore ran all stages and returned the best. An earlier "W32A16 − 1 dB" target either early-exited (before priors) or made AutoQuant abort at the W32 check.
* Bug worked around: 2.12's deprecated AutoQuant builds `AdaroundParameters` with `num_iterations=None` → `TypeError`. Fixed by `set_adaround_params(AdaroundParameters(loader, len(loader), default_num_iterations=10000))`.
* The 2.12 wheel exposes AutoQuant only as `aimet_onnx._deprecated.auto_quant_v2` (doc 10).
* As predicted in doc 06 §2: BN-fold and CLE do nothing for an LSTM + MLP head. **The gains come from range setting (priors), per-channel Gemm, and AdaRound.**

Exports: `out/<model>/final/model.{onnx,encodings}` (AutoQuant W8A16 deliverable, AdaRound-adjusted weights, encoding format 0.6.1) and `out/<model>/w8a8/` (comparison).

## 5. Held-out TEST evaluation (V2) — `06_evaluate_sim.py`
| Model | Variant | SQNR vs FP32 | max\|Δ\| (norm.) | Headline R² (FP32 → INT) | Natural-flag agreement | Injected-fault agreement |
|---|---|---|---|---|---|---|
| 1 h | **W8A16** | **57.9 dB** | 0.0017 | 0.8522 → 0.8526 | 99.78 % | 99.98 % |
| 1 h | W8A8 | 42.0 dB | 0.0122 | 0.8522 → 0.8543 | 99.64 % | 99.96 % |
| 24 h | **W8A16** | **58.7 dB** | 0.0029 | 0.3388 → 0.3388 | 99.94 % | 99.98 % |
| 24 h | W8A8 | 40.7 dB | 0.0180 | 0.3388 → 0.3359 | 99.59 % | 99.79 % |
| single | **W8A16** | **53.2 dB** | 0.0031 | 0.9499 → 0.9497 | 99.94 % | 99.94 % |
| single | W8A8 | 43.6 dB | 0.0112 | 0.9499 → 0.9498 | 99.88 % | 99.86 % |

* TEST SQNR ≈ validation SQNR, so no over-fitting to the selection subset.
* **W8A16:** max error ≤ 0.3 % of capacity, R² unchanged to the 4th decimal, ≥ 99.8 % identical anomaly decisions. **V2 passes** (target was ≥ 30 dB).
* **W8A8 also passes** in simulation (≥ 40 dB, |ΔR²| ≤ 0.003). Remember AIMET doesn't simulate LSTM internals (doc 03), so the on-device result (doc 14) decides whether W8A8 holds up on the HTP.

---
## 6. Revision after P4 findings (final configuration)
Two discoveries made while deploying (details in doc 14) changed the AIMET setup. **Sections 3–5 above are superseded by this section.**

1. **LSTM weights were not really 8-bit.** PyTorch's exporter reorders the LSTM gates with Slice/Concat, so the raw weight initializers reach the LSTM *through ops*, and AIMET placed **16-bit activation quantizers** on them. The earlier "W8A16" therefore had 16-bit LSTM weights (≈95 % of parameters). Fix: `lstm_weights_as_params()` walks back from every LSTM `W`/`R` input through the constant layout subgraph and sets those quantizers to **8-bit symmetric** (before calibration, and inside `AutoQuantWithPriors`). That makes the W8A16 label honest, and it matches what the HTP does (it quantizes the folded LSTM weights as 8-bit params).
2. **Per-channel Gemm weights mis-execute on the HTP** in our QNN path (14.6 dB on device vs 46.2 dB per-tensor, same model). AutoQuant now uses the **per-tensor** HTP v75 config, so AdaRound optimizes rounding for the scales actually deployed. The per-channel row is kept as simulation-only information.

### Final validation results (SQNR vs FP32, dB; 600 val windows)
| Config | 1 h | 24 h | single |
|---|---|---|---|
| W32A16 + priors (activation ceiling) | 77.5 | 80.7 | 63.0 |
| W8A8 + priors | 43.3 | 41.2 | 42.4 |
| W8A16 + priors (true 8-bit LSTM weights) | 47.6 | 43.3 | 49.9 |
| W8A16 + priors + per-channel Gemm (sim only) | 49.1 | 46.2 | 52.9 |
| **AutoQuant W8A16** (per-tensor, priors, AdaRound) | **50.3** | **47.1** | **51.1** |
| AdaRound gain over plain W8A16 | +2.7 | +3.8 | +1.2 |

(`W8A32` in the JSON only quantizes the Gemm weights, because the LSTM weight quantizers are activation-typed. Don't read it as "all weights at 8-bit".)

### Final held-out TEST results of the exported models (V2)
| Model | Variant | SQNR vs FP32 | max\|Δ\| | Headline R² FP32 → INT | Flag agreement natural / injected |
|---|---|---|---|---|---|
| 1 h | **W8A16** | **46.8 dB** | 0.0042 | 0.8522 → 0.8547 | 99.62 % / 99.98 % |
| 1 h | W8A8 | 41.4 dB | 0.0113 | 0.8522 → 0.8564 | 99.56 % / 99.94 % |
| 24 h | **W8A16** | **45.8 dB** | 0.0120 | 0.3388 → 0.3388 | 99.66 % / 99.92 % |
| 24 h | W8A8 | 40.6 dB | 0.0156 | 0.3388 → 0.3404 | 99.59 % / 99.66 % |
| single | **W8A16** | **51.8 dB** | 0.0054 | 0.9499 → 0.9499 | 99.92 % / 99.92 % |
| single | W8A8 | 43.9 dB | 0.0128 | 0.9499 → 0.9500 | 99.94 % / 99.92 % |

The 1 h model's TEST SQNR (46.8) is below its validation SQNR (50.3) because of the **Demand** outputs: their values are small in Nov–Dec, and SQNR is relative to signal energy (Solar columns ≈ 52–54 dB, Demand ≈ 37–44 dB). Re-loading the export reproduces validation exactly (51.38 → 51.38 dB in an intermediate run), so the export itself is lossless.
