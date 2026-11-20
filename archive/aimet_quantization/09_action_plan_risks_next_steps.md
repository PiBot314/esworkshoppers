# 09 — Action Plan, Risks, and Open Questions

> **Revision 2026-10-05.** Approximate data rebuild adopted (doc 05 banner). P1's exit gate is now "representative windows built + sanity deviations logged", and V0 becomes informational. Open questions (§3) were resolved by defaults, since the user asked to proceed:
> 1. Budget → **SQNR ≥ 30 dB vs FP32** + report R² drop on approximate data.
> 2. Bit-width → **W8A16 deliverable**. W8A8 also measured for comparison.
> 3. Rebuild fallback → **new baseline on rebuilt data** (option a).
> 4. Anomaly layer → **used as a verification metric** (flag agreement INT vs FP32), with a small host-side detector script.
> 5. Scope → **1 h pilot first, then the 24 h and single-node models**.
> 6. Device workdir → `/data/local/tmp/nz_aimet/` (new; nothing existing touched).

## 1. Sequenced action plan
Each phase ends with a numbered markdown log (`10_…`, `11_…`) recording commands run, outputs, and decisions.

| Phase | Work | Exit gate | Doc |
|---|---|---|---|
| **P0 Environment** | Create `.venv-aimet` (Py 3.10, aimet-onnx 2.12.0); build `qairt-fixed:2.47`; AIMET smoke test; converter smoke test | V1 passes; `CONVERT_OK` | 04 |
| **P1 Data rebuild** | Reconstruct `lstm_ready` → `lstm_normalized` from `../nodes`; build calib/val/test windows + `.raw` lists | Anchors in doc 05 §3, esp. **V0** (FP32 metrics reproduce) | 05 |
| **P2 Baseline + anomaly layer** | FP32 metrics per domain/horizon on rebuilt TEST; residual thresholds on VAL; fault-injection set | Baseline table written | 02 §4, 08 |
| **P3 AIMET (1 h pilot)** | E1–E3 sims → AutoQuant E4 (→ E5 AMP if needed) → QuantAnalyzer on misses | **V2** within 0.01 R² | 06 |
| **P4 QAIRT + device (1 h)** | convert with overrides → quantize → context binary → push 2.47 runtime → run CPU-FP32 / HTP-AIMET / HTP-QAIRT-only | **V3, V4, V5** | 07, 08 |
| **P5 Replicate** | Repeat P3–P4 for the 24 h and single-node models | same gates | 06–08 |
| **P6 Report** | Summary tables, plots, latency/size, honest limitations; new deploy script | Deliverables in doc 08 §4 | 08 |

Effort guess: P0 ≈ ½ day, **P1 ≈ 1–2 days (largest uncertainty)**, P2 ≈ ½ day, P3 ≈ 1 day, P4 ≈ 1 day, P5 ≈ 1 day, P6 ≈ ½ day.

## 2. Risk register
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Data rebuild can't reproduce training normalization exactly (KH95 schema, inverter voltage aggregation, two different builds behind 1 h vs 24 h metas) | **High** | Baseline metrics shift | Anchor-driven inference of mappings. Fallback: new FP32 baseline on rebuilt data (all INT-vs-FP32 comparisons remain valid), or fetch teammate's zip |
| AIMET 2.12 chokes on the dynamo-exported opset-18 graph | Medium | Blocks P3 | Topo-sort / onnxsim re-save without weight change; parity-check (V1) |
| **Sim-to-target gap** (AIMET doesn't quantize LSTM internals; HTP does) | **High** | AIMET metrics over-optimistic | Measure V4. Keep activations 16-bit. Calibrate QAIRT internals with the same windows. Per-layer debugger |
| W8A8 loses too much over 96 recurrent steps | High | Can't claim 8-bit | Default to W8A16 (doc 06 §5) |
| AdaRound/CLE give ≈0 benefit (head-only) | High | AutoQuant looks "useless" | Expected and explainable. The report credits scheme selection + AMP instead |
| Demand head already poor in FP32 (R² −0.34) | Certain | Confusing metrics | Excluded from objective, reported separately |
| Version mismatch device 2.37 vs host 2.47 | Certain if ignored | Load failures | Push 2.47 runtime to new dir |
| protobuf / root-owned files / adb server clashes | Medium | Friction | Gotchas G1–G3 (doc 03) |
| Host RAM pressure (~2 GB free at audit) | Medium | OOM during AIMET | Close apps. Batch windows lazily |

## 3. Open questions for you (please answer before P0/P1 execution)
1. **Accuracy budget:** is `allowed_accuracy_drop = 0.01` R² (Solar/Grid mean, VAL split) acceptable? Or do you have an evaluator-defined criterion?
2. **Bit-width headline:** is W8A16 acceptable as the deliverable, or is a strict "INT8" (W8A8) result required for the course?
3. **Fallback if the rebuild misses the anchors:** accept a new FP32 baseline on rebuilt data (option a), or stop and request `lstm_normalized.zip` (option b)?
4. **Anomaly layer:** should the residual-based anomaly detection (doc 02 §4) be implemented as part of this work, or only used as a verification metric?
5. **Scope of models:** all three ONNX files, or the 1 h pilot first and then decide?
6. **Device workdir:** OK to create `/data/local/tmp/nz_aimet/` on the QIDK (nothing existing touched)?

## 4. What happens next once answered
P0 starts. Scripts go under `aimet_quantization/scripts/` (numbered to match phases), outputs under `aimet_quantization/{data,out,results}/`, and every phase gets its own numbered log file continuing from `10_`.
