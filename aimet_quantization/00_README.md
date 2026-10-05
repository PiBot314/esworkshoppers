# AIMET Quantization & QIDK Deployment: Planning Documents

Planning record for post-training-quantizing the existing (frozen, **not retrained**) LSTM ONNX models in `../model2/qidk_deployment/` with Qualcomm **AIMET AutoQuant**, and deploying them to the **QIDK (SM8650, HTP v75)**. Created 2026-10-05.

| # | Document | Contents |
|---|---|---|
| 01 | [Codebase overview](01_codebase_overview.md) | Layout, script data-flow, artefact inventory, **8 inconsistencies found** |
| 02 | [Model identification](02_model_identification.md) | 2-layer LSTM **forecaster** (not an anomaly detector); architecture from ONNX graphs; residual-based anomaly layer |
| 03 | [Tooling audit](03_tooling_audit.md) | What exists, what's missing, verified blockers B1–B4 |
| 04 | [Environment remediation](04_environment_remediation.md) | AIMET 2.12 uv venv, fixed QAIRT image, device runtime |
| 05 | [Data rebuild](05_calibration_and_eval_data_rebuild.md) | Reconstructing `lstm_normalized/`, acceptance anchors, calibration-set design |
| 06 | [Quantization strategy](06_quantization_strategy_autoquant.md) | What AutoQuant does for an LSTM, eval callback, experiment grid, W8A16 target |
| 07 | [Conversion & deployment](07_conversion_and_qidk_deployment.md) | AIMET → QAIRT → HTP context binary → `qnn-net-run` on QIDK |
| 08 | [Verification plan](08_verification_plan.md) | V0–V5 ladder, metrics, fault injection, on-device perf |
| 09 | [Action plan, risks, questions](09_action_plan_risks_next_steps.md) | Phases P0–P6, risk register, open questions (resolved by defaults, see banner) |
| 10 | [P0 log: environment](10_P0_environment_log.md) | AIMET venv, deprecated AutoQuant import, LSTM per-tensor config, fixed QAIRT image, device runtime |
| 11 | [P1 log: data rebuild](11_P1_data_rebuild_log.md) | Reverse-engineered preprocessing, anchors, windows, calibration set |
| 12 | [P2 log: FP32 baseline + anomaly layer](12_P2_fp32_baseline_and_anomaly_log.md) | V0 results, residual detector, fault injection |
| 13 | [P3 log: AIMET quantization](13_P3_aimet_quantization_log.md) | Clipping diagnosis, range priors, AutoQuant/AdaRound, **§6 final config** |
| 14 | [P4 log: QIDK deployment](14_P4_qidk_deployment_log.md) | Working conversion path, 7 failures and fixes, on-device results, latency |
| 15 | [Summary, lessons, next steps](15_summary_lessons_and_next_steps.md) | Deliverables, plan vs reality, limitations |
| 16 | [Dashboard plan](16_dashboard_plan.md) | Live demo GUI served by the QIDK: options, decisions, revised fast plan (§7) |
| 17 | [Dashboard build log](17_dashboard_build_log.md) | Data pack, device server, frontend, launcher, verification |
| – | [README.md](README.md) · [README_DETAILED.md](README_DETAILED.md) · [TECHNICAL_QA.md](TECHNICAL_QA.md) | Final concise / complete readmes, technical Q&A |

## Decisions already taken (2026-10-05)
* Target: the **mideval LSTM forecasters** (not the `../project` autoencoder).
* Calibration/eval data: **rebuild from raw** `../nodes/parquet_sorted`.
* AIMET environment: **host uv venv, Python 3.10, aimet-onnx 2.12.0** (the last release with ONNX AutoQuant).

## Key facts at a glance
* The QIDK is connected (`d42f06b`, SM8650, Android 14, **no Python on device**).
* `qairt:latest` (QAIRT 2.47) is present, but its **ONNX converter is broken** until protobuf is pinned to 3.20.3 (fix verified).
* AIMET is not installed. ONNX **AutoQuant only exists up to aimet-onnx 2.12.0** (cp310).
* `lstm_normalized/` (calibration + eval data) is **missing**.

## Outcome (2026-10-05)
All three models run on the QIDK NPU as AIMET-quantized **W8A16**: 48.4 / 46.4 / 52.1 dB fidelity to FP32, |ΔR²| ≤ 0.004, ≥ 99.66 % identical anomaly decisions, 3.4–3.9 ms per inference. See doc 14 and `release/`.

`../model2/` is unchanged. On the device, only `/data/local/tmp/nz_aimet/` was created.
