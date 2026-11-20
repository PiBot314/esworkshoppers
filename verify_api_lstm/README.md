# Verification: existing LSTMs on the mentors' API

Evidence for `../archive/plan_reduced_shortterm.md` §2. Run from this folder (needs pandas, numpy, onnxruntime; e.g. `../../project/.venv`):

```bash
python fetch.py 2026-11-01 2026-11-16           # API -> raw/ (one JSON per node-day, ~18 MB)
python verify.py "$PWD/.." proxy                # build windows like training, run FP32 ONNX, score with 2025 thresholds
python verify.py "$PWD/.." zeros                # same, with unserved inputs filled with 0
python inject.py "$PWD/.." proxy                # flag clustering, injected-fault detection, threshold recalibration
```

`verify.py` writes `result_<mode>.json`; `inject.py` reads it. The NPU parity check (§2.5 of the plan) was run by hand with `qnn-net-run` on the board.
