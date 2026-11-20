# 10 — P0 Log: Environment Setup (executed 2026-10-05)

## 1. AIMET venv: done
```bash
uv python install 3.10                      # → CPython 3.10.19
uv venv --python 3.10 .venv-aimet
VIRTUAL_ENV=.venv-aimet uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "aimet-onnx==2.12.0" "onnxruntime==1.19.2" "onnx==1.16.2" "numpy<2" "protobuf>=3.20.2,<5" \
  torch pandas pyarrow scikit-learn matplotlib
```
Installed: aimet-onnx 2.12.0, onnx 1.16.2, onnxruntime 1.19.2, torch 2.14.1+cpu, numpy 1.26.4, protobuf 4.25.9.

**Finding:** in the 2.12.0 *wheel*, `from aimet_onnx.auto_quant_v2 import AutoQuant` raises
`NameError: 'AutoQuant' was deleted since aimet-onnx==2.3 … import from 'aimet_onnx._deprecated.auto_quant_v2'`.
The GitHub source file still exists at the 2.12 tag, but the public name was turned into a deprecation shim. The working import is:
```python
from aimet_onnx._deprecated.auto_quant_v2 import AutoQuant, AutoQuantWithAutoMixedPrecision
```
Same constructor as documented in doc 03 (`model, dummy_input, data_loader, eval_callback, param_bw, output_bw, quant_scheme, rounding_mode, use_cuda, device, config_file, results_dir, cache_id, strict_validation`).

## 2. AIMET smoke test: `scripts/00_smoke_test_aimet.py`
| Model | V1 pass-through max\|Δ\| | V1 with quantizers *disabled* | W8A16 sim max\|Δ\| vs FP32 (1 sample) |
|---|---|---|---|
| all_sensors_1h | 1.28e-4 | **1.2e-7** | 3.4e-3 |
| all_sensors_24h | 1.22e-4 | – | 6.3e-3 |
| single_node | 1.64e-4 | – | 5.2e-3 |

Reasoning:
* With every quantizer disabled, AIMET's rewritten graph is bit-identical to ORT (1.2e-7 = float rounding). **V1 passes.** The ~1e-4 in "float32" mode comes from AIMET's float quantize-dequantize emulation, not from a graph change.
* AIMET places weight quantizers on the raw initializers `lstm.weight_ih_l*`/`weight_hh_l*`. In the 1 h/24 h graphs those pass through Slice/Concat (gate reorder) before the LSTM. A per-tensor encoding is invariant to that permutation, so it maps 1:1 to the converted LSTM weights.

**Problem and fix: per-channel on LSTM.** The single-node model crashed in `enable_per_channel_quantization` (`channel_axis is None`). In that graph the PyTorch exporter **constant-folded** layer-0's `W` into an anonymous initializer `val_41` [1,256,10], and AIMET has no channel-axis rule for an LSTM `W` input. The HTP v75 config enables per-channel by default (Gemm is already explicitly per-tensor).
→ Created **`configs/htp_v75_lstm_per_tensor.json`** = stock `htp_quantsim_config_v75.json` + `"LSTM": {"per_channel_quantization": "False"}`.
Why this is the right choice beyond fixing the crash: `qairt-quantizer` quantizes LSTM weights **per-tensor by default**, so this config makes the AIMET simulation match what the HTP will actually run. This config is used for every model from here on.

## 3. Fixed QAIRT image: done
`docker/Dockerfile.qairt-fixed` (FROM `qairt:latest`, pins `protobuf==3.20.3`) → image **`qairt-fixed:2.47`**. The original image is untouched.
Check: `import onnx` → 1.16.1, and `qairt-converter` on the 1 h model → `CONVERT_OK`, run as the host user (`--user $(id -u):$(id -g)`), so no root-owned files.

## 4. Device runtime: done
Copied from the image into `runtime_2.47/` (123 MB) and pushed to **`/data/local/tmp/nz_aimet/`**:
`bin/{qnn-net-run, qnn-throughput-net-run, qnn-profile-viewer}`,
`lib/{libQnnHtp, libQnnHtpV75Stub, libQnnHtpV75Skel (hexagon-v75/unsigned), libQnnHtpPrepare, libQnnSystem, libQnnCpu, libQnnModelDlc, libQnnHtpNetRunExtensions, libQnnHtpProfilingReader}.so`.
Check: `qnn-net-run --version` on device → **QNN SDK v2.47.0.260601114230**. Existing folders (`qairt-2.37.*`, `campus_energy`) were not touched.

## 5. Exit gate
V1 ✅ · AutoQuant importable ✅ · converter ✅ · device runtime 2.47 ✅ → **P0 complete.**
