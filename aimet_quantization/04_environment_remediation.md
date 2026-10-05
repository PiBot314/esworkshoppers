# 04 — Environment Remediation (step-by-step)

None of these steps have been executed yet. They're listed in order, each with a check you can run afterwards.

## Step 4.1: AIMET environment on the host (fixes B2)
Decision: a host `uv` venv (Python 3.10). The QAIRT image stays untouched for AIMET purposes.

```bash
cd ~/Documents/Uni/Sem3/EmbeddedSystemWorkshop/mideval
uv python install 3.10
uv venv --python 3.10 .venv-aimet
source .venv-aimet/bin/activate
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "aimet-onnx==2.12.0" "onnxruntime==1.19.2" "onnx==1.16.2" "numpy<2" "protobuf>=3.20.2,<5" \
  torch pandas pyarrow scikit-learn matplotlib
```
Why these pins:
* `aimet-onnx==2.12.0`: the last release with `auto_quant_v2` (doc 03 §3), cp310 only.
* `onnxruntime==1.19.2`: the minimum AIMET 2.12 accepts. QuantSim adds custom ops to ORT, so the oldest supported version is the safest.
* `onnx==1.16.2`, `protobuf<5`: avoids the protobuf-gencode mismatch seen in the QAIRT image.
* `numpy<2`: required by aimet-onnx 2.12.
* CPU torch: AIMET imports torch utilities, and CPU is enough for ~140 k-param models. (GPU option: `aimet-onnx` GPU needs `onnxruntime-gpu` + CUDA 12. Not worth it here.)

**Check (smoke test, written as `scripts/00_smoke_test_aimet.py` in the execution phase):**
1. `import aimet_onnx; from aimet_onnx.auto_quant_v2 import AutoQuant` succeeds.
2. Load `all_sensors_lstm_qidk.onnx` and build a pass-through sim, `QuantizationSimModel(model, param_type="float32"/activation_type="float32", ...)` (2.12 API: `param_type`, `activation_type`, `quant_scheme`, `config_file`, `dummy_input`). Run the sample input. Output must match plain ORT (max |Δ| < 1e-5).
3. Build a W8A16 sim (`param_type=int8, activation_type=int16`) with the HTP v75 config (either the alias listed in `_config_file_aliases` or the path to `htp_quantsim_config_v75.json` inside the installed package). Call `sim.compute_encodings([{input_name: x}])` on the 2 sample windows and confirm the LSTM node gets quantizers on its inputs/weights/outputs.

If step 2 fails on opset 18 / dynamo-exported graph quirks (onnxsim already complained about the topological order of initializers), the fallback is to re-save the graph with `onnx.utils` topological sort and `onnxsim` **without changing weights**. That's a graph rewrite, not retraining. Verify parity.

## Step 4.2: Fix the QAIRT image (fixes B1)
Don't edit `qairt:latest`. Derive a new image so the original stays intact:

```bash
cd ~/Documents/Uni/Sem3/EmbeddedSystemWorkshop/mideval/qidk/Tools/qairt_docker
cat > Dockerfile.fix <<'EOF'
FROM qairt:latest
RUN . /usr/venv/bin/activate && pip install --no-cache-dir "protobuf==3.20.3"
EOF
docker build -f Dockerfile.fix -t qairt-fixed:2.47 .
```
**Check:**
```bash
docker run --rm --user $(id -u):$(id -g) -v "$PWD/../../..:/local:z" qairt-fixed:2.47 bash -c \
 'source /usr/venv/bin/activate && source $QAIRT_SDK_ROOT/bin/envsetup.sh >/dev/null &&
  python -c "import onnx;print(onnx.__version__)" &&
  qairt-converter --input_network /local/model2/qidk_deployment/all_sensors_lstm_qidk.onnx \
                  --output_path /tmp/t.dlc && echo CONVERT_OK'
```
(Already proven in a throwaway container: conversion succeeds after the protobuf pin.)

Then point `run_qairt_docker.sh` at it with `-i qairt-fixed:2.47`, or use `docker run` one-shots like the one above.

## Step 4.3: Device runtime (fixes part of B4)
Push the **2.47** HTP runtime into a fresh folder. Nothing existing gets overwritten.
```bash
SDK=/Tools/qairt_docker/qairt/current     # inside container; copy out to mideval/aimet_quantization/runtime_2.47/ first
D=/data/local/tmp/nz_aimet
adb -s d42f06b shell mkdir -p $D/lib $D/bin
adb push runtime_2.47/bin/aarch64-android/qnn-net-run             $D/bin/
adb push runtime_2.47/lib/aarch64-android/libQnnHtp.so            $D/lib/
adb push runtime_2.47/lib/aarch64-android/libQnnHtpV75Stub.so     $D/lib/
adb push runtime_2.47/lib/aarch64-android/libQnnHtpPrepare.so     $D/lib/
adb push runtime_2.47/lib/aarch64-android/libQnnSystem.so         $D/lib/
adb push runtime_2.47/lib/aarch64-android/libQnnCpu.so            $D/lib/
adb push runtime_2.47/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so $D/lib/
```
**Check:** `adb shell "cd $D && LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib ./bin/qnn-net-run --version"`, plus an FP32 run of the CPU backend on one sample (doc 07).

## Step 4.4: Hygiene
* Use the host `adb` only (gotcha G3).
* Always run containers with `--user $(id -u):$(id -g)` (gotcha G2).
* Free RAM before AIMET runs (≈12 GB was in use during the audit). AIMET plus 30 k windows of [96×58] float32 (≈730 MB) fits, but not alongside heavy browser/IDE use.
