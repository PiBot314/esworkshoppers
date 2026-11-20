# 07 — Conversion (AIMET → QAIRT) and Deployment to the QIDK

Target: SM8650 / HTP **v75**, QAIRT **2.47** on both host and device. Device workdir: `/data/local/tmp/nz_aimet` (new; nothing existing is modified).
All QAIRT commands run in `qairt-fixed:2.47` (doc 04 §4.2) as `--user $(id -u):$(id -g)` with `mideval/` mounted at `/local`.

## 1. Pipeline
```
FP32 ONNX (frozen) ──► AIMET AutoQuant (host venv) ──► model.onnx + model.encodings
                                                            │
     calib .raw + input_list.txt ───────────────┐           ▼
                                                 ├─► qairt-converter -q model.encodings  → model_fp.dlc
                                                 └─► qairt-quantizer --input_list ...    → model_w8a16.dlc
                                                            │
                         qnn-context-binary-generator (HTP v75) → model_w8a16.serialized.bin
                                                            │
                     adb push → qnn-net-run --retrieve_context (HTP) → outputs .raw → adb pull → compare
```

## 2. Commands (templates; `M=all_sensors_1h` shown)
```bash
# 2.1 Convert, with AIMET encodings as overrides
qairt-converter --input_network /local/aimet_quantization/out/$M/aq/model.onnx \
  --quantization_overrides      /local/aimet_quantization/out/$M/aq/model.encodings \
  --output_path                 /local/aimet_quantization/out/$M/qairt/${M}_fp.dlc

# 2.2 Quantize: overrides win; calib data covers the LSTM-internal tensors
qairt-quantizer --input_dlc  .../${M}_fp.dlc \
  --input_list /local/aimet_quantization/data/raw/$M/calib_input_list.txt \
  --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32 \
  --output_dlc .../${M}_w8a16.dlc --dump_encoding_json
#   variant A (control): same command WITHOUT AIMET overrides → isolates AIMET's contribution
#   variant B: --use_per_channel_quantization (if E3 won in AIMET)

# 2.3 Offline-prepare an HTP context binary for v75
qnn-context-binary-generator --backend $QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
  --model $QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnModelDlc.so \
  --dlc_path .../${M}_w8a16.dlc --binary_file ${M}_w8a16 \
  --output_dir .../ctx --config_file htp_v75_config.json   # backend ext. config: dsp_arch v75 / soc_model SM8650
```
Check after each step: `qairt-dlc-info` shows every tensor quantized (no unintended float fallback), and the IO datatypes are what we expect. Keep IO as float32 via `--preserve_io_datatype` if the runner should accept float `.raw` directly.

## 3. Push the runtime and model (device)
Runtime libs as in doc 04 §4.3, **plus** `libQnnHtpNetRunExtensions.so` (for the backend config) and `libQnnModelDlc.so` (only if running a DLC directly instead of a context binary).
```bash
D=/data/local/tmp/nz_aimet
adb -s d42f06b push ctx/${M}_w8a16.bin               $D/
adb -s d42f06b push data/raw/$M/test/                $D/test/      # float32 .raw test windows
adb -s d42f06b push data/raw/$M/test_input_list.txt  $D/           # paths relative to $D
```

## 4. Run on the HTP
```bash
adb -s d42f06b shell "cd $D && export LD_LIBRARY_PATH=$D/lib ADSP_LIBRARY_PATH=$D/lib && \
  ./bin/qnn-net-run --backend lib/libQnnHtp.so --retrieve_context ${M}_w8a16.bin \
     --input_list test_input_list.txt --output_dir out_htp --perf_profile burst --profiling_level basic"
adb -s d42f06b pull $D/out_htp aimet_quantization/out/$M/device/
```
Also run, for the verification matrix (doc 08):
* **CPU backend, FP32 DLC** (`--backend libQnnCpu.so --model libQnnModelDlc.so --dlc_path …_fp.dlc`). Proves the conversion itself is lossless on device.
* **HTP, QAIRT-only quantized** (variant A). The control for AIMET's contribution.
* Timing: `qnn-throughput-net-run` or the `--profiling_level` logs, then `qnn-profile-viewer`, for latency/throughput on HTP vs CPU.

## 5. Replacing `qidk_deploy.sh`
The new deploy script (`scripts/deploy_qidk.sh`, written in execution) will only use `adb push` + `qnn-net-run` (no device Python). Host-side Python then converts the pulled `.raw` outputs → Watts → net-zero table / anomaly flags, reusing the formatting of the existing runners. `model2/qidk_deployment/qidk_deploy.sh` is left untouched.

## 6. Optional: on-device app
For a live demo beyond `qnn-net-run`, the QIDK repo's Android apps (`qidk/Solutions/...`) show the SNPE/QNN Java/JNI pattern. That's a separate work item, only if a GUI demo is required.
