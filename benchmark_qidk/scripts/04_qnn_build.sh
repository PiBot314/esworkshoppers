#!/usr/bin/env bash
# Build QNN model libraries (Android aarch64) for each model in models.json, inside the qairt-fixed:2.47 image.
# Same converter path as aimet_quantization/scripts/08_qairt_build.sh:
#   fp32.so   FP32 graph, for the QNN CPU backend
#   fp16.so   NPU float16 (VARIANTS=fp16)
#   w8a8.so   8-bit weights and activations (VARIANTS=w8a8)
#   w8a16.so  8-bit weights / 16-bit activations, calibrated by QNN on 512 training windows, for the NPU (HTP)
# Usage: scripts/04_qnn_build.sh [model ...]   (default: all)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
B="$(cd "$HERE/.." && pwd)"
FINAL="$(cd "$B/.." && pwd)"
IMAGE="${IMAGE:-qairt-fixed:2.47}"
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=($(python3 -c "import json;print(' '.join(json.load(open('$HERE/models.json'))))"))

# calibration list with container paths
sed "s#^$FINAL#/local#" "$B/data/calib_list.txt" > "$B/data/calib_list_docker.txt"

for M in "${MODELS[@]}"; do
  ONNX=$(python3 -c "import json;print(json.load(open('$HERE/models.json'))['$M']['onnx'])")
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$FINAL:/local:z" -e M="$M" -e ONNX="$ONNX" \
    -e VARIANTS="${VARIANTS:-fp32 w8a16}" "$IMAGE" bash -c '
    source /usr/venv/bin/activate; source $QAIRT_SDK_ROOT/bin/envsetup.sh >/dev/null
    set -eo pipefail
    export PATH=$ANDROID_NDK_ROOT:$PATH
    O=/local/benchmark_qidk/models/$M; mkdir -p $O/logs; cd $O
    COMMON="--expand_lstm_op_structure --preserve_io layout --input_network /local/$ONNX"
    build() { local V=$1; shift
      qnn-onnx-converter $COMMON --output_path $O/$V.cpp "$@" > $O/logs/conv_$V.log 2>&1
      qnn-model-lib-generator -c $O/$V.cpp -b $O/$V.bin -o $O/lib -t aarch64-android > $O/logs/libgen_$V.log 2>&1
      echo "[$M] built $V"; }
    CAL=/local/benchmark_qidk/data/calib_list_docker.txt
    for V in $VARIANTS; do case $V in
      fp32)  build fp32 ;;
      fp16)  build fp16  --float_bitwidth 16 ;;                       # NPU float16, no quantization
      w8a16) build w8a16 --input_list $CAL --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32 ;;
      w8a8)  build w8a8  --input_list $CAL --act_bitwidth 8  --weights_bitwidth 8 --bias_bitwidth 32 ;;
    esac; done
  '
done
ls -la "$B"/models/*/lib/aarch64-android/
