#!/usr/bin/env bash
# P4: build QNN model libraries (Android aarch64) with QAIRT 2.47 inside the qairt-fixed image.
# Usage (host):  scripts/08_qairt_build.sh <model> [<model> ...]
#
# Path used (see doc 14 for why): qnn-onnx-converter (convert + quantize in one step, AIMET encodings as
# overrides, missing LSTM-internal encodings calibrated from the input list) with
#   --expand_lstm_op_structure  LSTM -> MatMul/Add/Sigmoid/Tanh/Mul per step (the fused quantized Lstm op failed
#                               HTP op validation)
#   --preserve_io layout        keep the ONNX [1, 96, F] input layout (default re-lays it out as [1, F, 96])
# then qnn-model-lib-generator -> lib<variant>.so
#
# Variants (out/<model>/qnn/lib/aarch64-android/):
#   libfp32.so          FP32 graph (device CPU backend, V3)
#   libaimet_w8a16.so   AIMET AutoQuant export + encodings, W8A16          <- deliverable (HTP)
#   libqairt_w8a16.so   control: original ONNX, QNN calibration only, W8A16 (HTP)
#   libaimet_w8a8.so    AIMET W8A8 export + encodings (HTP)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MIDEVAL="$(cd "$HERE/../.." && pwd)"
IMAGE="${IMAGE:-qairt-fixed:2.47}"

declare -A ONNX=( [all_sensors_1h]=all_sensors_lstm_qidk.onnx [all_sensors_24h]=all_sensors_24h_lstm_qidk.onnx
                  [single_node]=SL-PL00-00_lstm_qidk.onnx )

for M in "$@"; do
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$MIDEVAL:/local:z" -e M="$M" -e ONNX="${ONNX[$M]}" "$IMAGE" bash -c '
    source /usr/venv/bin/activate; source $QAIRT_SDK_ROOT/bin/envsetup.sh >/dev/null   # not set -u safe
    set -eo pipefail
    export PATH=$ANDROID_NDK_ROOT:$PATH
    A=/local/aimet_quantization; O=$A/out/$M/qnn; mkdir -p $O/logs; cd $O
    CAL=$A/data/raw/$M/calib_list.txt
    COMMON="--expand_lstm_op_structure --preserve_io layout"
    build() {   # $1 = variant, rest = converter args
      local V=$1; shift
      qnn-onnx-converter $COMMON --output_path $O/$V.cpp "$@" > $O/logs/conv_$V.log 2>&1
      qnn-model-lib-generator -c $O/$V.cpp -b $O/$V.bin -o $O/lib -t aarch64-android > $O/logs/libgen_$V.log 2>&1
      echo "[$M] built $V"
    }
    build fp32        --input_network /local/model2/qidk_deployment/$ONNX
    build aimet_w8a16 --input_network $A/out/$M/final/model.onnx --quantization_overrides $A/out/$M/final/model.qairt.encodings \
                      --input_list $CAL --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32
    build qairt_w8a16 --input_network /local/model2/qidk_deployment/$ONNX \
                      --input_list $CAL --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32
    build aimet_w8a8  --input_network $A/out/$M/w8a8/model.onnx --quantization_overrides $A/out/$M/w8a8/model.qairt.encodings \
                      --input_list $CAL --act_bitwidth 8 --weights_bitwidth 8 --bias_bitwidth 32
    ls -la $O/lib/aarch64-android/
  '
done
