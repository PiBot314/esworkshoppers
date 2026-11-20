#!/usr/bin/env bash
# Run every test window through each model on the QIDK and pull the outputs back:
#   htp_w8a16  NPU, QNN W8A16          cpu_fp32  QNN CPU backend, FP32          ort  ONNX Runtime CPU, FP32
# Outputs land in results/outputs/<model>_<variant>.bin (float32, windows x outputs).
# Usage: scripts/06_device_accuracy.sh [model ...]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
B="$(cd "$HERE/.." && pwd)"
DV=/data/local/tmp/nz_bench
RT=/data/local/tmp/nz_aimet           # QAIRT 2.47 runtime already on the board (bin/, lib/)
MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(baseline_h4 guided_h4 baseline_24h guided_24h cnn_lstm_24h)
mkdir -p "$B/results/outputs"

for M in "${MODELS[@]}"; do
  H=$(python3 -c "import json;print(json.load(open('$HERE/models.json'))['$M']['horizon'])")
  LIST=$([ "$H" = 96 ] && echo test_list_h96.txt || echo test_list.txt)   # paths inside are relative to $DV/data
  adb shell "mkdir -p $DV/qnn/$M" >/dev/null
  adb push "$B"/models/$M/lib/aarch64-android/lib*.so $DV/qnn/$M/ >/dev/null
  for VB in ${RUNS:-htp:w8a16 cpu:fp32}; do      # e.g. RUNS="htp:w8a8 htp:fp16"
    BE=${VB%%:*}; V=${VB#*:}; LIB=$([ $BE = htp ] && echo libQnnHtp.so || echo libQnnCpu.so)
    T0=$(date +%s)
    adb shell "cd $DV/data && rm -rf acc_tmp && export LD_LIBRARY_PATH=$RT/lib ADSP_LIBRARY_PATH='$RT/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' && \
      $RT/bin/qnn-net-run --backend $RT/lib/$LIB --model ../qnn/$M/lib$V.so --input_list $LIST --output_dir acc_tmp \
        --perf_profile burst > acc_tmp.log 2>&1; tail -1 acc_tmp.log; \
      n=\$(wc -l < $LIST); i=0; : > acc_out.bin; while [ \$i -lt \$n ]; do cat acc_tmp/Result_\$i/*.raw >> acc_out.bin; i=\$((i+1)); done; ls -l acc_out.bin"
    adb pull $DV/data/acc_out.bin "$B/results/outputs/${M}_${BE}_${V}.bin" >/dev/null
    echo "[$M] ${BE}_$V done in $(( $(date +%s) - T0 )) s"
  done
  if [ "${SKIP_ORT:-0}" != 1 ]; then
    adb shell "cd $DV/data && LD_LIBRARY_PATH=../ortlib ../bin/ort_bench ../onnx/$M.onnx 1 outputs $LIST acc_out.bin"
    adb pull $DV/data/acc_out.bin "$B/results/outputs/${M}_ort_fp32.bin" >/dev/null
    echo "[$M] ort_fp32 done"
  fi
done
adb shell "rm -rf $DV/data/acc_tmp $DV/data/acc_out.bin $DV/data/acc_tmp.log"
