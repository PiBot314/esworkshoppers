#!/usr/bin/env bash
# Latency on the QIDK, one input window, batch 1:
#   htp_w8a16   NPU (QNN HTP), 1000 inferences, perf profile burst; also power_saver for guided_24h
#   cpu_fp32    QNN CPU backend (FP32), 300 inferences
#   ort_t1/t4   ONNX Runtime CPU (FP32), 1 and 4 threads, 50 warm-up + 1000 timed runs
#   lightgbm    ONNX Runtime CPU, 1 and 4 threads
# Raw outputs: results/latency/<model>_<config>.txt
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
B="$(cd "$HERE/.." && pwd)"
DV=/data/local/tmp/nz_bench
RT=/data/local/tmp/nz_aimet
OUT="$B/results/latency"; mkdir -p "$OUT"
MODELS=(baseline_h4 guided_h4 baseline_24h guided_24h cnn_lstm_24h)
ENV="export LD_LIBRARY_PATH=$RT/lib ADSP_LIBRARY_PATH='$RT/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp'"
adb shell "cd $DV/data && head -1 test_list.txt > one.txt"

qnn() {  # model backend variant n perf_profile tag
  local M=$1 BE=$2 V=$3 N=$4 PP=$5 TAG=$6 LIB
  LIB=$([ $BE = htp ] && echo libQnnHtp.so || echo libQnnCpu.so)
  adb shell "cd $DV/data && rm -rf lat_tmp && $ENV && \
    $RT/bin/qnn-net-run --backend $RT/lib/$LIB --model ../qnn/$M/lib$V.so --input_list one.txt --num_inferences $N \
      --keep_num_outputs 1 --perf_profile $PP --profiling_level basic --output_dir lat_tmp > /dev/null 2>&1 && \
    $RT/bin/qnn-profile-viewer --input_log lat_tmp/qnn-profiling-data_0.log" > "$OUT/${M}_$TAG.txt" 2>&1
  echo "[$M] $TAG: $(grep -m1 'NetRun IPS' "$OUT/${M}_$TAG.txt" | sed 's/^ *//')"
}
ortb() {  # onnx-name threads input tag
  adb shell "cd $DV/data && LD_LIBRARY_PATH=../ortlib ../bin/ort_bench ../onnx/$1.onnx $2 latency $3 50 1000" > "$OUT/$1_$4.txt" 2>&1
  echo "[$1] $4: $(cat "$OUT/$1_$4.txt")"
}

if [ -n "${HTP_ONLY:-}" ]; then        # e.g. HTP_ONLY="w8a8 fp16": only these NPU variants, all models
  for M in "${MODELS[@]}"; do for V in $HTP_ONLY; do qnn $M htp $V 1000 burst htp_$V; done; sleep 5; done
  exit 0
fi
for M in "${MODELS[@]}"; do
  qnn $M htp w8a16 1000 burst htp_w8a16
  [ $M = guided_24h ] && qnn $M htp w8a16 1000 power_saver htp_w8a16_powersaver
  qnn $M cpu fp32 300 burst cpu_fp32
  ortb $M 1 test/00000.raw ort_t1
  ortb $M 4 test/00000.raw ort_t4
  sleep 10   # let the SoC cool a little between models
done
ortb lightgbm_24h 1 test_last_00000.raw ort_t1
ortb lightgbm_24h 4 test_last_00000.raw ort_t4
adb shell "rm -rf $DV/data/lat_tmp"
