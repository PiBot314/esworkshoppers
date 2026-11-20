#!/usr/bin/env bash
# Relative power on the QIDK.
# The board runs from its DC adapter and its battery gauge is simulated, so there is no current sensor to read.
# The PMIC does report the main supply rail voltage (vph_pwr); it sags in proportion to the current drawn.
# For each configuration: 15 s idle, then the model runs back-to-back for ~30 s (rail sampled for 20 s in the
# middle of that), then 15 s idle again. Extra load = mean(idle before, idle after) - load.
# Energy per inference is then proportional to (extra load / inference rate). Also measures one and four
# busy-looping CPU cores as reference loads. The whole sequence is repeated REPS times.
# Raw lines: results/power/power_raw.txt
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
B="$(cd "$HERE/.." && pwd)"
DV=/data/local/tmp/nz_bench
RT=/data/local/tmp/nz_aimet
REPS=${REPS:-2}
OUT="$B/results/power"; mkdir -p "$OUT"; RAW="$OUT/power_raw.txt"; : > "$RAW"
ENV="export LD_LIBRARY_PATH=$RT/lib ADSP_LIBRARY_PATH='$RT/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp'"
adb shell "cd $DV/data && head -1 test_list.txt > one.txt"

vph() { adb shell "su 0 $DV/bin/vphmon $1 20" | tr -d '\r'; }

# number of QNN inferences that take ~32 s, from the latency run's IPS
nq() { python3 - "$B/results/latency/$1_$2.txt" <<'EOF'
import re,sys
ips=float(re.search(r"NetRun IPS[^:]*:\s*([\d.]+)", open(sys.argv[1]).read()).group(1))
print(int(ips*32)+10)
EOF
}

start_load() {  # cfg model -> starts the load in the background on the board
  local C=$1 M=$2
  case $C in
    htp_w8a16|cpu_fp32)
      local BE=${C%%_*} V=${C#*_} LIB N
      LIB=$([ $BE = htp ] && echo libQnnHtp.so || echo libQnnCpu.so); N=$(nq $M $C)
      adb shell "cd $DV/data && rm -rf pw_tmp && $ENV && nohup $RT/bin/qnn-net-run --backend $RT/lib/$LIB \
        --model ../qnn/$M/lib$V.so --input_list one.txt --num_inferences $N --keep_num_outputs 1 --perf_profile burst \
        --profiling_level basic --output_dir pw_tmp > pw_load.log 2>&1 < /dev/null &" ;;
    ort_t1|ort_t4)
      local T=${C#ort_t} IN=test/00000.raw
      [ $M = lightgbm_24h ] && IN=test_last_00000.raw
      adb shell "cd $DV/data && LD_LIBRARY_PATH=../ortlib nohup ../bin/ort_bench ../onnx/$M.onnx $T loop $IN 32 \
        > pw_load.log 2>&1 < /dev/null &" ;;
    busy1|busy4)
      local K=${C#busy}
      adb shell "i=0; while [ \$i -lt $K ]; do nohup timeout 33 sh -c 'while :; do :; done' > /dev/null 2>&1 < /dev/null & i=\$((i+1)); done" ;;
  esac
}

rate() {  # inferences per second achieved during the load
  local C=$1
  case $C in
    htp_w8a16|cpu_fp32)
      adb shell "$RT/bin/qnn-profile-viewer --input_log $DV/data/pw_tmp/qnn-profiling-data_0.log" | \
        grep -m1 'NetRun IPS' | grep -oE '[0-9]+\.[0-9]+' | head -1 ;;
    ort_*) adb shell "cat $DV/data/pw_load.log" | grep -oE 'rate_per_s=[0-9.]+' | cut -d= -f2 ;;
    *) echo 0 ;;
  esac
}

CONFIGS=()
for M in guided_h4 guided_24h cnn_lstm_24h; do for C in htp_w8a16 cpu_fp32 ort_t1 ort_t4; do CONFIGS+=("$M:$C"); done; done
CONFIGS+=("lightgbm_24h:ort_t1" "lightgbm_24h:ort_t4" "ref:busy1" "ref:busy4")

IDLE=$(vph 15)
for r in $(seq 1 "$REPS"); do
  ORDER=("${CONFIGS[@]}"); [ $((r % 2)) -eq 0 ] && ORDER=($(printf '%s\n' "${CONFIGS[@]}" | tac))
  for MC in "${ORDER[@]}"; do
    M=${MC%%:*}; C=${MC#*:}
    start_load "$C" "$M"
    sleep 6                                  # skip start-up (model load, graph finalize)
    LOAD=$(vph 20)
    while adb shell "pidof qnn-net-run ort_bench timeout > /dev/null"; do sleep 1; done
    RATE=$(rate "$C")
    IDLE2=$(vph 15)
    echo "rep=$r model=$M cfg=$C rate=$RATE | idle_before: $IDLE | load: $LOAD | idle_after: $IDLE2" | tee -a "$RAW"
    IDLE=$IDLE2
  done
done
adb shell "rm -rf $DV/data/pw_tmp $DV/data/pw_load.log"
