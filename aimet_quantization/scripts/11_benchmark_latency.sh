#!/usr/bin/env bash
# P4: on-device latency. For each model/variant: 300 inferences of one window with basic profiling,
# then qnn-profile-viewer (on device) summarises per-inference execute times.
# Usage: scripts/11_benchmark_latency.sh <model> [backend:variant ...]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
AQ="$(cd "$HERE/.." && pwd)"
SERIAL="${SERIAL:-d42f06b}"
D=/data/local/tmp/nz_aimet
N=300
M=$1; shift
if [ $# -eq 0 ]; then VARIANTS=(cpu:fp32 htp:aimet_w8a16 htp:qairt_w8a16 htp:aimet_w8a8); else VARIANTS=("$@"); fi
adbr() { local i; for i in 1 2 3; do adb -s "$SERIAL" "$@" && return 0
  adb kill-server >/dev/null 2>&1; adb start-server >/dev/null 2>&1; timeout 60 adb wait-for-device || true; done; return 1; }

mkdir -p "$AQ/out/$M/latency"
adbr shell "cd $D && head -1 $M/test_list.txt > $M/one.txt"
for VB in "${VARIANTS[@]}"; do
  BE=${VB%%:*}; V=${VB#*:}
  LIB=$([ "$BE" = htp ] && echo libQnnHtp.so || echo libQnnCpu.so)
  adbr shell "cd $D && rm -rf $M/lat_${BE}_$V && export LD_LIBRARY_PATH=$D/lib ADSP_LIBRARY_PATH=$D/lib && \
    ./bin/qnn-net-run --backend lib/$LIB --model $M/models/lib$V.so --input_list $M/one.txt --num_inferences $N \
      --keep_num_outputs 1 --perf_profile burst --profiling_level basic --output_dir $M/lat_${BE}_$V > /dev/null 2>&1 && \
    ./bin/qnn-profile-viewer --input_log $M/lat_${BE}_$V/qnn-profiling-data_0.log" > "$AQ/out/$M/latency/${BE}_$V.txt" 2>&1
  echo "[$M] ${BE}:$V"; grep -iE "execute|finalize|init" "$AQ/out/$M/latency/${BE}_$V.txt" | head -8
done
