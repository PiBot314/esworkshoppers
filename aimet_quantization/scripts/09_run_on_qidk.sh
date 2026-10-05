#!/usr/bin/env bash
# P4: push model libraries + TEST windows to the QIDK, run them with qnn-net-run, pull outputs.
# Usage: scripts/09_run_on_qidk.sh <model> [variant ...]     (default variants: all four)
# Needs the 2.47 runtime already in $D/{bin,lib} (doc 10) and host adb.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
AQ="$(cd "$HERE/.." && pwd)"
SERIAL="${SERIAL:-d42f06b}"
D=/data/local/tmp/nz_aimet
M=$1; shift
if [ $# -eq 0 ]; then VARIANTS=(cpu:fp32 htp:aimet_w8a16 htp:qairt_w8a16 htp:aimet_w8a8); else VARIANTS=("$@"); fi
# The host adb connection to the QIDK dropped twice during long runs (device itself stayed up):
# wrap every call with a reconnect-and-retry.
ADB=adbr
adbr() {
  local i
  for i in 1 2 3; do
    adb -s "$SERIAL" "$@" && return 0
    echo "[adb] retry $i after reconnect" >&2
    adb kill-server >/dev/null 2>&1; adb start-server >/dev/null 2>&1; timeout 60 adb wait-for-device || true
  done
  return 1
}

$ADB shell "mkdir -p $D/$M/models"
if [ "$($ADB shell "ls $D/$M/test 2>/dev/null | wc -l" | tr -d '\r')" != "$(ls "$AQ/data/raw/$M/test" | wc -l)" ]; then
  $ADB push "$AQ/data/raw/$M/test" "$D/$M/" | tail -1
fi
$ADB push "$AQ/data/raw/$M/test_list.txt" "$D/$M/" > /dev/null

for VB in "${VARIANTS[@]}"; do
  BE=${VB%%:*}; V=${VB#*:}
  LIB=$([ "$BE" = htp ] && echo libQnnHtp.so || echo libQnnCpu.so)
  $ADB push "$AQ/out/$M/qnn/lib/aarch64-android/lib$V.so" "$D/$M/models/" > /dev/null
  OUTD=$M/out_${BE}_$V
  $ADB shell "cd $D && rm -rf $OUTD && export LD_LIBRARY_PATH=$D/lib ADSP_LIBRARY_PATH=$D/lib && \
     ./bin/qnn-net-run --backend lib/$LIB --model $M/models/lib$V.so --input_list $M/test_list.txt \
       --output_dir $OUTD --perf_profile burst --profiling_level basic > $M/run_${BE}_$V.log 2>&1"
  rm -rf "$AQ/out/$M/device/${BE}_$V"; mkdir -p "$AQ/out/$M/device"
  $ADB pull "$D/$OUTD" "$AQ/out/$M/device/${BE}_$V" > /dev/null
  $ADB pull "$D/$M/run_${BE}_$V.log" "$AQ/out/$M/device/" > /dev/null
  echo "[$M] ran ${BE}:$V -> out/$M/device/${BE}_$V"
done
