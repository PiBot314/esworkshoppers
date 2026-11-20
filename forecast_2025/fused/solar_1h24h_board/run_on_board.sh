#!/usr/bin/env bash
# Push libsolar_1h24h.so + test windows to the QIDK, run on the HTP, pull outputs (needs the QAIRT 2.47 runtime in $D/{bin,lib},
# as for aimet_quantization/scripts/09_run_on_qidk.sh).
set -euo pipefail
SERIAL="${SERIAL:-d42f06b}"; D=/data/local/tmp/nz_aimet
HERE="$(cd "$(dirname "$0")" && pwd)"
adb -s $SERIAL shell "mkdir -p $D/solar_1h24h"
adb -s $SERIAL push "$HERE/test" "$D/solar_1h24h/" | tail -1
adb -s $SERIAL push "$HERE/test_list.txt" "$HERE/libsolar_1h24h.so" "$D/solar_1h24h/" > /dev/null
adb -s $SERIAL shell "cd $D && rm -rf solar_1h24h/out && export LD_LIBRARY_PATH=$D/lib ADSP_LIBRARY_PATH=$D/lib && \
  ./bin/qnn-net-run --backend lib/libQnnHtp.so --model solar_1h24h/libsolar_1h24h.so --input_list solar_1h24h/test_list.txt \
    --output_dir solar_1h24h/out --perf_profile burst --profiling_level basic > solar_1h24h/run.log 2>&1"
rm -rf "$HERE/device_out"
adb -s $SERIAL pull "$D/solar_1h24h/out" "$HERE/device_out" > /dev/null
adb -s $SERIAL pull "$D/solar_1h24h/run.log" "$HERE/" > /dev/null
echo "pulled -> $HERE/device_out; score: python3 forecast_2025/b5_score_board.py solar_1h24h"
