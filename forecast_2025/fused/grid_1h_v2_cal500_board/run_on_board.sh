#!/usr/bin/env bash
# Push libgrid_1h_v2_cal500.so + test windows to the QIDK, run on the HTP, pull outputs (needs the QAIRT 2.47 runtime in $D/{bin,lib},
# as for aimet_quantization/scripts/09_run_on_qidk.sh).
set -euo pipefail
SERIAL="${SERIAL:-d42f06b}"; D=/data/local/tmp/nz_aimet
HERE="$(cd "$(dirname "$0")" && pwd)"
adb -s $SERIAL shell "mkdir -p $D/grid_1h_v2_cal500"
adb -s $SERIAL push "$HERE/test" "$D/grid_1h_v2_cal500/" | tail -1
adb -s $SERIAL push "$HERE/test_list.txt" "$HERE/libgrid_1h_v2_cal500.so" "$D/grid_1h_v2_cal500/" > /dev/null
adb -s $SERIAL shell "cd $D && rm -rf grid_1h_v2_cal500/out && export LD_LIBRARY_PATH=$D/lib ADSP_LIBRARY_PATH=$D/lib && \
  ./bin/qnn-net-run --backend lib/libQnnHtp.so --model grid_1h_v2_cal500/libgrid_1h_v2_cal500.so --input_list grid_1h_v2_cal500/test_list.txt \
    --output_dir grid_1h_v2_cal500/out --perf_profile burst --profiling_level basic > grid_1h_v2_cal500/run.log 2>&1"
rm -rf "$HERE/device_out"
adb -s $SERIAL pull "$D/grid_1h_v2_cal500/out" "$HERE/device_out" > /dev/null
adb -s $SERIAL pull "$D/grid_1h_v2_cal500/run.log" "$HERE/" > /dev/null
echo "pulled -> $HERE/device_out; score: python3 forecast_2025/b5_score_board.py grid_1h_v2_cal500"
