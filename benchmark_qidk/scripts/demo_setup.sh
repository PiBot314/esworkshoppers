#!/usr/bin/env bash
# Get the QIDK demo-ready in one command (run from the PC with the board on USB):
#   - copies any missing benchmark files to the board (models, test data, tools, demo.sh)
#   - installs the app, starts the nz_dashboard server, keeps the screen awake, opens the app
# After this the USB cable can be unplugged; everything runs from the touchscreen until the board reboots.
#   benchmark_qidk/scripts/demo_setup.sh            set up and open the app
#   benchmark_qidk/scripts/demo_setup.sh --check    only check that everything is in place
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
B="$(cd "$HERE/.." && pwd)"
AQ="$(cd "$B/../aimet_quantization" && pwd)"
DV=/data/local/tmp/nz_bench
MODELS=(baseline_h4 guided_h4 baseline_24h guided_24h cnn_lstm_24h)

adb get-state >/dev/null 2>&1 || { echo "QIDK not found: plug in the USB cable and check 'adb devices'"; exit 1; }

check() {
  local miss=0
  for f in demo.sh bin/ort_bench ortlib/libonnxruntime.so onnx/lightgbm_24h.onnx data/y_test_h4.bin data/y_test_h96.bin \
           data/test_list.txt data/test_last_00000.raw data/test/04831.raw; do
    adb shell "test -e $DV/$f" || { echo "missing on board: $f"; miss=1; }
  done
  for M in "${MODELS[@]}"; do
    adb shell "test -e $DV/onnx/$M.onnx" || { echo "missing on board: onnx/$M.onnx"; miss=1; }
    for p in fp32 w8a16 fp16 w8a8; do adb shell "test -e $DV/qnn/$M/lib$p.so" || { echo "missing on board: qnn/$M/lib$p.so"; miss=1; }; done
  done
  adb shell "test -x /data/local/tmp/nz_aimet/bin/qnn-net-run" || { echo "missing QNN runtime (nz_aimet)"; miss=1; }
  return $miss
}

if [ "${1:-}" = --check ]; then check && echo "all demo files are on the board"; exit; fi

if ! check >/dev/null 2>&1; then
  echo "copying benchmark files to the board…"
  adb shell "mkdir -p $DV/bin $DV/ortlib $DV/onnx $DV/data"
  adb push "$B/tools/bin/ort_bench" "$B/tools/bin/vphmon" $DV/bin/ >/dev/null
  adb push "$B/third_party/ort/jni/arm64-v8a/libonnxruntime.so" $DV/ortlib/ >/dev/null
  for M in "${MODELS[@]}"; do
    SRC=$(python3 -c "import json;print(json.load(open('$HERE/models.json'))['$M']['onnx'])")
    adb push "$B/../$SRC" $DV/onnx/$M.onnx >/dev/null
    [ -f "$B/../$SRC.data" ] && adb push "$B/../$SRC.data" $DV/onnx/ >/dev/null
    adb shell "mkdir -p $DV/qnn/$M"; adb push "$B"/models/$M/lib/aarch64-android/*.so $DV/qnn/$M/ >/dev/null
  done
  adb push "$B/models/lightgbm_24h.onnx" $DV/onnx/ >/dev/null
  adb push --sync "$B/data/test" $DV/data/ >/dev/null
  adb push "$B/data/test_list.txt" "$B/data/test_list_h96.txt" "$B/data/test_last_00000.raw" \
           "$B/data/y_test_h4.bin" "$B/data/y_test_h96.bin" $DV/data/ >/dev/null
fi
adb push "$B/board/demo.sh" $DV/ >/dev/null
adb shell "chmod +x $DV/bin/*"
check

# app (installed only if missing or older than the local build)
adb install -r "$AQ/dashboard/bin/nz_demo.apk" >/dev/null && echo "app installed"
# server (also pushes the old dashboard's models if needed)
NO_BROWSER=1 "$AQ/scripts/14_dashboard_start.sh" | tail -1
curl -sf "http://localhost:8080/api/demo?cmd=models" >/dev/null && echo "benchmark endpoint OK"
# screen: stay on while powered, wake, unlock (swipe), open the app on the Benchmark-capable build
# (the board reports no AC/USB power because its battery is simulated, so "stay on while plugged in" can't be relied on:
#  also set the screen timeout to the maximum and turn off the swipe lock screen; both persist across reboots)
adb shell "svc power stayon true; settings put system screen_off_timeout 2147483647; locksettings set-disabled true >/dev/null; \
  input keyevent KEYCODE_WAKEUP; sleep 1; input swipe 540 2000 540 500 250; sleep 1; \
  am start -n dev.netzero.qidk/.MainActivity" >/dev/null
echo "Ready: the app is open on the QIDK. Tap 'Benchmark' in the bottom bar. The USB cable can now be unplugged."
