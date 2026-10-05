#!/usr/bin/env bash
# One-command launcher for the QIDK demo dashboard (USB presentation setup).
#   scripts/14_dashboard_start.sh            push (if changed), (re)start server on the QIDK, adb forward, open browser
#   scripts/14_dashboard_start.sh --stop     stop the server
# Then open http://localhost:8080 (also reachable at http://<QIDK-Wi-Fi-IP>:8080 on the same network).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
AQ="$(cd "$HERE/.." && pwd)"
SERIAL="${SERIAL:-d42f06b}"
PORT="${PORT:-8080}"
D=/data/local/tmp/nz_aimet
adbr() { local i; for i in 1 2 3; do adb -s "$SERIAL" "$@" && return 0
  adb kill-server >/dev/null 2>&1; adb start-server >/dev/null 2>&1; timeout 60 adb wait-for-device || true; done; return 1; }

adbr shell "pkill nz_dashboard" >/dev/null 2>&1 || true
if [ "${1:-}" = "--stop" ]; then echo "stopped"; exit 0; fi

# runtime + model libraries (normally already there from the P4 runs)
if ! adbr shell "test -x $D/bin/qnn-net-run"; then adbr push "$AQ/release/qidk_runtime_2.47/bin" "$AQ/release/qidk_runtime_2.47/lib" "$D/" >/dev/null; fi
for M in all_sensors_1h all_sensors_24h single_node; do
  adbr shell "mkdir -p $D/$M/models"
  for V in aimet_w8a16 fp32; do
    adbr shell "test -f $D/$M/models/lib$V.so" || adbr push "$AQ/out/$M/qnn/lib/aarch64-android/lib$V.so" "$D/$M/models/" >/dev/null
  done
done
# server + web files (adb push only transfers what changed)
adbr shell "mkdir -p $D/dashboard"
adbr push --sync "$AQ/dashboard/bin/nz_dashboard" "$D/dashboard/" >/dev/null
adbr push --sync "$AQ/dashboard/www" "$D/dashboard/" >/dev/null
adbr shell "chmod +x $D/dashboard/nz_dashboard && $D/dashboard/nz_dashboard --daemon --port $PORT --root $D --log $D/dashboard/server.log"
adbr forward tcp:$PORT tcp:$PORT >/dev/null
for i in $(seq 1 20); do curl -sf "http://localhost:$PORT/api/status" >/dev/null && break; done
curl -sf "http://localhost:$PORT/api/status" && echo
WIFI=$(adbr shell "ip -4 addr show wlan0 2>/dev/null | grep -o 'inet [0-9.]*' | cut -d' ' -f2" | tr -d '\r')
echo "Dashboard:  http://localhost:$PORT   (USB)      |  http://${WIFI:-<no wifi>}:$PORT   (Wi-Fi)"
[ "${NO_BROWSER:-0}" = 1 ] || (xdg-open "http://localhost:$PORT" >/dev/null 2>&1 &)
