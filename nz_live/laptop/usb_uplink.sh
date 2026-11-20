#!/usr/bin/env bash
# USB fallback route from the QIDK to the campus API (19 Nov). The live loop (loop.ts) tries the API directly over
# Wi-Fi first, then http://127.0.0.1:19999 on the board. This script makes that second address work:
#   board 127.0.0.1:19999 --adb reverse (USB)--> laptop 127.0.0.1:19999 --socat--> campus API
# The laptop opens the connection to the API itself, so Docker's FORWARD=DROP (SYSTEM_DESIGN.md §9.2) doesn't apply.
# Runs as the systemd user service nz-usb-uplink.service (see README.md here); re-adds the reverse after a replug.
set -u
SERIAL=${SERIAL:-d42f06b}
PORT=${PORT:-19999}
API=${API:-10.2.203.214:9999}
log() { echo "$(date -Is) $*"; }

socat TCP-LISTEN:"$PORT",bind=127.0.0.1,reuseaddr,fork TCP:"$API",connect-timeout=10 &
RELAY=$!
trap 'kill $RELAY 2>/dev/null' EXIT
trap 'exit 0' TERM INT
log "relay 127.0.0.1:$PORT -> $API (pid $RELAY)"

was=""
while kill -0 "$RELAY" 2>/dev/null; do
	state=$(adb -s "$SERIAL" get-state 2>/dev/null || true)
	if [ "$state" = device ]; then
		if ! adb -s "$SERIAL" reverse --list 2>/dev/null | grep -q "tcp:$PORT"; then
			adb -s "$SERIAL" reverse "tcp:$PORT" "tcp:$PORT" >/dev/null && log "board connected: reverse tcp:$PORT set"
		fi
	elif [ "$was" = device ]; then
		log "board disconnected from USB"
	fi
	was=$state
	sleep 10
done
log "relay exited"
exit 1
