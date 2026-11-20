#!/usr/bin/env bash
# Tests A3 + A4 on the board (plan_reduced_shortterm.md §6): watch the live loop until $END (IST, HH:MM),
# save its status after every cycle, and inject demo faults after odd-numbered cycles (both signals).
#   bash live_run.sh 10:00 > ../runs/<date>.log
END=${1:-10:00}
OUT=$(dirname "$0")/../runs; mkdir -p "$OUT"
PL=(dropout damage40 dropout damage40); GR=(spike dropout spike damage40)
st() { adb shell 'curl -s -m 10 http://127.0.0.1:5173/api/live/status'; }
last=""; k=0
while [ "$(TZ=Asia/Kolkata date +%H:%M)" \< "$END" ]; do
	s=$(st); b=$(echo "$s" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("last_reading_checked") or "")' 2>/dev/null)
	if [ -n "$b" ] && [ "$b" != "$last" ]; then
		[ -n "$last" ] && k=$((k + 1))
		last=$b
		echo "$s" >> "$OUT/status.jsonl"
		echo "$(TZ=Asia/Kolkata date +%T) cycle $k checked $b"
		if [ $k -ge 1 ] && [ $((k % 2)) -eq 1 ] && [ $(((k - 1) / 2)) -lt 4 ]; then
			i=$(((k - 1) / 2))
			for x in "pl00 ${PL[$i]}" "grid ${GR[$i]}"; do
				set -- $x
				adb shell "curl -s -m 10 -X POST -H content-type:application/json -d '{\"signal\":\"$1\",\"kind\":\"$2\",\"steps\":1}' http://127.0.0.1:5173/api/live/inject"; echo
			done
		fi
	fi
	sleep 60
done
adb shell 'cat /data/local/tmp/nz_live/live/events.jsonl' > "$OUT/events.jsonl"
echo "done at $(TZ=Asia/Kolkata date +%T), $k cycles"
