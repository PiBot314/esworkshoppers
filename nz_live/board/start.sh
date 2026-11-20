#!/system/bin/sh
# nz_live supervisor on the QIDK. Starts and keeps running:
#   nz_dashboard  :8080  (existing NPU server, unchanged)
#   llama-server  :8090  (Qwen3-4B, localhost only)
#   dashboard     :5173  (SvelteKit build on Bun: map, net zero, assistant, live loop)
# Started at boot by init (nz_live.rc, as user shell), or by hand:
#   sh /data/local/tmp/nz_live/start.sh           (stays in the foreground; use setsid/nohup to detach)
#   sh /data/local/tmp/nz_live/start.sh --stop
# Logs: /data/local/tmp/nz_live/logs/
R=/data/local/tmp/nz_live
AIMET=/data/local/tmp/nz_aimet
LOG=$R/logs
RUN=$R/run
mkdir -p $LOG $RUN $R/tmp $R/home $R/live

stop_all() {
	for f in $RUN/*.pid; do
		[ -f "$f" ] || continue
		kill "$(cat "$f")" 2>/dev/null
		rm -f "$f"
	done
}

if [ "$1" = "--stop" ]; then
	stop_all
	# children of the loops, by exact name (never pkill -f: it matches the calling shell)
	pkill -x nz_dashboard; pkill -x llama-server; pkill -x ld-musl-aarch64
	echo "stopped"
	exit 0
fi

say() { echo "$(date '+%F %T') $*" >> $LOG/supervisor.log; }

rotate() {  # keep each log under ~5 MB, one old copy
	[ -f "$1" ] && [ "$(stat -c %s "$1" 2>/dev/null || echo 0)" -gt 5000000 ] && mv -f "$1" "$1.1"
}

# keep NAME CMD...  : run CMD in the foreground, restart it when it exits (back-off 2 s .. 60 s)
keep() {
	name=$1; shift
	(
		delay=2
		while :; do
			rotate $LOG/$name.log
			say "start $name"
			started=$(date +%s)
			"$@" >> $LOG/$name.log 2>&1 &
			echo $! > $RUN/$name.child.pid
			wait $!
			rc=$?
			say "$name exited rc=$rc"
			# ran for over 5 minutes: treat as a fresh failure, reset the back-off
			[ $(( $(date +%s) - started )) -gt 300 ] && delay=2
			sleep $delay
			delay=$(( delay * 2 )); [ $delay -gt 60 ] && delay=60
		done
	) &
	echo $! > $RUN/$name.pid
}

stop_all
pkill -x nz_dashboard; pkill -x llama-server; pkill -x ld-musl-aarch64
say "supervisor up (pid $$, uid $(id -u))"

# Wait for Android to finish booting (no-op when started by hand).
while [ "$(getprop sys.boot_completed)" != "1" ]; do sleep 2; done
svc power stayon true 2>/dev/null   # keep the board's own screen on while powered (for the app)

cd /   # Bun walks up from the working directory; shell can't list /data, so start from /

keep nz_dashboard "$AIMET/dashboard/nz_dashboard" --port 8080 --root "$AIMET"

keep llama-server "$R/bin/llama-server" \
	-m "$R/models/${SLM_MODEL_FILE:-Qwen3-4B-Instruct-2507-Q4_0-requant.gguf}" \
	--host 127.0.0.1 --port 8090 -t 6 --ctx-size 8192 --parallel 1 --jinja \
	--cache-reuse 256 --load-mode none

keep dashboard env \
	TMPDIR=$R/tmp HOME=$R/home \
	CAMPUS_DB=$R/data/campus.sqlite \
	SLM_URL=http://127.0.0.1:8090/v1 NPU_URL=http://127.0.0.1:8080 \
	LIVE=1 LIVE_DIR=$R/live LIVE_API="$(cat $R/live_api.url 2>/dev/null || echo http://10.2.203.214:9999,http://127.0.0.1:19999)" \
	HOST=0.0.0.0 PORT=5173 \
	"$R/lib/ld-musl-aarch64.so.1" --library-path "$R/lib" "$R/bin/bun" "$R/dashboard/build/index.js"

say "all started"
wait
