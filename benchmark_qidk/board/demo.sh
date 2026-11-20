#!/system/bin/sh
# On-board demo for the QIDK. Everything runs on the board; no PC needed once the files are pushed.
#
#   sh demo.sh models                          list models and the NPU precisions available for each
#   sh demo.sh forecast <model> <window> [p]   one real test window: NPU (precision p) vs QNN CPU vs ONNX Runtime CPU,
#                                              forecast vs actual, time per inference
#   sh demo.sh speed [p]                       time per inference for every model on NPU / QNN CPU / ONNX Runtime (~1 min)
#   sh demo.sh accuracy <model> [p] [n]        first n test windows (default 500, "all" for every one) on the NPU:
#                                              R² per domain, and agreement with the FP32 model (SQNR)
#
# models: baseline_h4 guided_h4 baseline_24h guided_24h cnn_lstm_24h (lightgbm_24h: speed only)
# p (NPU precision): w8a16 (default) | fp16 | w8a8
# Set JSON=1 to also print one line "JSON:{...}" at the end (used by the app).

B=/data/local/tmp/nz_bench
RT=/data/local/tmp/nz_aimet
D=$B/data
RUN=$B/run
TEST_START=1762747200                     # first test row, 2025-11-10 04:00 UTC
CAP_S=535813.358399029; CAP_G=887250.7668833333; CAP_D=30045.697428571428
export LD_LIBRARY_PATH=$RT/lib:$B/ortlib
export ADSP_LIBRARY_PATH="$RT/lib;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
MODELS="baseline_h4 guided_h4 baseline_24h guided_24h cnn_lstm_24h"
mkdir -p $RUN

die() { echo "error: $*"; [ "$JSON" = 1 ] && echo "JSON:{\"ok\":false,\"error\":\"$*\"}"; exit 1; }
horizon() { case $1 in *_h4) echo 4 ;; *) echo 96 ;; esac; }
ntest() { [ "$(horizon $1)" = 4 ] && echo 4832 || echo 4740; }
label() { case $1 in w8a16) echo "NPU W8A16" ;; fp16) echo "NPU FP16" ;; w8a8) echo "NPU W8A8" ;; esac; }
floats() { od -A n -t f4 -v "$1" | tr -s ' \t' '\n\n' | grep -v '^$'; }     # binary float32 -> one per line
check_model() { case " $MODELS " in *" $1 "*) ;; *) die "unknown model '$1' (one of: $MODELS)" ;; esac; }
check_prec() { case $1 in w8a16|fp16|w8a8) ;; *) die "precision must be w8a16, fp16 or w8a8" ;; esac
               [ -f $B/qnn/$2/lib$1.so ] || die "no $1 build of $2 on the board"; }

# qnn <backend-lib> <model.so> <list> <n> <outdir>  ->  prints "<ms per inference> <npu compute ms or ->"
qnn() {
  rm -rf $5
  $RT/bin/qnn-net-run --backend $RT/lib/$1 --model $2 --input_list $3 --num_inferences $4 --keep_num_outputs 1 \
    --perf_profile burst --profiling_level basic --output_dir $5 > $5.log 2>&1 || { echo "- -"; return; }
  $RT/bin/qnn-profile-viewer --input_log $5/qnn-profiling-data_0.log 2>/dev/null | awk '
    /Execute Stats \(Average\)/ {a=1} /Execute Stats \(Min\)/ {a=0}
    a && /NetRun:/ && !n {n=$2} a && /Accelerator \(execute\) time/ && !x {x=$(NF-1)}
    END {printf "%.3f %s\n", n/1000, (x ? sprintf("%.3f", x/1000) : "-")}'
}
ort_ms() { $B/bin/ort_bench $1 1 latency $2 20 ${3:-300} | sed -n 's/.*mean_ms=\([0-9.]*\).*/\1/p'; }

cmd=$1; shift
case $cmd in

models)
  echo "model          horizon  test windows  NPU precisions on board"
  J=""
  for M in $MODELS; do
    P=""; for p in w8a16 fp16 w8a8; do [ -f $B/qnn/$M/lib$p.so ] && P="$P $p"; done
    printf "%-14s %4s     %6s       %s\n" $M "$(horizon $M)" "$(ntest $M)" "$P"
    J="$J{\"name\":\"$M\",\"horizon\":$(horizon $M),\"windows\":$(ntest $M),\"prec\":\"$(echo $P)\"},"
  done
  [ "$JSON" = 1 ] && echo "JSON:{\"ok\":true,\"models\":[${J%,}]}"
  ;;

forecast)
  M=$1; W=${2:-0}; P=${3:-w8a16}
  check_model $M; check_prec $P $M
  H=$(horizon $M); N=$(ntest $M)
  [ "$W" -ge 0 ] 2>/dev/null && [ "$W" -lt "$N" ] || die "window must be 0..$((N-1))"
  IN=$D/test/$(printf %05d $W).raw
  echo "$IN" > $RUN/list.txt
  T0=$((TEST_START + (W + 96) * 900 + 19800))
  WHEN=$(date -u -d @$T0 "+%Y-%m-%d %H:%M")
  echo "== $M, test window $W: forecast for the $H steps from $WHEN IST =="

  set -- $(qnn libQnnHtp.so $B/qnn/$M/lib$P.so $RUN/list.txt 50 $RUN/npu); NPU_MS=$1; NPU_X=$2
  set -- $(qnn libQnnCpu.so $B/qnn/$M/libfp32.so $RUN/list.txt 10 $RUN/cpu); CPU_MS=$1
  $B/bin/ort_bench $B/onnx/$M.onnx 1 outputs $RUN/list.txt $RUN/ort.bin > /dev/null
  ORT_MS=$(ort_ms $B/onnx/$M.onnx $IN 200)
  [ -f $RUN/npu/Result_0/*.raw ] || die "NPU run failed: $(tail -2 $RUN/npu.log | tr '\n"' '  ')"

  dd if=$D/y_test_h$H.bin bs=4 skip=$((W * H * 3)) count=$((H * 3)) 2>/dev/null > $RUN/y.bin
  floats $RUN/y.bin > $RUN/y.txt
  floats $RUN/npu/Result_0/*.raw > $RUN/npu.txt
  floats $RUN/cpu/Result_0/*.raw > $RUN/cpu.txt
  floats $RUN/ort.bin > $RUN/ort.txt
  paste $RUN/y.txt $RUN/npu.txt $RUN/cpu.txt $RUN/ort.txt | awk -v H=$H -v cs=$CAP_S -v cg=$CAP_G -v cd=$CAP_D \
      -v lab="$(label $P)" -v nms=$NPU_MS -v nx=$NPU_X -v cms=$CPU_MS -v oms=$ORT_MS -v json=$JSON \
      -v m=$M -v p=$P -v w=$W -v when="$WHEN" '
    { k = NR - 1; s = int(k / 3); d = k % 3; cap = (d == 0 ? cs : (d == 1 ? cg : cd))
      y[s,d] = $1 * cap; a[s,d] = ($2 < 0 ? 0 : $2) * cap; b[s,d] = ($3 < 0 ? 0 : $3) * cap; c[s,d] = ($4 < 0 ? 0 : $4) * cap
      ea[d] += (a[s,d] > y[s,d] ? a[s,d] - y[s,d] : y[s,d] - a[s,d]); ec[d] += (c[s,d] > y[s,d] ? c[s,d] - y[s,d] : y[s,d] - c[s,d])
      df = $2 - $4; if (df < 0) df = -df; if (df > mx) mx = df }
    END {
      split("Solar Grid Demand", dn, " ")
      printf "\nTime per forecast:  %s %s ms (NPU compute %s ms) | CPU FP32 (QNN) %s ms | CPU FP32 (ONNX Runtime) %s ms\n", lab, nms, nx, cms, oms
      printf "\n  ahead      | actual kW: solar   grid  demand | %-9s: solar   grid  demand | ONNX Rt: solar   grid  demand\n", lab
      n = split((H == 4 ? "1 2 3 4" : "1 4 24 48 96"), st, " ")
      for (i = 1; i <= n; i++) { s = st[i] - 1
        printf "  %5d min  | %16.1f %6.1f %7.2f | %16.1f %6.1f %7.2f | %14.1f %6.1f %7.2f\n", st[i] * 15,
          y[s,0]/1000, y[s,1]/1000, y[s,2]/1000, a[s,0]/1000, a[s,1]/1000, a[s,2]/1000, c[s,0]/1000, c[s,1]/1000, c[s,2]/1000 }
      printf "\nMean abs. error over the %d steps (kW):  ", H
      for (d = 0; d < 3; d++) printf "%s %s %.1f / ONNX Rt %.1f   ", dn[d+1], lab, ea[d]/H/1000, ec[d]/H/1000
      printf "\nLargest NPU vs FP32 difference: %.2f %% of capacity\n", 100 * mx
      if (json == 1) {
        printf "JSON:{\"ok\":true,\"model\":\"%s\",\"prec\":\"%s\",\"window\":%d,\"start\":\"%s IST\",\"H\":%d,", m, p, w, when, H
        printf "\"time_ms\":{\"npu\":%s,\"npu_compute\":%s,\"qnn_cpu\":%s,\"ort\":%s},\"max_diff_pct\":%.3f", nms, (nx == "-" ? "null" : nx), cms, oms, 100 * mx
        split("actual npu qnn_cpu ort", key, " ")
        for (q = 1; q <= 4; q++) { printf ",\"%s\":[", key[q]
          for (s = 0; s < H; s++) for (d = 0; d < 3; d++) {
            v = (q == 1 ? y[s,d] : (q == 2 ? a[s,d] : (q == 3 ? b[s,d] : c[s,d]))); printf "%s%.1f", (s || d ? "," : ""), v }
          printf "]" }
        printf "}\n" } }'
  ;;

speed)
  P=${1:-w8a16}
  echo "$(head -1 $D/test_list.txt)" | sed "s#^#$D/#" > $RUN/one.txt
  echo "Time per inference on the QIDK, one window, batch 1 (ms)"
  printf "%-14s %14s %16s %14s %16s\n" model "$(label $P)" "(NPU compute)" "CPU QNN FP32" "CPU ONNX Rt FP32"
  J=""
  for M in $MODELS; do
    if [ -f $B/qnn/$M/lib$P.so ]; then set -- $(qnn libQnnHtp.so $B/qnn/$M/lib$P.so $RUN/one.txt 200 $RUN/sp); else set -- - -; fi
    NM=$1; NX=$2
    set -- $(qnn libQnnCpu.so $B/qnn/$M/libfp32.so $RUN/one.txt 30 $RUN/sp); CM=$1
    OM=$(ort_ms $B/onnx/$M.onnx $D/test/00000.raw 300)
    printf "%-14s %14s %16s %14s %16s\n" $M $NM $NX $CM $OM
    J="$J{\"model\":\"$M\",\"npu\":\"$NM\",\"npu_compute\":\"$NX\",\"qnn_cpu\":\"$CM\",\"ort\":\"$OM\"},"
  done
  OM=$(ort_ms $B/onnx/lightgbm_24h.onnx $D/test_last_00000.raw 300)
  printf "%-14s %14s %16s %14s %16s\n" lightgbm_24h - - - $OM
  J="$J{\"model\":\"lightgbm_24h\",\"npu\":\"-\",\"npu_compute\":\"-\",\"qnn_cpu\":\"-\",\"ort\":\"$OM\"}"
  echo "A forecast is needed once every 15 minutes (900,000 ms)."
  [ "$JSON" = 1 ] && echo "JSON:{\"ok\":true,\"prec\":\"$P\",\"rows\":[$J]}"
  ;;

accuracy)
  M=$1; P=${2:-w8a16}; NW=${3:-500}
  check_model $M; check_prec $P $M
  H=$(horizon $M); N=$(ntest $M); [ "$NW" = all ] && NW=$N; [ "$NW" -gt "$N" ] && NW=$N
  head -$NW $D/test_list.txt | sed "s#^#$D/#" > $RUN/acc_list.txt
  echo "== $M: first $NW test windows on the $(label $P) vs the FP32 model (ONNX Runtime on the CPU) =="
  rm -rf $RUN/acc
  $RT/bin/qnn-net-run --backend $RT/lib/libQnnHtp.so --model $B/qnn/$M/lib$P.so --input_list $RUN/acc_list.txt \
    --perf_profile burst --output_dir $RUN/acc > $RUN/acc.log 2>&1 || die "NPU run failed"
  i=0; : > $RUN/acc_npu.bin
  while [ $i -lt $NW ]; do cat $RUN/acc/Result_$i/*.raw >> $RUN/acc_npu.bin; i=$((i + 1)); done
  $B/bin/ort_bench $B/onnx/$M.onnx 1 outputs $RUN/acc_list.txt $RUN/acc_ort.bin > /dev/null
  head -c $((NW * H * 3 * 4)) $D/y_test_h$H.bin > $RUN/acc_y.bin
  floats $RUN/acc_y.bin > $RUN/y.txt; floats $RUN/acc_npu.bin > $RUN/npu.txt; floats $RUN/acc_ort.bin > $RUN/ort.txt
  paste $RUN/y.txt $RUN/npu.txt $RUN/ort.txt | awk -v json=$JSON -v m=$M -v p=$P -v nw=$NW -v lab="$(label $P)" '
    { d = (NR - 1) % 3; t = $1; a = ($2 < 0 ? 0 : $2); c = ($3 < 0 ? 0 : $3)
      n[d]++; st[d] += t; st2[d] += t * t; ra[d] += (t - a) ^ 2; rc[d] += (t - c) ^ 2
      sig += $3 ^ 2; noi += ($3 - $2) ^ 2 }
    END { split("Solar Grid Demand", dn, " ")
      sq = 10 * log(sig / (noi > 0 ? noi : 1e-30)) / log(10)
      printf "\n  domain   R² %-10s R² FP32 (ONNX Rt)\n", lab
      J = ""
      for (d = 0; d < 3; d++) { tot = st2[d] - st[d] ^ 2 / n[d]
        if (tot > 1e-12) { r2a = sprintf("%.3f", 1 - ra[d] / tot); r2c = sprintf("%.3f", 1 - rc[d] / tot) }
        else { r2a = "n/a"; r2c = "n/a" }          # actual values constant (e.g. meter offline) -> R² undefined
        printf "  %-7s  %8s     %8s\n", dn[d+1], r2a, r2c
        J = J sprintf("%s{\"domain\":\"%s\",\"r2_npu\":\"%s\",\"r2_fp32\":\"%s\"}", (d ? "," : ""), dn[d+1], r2a, r2c) }
      printf "\nAgreement NPU vs FP32: %.1f dB SQNR (above ~40 dB = effectively identical)\n", sq
      if (json == 1) printf "JSON:{\"ok\":true,\"model\":\"%s\",\"prec\":\"%s\",\"windows\":%d,\"sqnr_db\":%.1f,\"rows\":[%s]}\n", m, p, nw, sq, J }'
  ;;

*)
  sed -n '2,15p' $0
  ;;
esac
exit 0
