#!/usr/bin/env bash
# B5 prep: package a graph that passed in the x86 HTP emulator for the QIDK, so deploying is one push when the board
# is connected (20 Nov). Same conversion as c1_emulator.py's 'perrow' build (W8A16, bias 32, per-row weights + bias,
# --expand_lstm_op_structure --preserve_io layout), same calibration list as that graph's emulator run, but
# qnn-model-lib-generator -t aarch64-android (as aimet_quantization/scripts/08_qairt_build.sh).
#
#   forecast_2025/b5_board_package.sh STEM          (STEM = fused/<STEM>.onnx, after c1_emulator.py ran on it)
#
# -> forecast_2025/fused/<STEM>_board/
#      lib<STEM>.so          model library for the HTP backend (aarch64-android)
#      test/  test_list.txt  the emulator's test windows, paths relative to the device dir
#      run_on_board.sh       push + qnn-net-run on the HTP (burst, basic profiling) + pull (as 09_run_on_qidk.sh)
#      then: python3 forecast_2025/b5_score_board.py STEM   -> board SQNR vs FP32, vs the emulator, and timing
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FINAL="$(cd "$HERE/.." && pwd)"
IMAGE="${IMAGE:-qairt-fixed:2.47}"
S=$1
EMU="$HERE/fused/${S}_emu"
B="$HERE/fused/${S}_board"
[ -f "$EMU/raw_cal.txt" ] || { echo "run c1_emulator.py on fused/$S.onnx first"; exit 1; }
if [ ! -d "$EMU/raw_cal" ] || [ ! -d "$EMU/raw_test" ]; then   # moved out on 20 Nov (workspace clean-up)
  echo "the emulator's input windows are not in $EMU; restore them with"
  echo "  mv archive/forecast_2025_emulator_work/${S}_emu/raw_{cal,test} forecast_2025/fused/${S}_emu/"
  echo "or rerun c1_emulator.py (same windows: they are seeded by model file name)"; exit 1
fi
mkdir -p "$B/test"
DIMS=$(python3 - "$HERE/fused/$S.onnx" <<'EOF'
import sys, onnx
m = onnx.load(sys.argv[1])
print(" ".join("--input_dim %s 1,%s" % (i.name, ",".join(str(d.dim_value) for d in i.type.tensor_type.shape.dim[1:]))
               for i in m.graph.input))
EOF
)
REL=${B#$FINAL/}; RELEMU=${EMU#$FINAL/}
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$FINAL:/local:z" "$IMAGE" bash -c "
  source /usr/venv/bin/activate; source \$QAIRT_SDK_ROOT/bin/envsetup.sh >/dev/null
  set -eo pipefail
  export PATH=\$ANDROID_NDK_ROOT:\$PATH
  cd /local/$REL
  qnn-onnx-converter --input_network /local/forecast_2025/fused/$S.onnx --expand_lstm_op_structure --preserve_io layout \
     $DIMS --input_list /local/$RELEMU/raw_cal.txt --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32 \
     --use_per_row_quantization --enable_per_row_quantized_bias --output_path $S.cpp > conv.log 2>&1
  qnn-model-lib-generator -c $S.cpp -b $S.bin -o lib -t aarch64-android > libgen.log 2>&1
  cp lib/aarch64-android/lib$S.so .
  echo built lib$S.so
"
# test windows: same files the emulator used; device paths relative to the run directory
cp "$EMU"/raw_test/*.raw "$B/test/"
sed "s#/local/$RELEMU/raw_test/#$S/test/#g" "$EMU/raw_test.txt" > "$B/test_list.txt"
cat > "$B/run_on_board.sh" <<EOF
#!/usr/bin/env bash
# Push lib$S.so + test windows to the QIDK, run on the HTP, pull outputs (needs the QAIRT 2.47 runtime in \$D/{bin,lib},
# as for aimet_quantization/scripts/09_run_on_qidk.sh).
set -euo pipefail
SERIAL="\${SERIAL:-d42f06b}"; D=/data/local/tmp/nz_aimet
HERE="\$(cd "\$(dirname "\$0")" && pwd)"
adb -s \$SERIAL shell "mkdir -p \$D/$S"
adb -s \$SERIAL push "\$HERE/test" "\$D/$S/" | tail -1
adb -s \$SERIAL push "\$HERE/test_list.txt" "\$HERE/lib$S.so" "\$D/$S/" > /dev/null
adb -s \$SERIAL shell "cd \$D && rm -rf $S/out && export LD_LIBRARY_PATH=\$D/lib ADSP_LIBRARY_PATH=\$D/lib && \\
  ./bin/qnn-net-run --backend lib/libQnnHtp.so --model $S/lib$S.so --input_list $S/test_list.txt \\
    --output_dir $S/out --perf_profile burst --profiling_level basic > $S/run.log 2>&1"
rm -rf "\$HERE/device_out"
adb -s \$SERIAL pull "\$D/$S/out" "\$HERE/device_out" > /dev/null
adb -s \$SERIAL pull "\$D/$S/run.log" "\$HERE/" > /dev/null
echo "pulled -> \$HERE/device_out; score: python3 forecast_2025/b5_score_board.py $S"
EOF
chmod +x "$B/run_on_board.sh"
ls -la "$B" | head -8
