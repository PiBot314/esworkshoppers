#!/usr/bin/env bash
# Cross-compile tools/ort_bench.cpp and tools/vphmon.c for the QIDK with the NDK in the qairt image.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
B="$(cd "$HERE/.." && pwd)"
mkdir -p "$B/tools/bin"
docker run --rm --user "$(id -u):$(id -g)" -v "$B:/b:z" qairt-fixed:2.47 bash -c '
  source $QAIRT_SDK_ROOT/bin/envsetup.sh >/dev/null 2>&1
  TC=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin
  ORT=/b/third_party/ort
  $TC/aarch64-linux-android31-clang++ -O2 -std=c++17 -I$ORT/headers /b/tools/ort_bench.cpp \
      -L$ORT/jni/arm64-v8a -lonnxruntime -static-libstdc++ -o /b/tools/bin/ort_bench
  $TC/aarch64-linux-android31-clang -O2 /b/tools/vphmon.c -lm -o /b/tools/bin/vphmon
'
file "$B"/tools/bin/*
