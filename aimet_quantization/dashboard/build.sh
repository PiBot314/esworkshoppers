#!/usr/bin/env bash
# Build nz_dashboard for the QIDK (aarch64 Android, API 31) with the NDK inside the qairt image.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
docker run --rm --user "$(id -u):$(id -g)" -v "$HERE:/d:z" "${IMAGE:-qairt-fixed:2.47}" bash -c '
  CXX=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++
  $CXX -std=c++17 -O2 -static-libstdc++ -Wall -Wno-unused-function -o /d/bin/nz_dashboard /d/server/nz_dashboard.cpp'
ls -la "$HERE/bin/nz_dashboard"
