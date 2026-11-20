#!/usr/bin/env bash
# Build llama-server (and llama-bench, llama-cli) for the QIDK: aarch64 Android, CPU backend,
# tuned for the SM8650's Cortex-X4/A720 (dotprod + i8mm). Static, so only the binaries need pushing.
#   ANDROID_NDK_ROOT  default ~/Android/Sdk/ndk/26.2.11394342 (r26c, copied from the qairt-fixed:2.47 image)
# Output: nz_live/build/llama/bin/{llama-server,llama-bench,llama-cli}
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
NDK="${ANDROID_NDK_ROOT:-$HOME/Android/Sdk/ndk/26.2.11394342}"
SRC="$HERE/third_party/llama.cpp"          # git tag b11936 (2026-10-27)
OUT="$HERE/build/llama"
cmake -S "$SRC" -B "$OUT/cmake" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm \
  -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF \
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++"
cmake --build "$OUT/cmake" --target llama-server llama-bench llama-cli -j"$(nproc)"
mkdir -p "$OUT/bin"
cp "$OUT/cmake/bin/llama-server" "$OUT/cmake/bin/llama-bench" "$OUT/cmake/bin/llama-cli" "$OUT/bin/"
"$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" "$OUT"/bin/*
ls -la "$OUT/bin"
