#!/usr/bin/env bash
# Builds the Android app (dashboard/bin/nz_demo.apk) without Gradle: aapt2 → javac → d8 → zipalign → apksigner.
#   dashboard/android/build.sh            build
#   dashboard/android/build.sh --install  build and install on the connected device (SERIAL, default the QIDK)
# Needs: Android SDK (platforms;android-35, build-tools;36.0.0) in $ANDROID_HOME or ~/Android/Sdk, and a JDK ≥ 17
# (only the jdk.compiler module is used, so a JRE with jdk.compiler also works).
# The app bundles dashboard/www (app.js, style.css, uPlot, data/) plus the phone layout in android/web/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DASH="$(cd "$HERE/.." && pwd)"
SDK="${ANDROID_HOME:-$HOME/Android/Sdk}"
BT="$SDK/build-tools/${BUILD_TOOLS:-36.0.0}"
JAR="$SDK/platforms/${PLATFORM:-android-35}/android.jar"
OUT="$HERE/build"
APK="$DASH/bin/nz_demo.apk"
for f in "$BT/aapt2" "$BT/d8" "$JAR"; do [ -e "$f" ] || { echo "missing $f (install with sdkmanager)"; exit 1; }; done

rm -rf "$OUT"; mkdir -p "$OUT/assets/www" "$OUT/classes" "$OUT/dex"

# 1. web assets: shared dashboard files, then the mobile page on top
cp -r "$DASH/www/app.js" "$DASH/www/style.css" "$DASH/www/vendor" "$DASH/www/data" "$OUT/assets/www/"
cp "$HERE/web/"* "$OUT/assets/www/"

# 2. resources + manifest + assets (recorded .f32 arrays stored uncompressed for fast loading)
"$BT/aapt2" compile --dir "$HERE/res" -o "$OUT/res.zip"
"$BT/aapt2" link -o "$OUT/base.apk" -I "$JAR" --manifest "$HERE/AndroidManifest.xml" \
  -A "$OUT/assets" -0 f32 "$OUT/res.zip"

# 3. Java → dex
JAVAC=(java -m jdk.compiler/com.sun.tools.javac.Main)
command -v javac >/dev/null && JAVAC=(javac)
"${JAVAC[@]}" -source 11 -target 11 -Xlint:-options -encoding UTF-8 -classpath "$JAR" -d "$OUT/classes" \
  $(find "$HERE/src" -name '*.java')
"$BT/d8" --release --min-api 26 --lib "$JAR" --output "$OUT/dex" $(find "$OUT/classes" -name '*.class')

# 4. package, align, sign (local debug key, created once)
cp "$OUT/base.apk" "$OUT/unsigned.apk"
(cd "$OUT/dex" && zip -q "$OUT/unsigned.apk" classes.dex)
"$BT/zipalign" -f -p 4 "$OUT/unsigned.apk" "$OUT/aligned.apk"
KS="$HERE/debug.keystore"
[ -f "$KS" ] || keytool -genkeypair -keystore "$KS" -storepass android -keypass android -alias nzdemo \
  -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=Net-Zero NPU demo" >/dev/null 2>&1
mkdir -p "$DASH/bin"
"$BT/apksigner" sign --ks "$KS" --ks-pass pass:android --key-pass pass:android --out "$APK" "$OUT/aligned.apk"
rm -f "$APK.idsig"
echo "built $APK ($(du -h "$APK" | cut -f1))"

if [ "${1:-}" = "--install" ]; then
  adb -s "${SERIAL:-d42f06b}" install -r "$APK"
  adb -s "${SERIAL:-d42f06b}" shell am start -n dev.netzero.qidk/.MainActivity >/dev/null
fi
