#!/usr/bin/env bash
# Fingerprint the project's model binaries (not tracked by git, see .gitignore) into MODELS_SHA256.txt, which IS
# tracked: each commit then records exactly which model files were current. Rerun after training/exporting/building.
#   ./make_model_manifest.sh            -> rewrite MODELS_SHA256.txt
#   sha256sum -c MODELS_SHA256.txt      -> check the files on disk still match
# Scope: our trained/exported models and NPU/board builds. Not included: vendor runtimes (QAIRT libs), the LLM
# weights in qidk_export/models (third-party, unchanged), AIMET work outputs in aimet_quantization/out (intermediates;
# the deliverables are in aimet_quantization/release, which has its own SHA256SUMS).
set -euo pipefail
cd "$(dirname "$0")"
DIRS=(common model2 cnn_lstm_hybrid forecast_2025/models forecast_2025/fused benchmark_qidk/models)
{
  echo "# sha256  path   (generated $(date -u +%Y-%m-%dT%H:%MZ) by make_model_manifest.sh)"
  find "${DIRS[@]}" -type f \( -name '*.onnx' -o -name '*.onnx.data' -o -name '*.pt' -o -name '*.pth' -o -name '*.pkl' \
       -o -name '*.so' -o -name '*.dlc' -o -name '*_lgbm.txt' \) -not -path '*/__pycache__/*' -print0 \
    | sort -z | xargs -0 sha256sum
} > MODELS_SHA256.txt
echo "$(grep -vc '^#' MODELS_SHA256.txt) model files -> MODELS_SHA256.txt"
