#!/usr/bin/env bash
# Start the small language model the dashboard's assistant talks to.
# It runs entirely on this machine with llama.cpp and serves an OpenAI-compatible API.
#
#   LLAMA_SERVER  path to the llama-server binary
#   SLM_MODEL     path to a GGUF model with tool-calling support
#   SLM_PORT      port (the dashboard expects 8090 unless SLM_URL is set)
#   SLM_ARGS      extra llama-server arguments, replacing the defaults below
#
# Default binary: the Intel-GPU (SYCL) build, which runs this model at about 25 tokens/s on the
# Arc integrated GPU. The newer OpenVINO build in ~/ai/llama.cpp manages about 5 tokens/s.
# The older build cannot parse Qwen3's own chat template, so a compatible Qwen tool-calling
# template is passed explicitly (scripts/qwen-tools.jinja, from the llama.cpp repository).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/ollama/llama-cpp-ipex-llm-2.3.0b20250430-ubuntu-core/llama-server}"
SLM_MODEL="${SLM_MODEL:-$HOME/ollama/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
SLM_PORT="${SLM_PORT:-8090}"
SLM_ARGS="${SLM_ARGS:--ngl 99 -t 4 --chat-template-file $HERE/qwen-tools.jinja}"

[ -x "$LLAMA_SERVER" ] || { echo "llama-server not found at $LLAMA_SERVER (set LLAMA_SERVER)"; exit 1; }
[ -f "$SLM_MODEL" ]    || { echo "model not found at $SLM_MODEL (set SLM_MODEL)"; exit 1; }

# shellcheck disable=SC2086
exec "$LLAMA_SERVER" -m "$SLM_MODEL" --host 127.0.0.1 --port "$SLM_PORT" \
	--jinja --ctx-size 8192 --parallel 1 $SLM_ARGS "$@"
