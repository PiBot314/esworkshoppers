Dashboard + SLM bundle for the QIDK

  dashboard/   SvelteKit app (bun install, then bun --bun run dev)
  data/        campus.sqlite (set CAMPUS_DB=$PWD/data/campus.sqlite)
  models/      Qwen3-4B-Instruct-2507-Q4_K_M.gguf (set SLM_MODEL=$PWD/models/...)

Not included, install on the board:
  bun (aarch64 build)
  llama-server built for aarch64 (set LLAMA_SERVER). The x86 SYCL build on the laptop will not run on the QIDK.
  Drop "-ngl 99" via SLM_ARGS if the build has no GPU backend, keeping --chat-template-file dashboard/scripts/qwen-tools.jinja.

Run:
  cd dashboard && bun install
  CAMPUS_DB=../data/campus.sqlite bun --bun run dev --host   (terminal 1)
  SLM_MODEL=../models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf LLAMA_SERVER=/path/to/llama-server SLM_ARGS="-t 4 --chat-template-file scripts/qwen-tools.jinja" bash scripts/slm.sh   (terminal 2)
