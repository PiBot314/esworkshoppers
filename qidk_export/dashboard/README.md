# Campus sensor dashboard

A map of every sensor node on the IIIT-H campus, a net-zero carbon balance with a what-if for extra solar, and an
assistant (a small language model running locally) that answers questions about the data.

**Its only data source is `campus.sqlite`** (`../../data/global_schema/campus.sqlite`), opened read-only.
The map background comes from OpenStreetMap tiles; nothing else is fetched.

## Run

```sh
bun install
bun run dev          # http://localhost:5173
```

The first page load takes about 2 seconds while the server counts messages per node; after that it is cached.

If the database lives somewhere else:

```sh
CAMPUS_DB=/path/to/campus.sqlite bun run dev
```

The scripts run Vite under the Bun runtime (`bun --bun`), because the server reads the database with
Bun's built-in `bun:sqlite`. Running Vite under Node will not work.

## The three parts

| Part | Where | What it does |
|---|---|---|
| **Map** | `/` | every node on a map; click one for its details, data and a chart |
| **Net zero** | `/whatif` | the campus's 2025 carbon balance (grid + diesel, minus trees), and what more solar would change |
| **Assistant** | button at the top right | a small language model running on this machine that answers questions by looking things up in the database |

## The assistant (small language model)

The assistant is a local model served by llama.cpp. Start it in a second terminal:

```sh
bun run slm          # serves the model on http://127.0.0.1:8090
```

It defaults to the Intel-GPU llama.cpp build in `~/ollama/llama-cpp-ipex-llm-…/llama-server` and the model
`~/ollama/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (2.5 GB, about 25 tokens/s on the Arc integrated GPU).
Override with `LLAMA_SERVER`, `SLM_MODEL`, `SLM_PORT`, `SLM_ARGS`. The first question after starting takes
longer while the model reads its instructions; later ones reuse that work.
The dashboard finds it through `SLM_URL` (default `http://127.0.0.1:8090/v1`). Any server that speaks the
OpenAI chat API with tool calling works, so the same dashboard can later point at a model on the QIDK.

The model never sees the database directly. It can only call seven tools (`src/lib/server/tools.ts`):
`data_overview`, `search_nodes`, `get_node`, `get_stats`, `compare_nodes`, `net_zero`, `show_node`. The chat shows every
call it made, so an answer can be checked against the lookup behind it. It is a small model and does make
mistakes: treat the lookups as the facts and its sentences as a summary of them.

## The net-zero page

`src/lib/server/whatif.ts`. It reads the substation's grid power (`EM-CR-SB00-00`, circuit GD), the generator
output (`EM-CR-SB00-01`) and three solar nodes from `campus.sqlite`, averages them to 15 minutes, and builds a
carbon balance for 2025:

- **Caused** = grid kWh x 0.7 kg/kWh + diesel kWh x 0.8 kg/kWh.
- **Removed** = what the campus trees absorb in a year, estimated from one year of trunk growth with the
  Chave et al. (2014) biomass formula. It is an estimate with a wide range; the page shows low, central and high.
- **Net** = caused - removed. Net zero means net <= 0.
- **More solar** is subtracted from the campus demand, following the measured solar pattern. Solar beyond the
  demand at that moment is not counted.

Every factor and assumption is in `ASSUMPTIONS` at the top of that file and is listed on the page.
Water pumping, sewage treatment and waste are not included: the database has no usable history for them.

## How to read the map

- **Colour and 2-letter code** = the node's vertical (EM, SL, WM …).
- **Fill** = how much real (measured) data the node has:
  filled = 1,000 or more messages, outlined = fewer, dashed = none.
- **`×N`** = N nodes share that exact position (usually a building-level coordinate). Click it to list them.

In the node panel, pick a source row to switch between real, derived and synthetic data. Synthetic data
is never shown unless you pick it. Times are in IST. Periods ("last 7 days") count back from that
source's last message, because the data is historical.

## Layout

| Path | What |
|---|---|
| `src/lib/server/db.ts` | every SQL query: node list, per-node coverage, node detail, time series |
| `src/routes/+page.server.ts` | loads the node list for the map |
| `src/routes/api/node/[id]/` | JSON for one node; `series/` for chart data |
| `src/lib/components/CampusMap.svelte` | Leaflet map and markers |
| `src/lib/components/NodePanel.svelte` | the panel for one node |
| `src/lib/components/LineChart.svelte` | the chart |
| `src/lib/server/whatif.ts`, `src/routes/whatif/` | the net-zero balance and its page |
| `src/lib/server/tools.ts`, `src/routes/api/chat/` | the assistant's tools and the chat endpoint |
| `src/lib/components/ChatPanel.svelte` | the assistant panel |
| `scripts/slm.sh` | starts the local language model |
| `src/lib/format.ts` | colours, number / date formatting, unit conversion |
