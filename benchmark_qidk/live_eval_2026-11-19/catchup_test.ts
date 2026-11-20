// Catch-up test: one normal cycle, then the clock jumps 24 h (96 missed bins) and one cycle runs the catch-up.
import { readFileSync, rmSync, mkdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { liveStatus, runCycleForTest } from '/home/radmeerkat62445/Documents/Uni/Sem3/EmbeddedSystemWorkshop/final/qidk_export/dashboard/src/lib/server/live/loop.ts';
import { STEP } from '/home/radmeerkat62445/Documents/Uni/Sem3/EmbeddedSystemWorkshop/final/qidk_export/dashboard/src/lib/server/live/features.ts';
process.env.LIVE = '1';
const DIR = process.env.LIVE_DIR!, API = process.env.LIVE_API!, NPU = process.env.NPU_URL!;
const { t0 } = JSON.parse(readFileSync('/home/radmeerkat62445/Documents/Uni/Sem3/EmbeddedSystemWorkshop/final/qidk_export/dashboard/../../verify_api_lstm/grid.json', 'utf8'));
const T0 = t0 + 300 * STEP, T1 = T0 + 96 * STEP;
await fetch(`${API}/_clock?t=${T0 + STEP + 120}`, { method: 'POST' });
let a = performance.now(); await runCycleForTest();
console.log('normal cycle', ((performance.now() - a) / 1000).toFixed(1), 's', liveStatus().npu);
await fetch(`${API}/_clock?t=${T1 + STEP + 120}`, { method: 'POST' });
await new Promise((r) => setTimeout(r, Number(process.env.IDLE ?? 0) * 1000)); // runners idle out first, as after a long fetch
a = performance.now(); await runCycleForTest();
console.log('catch-up cycle', ((performance.now() - a) / 1000).toFixed(1), 's', liveStatus().npu);
const sh = readFileSync(join(DIR, 'shadow.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
const c: Record<string, number> = {};
for (const r of sh) if (r.t > T0 && r.t <= T1) c[r.model] = (c[r.model] ?? 0) + 1;
console.log('shadow points in the gap', c);
console.log('runners now', await (await fetch(`${NPU}/api/warm`)).text());
process.exit(0);
