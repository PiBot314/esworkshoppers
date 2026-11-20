// Replay test for the live loop (plan_reduced_shortterm.md, test A2): runs the REAL loop.ts, cycle by cycle,
// through the cached API days on a fake clock, and checks every flag against verify_api_lstm/verify.py.
//   1. ../../project/.venv/bin/python ../../verify_api_lstm/replay_server.py <final> 9998     (fake API + FP32 models)
//   2. LIVE_API=http://127.0.0.1:9998 NPU_URL=http://127.0.0.1:9998 LIVE_DIR=<tmp> bun scripts/live_replay.ts [bins] [inject]
// With NPU_URL pointing at the board's nz_dashboard instead (adb forward tcp:8080 tcp:8080), it checks the NPU path:
// the verdicts may then differ from verify.py's FP32 ones on readings close to the limit.
// `inject` adds 10 dropouts on PL00 (daytime) and 10 +50 % spikes on Grid, and reports how many were caught.
import { mkdirSync, readFileSync, rmSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { liveStatus, requestInject, runCycleForTest, type LiveEvent, type Signal } from '../src/lib/server/live/loop.ts';
import { istHour, STEP } from '../src/lib/server/live/features.ts';

process.env.LIVE = '1';
const DIR = process.env.LIVE_DIR!;
rmSync(join(DIR, 'events.jsonl'), { force: true });
mkdirSync(DIR, { recursive: true });
const API = process.env.LIVE_API!;
const V = join(resolve(import.meta.dir, '../../..'), 'verify_api_lstm');
const { t0, n } = JSON.parse(readFileSync(join(V, 'grid.json'), 'utf8'));
const res = JSON.parse(readFileSync(join(V, 'result_proxy.json'), 'utf8'));
const bins = Number(process.argv[2] ?? n);
const doInject = process.argv[3] === 'inject';

// verify.py's verdicts per (signal, bin start)
const py = new Map<string, boolean>();
for (const [model, d, sig] of [['all_sensors_1h', 1, 'grid'], ['single_node', 0, 'pl00']] as const)
	for (const [ts, , , , flags] of res[model] as [string, number, number[], number[], boolean[]][])
		py.set(`${sig}@${Date.parse(ts.replace(' ', 'T')) / 1000}`, flags[d]);

const first = t0 + 96 * STEP; // first bin with a full window before it
const last = Math.min(t0 + (n - 1) * STEP, first + (bins - 1) * STEP);
const injected = new Map<string, string>();
let k = 0;
const started = performance.now();
for (let T = first; T <= last; T += STEP, k++) {
	const h = istHour(T);
	const sig: Signal = injected.size % 2 ? 'grid' : 'pl00';
	if (doInject && injected.size < 20 && k % 37 === 0 && (sig === 'grid' || (h >= 9 && h <= 15))) {
		const kind = sig === 'pl00' ? 'dropout' : 'spike';
		requestInject(sig, kind, 1);
		injected.set(`${sig}@${T}`, kind);
	}
	await fetch(`${API}/_clock?t=${T + STEP + 120}`, { method: 'POST' });
	await runCycleForTest();
	const s = liveStatus();
	if (!s.api.ok || !s.npu.ok) { console.log('cycle failed at', new Date(T * 1000).toISOString(), s.api, s.npu); process.exit(1); }
	if (k % 96 === 95) console.log(`${s.last_reading_checked}: ${k + 1} bins, ${((performance.now() - started) / 1000).toFixed(0)} s`);
}

// from the file: the loop keeps only 7 days of events in memory, the replay covers 15
const events = readFileSync(join(DIR, 'events.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l) as LiveEvent);
const tsFlags = new Set(events.filter((e) => e.flagged && !e.injected).map((e) => `${e.signal}@${e.t}`));
for (const sig of ['grid', 'pl00'] as Signal[]) {
	let both = 0, onlyPy = 0, onlyTs = 0, notScored = 0;
	for (let T = first; T <= last; T += STEP) {
		const key = `${sig}@${T}`;
		if (injected.has(key)) continue;
		const p = py.get(key), t = tsFlags.has(key);
		if (p === undefined) { notScored += +t; continue; }
		if (p && t) both++; else if (p) onlyPy++; else if (t) onlyTs++;
	}
	console.log(`${sig}: flags in both ${both}, only verify.py ${onlyPy}, only live loop ${onlyTs}; loop flags on bins verify.py skipped: ${notScored}`);
}
if (doInject) {
	for (const sig of ['grid', 'pl00'] as Signal[]) {
		const mine = [...injected].filter(([key]) => key.startsWith(sig));
		const caught = mine.filter(([key]) => events.some((e) => `${e.signal}@${e.t}` === key && e.injected && e.flagged)).length;
		console.log(`injected ${sig}: ${caught}/${mine.length} caught`);
	}
}
const npuMs = events.map((e) => e.npu_ms).filter((x): x is number => x != null);
if (npuMs.length) console.log(`NPU execute ms (median of flagged events): ${npuMs.sort((a, b) => a - b)[npuMs.length >> 1]}`);
console.log(`${k} cycles in ${((performance.now() - started) / 1000).toFixed(0)} s`);
process.exit(0);
