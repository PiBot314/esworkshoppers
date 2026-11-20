// Self-test for the live loop's pure parts (plan_reduced_shortterm.md, tests A1 and A2):
//   A1: feature matrices built by features.ts == verify_api_lstm/verify.py's, from the same raw API files
//   A2: detect.ts flags == verify.py's flags, given verify.py's FP32 forecasts
// Run from qidk_export/dashboard after `DUMP=1 python verify.py <final> proxy` in verify_api_lstm:
//   bun scripts/live_selftest.ts
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { buildFrames, featureMatrix, actuals, istHour, STEP, type ModelMeta, type Reading } from '../src/lib/server/live/features.ts';
import { check } from '../src/lib/server/live/detect.ts';

const FINAL = resolve(import.meta.dir, '../../..');
const V = join(FINAL, 'verify_api_lstm');
const raw: Record<string, Reading[]> = {};
for (const f of readdirSync(join(V, 'raw')).sort()) {
	const node = f.replace(/_\d{4}-\d{2}-\d{2}\.json$/, '');
	for (const row of JSON.parse(readFileSync(join(V, 'raw', f), 'utf8'))) (raw[node] ??= []).push({ t: Date.parse(row.created_at) / 1000, row });
}
const { t0, n } = JSON.parse(readFileSync(join(V, 'grid.json'), 'utf8'));
const frames = buildFrames(raw, t0, n);

// --- A1 -------------------------------------------------------------------------------------
let ok = true;
const metas: Record<string, ModelMeta> = {};
for (const model of ['all_sensors_1h', 'single_node']) {
	const meta = (metas[model] = JSON.parse(readFileSync(join(FINAL, 'aimet_quantization/dashboard/www/data', model, 'meta.json'), 'utf8')));
	const X = featureMatrix(frames, meta, t0, n);
	writeFileSync(join(V, `X_${model}_ts.f64`), Buffer.from(X.buffer));
	console.log(`A1 ${model}: wrote ${n} x ${meta.features.length}`);
}

// --- A2 -------------------------------------------------------------------------------------
const res = JSON.parse(readFileSync(join(V, 'result_proxy.json'), 'utf8'));
const act = actuals(frames, n);
for (const [model, domain, key, name] of [['all_sensors_1h', 1, 'grid', 'Grid'], ['single_node', 0, 'pl00', 'PL00']] as const) {
	let same = 0, flagsTs = 0, flagsPy = 0, maxActDiff = 0;
	for (const [ts, , pred, actual, flags] of res[model] as [string, number, number[], number[], boolean[]][]) {
		const t = Date.parse(ts.replace(' ', 'T')) / 1000;
		const b = (t - t0) / STEP;
		const a = act[key][b];
		maxActDiff = Math.max(maxActDiff, Math.abs(a - actual[domain]));
		const c = check(metas[model], domain, istHour(t), a, pred[domain]);
		same += +(c.flagged === flags[domain]);
		flagsTs += +c.flagged; flagsPy += +flags[domain];
	}
	const N = res[model].length;
	console.log(`A2 ${name}: ${same}/${N} verdicts identical; flags TS ${flagsTs} / Python ${flagsPy}; max |actual diff| ${maxActDiff.toFixed(6)} W`);
	ok &&= same === N;
}
console.log(ok ? 'A2 PASS' : 'A2 FAIL');
