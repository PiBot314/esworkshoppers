// Turns raw API readings into model input windows, exactly like the training pipeline
// (aimet_quantization/scripts/01_rebuild_dataset.py + model2/generate_lstm_dataset.py) and
// verify_api_lstm/verify.py, which it is tested against (live_selftest.ts):
//   signed raw values -> 15-min UTC mean on one shared grid -> ffill(limit=4) + was_imputed
//   -> abs() on every column -> kW to W -> fillna(0) -> min-max with each feature's own range.
// Pure functions only: no I/O.

export const STEP = 900; // seconds per bin
const IST = 19800;

/** One API reading: unix seconds + the raw JSON row. */
export interface Reading { t: number; row: Record<string, unknown> }

const num = (v: unknown) => (typeof v === 'number' ? v : v == null || v === '' ? NaN : Number(v));
const pick = (r: Record<string, unknown>, k: string) => num(r[k]);

type Mapper = (r: Record<string, unknown>) => Record<string, number>;
const v3: Mapper = (r) => ({
	active_power_kW: pick(r, 'active_power_3'), reactive_power_kVAr: pick(r, 'reactive_power_3'),
	voltage_avg_V: pick(r, 'voltage_avg'), neutral_current_A: pick(r, 'nphaselc')
});
// API names the SR-EM fields correctly: the 2025 firmware transposition must NOT be re-applied.
const srem: Mapper = (r) => ({
	power_W: pick(r, 'power'), voltage_V: pick(r, 'voltage'), current_A: pick(r, 'current'), frequency_Hz: pick(r, 'frequency')
});

/** API node -> columns, signed, in the units the meter reports. The first column is the node's target. */
export const MAP: Record<string, Mapper> = {
	'EM-CR-SB00-00': (r) => ({
		gd_power_W: pick(r, 'gdpower'), t1_power_W: pick(r, 't1power'), t2_power_W: pick(r, 't2power'),
		hv_voltage_VLL: pick(r, 'gdvll'), lt1_voltage_VLN: pick(r, 'lt1vln'), lt2_voltage_VLN: pick(r, 'lt2vln')
	}),
	'EM-CR-KH95-00': (r) => ({
		power_total_W: pick(r, 'r_power') + pick(r, 'y_power') + pick(r, 'b_power'),
		power_R_W: pick(r, 'r_power'), power_B_W: pick(r, 'b_power'), voltage_V: pick(r, 'voltage'), power_factor: pick(r, 'power_factor')
	}),
	'SR-EM-KH04-01': srem,
	'SR-EM-KH04-02': srem,
	'SL-PL00-00': v3,
	// this meter reports W, not kW (same as 2025)
	'SL-NI01-00': (r) => ({
		active_power_W: pick(r, 'active_power_3'), reactive_power_VAr: pick(r, 'reactive_power_3'),
		voltage_avg_V: pick(r, 'voltage_avg'), neutral_current_A: pick(r, 'nphaselc')
	}),
	// active and reactive registers swapped on this meter (same rule as 2025)
	'SL-VN95-00': (r) => ({ ...v3(r), active_power_kW: pick(r, 'reactive_power_3'), reactive_power_kVAr: pick(r, 'active_power_3') })
};
export const NODES = Object.keys(MAP);

/** Per node: column name -> values on the grid (NaN = no data). */
export type Frames = Record<string, Record<string, Float64Array>>;

/** Build every served node on the grid t0, t0+STEP, ... (n bins). */
export function buildFrames(raw: Record<string, Reading[]>, t0: number, n: number): Frames {
	const frames: Frames = {};
	for (const node of NODES) {
		const rows = raw[node] ?? [];
		const mapped = rows.map((r) => ({ t: r.t, v: MAP[node](r.row) }));
		const cols = Object.keys(MAP[node]({}));
		const target = cols[0];
		const out: Record<string, Float64Array> = {};
		let missing: Uint8Array | undefined;
		for (const c of cols) {
			const sum = new Float64Array(n), cnt = new Float64Array(n);
			for (const { t, v } of mapped) {
				const b = Math.floor((t - t0) / STEP);
				if (b < 0 || b >= n) continue;
				const x = v[c];
				if (Number.isFinite(x)) { sum[b] += x; cnt[b]++; }
			}
			const col = new Float64Array(n);
			for (let b = 0; b < n; b++) col[b] = cnt[b] ? sum[b] / cnt[b] : NaN;
			if (c === target) missing = col.map((x) => (Number.isNaN(x) ? 1 : 0)) as unknown as Uint8Array;
			// ffill(limit=4)
			let last = NaN, run = 0;
			for (let b = 0; b < n; b++) {
				if (Number.isNaN(col[b])) {
					run++;
					if (!Number.isNaN(last) && run <= 4) col[b] = last;
				} else { last = col[b]; run = 0; }
			}
			out[c] = col;
		}
		const imp = new Float64Array(n);
		for (let b = 0; b < n; b++) imp[b] = missing![b] && !Number.isNaN(out[target][b]) ? 1 : 0;
		for (const c of cols) out[c] = out[c].map(Math.abs);
		for (const [kw, w] of [['active_power_kW', 'active_power_W'], ['reactive_power_kVAr', 'reactive_power_VAr']] as const) {
			if (out[kw]) { out[w] = out[kw].map((x) => x * 1000); delete out[kw]; }
		}
		out.was_imputed = imp;
		frames[node] = out;
	}
	return frames;
}

export interface Feature { name: string; node: string; col: string; min: number; max: number; cyclic: boolean }
export interface ModelMeta {
	model: string; domains: string[]; horizon: number; lookback: number; features: Feature[];
	model_capacities_W: number[]; anomaly: { K: number; floor_frac: number; median_W: number[][]; limit_W: number[][] };
}

/** Inputs the API doesn't serve: which ones are estimated from a neighbour, and how. */
const NEIGHBOUR: Record<string, string> = { 'SL-NI03-00': 'SL-NI01-00', 'SL-NI03-01': 'SL-NI01-00', 'SL-VN02-01': 'SL-VN95-00' };
export const ESTIMATED_INPUTS = ['SL-NI03-00, SL-NI03-01 (copied from SL-NI01-00)', 'SL-VN02-01 (copied from SL-VN95-00)', 'SR-EM-KH04-00 (0)'];

function cyclic(t: number, col: string) {
	const d = new Date((t + IST) * 1000);
	const hod = d.getUTCHours() + d.getUTCMinutes() / 60;
	const dow = (d.getUTCDay() + 6) % 7; // Monday = 0, as pandas
	const doy = Math.floor((Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()) - Date.UTC(d.getUTCFullYear(), 0, 1)) / 86400000) + 1;
	const [name, fn] = col.split('_') as [string, 'sin' | 'cos'];
	const [val, period] = name === 'hod' ? [hod, 24] : name === 'dow' ? [dow, 7] : [doy, 365.25];
	return Math[fn]((2 * Math.PI * val) / period);
}

/** Feature matrix [n x F] (row-major, float64) for one model, before windowing. */
export function featureMatrix(frames: Frames, meta: ModelMeta, t0: number, n: number): Float64Array {
	const F = meta.features.length;
	const X = new Float64Array(n * F);
	const cap = (name: string) => meta.features.find((f) => f.name === name)!.max;
	meta.features.forEach((f, j) => {
		let src: (b: number) => number;
		if (f.cyclic) src = (b) => cyclic(t0 + b * STEP, f.col);
		else if (frames[f.node]?.[f.col]) { const c = frames[f.node][f.col]; src = (b) => c[b]; }
		else if (f.col === 'active_power_W' && NEIGHBOUR[f.node]) {
			const peer = NEIGHBOUR[f.node], c = frames[peer].active_power_W;
			const k = cap(`${f.node}__active_power_W`) / cap(`${peer}__active_power_W`);
			src = (b) => c[b] * k;
		} else if (f.col === 'ac_voltage_avg_V') { const c = frames['SL-NI01-00'].voltage_avg_V; src = (b) => c[b]; }
		else src = () => 0;
		const span = f.max - f.min;
		for (let b = 0; b < n; b++) {
			const x = src(b);
			X[b * F + j] = ((Number.isNaN(x) ? 0 : x) - f.min) / span;
		}
	});
	return X;
}

/** The 96-step window ending at bin `end` (inclusive), as float32 for the NPU. */
export function windowAt(X: Float64Array, F: number, end: number, lookback = 96): Float32Array {
	return new Float32Array(X.subarray((end - lookback + 1) * F, (end + 1) * F));
}

/** Actual readings (W) used by the anomaly check, per bin. NaN where the substation has no data. */
export function actuals(frames: Frames, n: number) {
	const z = (x: number) => (Number.isNaN(x) ? 0 : x);
	const gd = frames['EM-CR-SB00-00'].gd_power_W, kh = frames['EM-CR-KH95-00'].power_total_W, pl = frames['SL-PL00-00'].active_power_W;
	const grid = new Float64Array(n), pl00 = new Float64Array(n);
	for (let b = 0; b < n; b++) {
		grid[b] = Number.isNaN(gd[b]) ? NaN : gd[b] + z(kh[b]);
		pl00[b] = z(pl[b]);
	}
	return { grid, pl00 };
}

export const istHour = (t: number) => new Date((t + IST) * 1000).getUTCHours();
