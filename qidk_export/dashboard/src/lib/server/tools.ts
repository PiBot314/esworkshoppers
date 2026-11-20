// The tools the language model may call. Each one reads campus.sqlite (through db.ts /
// whatif.ts) and returns a small JSON object: small on purpose, the model has a short memory.
import { all, channelsOf, getCoverage, getNodeDetail, getNodes, getSources } from '#lib/server/db.ts';
import { runWhatIf, summariseWhatIf } from '#lib/server/whatif.ts';
import type { Channel } from '#lib/types.ts';
import { activeModelsInfo, enabled as liveEnabled, eventById, listModels, liveStatus, recentEvents, resultsBundle, SIGNALS, type LiveEvent } from '#lib/server/live/loop.ts';

const IST = 19800;
const round = (v: number | null) => (v == null ? null : Math.abs(v) >= 100 ? Math.round(v) : Math.round(v * 1000) / 1000);
const istDate = (ts: number) => new Date((ts + IST) * 1000).toISOString().slice(0, 10);
const istTime = (ts: number) => new Date((ts + IST) * 1000).toISOString().slice(0, 16).replace('T', ' ') + ' IST';
const parseDay = (s: unknown, end = false) => {
	if (typeof s !== 'string' || !/^\d{4}-\d{2}(-\d{2})?$/.test(s)) return null;
	const d = s.length === 7 ? (end ? new Date(Date.UTC(+s.slice(0, 4), +s.slice(5, 7), 1)) : new Date(s + '-01T00:00:00Z')) : new Date(s + 'T00:00:00Z');
	return Math.floor(d.getTime() / 1000) - IST + (end && s.length === 10 ? 86400 : 0);
};

function findNode(id: unknown) {
	const q = String(id ?? '').trim().toUpperCase();
	const nodes = getNodes();
	const hit = nodes.find((n) => n.node_id.toUpperCase() === q);
	if (hit) return { node: hit };
	const near = nodes.filter((n) => q && n.node_id.toUpperCase().includes(q)).slice(0, 5).map((n) => n.node_id);
	return { error: `No node called "${id}".`, did_you_mean: near.length ? near : undefined, hint: 'Use search_nodes to find the right ID.' };
}

const LEVEL = { history: 'a year of real data', few: 'only a few real points', none: 'no real data' } as const;
const describe = (n: ReturnType<typeof getNodes>[number]) => ({
	id: n.node_id,
	type: n.subtype_name ? `${n.vertical_name} / ${n.subtype_name}` : n.vertical_name,
	place: [n.site_name, n.floor].filter(Boolean).join(', ') || 'unknown',
	real_data: LEVEL[n.level],
	real_messages: n.measured
});

/** Pick the source to read: the named one, else the measured source with the most messages. */
function pickSource(nodeId: string, wanted?: unknown) {
	const cov = (getCoverage().get(nodeId) ?? []).filter((c) => c.payload_schema_id != null);
	const w = String(wanted ?? '').toLowerCase();
	const named = w && cov.filter((c) => c.source_code === w || c.source_kind === w).sort((a, b) => b.n - a.n)[0];
	return named || cov.filter((c) => c.source_kind === 'measured').sort((a, b) => b.n - a.n)[0] || cov.sort((a, b) => b.n - a.n)[0];
}

/** General words people use, mapped to the quantity they almost always mean. */
const SYNONYM: Record<string, string> = { power: 'active_power', load: 'active_power', demand: 'active_power', voltage: 'voltage_ln', energy: 'active_energy', generation: 'active_power' };

function pickChannel(channels: Channel[], wanted: unknown) {
	const norm = (s: string) => s.toLowerCase().replace(/[\s_-]+/g, '');
	const w = norm(String(wanted ?? ''));
	const numeric = channels.filter((c) => c.value_kind !== 'text' && c.value_kind !== 'time');
	// When several channels fit a general word, prefer the whole-device total over a single phase.
	const total = (list: Channel[]) => list.find((c) => c.phase == null || c.phase === '3P') ?? list[0] ?? null;
	return (
		numeric.find((c) => norm(c.raw_name) === w) ??
		total(numeric.filter((c) => c.quantity === SYNONYM[w])) ??
		total(numeric.filter((c) => norm(c.quantity) === w || norm(c.quantity_name) === w)) ??
		total(numeric.filter((c) => norm(c.raw_name).includes(w) || w.includes(norm(c.raw_name)))) ??
		total(numeric.filter((c) => norm(c.quantity).includes(w) || norm(c.quantity_name).includes(w)))
	);
}
const unit = (c: Channel) => (c.scale != null ? (c.canonical_unit ?? '') : `unverified (labelled ${c.raw_unit ?? '?'})`);

type Args = Record<string, unknown>;
export interface ToolOutput { result: unknown; ui?: { action: 'show_node'; node_id: string } | { action: 'what_if'; href: string } }

const impl: Record<string, (a: Args) => ToolOutput> = {
	data_overview() {
		const nodes = getNodes();
		const by: Record<string, { name: string; nodes: number; with_year_of_real_data: number; with_no_real_data: number }> = {};
		for (const n of nodes) {
			const v = (by[n.vertical] ??= { name: n.vertical_name, nodes: 0, with_year_of_real_data: 0, with_no_real_data: 0 });
			v.nodes++;
			if (n.level === 'history') v.with_year_of_real_data++;
			if (n.level === 'none') v.with_no_real_data++;
		}
		const totals = new Map<number, number>();
		for (const list of getCoverage().values()) for (const c of list) totals.set(c.source_id, (totals.get(c.source_id) ?? 0) + c.n);
		return {
			result: {
				total_nodes: nodes.length,
				node_types: by,
				nodes_with_a_year_of_real_data: nodes.filter((n) => n.level === 'history').map((n) => n.node_id),
				data_sources: getSources().map((s) => ({ name: s.code, kind: s.kind, messages: totals.get(s.id) ?? 0 })),
				note: 'This database is historical, mostly the year 2025. For live data use live_status, recent_anomalies and explain_anomaly. "synthetic" sources are generated, not measured.'
			}
		};
	},

	search_nodes(a) {
		const q = String(a.query ?? '').toLowerCase().trim();
		const v = String(a.type ?? '').toUpperCase().trim();
		const words = q.split(/\s+/).filter(Boolean);
		const hits = getNodes().filter((n) => {
			if (v && n.vertical !== v) return false;
			if (a.only_with_real_history && n.level !== 'history') return false;
			const text = `${n.node_id} ${n.vertical_name} ${n.subtype_name ?? ''} ${n.site_name ?? ''} ${n.floor ?? ''}`.toLowerCase();
			return words.every((w) => text.includes(w));
		});
		hits.sort((x, y) => y.measured - x.measured);
		return { result: { matches: hits.length, showing: Math.min(hits.length, 10), nodes: hits.slice(0, 10).map(describe), note: 'Sorted by number of messages, which says nothing about how large the values are.' } };
	},

	get_node(a) {
		const f = findNode(a.node_id);
		if (!f.node) return { result: f };
		const d = getNodeDetail(f.node.node_id)!;
		const src = pickSource(f.node.node_id);
		const channels = src ? (d.channels[src.source_id] ?? []) : d.declared;
		const latest = src ? d.latest[src.source_id] : null;
		const byId = new Map(channels.map((c) => [c.id, c]));
		return {
			result: {
				...describe(f.node),
				device: d.node.device,
				position_quality: f.node.location_quality,
				data: d.coverage.map((c) => ({ source: c.source_code, kind: c.source_kind, messages: c.n, from: istDate(c.first_ts), to: istDate(c.last_ts) })),
				values_it_reports: channels.slice(0, 40).map((c) => `${c.raw_name} (${c.quantity_name}${c.scale != null && c.canonical_unit ? ', ' + c.canonical_unit : ''})`),
				last_message: latest && latest.values.length
					? { source: src!.source_code, time: istTime(latest.ts), values: Object.fromEntries(latest.values.slice(0, 14).flatMap((v) => {
							const c = byId.get(v.channel_id);
							if (!c) return [];
							return [[c.raw_name, v.value != null ? `${round(c.scale != null ? v.value * c.scale + c.offset : v.value)} ${unit(c)}`.trim() : v.value_text]];
						})) }
					: 'no decoded values'
			}
		};
	},

	get_stats(a) {
		const f = findNode(a.node_id);
		if (!f.node) return { result: f };
		const id = f.node.node_id;
		const src = pickSource(id, a.source);
		if (!src) return { result: { error: `${id} has no data in the database.` } };
		const channels = channelsOf(src.payload_schema_id!);
		const ch = pickChannel(channels, a.value);
		if (!ch) return { result: { error: `${id} has no value matching "${a.value}".`, available_values: channels.map((c) => c.raw_name).slice(0, 40) } };
		const from = parseDay(a.start) ?? src.first_ts;
		const to = parseDay(a.end, true) ?? src.last_ts + 1;
		const conv = (v: number | null) => (v == null ? null : round(ch.scale != null ? v * ch.scale + ch.offset : v));
		const where = `FROM sample s JOIN reading r ON r.sample_id = s.id AND r.channel_id = ?1
			WHERE s.node_id = ?2 AND s.source_id = ?3 AND s.ts >= ?4 AND s.ts < ?5 AND r.value IS NOT NULL`;
		const p = [ch.id, id, src.source_id, from, to] as const;
		const tot = all<{ n: number; avg: number | null; min: number | null; max: number | null; t0: number | null; t1: number | null }>(
			`SELECT COUNT(*) n, AVG(r.value) avg, MIN(r.value) min, MAX(r.value) max, MIN(s.ts) t0, MAX(s.ts) t1 ${where}`, ...p)[0];
		if (!tot.n) return { result: { node: id, value: ch.raw_name, source: src.source_code, error: 'No values in that period.', data_exists: `${istDate(src.first_ts)} to ${istDate(src.last_ts)}` } };
		const peak = all<{ ts: number }>(`SELECT s.ts ${where} ORDER BY r.value DESC LIMIT 1`, ...p)[0];
		const out: Record<string, unknown> = {
			node: id, value: ch.raw_name, meaning: ch.quantity_name, unit: unit(ch),
			source: src.source_code, kind: src.source_kind,
			period: `${istDate(tot.t0!)} to ${istDate(tot.t1!)}`, readings: tot.n,
			data_exists_for: `${istDate(src.first_ts)} to ${istDate(src.last_ts)}`,
			average: conv(tot.avg), minimum: conv(tot.min), maximum: conv(tot.max), maximum_at: istTime(peak.ts)
		};
		if (ch.value_kind === 'cumulative') {
			out.note = 'This is a counter that only goes up; average/min/max are not meaningful. Use increase_over_period.';
			out.increase_over_period = conv(tot.max! - tot.min!);
			delete out.average;
		}
		if (ch.unit_note) out.unit_warning = ch.unit_note;
		const g = String(a.group_by ?? '');
		const expr = g === 'hour' ? `CAST(strftime('%H', s.ts + ${IST}, 'unixepoch') AS INTEGER)`
			: g === 'month' ? `strftime('%Y-%m', s.ts + ${IST}, 'unixepoch')`
			: g === 'weekday' ? `strftime('%w', s.ts + ${IST}, 'unixepoch')` : null;
		if (expr && ch.value_kind !== 'cumulative') {
			const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
			const rows = all<{ k: string | number; avg: number; max: number }>(`SELECT ${expr} k, AVG(r.value) avg, MAX(r.value) max ${where} GROUP BY 1 ORDER BY 1`, ...p);
			const label = (k: string | number) => (g === 'weekday' ? days[+k] : g === 'hour' ? `${k}:00` : String(k));
			out[`average_by_${g}`] = Object.fromEntries(rows.map((r) => [label(r.k), conv(r.avg)]));
			if (rows.length) {
				const top = rows.reduce((x, y) => (y.avg > x.avg ? y : x)), low = rows.reduce((x, y) => (y.avg < x.avg ? y : x));
				out[`highest_${g}`] = `${label(top.k)} (average ${conv(top.avg)})`;
				out[`lowest_${g}`] = `${label(low.k)} (average ${conv(low.avg)})`;
			}
		}
		return { result: out };
	},

	compare_nodes(a) {
		const words = String(a.query ?? '').toLowerCase().split(/\s+/).filter(Boolean);
		const v = String(a.type ?? '').toUpperCase().trim();
		const nodes = getNodes().filter((n) => {
			if (n.level !== 'history' || (v && n.vertical !== v)) return false;
			const text = `${n.node_id} ${n.vertical_name} ${n.subtype_name ?? ''} ${n.site_name ?? ''}`.toLowerCase();
			return words.every((w) => text.includes(w));
		}).slice(0, 14);
		if (!nodes.length) return { result: { error: 'No nodes with a year of real data match.', hint: 'Only about 13 nodes have real history; try search_nodes.' } };
		const from = parseDay(a.start), to = parseDay(a.end, true);
		const rows = nodes.map((n) => {
			const src = pickSource(n.node_id);
			const ch = src && pickChannel(channelsOf(src.payload_schema_id!), a.value);
			if (!src || !ch) return { node: n.node_id, place: n.site_name, result: 'does not report this value' };
			const t = all<{ n: number; avg: number | null; max: number | null; min: number | null }>(
				`SELECT COUNT(*) n, AVG(r.value) avg, MAX(r.value) max, MIN(r.value) min FROM sample s JOIN reading r ON r.sample_id = s.id AND r.channel_id = ?1
				 WHERE s.node_id = ?2 AND s.source_id = ?3 AND s.ts >= ?4 AND s.ts < ?5 AND r.value IS NOT NULL`,
				ch.id, n.node_id, src.source_id, from ?? src.first_ts, to ?? src.last_ts + 1)[0];
			if (!t.n) return { node: n.node_id, place: n.site_name, result: 'has messages but no values for this' };
			const c = (x: number | null) => (x == null ? null : round(ch.scale != null ? x * ch.scale + ch.offset : x));
			const row: Record<string, unknown> = { node: n.node_id, place: n.site_name, value: ch.raw_name, unit: unit(ch), average: c(t.avg), maximum: c(t.max), readings: t.n };
			if ((t.avg ?? 0) < 0) row.note = 'negative numbers: this meter reports generation as negative, or its unit is unreliable';
			return row;
		});
		// Do the ranking here: a small model misreads a list of numbers more easily than one named winner.
		const usable = rows.filter((r) => typeof r.average === 'number' && (r.average as number) >= 0) as { node: string; average: number }[];
		usable.sort((x, y) => y.average - x.average);
		const skipped = rows.filter((r) => !usable.includes(r as never)).map((r) => r.node);
		return {
			result: {
				highest_average: usable[0] ? `${usable[0].node} (${usable[0].average})` : 'none',
				lowest_average: usable.length > 1 ? `${usable[usable.length - 1].node} (${usable[usable.length - 1].average})` : undefined,
				ranking_high_to_low: usable.map((r) => r.node),
				left_out_of_ranking: skipped.length ? { nodes: skipped, why: 'no values, or negative / unreliable values' } : undefined,
				details: rows,
				note: 'Real measured data.'
			}
		};
	},

	net_zero(a) {
		const r = runWhatIf({ extra_solar_pct: Number(a.extra_solar_pct) || 0, export_surplus: a.net_metering === true });
		return { result: summariseWhatIf(r), ui: { action: 'what_if', href: `/whatif?solar=${r.input.extra_solar_pct}${r.input.export_surplus ? '&export=1' : ''}` } };
	},

	show_node(a) {
		const f = findNode(a.node_id);
		if (!f.node) return { result: f };
		return { result: { ok: true, shown: f.node.node_id }, ui: { action: 'show_node', node_id: f.node.node_id } };
	}
};

// ---- live data (the campus API, the board's NPU forecasters and the anomaly rule; live/loop.ts) ----------

const SOURCE = 'Campus API: synthetic 2026 data replayed in real time. Forecasts from the LSTM models on the QIDK NPU.';
const NOTE = 'A flag means the reading was far from the forecast. It is not proof of a fault. Most natural flags on this synthetic data are false alarms.';

function explainEvent(e: LiveEvent) {
	const hints: string[] = [];
	if (e.injected) hints.push(`This was a demo fault (${e.injected}) injected from the dashboard, not a real event.`);
	hints.push(`The reading was ${Math.abs(e.residual_kW)} kW ${e.residual_kW < 0 ? 'below' : 'above'} the forecast, ${e.flagged ? `${e.excess_kW} kW beyond` : 'within'} the allowed range (±${e.limit_kW} kW around the usual error at ${e.hour}:00).`);
	if (e.signal === 'pl00') {
		const ch = (now: number | null, before: number | null) => (now == null || before == null || before < 1 ? null : (now - before) / before);
		const ni = ch(e.peers['SL-NI01-00 now kW'], e.peers['SL-NI01-00 15 min earlier kW']);
		const vn = ch(e.peers['SL-VN95-00 now kW'], e.peers['SL-VN95-00 15 min earlier kW']);
		const own = e.recent_kW.at(-1) ? (e.actual_kW - e.recent_kW.at(-1)!) / e.recent_kW.at(-1)! : null;
		if (ni != null && vn != null && own != null) {
			const together = Math.sign(ni) === Math.sign(own) && Math.sign(vn) === Math.sign(own) && Math.abs(ni) > 0.25 && Math.abs(vn) > 0.25;
			hints.push(together ? 'The other solar meters changed the same way at the same time, so this looks campus-wide (for example weather), not a fault at this plant.' : 'The other solar meters did not change the same way, so the change is specific to this plant.');
		}
	}
	if (e.signal === 'grid' && e.estimated_inputs) hints.push('3 of the 6 solar inputs to this model are estimated from neighbouring meters, so its forecasts are less certain than in 2025.');
	return {
		id: e.id, time: e.time, signal: e.label, flagged: e.flagged, injected: e.injected,
		actual_kW: e.actual_kW, forecast_kW: e.forecast_kW, limit_kW: e.limit_kW,
		last_4_readings_kW: e.recent_kW, at_the_same_moment: e.peers,
		forecast_by: `${e.model} on the ${e.variant === 'aimet_w8a16' ? 'NPU' : 'CPU'}${e.npu_ms != null ? ` (${e.npu_ms.toFixed(1)} ms)` : ''}`,
		how_often_natural_flags_happen: SIGNALS[e.signal].natural_flag_rate,
		what_it_means: hints, note: NOTE
	};
}

Object.assign(impl, {
	live_status() {
		if (!liveEnabled()) return { result: { error: 'Live data is turned off on this dashboard.' } };
		const s = liveStatus();
		const act = activeModelsInfo() as Record<string, { id: string; label?: string }>;
		return { result: { source: SOURCE, now: s.now, api_ok: s.api.ok, npu_ok: s.npu.ok, last_reading_checked: s.last_reading_checked, grid: s.latest.grid ?? null, palash_solar: s.latest.pl00 ?? null, campus_solar: s.latest.campus_solar ?? null, last_24h: s.last_24h, forecast_by: { grid: act.grid?.label ?? act.grid?.id, palash_solar: act.pl00?.label ?? act.pl00?.id }, note: NOTE } };
	},
	recent_anomalies(a: Args) {
		if (!liveEnabled()) return { result: { error: 'Live data is turned off on this dashboard.' } };
		const hours = Math.max(1, Math.min(168, Number(a.hours ?? 24) || 24));
		const sig = a.signal === 'grid' || a.signal === 'pl00' ? a.signal : undefined;
		const ev = recentEvents(hours, sig);
		return { result: {
			source: SOURCE, hours, total: ev.length, natural: ev.filter((e) => e.flagged && !e.injected).length, injected: ev.filter((e) => e.injected).length,
			newest: ev.slice(0, 5).map((e) => ({ id: e.id, time: e.time, signal: e.label, actual_kW: e.actual_kW, forecast_kW: e.forecast_kW, beyond_limit_kW: e.flagged ? e.excess_kW : 0, injected: e.injected })),
			note: NOTE
		} };
	},
	model_results(a: Args) {
		const hours = Math.max(1, Math.min(192, Number(a.hours ?? 24) || 24));
		const b = resultsBundle(hours) as { live: Record<string, { rows: Record<string, unknown>[] }>; offline: { split: string; rows: Record<string, unknown>[]; importance: { by_node: [string, number][]; by_quantity: [string, number][] } } | null };
		const pick = (r: Record<string, unknown>) => ({ forecast: r.label, role: r.kind, readings: r.n, mae_kW: r.mae_kW, r2: r.r2, skill_vs_persistence: r.skill_vs_persistence, would_flag_rate: r.flag_rate });
		return { result: {
			offline_2025_test: b.offline ? {
				split: b.offline.split,
				r2_and_skill_by_horizon: b.offline.rows.map((r) => ({ forecast: r.id, ...Object.fromEntries(['15min', '1h', '24h'].filter((h) => r[h]).map((h) => [h, r[h]])) })),
				what_inputs_matter_by_meter: b.offline.importance.by_node.slice(0, 5), by_kind: b.offline.importance.by_quantity.slice(0, 4)
			} : 'not available',
			live_shadow_last_hours: { hours, grid: b.live.grid?.rows.map(pick) ?? [], palash_solar: b.live.pl00?.rows.map(pick) ?? [] },
			how_to_read: 'Skill = 1 - error(model)/error(persistence); above 0 = better than "next reading = current reading". Persistence and "same time yesterday" are baselines that need no training. Live data is synthetic, so "would flag" rates are false alarms. Importance is LightGBM gain importance, not SHAP.',
			key_findings_2025: [
				'15 minutes ahead: persistence beats every model, for both solar and grid.',
				'1 hour ahead, solar: the guided LSTM beats both baselines (R² 0.953, skill +0.45). 1 hour ahead, grid: persistence beats every model.',
				'24 hours ahead, solar: same time yesterday (R² 0.915) beats every model, because no model has weather data.',
				'24 hours ahead, grid: LightGBM (R² 0.20) beats same time yesterday (R² -0.05), but every grid forecast a day ahead is weak.'
			]
		} };
	},
	list_models() {
		const act = activeModelsInfo() as Record<string, { id: string }>;
		return { result: {
			models: listModels().map((m) => ({
				id: m.id, name: m.label, family: m.family, horizon: m.horizon <= 4 ? '1 hour' : '24 hours',
				test_R2_15min: m.test_first_step_R2 ? { solar: m.test_first_step_R2.Solar, grid: m.test_first_step_R2.Grid } : 'not scored',
				npu_vs_fp32_sqnr_dB: m.sqnr_db ?? null, npu_ms: m.npu_ms ?? null,
				used_now_for: (['grid', 'pl00'] as const).filter((s) => act[s]?.id === m.id).map((s) => (s === 'grid' ? 'campus grid draw' : 'Palash solar'))
			})),
			note: 'R² of the +15-minute forecast on the 2025 test split (higher is better, 1 = perfect). Only people can switch models, on the Live page; you cannot.'
		} };
	},
	explain_anomaly(a: Args) {
		if (!liveEnabled()) return { result: { error: 'Live data is turned off on this dashboard.' } };
		const e = a.id != null ? eventById(Number(a.id)) : recentEvents(168)[0];
		if (!e) return { result: { error: a.id != null ? `No event with id ${a.id}.` : 'No anomalies in the last 7 days.' } };
		return { result: explainEvent(e) };
	}
} satisfies Record<string, (a: Args) => ToolOutput>);

export function runTool(name: string, args: Args): ToolOutput {
	const fn = impl[name];
	if (!fn) return { result: { error: `Unknown tool ${name}` } };
	try {
		return fn(args ?? {});
	} catch (e) {
		return { result: { error: e instanceof Error ? e.message : String(e) } };
	}
}

const str = (description: string) => ({ type: 'string', description });
const fn = (name: string, description: string, properties: Record<string, unknown>, required: string[] = []) => ({
	type: 'function', function: { name, description, parameters: { type: 'object', properties, required } }
});
export const TOOL_SPECS = [
	fn('data_overview', 'Counts of sensor nodes by type, which nodes have a year of real data, and the data sources. Only for questions about what data exists. Contains no measurements.', {}),
	fn('search_nodes', 'Find sensor nodes by words in their ID, type, or building name. Returns up to 10 nodes.', {
		query: str('Words to look for, e.g. "solar nilgiri", "kohli", "water flow". May be empty.'),
		type: { type: 'string', enum: ['AQ', 'CM', 'EM', 'SL', 'SR', 'WE', 'WM', 'WN'], description: 'Optional node type: AQ air quality, CM crowd, EM energy, SL solar, SR smart rooms, WE weather, WM water, WN street poles.' },
		only_with_real_history: { type: 'boolean', description: 'true = only nodes with a year of real data.' }
	}),
	fn('get_node', 'Everything about one node: what and where it is, how much data it has, which values it reports, and its last message.', { node_id: str('Node ID, e.g. EM-CR-SB00-00') }, ['node_id']),
	fn('get_stats', 'Average, minimum, maximum (and when the maximum happened) of one value of one node over a period. Uses real measured data unless source says otherwise.', {
		node_id: str('Node ID, e.g. EM-CR-SB00-00'),
		value: str('Which value: a name from get_node (e.g. "GDPower", "Power") or a general word like "power", "voltage", "energy".'),
		start: str('Optional first day, YYYY-MM-DD or YYYY-MM.'),
		end: str('Optional last day, YYYY-MM-DD or YYYY-MM.'),
		group_by: { type: 'string', enum: ['hour', 'weekday', 'month'], description: 'Optional: also return the average per hour of day, per weekday, or per month.' },
		source: str('Optional: "synthetic" or "derived" to read generated data instead of measured.')
	}, ['node_id', 'value']),
	fn('compare_nodes', 'Compare the average and maximum of one value across several nodes that have real data, e.g. which solar node produces most. Returns one row per node.', {
		value: str('Which value, e.g. "power", "voltage", "energy".'),
		type: { type: 'string', enum: ['EM', 'SL', 'SR'], description: 'Node type to compare: EM energy, SL solar, SR smart rooms.' },
		query: str('Optional words to narrow the nodes, e.g. a building name.'),
		start: str('Optional first day, YYYY-MM-DD or YYYY-MM.'),
		end: str('Optional last day, YYYY-MM-DD or YYYY-MM.')
	}, ['value']),
	fn('net_zero', "The campus's carbon balance for 2025: emissions from grid electricity and diesel, minus what the trees absorb, and whether that reaches net zero. Optionally with more solar panels.", {
		extra_solar_pct: { type: 'number', description: 'Optional extra solar as a percentage of the solar measured today, 0 to 2000. 100 = double the solar. Leave out for the situation today.' },
		net_metering: { type: 'boolean', description: 'Optional: true = surplus solar is exported and credited (net metering). Default false = surplus is wasted (no export, no storage).' }
	}),
	fn('show_node', 'Open a node on the dashboard map so the user can see it. Use when the user asks to show, open or go to a node.', { node_id: str('Node ID') }, ['node_id']),
	fn('live_status', 'Live right now: the latest checked reading and the next-hour forecast for campus grid draw and the Palash solar plant, whether the board and the data feed are working, and how many anomalies were flagged in the last 24 hours.', {}),
	fn('recent_anomalies', 'List live anomaly flags: readings that were far from the forecast. Newest first, up to 5, each with an id.', {
		hours: { type: 'number', description: 'How far back to look, in hours (default 24, max 168).' },
		signal: { type: 'string', enum: ['grid', 'pl00'], description: 'Optional: grid = campus grid draw, pl00 = Palash solar plant.' }
	}),
	fn('model_results', 'Results and interpretability: how accurate each forecasting model is compared with simple baselines (persistence = next reading equals the current one; same time yesterday), on 2025 data at 15 min, 1 h and 24 h and live in shadow mode, and which inputs the models rely on. For "is the model better than persistence", "results", "baseline", "skill", "which inputs matter".', {
		hours: { type: 'number', description: 'Live window in hours (default 24, max 192).' }
	}),
	fn('list_models', 'The forecasting models on the board: which one is used now for grid and Palash solar, and how accurate and fast each one is (R², NPU vs FP32 agreement, NPU milliseconds). For questions like "which model", "how accurate", "compare the models".', {}),
	fn('explain_anomaly', 'Details of one live anomaly flag: actual vs forecast, the allowed range, the readings before it, what other meters did at the same moment, whether it was an injected demo fault, and what that suggests.', {
		id: { type: 'number', description: 'Event id from recent_anomalies. Leave out for the newest one.' }
	})
];
