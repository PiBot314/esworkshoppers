// Net-zero balance and "what if we had more solar": replays a year of measured campus
// grid draw from campus.sqlite and turns it into a carbon balance.
//
//   emissions = activity x emission factor, summed over sources
//   net       = emissions - what the campus trees absorb
import { all, channelsOf, getCoverage } from '#lib/server/db.ts';
import type { NetZeroRow, WhatIfInput, WhatIfResult } from '#lib/types.ts';

const SLOT = 900; // seconds: 15 minutes
const DT = SLOT / 3600; // hours
const IST = 19800;

/** Where the signals come from. Everything else is looked up in the database. */
const LOAD = { node: 'EM-CR-SB00-00', quantity: 'active_power', circuit: 'GD' };
const GENERATOR = { node: 'EM-CR-SB00-01', circuits: ['G1', 'G2'] };
/** Solar nodes with a clean, near-complete year. sign -1 = the meter reports generation as negative. */
const SOLAR = [
	{ node: 'SL-NI03-00', sign: 1 },
	{ node: 'SL-NI03-01', sign: 1 },
	{ node: 'SL-PL00-00', sign: -1 }
];

export const ASSUMPTIONS = {
	/** India grid factor (CEA), from papers/carbon_calculations/Calculations.docx */
	grid_kg_per_kwh: 0.7,
	/** Typical diesel generator. NOT from the project documents: replace with the DG set figure. */
	diesel_kg_per_kwh: 0.8,
	trees: {
		/** IIIT-H's own figure: "over 2,000 trees" on the 66-acre campus (blogs.iiit.ac.in, July 2025).
		 *  The 1,200 saplings planted that month are left out: they are too small to matter yet. */
		count: 2000,
		/** the count used in Calculations.docx; no source is given for it there */
		count_in_document: 46000,
		dbh_cm: 26.4, // Calculations.docx: average trunk diameter at 1.3 m
		/** Chave et al. (2014) environmental stress E at the campus (17.4455 N, 78.3489 E), read from
		 *  their global raster. High = hot, dry, seasonal: trees are shorter for the same trunk. */
		climate_stress_E: 0.937,
		wood_density: 0.6, // g/cm3, assumed: no species list for the campus was found
		/** trunk diameter growth per year, cm: low / central / high (assumed range for tropical trees) */
		growth_cm_per_year: [0.3, 0.5, 0.8] as [number, number, number],
		root_ratio: 0.26, // below-ground = 26 % of above-ground (Calculations.docx)
		carbon_fraction: 0.47, // Calculations.docx
		co2_per_carbon: 3.67, // Calculations.docx
		/** the figure in Calculations.docx, kept for comparison only */
		document_kg_per_year: 6190968
	}
};

/** Above-ground biomass of one tree, kg. Chave et al. (2014), the form that uses the site's climate
 *  instead of a measured height: exp(-1.803 - 0.976 E + 0.976 ln(density) + 2.673 ln(D) - 0.0299 ln(D)^2) */
function agb(dbh: number, t = ASSUMPTIONS.trees) {
	const lnD = Math.log(dbh);
	return Math.exp(-1.803 - 0.976 * t.climate_stress_E + 0.976 * Math.log(t.wood_density) + 2.673 * lnD - 0.0299 * lnD * lnD);
}

/** CO2 the campus trees absorb in one year, kg: the biomass a year of growth adds, converted to CO2. */
export function treeSequestration() {
	const t = ASSUMPTIONS.trees;
	const perTree = (growth: number) =>
		(agb(t.dbh_cm + growth) - agb(t.dbh_cm)) * (1 + t.root_ratio) * t.carbon_fraction * t.co2_per_carbon;
	const [lo, mid, hi] = t.growth_cm_per_year.map(perTree);
	const lnD = Math.log(t.dbh_cm);
	return {
		count: t.count,
		low: lo * t.count, central: mid * t.count, high: hi * t.count,
		per_tree_low: lo, per_tree_central: mid, per_tree_high: hi,
		biomass_per_tree: agb(t.dbh_cm),
		/** tree height the same paper's climate model implies for this trunk size, m */
		implied_height_m: Math.exp(0.893 - t.climate_stress_E + 0.76 * lnD - 0.034 * lnD * lnD),
		/** the same per-tree rate with the tree count from Calculations.docx */
		with_document_count: { count: t.count_in_document, low: lo * t.count_in_document, central: mid * t.count_in_document, high: hi * t.count_in_document }
	};
}

function measuredSource(node: string) {
	const best = (getCoverage().get(node) ?? [])
		.filter((c) => c.source_kind === 'measured' && c.payload_schema_id != null)
		.sort((a, b) => b.n - a.n)[0];
	if (!best) throw new Error(`No measured data for ${node}`);
	return best;
}

/** 15-minute averages of one power channel of one node, in kW. */
function slotSeries(node: string, pick: (c: ReturnType<typeof channelsOf>[number]) => boolean) {
	const cov = measuredSource(node);
	const ch = channelsOf(cov.payload_schema_id!).find(pick);
	if (!ch || ch.scale == null) throw new Error(`No usable power channel on ${node}`);
	const rows = all<{ t: number; v: number }>(
		`SELECT (s.ts / ${SLOT}) * ${SLOT} AS t, AVG(r.value) AS v
		 FROM sample s JOIN reading r ON r.sample_id = s.id AND r.channel_id = ?1
		 WHERE s.node_id = ?2 AND s.source_id = ?3 AND r.value IS NOT NULL
		 GROUP BY 1 ORDER BY 1`,
		ch.id, node, cov.source_id
	);
	for (const r of rows) r.v = r.v * ch.scale + ch.offset;
	return { rows, source: cov.source_code };
}

interface Base {
	source: string;
	t: Int32Array; // slot start, unix s
	load: Float64Array; // kW, measured grid draw
	solar: Float64Array; // kW, measured generation of the SOLAR nodes
	month: string[]; // 'YYYY-MM' (IST) per slot
	months: string[];
	dieselKwh: number;
}

let base: Base | undefined;
function getBase(): Base {
	if (base) return base;
	const load = slotSeries(LOAD.node, (c) => c.quantity === LOAD.quantity && c.circuit === LOAD.circuit);
	const solarAt = new Map<number, number>();
	for (const s of SOLAR) {
		const { rows } = slotSeries(s.node, (c) => c.quantity === 'active_power' && (c.phase === '3P' || c.phase == null));
		for (const r of rows) solarAt.set(r.t, (solarAt.get(r.t) ?? 0) + Math.max(0, r.v * s.sign));
	}
	let dieselKwh = 0;
	for (const g of GENERATOR.circuits) {
		try {
			const { rows } = slotSeries(GENERATOR.node, (c) => c.quantity === 'active_power' && c.circuit === g);
			for (const r of rows) dieselKwh += Math.max(0, r.v) * DT;
		} catch { /* generator channel missing: count nothing */ }
	}
	const n = load.rows.length;
	const b: Base = {
		source: load.source, dieselKwh,
		t: new Int32Array(n), load: new Float64Array(n), solar: new Float64Array(n), month: new Array(n), months: []
	};
	load.rows.forEach((r, i) => {
		b.t[i] = r.t;
		b.load[i] = Math.max(0, r.v);
		b.solar[i] = solarAt.get(r.t) ?? 0;
		b.month[i] = new Date((r.t + IST) * 1000).toISOString().slice(0, 7);
		if (b.months[b.months.length - 1] !== b.month[i]) b.months.push(b.month[i]);
	});
	return (base = b);
}

export function runWhatIf(input: WhatIfInput): WhatIfResult {
	const b = getBase();
	const A = ASSUMPTIONS;
	const n = b.t.length;
	const k = Math.min(20, Math.max(0, (input.extra_solar_pct || 0) / 100));
	const exportSurplus = !!input.export_surplus;

	// Extra panels follow the measured solar pattern. What they produce beyond the campus's
	// demand at that moment is not counted (no export credit, no storage).
	const grid = new Float64Array(n);
	const monthly = new Map<string, { days: Set<number>; before: number; after: number }>();
	let gridBefore = 0, gridAfter = 0, solarNow = 0, extra = 0, unused = 0, floor = 0;
	// "No production": below 1 % of the year's peak (meters report small non-zero values at night)
	let peak = 0;
	for (let i = 0; i < n; i++) peak = Math.max(peak, b.solar[i]);
	const dark = 0.01 * peak;
	for (let i = 0; i < n; i++) {
		const add = k * b.solar[i];
		grid[i] = Math.max(0, b.load[i] - add);
		gridBefore += b.load[i] * DT;
		gridAfter += grid[i] * DT;
		solarNow += b.solar[i] * DT;
		extra += add * DT;
		unused += Math.max(0, add - b.load[i]) * DT;
		if (b.solar[i] < dark) floor += b.load[i] * DT; // no number of panels covers these slots without storage
		const m = monthly.get(b.month[i]) ?? { days: new Set<number>(), before: 0, after: 0 };
		m.days.add(Math.floor((b.t[i] + IST) / 86400));
		m.before += b.load[i] * DT;
		m.after += grid[i] * DT;
		monthly.set(b.month[i], m);
	}

	const trees = treeSequestration();
	const row = (gridKwh: number, solarKwh: number, exportedKwh = 0): NetZeroRow => {
		const grid_kg = (gridKwh - exportedKwh) * A.grid_kg_per_kwh, diesel_kg = b.dieselKwh * A.diesel_kg_per_kwh;
		const total_kg = grid_kg + diesel_kg;
		return {
			grid_kwh: gridKwh, grid_kg, diesel_kwh: b.dieselKwh, diesel_kg, total_kg,
			trees_kg: trees.central,
			net_kg: total_kg - trees.central,
			net_low_kg: total_kg - trees.high, // most absorption = lowest net
			net_high_kg: total_kg - trees.low,
			solar_kwh: solarKwh, solar_avoided_kg: solarKwh * A.grid_kg_per_kwh, exported_kwh: exportedKwh
		};
	};
	const before = row(gridBefore, solarNow);
	// Without net metering the surplus is wasted; with it, the surplus is exported and credited at the grid factor.
	const after = exportSurplus ? row(gridAfter, solarNow + extra, unused) : row(gridAfter, solarNow + extra - unused);

	// One day in detail: the sunniest day of the chosen month (default: the month with most solar).
	const sunByDay = new Map<number, number>();
	const dayOf = (i: number) => Math.floor((b.t[i] + IST) / 86400);
	const wanted = b.months.includes(input.month ?? '') ? input.month! : null;
	for (let i = 0; i < n; i++) {
		if (wanted && b.month[i] !== wanted) continue;
		sunByDay.set(dayOf(i), (sunByDay.get(dayOf(i)) ?? 0) + b.solar[i]);
	}
	const bestDay = [...sunByDay.entries()].sort((x, y) => y[1] - x[1])[0][0];
	const points: WhatIfResult['day']['points'] = [];
	for (let i = 0; i < n; i++) if (dayOf(i) === bestDay) points.push({ t: b.t[i], before: b.load[i], after: grid[i] });

	// How much more solar would bring net emissions to zero, if every extra kWh counts (net metering or storage).
	// Diesel can only be offset that way too, so it's part of the gap. Without export or storage there is a floor:
	// the load in slots where the panels produce nothing (nights) stays on the grid however many panels are added.
	const gapKwh = Math.max(0, before.net_kg) / A.grid_kg_per_kwh;
	const minNetNoExport = floor * A.grid_kg_per_kwh + b.dieselKwh * A.diesel_kg_per_kwh - trees.central;

	return {
		input: { extra_solar_pct: k * 100, month: wanted ?? new Date(bestDay * 86400000).toISOString().slice(0, 7), export_surplus: exportSurplus },
		source: b.source,
		period: { from: b.t[0], to: b.t[n - 1] + SLOT, days: new Set(Array.from(b.t, (t) => Math.floor((t + IST) / 86400))).size },
		months: b.months,
		monthly: [...monthly.entries()].map(([month, m]) => ({
			month, days: m.days.size,
			co2_before_kg: m.before * A.grid_kg_per_kwh, co2_after_kg: m.after * A.grid_kg_per_kwh
		})),
		day: { date: bestDay * 86400 - IST, points },
		solar: { nodes: SOLAR.map((s) => s.node), measured_kwh: solarNow, extra_kwh: extra, unused_kwh: unused },
		netzero: { before, after },
		trees: { ...trees, document_kg: A.trees.document_kg_per_year, net_with_document_count_kg: before.total_kg - trees.with_document_count.central },
		to_reach_zero: {
			extra_solar_kwh: gapKwh, times_todays_solar: solarNow ? gapKwh / solarNow : null, extra_pct: solarNow ? (gapKwh / solarNow) * 100 : null,
			floor_grid_kwh: floor, min_net_without_export_kg: minNetNoExport, possible_without_export: minNetNoExport <= 0
		},
		assumptions: A
	};
}

/** Compact version for the language model: a few numbers, rounded. */
export function summariseWhatIf(r: WhatIfResult) {
	const t = (kg: number) => Math.round(kg / 1000);
	const b = r.netzero.before, a = r.netzero.after;
	const side = (x: NetZeroRow) => ({
		grid_electricity_mwh: Math.round(x.grid_kwh / 1000),
		emissions_t_co2: t(x.total_kg),
		absorbed_by_trees_t_co2: t(x.trees_kg),
		net_t_co2: t(x.net_kg),
		net_range_t_co2: `${t(x.net_low_kg)} to ${t(x.net_high_kg)}`,
		solar_generated_mwh: Math.round(x.solar_kwh / 1000)
	});
	const z = r.to_reach_zero;
	return {
		scenario: `${Math.round(r.input.extra_solar_pct)}% extra solar, ${r.input.export_surplus ? 'with net metering (surplus exported and credited)' : 'no export or storage (surplus wasted)'}`,
		replayed: `${r.period.days} days of measured campus grid electricity in 2025`,
		today: side(b),
		with_scenario: r.input.extra_solar_pct > 0 ? side(a) : 'same as today (no extra solar asked for)',
		net_zero_reached: (r.input.extra_solar_pct > 0 ? a.net_kg : b.net_kg) <= 0,
		extra_solar_needed_for_net_zero: z.extra_pct != null
			? `about +${Math.round(z.extra_pct)}% more than today's measured solar (${Math.round((1 + z.times_todays_solar!) * 10) / 10} times today's in total), and only with net metering or storage`
			: 'unknown',
		without_export_or_storage: z.possible_without_export
			? 'reachable with solar alone'
			: `not reachable with solar alone: night-time load keeps at least ${t(z.min_net_without_export_kg)} t CO2 a year however many panels are added`,
		solar_baseline_note: `"Today's measured solar" counts only ${r.solar.nodes.length} solar meters with a clean 2025 (${Math.round(r.solar.measured_kwh / 1000 / r.period.days * 1000)} kWh/day); other meters are partial or missing, so the real multiple is lower.`,
		trees_note: `${r.trees.count} trees (the institute's own count) at about ${Math.round(r.trees.per_tree_central)} kg CO2 per tree per year. Trees offset only a small part of the emissions.`,
		caveat: 'The tree absorption is an estimate. Water, sewage and waste emissions are not included. Net above zero means not net zero.'
	};
}
