export type SourceKind = 'measured' | 'derived' | 'synthetic';
export type Level = 'history' | 'few' | 'none';

export interface SourceInfo { id: number; code: string; kind: SourceKind; description: string | null }

export interface Coverage {
	source_id: number; source_code: string; source_kind: SourceKind;
	payload_schema_id: number | null; n: number; first_ts: number; last_ts: number;
}

export interface NodeSummary {
	node_id: string; vertical: string; vertical_name: string; subtype: string; subtype_name: string | null;
	site_code: string | null; site_name: string | null; floor_code: string | null; floor: string | null;
	latitude: number | null; longitude: number | null; location_quality: string; is_derived: number;
	/** real messages / derived + synthetic messages */
	measured: number; other: number; level: Level;
}

export interface Channel {
	id: number; position: number; raw_name: string; quantity: string; quantity_name: string;
	value_kind: string; canonical_unit: string | null; circuit: string | null; phase: string | null;
	qualifier: string; raw_unit: string | null; scale: number | null; offset: number; unit_note: string | null;
}

export interface Latest {
	ts: number; version: string | null; raw_payload: string | null;
	values: { channel_id: number; value: number | null; value_text: string | null }[];
}

export interface NodeDetail {
	node: {
		node_id: string; om2m_path: string | null; descriptor_created: string | null;
		payload_schema_id: number | null; node_number: string | null; device: string | null;
		controller: string | null; site_confidence: string | null; site_note: string | null;
		vertical_description: string | null; subtype_description: string | null;
	};
	versions: { version: string; valid_from: string | null; valid_to: string | null }[];
	coverage: (Coverage & { undecoded: number; top: number })[];
	/** channels and latest message, keyed by source id */
	channels: Record<number, Channel[]>;
	latest: Record<number, Latest>;
	declared: Channel[];
}

export interface Series {
	bucket: number; from: number; to: number;
	points: { t: number; avg: number; min: number; max: number; n: number }[];
}

export interface WhatIfInput {
	/** extra solar as a percentage of today's measured solar */
	extra_solar_pct: number;
	month?: string;
	/** count solar beyond the campus's demand at that moment as exported and credited (net metering) */
	export_surplus?: boolean;
}
export interface NetZeroRow {
	grid_kwh: number; grid_kg: number; diesel_kwh: number; diesel_kg: number; total_kg: number;
	/** central tree estimate, and net emissions with it */
	trees_kg: number; net_kg: number;
	/** net emissions at the high and low ends of the tree estimate */
	net_low_kg: number; net_high_kg: number;
	solar_kwh: number; solar_avoided_kg: number;
	/** surplus solar exported and credited against grid electricity (net metering only) */
	exported_kwh: number;
}
export interface WhatIfResult {
	input: Required<WhatIfInput>;
	source: string;
	period: { from: number; to: number; days: number };
	months: string[];
	monthly: { month: string; days: number; co2_before_kg: number; co2_after_kg: number }[];
	day: { date: number; points: { t: number; before: number; after: number }[] };
	solar: { nodes: string[]; measured_kwh: number; extra_kwh: number; unused_kwh: number };
	netzero: { before: NetZeroRow; after: NetZeroRow };
	trees: {
		count: number; low: number; central: number; high: number;
		per_tree_low: number; per_tree_central: number; per_tree_high: number;
		biomass_per_tree: number; implied_height_m: number;
		with_document_count: { count: number; low: number; central: number; high: number };
		document_kg: number; net_with_document_count_kg: number;
	};
	to_reach_zero: {
		/** extra clean energy a year that would bring net to zero, if every kWh counts (net metering or storage) */
		extra_solar_kwh: number; times_todays_solar: number | null; extra_pct: number | null;
		/** without export or storage: grid electricity that remains however much solar is added (load while panels produce nothing) */
		floor_grid_kwh: number; min_net_without_export_kg: number; possible_without_export: boolean;
	};
	assumptions: unknown;
}
