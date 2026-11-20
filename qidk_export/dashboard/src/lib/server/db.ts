// The dashboard's ONLY data source: campus.sqlite, opened read-only.
import { Database } from 'bun:sqlite';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import type { Coverage, NodeDetail, NodeSummary, Series, SourceInfo, Channel, Latest } from '#lib/types.ts';

const DB_PATH = resolve(process.env.CAMPUS_DB ?? '../data/campus.sqlite');

// Opened on first use, not at import: `vite build` imports this module while analysing routes,
// and the build machine need not have the database.
let handle: Database | undefined;
function db(): Database {
	if (handle) return handle;
	if (!existsSync(DB_PATH)) throw new Error(`campus.sqlite not found at ${DB_PATH}. Set CAMPUS_DB to its path.`);
	return (handle = new Database(DB_PATH, { readonly: true }));
}

export const all = <T>(sql: string, ...p: (string | number)[]) => db().query(sql).all(...p) as T[];
export const get = <T>(sql: string, ...p: (string | number)[]) => db().query(sql).get(...p) as T | null;

/** A node counts as having "history" once it has this many real messages. */
const HISTORY_MIN = 1000;

// ---- sources ---------------------------------------------------------------

let sourcesCache: SourceInfo[] | undefined;
export function getSources(): SourceInfo[] {
	return (sourcesCache ??= all<SourceInfo>('SELECT id, code, kind, description FROM source ORDER BY id'));
}

// ---- coverage: how much data each node has, per source ----------------------
// One scan of the sample table (about 2 s), then kept in memory. The database is
// read-only, so this never goes stale while the server runs.

let coverageCache: Map<string, Coverage[]> | undefined;
export function getCoverage(): Map<string, Coverage[]> {
	if (coverageCache) return coverageCache;
	const rows = all<Coverage & { node_id: string }>(
		`SELECT node_id, source_id, payload_schema_id, COUNT(*) AS n, MIN(ts) AS first_ts, MAX(ts) AS last_ts
		 FROM sample GROUP BY node_id, source_id, payload_schema_id`
	);
	const kinds = new Map(getSources().map((s) => [s.id, s]));
	const map = new Map<string, Coverage[]>();
	for (const { node_id, ...r } of rows) {
		const src = kinds.get(r.source_id)!;
		const list = map.get(node_id) ?? [];
		list.push({ ...r, source_code: src.code, source_kind: src.kind });
		map.set(node_id, list);
	}
	return (coverageCache = map);
}

// ---- node list (for the map) -----------------------------------------------

let nodesCache: NodeSummary[] | undefined;
export function getNodes(): NodeSummary[] {
	if (nodesCache) return nodesCache;
	const cov = getCoverage();
	const rows = all<Omit<NodeSummary, 'measured' | 'other' | 'level'>>(
		`SELECT n.node_id, n.vertical, v.name AS vertical_name, n.subtype, sb.name AS subtype_name,
		        n.site_code, st.name AS site_name, n.floor_code, f.meaning AS floor,
		        n.latitude, n.longitude, n.location_quality, n.is_derived
		 FROM node n
		 JOIN vertical v ON v.code = n.vertical
		 LEFT JOIN subtype sb ON sb.vertical = n.vertical AND sb.code = n.subtype
		 LEFT JOIN site st ON st.code = n.site_code
		 LEFT JOIN floor_code f ON f.code = n.floor_code
		 ORDER BY n.node_id`
	);
	return (nodesCache = rows.map((r) => {
		let measured = 0;
		let other = 0;
		for (const c of cov.get(r.node_id) ?? []) {
			if (c.source_kind === 'measured') measured += c.n;
			else other += c.n;
		}
		const level = measured >= HISTORY_MIN ? 'history' : measured > 0 ? 'few' : 'none';
		return { ...r, measured, other, level };
	}));
}

export function getVerticals() {
	return all<{ code: string; name: string; description: string | null }>(
		'SELECT code, name, description FROM vertical ORDER BY code'
	);
}

// ---- one node --------------------------------------------------------------

export function channelsOf(payloadSchemaId: number): Channel[] {
	return all<Channel>(
		`SELECT c.id, c.position, c.raw_name, c.quantity, q.name AS quantity_name, q.value_kind,
		        q.canonical_unit, c.circuit, c.phase, c.qualifier, c.raw_unit, c.scale, c."offset", c.unit_note
		 FROM channel c JOIN quantity q ON q.code = c.quantity
		 WHERE c.payload_schema_id = ? ORDER BY c.position`,
		payloadSchemaId
	);
}

function latestOf(nodeId: string, sourceId: number, ts: number): Latest {
	const s = get<{ id: number; version: string | null; raw_payload: string | null }>(
		'SELECT id, version, raw_payload FROM sample WHERE node_id = ? AND ts = ? AND source_id = ? ORDER BY id DESC LIMIT 1',
		nodeId, ts, sourceId
	);
	const values = s
		? all<{ channel_id: number; value: number | null; value_text: string | null }>(
				'SELECT channel_id, value, value_text FROM reading WHERE sample_id = ?', s.id
			)
		: [];
	return { ts, version: s?.version ?? null, raw_payload: s?.raw_payload ?? null, values };
}

export function getNodeDetail(nodeId: string): NodeDetail | null {
	const node = get<NodeDetail['node']>(
		`SELECT n.node_id, n.om2m_path, n.descriptor_created, n.payload_schema_id, n.node_number,
		        dm.device, dm.controller, st.confidence AS site_confidence, st.note AS site_note,
		        v.description AS vertical_description, sb.description AS subtype_description
		 FROM node n
		 JOIN vertical v ON v.code = n.vertical
		 LEFT JOIN subtype sb ON sb.vertical = n.vertical AND sb.code = n.subtype
		 LEFT JOIN site st ON st.code = n.site_code
		 LEFT JOIN device_model dm ON dm.id = n.device_model_id
		 WHERE n.node_id = ?`,
		nodeId
	);
	if (!node) return null;

	const versions = all<NodeDetail['versions'][number]>(
		'SELECT version, valid_from, valid_to FROM node_version WHERE node_id = ? ORDER BY valid_from', nodeId
	);

	// Merge coverage rows that differ only by layout into one row per source,
	// remembering the layout that carries the most messages.
	const bySource = new Map<number, NodeDetail['coverage'][number]>();
	for (const c of getCoverage().get(nodeId) ?? []) {
		const cur = bySource.get(c.source_id);
		if (!cur) {
			bySource.set(c.source_id, { ...c, undecoded: c.payload_schema_id == null ? c.n : 0, top: c.n });
			continue;
		}
		if (c.payload_schema_id == null) cur.undecoded += c.n;
		else if (cur.payload_schema_id == null || c.n > cur.top) {
			cur.payload_schema_id = c.payload_schema_id;
			cur.top = c.n;
		}
		cur.n += c.n;
		cur.first_ts = Math.min(cur.first_ts, c.first_ts);
		cur.last_ts = Math.max(cur.last_ts, c.last_ts);
	}
	const coverage = [...bySource.values()].sort((a, b) => a.source_id - b.source_id);

	const channels: Record<number, Channel[]> = {};
	const latest: Record<number, Latest> = {};
	for (const c of coverage) {
		if (c.payload_schema_id != null) channels[c.source_id] = channelsOf(c.payload_schema_id);
		latest[c.source_id] = latestOf(nodeId, c.source_id, c.last_ts);
	}
	// What the node says it sends (its descriptor layout), even when no data exists.
	const declared = node.payload_schema_id != null ? channelsOf(node.payload_schema_id) : [];

	return { node, versions, coverage, channels, latest, declared };
}

// ---- time series -----------------------------------------------------------

const BUCKETS = [60, 300, 900, 3600, 3 * 3600, 6 * 3600, 86400, 7 * 86400];
const TARGET_POINTS = 360;

export function getSeries(nodeId: string, channelId: number, sourceId: number, from: number, to: number): Series {
	const span = Math.max(1, to - from);
	const bucket = BUCKETS.find((b) => span / b <= TARGET_POINTS) ?? BUCKETS[BUCKETS.length - 1];
	const points = all<{ t: number; avg: number; min: number; max: number; n: number }>(
		`SELECT (s.ts / ?1) * ?1 AS t, AVG(r.value) AS avg, MIN(r.value) AS min, MAX(r.value) AS max, COUNT(*) AS n
		 FROM sample s
		 JOIN reading r ON r.sample_id = s.id AND r.channel_id = ?2
		 WHERE s.node_id = ?3 AND s.source_id = ?4 AND s.ts BETWEEN ?5 AND ?6 AND r.value IS NOT NULL
		 GROUP BY 1 ORDER BY 1`,
		bucket, channelId, nodeId, sourceId, from, to
	);
	return { bucket, from, to, points };
}
