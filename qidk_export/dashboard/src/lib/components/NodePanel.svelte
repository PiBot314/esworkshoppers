<script lang="ts">
	import type { Channel, NodeDetail, NodeSummary, Series } from '#lib/types.ts';
	import { convert, fmtCount, fmtDate, fmtDateTime, fmtNum, isChartable, tagsOf, unitOf, colorOf, inkOn, LEVEL_LABEL } from '#lib/format.ts';
	import LineChart from './LineChart.svelte';

	interface Props { summary: NodeSummary; onclose: () => void }
	let { summary, onclose }: Props = $props();

	const RANGES = [
		{ key: 'day', label: 'Last day', seconds: 86400 },
		{ key: 'week', label: 'Last 7 days', seconds: 7 * 86400 },
		{ key: 'month', label: 'Last 30 days', seconds: 30 * 86400 },
		{ key: 'all', label: 'Everything', seconds: Infinity }
	] as const;
	type RangeKey = (typeof RANGES)[number]['key'];

	let detail = $state<NodeDetail | null>(null);
	let failed = $state('');
	let sourceId = $state<number | null>(null);
	let channelId = $state<number | null>(null);
	let range = $state<RangeKey>('week');
	let series = $state<Series | null>(null);
	let loadingSeries = $state(false);

	const cov = $derived(detail?.coverage.find((c) => c.source_id === sourceId) ?? null);
	const channels = $derived((sourceId != null && detail?.channels[sourceId]) || []);
	const chartable = $derived(channels.filter(isChartable));
	const channel = $derived(chartable.find((c) => c.id === channelId) ?? null);
	const latest = $derived(sourceId != null ? (detail?.latest[sourceId] ?? null) : null);
	const latestRows = $derived.by(() => {
		if (!latest) return [];
		const byId = new Map(channels.map((c) => [c.id, c]));
		return latest.values
			.map((v) => ({ v, c: byId.get(v.channel_id) }))
			.filter((r): r is { v: typeof r.v; c: Channel } => !!r.c)
			.sort((a, b) => a.c.position - b.c.position);
	});

	function pickSource(id: number) {
		sourceId = id;
		const c = detail!.coverage.find((x) => x.source_id === id)!;
		const list = (detail!.channels[id] ?? []).filter(isChartable);
		// Keep the same quantity selected when switching source, else prefer a live value over a counter.
		const prev = channel;
		const next =
			(prev && list.find((x) => x.raw_name === prev.raw_name)) ??
			list.find((x) => x.value_kind === 'instant') ?? list[0];
		channelId = next?.id ?? null;
		range = c.n < 1000 ? 'all' : range;
		series = null;
	}

	// Load the node whenever a different one is selected.
	$effect(() => {
		const id = summary.node_id;
		const ctl = new AbortController();
		detail = null;
		failed = '';
		sourceId = channelId = null;
		series = null;
		fetch(`/api/node/${encodeURIComponent(id)}`, { signal: ctl.signal })
			.then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
			.then((d: NodeDetail) => {
				detail = d;
				// Start on real data when there is any, and on the source with the most of it.
				const best = [...d.coverage].sort(
					(a, b) => Number(b.source_kind === 'measured') - Number(a.source_kind === 'measured') || b.n - a.n
				)[0];
				if (best) pickSource(best.source_id);
			})
			.catch((e) => {
				if (e.name !== 'AbortError') failed = String(e.message ?? e);
			});
		return () => ctl.abort();
	});

	// Load the chart whenever source, channel or range changes.
	$effect(() => {
		if (!cov || !channel) return;
		const secs = RANGES.find((r) => r.key === range)!.seconds;
		const to = cov.last_ts;
		const from = Math.max(cov.first_ts, secs === Infinity ? cov.first_ts : to - secs);
		const q = new URLSearchParams({ channel: String(channel.id), source: String(cov.source_id), from: String(from), to: String(to) });
		const ctl = new AbortController();
		loadingSeries = true;
		fetch(`/api/node/${encodeURIComponent(summary.node_id)}/series?${q}`, { signal: ctl.signal })
			.then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
			.then((s: Series) => {
				series = s;
				loadingSeries = false;
			})
			.catch((e) => {
				if (e.name !== 'AbortError') loadingSeries = false;
			});
		return () => ctl.abort();
	});

	const chartPoints = $derived.by(() => {
		const s = series, c = channel;
		if (!s || !c) return [];
		return s.points.map((p) => {
			const a = convert(c, p.avg), lo = convert(c, p.min), hi = convert(c, p.max);
			return { t: p.t, n: p.n, avg: a, min: Math.min(lo, hi), max: Math.max(lo, hi) };
		});
	});

	const KIND_NOTE = {
		measured: 'real sensor data',
		derived: 'computed from real data plus assumptions',
		synthetic: 'generated, not measured'
	} as const;
	const LOCATION_NOTE: Record<string, string> = {
		building: 'Position is the building, shared with other nodes. Not the exact spot.',
		suspect: 'This position is known to be wrong.',
		none: 'No position recorded.'
	};
	const place = $derived(
		[summary.site_name ?? summary.site_code, summary.floor ?? (summary.floor_code ? `floor code ${summary.floor_code}` : null)]
			.filter(Boolean).join(', ')
	);
</script>

<article>
	<header>
		<div class="title">
			<span class="badge" style:background={colorOf(summary.vertical)} style:color={inkOn(summary.vertical)}>{summary.vertical}</span>
			<h2>{summary.node_id}</h2>
			<button class="close" onclick={onclose} aria-label="Close node details">✕</button>
		</div>
		<p class="what">{summary.vertical_name}{summary.subtype_name ? ` · ${summary.subtype_name}` : ''}</p>
		{#if place}<p class="where">{place}</p>{/if}
		{#if summary.is_derived}<p class="note">Not a physical device: its values are computed from other nodes.</p>{/if}
		{#if LOCATION_NOTE[summary.location_quality]}<p class="note">{LOCATION_NOTE[summary.location_quality]}</p>{/if}
	</header>

	{#if failed}
		<p class="note warn">Could not load this node ({failed}).</p>
	{:else if !detail}
		<p class="loading">Loading…</p>
	{:else}
		<section>
			<h3>Device</h3>
			<dl>
				<dt>Instrument</dt><dd>{detail.node.device ?? 'not recorded'}</dd>
				<dt>Controller</dt><dd>{detail.node.controller ?? 'not recorded'}</dd>
				{#if detail.versions.length}
					<dt>Firmware</dt>
					<dd>{detail.versions[detail.versions.length - 1].version}{detail.versions.length > 1 ? ` (${detail.versions.length} versions so far)` : ''}</dd>
				{/if}
				{#if detail.node.om2m_path}<dt>Server path</dt><dd class="mono">{detail.node.om2m_path}</dd>{/if}
			</dl>
		</section>

		<section>
			<h3>Data in the database</h3>
			{#if !detail.coverage.length}
				<p class="note">Nothing. This node is in the catalog but has never contributed a message to the database.</p>
			{:else}
				<p class="hint">{LEVEL_LABEL[summary.level]}. Pick a row to look at that data.</p>
				<div class="sources" role="radiogroup" aria-label="Data source">
					{#each detail.coverage as c (c.source_id)}
						<button role="radio" aria-checked={c.source_id === sourceId} class:on={c.source_id === sourceId} onclick={() => pickSource(c.source_id)}>
							<span class="src">{c.source_code}</span>
							<span class="kind {c.source_kind}">{c.source_kind}</span>
							<span class="count num">{fmtCount(c.n)} {c.n === 1 ? 'message' : 'messages'}</span>
							<span class="span">{c.n === 1 || fmtDate(c.first_ts) === fmtDate(c.last_ts) ? fmtDate(c.last_ts) : `${fmtDate(c.first_ts)} to ${fmtDate(c.last_ts)}`}</span>
						</button>
					{/each}
				</div>
				{#if cov}
					<p class="hint">
						<strong>{cov.source_kind}</strong>: {KIND_NOTE[cov.source_kind]}.
						{#if cov.undecoded}{fmtCount(cov.undecoded)} of these messages could not be decoded.{/if}
					</p>
				{/if}
			{/if}
		</section>

		{#if cov && channel}
			<section>
				<h3>Over time</h3>
				<div class="filters">
					<select bind:value={channelId} aria-label="Value to plot">
						{#each chartable as c (c.id)}
							<option value={c.id}>{c.raw_name} ({c.quantity_name})</option>
						{/each}
					</select>
					<div class="seg" role="radiogroup" aria-label="Period">
						{#each RANGES as r (r.key)}
							<button role="radio" aria-checked={range === r.key} class:on={range === r.key} onclick={() => (range = r.key)}>{r.label}</button>
						{/each}
					</div>
				</div>
				<p class="hint">
					{channel.quantity_name}{unitOf(channel) ? `, ${unitOf(channel)}` : ''}.
					{#if series}Each point is the average over {series.bucket >= 86400 ? `${series.bucket / 86400} day` : series.bucket >= 3600 ? `${series.bucket / 3600} h` : `${series.bucket / 60} min`}; the band is min to max.{/if}
					Periods are counted back from this source's last message.
				</p>
				{#if channel.value_kind === 'cumulative'}
					<p class="note">This is a counter that only goes up. Its slope is the usage, not its height.</p>
				{/if}
				{#if channel.scale == null}
					<p class="note warn">⚠ Unit not trusted, so values are shown as stored with no unit{channel.unit_note ? ` (${channel.unit_note})` : channel.raw_unit ? ` (labelled ${channel.raw_unit})` : ''}.</p>
				{/if}
				<div class:stale={loadingSeries}>
					{#if series}
						<LineChart points={chartPoints} bucket={series.bucket} from={series.from} to={series.to} unit={unitOf(channel)} label={channel.raw_name} />
					{:else}
						<p class="loading">Loading…</p>
					{/if}
				</div>
			</section>
		{/if}

		{#if cov && latest}
			<section>
				<h3>Last message from this source</h3>
				<p class="hint">{fmtDateTime(latest.ts)} IST{latest.version ? ` · ${latest.version}` : ''}</p>
				{#if latestRows.length}
					<table>
						<tbody>
							{#each latestRows as { v, c } (c.id)}
								<tr>
									<td>
										{c.raw_name}
										{#each tagsOf(c) as t (t)}<span class="tag">{t}</span>{/each}
										<small>{c.quantity_name}</small>
									</td>
									<td class="num">
										{#if v.value != null}
											<strong>{fmtNum(convert(c, v.value))}</strong> {unitOf(c)}{#if c.scale == null && c.raw_unit && c.raw_unit !== 'n/a'}<span title={c.unit_note ?? `Labelled ${c.raw_unit}, but that unit is not trusted`}> ⚠</span>{/if}
										{:else}
											<span class="mono">{v.value_text}</span>
										{/if}
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				{:else if latest.raw_payload}
					<p class="note">This message could not be decoded. Original text:</p>
					<pre>{latest.raw_payload}</pre>
				{:else}
					<p class="note">The message has a timestamp but no values.</p>
				{/if}
			</section>
		{/if}

		{#if detail.declared.length}
			<section>
				<h3>What this node says it sends</h3>
				<p class="hint">From its descriptor on the campus server, whether or not any data arrived.</p>
				<ul class="declared">
					{#each detail.declared as c (c.id)}
						<li>{c.raw_name}{c.raw_unit && c.raw_unit !== 'n/a' ? ` (${c.raw_unit})` : ''}</li>
					{/each}
				</ul>
			</section>
		{/if}
	{/if}
</article>

<style>
	article { display: grid; grid-template-columns: minmax(0, 1fr); gap: 16px; padding: 16px; align-content: start; }
	.title { display: flex; align-items: center; gap: 8px; }
	h2 { font-size: 18px; font-weight: 650; flex: 1; overflow-wrap: anywhere; }
	h3 { font-size: 12px; font-weight: 600; color: var(--ink-2); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px; }
	.badge { font-size: 11px; font-weight: 700; padding: 3px 6px; border-radius: 4px; }
	.close { border: 0; background: none; font-size: 16px; padding: 4px 8px; border-radius: 6px; color: var(--ink-2); }
	.close:hover { background: var(--wash); }
	.what { margin-top: 6px; font-weight: 550; }
	.where { color: var(--ink-2); }
	.hint, .note, .loading { font-size: 12.5px; color: var(--ink-2); }
	.note { margin-top: 4px; }
	.note.warn { color: var(--warn-ink); background: var(--warn-bg); padding: 6px 8px; border-radius: 6px; }
	section { border-top: 1px solid var(--grid); padding-top: 12px; display: grid; grid-template-columns: minmax(0, 1fr); gap: 6px; }
	dl { display: grid; grid-template-columns: auto 1fr; gap: 3px 12px; margin: 0; font-size: 13px; }
	dt { color: var(--ink-2); }
	dd { margin: 0; overflow-wrap: anywhere; }
	.mono { font-family: ui-monospace, monospace; font-size: 12px; overflow-wrap: anywhere; }
	.sources { display: grid; gap: 4px; }
	.sources button {
		display: grid; grid-template-columns: 1fr auto; gap: 2px 8px; text-align: left; padding: 7px 9px;
		background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
	}
	.sources button:hover { background: var(--wash); }
	.sources button.on { border-color: var(--ink); box-shadow: inset 0 0 0 1px var(--ink); }
	.src { font-weight: 600; font-family: ui-monospace, monospace; font-size: 12.5px; }
	.kind { font-size: 11px; padding: 1px 6px; border-radius: 10px; border: 1px solid var(--axis); color: var(--ink-2); justify-self: end; }
	.kind.measured { border-color: var(--ink); color: var(--ink); font-weight: 600; }
	.kind.synthetic { border-style: dashed; }
	.count, .span { font-size: 12px; color: var(--ink-2); }
	.span { justify-self: end; }
	.count { text-align: left; }
	.filters { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
	select { flex: 1 1 100%; min-width: 0; max-width: 100%; padding: 5px 6px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface); }
	.seg { display: flex; flex: 1 1 100%; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
	.seg button { flex: 1; border: 0; background: var(--surface); padding: 5px 8px; font-size: 12px; color: var(--ink-2); }
	.seg button + button { border-left: 1px solid var(--border); }
	.seg button.on { background: var(--ink); color: var(--surface); }
	.stale { opacity: 0.45; transition: opacity 0.15s; }
	td { padding: 5px 0; border-bottom: 1px solid var(--grid); vertical-align: top; font-size: 13px; }
	td small { display: block; color: var(--muted); font-size: 11.5px; }
	td.num { white-space: nowrap; padding-left: 10px; }
	.tag { font-size: 10.5px; border: 1px solid var(--axis); border-radius: 4px; padding: 0 4px; margin-left: 4px; color: var(--ink-2); }
	pre { margin: 0; padding: 8px; background: var(--wash); border-radius: 6px; font-size: 11.5px; white-space: pre-wrap; overflow-wrap: anywhere; max-height: 160px; overflow: auto; }
	.declared { margin: 0; padding: 0; list-style: none; display: flex; flex-wrap: wrap; gap: 4px; }
	.declared li { font-size: 12px; padding: 2px 7px; background: var(--wash); border-radius: 10px; }
</style>
