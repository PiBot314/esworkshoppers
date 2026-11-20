<script lang="ts">
	// Two series ("today" vs "scenario") on one axis: as paired bars per category, or as two lines over time.
	import { fmtNum } from '#lib/format.ts';

	interface Row { key: string | number; label: string; before: number; after: number; extra?: string }
	interface Props {
		kind: 'bars' | 'lines';
		rows: Row[];
		unit: string;
		names: [string, string];
		title: string;
		/** show only every n-th x label */
		labelEvery?: number;
	}
	let { kind, rows, unit, names, title, labelEvery = 1 }: Props = $props();

	const H = 240;
	const M = { top: 10, right: 12, bottom: 26, left: 52 };
	let width = $state(600);
	let hover = $state<number | null>(null);

	const innerW = $derived(Math.max(10, width - M.left - M.right));
	const innerH = H - M.top - M.bottom;
	const ticks = $derived.by(() => {
		const hi = Math.max(1, ...rows.map((r) => Math.max(r.before, r.after)));
		const raw = hi / 4;
		const mag = 10 ** Math.floor(Math.log10(raw));
		const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw)!;
		const out: number[] = [];
		for (let v = 0; v < hi + step * 0.999; v += step) out.push(+v.toPrecision(12));
		return out;
	});
	const top = $derived(ticks[ticks.length - 1]);
	const y = (v: number) => M.top + innerH - (v / top) * innerH;
	const band = $derived(innerW / Math.max(1, rows.length));
	const cx = (i: number) => M.left + (kind === 'bars' ? band * (i + 0.5) : rows.length > 1 ? (innerW * i) / (rows.length - 1) : innerW / 2);
	const barW = $derived(Math.max(3, Math.min(18, (band - 8) / 2 - 1)));
	const path = (k: 'before' | 'after') => rows.map((r, i) => `${i ? 'L' : 'M'}${cx(i).toFixed(1)},${y(r[k]).toFixed(1)}`).join('');
	// Rounded top, square bottom: the bar stays anchored to the baseline.
	function bar(x: number, v: number) {
		const h = Math.max(0, M.top + innerH - y(v)), r = Math.min(3, barW / 2, h);
		const b = M.top + innerH;
		return `M${x},${b}v${-(h - r)}q0,${-r} ${r},${-r}h${barW - 2 * r}q${r},0 ${r},${r}v${h - r}z`;
	}
	function pick(e: PointerEvent & { currentTarget: SVGElement }) {
		const px = e.clientX - e.currentTarget.getBoundingClientRect().left - M.left;
		const i = kind === 'bars' ? Math.floor(px / band) : Math.round((px / innerW) * (rows.length - 1));
		hover = rows.length ? Math.min(rows.length - 1, Math.max(0, i)) : null;
	}
	const u = $derived(unit ? ` ${unit}` : '');
	const tipLeft = $derived(hover == null ? 0 : Math.min(Math.max(cx(hover), 100), width - 100));
</script>

<div class="legend" aria-hidden="true">
	<span><i class="k1" class:line={kind === 'lines'}></i>{names[0]}</span>
	<span><i class="k2" class:line={kind === 'lines'}></i>{names[1]}</span>
</div>
<div class="chart" bind:clientWidth={width}>
	<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
	<svg
		{width} height={H} role="img" tabindex="0"
		aria-label="{title}. Use left and right arrow keys to read values."
		onpointermove={pick}
		onpointerleave={() => (hover = null)}
		onblur={() => (hover = null)}
		onkeydown={(e) => {
			if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
			const d = e.key === 'ArrowRight' ? 1 : -1;
			hover = Math.min(rows.length - 1, Math.max(0, (hover ?? (d > 0 ? -1 : rows.length)) + d));
		}}
	>
		{#each ticks as t (t)}
			<line class="grid" x1={M.left} x2={M.left + innerW} y1={y(t)} y2={y(t)} />
			<text class="tick" x={M.left - 8} y={y(t)} dy="0.32em" text-anchor="end">{fmtNum(t)}</text>
		{/each}
		{#if kind === 'bars' && hover != null}
			<rect class="hl" x={M.left + band * hover} y={M.top} width={band} height={innerH} />
		{/if}
		{#if kind === 'bars'}
			{#each rows as r, i (r.key)}
				<path class="s1" d={bar(cx(i) - barW - 1, r.before)} />
				<path class="s2" d={bar(cx(i) + 1, r.after)} />
			{/each}
		{:else}
			<path class="l1" d={path('before')} />
			<path class="l2" d={path('after')} />
			{#if hover != null}
				<line class="cross" x1={cx(hover)} x2={cx(hover)} y1={M.top} y2={M.top + innerH} />
				<circle class="s1 ring" cx={cx(hover)} cy={y(rows[hover].before)} r="4" />
				<circle class="s2 ring" cx={cx(hover)} cy={y(rows[hover].after)} r="4" />
			{/if}
		{/if}
		<line class="axis" x1={M.left} x2={M.left + innerW} y1={M.top + innerH} y2={M.top + innerH} />
		{#each rows as r, i (r.key)}
			{#if i % labelEvery === 0}
				<text class="tick" x={cx(i)} y={H - 8} text-anchor="middle">{r.label}</text>
			{/if}
		{/each}
	</svg>
	{#if hover != null}
		{@const r = rows[hover]}
		<div class="tip" style:left="{tipLeft}px">
			<span class="when">{r.label}</span>
			<span><i class="k1 line"></i><strong>{fmtNum(r.before)}{u}</strong> {names[0]}</span>
			<span><i class="k2 line"></i><strong>{fmtNum(r.after)}{u}</strong> {names[1]}</span>
			{#if r.extra}<span>{r.extra}</span>{/if}
		</div>
	{/if}
</div>
<details>
	<summary>Show as table</summary>
	<div class="scroll">
		<table>
			<thead><tr><th></th><th class="num">{names[0]}{u}</th><th class="num">{names[1]}{u}</th></tr></thead>
			<tbody>
				{#each rows as r (r.key)}
					<tr><td>{r.label}</td><td class="num">{fmtNum(r.before)}</td><td class="num">{fmtNum(r.after)}</td></tr>
				{/each}
			</tbody>
		</table>
	</div>
</details>

<style>
	.legend { display: flex; gap: 16px; font-size: 12px; color: var(--ink-2); margin-bottom: 4px; }
	.legend span, .tip span { display: flex; align-items: center; gap: 6px; }
	i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; flex: none; }
	i.line { height: 2px; width: 14px; border-radius: 1px; }
	.k1 { background: var(--series-1); }
	.k2 { background: var(--series-2); }
	.chart { position: relative; width: 100%; }
	svg { display: block; touch-action: pan-y; }
	.grid { stroke: var(--grid); }
	.axis { stroke: var(--axis); }
	.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
	.s1 { fill: var(--series-1); }
	.s2 { fill: var(--series-2); }
	.l1, .l2 { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
	.l1 { stroke: var(--series-1); }
	.l2 { stroke: var(--series-2); }
	.ring { stroke: var(--surface); stroke-width: 2; }
	.hl { fill: var(--wash); }
	.cross { stroke: var(--axis); }
	.tip {
		position: absolute; top: 0; transform: translateX(-50%); display: grid; gap: 2px; padding: 6px 9px;
		pointer-events: none; white-space: nowrap; background: var(--surface); border: 1px solid var(--border);
		border-radius: 6px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12); font-size: 12px; color: var(--ink-2);
	}
	.tip strong { color: var(--ink); }
	.when { font-weight: 600; color: var(--ink); }
	details { margin-top: 4px; font-size: 12px; color: var(--ink-2); }
	summary { cursor: pointer; }
	.scroll { max-height: 220px; overflow: auto; margin-top: 6px; }
	th, td { padding: 3px 6px; border-bottom: 1px solid var(--grid); text-align: left; }
	th.num, td.num { text-align: right; }
	th { position: sticky; top: 0; background: var(--surface); font-weight: 600; }
</style>
