<script lang="ts">
	import { fmtDateTime, fmtNum, fmtTick } from '#lib/format.ts';

	interface Point { t: number; avg: number; min: number; max: number; n: number }
	interface Props {
		points: Point[];
		/** bucket width in seconds: gaps longer than this break the line */
		bucket: number;
		from: number;
		to: number;
		unit: string;
		label: string;
	}
	let { points, bucket, from, to, unit, label }: Props = $props();

	const H = 230;
	const M = { top: 12, right: 14, bottom: 26, left: 58 };
	let width = $state(480);
	let hover = $state<Point | null>(null);

	const innerW = $derived(Math.max(10, width - M.left - M.right));
	const innerH = H - M.top - M.bottom;
	const span = $derived(Math.max(1, to - from));

	function niceTicks(lo: number, hi: number, count = 4) {
		if (lo === hi) {
			const pad = Math.abs(lo) * 0.1 || 1;
			lo -= pad;
			hi += pad;
		}
		const raw = (hi - lo) / count;
		const mag = 10 ** Math.floor(Math.log10(raw));
		const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw)!;
		const start = Math.floor(lo / step) * step;
		const ticks: number[] = [];
		for (let v = start; v < hi + step * 0.999; v += step) ticks.push(+v.toPrecision(12));
		return ticks;
	}

	const yTicks = $derived.by(() => {
		if (!points.length) return [0, 1];
		let lo = Infinity, hi = -Infinity;
		for (const p of points) {
			if (p.min < lo) lo = p.min;
			if (p.max > hi) hi = p.max;
		}
		return niceTicks(lo, hi);
	});
	const y0 = $derived(yTicks[0]);
	const y1 = $derived(yTicks[yTicks.length - 1]);
	const x = (t: number) => M.left + ((t - from) / span) * innerW;
	const y = (v: number) => M.top + innerH - ((v - y0) / (y1 - y0 || 1)) * innerH;
	const xTicks = $derived(Array.from({ length: 5 }, (_, i) => from + (span * i) / 4));

	// Split into runs so that missing periods show as gaps, not as straight lines.
	const runs = $derived.by(() => {
		const out: Point[][] = [];
		let cur: Point[] = [];
		for (const p of points) {
			if (cur.length && p.t - cur[cur.length - 1].t > bucket * 1.5) {
				out.push(cur);
				cur = [];
			}
			cur.push(p);
		}
		if (cur.length) out.push(cur);
		return out;
	});
	const linePath = (r: Point[]) => r.map((p, i) => `${i ? 'L' : 'M'}${x(p.t).toFixed(1)},${y(p.avg).toFixed(1)}`).join('');
	const bandPath = (r: Point[]) =>
		r.map((p, i) => `${i ? 'L' : 'M'}${x(p.t).toFixed(1)},${y(p.max).toFixed(1)}`).join('') +
		[...r].reverse().map((p) => `L${x(p.t).toFixed(1)},${y(p.min).toFixed(1)}`).join('') + 'Z';
	const hasBand = $derived(points.some((p) => p.n > 1 && p.max !== p.min));

	function nearest(clientX: number, target: SVGElement) {
		if (!points.length) return null;
		const box = target.getBoundingClientRect();
		const t = from + ((clientX - box.left - M.left) / innerW) * span;
		let lo = 0, hi = points.length - 1;
		while (hi - lo > 1) {
			const mid = (lo + hi) >> 1;
			if (points[mid].t < t) lo = mid;
			else hi = mid;
		}
		return Math.abs(points[lo].t - t) <= Math.abs(points[hi].t - t) ? points[lo] : points[hi];
	}
	function step(dir: number) {
		if (!points.length) return;
		const i = hover ? points.indexOf(hover) : dir > 0 ? -1 : points.length;
		hover = points[Math.min(points.length - 1, Math.max(0, i + dir))];
	}
	const tipLeft = $derived(hover ? Math.min(Math.max(x(hover.t), 90), width - 90) : 0);
	const u = $derived(unit ? ` ${unit}` : '');
</script>

<div class="chart" bind:clientWidth={width}>
	<!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
	<svg
		{width}
		height={H}
		role="img"
		aria-label="{label} over time. Use left and right arrow keys to read values."
		tabindex="0"
		onpointermove={(e) => (hover = nearest(e.clientX, e.currentTarget))}
		onpointerleave={() => (hover = null)}
		onblur={() => (hover = null)}
		onkeydown={(e) => {
			if (e.key === 'ArrowRight') step(1);
			else if (e.key === 'ArrowLeft') step(-1);
		}}
	>
		{#each yTicks as t (t)}
			<line class="grid" x1={M.left} x2={M.left + innerW} y1={y(t)} y2={y(t)} />
			<text class="tick" x={M.left - 8} y={y(t)} dy="0.32em" text-anchor="end">{fmtNum(t)}</text>
		{/each}
		<line class="axis" x1={M.left} x2={M.left + innerW} y1={M.top + innerH} y2={M.top + innerH} />
		{#each xTicks as t, i (i)}
			<text class="tick" x={x(t)} y={H - 8} text-anchor={i === 0 ? 'start' : i === 4 ? 'end' : 'middle'}>
				{fmtTick(t, span)}
			</text>
		{/each}

		{#each runs as r, i (i)}
			{#if hasBand}<path class="band" d={bandPath(r)} />{/if}
			{#if r.length === 1}
				<circle class="dot" cx={x(r[0].t)} cy={y(r[0].avg)} r="3" />
			{:else}
				<path class="line" d={linePath(r)} />
			{/if}
		{/each}

		{#if hover}
			<line class="cross" x1={x(hover.t)} x2={x(hover.t)} y1={M.top} y2={M.top + innerH} />
			<circle class="dot ring" cx={x(hover.t)} cy={y(hover.avg)} r="4" />
		{/if}
	</svg>

	{#if hover}
		<div class="tip" style:left="{tipLeft}px">
			<strong>{fmtNum(hover.avg)}{u}</strong>
			{#if hover.n > 1}<span>range {fmtNum(hover.min)} to {fmtNum(hover.max)} · {hover.n} messages</span>{/if}
			<span>{fmtDateTime(hover.t)}</span>
		</div>
	{/if}
	{#if !points.length}
		<p class="empty">No values in this period.</p>
	{/if}
</div>

<details>
	<summary>Show as table ({points.length} rows)</summary>
	<div class="scroll">
		<table>
			<thead><tr><th>Time (IST)</th><th class="num">Average{u}</th><th class="num">Min</th><th class="num">Max</th></tr></thead>
			<tbody>
				{#each points as p (p.t)}
					<tr><td>{fmtDateTime(p.t)}</td><td class="num">{fmtNum(p.avg)}</td><td class="num">{fmtNum(p.min)}</td><td class="num">{fmtNum(p.max)}</td></tr>
				{/each}
			</tbody>
		</table>
	</div>
</details>

<style>
	.chart { position: relative; width: 100%; }
	svg { display: block; touch-action: pan-y; }
	.grid { stroke: var(--grid); stroke-width: 1; }
	.axis { stroke: var(--axis); stroke-width: 1; }
	.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
	.line { fill: none; stroke: var(--series-1); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
	.band { fill: var(--series-1); opacity: 0.16; }
	.dot { fill: var(--series-1); }
	.dot.ring { stroke: var(--surface); stroke-width: 2; }
	.cross { stroke: var(--axis); stroke-width: 1; }
	.tip {
		position: absolute; top: 0; transform: translateX(-50%);
		display: grid; gap: 1px; padding: 6px 8px; pointer-events: none; white-space: nowrap;
		background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12); font-size: 12px; color: var(--ink-2);
	}
	.tip strong { color: var(--ink); font-size: 13px; }
	.empty { position: absolute; inset: 40% 0 auto; text-align: center; color: var(--muted); }
	details { margin-top: 4px; font-size: 12px; color: var(--ink-2); }
	summary { cursor: pointer; }
	.scroll { max-height: 220px; overflow: auto; margin-top: 6px; }
	th, td { padding: 3px 6px; border-bottom: 1px solid var(--grid); text-align: left; }
	th.num, td.num { text-align: right; }
	th { position: sticky; top: 0; background: var(--surface); font-weight: 600; }
</style>
