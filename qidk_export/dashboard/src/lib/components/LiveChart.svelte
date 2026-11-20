<script lang="ts" module>
	export interface Pt {
		t: number; actual_kW: number; forecast_kW: number; lo_kW: number; hi_kW: number;
		flagged: boolean; injected: string | null; backfill?: boolean;
	}
</script>

<script lang="ts">
	// Actual vs NPU forecast (made 15 min earlier) with the normal range, flagged readings, and the next-hour
	// forecast past "now". One y-axis (kW). Hover shows a crosshair and the reading's numbers.
	interface Props { points: Pt[]; ahead: { t: number; kW: number }[]; label: string }
	let { points, ahead, label }: Props = $props();

	const STEP = 900, IST = 19800;
	const H = 260, M = { top: 24, right: 64, bottom: 26, left: 52 };
	let width = $state(600);
	let hi = $state<number | null>(null);

	const hhmm = (t: number) => new Date((t + IST) * 1000).toISOString().slice(11, 16);
	const innerW = $derived(Math.max(10, width - M.left - M.right));
	const innerH = H - M.top - M.bottom;
	const now = $derived(points.at(-1)?.t ?? 0);
	const t0 = $derived(points[0]?.t ?? now - 86400);
	const t1 = $derived(Math.max(now, ahead.at(-1)?.t ?? now));

	function nice(lo: number, hi: number, count = 4) {
		if (!(hi > lo)) { const p = Math.abs(lo) * 0.1 || 1; lo -= p; hi += p; }
		const raw = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(raw));
		const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw)!;
		const ticks: number[] = [];
		for (let v = Math.floor(lo / step) * step; v < hi + step * 0.999; v += step) ticks.push(+v.toPrecision(12));
		return ticks;
	}
	const yTicks = $derived.by(() => {
		let lo = Infinity, hi = -Infinity;
		for (const p of points) { lo = Math.min(lo, p.actual_kW, p.lo_kW); hi = Math.max(hi, p.actual_kW, p.hi_kW); }
		for (const a of ahead) { lo = Math.min(lo, a.kW); hi = Math.max(hi, a.kW); }
		return Number.isFinite(lo) ? nice(Math.min(0, lo), hi) : [0, 1];
	});
	const y0 = $derived(yTicks[0]), y1 = $derived(yTicks.at(-1)!);
	const x = (t: number) => M.left + ((t - t0) / Math.max(1, t1 - t0)) * innerW;
	const y = (v: number) => M.top + innerH - ((v - y0) / (y1 - y0 || 1)) * innerH;
	const xTicks = $derived.by(() => {
		const out: number[] = [], every = (t1 - t0) > 12 * 3600 ? 6 * 3600 : 2 * 3600;
		for (let t = Math.ceil((t0 + IST) / every) * every - IST; t <= t1; t += every) out.push(t);
		return out;
	});

	// runs: consecutive points without gaps, so missing readings show as breaks
	const runs = $derived.by(() => {
		const out: Pt[][] = []; let cur: Pt[] = [];
		for (const p of points) { if (cur.length && p.t - cur.at(-1)!.t > STEP * 1.5) { out.push(cur); cur = []; } cur.push(p); }
		if (cur.length) out.push(cur);
		return out;
	});
	const line = (run: Pt[], k: 'actual_kW' | 'forecast_kW') => run.map((p, i) => `${i ? 'L' : 'M'}${x(p.t).toFixed(1)},${y(p[k]).toFixed(1)}`).join('');
	const band = (run: Pt[]) =>
		run.map((p, i) => `${i ? 'L' : 'M'}${x(p.t).toFixed(1)},${y(p.hi_kW).toFixed(1)}`).join('') +
		[...run].reverse().map((p) => `L${x(p.t).toFixed(1)},${y(p.lo_kW).toFixed(1)}`).join('') + 'Z';
	const aheadPath = $derived.by(() => {
		const last = points.at(-1);
		const pts = [...(last ? [{ t: last.t, kW: last.forecast_kW }] : []), ...ahead];
		return pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.t).toFixed(1)},${y(p.kW).toFixed(1)}`).join('');
	});
	const marks = $derived(points.filter((p) => p.flagged || p.injected));
	// direct labels at the line ends, pushed apart when they would overlap
	const endLabels = $derived.by(() => {
		const L = points.at(-1), A = ahead.at(-1);
		const out = [
			...(L ? [{ text: 'actual', x: x(L.t) + 6, y: y(L.actual_kW) }] : []),
			...(A ? [{ text: 'forecast', x: x(A.t) + 6, y: y(A.kW) }] : [])
		];
		if (out.length === 2 && Math.abs(out[0].y - out[1].y) < 13) {
			const mid = (out[0].y + out[1].y) / 2, up = out[0].y <= out[1].y ? 0 : 1;
			out[up].y = mid - 7; out[1 - up].y = mid + 7;
		}
		return out;
	});
	const hover = $derived(hi == null ? null : points[hi]);

	function move(e: PointerEvent) {
		if (!points.length) return;
		const r = (e.currentTarget as SVGElement).getBoundingClientRect();
		const t = t0 + ((e.clientX - r.left - M.left) / innerW) * (t1 - t0);
		let best = 0;
		for (let i = 1; i < points.length; i++) if (Math.abs(points[i].t - t) < Math.abs(points[best].t - t)) best = i;
		hi = Math.abs(points[best].t - t) < STEP * 2 ? best : null;
	}
	const status = (p: Pt) => (p.flagged ? 'Flagged' : 'Normal') + (p.injected ? ` · injected ${p.injected}` : '');
</script>

<figure>
	<div class="legend" aria-hidden="true">
		<span><i class="sw actual"></i>Actual</span>
		<span><i class="sw fc"></i>NPU forecast (15 min ahead)</span>
		<span><i class="sw band"></i>Normal range</span>
		<span><i class="sw flag"></i>Flagged</span>
		<span><i class="sw inj"></i>Injected test fault</span>
	</div>
	<div class="plot" bind:clientWidth={width}>
		{#if !points.length}
			<p class="empty">No checked readings yet. The first point appears after the next 15-minute check.</p>
		{:else}
			<svg {width} height={H} role="img" aria-label={label} onpointermove={move} onpointerleave={() => (hi = null)}>
				{#each yTicks as v (v)}
					<line class="grid" x1={M.left} x2={M.left + innerW} y1={y(v)} y2={y(v)} />
					<text class="tick" x={M.left - 8} y={y(v)} text-anchor="end" dominant-baseline="middle">{v}</text>
				{/each}
				{#each xTicks as t (t)}
					<text class="tick" x={x(t)} y={H - 8} text-anchor="middle">{hhmm(t)}</text>
				{/each}
				<text class="tick" x={M.left - 8} y={10} text-anchor="end">kW</text>

				{#each runs as run, i (i)}<path class="band" d={band(run)} />{/each}
				<line class="now" x1={x(now)} x2={x(now)} y1={M.top} y2={M.top + innerH} />
				<text class="tick" x={x(now) + 4} y={M.top + 10}>now</text>
				{#each runs as run, i (i)}<path class="fc" d={line(run, 'forecast_kW')} />{/each}
				<path class="fc ahead" d={aheadPath} />
				{#each runs as run, i (i)}<path class="actual" d={line(run, 'actual_kW')} />{/each}

				{#each marks as p (p.t)}
					{#if p.flagged}<circle class="flag" cx={x(p.t)} cy={y(p.actual_kW)} r="5" />{/if}
					{#if p.injected}<circle class="inj" cx={x(p.t)} cy={y(p.actual_kW)} r="8" />{/if}
				{/each}

				{#each endLabels as l (l.text)}
					<text class="end" x={l.x} y={l.y} dominant-baseline="middle">{l.text}</text>
				{/each}

				{#if hover}
					<line class="cross" x1={x(hover.t)} x2={x(hover.t)} y1={M.top} y2={M.top + innerH} />
					<circle class="dot" cx={x(hover.t)} cy={y(hover.actual_kW)} r="4" />
				{/if}
			</svg>
			{#if hover}
				<div class="tip" style:left="{Math.min(x(hover.t) + 12, width - 210)}px">
					<strong>{hhmm(hover.t)} IST</strong>
					<dl>
						<dt>Actual</dt><dd class="num">{hover.actual_kW} kW</dd>
						<dt>Forecast</dt><dd class="num">{hover.forecast_kW} kW</dd>
						<dt>Normal range</dt><dd class="num">{hover.lo_kW} – {hover.hi_kW}</dd>
						<dt>Status</dt><dd class:bad={hover.flagged}>{status(hover)}</dd>
					</dl>
					{#if hover.backfill}<small>Backfilled after a restart</small>{/if}
				</div>
			{/if}
		{/if}
	</div>
</figure>

<style>
	figure { margin: 0; }
	.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; color: var(--ink-2); margin-bottom: 6px; }
	.legend span { display: inline-flex; align-items: center; gap: 6px; }
	.sw { display: inline-block; width: 16px; height: 0; border-top: 2px solid; }
	.sw.actual { border-color: var(--series-1); }
	.sw.fc { border-color: var(--series-2); border-top-style: dashed; }
	.sw.band { height: 10px; border: 0; background: color-mix(in srgb, var(--series-2) 18%, transparent); }
	.sw.flag { width: 10px; height: 10px; border: 0; border-radius: 50%; background: var(--bad); }
	.sw.inj { width: 12px; height: 12px; border: 2px solid var(--ink); border-radius: 50%; }
	.plot { position: relative; min-height: 120px; }
	svg { display: block; touch-action: pan-y; }
	.grid { stroke: var(--grid); }
	.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
	.band { fill: color-mix(in srgb, var(--series-2) 16%, transparent); }
	.fc { fill: none; stroke: var(--series-2); stroke-width: 2; stroke-dasharray: 5 4; }
	.fc.ahead { opacity: 0.9; }
	.actual { fill: none; stroke: var(--series-1); stroke-width: 2; stroke-linejoin: round; }
	.now { stroke: var(--axis); stroke-dasharray: 2 3; }
	.flag { fill: var(--bad); stroke: var(--surface); stroke-width: 2; }
	.inj { fill: none; stroke: var(--ink); stroke-width: 2; }
	.end { fill: var(--ink-2); font-size: 11px; }
	.cross { stroke: var(--ink-2); stroke-width: 1; }
	.dot { fill: var(--series-1); stroke: var(--surface); stroke-width: 2; }
	.empty { color: var(--muted); padding: 40px 0; text-align: center; }
	.tip {
		position: absolute; top: 8px; width: 200px; pointer-events: none; padding: 8px 10px; border-radius: 8px;
		background: var(--surface); border: 1px solid var(--border); box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12); font-size: 12px;
	}
	.tip dl { display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; margin: 4px 0 0; }
	.tip dt { color: var(--muted); }
	.tip dd { margin: 0; }
	.tip .bad { color: var(--bad); font-weight: 600; }
	.tip small { color: var(--muted); }
</style>
