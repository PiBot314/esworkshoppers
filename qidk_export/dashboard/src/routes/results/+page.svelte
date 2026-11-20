<script lang="ts">
	// Results & interpretability: are the forecasters worth it, against the two baselines every model must beat?
	// Offline: 2025 test split (nz_live/scripts/offline_results.py). Live: shadow mode on the API (loop.ts liveResults).
	import { onMount } from 'svelte';
	import { chat, ui } from '#lib/ui.svelte.ts';

	type Dom = 'Solar' | 'Grid';
	type Score = Record<Dom, { R2: number; skill_vs_persistence: number }>;
	interface ORow { id: string; kind: 'baseline' | 'model'; horizon_steps?: number; '15min'?: Score; '1h'?: Score; '24h'?: Score }
	interface Offline {
		split: string; horizons: ('15min' | '1h' | '24h')[]; rows: ORow[];
		importance: { source: string; by_node: [string, number][]; by_quantity: [string, number][]; top_features: [string, number][] };
	}
	interface LRow { model: string; label: string; kind: string; n: number; mae_kW: number | null; rmse_kW: number | null; r2: number | null; skill_vs_persistence: number | null; flag_rate: number | null }
	interface Bundle {
		enabled: boolean; hours: number;
		live: Record<'grid' | 'pl00', { label: string; from: string | null; to: string | null; rows: LRow[] }>;
		offline: Offline | null;
		limits: Record<'grid' | 'pl00', { model: string; limit_kW: number[]; median_kW: number[] } | null>;
		models: { id: string; label: string }[];
	}

	let data = $state<Bundle | null>(null);
	let hours = $state(24);
	let failed = $state(false);
	async function load() {
		try { data = await (await fetch(`/api/live/results?hours=${hours}`)).json(); failed = false; } catch { failed = true; }
	}
	onMount(() => { load(); const t = setInterval(load, 60000); return () => clearInterval(t); });

	const H = { '15min': '+15 min', '1h': '1 hour', '24h': '24 hours' } as const;
	const name = (id: string) => ({ persistence: 'Persistence (next = current)', yesterday: 'Same time yesterday', lightgbm_24h: 'LightGBM, 24 h (CPU only)' } as Record<string, string>)[id]
		?? data?.models.find((m) => m.id === id)?.label ?? id;
	const fmt = (x: number | null | undefined, d = 3) => (x == null ? '–' : x.toFixed(d));
	const sk = (x: number | null | undefined) => (x == null ? '–' : `${x > 0 ? '+' : ''}${x.toFixed(2)}`);

	// headline verdicts: best model vs best baseline, per horizon and quantity
	const verdicts = $derived.by(() => {
		const o = data?.offline;
		if (!o) return [];
		const out: { h: string; dom: Dom; model: ORow; base: ORow; win: boolean }[] = [];
		for (const h of o.horizons)
			for (const dom of ['Solar', 'Grid'] as Dom[]) {
				const has = o.rows.filter((r) => r[h]);
				const best = (k: string) => has.filter((r) => r.kind === k).sort((a, b) => b[h]![dom].R2 - a[h]![dom].R2)[0];
				const model = best('model'), base = best('baseline');
				if (model && base) out.push({ h, dom, model, base, win: model[h]![dom].R2 > base[h]![dom].R2 });
			}
		return out;
	});

	function ask(q: string) { chat.pending = q; ui.chatOpen = true; }
</script>

<svelte:head><title>Results · Campus sensors</title></svelte:head>

<main>
	<header>
		<h1>Results: are the forecasters worth it?</h1>
		<p class="lede">
			Every model has to beat two forecasts that need no training: <strong>persistence</strong> (the next reading equals the
			current one) and <strong>same time yesterday</strong>. Skill = 1 − error(model) / error(persistence): above 0 means better than persistence.
		</p>
	</header>

	{#if failed && !data}
		<p class="panel warn">Can't load the results.</p>
	{:else if !data}
		<p class="muted">Loading…</p>
	{:else}
		{#if data.offline}
			<section class="panel">
				<h2>Verdict on 2025 data</h2>
				<p class="sub">{data.offline.split}. R² of the forecast (1 = perfect). Grid = substation draw + KH95; Solar = campus total.</p>
				<div class="scroll"><table>
					<thead><tr><th>Horizon</th><th>Quantity</th><th>Best model</th><th class="num">R²</th><th>Best baseline</th><th class="num">R²</th><th>Who wins</th></tr></thead>
					<tbody>
						{#each verdicts as v (v.h + v.dom)}
							<tr>
								<td>{H[v.h as keyof typeof H]}</td><td>{v.dom}</td>
								<td>{name(v.model.id)}</td><td class="num">{fmt(v.model[v.h as '1h']![v.dom].R2)}</td>
								<td>{name(v.base.id)}</td><td class="num">{fmt(v.base[v.h as '1h']![v.dom].R2)}</td>
								<td><span class="pill {v.win ? 'ok' : 'bad'}">{v.win ? '✓ model' : '✗ baseline'}</span></td>
							</tr>
						{/each}
					</tbody>
				</table></div>
				<p class="note">
					<strong>Reading it:</strong> at +15 minutes nothing beats persistence: campus load and solar change little in 15 minutes.
					The models earn their place for solar a few steps ahead (1 hour), and for grid draw a day ahead, where every forecast is weak.
					For 24-hour solar, "same time yesterday" wins: without weather data no model knows tomorrow's clouds.
				</p>
				<button class="link" onclick={() => ask('Are the forecasting models better than just using the last reading?')}>Ask the assistant about this</button>
			</section>

			<section class="panel">
				<h2>All models and baselines, 2025 test split</h2>
				<div class="scroll"><table class="grid">
					<thead>
						<tr><th rowspan="2">Forecast</th>{#each data.offline.horizons as h (h)}<th colspan="4" class="c">{H[h]}</th>{/each}</tr>
						<tr>{#each data.offline.horizons as h (h)}<th class="num">Solar R²</th><th class="num">skill</th><th class="num">Grid R²</th><th class="num">skill</th>{/each}</tr>
					</thead>
					<tbody>
						{#each data.offline.rows as r (r.id)}
							<tr class:base={r.kind === 'baseline'}>
								<td>{name(r.id)}{#if r.kind === 'baseline'} <span class="tag">baseline</span>{/if}</td>
								{#each data.offline.horizons as h (h)}
									{#if r[h]}
										{#each ['Solar', 'Grid'] as const as d (d)}
											<td class="num">{fmt(r[h]![d].R2)}</td>
											<td class="num {r[h]![d].skill_vs_persistence > 0.005 ? 'pos' : r[h]![d].skill_vs_persistence < -0.005 ? 'neg' : ''}">{sk(r[h]![d].skill_vs_persistence)}</td>
										{/each}
									{:else}
										<td class="num muted" colspan="4">not a {H[h]} model</td>
									{/if}
								{/each}
							</tr>
						{/each}
					</tbody>
				</table></div>
				<p class="sub">Skill is computed over every step of the horizon. Persistence's skill is 0 by definition. LightGBM uses only the last reading of the window.</p>
			</section>
		{/if}

		<section class="panel">
			<div class="row">
				<h2>Live, on the campus API (shadow mode)</h2>
				<label class="inline">Window
					<select bind:value={hours} onchange={load}>
						<option value={24}>last 24 hours</option><option value={72}>last 3 days</option><option value={168}>last 7 days</option>
					</select>
				</label>
			</div>
			<p class="sub">
				Every 15 minutes the active model, every other Grid model (in shadow: forecast, never alerted) and both baselines forecast the same reading.
				Synthetic data replayed live, so "would flag" counts are false alarms by construction. Injected test faults are left out.
			</p>
			{#if !data.enabled}
				<p class="muted">The live loop is off on this server.</p>
			{:else}
				{#each ['grid', 'pl00'] as const as sig (sig)}
					{@const L = data.live[sig]}
					<h3>{sig === 'grid' ? 'Campus grid draw' : 'Palash solar plant'} <span class="muted">{L.from ? `${L.from} → ${L.to}` : 'no data yet'}</span></h3>
					<div class="scroll"><table>
						<thead><tr><th>Forecast</th><th>Role</th><th class="num">Readings</th><th class="num">MAE kW</th><th class="num">RMSE kW</th><th class="num">R²</th><th class="num">Skill vs persistence</th><th class="num">Would flag</th></tr></thead>
						<tbody>
							{#each L.rows as r (r.model)}
								<tr class:active={r.kind === 'active'} class:base={r.kind === 'baseline'}>
									<td>{r.label}</td><td>{r.kind}</td><td class="num">{r.n}</td><td class="num">{fmt(r.mae_kW, 1)}</td><td class="num">{fmt(r.rmse_kW, 1)}</td>
									<td class="num">{fmt(r.r2)}</td>
									<td class="num {r.skill_vs_persistence != null && r.skill_vs_persistence > 0.005 ? 'pos' : r.skill_vs_persistence != null && r.skill_vs_persistence < -0.005 ? 'neg' : ''}">{sk(r.skill_vs_persistence)}</td>
									<td class="num">{r.flag_rate == null ? '–' : `${(r.flag_rate * 100).toFixed(1)} %`}</td>
								</tr>
							{/each}
						</tbody>
					</table></div>
				{/each}
			{/if}
		</section>

		<section class="panel">
			<h2>What the models look at</h2>
			{#if data.offline}
				<p class="sub">{data.offline.importance.source}</p>
				<div class="imp">
					{#each [['By meter', data.offline.importance.by_node], ['By kind of input', data.offline.importance.by_quantity]] as [title, rows] (title)}
						<div>
							<h3>{title}</h3>
							{#each rows as [k, v] ((title as string) + k)}
								<div class="bar"><span class="k">{k}</span><span class="track"><i style:width="{Math.max(1, (v as number) * 100 / ((rows as [string, number][])[0][1]))}%"></i></span><span class="v">{((v as number) * 100).toFixed(1)}%</span></div>
							{/each}
						</div>
					{/each}
				</div>
				<p class="note">
					Time of day, week and year carry most of the weight, then the substation's own recent readings: the model mostly learns
					the daily and seasonal shape plus "where load is now", which is close to what persistence and yesterday already capture.
				</p>
			{/if}

			{#each ['grid', 'pl00'] as const as sig (sig)}
				{@const lim = data.limits[sig]}
				{#if lim}
					{@const max = Math.max(...lim.limit_kW)}
					<h3>Allowed deviation by hour of day: {sig === 'grid' ? 'grid draw' : 'Palash solar'} <span class="muted">({name(lim.model)})</span></h3>
					<div class="hours" role="img" aria-label="Allowed deviation per IST hour">
						{#each lim.limit_kW as v, h (h)}
							<div class="hb" title="{String(h).padStart(2, '0')}:00 IST: ±{v} kW (typical error {lim.median_kW[h]} kW)"><i style:height="{(v / max) * 100}%"></i><span>{h % 3 === 0 ? h : ''}</span></div>
						{/each}
					</div>
					<p class="sub">Taller = the model was less sure at that hour on 2025 validation data, so a reading must be further off to be flagged. Hover for the numbers.</p>
				{/if}
			{/each}
		</section>
	{/if}
</main>

<style>
	main { max-width: 1200px; margin: 0 auto; padding: 18px 16px 40px; display: flex; flex-direction: column; gap: 14px; }
	h1 { font-size: 20px; font-weight: 650; }
	h2 { font-size: 15px; font-weight: 650; }
	h3 { font-size: 13px; font-weight: 650; margin: 12px 0 4px; }
	.lede { color: var(--ink-2); max-width: 85ch; margin-top: 4px; }
	.panel { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; min-width: 0; }
	.warn { background: var(--warn-bg); color: var(--warn-ink); }
	.sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
	.note { font-size: 13px; color: var(--ink-2); margin-top: 10px; max-width: 95ch; }
	.muted { color: var(--muted); font-weight: 400; }
	.scroll { overflow-x: auto; }
	table { margin-top: 8px; font-size: 13px; }
	th { text-align: left; font-weight: 550; color: var(--muted); font-size: 12px; border-bottom: 1px solid var(--border); padding: 4px 8px 4px 0; white-space: nowrap; }
	th.c { text-align: center; }
	td { padding: 5px 8px 5px 0; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; white-space: nowrap; }
	tr.base td { background: var(--wash); }
	tr.active td { font-weight: 650; }
	.pos { color: var(--ok); font-weight: 600; }
	.neg { color: var(--bad); }
	.tag { font-size: 11px; color: var(--muted); border: 1px solid var(--border); border-radius: 4px; padding: 0 4px; }
	.pill { padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; }
	.pill.ok { background: color-mix(in srgb, var(--ok) 14%, transparent); color: var(--ok); }
	.pill.bad { background: color-mix(in srgb, var(--bad) 14%, transparent); color: var(--bad); }
	.link { border: 0; background: none; color: var(--series-1); padding: 0; font-weight: 550; margin-top: 8px; }
	.row { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px; }
	.inline { font-size: 12px; color: var(--ink-2); display: flex; gap: 6px; align-items: center; }
	select { padding: 3px 6px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface); }
	.imp { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
	.bar { display: grid; grid-template-columns: minmax(0, 11rem) 1fr 3.5rem; gap: 8px; align-items: center; font-size: 12px; margin: 3px 0; }
	.bar .k { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--ink-2); }
	.track { height: 8px; background: var(--wash); border-radius: 4px; overflow: hidden; }
	.track i { display: block; height: 100%; background: var(--series-1); border-radius: 4px; }
	.bar .v { text-align: right; font-variant-numeric: tabular-nums; color: var(--ink-2); }
	.hours { display: grid; grid-template-columns: repeat(24, 1fr); gap: 2px; height: 90px; align-items: end; margin-top: 6px; }
	.hb { height: 100%; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; }
	.hb i { display: block; width: 100%; background: var(--series-2); border-radius: 3px 3px 0 0; min-height: 2px; }
	.hb span { font-size: 10px; color: var(--muted); height: 14px; }
	@media (max-width: 800px) { .imp { grid-template-columns: 1fr; } }
</style>
