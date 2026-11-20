<script lang="ts">
	// Front page: the live NPU forecasting loop (src/lib/server/live). Polls status + history every 30 s.
	import { onMount } from 'svelte';
	import LiveChart, { type Pt } from '#lib/components/LiveChart.svelte';
	import { chat, ui } from '#lib/ui.svelte.ts';

	type Sig = 'grid' | 'pl00';
	interface Latest {
		reading_time: string; actual_kW: number; forecast_for_this_reading_kW: number; limit_kW: number; flagged: boolean;
		next_hour_forecast_kW: { time: string; kW: number }[];
	}
	interface Status {
		enabled: boolean; source: string;
		api: { ok: boolean; error: string | null };
		npu: { ok: boolean; variant: string | null; last_ms: number | null; error: string | null };
		now: string | null; last_reading_checked: string | null; next_check: string | null;
		minutes_since_last_reading: Record<string, number | null>;
		latest: Partial<Record<Sig, Latest>> & { campus_solar?: { note: string; next_hour_forecast_kW: { time: string; kW: number }[] } };
		last_24h: { natural_flags: number; injected_faults: number };
		pending_injection: Partial<Record<Sig, { kind: string; left: number }>>;
	}
	interface Ev { id: number; time: string; signal: Sig; actual_kW: number; forecast_kW: number; excess_kW: number; flagged: boolean; injected: string | null; npu_ms: number | null; late?: boolean }
	interface Hist { backfill: string; signals: Record<Sig, { model: string; label: string; natural_flag_rate: string; points: Pt[] }> }
	interface Model {
		id: string; label: string; family: string; horizon: number; signals: Sig[]; builtin: boolean;
		test_first_step_R2?: Record<string, number>; sqnr_db?: number; npu_ms?: number;
	}
	interface Models { models: Model[]; active: Record<Sig, Model> }

	const SIG: Record<Sig, { title: string; sub: string; faults: [string, string][] }> = {
		grid: { title: 'Campus grid draw', sub: 'Substation + KH95 · 11 meters, 58 inputs', faults: [['spike', 'Spike +50 %'], ['dropout', 'Dropout'], ['damage40', '−40 %']] },
		pl00: { title: 'Palash solar plant', sub: 'SL-PL00-00 · 1 meter, 10 inputs', faults: [['dropout', 'Dropout'], ['damage40', 'Panel damage −40 %'], ['spike', 'Spike +50 %']] }
	};

	let status = $state<Status | null>(null);
	let hist = $state<Hist | null>(null);
	let events = $state<Ev[]>([]);
	let models = $state<Models | null>(null);
	let switching = $state(false);
	let failed = $state(false);
	let note = $state('');

	async function poll() {
		try {
			const [s, h, e, m] = await Promise.all(['/api/live/status', '/api/live/history?hours=24', '/api/live/events', '/api/live/models'].map((u) => fetch(u).then((r) => r.json())));
			status = s; hist = h; events = (e.events as Ev[]).slice().reverse(); models = m; failed = false;
		} catch { failed = true; }
	}
	onMount(() => { poll(); const t = setInterval(poll, 30000); return () => clearInterval(t); });

	const ahead = (sig: Sig) => {
		const last = hist?.signals[sig]?.points.at(-1);
		const f = status?.latest[sig]?.next_hour_forecast_kW ?? [];
		return last ? f.map((p, i) => ({ t: last.t + (i + 1) * 900, kW: p.kW })) : [];
	};
	const hhmm = (s: string | null | undefined) => s?.slice(11, 16) ?? '–';
	const dataAge = $derived(status?.minutes_since_last_reading['EM-CR-SB00-00'] ?? null);
	const variantLabel = (v: string | null) => (v === 'aimet_w8a16' ? 'Hexagon NPU · AIMET W8A16' : v === 'fp32' ? 'CPU fallback · FP32' : '–');

	async function inject(signal: Sig, kind: string, label: string) {
		const r = await fetch('/api/live/inject', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ signal, kind, steps: 1 }) });
		const j = await r.json().catch(() => ({}));
		note = r.ok ? `${label} queued on ${SIG[signal].title.toLowerCase()}: applies to the ${hhmm(j.applies_from)} reading.` : (j.message ?? 'Failed');
		poll();
	}
	const horizonLabel = (h: number) => (h <= 4 ? '1 h' : '24 h');
	const r2 = (m: Model, d: string) => m.test_first_step_R2?.[d]?.toFixed(2) ?? '–';
	const forSig = (sig: Sig) => models?.models.filter((m) => m.signals.includes(sig)) ?? [];
	async function switchTo(sig: Sig, id: string) {
		switching = true;
		const r = await fetch('/api/live/models', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ signal: sig, model: id }) });
		const j = await r.json().catch(() => ({}));
		const m = models?.models.find((x) => x.id === id);
		note = j.ok ? `${SIG[sig].title}: now forecast by ${m?.label ?? id}. The last 24 h are being re-forecast with it; the next live check uses it.` : (j.error ?? 'Switch failed');
		switching = false;
		poll();
	}

	function ask(e: Ev) {
		chat.pending = `What happened with ${e.signal === 'grid' ? 'grid draw' : 'palash solar'} at ${hhmm(e.time)}? (live anomaly #${e.id})`;
		ui.chatOpen = true;
	}
</script>

<svelte:head><title>Live · Campus sensors</title></svelte:head>

<main>
	<header>
		<div>
			<h1>Live forecasting on the Hexagon NPU</h1>
			<p class="lede">
				Every 15 minutes the board fetches the campus API, runs the forecasting LSTMs on its NPU, and flags readings
				that land outside the normal range around the forecast.
			</p>
		</div>
		<p class="src">Data: campus API, <strong>synthetic 2026 data replayed in real time</strong>. A flag means “unusual compared with the forecast”, not a confirmed fault.</p>
	</header>

	{#if failed && !status}
		<p class="panel warn">Can't reach the dashboard server.</p>
	{:else if status && !status.enabled}
		<p class="panel warn">The live loop is off on this server (start it with <code>LIVE=1</code>). The map, Net zero and the NPU lab still work.</p>
	{:else if status}
		<section class="tiles" aria-label="System status">
			<div class="tile">
				<span class="k">Inference</span>
				<span class="v"><i class="st {status.npu.ok ? 'ok' : 'bad'}" aria-hidden="true"></i>{status.npu.last_ms != null ? `${status.npu.last_ms.toFixed(1)} ms` : status.npu.ok ? 'OK' : '–'}</span>
				<span class="s">{status.npu.ok ? variantLabel(status.npu.variant) : (status.npu.error ?? 'not run yet')}</span>
			</div>
			<div class="tile">
				<span class="k">Data feed</span>
				<span class="v"><i class="st {status.api.ok && dataAge != null && dataAge < 20 ? 'ok' : 'bad'}" aria-hidden="true"></i>{dataAge != null ? `${dataAge} min old` : '–'}</span>
				<span class="s">{status.api.ok ? `Campus API reachable${status.api.route ? ` · ${status.api.route}` : ''}` : (status.api.error ?? 'no data yet')}</span>
			</div>
			<div class="tile">
				<span class="k">Checks</span>
				<span class="v">{hhmm(status.last_reading_checked)} <small>→ {hhmm(status.next_check)}</small></span>
				<span class="s">last reading checked → next check (IST)</span>
			</div>
			<div class="tile">
				<span class="k">Flags, last 24 h</span>
				<span class="v">{status.last_24h.natural_flags} <small>natural</small> · {status.last_24h.injected_faults} <small>injected</small></span>
				<span class="s">natural flags on synthetic data are false alarms</span>
			</div>
		</section>

		<section class="sigs">
			{#each ['grid', 'pl00'] as const as sig (sig)}
				{@const L = status.latest[sig]}
				{@const H = hist?.signals[sig]}
				<article class="panel">
					<div class="head">
						<div>
							<h2>{SIG[sig].title}</h2>
							<p class="sub">{SIG[sig].sub}</p>
							{#if forSig(sig).length > 1}
								<label class="model">
									<span>Model</span>
									<select disabled={switching} value={models?.active[sig]?.id} onchange={(e) => switchTo(sig, e.currentTarget.value)}>
										{#each forSig(sig) as m (m.id)}
											<option value={m.id}>{m.label} · {horizonLabel(m.horizon)}{m.test_first_step_R2 ? ` · Grid R² ${r2(m, 'Grid')}` : ''}</option>
										{/each}
									</select>
								</label>
							{:else if models}
								<p class="sub">Model: {models.active[sig]?.label ?? models.active[sig]?.id} (the only model for this plant)</p>
							{/if}
						</div>
						{#if L}
							<span class="pill {L.flagged ? 'bad' : 'ok'}">{L.flagged ? '⚠ Flagged' : '✓ Normal'}</span>
						{/if}
					</div>
					{#if L}
						<dl class="nums">
							<div><dt>Actual {hhmm(L.reading_time)}</dt><dd>{L.actual_kW} <small>kW</small></dd></div>
							<div><dt>NPU forecast</dt><dd>{L.forecast_for_this_reading_kW} <small>kW</small></dd></div>
							<div><dt>Allowed deviation</dt><dd>±{L.limit_kW} <small>kW</small></dd></div>
							<div><dt>Next hour</dt><dd>{L.next_hour_forecast_kW.at(-1)?.kW ?? '–'} <small>kW at {hhmm(L.next_hour_forecast_kW.at(-1)?.time)}</small></dd></div>
						</dl>
					{/if}
					{#if !L}<p class="muted small">Latest numbers appear after the next live check{status.next_check ? ` (${hhmm(status.next_check)})` : ''}.</p>{/if}
					<LiveChart points={H?.points ?? []} ahead={ahead(sig)} label="{SIG[sig].title}: actual vs NPU forecast, last 24 hours" />
					{#if hist?.backfill === 'running'}<p class="muted">Filling in the last 24 h from past windows…</p>{/if}
					<div class="faults">
						<span class="muted">Test the detector:</span>
						{#each SIG[sig].faults as [kind, label] (kind)}
							<button onclick={() => inject(sig, kind, label)}>{label}</button>
						{/each}
						{#if status.pending_injection[sig]}<span class="muted">queued: {status.pending_injection[sig]!.kind}</span>{/if}
					</div>
				</article>
			{/each}
		</section>
		{#if note}<p class="note" role="status">{note}</p>{/if}

		<section class="bottom">
			<article class="panel">
				<h2>Recent flags</h2>
				{#if events.length}
					<table>
						<thead><tr><th>Time (IST)</th><th>Signal</th><th class="num">Actual</th><th class="num">Forecast</th><th class="num">Past limit</th><th>Kind</th><th></th></tr></thead>
						<tbody>
							{#each events.slice(0, 8) as e (e.id)}
								<tr>
									<td>{e.time.slice(5, 16)}</td>
									<td>{e.signal === 'grid' ? 'Grid' : 'Palash solar'}</td>
									<td class="num">{e.actual_kW} kW</td>
									<td class="num">{e.forecast_kW} kW</td>
									<td class="num">{e.flagged ? `${e.excess_kW} kW` : 'not flagged'}</td>
									<td>{e.injected ? `injected ${e.injected}` : e.late ? 'natural (found after an outage)' : 'natural'}</td>
									<td><button class="link" onclick={() => ask(e)}>Ask</button></td>
								</tr>
							{/each}
						</tbody>
					</table>
				{:else}
					<p class="muted">No flags yet.</p>
				{/if}
			</article>
			<article class="panel">
				<h2>Campus solar, next hour</h2>
				<p class="sub">all_sensors_1h · forecast only</p>
				{#if status.latest.campus_solar}
					<table>
						<tbody>
							{#each status.latest.campus_solar.next_hour_forecast_kW as f (f.time)}
								<tr><td>{hhmm(f.time)}</td><td class="num">{f.kW} kW</td></tr>
							{/each}
						</tbody>
					</table>
					<p class="muted small">{status.latest.campus_solar.note}</p>
				{/if}
				<p class="more"><a href="/npu">NPU lab →</a> <span class="muted">replays, day-ahead, single runs and the FP32 vs NPU comparison</span></p>
			</article>
		</section>
		{#if models}
			<section class="panel">
				<h2>Forecasting models on the board</h2>
				<p class="sub">
					R² of the +15-minute forecast on the 2025 test split, same windows for every model. SQNR: NPU (W8A16) vs FP32 on one
					validation window; NPU ms: execution time on the Hexagon NPU. Switch the model behind Grid in its card above.
				</p>
				<div class="scroll"><table class="models">
					<thead><tr><th>Model</th><th>Family</th><th>Horizon</th><th class="num">Solar R²</th><th class="num">Grid R²</th><th class="num">SQNR</th><th class="num">NPU ms</th><th>Used for</th></tr></thead>
					<tbody>
						{#each models.models as m (m.id)}
							{@const used = (['grid', 'pl00'] as const).filter((s) => models!.active[s]?.id === m.id)}
							<tr class:active={used.length}>
								<td><strong>{m.label}</strong> <code>{m.id}</code></td>
								<td>{m.family}</td>
								<td>{horizonLabel(m.horizon)}</td>
								<td class="num">{r2(m, 'Solar')}</td>
								<td class="num">{r2(m, 'Grid')}</td>
								<td class="num">{m.sqnr_db != null ? `${m.sqnr_db} dB` : '–'}</td>
								<td class="num">{m.npu_ms ?? '–'}</td>
								<td>{used.map((s) => SIG[s].title).join(', ') || '–'}</td>
							</tr>
						{/each}
					</tbody>
				</table></div>
				<p class="muted small">LightGBM (24 h) runs only on the CPU and isn't switchable yet. 24 h models are checked on their first step, like the 1 h ones.</p>
			</section>
		{/if}
	{:else}
		<p class="muted">Loading…</p>
	{/if}
</main>

<style>
	main { max-width: 1400px; margin: 0 auto; padding: 18px 16px 40px; display: flex; flex-direction: column; gap: 14px; }
	header { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: end; gap: 8px 24px; }
	h1 { font-size: 20px; font-weight: 650; }
	h2 { font-size: 15px; font-weight: 650; }
	.lede { color: var(--ink-2); max-width: 70ch; margin-top: 4px; }
	.src { color: var(--muted); font-size: 12px; max-width: 46ch; }
	.src strong { color: var(--ink-2); font-weight: 600; }
	.panel { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; min-width: 0; }
	.panel.warn { background: var(--warn-bg); color: var(--warn-ink); }
	.tiles { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
	.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px; display: flex; flex-direction: column; gap: 2px; }
	.k { font-size: 12px; color: var(--muted); font-weight: 550; }
	.v { font-size: 20px; font-weight: 650; font-variant-numeric: tabular-nums; display: flex; flex-wrap: wrap; align-items: center; gap: 0 8px; }
	.v small { font-size: 13px; font-weight: 500; color: var(--ink-2); }
	.s { font-size: 12px; color: var(--ink-2); }
	.st { width: 9px; height: 9px; border-radius: 50%; flex: none; }
	.st.ok { background: var(--ok); }
	.st.bad { background: var(--bad); }
	.sigs { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
	.head { display: flex; justify-content: space-between; align-items: start; gap: 10px; }
	.sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
	.pill { padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; white-space: nowrap; }
	.pill.ok { background: color-mix(in srgb, var(--ok) 14%, transparent); color: var(--ok); }
	.pill.bad { background: color-mix(in srgb, var(--bad) 14%, transparent); color: var(--bad); }
	.nums { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 12px 0; }
	.nums dt { font-size: 12px; color: var(--muted); }
	.nums dd { margin: 0; font-size: 18px; font-weight: 650; font-variant-numeric: tabular-nums; }
	.nums small { font-size: 12px; font-weight: 500; color: var(--ink-2); }
	.faults { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 10px; font-size: 12px; }
	.faults button { padding: 4px 10px; border-radius: 6px; border: 1px solid var(--border); background: var(--wash); }
	.faults button:hover { border-color: var(--ink-2); }
	.note { font-size: 13px; color: var(--ink-2); }
	.bottom { display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); gap: 14px; align-items: start; }
	table { margin-top: 8px; font-size: 13px; }
	th { text-align: left; font-weight: 550; color: var(--muted); font-size: 12px; border-bottom: 1px solid var(--border); padding: 4px 6px; }
	td { padding: 5px 6px; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
	.link { border: 0; background: none; color: var(--series-1); padding: 0; font-weight: 550; }
	.muted { color: var(--muted); }
	.model { display: flex; align-items: center; gap: 8px; margin-top: 6px; font-size: 12px; color: var(--muted); }
	.model select { max-width: 100%; padding: 3px 6px; border-radius: 6px; border: 1px solid var(--border); background: var(--surface); font-size: 12px; }
	.scroll { overflow-x: auto; }
	.models code { color: var(--muted); font-size: 11px; }
	.models tr.active td { background: var(--wash); }
	.models td:first-child { white-space: normal; }
	.small { font-size: 12px; margin-top: 8px; }
	.more { margin-top: 12px; font-size: 13px; }
	.more a { color: var(--series-1); font-weight: 600; text-decoration: none; }
	@media (max-width: 1100px) {
		.sigs, .bottom { grid-template-columns: 1fr; }
	}
	@media (max-width: 700px) {
		.tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); }
		.nums { grid-template-columns: repeat(2, minmax(0, 1fr)); }
		.bottom table th:nth-child(4), .bottom table td:nth-child(4) { display: none; }
	}
</style>
