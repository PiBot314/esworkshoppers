<script lang="ts">
	// Live chip + anomaly banner + demo fault buttons (plan_reduced_shortterm.md §4.4).
	// Polls /api/live/status and /api/live/events every 30 s. No SLM call per flag: "Ask about it" hands
	// the question to the assistant, which then uses the live tools.
	import { onMount } from 'svelte';
	import { chat, ui } from '#lib/ui.svelte.ts';

	interface Ev { id: number; time: string; signal: 'grid' | 'pl00'; label: string; actual_kW: number; forecast_kW: number; flagged: boolean; injected: string | null; late?: boolean }
	interface Status {
		enabled: boolean; api: { ok: boolean; error: string | null }; npu: { ok: boolean; variant: string | null; last_ms: number | null; error: string | null };
		now: string | null; last_reading_checked: string | null; next_check: string | null;
		minutes_since_last_reading: Record<string, number | null>; last_24h: { natural_flags: number; injected_faults: number };
		pending_injection: Record<string, { kind: string; left: number }>;
	}

	let status = $state<Status | null>(null);
	let banner = $state<Ev | null>(null);
	let open = $state(false);
	let note = $state('');
	let since = 0;
	let first = true;

	const age = $derived(status?.minutes_since_last_reading['EM-CR-SB00-00'] ?? null);
	const level = $derived(
		!status?.enabled ? 'off' : status.api.ok && status.npu.ok && age != null && age < 20 ? 'ok' : 'warn'
	);
	const short = (e: Ev) => (e.signal === 'grid' ? 'Grid' : 'Palash solar');

	async function poll() {
		try {
			status = await (await fetch('/api/live/status')).json();
			const r = await (await fetch(`/api/live/events?since=${since}`)).json();
			const evs = r.events as Ev[];
			if (evs.length) {
				since = evs[evs.length - 1].id;
				// on the first poll only remember where we are; don't pop up old flags
				if (!first) banner = evs.findLast((e) => (e.flagged || e.injected) && !e.late) ?? banner;
			}
			first = false;
		} catch {
			status = null;
		}
	}
	onMount(() => {
		poll();
		const t = setInterval(poll, 30000);
		return () => clearInterval(t);
	});

	function askAbout(e: Ev) {
		chat.pending = `What happened with ${short(e).toLowerCase()} at ${e.time.slice(11, 16)}? (live anomaly #${e.id})`;
		ui.chatOpen = true;
		banner = null;
	}

	async function inject(signal: 'grid' | 'pl00', kind: string) {
		const r = await fetch('/api/live/inject', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ signal, kind, steps: 1 }) });
		const j = await r.json().catch(() => ({}));
		note = r.ok ? `Injected ${kind} on ${signal === 'grid' ? 'Grid' : 'Palash solar'}: applies to the ${j.applies_from} reading.` : (j.message ?? 'Failed');
		poll();
	}
</script>

<div class="live">
	<button class="chip {level}" aria-expanded={open} onclick={() => (open = !open)} title="Live data status">
		<span class="dot"></span>
		{level === 'off' ? 'Live off' : level === 'ok' ? 'Live' : 'Live: no data'}
	</button>

	{#if open}
		<div class="pop" role="dialog" aria-label="Live data">
			{#if !status}
				<p>Can't reach the dashboard server.</p>
			{:else if !status.enabled}
				<p>The live loop is off on this server (start it with <code>LIVE=1</code>).</p>
			{:else}
				<dl>
					<dt>API time</dt><dd>{status.now ?? "not synced yet"}</dd>
					<dt>Last reading checked</dt><dd>{status.last_reading_checked ?? '–'}</dd>
					<dt>Next check</dt><dd>{status.next_check ?? '–'}</dd>
					<dt>Data feed</dt><dd>{status.api.ok ? `OK, substation data ${age ?? '?'} min old` : status.api.error ?? 'not yet'}</dd>
					<dt>Forecasts</dt><dd>{status.npu.ok ? `${status.npu.variant === 'aimet_w8a16' ? 'NPU' : 'CPU (NPU failed)'}${status.npu.last_ms != null ? `, ${status.npu.last_ms.toFixed(1)} ms` : ''}` : status.npu.error ?? 'not yet'}</dd>
					<dt>Last 24 h</dt><dd>{status.last_24h.natural_flags} flags, {status.last_24h.injected_faults} demo faults</dd>
				</dl>
				<p class="hint">Flags on this synthetic data are mostly false alarms. Demo faults change the next reading:</p>
				<div class="btns">
					<button onclick={() => inject('pl00', 'dropout')}>Solar dropout</button>
					<button onclick={() => inject('pl00', 'damage40')}>Solar −40 %</button>
					<button onclick={() => inject('grid', 'spike')}>Grid spike</button>
				</div>
				{#if note}<p class="hint">{note}</p>{/if}
			{/if}
		</div>
	{/if}

	{#if banner}
		<div class="banner" role="status">
			<span>
				<b>{banner.time.slice(11, 16)}</b> · {short(banner)}
				{Math.abs(Math.round(banner.actual_kW - banner.forecast_kW))} kW {banner.actual_kW < banner.forecast_kW ? 'below' : 'above'} forecast{banner.injected ? ' (demo fault)' : ''}
			</span>
			<button class="link" onclick={() => askAbout(banner!)}>Ask about it</button>
			<button class="x" aria-label="Dismiss" onclick={() => (banner = null)}>×</button>
		</div>
	{/if}
</div>

<style>
	.live { position: relative; display: flex; align-items: center; }
	.chip {
		display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 999px;
		border: 1px solid var(--border); background: var(--surface); color: var(--ink-2); font-weight: 550; font-size: 13px;
	}
	.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
	.chip.ok .dot { background: var(--ok); }
	.chip.warn { color: var(--warn-ink); background: var(--warn-bg); }
	.chip.warn .dot { background: var(--warn-ink); }
	.pop {
		position: absolute; top: 34px; right: 0; z-index: 1000; width: min(340px, calc(100vw - 32px)); padding: 12px 14px;
		background: var(--surface); border: 1px solid var(--border); border-radius: 8px; box-shadow: 0 6px 24px rgba(0, 0, 0, 0.15);
	}
	dl { display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; margin: 0 0 8px; }
	dt { color: var(--muted); }
	dd { margin: 0; overflow-wrap: anywhere; }
	p { margin: 0 0 6px; }
	.hint { color: var(--ink-2); font-size: 13px; }
	.btns { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 6px; }
	.btns button { padding: 4px 10px; border-radius: 6px; border: 1px solid var(--border); background: var(--wash); color: var(--ink); }
	.banner {
		position: fixed; top: 54px; left: 50%; transform: translateX(-50%); z-index: 1000; max-width: calc(100vw - 32px);
		display: flex; align-items: center; gap: 12px; padding: 8px 12px; border-radius: 8px;
		background: var(--warn-bg); color: var(--warn-ink); border: 1px solid var(--border); box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
	}
	.link { border: 0; background: none; color: inherit; font-weight: 650; text-decoration: underline; padding: 0; }
	.x { border: 0; background: none; color: inherit; font-size: 18px; line-height: 1; padding: 0 2px; }
</style>
