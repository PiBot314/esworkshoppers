<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { onMount, tick } from 'svelte';
	import { chat, ui } from '#lib/ui.svelte.ts';

	const turns = chat.turns;
	let draft = $state('');
	let busy = $state(false);
	let status = $state<{ ok: boolean; model: string | null } | null>(null);
	let scroller: HTMLDivElement;
	let ctl: AbortController | null = null;

	const EXAMPLES = [
		'What data do we actually have?',
		'When does the campus use the most power?',
		'Is the campus net zero?',
		'Is anything unusual right now?'
	];
	const STEP_LABEL: Record<string, (a: Record<string, unknown>) => string> = {
		data_overview: () => 'Looked at what data exists',
		search_nodes: (a) => `Searched nodes${a.query ? ` for "${a.query}"` : ''}${a.type ? ` of type ${a.type}` : ''}`,
		get_node: (a) => `Read node ${a.node_id}`,
		get_stats: (a) => `Computed ${a.value} for ${a.node_id}${a.start || a.end ? ` (${a.start ?? '…'} to ${a.end ?? '…'})` : ''}${a.group_by ? ` by ${a.group_by}` : ''}`,
		compare_nodes: (a) => `Compared ${a.value} across ${a.type ?? ''} nodes${a.query ? ` matching "${a.query}"` : ''}`,
		net_zero: (a) => `Worked out the carbon balance${a.extra_solar_pct ? ` with +${a.extra_solar_pct}% solar` : ''}`,
		show_node: (a) => `Opened ${a.node_id} on the map`,
		live_status: () => 'Checked the live data and the NPU forecasts',
		recent_anomalies: (a) => `Looked up live anomaly flags (last ${a.hours ?? 24} h${a.signal ? `, ${a.signal}` : ''})`,
		explain_anomaly: (a) => `Read the details of live anomaly ${a.id != null ? '#' + a.id : '(newest)'}`
	};

	async function checkStatus() {
		try { status = await (await fetch('/api/chat')).json(); } catch { status = { ok: false, model: null }; }
	}
	onMount(checkStatus);

	// A question handed over by the live banner ("Ask about it").
	$effect(() => {
		const q = chat.pending;
		if (q && !busy) { chat.pending = null; ask(q); }
	});

	async function scrollDown() {
		await tick();
		scroller?.scrollTo({ top: scroller.scrollHeight });
	}

	async function ask(text: string) {
		text = text.trim();
		if (!text || busy) return;
		draft = '';
		busy = true;
		turns.push({ role: 'user', content: text });
		const history = turns.map((t) => ({ role: t.role, content: t.content }));
		turns.push({ role: 'assistant', content: '', steps: [] });
		const reply = turns[turns.length - 1];
		scrollDown();
		ctl = new AbortController();
		try {
			const res = await fetch('/api/chat', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({ messages: history, context: { page: page.url.pathname, node: page.url.pathname === '/map' ? ui.selectedId : null } }),
				signal: ctl.signal
			});
			const reader = res.body!.getReader();
			const dec = new TextDecoder();
			let buf = '';
			for (;;) {
				const { done, value } = await reader.read();
				if (done) break;
				buf += dec.decode(value, { stream: true });
				let nl;
				while ((nl = buf.indexOf('\n')) >= 0) {
					const line = buf.slice(0, nl);
					buf = buf.slice(nl + 1);
					if (!line) continue;
					const ev = JSON.parse(line);
					if (ev.type === 'text') reply.content += ev.text;
					else if (ev.type === 'error') reply.error = ev.message;
					else if (ev.type === 'tool') {
						reply.steps!.push({ name: ev.name, args: ev.args ?? {}, href: ev.ui?.action === 'what_if' ? ev.ui.href : undefined });
						if (ev.ui?.action === 'show_node') {
							ui.stack = [];
							ui.selectedId = ev.ui.node_id;
							if (page.url.pathname !== '/map') goto('/map');
						}
					}
					scrollDown();
				}
			}
		} catch (e) {
			if (!(e instanceof DOMException && e.name === 'AbortError')) reply.error = 'Could not reach the dashboard server.';
		}
		reply.content = reply.content.trim();
		if (!reply.content && !reply.error && !reply.steps!.length) reply.error = 'The model gave no answer. Try rephrasing.';
		busy = false;
		ctl = null;
		if (reply.error) checkStatus();
		scrollDown();
	}
</script>

<section class="chat" aria-label="Assistant">
	<header>
		<div>
			<h2>Assistant</h2>
			<p>
				{#if status?.ok}
					Small language model on this device: {status.model}
				{:else if status}
					The language model is not running (it starts with the board, or <code>bun run slm</code> on a laptop)
				{:else}
					Checking the model…
				{/if}
			</p>
		</div>
		<button class="x" onclick={() => (ui.chatOpen = false)} aria-label="Close assistant">✕</button>
	</header>

	<div class="log" bind:this={scroller} aria-live="polite">
		{#if !turns.length}
			<p class="intro">
				Ask about the campus sensors. It looks things up in <code>campus.sqlite</code> and shows each lookup it made.
				It is a small model, so check anything important against the lookup it shows.
			</p>
			<div class="examples">
				{#each EXAMPLES as e (e)}<button onclick={() => ask(e)} disabled={busy}>{e}</button>{/each}
			</div>
		{/if}
		{#each turns as t, i (i)}
			<div class="turn {t.role}">
				{#each t.steps ?? [] as s, j (j)}
					<p class="step">
						<span aria-hidden="true">↳</span> {(STEP_LABEL[s.name] ?? (() => s.name))(s.args)}
						{#if s.href}<a href={s.href}>Open in Net zero</a>{/if}
					</p>
				{/each}
				{#if t.content}<p class="bubble">{t.content}</p>{/if}
				{#if t.role === 'assistant' && busy && i === turns.length - 1 && !t.content}<p class="step">Thinking…</p>{/if}
				{#if t.error}<p class="err">{t.error}</p>{/if}
			</div>
		{/each}
	</div>

	<form onsubmit={(e) => { e.preventDefault(); ask(draft); }}>
		<input bind:value={draft} placeholder="Ask about a node, a period, or net zero" aria-label="Your question" />
		{#if busy}
			<button type="button" onclick={() => ctl?.abort()}>Stop</button>
		{:else}
			<button type="submit" disabled={!draft.trim()}>Send</button>
		{/if}
	</form>
</section>

<style>
	.chat {
		position: fixed; right: 16px; bottom: 16px; z-index: 2000; width: min(400px, calc(100vw - 32px)); height: min(600px, calc(100vh - 78px));
		display: flex; flex-direction: column; background: var(--surface); border: 1px solid var(--border);
		border-radius: 12px; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.22); overflow: hidden;
	}
	header { display: flex; align-items: start; gap: 8px; padding: 12px 14px; border-bottom: 1px solid var(--grid); }
	header div { flex: 1; min-width: 0; }
	h2 { font-size: 15px; font-weight: 650; }
	header p { font-size: 12px; color: var(--ink-2); overflow-wrap: anywhere; }
	.x { border: 0; background: none; font-size: 15px; padding: 2px 8px; border-radius: 6px; color: var(--ink-2); }
	.x:hover { background: var(--wash); }
	.log { flex: 1; overflow-y: auto; padding: 14px; display: flex; flex-direction: column; gap: 12px; }
	.intro { font-size: 13px; color: var(--ink-2); }
	.examples { display: grid; gap: 6px; }
	.examples button { text-align: left; padding: 8px 10px; border: 1px solid var(--border); border-radius: 8px; background: var(--plane); }
	.examples button:hover { background: var(--wash); }
	.turn { display: grid; gap: 4px; }
	.turn.user { justify-items: end; }
	.bubble { white-space: pre-wrap; overflow-wrap: anywhere; padding: 8px 11px; border-radius: 12px; max-width: 92%; background: var(--wash); }
	.user .bubble { background: var(--ink); color: var(--surface); }
	.step { font-size: 12px; color: var(--ink-2); }
	.step a { margin-left: 6px; color: var(--ink); font-weight: 600; }
	.err { font-size: 12.5px; color: var(--warn-ink); background: var(--warn-bg); padding: 6px 9px; border-radius: 8px; }
	form { display: flex; gap: 8px; padding: 10px; border-top: 1px solid var(--grid); }
	input { flex: 1; min-width: 0; padding: 8px 10px; border: 1px solid var(--border); border-radius: 8px; background: var(--plane); }
	form button { padding: 8px 14px; border-radius: 8px; border: 0; background: var(--ink); color: var(--surface); font-weight: 600; }
	form button:disabled { opacity: 0.4; cursor: default; }
	code { font-family: ui-monospace, monospace; font-size: 11.5px; }
</style>
