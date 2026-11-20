<script lang="ts">
	import CampusMap from '#lib/components/CampusMap.svelte';
	import NodePanel from '#lib/components/NodePanel.svelte';
	import { colorOf, inkOn, fmtCount, LEVEL_LABEL } from '#lib/format.ts';
	import type { Level } from '#lib/types.ts';
	import { ui } from '#lib/ui.svelte.ts';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();

	const LEVELS: Level[] = ['history', 'few', 'none'];
	let offVerticals = $state<string[]>([]);
	let offLevels = $state<Level[]>([]);
	let search = $state('');

	const byId = $derived(new Map(data.nodes.map((n) => [n.node_id, n])));
	const verticals = $derived(
		data.verticals
			.map((v) => ({ ...v, count: data.nodes.filter((n) => n.vertical === v.code).length }))
			.filter((v) => v.count)
			.sort((a, b) => b.count - a.count)
	);
	const levelCount = $derived(Object.fromEntries(LEVELS.map((l) => [l, data.nodes.filter((n) => n.level === l).length])));

	const filtered = $derived.by(() => {
		const q = search.trim().toLowerCase();
		return data.nodes.filter(
			(n) =>
				!offVerticals.includes(n.vertical) &&
				!offLevels.includes(n.level) &&
				(!q || `${n.node_id} ${n.site_name ?? ''} ${n.vertical_name} ${n.subtype_name ?? ''}`.toLowerCase().includes(q))
		);
	});
	const selected = $derived(ui.selectedId ? (byId.get(ui.selectedId) ?? null) : null);
	const stackNodes = $derived(ui.stack.map((id) => byId.get(id)!).filter(Boolean));

	const toggle = <T,>(list: T[], v: T) => (list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);

	function pick(ids: string[]) {
		if (ids.length === 1) {
			ui.selectedId = ids[0];
			ui.stack = [];
		} else {
			ui.stack = ids;
			ui.selectedId = null;
		}
	}
	function select(id: string) {
		ui.selectedId = id;
	}
	function close() {
		ui.selectedId = null;
	}
	const panelOpen = $derived(!!selected || ui.stack.length > 0);
</script>

<div class="app" class:open={panelOpen}>
	<aside class="left">
		<header>
			<p>{fmtCount(data.nodes.length)} nodes in <code>campus.sqlite</code>. Showing {fmtCount(filtered.length)}.</p>
		</header>

		<input type="search" placeholder="Search by name, building or type" bind:value={search} aria-label="Search nodes" />

		<section>
			<h2>Type</h2>
			<div class="chips">
				{#each verticals as v (v.code)}
					{@const on = !offVerticals.includes(v.code)}
					<button class="chip" class:off={!on} aria-pressed={on} title={v.description ?? v.name} onclick={() => (offVerticals = toggle(offVerticals, v.code))}>
						<span class="sw" style:background={colorOf(v.code)} style:color={inkOn(v.code)}>{v.code}</span>
						<span class="name">{v.name}</span>
						<span class="n">{v.count}</span>
					</button>
				{/each}
			</div>
		</section>

		<section>
			<h2>How much real data</h2>
			<div class="chips">
				{#each LEVELS as l (l)}
					{@const on = !offLevels.includes(l)}
					<button class="chip" class:off={!on} aria-pressed={on} onclick={() => (offLevels = toggle(offLevels, l))}>
						<span class="sw level {l}"></span>
						<span class="name">{LEVEL_LABEL[l]}</span>
						<span class="n">{levelCount[l]}</span>
					</button>
				{/each}
			</div>
			<p class="hint">Filled marker: 1,000 or more real messages. Outlined: fewer. Dashed: none.</p>
		</section>

		<section class="list">
			<h2>Nodes</h2>
			<ul>
				{#each filtered as n (n.node_id)}
					<li>
						<button class:on={n.node_id === ui.selectedId} onclick={() => select(n.node_id)}>
							<span class="sw level {n.level}" style:--c={colorOf(n.vertical)}></span>
							<span class="id">{n.node_id}</span>
							<span class="site">{n.site_name ?? n.site_code ?? ''}</span>
						</button>
					</li>
				{:else}
					<li class="hint">No nodes match.</li>
				{/each}
			</ul>
		</section>
	</aside>

	<main>
		<CampusMap nodes={filtered} selectedId={ui.selectedId} onpick={pick} />
	</main>

	{#if panelOpen}
		<aside class="right">
			{#if stackNodes.length}
				<div class="stack">
					<div class="stack-head">
						<h2>{stackNodes.length} nodes share this position</h2>
						{#if !selected}<button class="close" onclick={() => (ui.stack = [])} aria-label="Close">✕</button>{/if}
					</div>
					<p class="hint">Their recorded position is the same point, usually the building. Pick one.</p>
					<ul class:compact={!!selected}>
						{#each stackNodes as n (n.node_id)}
							<li>
								<button class:on={n.node_id === ui.selectedId} onclick={() => select(n.node_id)}>
									<span class="sw level {n.level}" style:--c={colorOf(n.vertical)}></span>
									<span class="id">{n.node_id}</span>
									<span class="site">{n.subtype_name ?? n.vertical_name}{n.floor ? `, ${n.floor}` : ''}</span>
								</button>
							</li>
						{/each}
					</ul>
				</div>
			{/if}
			{#if selected}
				<NodePanel summary={selected} onclose={close} />
			{/if}
		</aside>
	{/if}
</div>

<style>
	.app { display: grid; grid-template-columns: 300px minmax(0, 1fr); height: 100%; }
	.app.open { grid-template-columns: 300px minmax(0, 1fr) 440px; }
	aside { background: var(--surface); overflow-y: auto; min-height: 0; }
	.left { border-right: 1px solid var(--border); display: flex; flex-direction: column; gap: 14px; padding: 16px; }
	.right { border-left: 1px solid var(--border); }
	main { min-height: 0; min-width: 0; }
	header p { color: var(--ink-2); font-size: 12.5px; margin-top: 2px; }
	h2 { font-size: 12px; font-weight: 600; color: var(--ink-2); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px; }
	.hint { font-size: 12px; color: var(--ink-2); margin-top: 6px; }
	input[type='search'] { width: 100%; padding: 7px 9px; border: 1px solid var(--border); border-radius: 8px; background: var(--plane); }
	.chips { display: grid; gap: 2px; }
	.chip { display: grid; grid-template-columns: 30px 1fr auto; align-items: center; gap: 8px; padding: 4px 6px; border: 0; background: none; border-radius: 6px; text-align: left; }
	.chip:hover { background: var(--wash); }
	.chip.off { opacity: 0.4; }
	.chip.off .name { text-decoration: line-through; }
	.sw { display: grid; place-items: center; height: 18px; border-radius: 9px; font-size: 10px; font-weight: 700; }
	.n { color: var(--ink-2); font-size: 12px; font-variant-numeric: tabular-nums; }
	.sw.level { --c: var(--ink-2); width: 14px; height: 14px; border-radius: 50%; justify-self: center; background: var(--c); }
	.sw.level.few { background: var(--surface); border: 3px solid var(--c); }
	.sw.level.none { background: var(--surface); border: 2px dashed var(--muted); }
	.list { flex: 1; min-height: 160px; display: flex; flex-direction: column; }
	ul { list-style: none; margin: 0; padding: 0; }
	.list ul { overflow-y: auto; flex: 1; }
	li button { width: 100%; display: grid; grid-template-columns: 18px auto 1fr; align-items: center; gap: 8px; padding: 4px 6px; border: 0; background: none; border-radius: 6px; text-align: left; }
	li button:hover { background: var(--wash); }
	li button.on { background: var(--ink); color: var(--surface); }
	li button.on .site { color: inherit; opacity: 0.75; }
	.id { font-family: ui-monospace, monospace; font-size: 12.5px; white-space: nowrap; }
	.site { color: var(--ink-2); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; text-align: right; }
	.stack { padding: 16px 16px 8px; border-bottom: 1px solid var(--grid); }
	.stack-head { display: flex; align-items: start; gap: 8px; }
	.stack-head h2 { flex: 1; }
	.stack ul { margin-top: 8px; }
	.stack ul.compact { max-height: 132px; overflow-y: auto; }
	.close { border: 0; background: none; font-size: 16px; padding: 2px 8px; border-radius: 6px; color: var(--ink-2); }
	.close:hover { background: var(--wash); }
	code { font-family: ui-monospace, monospace; font-size: 12px; }

	@media (max-width: 900px) {
		.app, .app.open { grid-template-columns: 1fr; grid-template-rows: auto 60vh auto; height: auto; }
		.left { max-height: 45vh; border-right: 0; border-bottom: 1px solid var(--border); }
		.right { border-left: 0; border-top: 1px solid var(--border); }
	}
</style>
