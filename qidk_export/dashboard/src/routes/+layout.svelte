<script lang="ts">
	import '../app.css';
	import { page } from '$app/state';
	import ChatPanel from '#lib/components/ChatPanel.svelte';
	import LiveBar from '#lib/components/LiveBar.svelte';
	import { ui } from '#lib/ui.svelte.ts';
	import type { LayoutProps } from './$types';

	let { children }: LayoutProps = $props();
	const TABS = [
		{ href: '/', label: 'Live' },
		{ href: '/map', label: 'Map' },
		{ href: '/whatif', label: 'Net zero' },
		{ href: '/results', label: 'Results' },
		{ href: '/npu', label: 'NPU lab' }
	];
</script>

<svelte:head>
	<title>Campus sensors</title>
</svelte:head>

<div class="shell">
	<nav>
		<strong>Campus sensors</strong>
		<div class="tabs">
			{#each TABS as t (t.href)}
				<a href={t.href} aria-current={page.url.pathname === t.href ? 'page' : undefined}>{t.label}</a>
			{/each}
		</div>
		<LiveBar />
		<button class="ask" aria-pressed={ui.chatOpen} onclick={() => (ui.chatOpen = !ui.chatOpen)}>
			<span class="long">{ui.chatOpen ? 'Close assistant' : 'Ask the assistant'}</span><span class="short">{ui.chatOpen ? 'Close' : 'Ask'}</span>
		</button>
	</nav>
	<div class="page">{@render children()}</div>
	{#if ui.chatOpen}<ChatPanel />{/if}
</div>

<style>
	.shell { height: 100vh; display: flex; flex-direction: column; }
	nav {
		flex: none; height: 46px; display: flex; align-items: center; gap: 20px; padding: 0 16px;
		background: var(--surface); border-bottom: 1px solid var(--border);
	}
	strong { font-size: 15px; font-weight: 650; }
	.tabs { display: flex; gap: 4px; flex: 1; min-width: 0; overflow-x: auto; scrollbar-width: none; }
	.tabs a { white-space: nowrap; }
	a { padding: 6px 12px; border-radius: 6px; color: var(--ink-2); text-decoration: none; font-weight: 550; }
	a:hover { background: var(--wash); }
	a[aria-current='page'] { background: var(--ink); color: var(--surface); }
	.ask { padding: 6px 12px; border-radius: 6px; border: 1px solid var(--ink); background: var(--surface); font-weight: 550; }
	.ask[aria-pressed='true'] { background: var(--ink); color: var(--surface); }
	.ask { white-space: nowrap; flex: none; }
	.short { display: none; }
	@media (max-width: 600px) { .long { display: none; } .short { display: inline; } }
	.page { flex: 1; min-height: 0; overflow: auto; }
	@media (max-width: 900px) {
		.shell { height: auto; min-height: 100vh; }
		.page { overflow: visible; }
		nav { gap: 10px; padding: 0 10px; }
		strong { display: none; }
	}
</style>
