<script lang="ts">
	import 'leaflet/dist/leaflet.css';
	import { onMount } from 'svelte';
	import type * as Leaflet from 'leaflet';
	import type { NodeSummary } from '#lib/types.ts';
	import { colorOf, inkOn, OTHER_COLOR, LEVEL_LABEL } from '#lib/format.ts';

	interface Props {
		nodes: NodeSummary[];
		selectedId: string | null;
		/** called with every node at the clicked spot (one, or a stack) */
		onpick: (ids: string[]) => void;
	}
	let { nodes, selectedId, onpick }: Props = $props();

	let el: HTMLDivElement;
	let L: typeof Leaflet | undefined = $state();
	let map: Leaflet.Map | undefined;
	let layer: Leaflet.LayerGroup | undefined;

	// Nodes that share one coordinate (e.g. a building-level position) become one stacked marker.
	const keyOf = (n: NodeSummary) => `${n.latitude!.toFixed(5)},${n.longitude!.toFixed(5)}`;
	const groups = $derived.by(() => {
		const m = new Map<string, NodeSummary[]>();
		for (const n of nodes) {
			if (n.latitude == null || n.longitude == null) continue;
			const k = keyOf(n);
			m.get(k)?.push(n) ?? m.set(k, [n]);
		}
		return [...m.values()];
	});

	function icon(group: NodeSummary[], selected: boolean) {
		const first = group[0];
		const div = document.createElement('div');
		if (group.length === 1) {
			div.className = `pin ${first.level}`;
			div.textContent = first.vertical;
			div.style.setProperty('--c', colorOf(first.vertical));
			div.style.setProperty('--ink', inkOn(first.vertical));
		} else {
			const uniform = group.every((n) => n.vertical === first.vertical);
			// A stack looks as strong as its best-covered member.
			const best = group.some((n) => n.level === 'history') ? 'history' : group.some((n) => n.level === 'few') ? 'few' : 'none';
			div.className = `pin stack ${best}`;
			div.textContent = uniform ? `${first.vertical} ×${group.length}` : `×${group.length}`;
			div.style.setProperty('--c', uniform ? colorOf(first.vertical) : OTHER_COLOR);
			div.style.setProperty('--ink', uniform ? inkOn(first.vertical) : '#ffffff');
		}
		if (selected) div.classList.add('selected');
		return L!.divIcon({ html: div, className: 'pin-wrap', iconSize: [0, 0] });
	}

	onMount(() => {
		let dead = false;
		import('leaflet').then((mod) => {
			if (dead) return;
			const leaflet = (mod as unknown as { default?: typeof Leaflet }).default ?? mod;
			map = leaflet.map(el, { zoomControl: true });
			leaflet
				.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
					maxZoom: 19,
					attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
				})
				.addTo(map);
			layer = leaflet.layerGroup().addTo(map);
			// Frame the campus; ignore positions known to be wrong.
			const pts = nodes
				.filter((n) => n.latitude != null && n.location_quality !== 'suspect')
				.map((n) => [n.latitude!, n.longitude!] as [number, number]);
			if (pts.length) map.fitBounds(leaflet.latLngBounds(pts).pad(0.02));
			else map.setView([17.4455, 78.3489], 17);
			L = leaflet;
		});
		return () => {
			dead = true;
			map?.remove();
		};
	});

	// Redraw markers whenever the filtered set or the selection changes.
	$effect(() => {
		if (!L || !layer) return;
		layer.clearLayers();
		for (const g of groups) {
			const selected = g.some((n) => n.node_id === selectedId);
			const title =
				g.length === 1
					? `${g[0].node_id} · ${LEVEL_LABEL[g[0].level]}`
					: `${g.length} nodes share this position`;
			L.marker([g[0].latitude!, g[0].longitude!], {
				icon: icon(g, selected),
				title,
				keyboard: true,
				zIndexOffset: selected ? 1000 : g.length > 1 ? 500 : 0
			})
				.on('click', () => onpick(g.map((n) => n.node_id)))
				.addTo(layer);
		}
	});

	// Bring a node picked from the list into view.
	$effect(() => {
		if (!L || !map || !selectedId) return;
		const n = nodes.find((x) => x.node_id === selectedId);
		if (!n || n.latitude == null || n.longitude == null) return;
		const at = L.latLng(n.latitude, n.longitude);
		if (!map.getBounds().pad(-0.1).contains(at)) map.panTo(at);
	});
</script>

<div class="map" bind:this={el} aria-label="Map of campus sensor nodes"></div>

<style>
	.map { height: 100%; width: 100%; background: var(--plane); }
	.map :global(.pin-wrap) { overflow: visible; }
	.map :global(.pin) {
		position: absolute;
		transform: translate(-50%, -50%);
		min-width: 24px;
		height: 24px;
		padding: 0 4px;
		border-radius: 12px;
		display: grid;
		place-items: center;
		font: 600 10px/1 system-ui, sans-serif;
		white-space: nowrap;
		cursor: pointer;
		/* light values on purpose: the basemap is always light */
		background: var(--c);
		color: var(--ink);
		border: 2px solid #ffffff;
		box-shadow: 0 0 0 1px rgba(11, 11, 11, 0.25);
	}
	.map :global(.pin.few) {
		background: #ffffff;
		color: #0b0b0b;
		border: 3px solid var(--c);
	}
	.map :global(.pin.none) {
		background: #ffffff;
		color: #52514e;
		border: 2px dashed #898781;
		box-shadow: none;
	}
	.map :global(.pin.stack) { height: 26px; padding: 0 7px; border-radius: 13px; font-size: 11px; }
	.map :global(.pin.selected) { box-shadow: 0 0 0 3px #0b0b0b, 0 0 0 5px #ffffff; }
	.map :global(.pin:hover) { filter: brightness(1.08); }
</style>
