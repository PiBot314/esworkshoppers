<script lang="ts">
	import { afterNavigate, replaceState } from '$app/navigation';
	import { page } from '$app/state';
	import CompareChart from '#lib/components/CompareChart.svelte';
	import { fmtCount, fmtDate, fmtNum } from '#lib/format.ts';
	import type { WhatIfResult } from '#lib/types.ts';

	let solar = $state(0);
	let exportSurplus = $state(false);
	let month = $state('');
	let ready = $state(false);
	let result = $state<WhatIfResult | null>(null);
	let loading = $state(false);
	let failed = $state('');

	// The address bar holds the scenario, so a link (or the assistant) can open a specific one.
	afterNavigate(() => {
		const q = page.url.searchParams;
		if (q.has('solar')) solar = Math.min(2000, Math.max(0, Number(q.get('solar')) || 0));
		exportSurplus = q.get('export') === '1';
		if (q.has('month')) month = q.get('month') ?? '';
		ready = true;
	});

	$effect(() => {
		if (!ready) return;
		const q = new URLSearchParams({ solar: String(solar) });
		if (month) q.set('month', month);
		if (exportSurplus) q.set('export', '1');
		const ctl = new AbortController();
		const timer = setTimeout(() => {
			loading = true;
			replaceState(`?${q}`, {});
			fetch(`/api/whatif?${q}`, { signal: ctl.signal })
				.then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
				.then((d: WhatIfResult) => { result = d; failed = ''; loading = false; })
				.catch((e) => { if (e.name !== 'AbortError') { failed = String(e.message ?? e); loading = false; } });
		}, 120);
		return () => { clearTimeout(timer); ctl.abort(); };
	});

	const t = (kg: number) => fmtNum(kg / 1000);
	const mwh = (kwh: number) => fmtNum(kwh / 1000);
	const signed = (kg: number) => `${kg > 0 ? '+' : kg < 0 ? '−' : ''}${fmtNum(Math.abs(kg) / 1000)}`;
	const monthName = (m: string) => new Date(m + '-01T00:00:00Z').toLocaleString('en-GB', { month: 'short', timeZone: 'UTC' });
	const hhmm = (ts: number) => new Date((ts + 19800) * 1000).toISOString().slice(11, 16);
	const pct = (x: number) => fmtCount(Math.round(x / 25) * 25);
	function setToZero() {
		if (!result?.to_reach_zero.extra_pct) return;
		exportSurplus = true;
		solar = Math.min(2000, Math.ceil(result.to_reach_zero.extra_pct / 25) * 25);
	}

	const nb = $derived(result?.netzero.before);
	const na = $derived(result?.netzero.after);
	const monthRows = $derived((result?.monthly ?? []).map((m) => ({
		key: m.month, label: monthName(m.month), before: m.co2_before_kg / 1000, after: m.co2_after_kg / 1000,
		extra: `${m.days} days of data`
	})));
	const dayRows = $derived((result?.day.points ?? []).map((p) => ({ key: p.t, label: hhmm(p.t), before: p.before, after: p.after })));
</script>

<svelte:head><title>Net zero · Campus sensors</title></svelte:head>

<div class="wrap">
	<header>
		<h1>Is the campus net zero?</h1>
		<p>
			Net zero means that over a year the campus causes no more CO₂ than it removes.
			This page adds up what the 2025 meter data shows, and lets you ask what more solar would change.
		</p>
	</header>

	{#if failed}
		<p class="warn">Could not work out the balance ({failed}).</p>
	{:else if !result || !nb || !na}
		<p class="hint">Loading the 2025 data…</p>
	{:else}
		<div class="tiles" class:stale={loading}>
			<div class="tile big">
				<p class="tl">Net emissions{solar ? ` with +${solar}% solar${exportSurplus ? ', net metering' : ''}` : ', 2025'}</p>
				<p class="tv">{signed(na.net_kg)} t CO₂</p>
				<p class="td">
					{na.net_kg <= 0 ? 'Net zero reached on the central estimate.' : 'Not net zero.'}
					<span>Range {signed(na.net_low_kg)} to {signed(na.net_high_kg)} t. Trees offset about {fmtNum((na.trees_kg / na.total_kg) * 100)}% of what is caused.</span>
				</p>
			</div>
			<div class="tile">
				<p class="tl">Caused: grid electricity + diesel{na.exported_kwh > 0 ? ', less export credit' : ''}</p>
				<p class="tv">{t(na.total_kg)} t</p>
				<p class="td"><span>{mwh(na.grid_kwh)} MWh bought from the grid{na.exported_kwh > 0 ? `, ${mwh(na.exported_kwh)} MWh exported` : ''}</span></p>
			</div>
			<div class="tile">
				<p class="tl">Removed: campus trees</p>
				<p class="tv">{t(na.trees_kg)} t</p>
				<p class="td"><span>{fmtCount(result.trees.count)} trees; estimate, {t(result.trees.low)} to {t(result.trees.high)} t</span></p>
			</div>
			<div class="tile">
				<p class="tl">Solar generated on campus</p>
				<p class="tv">{mwh(na.solar_kwh)} MWh</p>
				<p class="td"><span>avoids {t(na.solar_avoided_kg)} t of CO₂</span></p>
			</div>
		</div>

		<div class="filters">
			<label>
				<span>What if the campus had more solar? <strong>+{fmtCount(solar)}%</strong> more than today's measured solar{solar ? ` (${fmtNum(1 + solar / 100)}× today's in total)` : ''}</span>
				<input type="range" min="0" max="2000" step="25" bind:value={solar} />
			</label>
			<label class="check">
				<input type="checkbox" bind:checked={exportSurplus} />
				<span><strong>Net metering:</strong> solar beyond what the campus uses at that moment is exported to the grid and credited. Off = it's wasted (no export, no storage).</span>
			</label>
			{#if result.to_reach_zero.extra_pct}
				{@const z = result.to_reach_zero}
				<p class="hint">
					<strong>To reach zero</strong> (central estimate), the campus needs about <strong>+{pct(z.extra_pct!)}%</strong> more solar
					({fmtNum(1 + z.times_todays_solar!)}× today's in total, {mwh(z.extra_solar_kwh)} MWh a year more), <strong>and only with net metering or storage</strong>.
					<button class="link" onclick={setToZero}>Show this</button>
				</p>
				{#if !z.possible_without_export}
					<p class="hint">
						<strong>Without export or storage, solar alone can't get there:</strong> the load while the panels produce almost nothing (nights; under 1% of their peak)
						stays on the grid, {mwh(z.floor_grid_kwh)} MWh a year, so net emissions can't fall below about {signed(z.min_net_without_export_kg)} t however many panels are added.
					</p>
				{/if}
				<p class="hint">
					“Today's measured solar” counts only {result.solar.nodes.length} solar meters with a clean 2025 ({fmtCount(Math.round(result.solar.measured_kwh / result.period.days))} kWh a day).
					The campus's other meters are partial or missing that year, so its real solar is higher and the multiple above is an overestimate.
				</p>
			{/if}
		</div>

		<div class:stale={loading}>
			<section>
				<h2>The balance</h2>
				<p class="hint">Emissions = activity × emission factor, summed over sources. Net = emissions − what the trees absorb.</p>
				<div class="scrollx">
					<table>
						<thead><tr><th>Source</th><th class="num">Activity in 2025</th><th class="num">Factor</th><th class="num">Today, t CO₂</th><th class="num">With +{solar}% solar</th></tr></thead>
						<tbody>
							<tr><td>Grid electricity bought</td><td class="num">{mwh(nb.grid_kwh)} MWh</td><td class="num">0.7 kg/kWh</td><td class="num">{t(nb.grid_kg)}</td><td class="num">{t(na.grid_kwh * 0.7)}</td></tr>
							{#if na.exported_kwh > 0}
								<tr><td>Surplus solar exported (net metering credit)</td><td class="num">{mwh(na.exported_kwh)} MWh</td><td class="num">0.7 kg/kWh</td><td class="num">–</td><td class="num">−{t(na.exported_kwh * 0.7)}</td></tr>
							{/if}
							<tr><td>Diesel generators</td><td class="num">{mwh(nb.diesel_kwh)} MWh</td><td class="num">0.8 kg/kWh</td><td class="num">{t(nb.diesel_kg)}</td><td class="num">{t(na.diesel_kg)}</td></tr>
							<tr class="total"><td>Total caused</td><td></td><td></td><td class="num">{t(nb.total_kg)}</td><td class="num">{t(na.total_kg)}</td></tr>
							<tr><td>Absorbed by trees</td><td class="num">{fmtCount(result.trees.count)} trees</td><td class="num">{fmtNum(result.trees.per_tree_central)} kg/tree</td><td class="num">−{t(nb.trees_kg)}</td><td class="num">−{t(na.trees_kg)}</td></tr>
							<tr class="total"><td>Net</td><td></td><td></td><td class="num">{signed(nb.net_kg)}</td><td class="num">{signed(na.net_kg)}</td></tr>
						</tbody>
					</table>
				</div>
				<p class="hint">
					Solar is not subtracted as a separate line: what the panels generate already shows up as less electricity bought from the grid.
					{#if result.solar.unused_kwh > 1000}
						Of the extra solar, {mwh(result.solar.unused_kwh)} MWh a year ({fmtNum((result.solar.unused_kwh / result.solar.extra_kwh) * 100)}%) is more than the campus needs at that moment:
						{exportSurplus ? 'with net metering it is exported and credited (the credit row above).' : 'without net metering it is wasted and not counted.'}
					{/if}
				</p>
			</section>

			<section>
				<h2>CO₂ from grid electricity, month by month</h2>
				{#if exportSurplus}<p class="hint">Bars show electricity bought from the grid; the net-metering credit for exported surplus isn't included month by month.</p>{/if}
				<CompareChart kind="bars" rows={monthRows} unit="t" names={['Today', `With +${solar}% solar`]} title="Monthly CO2 from grid electricity, today versus scenario" />
			</section>

			<section>
				<div class="row">
					<h2>A sunny day in detail: {fmtDate(result.day.date)}</h2>
					<label class="inline">Month
						<select bind:value={month}>
							<option value="">Sunniest month</option>
							{#each result.months as m (m)}<option value={m}>{monthName(m)} {m.slice(0, 4)}</option>{/each}
						</select>
					</label>
				</div>
				<p class="hint">Power drawn from the grid through the day. More solar pulls the middle of the day down; it does nothing at night.</p>
				<CompareChart kind="lines" rows={dayRows} unit="kW" names={['Today', `With +${solar}% solar`]} title="Grid power through the day, today versus scenario" labelEvery={12} />
			</section>

			<section>
				<h2>How the tree number is worked out for this campus</h2>
				<ol class="assume">
					<li><strong>How many trees:</strong> {fmtCount(result.trees.count)}. IIIT-H's own figure is "over 2,000 trees" on its 66-acre campus (institute blog, July 2025). The 1,200 saplings planted that month are too small to count yet.</li>
					<li><strong>How big one tree is:</strong> biomass = exp(−1.803 − 0.976·E + 0.976·ln(density) + 2.673·ln(D) − 0.0299·ln(D)²), from Chave et al. 2014. E is the climate stress at the site: 0.937 at the campus, read from their global map. Hyderabad is hot, dry and seasonal, so trees are shorter for the same trunk (about {fmtNum(result.trees.implied_height_m)} m at 26.4 cm). That gives about {fmtNum(result.trees.biomass_per_tree)} kg per tree.</li>
					<li><strong>How much it grows:</strong> the trunk widens by 0.3 to 0.8 cm a year (0.5 central). The biomass that adds, plus 26% for roots, is 47% carbon; carbon × 3.67 = CO₂.</li>
					<li><strong>Result:</strong> about {fmtNum(result.trees.per_tree_central)} kg of CO₂ per tree per year ({fmtNum(result.trees.per_tree_low)} to {fmtNum(result.trees.per_tree_high)}), times {fmtCount(result.trees.count)} trees = {t(result.trees.central)} t a year. The project brief quotes about 21.7 kg per tree, inside this range.</li>
				</ol>
				<p class="warn">
					<strong>The tree count decides this.</strong> <code>Calculations.docx</code> uses 46,000 trees and gives no source for it; that is one tree for every 6 m² of the whole campus, buildings included.
					With 46,000 trees at the same per-tree rate, absorption would be {t(result.trees.with_document_count.central)} t and net emissions {signed(result.trees.net_with_document_count_kg)} t: still not net zero.
					That document's own total, {t(result.trees.document_kg)} t a year, also uses a formula that gives 3,091 kg per tree and divides the whole stock by 50 years.
					<strong>Still assumed:</strong> wood density 0.6 g/cm³ (no species list for the campus was found) and the growth rate. The 26.4 cm trunk diameter is from the document.
				</p>
			</section>

			<section>
				<h2>What is and is not counted</h2>
				<ul class="assume">
					<li><strong>Counted:</strong> grid electricity from the substation meter <code>EM-CR-SB00-00</code> ({result.period.days} of 365 days in 2025, source <code>{result.source}</code>), and generator output from <code>EM-CR-SB00-01</code>.</li>
					<li><strong>Not counted:</strong> water pumping, sewage treatment and waste. The database has no usable history for them.</li>
					<li>Grid factor 0.7 kg CO₂/kWh is from <code>Calculations.docx</code>. The diesel factor 0.8 kg/kWh is a typical value, not from the project documents.</li>
					<li>"Today's solar" is the sum of {result.solar.nodes.join(', ')} ({mwh(result.solar.measured_kwh)} MWh in the year): the solar meters with a clean, near-complete 2025. SL-NI01-00, SL-VN02-01 and SL-PL96-00 have only part of the year, and SL-VN95-00 has no measured 2025 data here, so real solar is higher and every "× today" figure on this page is an overestimate.</li>
					<li>Extra solar is assumed to follow the same daily pattern and to sit behind the campus meter.</li>
				</ul>
			</section>
		</div>
	{/if}
</div>

<style>
	.wrap { max-width: 1040px; margin: 0 auto; padding: 24px 20px 60px; display: grid; gap: 16px; }
	h1 { font-size: 22px; font-weight: 650; }
	header p { color: var(--ink-2); margin-top: 6px; max-width: 75ch; }
	h2 { font-size: 15px; font-weight: 650; }
	.hint { font-size: 12.5px; color: var(--ink-2); max-width: 90ch; }
	.hint strong { color: var(--ink); }
	.warn { font-size: 12.5px; color: var(--warn-ink); background: var(--warn-bg); padding: 8px 10px; border-radius: 8px; }
	.filters { display: grid; gap: 10px; padding: 14px 16px; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
	.filters label { display: grid; gap: 6px; font-size: 13px; color: var(--ink-2); }
	.filters strong { color: var(--ink); }
	input[type='range'] { width: 100%; accent-color: var(--ink); }
	.filters .check { display: flex; align-items: start; gap: 8px; font-size: 12.5px; }
	.check input { margin-top: 2px; accent-color: var(--ink); }
	.link { border: 0; background: none; padding: 0; color: var(--series-1); font: inherit; font-weight: 600; cursor: pointer; }
	.stale { opacity: 0.5; transition: opacity 0.15s; }
	.tiles { display: grid; grid-template-columns: 1.5fr repeat(3, 1fr); gap: 10px; }
	.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
	.tile.big { border-color: var(--ink); }
	.tl { font-size: 12px; color: var(--ink-2); }
	.tv { font-size: 22px; font-weight: 650; margin-top: 2px; }
	.big .tv { font-size: 30px; }
	.td { font-size: 12.5px; font-weight: 600; }
	.td span { color: var(--ink-2); font-weight: 400; }
	section { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px; display: grid; grid-template-columns: minmax(0, 1fr); gap: 8px; margin-bottom: 16px; }
	.row { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; justify-content: space-between; }
	.inline { font-size: 12.5px; color: var(--ink-2); display: flex; gap: 6px; align-items: center; }
	select { padding: 4px 6px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface); }
	.scrollx { overflow-x: auto; }
	th, td { padding: 6px 8px 6px 0; border-bottom: 1px solid var(--grid); text-align: left; font-size: 13px; white-space: nowrap; }
	th { font-weight: 600; color: var(--ink-2); font-size: 12px; }
	th.num, td.num { text-align: right; padding-right: 0; padding-left: 12px; }
	tr.total td { font-weight: 650; }
	.assume { margin: 0; padding-left: 18px; display: grid; gap: 5px; font-size: 12.5px; color: var(--ink-2); }
	code { font-family: ui-monospace, monospace; font-size: 12px; }
	@media (max-width: 800px) { .tiles { grid-template-columns: 1fr 1fr; } .tile.big { grid-column: 1 / -1; } }
</style>
