<!-- The NPU forecaster dashboard (nz_dashboard), embedded through the /npu-app/ proxy so it shares this origin. -->
<svelte:head>
	<title>NPU lab · Campus sensors</title>
</svelte:head>

<iframe src="/npu-app#live" title="NPU lab: replays, benchmarks and single NPU runs"></iframe>

<style>
	iframe { display: block; width: 100%; height: 100%; min-height: calc(100vh - 46px); border: 0; background: var(--surface); }
</style>
