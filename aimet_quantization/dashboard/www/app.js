/* QIDK Net-Zero demo dashboard.
 * Replay/plant/day-ahead views use the outputs the QIDK NPU recorded for every test window (P4).
 * "Run on NPU" sends a window to /api/run on the QIDK, which executes it live on the Hexagon HTP.
 * Anomaly rule = scripts/anomaly.py: flag if |actual - forecast - median_h| > limit_h (h = IST hour). */
"use strict";

const STEP_MS = 15 * 60 * 1000, IST_MS = 5.5 * 3600 * 1000, LOOKBACK = 96;
const UNRELIABLE = new Set(["Demand"]);   // FP32 R² < 0 -> not used for anomaly verdicts (doc 12)
const COLORS = { Solar: "#f5b83d", Grid: "#4ea8de", Demand: "#b07cf0", Solar_PL00: "#f5b83d" };
const MODELS = {};
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const fmtW = (w) => (w == null || isNaN(w)) ? "–" : Math.abs(w) >= 1e3 ? (w / 1e3).toFixed(1) + " kW" : w.toFixed(0) + " W";
const fmtT = (ms) => new Date(ms + IST_MS).toISOString().slice(0, 16).replace("T", " ") + " IST";
const hourIST = (ms) => new Date(ms + IST_MS).getUTCHours();
const tz = (ts) => uPlot.tzDate(new Date(ts * 1e3), "Asia/Kolkata");

// ---------------------------------------------------------------- data
async function loadModel(name) {
  const base = `data/${name}/`;
  const meta = await (await fetch(base + "meta.json")).json();
  const f32 = async (f) => new Float32Array(await (await fetch(base + f)).arrayBuffer());
  const [tl, y, npu, fp32] = await Promise.all(["timeline.f32", "y.f32", "npu.f32", "fp32.f32"].map(f32));
  const m = { name, meta, tl, y, npu, fp32, F: meta.features.length, H: meta.horizon, D: meta.domains.length,
              n: meta.n, start: Date.parse(meta.timeline_start_utc), inj: new Map() };
  m.HD = m.H * m.D;
  return m;
}
const tTarget = (m, i, h = 0) => m.start + (i + LOOKBACK + h) * STEP_MS;   // time of forecast step h of window i
const pred = (m, i, h, d, arr) => (arr || m.npu)[i * m.HD + h * m.D + d] * m.meta.model_capacities_W[d];
const actualBase = (m, i, h, d) => m.y[i * m.HD + h * m.D + d] * m.meta.data_capacities_W[d];
function actual(m, i, d) {                                    // first-step actual incl. injected faults
  const base = actualBase(m, i, 0, d), f = m.inj.get(i + ":" + d);
  if (!f) return base;
  const cap = m.meta.model_capacities_W[d];
  return { dropout: 0, spike: base + 0.5 * cap, panel: 0.6 * base, stuck: f.value }[f.type];
}
function check(m, i, d, act) {                              // -> {r, med, lim, flag}
  const h = hourIST(tTarget(m, i)), med = m.meta.anomaly.median_W[h][d], lim = m.meta.anomaly.limit_W[h][d];
  const r = act - pred(m, i, 0, d);
  return { r, med, lim, flag: Math.abs(r - med) > lim };
}

// ---------------------------------------------------------------- charts
function chartOpts(el, title, series, extra = {}) {
  return Object.assign({
    title, width: Math.max(300, el.clientWidth - 4), height: extra.height || 220, tzDate: tz,
    legend: { live: true }, cursor: { drag: { x: false, y: false } },
    scales: { x: { time: true } },
    axes: [{ stroke: "#8b98a5", grid: { stroke: "#2b3642" }, ticks: { stroke: "#2b3642" } },
           { stroke: "#8b98a5", grid: { stroke: "#2b3642" }, ticks: { stroke: "#2b3642" }, size: 70,
             values: (u, v) => v.map((x) => fmtW(x)) }],
    series: [{}, ...series],
  }, extra);
}
const pointsOnly = (label, color, size = 9) =>
  ({ label, stroke: color, fill: color, width: 0, paths: () => null, points: { show: true, size, fill: color, stroke: color } });
function resizeAll() { for (const u of window.__charts || []) u.setSize({ width: Math.max(300, u.root.parentElement.clientWidth - 4), height: u.height }); }
function track(u) { (window.__charts = window.__charts || []).push(u); return u; }
window.addEventListener("resize", () => { clearTimeout(window.__rz); window.__rz = setTimeout(resizeAll, 150); });

// ---------------------------------------------------------------- replay view (campus 1 h and single plant)
function makeReplay(section, m, plotDomains, liveModelName) {
  section.appendChild($("#replay-tpl").content.cloneNode(true));
  let start = 96 * 3;
  while (start < m.n - 1 && hourIST(tTarget(m, start)) !== 11) start++;
  const st = { idx: start, playing: false, timer: null, span: 192 };
  const slider = $(".slider", section), charts = $(".charts", section);
  slider.max = m.n - 1;
  const domSel = $(".inj-domain", section);
  m.meta.domains.forEach((d, j) => domSel.add(new Option(d, j)));

  const plots = plotDomains.map((d) => {
    const div = document.createElement("div"); charts.appendChild(div);
    const c = COLORS[m.meta.domains[d]] || "#f5b83d";
    return { d, u: track(new uPlot(chartOpts(charts, `${m.meta.domains[d]} — actual vs NPU forecast (+15 min) with typical band`, [
      { label: "band hi", stroke: "rgba(120,190,140,.5)", width: 1 },
      { label: "band lo", stroke: "rgba(120,190,140,.5)", width: 1 },
      { label: "NPU forecast", stroke: c, width: 2, dash: [6, 4] },
      { label: "FP32 forecast", stroke: "#9aa5b1", width: 1, show: false },
      { label: "actual", stroke: "#e6edf3", width: 2 },
      pointsOnly("anomaly", "#f85149", 14),
      pointsOnly("injected", "#ff9f43", 9),
    ], { bands: [{ series: [1, 2], fill: "rgba(120,190,140,.16)" }] }), [[]], div)) };
  });
  const rdiv = document.createElement("div"); charts.appendChild(rdiv);
  const d0 = plotDomains[0];
  const ru = track(new uPlot(chartOpts(charts, `${m.meta.domains[d0]} residual (actual − forecast − typical bias) vs alarm limit`, [
    { label: "+limit", stroke: "rgba(248,81,73,.6)", width: 1 },
    { label: "−limit", stroke: "rgba(248,81,73,.6)", width: 1 },
    { label: "residual", stroke: "#58a6ff", width: 1.5 },
    pointsOnly("anomaly", "#f85149", 14),
  ], { height: 170, bands: [{ series: [1, 2], fill: "rgba(120,190,140,.10)" }] }), [[]], rdiv));

  // notable natural anomalies (no injection) for the "jump to" list
  const moments = [];
  for (let i = 0; i < m.n; i++) {
    const c = check(m, i, d0, actualBase(m, i, 0, d0));
    if (c.flag) moments.push({ i, score: Math.abs(c.r - c.med) / c.lim });
  }
  moments.sort((a, b) => b.score - a.score);
  const msel = $(".moments", section);
  moments.slice(0, 20).sort((a, b) => a.i - b.i).forEach((mo) =>
    msel.add(new Option(`${fmtT(tTarget(m, mo.i))} · anomaly ×${mo.score.toFixed(1)}`, mo.i)));
  msel.onchange = () => { if (msel.value !== "") go(+msel.value); msel.value = ""; };

  function render() {
    const i1 = st.idx, i0 = Math.max(0, i1 - st.span + 1), N = i1 - i0 + 1;
    const ts = new Array(N);
    for (let k = 0; k < N; k++) ts[k] = tTarget(m, i0 + k) / 1000;
    let anomalies = 0;
    for (const p of plots) {
      const hi = [], lo = [], fc = [], f32 = [], act = [], an = [], inj = [];
      for (let k = 0; k < N; k++) {
        const i = i0 + k, a = actual(m, i, p.d), c = check(m, i, p.d, a), f = pred(m, i, 0, p.d);
        hi.push(f + c.med + c.lim); lo.push(f + c.med - c.lim);
        fc.push(f); f32.push(pred(m, i, 0, p.d, m.fp32)); act.push(a);
        an.push(c.flag ? a : null); inj.push(m.inj.has(i + ":" + p.d) ? a : null);
        if (c.flag) anomalies++;
      }
      p.u.setData([ts, hi, lo, fc, f32, act, an, inj]);
    }
    const rp = [], rm = [], rr = [], ra = [];
    for (let k = 0; k < N; k++) {
      const i = i0 + k, c = check(m, i, d0, actual(m, i, d0));
      rp.push(c.lim); rm.push(-c.lim); rr.push(c.r - c.med); ra.push(c.flag ? c.r - c.med : null);
    }
    ru.setData([ts, rp, rm, rr, ra]);
    slider.value = i1;
    $(".now", section).textContent = fmtT(tTarget(m, i1));
    $(".t-target", section).textContent = fmtT(tTarget(m, i1));
    // verdicts
    let html = "<tr><th></th><th>actual</th><th>forecast</th><th>residual</th><th>limit</th><th></th></tr>";
    for (let d = 0; d < m.D; d++) {
      const a = actual(m, i1, d), c = check(m, i1, d, a);
      html += `<tr><td style="color:${COLORS[m.meta.domains[d]]}">${m.meta.domains[d]}</td><td>${fmtW(a)}</td><td>${fmtW(pred(m, i1, 0, d))}</td>
        <td>${fmtW(c.r - c.med)}</td><td>±${fmtW(c.lim)}</td>${UNRELIABLE.has(m.meta.domains[d]) ? '<td class="na" title="Demand head has negative R² in FP32: not used for anomaly detection">n/a</td>' : `<td class="${c.flag ? "bad" : "ok"}">${c.flag ? "ANOMALY" : "ok"}</td>`}</tr>`;
    }
    $(".verdicts", section).innerHTML = html;
    let fh = "<tr><th>step</th>" + m.meta.domains.map((d) => `<th>${d}</th>`).join("") + "</tr>";
    for (let h = 0; h < m.H; h++)
      fh += `<tr><td>+${(h + 1) * 15} min</td>` + m.meta.domains.map((_, d) => `<td>${fmtW(Math.max(d === 0 ? 0 : -Infinity, pred(m, i1, h, d)))}</td>`).join("") + "</tr>";
    $(".fc", section).innerHTML = fh;
    $(".inj-list", section).textContent = m.inj.size ? `${m.inj.size} injected fault reading(s) active` : "";
    setKpi("anom", anomalies, "anomalies in view");
  }
  function go(i) { st.idx = Math.max(0, Math.min(m.n - 1, i)); render(); }
  function play(on) {
    st.playing = on ?? !st.playing;
    $(".play", section).textContent = st.playing ? "❚❚ Pause" : "▶ Play";
    clearInterval(st.timer);
    if (st.playing) st.timer = setInterval(() => { if (st.idx >= m.n - 1) return play(false); go(st.idx + 1); },
                                           1000 / +$(".speed", section).value);
  }
  function inject(type) {
    const d = +domSel.value, L = +$(".inj-len", section).value, value = actualBase(m, st.idx, 0, d);
    for (let k = 0; k < L && st.idx + k < m.n; k++) m.inj.set((st.idx + k) + ":" + d, { type, value });
    render();
  }
  $(".play", section).onclick = () => play();
  $(".speed", section).onchange = () => st.playing && play(true);
  slider.oninput = () => go(+slider.value);
  $(".jump", section).onchange = (e) => {
    const t = Date.parse(e.target.value + ":00+05:30");
    if (!isNaN(t)) go(Math.round((t - m.start) / STEP_MS) - LOOKBACK);
  };
  $$(".inject [data-fault]", section).forEach((b) => (b.onclick = () => inject(b.dataset.fault)));
  $(".clear", section).onclick = () => { m.inj.clear(); render(); };
  $(".to-live", section).onclick = () => Live.open(liveModelName, st.idx);
  render();
  return {
    m, render, go, play, inject, clear: () => { m.inj.clear(); render(); },
    step: (k) => go(st.idx + k),
    nextAnomaly() {
      for (let i = st.idx + 1; i < m.n; i++) for (let d = 0; d < m.D; d++)
        if (!UNRELIABLE.has(m.meta.domains[d]) && check(m, i, d, actual(m, i, d)).flag) return go(i);
    },
    get idx() { return st.idx; },
  };
}

// ---------------------------------------------------------------- day-ahead view
function makeDayAhead(m, typ) {
  const sl = $("#da-slider"); sl.max = m.n - 1;
  const el = $("#da-chart"), net = $("#da-net"), te = $("#typ-chart");
  const u = track(new uPlot(chartOpts(el, "Day-ahead forecast (NPU) vs actual", [
    { label: "Solar forecast", stroke: COLORS.Solar, width: 2, dash: [6, 4] }, { label: "Solar actual", stroke: COLORS.Solar, width: 1.5 },
    { label: "Grid forecast", stroke: COLORS.Grid, width: 2, dash: [6, 4] }, { label: "Grid actual", stroke: COLORS.Grid, width: 1.5 },
  ], { height: 300 }), [[]], el));
  const un = track(new uPlot(chartOpts(net, "Room demand (sub-meters): forecast vs actual", [
    { label: "Demand forecast", stroke: COLORS.Demand, width: 2, dash: [6, 4] }, { label: "Demand actual", stroke: COLORS.Demand, width: 1.5 },
  ], { height: 180 }), [[]], net));
  const hours = [...Array(24).keys()];
  const tu = track(new uPlot(Object.assign(chartOpts(te, "", [
    { label: "P90", stroke: "rgba(120,190,140,.5)" }, { label: "P10", stroke: "rgba(120,190,140,.5)" },
    { label: "median", stroke: "#e6edf3", width: 2 },
    { label: "this day forecast", stroke: COLORS.Solar, width: 2, dash: [6, 4] }, { label: "this day actual", stroke: COLORS.Solar, width: 1.5 },
  ], { height: 220, bands: [{ series: [1, 2], fill: "rgba(120,190,140,.16)" }] }), {
    scales: { x: { time: false } },
    axes: [{ stroke: "#8b98a5", grid: { stroke: "#2b3642" }, values: (u, v) => v.map((x) => x + ":00") },
           { stroke: "#8b98a5", grid: { stroke: "#2b3642" }, size: 70, values: (u, v) => v.map((x) => fmtW(x)) }],
  }), [[]], te));
  function render(i) {
    const ts = [], sf = [], sa = [], gf = [], ga = [], df = [], da = [];
    const hf = Array(24).fill(0), ha = Array(24).fill(0), hc = Array(24).fill(0);
    let E = [[0, 0], [0, 0], [0, 0]];
    for (let h = 0; h < m.H; h++) {
      const t = tTarget(m, i, h); ts.push(t / 1000);
      const v = [0, 1, 2].map((d) => [Math.max(d === 0 ? 0 : -1e12, pred(m, i, h, d)), actualBase(m, i, h, d)]);
      sf.push(v[0][0]); sa.push(v[0][1]); gf.push(v[1][0]); ga.push(v[1][1]); df.push(v[2][0]); da.push(v[2][1]);
      v.forEach((x, d) => { E[d][0] += x[0] * 0.25 / 1000; E[d][1] += x[1] * 0.25 / 1000; });
      const hh = hourIST(t); hf[hh] += v[0][0]; ha[hh] += v[0][1]; hc[hh]++;
    }
    u.setData([ts, sf, sa, gf, ga]); un.setData([ts, df, da]);
    tu.setData([hours, typ["90"][0], typ["10"][0], typ["50"][0], hf.map((x, k) => x / (hc[k] || 1)), ha.map((x, k) => x / (hc[k] || 1))]);
    $("#da-time").textContent = fmtT(tTarget(m, i, 0) - STEP_MS) + "  →  next 24 h";
    $("#da-energy").innerHTML = "<tr><th></th><th>forecast</th><th>actual</th><th>error</th></tr>" + ["Solar", "Grid", "Demand"].map((d, k) =>
      `<tr><td style="color:${COLORS[d]}">${d}</td><td>${E[k][0].toFixed(0)} kWh</td><td>${E[k][1].toFixed(0)} kWh</td><td>${(100 * (E[k][0] - E[k][1]) / Math.max(1, E[k][1])).toFixed(1)} %</td></tr>`).join("") +
      `<tr><td>Net (demand − solar)</td><td>${(E[2][0] - E[0][0]).toFixed(0)} kWh</td><td>${(E[2][1] - E[0][1]).toFixed(0)} kWh</td><td></td></tr>`;
  }
  sl.value = Math.min(m.n - 1, 96 * 2 + 72); sl.oninput = () => render(+sl.value);
  render(+sl.value);
  return { m, step: (k) => { sl.value = Math.max(0, Math.min(m.n - 1, +sl.value + k)); render(+sl.value); } };
}

// ---------------------------------------------------------------- live NPU view
const Live = {
  m: null, idx: 0, win: null, orig: null, nextRow: null, times: null, source: "", lastResult: null,
  init() {
    $("#lv-model").onchange = () => this.open($("#lv-model").value, 0);
    $("#lv-index").onchange = () => this.open(this.m.name, +$("#lv-index").value);
    $("#lv-template").onclick = () => this.download();
    $("#lv-file").onchange = (e) => e.target.files[0] && this.upload(e.target.files[0]);
    $("#lv-reset").onclick = () => this.open(this.m.name, this.idx);
    $("#ed-apply").onclick = () => this.edit();
    $("#ed-feature").onchange = () => this.drawEdit();
    $("#lv-run-npu").onclick = () => this.run("aimet_w8a16");
    $("#lv-run-cpu").onclick = () => this.run("fp32");
    const el = $("#ed-chart");
    this.eu = track(new uPlot(chartOpts(el, "", [
      { label: "original", stroke: "#9aa5b1", width: 1.5 }, { label: "edited / uploaded", stroke: "#f85149", width: 2 }], { height: 170 }), [[]], el));
  },
  ensureChart() {
    const el = $("#lv-chart"); el.innerHTML = "";
    const m = this.m, series = [];
    m.meta.domains.forEach((d) => series.push(
      { label: `${d} live`, stroke: COLORS[d], width: 2.5 },
      { label: `${d} recorded NPU`, stroke: COLORS[d], width: 1, dash: [3, 3] },
      { label: `${d} actual`, stroke: "#e6edf3", width: 1 }));
    this.lu = track(new uPlot(chartOpts(el, "Live result vs recorded NPU output and actual", series, { height: 240 }), [[]], el));
  },
  open(name, idx) {
    showTab("live");
    const m = MODELS[name]; this.m = m; this.idx = Math.max(0, Math.min(m.n - 1, idx | 0));
    $("#lv-model").value = name; $("#lv-index").max = m.n - 1; $("#lv-index").value = this.idx;
    this.win = m.tl.slice(this.idx * m.F, (this.idx + LOOKBACK) * m.F);
    this.orig = this.win.slice();
    this.nextRow = this.idx + LOOKBACK < m.tl.length / m.F ? m.tl.slice((this.idx + LOOKBACK) * m.F, (this.idx + LOOKBACK + 1) * m.F) : null;
    this.times = [...Array(LOOKBACK + 1).keys()].map((r) => m.start + (this.idx + r) * STEP_MS);
    this.source = "test"; this.edited = false;
    $("#lv-source").textContent = "(test window from the replay data — unedited)";
    $("#lv-time").textContent = `inputs ${fmtT(this.times[0])} … ${fmtT(this.times[95])}`;
    const fs = $("#ed-feature"); fs.innerHTML = "";
    m.meta.features.forEach((f, j) => { if (!f.cyclic && f.col !== "was_imputed") fs.add(new Option(f.name, j)); });
    const pref = m.meta.features.findIndex((f) => /active_power_W$/.test(f.name));
    if (pref >= 0) fs.value = pref;
    this.ensureChart(); this.drawEdit(); this.msg("");
    $("#lv-fc").innerHTML = $("#lv-anom").innerHTML = "";
  },
  phys(j, v) { const f = this.m.meta.features[j]; return f.col === "was_imputed" ? v : v * (f.max - f.min) + f.min; },
  norm(j, v) { const f = this.m.meta.features[j]; return f.col === "was_imputed" ? v : (v - f.min) / ((f.max - f.min) || 1); },
  drawEdit() {
    const j = +$("#ed-feature").value, F = this.m.F;
    const ts = this.times.slice(0, LOOKBACK).map((t) => t / 1000);
    this.eu.setData([ts, ts.map((_, r) => this.phys(j, this.orig[r * F + j])), ts.map((_, r) => this.phys(j, this.win[r * F + j]))]);
  },
  edit() {
    const j = +$("#ed-feature").value, op = $("#ed-op").value, k = +$("#ed-k").value, n = Math.max(1, Math.min(96, +$("#ed-steps").value));
    const F = this.m.F, f = this.m.meta.features[j], r0 = LOOKBACK - n, frozen = this.phys(j, this.win[r0 * F + j]);
    for (let r = r0; r < LOOKBACK; r++) {
      const v = this.phys(j, this.win[r * F + j]);
      const nv = { zero: 0, scale: v * k, freeze: frozen, spike: v + k * f.max }[op];
      this.win[r * F + j] = this.norm(j, nv);
    }
    this.edited = true;
    $("#lv-source").textContent = `(${this.source === "csv" ? "uploaded CSV" : "test window"} — edited: ${f.name} ${op} over last ${n} steps)`;
    this.drawEdit();
  },
  cyclicNorm(t) {          // the 6 cyclic features for timestamp t (ms), MinMax-normalized like training
    const d = new Date(t + IST_MS), hod = d.getUTCHours() + d.getUTCMinutes() / 60, dow = (d.getUTCDay() + 6) % 7;
    const doy = Math.floor((Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()) - Date.UTC(d.getUTCFullYear(), 0, 1)) / 864e5) + 1;
    const v = { hod_sin: Math.sin(2 * Math.PI * hod / 24), hod_cos: Math.cos(2 * Math.PI * hod / 24),
                dow_sin: Math.sin(2 * Math.PI * dow / 7), dow_cos: Math.cos(2 * Math.PI * dow / 7),
                doy_sin: Math.sin(2 * Math.PI * doy / 365.25), doy_cos: Math.cos(2 * Math.PI * doy / 365.25) };
    const out = {};
    this.m.meta.features.forEach((f, j) => { if (f.cyclic) out[j] = (v[f.col] - f.min) / (f.max - f.min); });
    return out;
  },
  download() {
    const { text, R } = this.buildCSV(), a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
    a.download = `${this.m.name}_window_${this.idx}.csv`; a.click();
    this.msg(`Template written: ${R} rows (${R > 96 ? "row 97 = actual next reading" : "no next reading"}), physical units.`, "ok");
  },
  async selfTest() {   // template -> real upload parser -> compare with the original window -> run on NPU
    const ref = this.win.slice(), refNext = this.nextRow && this.nextRow.slice(), { text } = this.buildCSV();
    await this.upload({ name: "selftest.csv", text: async () => text });
    let d = 0, dc = 0;
    this.m.meta.features.forEach((f, j) => { for (let r = 0; r < LOOKBACK; r++) {
      const e = Math.abs(this.win[r * this.m.F + j] - ref[r * this.m.F + j]); if (f.cyclic) dc = Math.max(dc, e); else d = Math.max(d, e); } });
    this.msg(`CSV self-test: max |Δ| sensors ${d.toExponential(1)}, cyclic (recomputed from timestamps) ${dc.toExponential(1)}` +
             (refNext ? `, next reading ${Math.max(...this.nextRow.map((v, j) => Math.abs(v - refNext[j]))).toExponential(1)}` : ""), d < 1e-4 && dc < 1e-4 ? "ok" : "err");
    await this.run("aimet_w8a16");
  },
  buildCSV() {
    const m = this.m, F = m.F, rows = [["timestamp", ...m.meta.features.map((f) => f.name)].join(",")];
    const R = this.nextRow ? LOOKBACK + 1 : LOOKBACK;
    for (let r = 0; r < R; r++) {
      const src = r < LOOKBACK ? this.win.subarray(r * F, (r + 1) * F) : this.nextRow;
      rows.push([new Date(this.times[r]).toISOString(), ...[...src].map((v, j) => +this.phys(j, v).toPrecision(9))].join(","));
    }
    return { text: rows.join("\n") + "\n", R };
  },
  async upload(file) {
    try {
      const m = this.m, F = m.F, text = await file.text();
      const lines = text.trim().split(/\r?\n/).filter((l) => l.trim());
      const head = lines[0].split(",").map((h) => h.trim());
      const ti = head.indexOf("timestamp");
      if (ti < 0) throw new Error("missing 'timestamp' column");
      const col = {}; head.forEach((h, k) => (col[h] = k));
      const need = m.meta.features.filter((f) => !f.cyclic && f.col !== "was_imputed").map((f) => f.name);
      const missing = need.filter((n) => !(n in col));
      if (missing.length) throw new Error(`missing ${missing.length} column(s): ${missing.slice(0, 4).join(", ")}${missing.length > 4 ? " …" : ""}`);
      const body = lines.slice(1);
      if (body.length < 96 || body.length > 97) throw new Error(`need 96 rows (+ optional 97th), got ${body.length}`);
      const normalized = $("#lv-normalized").checked, win = new Float32Array(96 * F), times = [];
      let next = null, outOfRange = 0;
      body.forEach((line, r) => {
        const c = line.split(","), t = Date.parse(c[ti]);
        if (isNaN(t)) throw new Error(`row ${r + 2}: bad timestamp '${c[ti]}'`);
        times.push(t);
        const cyc = this.cyclicNorm(t), row = new Float32Array(F);
        m.meta.features.forEach((f, j) => {
          if (f.cyclic) { row[j] = cyc[j]; return; }
          const raw = f.name in col ? c[col[f.name]] : "0";
          const v = parseFloat(raw);
          if (isNaN(v)) throw new Error(`row ${r + 2}, ${f.name}: not a number ('${raw}')`);
          row[j] = normalized ? v : this.norm(j, v);
          if (row[j] < -0.05 || row[j] > 1.05) outOfRange++;
        });
        if (r < 96) win.set(row, r * F); else next = row;
      });
      for (let r = 1; r < times.length; r++)
        if (times[r] - times[r - 1] !== STEP_MS) throw new Error(`rows ${r + 1}→${r + 2}: timestamps must be 15 min apart`);
      this.win = win; this.orig = win.slice(); this.nextRow = next; this.times = times.length === 97 ? times : [...times, times[95] + STEP_MS];
      this.source = "csv"; this.edited = false;
      $("#lv-source").textContent = `(uploaded: ${file.name})`;
      $("#lv-time").textContent = `inputs ${fmtT(this.times[0])} … ${fmtT(this.times[95])}`;
      this.drawEdit();
      this.msg(`Loaded ${file.name}: 96 rows${next ? " + next reading" : ""}, ${normalized ? "normalized" : "physical units → normalized"}` +
               (outOfRange ? ` · ⚠ ${outOfRange} value(s) outside the training range` : ""), outOfRange ? "err" : "ok");
    } catch (e) { this.msg("CSV rejected: " + e.message, "err"); }
    if ($("#lv-file")) $("#lv-file").value = "";
  },
  targetsFromRow(row) {    // actual Solar/Grid/Demand (W) from one row of sensor features, as in the training script
    const m = this.m, F = m.meta.features, val = (name) => { const j = F.findIndex((f) => f.name === name); return j < 0 ? 0 : this.phys(j, row[j]); };
    if (m.name === "single_node") return [val("active_power_W")];
    const sum = (re) => F.reduce((s, f, j) => s + (re.test(f.name) ? this.phys(j, row[j]) : 0), 0);
    return [sum(/^SL-.*__active_power_W$/), val("EM-CR-SB00-00__gd_power_W") + val("EM-CR-KH95-00__power_total_W"), sum(/^SR-EM-.*__power_W$/)];
  },
  async run(variant) {
    const m = this.m, btns = [$("#lv-run-npu"), $("#lv-run-cpu")];
    btns.forEach((b) => (b.disabled = true));
    $("#lv-status").textContent = variant === "fp32" ? "running FP32 on the QIDK CPU…" : "running on the QIDK Hexagon NPU…";
    const t0 = performance.now();
    try {
      const res = await fetch(`api/run?model=${m.name}&variant=${variant}`, {
        method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: this.win.buffer.slice(0) });
      const r = await res.json();
      if (!r.ok) throw new Error(r.error + (r.log ? " — " + r.log : ""));
      r.roundtrip_ms = performance.now() - t0;
      this.show(r);
      $("#lv-status").textContent = "";
    } catch (e) { $("#lv-status").innerHTML = `<span class="bad">${e.message}</span>`; }
    btns.forEach((b) => (b.disabled = false));
  },
  show(r) {
    const m = this.m, caps = m.meta.model_capacities_W, out = r.out;
    const W = (arr, h, d) => arr[h * m.D + d] * caps[d];
    const unedited = this.source === "test" && !this.edited;
    let maxdiff = null;
    if (unedited) {
      maxdiff = 0;
      for (let k = 0; k < m.HD; k++) maxdiff = Math.max(maxdiff, Math.abs(out[k] - (r.variant === "fp32" ? m.fp32 : m.npu)[this.idx * m.HD + k]));
    }
    $("#lv-timing").innerHTML = [
      ["backend", r.backend], ["model", `${m.name} · ${r.variant === "fp32" ? "FP32" : "AIMET W8A16"}`],
      ["NPU/CPU execute", r.execute_us > 0 ? (r.execute_us / 1000).toFixed(2) + " ms" : "–"],
      ["graph prepare (per launch)", r.finalize_us > 0 ? (r.finalize_us / 1000).toFixed(0) + " ms" : "–"],
      ["qnn-net-run total", r.wall_ms.toFixed(0) + " ms"], ["browser round trip", r.roundtrip_ms.toFixed(0) + " ms"],
      ["match vs recorded run", maxdiff == null ? "n/a (custom input)" : `max |Δ| = ${maxdiff.toExponential(1)}`],
    ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
    if (r.variant !== "fp32") setKpi("npu", r.execute_us / 1000, "ms NPU (live)");
    const Hs = Math.min(m.H, 8);
    $("#lv-fc").innerHTML = "<tr><th>step</th>" + m.meta.domains.map((d) => `<th>${d}</th>`).join("") + "</tr>" +
      [...Array(Hs).keys()].map((h) => `<tr><td>+${(h + 1) * 15} min</td>` + m.meta.domains.map((_, d) => `<td>${fmtW(Math.max(d === 0 ? 0 : -Infinity, W(out, h, d)))}</td>`).join("") + "</tr>").join("") +
      (m.H > Hs ? `<tr><td colspan="${m.D + 1}">… ${m.H} steps in the chart</td></tr>` : "");
    // chart: live vs recorded vs actual over the horizon
    const ts = [...Array(m.H).keys()].map((h) => (this.times[95] + (h + 1) * STEP_MS) / 1000), data = [ts];
    m.meta.domains.forEach((_, d) => {
      data.push(ts.map((_, h) => W(out, h, d)));
      data.push(ts.map((_, h) => (this.source === "test" ? pred(m, this.idx, h, d) : null)));
      data.push(ts.map((_, h) => (this.source === "test" ? actualBase(m, this.idx, h, d) : null)));
    });
    this.lu.setData(data);
    // anomaly check of the next reading
    if (!this.nextRow) { $("#lv-anom").innerHTML = "<tr><td>no next reading supplied (row 97)</td></tr>"; return; }
    const act = this.targetsFromRow(this.nextRow), h = hourIST(this.times[96] ?? this.times[95] + STEP_MS);
    $("#lv-anom").innerHTML = "<tr><th></th><th>actual</th><th>forecast</th><th>residual</th><th>limit</th><th></th></tr>" +
      m.meta.domains.map((dn, d) => {
        const f = W(out, 0, d), med = m.meta.anomaly.median_W[h][d], lim = m.meta.anomaly.limit_W[h][d], res = act[d] - f - med;
        const bad = Math.abs(res) > lim;
        const v = UNRELIABLE.has(dn) ? '<td class="na">n/a</td>' : `<td class="${bad ? "bad" : "ok"}">${bad ? "ANOMALY" : "ok"}</td>`;
        return `<tr><td style="color:${COLORS[dn]}">${dn}</td><td>${fmtW(act[d])}</td><td>${fmtW(f)}</td><td>${fmtW(res)}</td><td>±${fmtW(lim)}</td>${v}</tr>`;
      }).join("");
  },
  msg(t, cls = "") { const el = $("#lv-msg"); el.textContent = t; el.className = "msg " + cls; },
};

// ---------------------------------------------------------------- header, tabs, keyboard
const kpi = {};
function setKpi(id, v, label) {
  if (!kpi[id]) { const d = document.createElement("div"); d.className = "kpi"; d.innerHTML = '<div class="v"></div><div class="l"></div>'; $("#kpis").appendChild(d); kpi[id] = d; }
  $(".v", kpi[id]).textContent = typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(v < 10 ? 2 : 1)) : v;
  $(".l", kpi[id]).textContent = label;
}
let activeTab = "replay";
function showTab(t) {
  activeTab = t;
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === t));
  $$(".tab").forEach((s) => s.classList.toggle("active", s.id === "tab-" + t));
  const model = { replay: "all_sensors_1h", plant: "single_node", dayahead: "all_sensors_24h", live: Live.m?.name || "all_sensors_1h", hood: "all_sensors_1h" }[t];
  const dv = MODELS[model]?.meta.device;
  if (dv) {
    setKpi("npu", dv.npu_ms, "ms / inference (NPU)");
    setKpi("sqnr", dv.sqnr_db, "dB fidelity vs FP32");
    setKpi("r2", `${dv.r2_fp32.toFixed(3)}→${dv.r2_npu.toFixed(3)}`, "R² FP32 → NPU");
    setKpi("agree", (100 * dv.flag_agreement).toFixed(2) + " %", "same anomaly verdicts");
  }
  ({ replay: views.replay, plant: views.plant })[t]?.render();
  requestAnimationFrame(resizeAll);
}
const views = {};

function buildHood() {
  const rows = [["", "1 h", "24 h", "plant"]];
  const ms = ["all_sensors_1h", "all_sensors_24h", "single_node"].map((n) => MODELS[n].meta.device);
  rows.push(["NPU vs FP32 (SQNR)", ...ms.map((d) => d.sqnr_db.toFixed(1) + " dB")]);
  rows.push(["QNN-only quantization", ...ms.map((d) => d.qairt_only_sqnr_db.toFixed(1) + " dB")]);
  rows.push(["R² FP32 → NPU", ...ms.map((d) => `${d.r2_fp32.toFixed(3)}→${d.r2_npu.toFixed(3)}`)]);
  rows.push(["same anomaly verdicts", ...ms.map((d) => (100 * d.flag_agreement).toFixed(2) + " %")]);
  rows.push(["NPU execute", ...ms.map((d) => d.npu_ms.toFixed(2) + " ms")]);
  rows.push(["FP32 graph on device CPU", "bit-exact", "bit-exact", "bit-exact"]);
  $("#hood-table").innerHTML = rows.map((r, k) => "<tr>" + r.map((c) => k ? `<td>${c}</td>` : `<th>${c}</th>`).join("") + "</tr>").join("");
  const m = MODELS.all_sensors_1h, el = $("#hood-chart"), ts = [], dif = [];
  for (let i = 0; i < m.n; i++) {
    let mx = 0;
    for (let k = 0; k < m.HD; k++) mx = Math.max(mx, Math.abs(m.npu[i * m.HD + k] - m.fp32[i * m.HD + k]));
    ts.push(tTarget(m, i) / 1000); dif.push(100 * mx);
  }
  $("#hood-model").textContent = "(all_sensors_1h, " + m.n + " windows)";
  track(new uPlot(chartOpts(el, "", [{ label: "max |NPU − FP32| (% of capacity)", stroke: "#58a6ff", width: 1 }],
    { height: 220, axes: [{ stroke: "#8b98a5", grid: { stroke: "#2b3642" } }, { stroke: "#8b98a5", grid: { stroke: "#2b3642" }, size: 64, values: (u, v) => v.map((x) => x.toFixed(2) + " %") }] }), [ts, dif], el));
}

document.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
  const v = views[activeTab];
  if (/^[1-5]$/.test(e.key)) return showTab(["replay", "plant", "dayahead", "live", "hood"][+e.key - 1]);
  if (!v) { if (e.key === "r" && activeTab === "live") Live.run("aimet_w8a16"); return; }
  const big = e.shiftKey ? 96 : 1;
  switch (e.key) {
    case " ": e.preventDefault(); v.play?.(); break;
    case "ArrowRight": v.step(big); break;
    case "ArrowLeft": v.step(-big); break;
    case "d": v.inject?.("dropout"); break;
    case "s": v.inject?.("spike"); break;
    case "f": v.inject?.("stuck"); break;
    case "p": v.inject?.("panel"); break;
    case "c": v.clear?.(); break;
    case "n": v.nextAnomaly?.(); break;
    case "r": if (v.m) Live.open(v.m.name === "single_node" ? "single_node" : "all_sensors_1h", v.idx); break;
  }
});

(async function main() {
  $$("#tabs button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
  fetch("api/status").then((r) => r.json()).then((s) => {
    $("#device").innerHTML = `<b>● live</b> ${s.device.trim()} · ${s.soc.trim()}<br>${s.server}`;
  }).catch(() => ($("#device").innerHTML = '<span class="bad">server offline</span>'));
  for (const n of ["all_sensors_1h", "single_node", "all_sensors_24h"]) MODELS[n] = await loadModel(n);
  views.replay = makeReplay($("#tab-replay"), MODELS.all_sensors_1h, [0, 1], "all_sensors_1h");
  views.plant = makeReplay($("#tab-plant"), MODELS.single_node, [0], "single_node");
  views.dayahead = makeDayAhead(MODELS.all_sensors_24h, MODELS.all_sensors_1h.meta.typical_W);
  Live.init(); Live.open("all_sensors_1h", views.replay.idx);
  const h = location.hash.slice(1);
  showTab(["replay", "plant", "dayahead", "live", "hood"].includes(h) ? h : h === "live-run" ? "live" : "replay");
  if (h === "live-run") Live.run("aimet_w8a16");      // test hooks: /#live-run fires one NPU run on load,
  if (h.startsWith("inject-")) views.replay.inject(h.slice(7));   // /#inject-dropout etc. injects at the current reading
  if (h === "csv-selftest") { showTab("live"); Live.selfTest(); }
  buildHood();
})();
