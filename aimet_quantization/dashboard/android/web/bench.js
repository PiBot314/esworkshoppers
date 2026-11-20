/* "Benchmark" tab: runs benchmark_qidk/board/demo.sh on the QIDK through nz_dashboard (GET api/demo).
 * Models: the teammate's baseline / guided LSTMs (1 h, 24 h) and the CNN-LSTM. Loaded after app.js. */
"use strict";
(function () {
  const TEST_START = 1762747200;            // first test row, 2025-11-10 04:00 UTC (benchmark_qidk/data/info.json)
  const NAMES = { baseline_h4: "Baseline LSTM · 1 h", guided_h4: "Guided LSTM · 1 h", baseline_24h: "Baseline LSTM · 24 h",
                  guided_24h: "Guided LSTM · 24 h", cnn_lstm_24h: "CNN-LSTM · 24 h", lightgbm_24h: "LightGBM · 24 h" };
  const PREC = { w8a16: "NPU W8A16", fp16: "NPU FP16", w8a8: "NPU W8A8" };
  const sec = $("#tab-bench"), st = $("#bn-status"), out = $("#bn-out"), pre = $("#bn-text");
  let busy = false, chart = null;

  const ms = (v) => (v === null || v === undefined || v === "-" ? "–" : (+v < 1 ? (+v).toFixed(2) : (+v).toFixed(1)) + " ms");
  const table = (rows) => "<table>" + rows.map((r, i) => "<tr>" + r.map((c) => (i ? `<td>${c}</td>` : `<th>${c}</th>`)).join("") + "</tr>").join("") + "</table>";

  async function call(params, what) {
    if (busy) return null;
    busy = true;
    $$("#tab-bench button").forEach((b) => (b.disabled = true));
    const t0 = Date.now();
    const tick = setInterval(() => (st.textContent = `${what} on the QIDK… ${Math.round((Date.now() - t0) / 1000)} s`), 500);
    try {
      const r = await (await fetch("api/demo?" + new URLSearchParams(params))).json();
      st.textContent = r.ok ? `Done in ${((Date.now() - t0) / 1000).toFixed(1)} s (measured on the board).` : "Failed: " + (r.text || r.error);
      pre.textContent = r.text || "";
      return r.ok ? r.data : null;
    } catch (e) {
      st.innerHTML = `<span class="bad">${e.message || e}</span>`;
      return null;
    } finally {
      clearInterval(tick);
      busy = false;
      $$("#tab-bench button").forEach((b) => (b.disabled = false));
    }
  }

  function plot(d) {
    const el = $("#bn-chart");
    el.innerHTML = "";
    if (chart) { window.__charts = (window.__charts || []).filter((u) => u !== chart); chart.destroy(); }
    const t0 = TEST_START + (d.window + 96) * 900, x = [], cols = { actual: [[], []], npu: [[], []], ort: [[], []] };
    for (let s = 0; s < d.H; s++) {
      x.push(t0 + s * 900);
      for (const k of Object.keys(cols)) { cols[k][0].push(d[k][3 * s]); cols[k][1].push(d[k][3 * s + 1]); }
    }
    const pts = d.H <= 4 ? { points: { show: true, size: 7 } } : {};
    chart = track(new uPlot(chartOpts(el, `Forecast vs actual · ${NAMES[d.model]}`, [
      { label: "Solar actual", stroke: "#f2c94c", width: 2, ...pts },
      { label: `Solar ${PREC[d.prec]}`, stroke: "#f2994a", width: 2, dash: [6, 4], ...pts },
      { label: "Solar CPU FP32", stroke: "#bb6bd9", width: 1, dash: [2, 3], ...pts },
      { label: "Grid actual", stroke: "#56ccf2", width: 2, ...pts },
      { label: `Grid ${PREC[d.prec]}`, stroke: "#2f80ed", width: 2, dash: [6, 4], ...pts },
      { label: "Grid CPU FP32", stroke: "#27ae60", width: 1, dash: [2, 3], ...pts },
    ], { height: 260 }), [x, cols.actual[0], cols.npu[0], cols.ort[0], cols.actual[1], cols.npu[1], cols.ort[1]], el));
  }

  $("#bn-forecast").onclick = async () => {
    const p = { cmd: "forecast", model: $("#bn-model").value, window: $("#bn-window").value || 0, prec: $("#bn-prec").value };
    const d = await call(p, "Running one forecast");
    if (!d) return;
    out.innerHTML = `<h3>Time per forecast <small>(window ${d.window}, from ${d.start})</small></h3>` + table([
      ["Runs on", "Time"],
      [PREC[d.prec], ms(d.time_ms.npu) + `<br><small>NPU compute ${ms(d.time_ms.npu_compute)}</small>`],
      ["CPU FP32 · QNN", ms(d.time_ms.qnn_cpu)],
      ["CPU FP32 · ONNX Runtime <small>(1 thread)</small>", ms(d.time_ms.ort)],
    ]) + `<p class="hint">NPU output differs from the FP32 model by at most ${d.max_diff_pct.toFixed(2)} % of capacity.</p>`;
    plot(d);
  };

  $("#bn-speed").onclick = async () => {
    const d = await call({ cmd: "speed", prec: $("#bn-prec").value }, "Timing every model");
    if (!d) return;
    $("#bn-chart").innerHTML = "";
    out.innerHTML = `<h3>Time per inference <small>(one window, batch 1)</small></h3>` + table([
      ["Model", PREC[d.prec], "CPU · QNN", "CPU · ONNX Rt"],
      ...d.rows.map((r) => [NAMES[r.model], ms(r.npu) + (r.npu_compute !== "-" ? `<br><small>NPU ${ms(r.npu_compute)}</small>` : ""), ms(r.qnn_cpu), ms(r.ort)]),
    ]) + `<p class="hint">One forecast is needed every 15 minutes, so every option here is fast enough.</p>`;
  };

  $("#bn-accuracy").onclick = async () => {
    const n = $("#bn-n").value;
    const d = await call({ cmd: "accuracy", model: $("#bn-model").value, prec: $("#bn-prec").value, n }, `Checking ${n} test windows`);
    if (!d) return;
    $("#bn-chart").innerHTML = "";
    out.innerHTML = `<h3>Accuracy · ${NAMES[d.model]} <small>(first ${d.windows} test windows)</small></h3>` + table([
      ["Domain", `R² ${PREC[d.prec]}`, "R² FP32"], ...d.rows.map((r) => [r.domain, r.r2_npu, r.r2_fp32]),
    ]) + `<p class="hint">NPU vs FP32 agreement: <b>${d.sqnr_db} dB</b> (above ~40 dB the outputs are effectively identical).
      R² “n/a”: the actual values are constant over these windows (meter offline).</p>`;
  };

  $("#bn-random").onclick = () => {
    const max = $("#bn-model").value.endsWith("_h4") ? 4831 : 4739;
    $("#bn-window").value = Math.floor(Math.random() * max);
  };
})();
