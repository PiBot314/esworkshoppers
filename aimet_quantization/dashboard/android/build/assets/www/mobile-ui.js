/* Loaded after app.js: touch controls, phone-sized charts, server status and app glue.
 * Uses app.js globals (views, Live, showTab, activeTab, chartOpts, MODELS). */
"use strict";

// Phone-sized axes: smaller font and a narrower y-axis gutter so the plot gets the width.
{
  const base = chartOpts;
  chartOpts = function (el, title, series, extra) {
    const o = base(el, title, series, extra);
    (o.axes || []).forEach((a, k) => {
      a.font = "11px system-ui, sans-serif";
      // power axes: "600 kW" instead of "600.0 kW", in a gutter wide enough for it
      if (k === 1 && a.values && /W$/.test(String(a.values(null, [1000])[0]))) {
        a.values = (u, v) => v.map((x) => Math.abs(x) >= 1e4 ? Math.round(x / 1e3) + " kW" : Math.abs(x) >= 1e3 ? +(x / 1e3).toFixed(1) + " kW" : Math.round(x) + " W");
        a.size = 58;
      } else if (a.size >= 64) a.size = 62;
    });
    return o;
  };
}

function checkServer() {
  const dev = $("#device");
  fetch("api/status").then((r) => r.json()).then((s) => {
    dev.className = "device on";
    dev.innerHTML = `<b>● NPU live</b> ${s.device.trim()} · ${s.soc.trim()}`;
  }).catch(() => {
    dev.className = "device off";
    dev.innerHTML = `<span class="bad">● NPU server offline</span> — replay works offline · tap to set address`;
  });
}
window.__nzServerChanged = checkServer;
window.__nzBack = () => { if (activeTab === "replay") return false; showTab("replay"); scrollTo(0, 0); return true; };

$("#settings").onclick = $("#device").onclick = () => window.NZ ? NZ.openSettings() : checkServer();
$$("#tabs button").forEach((b) => b.addEventListener("click", () => scrollTo(0, 0)));

(function whenReady() {
  if (!views.dayahead || !$("#hood-table").innerHTML) return setTimeout(whenReady, 50);
  for (const [id, v] of [["#tab-replay", views.replay], ["#tab-plant", views.plant]]) {
    const s = $(id);
    $(".back", s).onclick = () => v.step(-1);
    $(".fwd", s).onclick = () => v.step(1);
    $(".next-anom", s).onclick = () => v.nextAnomaly();
  }
  $$(".da-step").forEach((b) => (b.onclick = () => views.dayahead.step(+b.dataset.k)));
  $("#app-info").innerHTML = window.NZ
    ? `Running on ${NZ.appDevice()}.<br>Replay, plant, day-ahead and pipeline views use the NPU outputs recorded on the QIDK and are bundled in the app. “Run on NPU” sends the window to <code>${NZ.getServer()}</code> (tap ⚙ to change).`
    : "Browser build.";
  checkServer();
  setInterval(checkServer, 10000);
  $("#splash").classList.add("gone");
  setTimeout(() => $("#splash").remove(), 400);
  requestAnimationFrame(resizeAll);
})();
