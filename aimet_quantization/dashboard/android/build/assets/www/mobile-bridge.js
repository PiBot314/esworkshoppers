/* Loaded before app.js. Inside the Android app (window.NZ present):
 *  - fetch("api/...") goes through the native bridge to the nz_dashboard server on the QIDK
 *    (a page served from the app's https origin cannot call a plain-http server directly);
 *  - <a download href="blob:..."> clicks save the file to Downloads.
 * In a normal browser nothing changes, so the page can also be opened from the server. */
"use strict";
(function () {
  if (!window.NZ) return;
  const realFetch = window.fetch.bind(window), pending = new Map();
  let seq = 0;

  window.__nzDone = (id, status, text) => {
    const p = pending.get(id);
    if (!p) return;
    pending.delete(id);
    if (status >= 200 && status <= 599) p.resolve(new Response(text, { status, headers: { "Content-Type": "application/json" } }));
    else p.reject(new Error(text));
  };

  function toBase64(buf) {
    const bytes = new Uint8Array(buf);
    let s = "";
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  }

  window.fetch = (url, opts = {}) => {
    const u = String(url);
    if (!u.startsWith("api/")) return realFetch(url, opts);
    return new Promise((resolve, reject) => {
      const id = ++seq;
      pending.set(id, { resolve, reject });
      NZ.request(id, opts.method || "GET", u, opts.body ? toBase64(opts.body) : "");
    });
  };

  const realClick = HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click = function () {
    if (this.download && this.href.startsWith("blob:")) {
      const name = this.download;
      realFetch(this.href).then((r) => r.text()).then((t) => NZ.saveFile(name, t));
      return;
    }
    return realClick.call(this);
  };
})();
