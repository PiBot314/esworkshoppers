"""Stand-in for the campus API and nz_dashboard, for replaying the live loop fast (plan test A2).

Serves the cached raw/ files as if they were live, up to a fake clock the test sets:
  GET  /synthetic/status                  -> {"replay_now": <fake clock>}
  GET  /verticals/all/temporal?from&to&node_id   (rows with created_at in [from, to] and <= the fake clock)
  POST /_clock?t=<unix seconds>           (test only: set the fake clock)
and, like nz_dashboard but with the FP32 ONNX models on the CPU (the ones verify.py uses):
  POST /api/run?model=<m>&variant=<v>     float32 [1,96,F] -> {"ok":true,"out":[...],"execute_us":-1}
  GET  /data/<model>/meta.json

  ../../project/.venv/bin/python replay_server.py <final> [port=9998]
"""
import bisect, glob, json, os, sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import numpy as np, onnxruntime as ort

FINAL = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9998
HERE = os.path.dirname(os.path.abspath(__file__))
WWW = os.path.join(FINAL, "aimet_quantization/dashboard/www/data")
ONNX = {"all_sensors_1h": "all_sensors_lstm_qidk.onnx", "single_node": "SL-PL00-00_lstm_qidk.onnx"}
sessions = {m: ort.InferenceSession(os.path.join(FINAL, "model2/qidk_deployment", f), providers=["CPUExecutionProvider"])
            for m, f in ONNX.items()}
# switchable models (nz_live/models/<id>/meta.json, plan_model_switcher.md): served under their registry id
for mp in glob.glob(os.path.join(FINAL, "nz_live", "models", "*", "meta.json")):
    src = json.load(open(mp)).get("source_onnx")
    if src:
        sessions[os.path.basename(os.path.dirname(mp))] = ort.InferenceSession(os.path.join(FINAL, src), providers=["CPUExecutionProvider"])

ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
rows = {}
for f in sorted(glob.glob(os.path.join(HERE, "raw", "*.json"))):
    node = os.path.basename(f).rsplit("_", 1)[0]
    rows.setdefault(node, []).extend(json.load(open(f)))
times = {}
for node, rs in rows.items():
    rs.sort(key=lambda r: ts(r["created_at"]))
    times[node] = [ts(r["created_at"]) for r in rs]
clock = [0.0]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/synthetic/status":
            return self.send({"replay_now": datetime.fromtimestamp(clock[0], timezone.utc).isoformat()})
        if u.path == "/verticals/all/temporal":
            node, t = q["node_id"], times.get(q["node_id"], [])
            lo, hi = ts(q["from"]), min(ts(q["to"]), clock[0])
            i, j = bisect.bisect_left(t, lo), bisect.bisect_right(t, hi)
            return self.send({node: {"data": rows.get(node, [])[i:j]}})
        if u.path.startswith("/data/") and u.path.endswith("/meta.json"):
            return self.send(json.load(open(os.path.join(WWW, u.path.split("/")[2], "meta.json"))))
        self.send({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        if u.path == "/_clock":
            clock[0] = float(q["t"])
            return self.send({"ok": True})
        if u.path == "/api/run":
            s = sessions[q["model"]]
            x = np.frombuffer(body, np.float32).reshape(1, 96, -1)
            out = s.run(None, {s.get_inputs()[0].name: x})[0].ravel()
            return self.send({"ok": True, "out": out.tolist(), "execute_us": -1})
        self.send({"error": "not found"}, 404)


print(f"replay server on :{PORT}", flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
