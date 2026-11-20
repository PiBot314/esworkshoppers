"""Test A7 (short version): ask the board's assistant live and existing questions, record the tools it calls,
its answer, and the time taken. Talks to the dashboard over the hotspot, like a phone would.
  python3 slm_questions.py http://10.42.0.237:5173 > ../runs/a7.jsonl"""
import json, sys, time, urllib.request
BASE = sys.argv[1]
Q = [  # (question, tool the plan expects first)
    ("Is anything unusual right now?", "live_status|recent_anomalies"),
    ("What is the campus grid draw right now, and what is forecast for the next hour?", "live_status"),
    ("How much is the Palash solar plant producing right now?", "live_status"),
    ("Were there any anomalies in the last few hours?", "recent_anomalies"),
    ("Explain the most recent anomaly.", "explain_anomaly|recent_anomalies"),
    ("Is the latest flag a real fault?", "recent_anomalies|explain_anomaly"),
    ("Are the forecasts running on the NPU?", "live_status"),
    ("Can you forecast campus solar for the next hour?", "live_status"),
    ("Is the campus net zero?", "net_zero"),
    ("When does the campus use the most power?", "get_stats|search_nodes|data_overview|compare_nodes"),
]
for q, want in Q:
    t0 = time.time()
    req = urllib.request.Request(f"{BASE}/api/chat", json.dumps({"messages": [{"role": "user", "content": q}], "context": {"page": "/"}}).encode(),
                                 {"content-type": "application/json"})
    tools, text, err = [], "", None
    try:
        with urllib.request.urlopen(req, timeout=400) as r:
            for line in r:
                if not line.strip():
                    continue
                o = json.loads(line)
                if o["type"] == "tool": tools.append(o["name"])
                elif o["type"] == "text": text += o["text"]
                elif o["type"] == "error": err = o["message"]
    except Exception as e:
        err = str(e)
    ok = bool(tools) and tools[0] in want.split("|")
    print(json.dumps({"q": q, "want": want, "tools": tools, "right_tool": ok, "s": round(time.time() - t0, 1), "answer": text.strip(), "error": err}), flush=True)
