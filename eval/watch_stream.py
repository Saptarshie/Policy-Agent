"""Read /api/ask/stream frame by frame, printing when each one arrives."""
import json
import sys
import time

import requests

BASE = "http://127.0.0.1:8000"
q = sys.argv[1] if len(sys.argv) > 1 else "How many days per week can I work remotely?"

t0 = time.time()
with requests.post(f"{BASE}/api/ask/stream",
                   json={"question": q, "history": []},
                   stream=True, timeout=600) as r:
    print(f"HTTP {r.status_code} {r.headers.get('content-type')}  ({time.time()-t0:.1f}s)")
    event, data = "message", ""
    for line in r.iter_lines(decode_unicode=True):
        if line is None:
            continue
        if line == "":
            if event != "message" or data:
                stamp = f"+{time.time()-t0:6.1f}s"
                if event == "step":
                    d = json.loads(data)
                    label = d.get("type")
                    if label == "tool_call":
                        label += f" {d.get('tool')}({json.dumps(d.get('arguments'))[:60]})"
                    elif label in ("thought", "final"):
                        label += " " + (d.get("content") or "")[:60].replace("\n", " ")
                    print(f"{stamp}  step   {label}")
                elif event == "final":
                    d = json.loads(data)
                    print(f"{stamp}  FINAL  {d.get('tool_calls')} tools, "
                          f"{d.get('elapsed_s')}s, {len(d.get('citations') or [])} citations, "
                          f"warnings={d.get('citation_warnings')}")
                else:
                    d = json.loads(data) if data else {}
                    print(f"{stamp}  {event:6} {json.dumps(d)[:90]}")
            event, data = "message", ""
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event: "):
            event = line[7:].strip()
        elif line.startswith("data: "):
            data += line[6:]
print(f"stream closed at +{time.time()-t0:.1f}s")
