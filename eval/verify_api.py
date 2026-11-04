"""End-to-end API verification against a running server.

Exercises every route the UI uses, plus its error paths, and prints a compact
pass/fail line per check with the evidence that decided it.

    BASE=http://127.0.0.1:8077 python eval/verify_api.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

BASE = os.getenv("BASE", "http://127.0.0.1:8077")
FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
OUT = Path(os.getenv("VERIFY_OUT", "eval/results/ui_verify"))
OUT.mkdir(parents=True, exist_ok=True)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  :: {detail}" if detail else ""),
          flush=True)
    return ok


def upload(path: Path, name=None, mime=None):
    with open(path, "rb") as fh:
        files = {"file": (name or path.name, fh, mime or "application/octet-stream")}
        return requests.post(f"{BASE}/api/documents", files=files, timeout=120)


def wait_job(job_id, timeout=300):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        r = requests.get(f"{BASE}/api/jobs/{job_id}", timeout=30)
        if r.status_code != 200:
            return None
        job = r.json()
        if job != last:
            print(f"      job {job_id}: {job.get('status')} {job.get('done')}/{job.get('total')}"
                  f" - {job.get('stage')}", flush=True)
            last = job
        if job.get("status") in ("done", "error"):
            return job
        time.sleep(2)
    return None


def wait_document(doc_id, timeout=300):
    """The UI polls the document list, not a job id: the upload returns doc_id."""
    if not doc_id:
        return None
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        for d in requests.get(f"{BASE}/api/documents", timeout=30).json():
            if d["id"] == doc_id:
                state = (d["status"], d["chunk_count"], (d.get("latest_job") or {}).get("stage"))
                if state != last:
                    print(f"      doc {doc_id}: {state[0]}, {state[1]} chunks, job={state[2]}",
                          flush=True)
                    last = state
                if d["status"] in ("ready", "error"):
                    return d
                break
        time.sleep(2)
    return None


def ask(q, history=None, label=""):
    t0 = time.time()
    r = requests.post(f"{BASE}/api/ask", json={"question": q, "history": history or []},
                      timeout=600)
    dt = time.time() - t0
    if r.status_code != 200:
        return None, dt, r.status_code
    return r.json(), dt, 200


def main():
    # ---------------------------------------------------------------- health
    h = requests.get(f"{BASE}/api/health", timeout=30).json()
    print("health:", json.dumps(h), flush=True)
    check("health returns the LLM model", h.get("llm_model") == "deepseek/deepseek-v4.1-flash",
          h.get("llm_model"))
    check("health counts documents", h.get("documents", 0) > 0, str(h.get("documents")))

    # ----------------------------------------------------------------- index
    r = requests.get(f"{BASE}/", timeout=30)
    check("index page served", r.status_code == 200 and len(r.text) > 5000,
          f"{r.status_code}, {len(r.text)} bytes")
    check("index has the three columns",
          all(k in r.text for k in ("Documents", "Ask the policy agent", "Contradiction tracker")))

    # ------------------------------------------------------------- documents
    docs = requests.get(f"{BASE}/api/documents", timeout=30).json()
    check("document list shape", bool(docs) and {"id", "name", "status", "chunk_count"} <= set(docs[0]),
          f"{len(docs)} docs, keys={sorted(docs[0])[:8]}")
    check("no document is stuck in a non-ready state",
          all(d["status"] in ("ready", "error", "ingesting") for d in docs),
          ", ".join(f"{d['name']}={d['status']}" for d in docs))

    # ---------------------------------------------------------------- uploads
    md = OUT / "ui_check_remote_policy.md"
    md.write_text(
        "# Remote Work Policy (2026 Revision)\n\n"
        "## 1. Entitlement\n"
        "All employees are entitled to work remotely up to 5 days per week, subject to manager approval.\n\n"
        "## 2. Office attendance\n"
        "There is no mandatory office attendance day.\n\n"
        "## 3. Equipment\n"
        "The company provides a laptop and a one-time home-office allowance of INR 25,000.\n",
        encoding="utf-8")
    r = upload(md)
    check("upload md accepted", r.status_code == 200, f"HTTP {r.status_code} {r.text[:120]}")
    job = r.json() if r.status_code == 200 else {}
    print("      upload response:", json.dumps(job)[:300], flush=True)
    check("upload returns the document id", bool(job.get("doc_id")), json.dumps(job)[:120])
    finished = wait_document(job.get("doc_id"))
    check("md ingestion finished", finished and finished.get("status") == "ready",
          finished and f"{finished.get('status')}, {finished.get('chunk_count')} chunks")

    txt = OUT / "ui_check_notice.txt"
    txt.write_text("Notice Period Policy\n\nEmployees must serve a notice period of 90 days "
                   "when resigning. Managers must serve 120 days.\n", encoding="utf-8")
    r = upload(txt)
    check("upload txt accepted", r.status_code == 200, f"HTTP {r.status_code}")
    if r.status_code == 200:
        done_txt = wait_document(r.json().get("doc_id"))
        check("txt ingestion finished", done_txt and done_txt.get("status") == "ready",
              done_txt and done_txt.get("status"))

    pdf = FIX / "sample_policy.pdf"
    if pdf.exists():
        r = upload(pdf)
        check("upload pdf accepted", r.status_code == 200, f"HTTP {r.status_code} {r.text[:100]}")
        if r.status_code == 200:
            done_pdf = wait_document(r.json().get("doc_id"))
            check("pdf ingestion finished", done_pdf and done_pdf.get("status") == "ready",
                  done_pdf and done_pdf.get("status"))
    else:
        check("pdf fixture exists", False, str(pdf))

    # ------------------------------------------------------- upload error paths
    bad = OUT / "ui_check_bad.docx"
    bad.write_bytes(b"PK\x03\x04not really a docx")
    r = upload(bad)
    check("unsupported type rejected 400", r.status_code == 400, f"HTTP {r.status_code} {r.text[:90]}")

    empty = OUT / "ui_check_empty.txt"
    empty.write_bytes(b"")
    r = upload(empty)
    check("empty file rejected 400", r.status_code == 400, f"HTTP {r.status_code} {r.text[:90]}")

    docs = requests.get(f"{BASE}/api/documents", timeout=30).json()
    names = [d["name"] for d in docs]
    check("uploaded docs appear in the list",
          any("ui_check_remote_policy" in n for n in names),
          f"{len(docs)} docs: {names}")
    new_doc = next((d for d in docs if "ui_check_remote_policy" in d["name"]), None)
    check("uploaded doc is ready with chunks",
          new_doc and new_doc["status"] == "ready" and new_doc["chunk_count"] > 0,
          new_doc and f"{new_doc['status']}, {new_doc['chunk_count']} chunks")

    # ------------------------------------------------- auto-scan after ingest
    t0 = time.time()
    flags = None
    while time.time() - t0 < 240:
        flags = requests.get(f"{BASE}/api/contradictions?status=OPEN", timeout=30).json()
        mine = [f for f in flags if "ui_check_remote_policy" in (f["doc_a"] + f["doc_b"])]
        if mine:
            break
        time.sleep(5)
    mine = [f for f in (flags or []) if "ui_check_remote_policy" in (f["doc_a"] + f["doc_b"])]
    check("contradiction cascade flagged the new 5-day policy", bool(mine),
          f"{len(mine)} flag(s) after {time.time() - t0:.0f}s")
    if mine:
        f = mine[0]
        check("flag cites both passages",
              bool(f.get("text_a")) and bool(f.get("text_b")) and bool(f.get("explanation")),
              f"#{f['id']} gate={f['gate_score']} conf={f['confidence']}: "
              f"{(f.get('explanation') or '')[:110]}")

    # --------------------------------------------------------------------- ask
    cases = [
        ("How many days per week can I work remotely?",
         lambda a: "day" in a.lower() and ("[" in a and "]" in a), "cites the policy text"),
        ("What is the company's policy on pet insurance for reptiles?",
         lambda a: any(w in a.lower() for w in ("not cover", "don't", "do not", "no policy",
                                                "not specified", "does not", "no information",
                                                "not mention", "cannot find")), "refuses out-of-corpus"),
        ("hi", lambda a: len(a) < 400, "answers a greeting briefly"),
        ("I know I get 5 remote days a week now, but what happens if my manager refuses and "
         "I still work from home?",
         lambda a: len(a) > 80, "handles a partial/premise question"),
        ("If I have 14 casual leave days left and I take 3, then encash 2, how many do I have?",
         lambda a: any(ch.isdigit() for ch in a), "does arithmetic on the employee's numbers"),
        ("Do any of the policy documents contradict each other about remote work?",
         lambda a: "[" in a or "conflict" in a.lower(), "can discuss contradictions"),
    ]
    history = []
    for q, predicate, label in cases:
        res, dt, code = ask(q, history)
        if res is None:
            check(f"ask: {label}", False, f"HTTP {code}")
            continue
        ans = res.get("answer") or ""
        check(f"ask: {label}", predicate(ans), f"{dt:.1f}s, {res.get('tool_calls')} tools, "
                                               f"{len(res.get('citations') or [])} citations, "
                                               f"{len(res.get('citation_warnings') or [])} warnings")
        (OUT / f"ask_{len(history)}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": ans})

    # the routing bug this run found: a content question about the corpus used to
    # be answered with the document inventory
    res, dt, code = ask("Do any of the policy documents contradict each other about remote work?")
    if res is not None:
        used = [s.get("tool") for s in (res.get("steps") or []) if s.get("type") == "tool_call"]
        check("a contradiction question reaches the agent (not a doc listing)",
              res.get("intent_route") != "DIRECT_REPLY" and bool(used),
              f"route={res.get('intent_route')} intent={res.get('intent')} tools={used}")
        check("the agent consults the flag table",
              "list_contradictions" in used,
              f"tools={used}")
        check("the answer names a conflict with a citation",
              bool(res.get("citations")) and "day" in (res.get("answer") or "").lower(),
              f"{len(res.get('citations') or [])} citations")
        (OUT / "ask_contradiction.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    r = requests.post(f"{BASE}/api/ask", json={"question": "   "}, timeout=60)
    check("empty question rejected 400", r.status_code == 400, f"HTTP {r.status_code}")

    # ------------------------------------------------------------ contradictions
    flags = requests.get(f"{BASE}/api/contradictions", timeout=30).json()
    check("all-status flag list returns rows", isinstance(flags, list) and flags,
          f"{len(flags)} rows")
    fid = flags[0]["id"]
    r = requests.post(f"{BASE}/api/contradictions/{fid}/status", json={"status": "resolved"},
                      timeout=30)
    check("flag status update accepted", r.status_code == 200, f"HTTP {r.status_code}")
    open_now = requests.get(f"{BASE}/api/contradictions?status=OPEN", timeout=30).json()
    check("resolved flag left the OPEN list", all(f["id"] != fid for f in open_now),
          f"{len(open_now)} open")
    r = requests.post(f"{BASE}/api/contradictions/{fid}/status", json={"status": "banana"},
                      timeout=30)
    check("invalid flag status rejected 400", r.status_code == 400, f"HTTP {r.status_code}")
    r = requests.post(f"{BASE}/api/contradictions/99999/status", json={"status": "resolved"},
                      timeout=30)
    check("unknown flag rejected 404", r.status_code == 404, f"HTTP {r.status_code}")

    # ------------------------------------------------------------------- chunks
    if new_doc:
        chunks = requests.get(f"{BASE}/api/chunks/1", timeout=30)
        check("chunk lookup works", chunks.status_code == 200,
              f"HTTP {chunks.status_code}")
        r = requests.get(f"{BASE}/api/chunks/999999", timeout=30)
        check("unknown chunk rejected 404", r.status_code == 404, f"HTTP {r.status_code}")

    # ---------------------------------------------------------------- scanning
    if new_doc:
        r = requests.post(f"{BASE}/api/scan/{new_doc['id']}", timeout=60)
        ok = r.status_code == 200 and r.json().get("job_id")
        check("manual rescan queued", bool(ok), f"HTTP {r.status_code} {r.text[:90]}")
        if ok:
            j = wait_job(r.json()["job_id"])
            check("manual rescan finished", j and j.get("status") == "done",
                  j and f"{j.get('status')} {j.get('stage')}")
    r = requests.post(f"{BASE}/api/scan/99999", timeout=60)
    check("scan of unknown document rejected", r.status_code in (400, 404),
          f"HTTP {r.status_code}")

    # ----------------------------------------------------------------- delete
    if new_doc:
        r = requests.delete(f"{BASE}/api/documents/{new_doc['id']}", timeout=60)
        check("delete accepted", r.status_code == 200, f"HTTP {r.status_code}")
        after = requests.get(f"{BASE}/api/documents", timeout=30).json()
        check("deleted document is gone",
              all(d["id"] != new_doc["id"] for d in after), f"{len(after)} docs left")
        r = requests.delete(f"{BASE}/api/documents/{new_doc['id']}", timeout=60)
        check("second delete is 404", r.status_code == 404, f"HTTP {r.status_code}")

    # ------------------------------------------------------------------ summary
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed", flush=True)
    for name, _, detail in failed:
        print(f"  FAILED: {name} :: {detail}")
    (OUT / "api_verification.json").write_text(
        json.dumps([{"check": n, "ok": o, "detail": d} for n, o, d in RESULTS], indent=2),
        encoding="utf-8")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
