#!/usr/bin/env python3
"""Policy RAG Agent - command line interface.

Usage examples:
    python run.py upload sample_docs/remote_work_policy_v1.txt
    python run.py list
    python run.py ask "How many days per week can I work remotely?"
    python run.py chat
    python run.py scan --doc remote_work_policy_v2
    python run.py contradictions
    python run.py resolve 1 resolved
    python run.py serve
    python run.py selftest
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .admin import AdminAPI
from .config import get_config


def _api() -> AdminAPI:
    return AdminAPI()


def _print_progress(p: dict) -> None:
    total = max(1, p.get("total") or 1)
    done = p.get("done") or 0
    pct = int(100 * done / total)
    bar = "#" * (pct // 5)
    sys.stdout.write(
        f"\r  [{bar:<20}] {pct:3d}%  {p.get('stage', ''):<40}"
    )
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n")


# ------------------------------------------------------------------ commands

def cmd_upload(args) -> int:
    api = _api()
    from pathlib import Path
    rc = 0
    for p in args.paths:
        path = Path(p)
        print(f"Uploading {path.name} ...")
        try:
            result = api.upload_document(path, name=args.name if len(args.paths) == 1 else None,
                                         on_progress=_print_progress)
        except Exception as e:
            print(f"  ERROR: {e}")
            rc = 1
            continue
        print(f"  ingested as '{api.db.get_document(result['doc_id'])['name']}' "
              f"({result['windows']} windows, {result['chunks']} chunks)")
        for w in result.get("warnings", [])[:5]:
            print(f"  warning: {w}")
        print("  contradiction scan started in background "
              "(see `python run.py contradictions`)")
    return rc


def cmd_list(args) -> int:
    api = _api()
    docs = api.list_documents()
    if not docs:
        print("No documents ingested yet. Try: python run.py upload <file>")
        return 0
    print(f"{len(docs)} document(s):\n")
    for d in docs:
        job = d.get("latest_job") or {}
        prog = ""
        if job and job.get("status") == "running":
            prog = f" | {job.get('kind')}: {job.get('done', 0)}/{job.get('total', 0)} {job.get('stage', '')}"
        print(f"  [{d['id']}] {d['name']}  ({d.get('format') or '?'}, "
              f"{d.get('chunk_count', 0)} chunks, {d.get('status')}){prog}")
        if d.get("open_contradictions"):
            print(f"      !! {d['open_contradictions']} OPEN contradiction flag(s)")
        s = (d.get("summary") or "").strip()
        if s:
            print(f"      summary: {s[:180]}")
    return 0


def cmd_delete(args) -> int:
    api = _api()
    ok = api.delete_document(args.doc)
    print("deleted." if ok else f"not found: {args.doc}")
    return 0 if ok else 1


def _print_result(result, trace: bool) -> None:
    if result.error:
        print(f"\n[agent error] {result.error}\n")
    print(result.answer or "(no answer)")
    if result.citations:
        print("\nCitations:")
        for c in result.citations:
            print(f"  - {c}")
    if result.citation_warnings:
        print("\nCitation warnings (verify these!):")
        for w in result.citation_warnings:
            print(f"  !! {w}")
    if trace:
        print(f"\n--- ReAct trace ({result.tool_calls} tool call(s), "
              f"{result.elapsed_s:.1f}s) ---")
        for i, s in enumerate(result.steps, 1):
            t = s["type"]
            if t == "thought":
                print(f"  {i}. THOUGHT: {s['content'][:200]}")
            elif t == "tool_call":
                print(f"  {i}. TOOL {s['tool']}({s['arguments']})")
                if s.get("schema_repairs"):
                    print(f"     [schema repairs applied: {', '.join(s['schema_repairs'])}]")
                print(f"     -> {s['observation'][:160]}")
            elif t == "tool_error":
                print(f"  {i}. TOOL_ERROR {s['tool']}: {s['problems']}")
            else:
                print(f"  {i}. FINAL: {s['content'][:200]}")
    print()


def cmd_ask(args) -> int:
    api = _api()
    cfg = api.cfg
    # one line so a reviewer can see which stack answered the question
    print(f"[retrieval={cfg.retrieval_mode} intent={'on' if cfg.intent_enabled else 'off'}"
          f" prompts={cfg.prompt_intent}/{cfg.prompt_agent}/{cfg.prompt_judge}"
          f" judge={'on' if cfg.verify_answers else 'off'}]\n")
    print("Thinking (ReAct loop) ...\n")
    result = api.ask(args.question)
    _print_result(result, trace=args.trace)
    return 0 if result.answer else 1


def cmd_chat(args) -> int:
    api = _api()
    print("Policy QA chat (type 'exit' to quit, '/trace' to toggle trace, "
          "/docs to list documents)\n")
    history: List[dict] = []
    trace = False
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        if q.lower() == "/trace":
            trace = not trace
            print(f"trace = {trace}\n")
            continue
        if q.lower() == "/docs":
            cmd_list(args)
            continue
        print("\nagent> thinking ...")
        result = api.ask(q, history=history[-10:])
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": result.answer})
        print()
        _print_result(result, trace=trace)
    return 0


def cmd_scan(args) -> int:
    api = _api()
    if args.doc:
        result = api.scan(args.doc, blocking=True)
        print(f"scan finished: {result.get('flagged', 0)} contradiction(s) flagged "
              f"(job #{result.get('job_id')})")
    else:
        results = api.scan(blocking=True)
        total = sum(r.get("flagged", 0) for r in results)
        print(f"scanned {len(results)} document(s), {total} contradiction(s) flagged")
    return 0


def cmd_contradictions(args) -> int:
    api = _api()
    flags = api.list_contradictions(args.status)
    if not flags:
        print(f"No {args.status or ''} contradictions.".strip())
        return 0
    for f in flags:
        ver = "LLM-verified" if f["verified"] else "UNVERIFIED (heuristic only)"
        print(f"#{f['id']} [{f['status']}] {ver}  (gate score {f['gate_score']:.2f}, "
              f"confidence {f['confidence']:.2f}, type: {f['conflict_type'] or '-'})")
        print(f"   A: {f['doc_a']} § {f['section_a'] or '-'}  (chunk {f['chunk_a_id']})")
        print(f"      \"{(f['text_a'] or '')[:220]}...\"")
        print(f"   B: {f['doc_b']} § {f['section_b'] or '-'}  (chunk {f['chunk_b_id']})")
        print(f"      \"{(f['text_b'] or '')[:220]}...\"")
        print(f"   why: {f['explanation']}")
        for p in (f.get("points") or [])[:4]:
            print(f"     - topic: {p.get('topic', '')}")
            print(f"       A says: {p.get('claim_a', '')[:160]}")
            print(f"       B says: {p.get('claim_b', '')[:160]}")
        print(f"   gate signals: {'; '.join(f.get('gate_signals') or [])[:300]}")
        print(f"   detected: {f['detected_at']}   resolve: python run.py resolve {f['id']} resolved|dismissed")
        print()
    return 0


def cmd_resolve(args) -> int:
    api = _api()
    ok = api.set_contradiction_status(args.id, args.status)
    print("updated." if ok else f"not found: {args.id}")
    return 0 if ok else 1


def cmd_status(args) -> int:
    api = _api()
    job = api.job_status(args.job_id)
    if not job:
        print(f"no job {args.job_id}")
        return 1
    print(f"job #{job['id']} [{job['kind']}] doc={job['doc_name']} "
          f"status={job['status']} progress={job['done']}/{job['total']}")
    print(f"  stage: {job['stage']}")
    if job.get("error"):
        print(f"  error: {job['error']}")
    return 0


def cmd_serve(args) -> int:
    cfg = get_config()
    host = args.host or cfg.app_host
    port = args.port or cfg.app_port
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required for the web UI: pip install uvicorn fastapi")
        return 1
    from .main import app  # noqa: F401  (imports fastapi lazily)
    print(f"Starting web UI on http://{host}:{port} ...")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


def cmd_selftest(args) -> int:
    import unittest
    from pathlib import Path
    tests_dir = Path(__file__).resolve().parent.parent / "tests"
    suite = unittest.defaultTestLoader.discover(str(tests_dir))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


# ---------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("upload", help="upload + ingest policy docs (txt/md/pdf)")
    s.add_argument("paths", nargs="+")
    s.add_argument("--name", help="document display name (single file only)")
    s.set_defaults(fn=cmd_upload)

    s = sub.add_parser("list", help="list ingested documents")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("delete", help="delete a document (name or id)")
    s.add_argument("doc")
    s.set_defaults(fn=cmd_delete)

    s = sub.add_parser("ask", help="ask a one-shot question")
    s.add_argument("question")
    s.add_argument("--trace", action="store_true", help="show the ReAct trace")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("chat", help="interactive chat session")
    s.set_defaults(fn=cmd_chat)

    s = sub.add_parser("scan", help="run the contradiction cascade")
    s.add_argument("--doc", help="document name or id (default: all ready docs)")
    s.set_defaults(fn=cmd_scan)

    s = sub.add_parser("contradictions", help="list contradiction flags")
    s.add_argument("--status", default=None,
                   choices=["OPEN", "resolved", "dismissed"])
    s.set_defaults(fn=cmd_contradictions)

    s = sub.add_parser("resolve", help="set a contradiction flag status")
    s.add_argument("id", type=int)
    s.add_argument("status", choices=["resolved", "dismissed", "OPEN"])
    s.set_defaults(fn=cmd_resolve)

    s = sub.add_parser("status", help="show a background job's progress")
    s.add_argument("job_id", type=int)
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("serve", help="start the web admin UI")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("selftest", help="run the offline unit tests")
    s.set_defaults(fn=cmd_selftest)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
