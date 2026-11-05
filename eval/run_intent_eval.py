#!/usr/bin/env python3
"""Intent-layer ablation: regex baseline vs LLM triage prompt variants.

    python eval/run_intent_eval.py --variant legacy,v1,v2
    python eval/run_intent_eval.py --variant v1 --verbose

Scored per case:
  * intent  - category match (``accept`` lists cover genuinely ambiguous inputs)
  * route   - REACT_LOOP | DIRECT_REPLY | FAST_REFUSE match
  * safety  - a workplace question must never be FAST_REFUSEd (that would be a
              coverage outage), and non-HR/injection input must be refused
  * latency - classifier wall time

Results are appended to a JSONL file keyed by (variant, case id), so an
interrupted run resumes and repeated questions cost nothing (the router's
disk cache).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.intent import IntentRouter  # noqa: E402
from app.agent.intent_legacy import classify_intent_legacy  # noqa: E402
from app.config import get_config  # noqa: E402
from app.llm import OpenAICompatClient  # noqa: E402
from app.prompts import INTENT_SYSTEM  # noqa: E402

CASES = ROOT / "eval" / "intent_cases.jsonl"
HARD_CASES = ROOT / "eval" / "intent_hard.jsonl"
RESULTS = ROOT / "eval" / "results" / "intent_runs.jsonl"

VARIANTS = {
    "legacy": None,          # regex baseline, no prompt
    "v1": INTENT_SYSTEM,
    "v2": None,              # filled in from prompts.INTENT_SYSTEM_V2 when present
    "v3": None,              # v1 + the conditions list (prompts.INTENT_SYSTEM_V3)
}


def load_cases(hard: bool = False) -> list:
    files = [CASES] + ([HARD_CASES] if hard else [])
    out = []
    for path in files:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(json.loads(line))
    return out


def load_results() -> dict:
    done = {}
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            done[(row["variant"], row["id"])] = row
    return done


def append_result(row: dict) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_router(variant: str, cfg):
    if variant == "legacy":
        return None
    if variant in ("v2", "v3"):
        import app.prompts as prompts
        prompt = getattr(prompts, f"INTENT_SYSTEM_{variant.upper()}", None)
        if prompt is None:
            raise SystemExit(f"INTENT_SYSTEM_{variant.upper()} is not defined yet")
    else:
        prompt = VARIANTS[variant]
    llm = OpenAICompatClient(
        base_url=cfg.llm_base_url, api_key=cfg.llm_api_key, model=cfg.llm_model,
        max_output_tokens=cfg.intent_max_tokens, temperature=0.0,
    )
    return IntentRouter(
        llm=llm, model=cfg.intent_model or cfg.llm_model,
        cache_path=str(ROOT / "eval" / "results" / f"intent_cache_{variant.replace('+', '_')}.json"),
        prompt=prompt, prompt_version=variant,
        max_tokens=cfg.intent_max_tokens,
    )


def score_case(case: dict, plan) -> dict:
    from app.agent.intent import CATEGORY_TO_ROUTE
    accept = case.get("accept") or []
    intent_ok = plan.intent in accept
    if case.get("routes"):
        routes = set(case["routes"])          # explicitly mixed-route case
    else:
        routes = {CATEGORY_TO_ROUTE[a] for a in accept}
    route_ok = plan.route in routes
    workplace = bool({"POLICY_DIRECT_LOOKUP", "IMPLICIT_SITUATION_OR_GRIEVANCE",
                      "MULTI_HOP_COMPARISON", "SENSITIVE_HR_ESCALATION"} & set(accept))
    false_refusal = bool(workplace and plan.route == "FAST_REFUSE"
                         and "FAST_REFUSE" not in routes)
    missed_refusal = bool(
        ({"CHITCHAT_OR_OUT_OF_SCOPE", "ADVERSARIAL_OR_MALICIOUS"} & set(accept))
        and plan.route != "FAST_REFUSE" and "FAST_REFUSE" in routes)
    return {
        "intent_ok": intent_ok,
        "route_ok": route_ok,
        "false_refusal": false_refusal,
        "missed_refusal": missed_refusal,
    }


def report(variant: str, rows: list, total_cases: int) -> None:
    n = len(rows)
    if not n:
        print(f"{variant}: no rows")
        return
    acc = sum(r["intent_ok"] for r in rows) / n
    route = sum(r["route_ok"] for r in rows) / n
    fr = sum(r["false_refusal"] for r in rows)
    mr = sum(r["missed_refusal"] for r in rows)
    lat = [r["latency_s"] for r in rows if r["latency_s"]]
    p50 = sorted(lat)[len(lat) // 2] if lat else 0.0
    fallbacks = sum(1 for r in rows if r.get("decided_by") == "fallback")
    cache_hits = sum(1 for r in rows if r.get("decided_by") == "cache")

    per_cat = defaultdict(lambda: [0, 0])
    for r in rows:
        gold = r["accept"][0]
        per_cat[gold][1] += 1
        per_cat[gold][0] += int(r["intent_ok"])

    print(f"\n=== {variant}  ({n}/{total_cases} cases) ===")
    print(f"  intent accuracy : {acc:5.1%}")
    print(f"  route accuracy  : {route:5.1%}   <- the metric that changes behaviour")
    print(f"  false refusals  : {fr}   (workplace question refused before retrieval = outage)")
    print(f"  missed refusals : {mr}   (chit-chat/injection sent to the agent)")
    print(f"  classifier p50  : {p50:.2f}s   (n={len(lat)})")
    print(f"  fail-open       : {fallbacks}   cache hits: {cache_hits}")

    # requires_table_check calibration (only on labelled cases)
    labelled = [r for r in rows if r.get("table") is not None]
    if labelled:
        tp = sum(1 for r in labelled if r["table"] and r["requires_table_check"])
        fp = sum(1 for r in labelled if not r["table"] and r["requires_table_check"])
        fn = sum(1 for r in labelled if r["table"] and not r["requires_table_check"])
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        print(f"  table flag      : P={prec:.0%} R={rec:.0%} "
              f"(tp={tp} fp={fp} fn={fn} of {len(labelled)} labelled)")
    print("  per category:")
    for cat, (ok, tot) in sorted(per_cat.items()):
        print(f"    {cat:34} {ok}/{tot}")
    wrong = [r for r in rows if not r["intent_ok"]]
    if wrong:
        print("  misclassified:")
        for r in wrong:
            print(f"    - {r['id']}: got {r['intent']} want {r['accept']} :: {r['text'][:70]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="legacy,v1",
                    help="comma list: legacy,v1,v2")
    ap.add_argument("--hard", action="store_true",
                    help="include the adversarial/ambiguous hard set")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--briefs-from", default="",
                    help="DATA_DIR whose indexed documents supply the CORPUS block "
                         "(one brief per ready document). Rows are recorded as "
                         "<variant>+corpus so they never mix with the baseline.")
    args = ap.parse_args()

    cfg = get_config()
    if not cfg.llm_configured:
        print("LLM not configured; set LLM_BASE_URL/LLM_MODEL in .env")
        return 2

    cases = load_cases(hard=args.hard)
    if args.limit:
        cases = cases[: args.limit]
    done = load_results()

    briefs = None
    if args.briefs_from:
        from app.db import Database
        db = Database(Path(args.briefs_from) / "policy_rag.db")
        briefs = [{"name": d["name"], "brief": (d.get("summary") or "").strip()}
                  for d in db.list_documents() if d.get("status") == "ready"]
        db.close()
        print(f"corpus block: {len(briefs)} document(s) from {args.briefs_from}")

    for variant in [v.strip() for v in args.variant.split(",") if v.strip()]:
        label = variant + ("+corpus" if briefs is not None else "")
        router = build_router(variant, cfg)
        if briefs is not None:
            # the cache key already carries a corpus fingerprint, so one cache file
            # per prompt variant is safe: verdicts with and without briefs coexist
            router.prompt_version = label
        rows = []
        for case in cases:
            prev = done.get((label, case["id"]))
            if prev is not None:
                rows.append(prev)
                continue
            t0 = time.time()
            if variant == "legacy":
                plan = classify_intent_legacy(case["text"])
            else:
                plan = router.classify(case["text"], corpus=briefs)
            latency = time.time() - t0
            row = {
                "variant": label, "id": case["id"], "text": case["text"],
                "accept": case["accept"], "intent": plan.intent,
                "route": plan.route, "de_noised_question": plan.de_noised_question,
                "search_queries": plan.search_queries,
                "requires_table_check": plan.requires_table_check,
                "table": case.get("table"),
                "refusal_reason": plan.refusal_reason,
                "decided_by": plan.decided_by, "notes": plan.notes,
                "latency_s": round(latency, 2),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            row.update(score_case(case, plan))
            append_result(row)
            rows.append(row)
            if args.verbose:
                mark = "ok " if row["intent_ok"] else "BAD"
                print(f"  [{mark}] {case['id']:4} {plan.intent:32} "
                      f"{plan.route:13} {latency:4.1f}s  {case['text'][:60]}")
        report(label, rows, len(cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
