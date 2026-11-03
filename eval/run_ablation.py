#!/usr/bin/env python3
"""End-to-end ablation: does the triage layer + prompt v2 actually help?

Variants (each is the SAME corpus, same retriever, same agent loop):

    A  regex intent  + agent prompt v1 + judge v1        (the original system)
    B  LLM intent    + agent prompt v1 + judge v1        (isolates the triage layer)
    C  LLM intent    + agent prompt v2 + judge v2        (isolates the prompt work)
    D  LLM intent    + agent prompt v2, judge OFF        (cost/latency of verification)

    python eval/run_ablation.py --variant A,B,C --limit 0
    python eval/run_ablation.py --variant D --only e01,q08

Scored per case: refusal correctness, required answer pattern, citation
presence/validity, faithfulness verdict, latency, tool calls. Results are
appended to eval/results/ablation_runs.jsonl keyed by (variant, case id) so a
timeout costs only the unfinished cases.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.admin import AdminAPI  # noqa: E402
from app.agent.intent_legacy import classify_intent_legacy  # noqa: E402
from app.config import get_config  # noqa: E402
from app.prompts import (  # noqa: E402
    AGENT_SYSTEM, AGENT_SYSTEM_V2, FAITHFULNESS_SYSTEM, FAITHFULNESS_SYSTEM_V2,
    INTENT_SYSTEM,
)

RESULTS = ROOT / "eval" / "results" / "ablation_runs.jsonl"

VARIANTS = {
    "A": dict(intent="legacy", agent=AGENT_SYSTEM, judge=FAITHFULNESS_SYSTEM, verify=True),
    "B": dict(intent="llm", agent=AGENT_SYSTEM, judge=FAITHFULNESS_SYSTEM, verify=True),
    "C": dict(intent="llm", agent=AGENT_SYSTEM_V2, judge=FAITHFULNESS_SYSTEM_V2, verify=True),
    "D": dict(intent="llm", agent=AGENT_SYSTEM_V2, judge=FAITHFULNESS_SYSTEM_V2, verify=False),
}


class LegacyRouter:
    """Adapter so the regex baseline can be driven through the same pipeline."""

    def classify(self, question: str):
        return classify_intent_legacy(question)


def load_cases() -> list:
    cases = []
    for name in ("eval_set.jsonl", "qa_extra.jsonl"):
        path = ROOT / "eval" / name
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                cases.append(json.loads(line))
    return cases


def _target() -> Path:
    """Where rows are written and resumed from (per-run file in parallel runs)."""
    override = os.environ.get("ABLATION_OUT")
    return Path(override) if override else RESULTS


def load_done() -> dict:
    done = {}
    target = _target()
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                done[(row["variant"], row["id"])] = row
    return done


def build_api(cfg, variant: str, router_cache: Path) -> AdminAPI:
    spec = VARIANTS[variant]
    if spec["intent"] == "legacy":
        router = LegacyRouter()
    else:
        from app.llm import OpenAICompatClient
        from app.agent.intent import IntentRouter
        llm = OpenAICompatClient(
            base_url=cfg.llm_base_url, api_key=cfg.llm_api_key, model=cfg.llm_model,
            max_output_tokens=cfg.intent_max_tokens, temperature=0.0)
        router = IntentRouter(
            llm=llm, model=cfg.intent_model or cfg.llm_model,
            cache_path=str(router_cache), prompt=INTENT_SYSTEM,
            prompt_version="v1", max_tokens=cfg.intent_max_tokens)

    # a per-variant config so verification can be toggled without mutating the
    # process-wide cached config
    from dataclasses import replace
    cfg_v = replace(cfg, verify_answers=spec["verify"])
    return AdminAPI(config=cfg_v, intent_router=router,
                    agent_prompt=spec["agent"], judge_prompt=spec["judge"])


# strict: the answer IS a refusal. Keep this narrow - the escalation note for
# sensitive cases also says "contact your HR business partner" and must not be
# mistaken for a refusal.
REFUSAL_RE = re.compile(
    r"(?i)ingested policy documents do not cover|"
    r"don'?t have enough information|do not have enough information")


def score(case: dict, result) -> dict:
    answer = result.answer or ""
    is_refusal = bool(REFUSAL_RE.search(answer)) and not result.citations
    if case.get("expect_refusal"):
        pattern_ok = bool(REFUSAL_RE.search(answer))
    else:
        rx = case.get("expect_answer_regex")
        pattern_ok = bool(re.search(rx, answer, re.I)) if rx else True
        if pattern_ok and is_refusal and not case.get("allow_refusal"):
            pattern_ok = False        # a refusal cannot satisfy a factual case
    doc_ok = True
    hint = (case.get("expect_doc") or "").lower()
    if hint and result.citations:
        doc_ok = hint in " ".join(result.citations).lower()
    return {
        "pattern_ok": pattern_ok,
        "doc_ok": doc_ok,
        "refused": is_refusal,
        "citation_warnings": len(result.citation_warnings),
        "faithful": result.faithful,
        "error": result.error or "",
    }


def report(variant: str, rows: list) -> None:
    n = len(rows)
    if not n:
        return
    pass_rate = sum(1 for r in rows if r["pattern_ok"] and r["doc_ok"] and not r["error"]) / n
    cite = sum(1 for r in rows if r["citation_warnings"] == 0) / n
    lat = sorted(r["elapsed_s"] for r in rows)
    p50 = lat[len(lat) // 2]
    tools = sum(r["tool_calls"] for r in rows) / n
    unfaithful = sum(1 for r in rows if r["faithful"] is False)
    errors = sum(1 for r in rows if r["error"])
    print(f"\n=== variant {variant} ({n} cases) ===")
    print(f"  pass (answer + citation) : {pass_rate:5.1%}")
    print(f"  clean citations          : {cite:5.1%}")
    print(f"  latency p50              : {p50:5.1f}s   mean tools {tools:.1f}")
    print(f"  faithfulness FAILED      : {unfaithful}")
    print(f"  agent errors             : {errors}")
    by_type = defaultdict(lambda: [0, 0])
    for r in rows:
        t = r.get("type") or "extra"
        by_type[t][1] += 1
        by_type[t][0] += int(r["pattern_ok"] and r["doc_ok"] and not r["error"])
    print("  by type:")
    for t, (ok, tot) in sorted(by_type.items()):
        print(f"    {t:10} {ok}/{tot}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A,B,C")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma list of case ids")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="", help="append rows here (default: network)")
    args = ap.parse_args()

    cfg = get_config()
    if not cfg.llm_configured:
        print("LLM not configured; set LLM_BASE_URL/LLM_MODEL in .env")
        return 2

    cases = load_cases()
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in wanted]
    elif args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("no cases selected")
        return 2

    done = load_done()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    for variant in [v.strip() for v in args.variant.split(",") if v.strip()]:
        if variant not in VARIANTS:
            raise SystemExit(f"unknown variant {variant}")
        api = build_api(cfg, variant, RESULTS.parent / "intent_cache_ablation.json")
        rows = []
        for case in cases:
            prev = done.get((variant, case["id"]))
            if prev is not None:
                rows.append(prev)
                continue
            t0 = time.time()
            try:
                result = api.ask(case["question"])
                row = {
                    "variant": variant, "id": case["id"], "type": case.get("type", "extra"),
                    "question": case["question"], "answer": (result.answer or "")[:1500],
                    "citations": result.citations,
                    "intent": result.intent, "intent_route": result.intent_route,
                    "de_noised_question": (result.intent_plan or {}).get("de_noised_question", ""),
                    "tool_calls": result.tool_calls,
                    "elapsed_s": round(time.time() - t0, 1),
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                row.update(score(case, result))
            except Exception as e:                    # never lose the whole run
                row = {
                    "variant": variant, "id": case["id"], "type": case.get("type", "extra"),
                    "question": case["question"], "answer": "",
                    "citations": [], "intent": "", "intent_route": "",
                    "de_noised_question": "", "tool_calls": 0,
                    "elapsed_s": round(time.time() - t0, 1),
                    "pattern_ok": False, "doc_ok": False, "refused": False,
                    "citation_warnings": 0, "faithful": None,
                    "error": f"{type(e).__name__}: {e}",
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            with open(_target(), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            if args.verbose:
                mark = "ok " if (row["pattern_ok"] and row["doc_ok"]) else "BAD"
                print(f"  [{mark}] {case['id']:4} {row['intent_route']:13} "
                      f"{row['elapsed_s']:5.1f}s {case['question'][:60]}")
        report(variant, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
