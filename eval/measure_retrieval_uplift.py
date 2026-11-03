#!/usr/bin/env python3
"""Does de-noising actually improve retrieval? (objective, label-only)

For every case in eval_set/qa_extra that names a gold document, run the same
retriever three times - raw question, intent-layer de-noised question, and the
intent layer's first search query - and report hit@3 at the DOCUMENT level.

Needs: eval/results/ablation_runs.jsonl (any variant with the LLM intent layer,
i.e. B/C/D) and the ingested corpus.

    python eval/measure_retrieval_uplift.py [--variant B]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.admin import AdminAPI  # noqa: E402

RUNS = ROOT / "eval" / "results" / "ablation_runs.jsonl"


def load_gold() -> dict:
    gold = {}
    for name in ("eval_set.jsonl", "qa_extra.jsonl"):
        for line in (ROOT / "eval" / name).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            hint = (c.get("expect_doc") or "").lower()
            if hint:
                gold[c["id"]] = hint
    return gold


def docs_for(api: AdminAPI, query: str, k: int = 3) -> set:
    try:
        hits = api.rag.search(query, top_k=k)
    except Exception:
        return set()
    return {h.doc_name.lower() for h in hits}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="B")
    args = ap.parse_args()

    gold = load_gold()
    if not RUNS.exists():
        print("no ablation results yet:", RUNS)
        return 2
    rows = [json.loads(l) for l in RUNS.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["variant"] == args.variant and r["id"] in gold]
    if not rows:
        print(f"no rows for variant {args.variant}")
        return 2

    api = AdminAPI()
    stats = {"raw": 0, "denoised": 0, "query": 0, "n": 0}
    changed_up, changed_down = [], []

    for r in rows:
        g = gold[r["id"]]
        q = r["question"]
        denoised = r.get("de_noised_question") or q
        queries = r.get("search_queries") or []
        q1 = queries[0] if queries else q

        raw_hit = any(g in d for d in docs_for(api, q))
        den_hit = any(g in d for d in docs_for(api, denoised))
        q_hit = any(g in d for d in docs_for(api, q1))

        stats["n"] += 1
        stats["raw"] += raw_hit
        stats["denoised"] += den_hit
        stats["query"] += q_hit
        if den_hit and not raw_hit:
            changed_up.append(r["id"])
        if raw_hit and not den_hit:
            changed_down.append(r["id"])
        print(f"  {r['id']:5} raw={'Y' if raw_hit else 'n'} "
              f"denoised={'Y' if den_hit else 'n'} query={'Y' if q_hit else 'n'}  "
              f"{q[:60]!r}")

    n = stats["n"]
    print(f"\nn={n} gold-document hit@3")
    print(f"  raw question      : {stats['raw']:2}/{n} = {stats['raw']/n:.0%}")
    print(f"  de-noised question: {stats['denoised']:2}/{n} = {stats['denoised']/n:.0%}")
    print(f"  first search query: {stats['query']:2}/{n} = {stats['query']/n:.0%}")
    print(f"  gained by de-noising: {changed_up or '-'}")
    print(f"  lost by de-noising  : {changed_down or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
