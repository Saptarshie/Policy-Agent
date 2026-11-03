#!/usr/bin/env python3
"""Retriever comparison - vector vs keyword vs hybrid (RRF).

Label-only, so it is fast, free and repeatable: no LLM call is made. Two
question sets are scored separately:

  * eval_set + qa_extra  - natural employee questions (paraphrase-heavy).
  * retrieval_probes     - exact-token lookups (policy numbers, tier names,
    currency amounts), the case where dense retrieval is known to be weak.

Metrics per mode, at k = 1/3/6:

  doc@k  the expected document appears in the top-k passages
  ans@k  a top-k passage *from that document* also contains the expected
         answer text (regex) - i.e. retrieval put the answer in front of the
         model, which is the only thing the agent can actually use

    python eval/measure_retrieval.py --k 1,3,6
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.admin import AdminAPI  # noqa: E402

MODES = ("vector", "keyword", "hybrid")


def load(name):
    path = ROOT / "eval" / name
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def score(api, cases, ks):
    out = {}
    for mode in MODES:
        doc_hits = {k: 0 for k in ks}
        ans_hits = {k: 0 for k in ks}
        n = 0
        for c in cases:
            expect = (c.get("expect_doc") or "").lower()
            pat = c.get("expect_pattern") or c.get("expect_answer_regex") or ""
            if not expect:
                continue
            n += 1
            hits = api.rag.search(c["question"], top_k=max(ks), mode=mode)
            try:
                rx = re.compile(pat, re.I) if pat else None
            except re.error:
                rx = None
            for k in ks:
                top = hits[:k]
                doc_hits[k] += any(expect in h.doc_name.lower() for h in top)
                if rx:
                    ans_hits[k] += any(expect in h.doc_name.lower() and
                                       rx.search(h.text or "") for h in top)
        out[mode] = (doc_hits, ans_hits, n)
    return out


def table(title, res, ks):
    print()
    print(title)
    head = "mode".ljust(9) + "".join(("doc@%d" % k).rjust(9) for k in ks) +            "".join(("ans@%d" % k).rjust(9) for k in ks)
    print(head)
    print("-" * len(head))
    for mode in MODES:
        doc_hits, ans_hits, n = res[mode]
        line = mode.ljust(9)
        line += "".join(("%3d/%-5d" % (doc_hits[k], n)) for k in ks)
        line += "".join(("%3d/%-5d" % (ans_hits[k], n)) for k in ks)
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="1,3,6")
    args = ap.parse_args()
    ks = [int(x) for x in args.k.split(",")]

    api = AdminAPI()
    if not api.db.list_documents():
        print("no documents ingested - run run.py upload first")
        return 2

    natural = load("eval_set.jsonl") + load("qa_extra.jsonl")
    probes = load("retrieval_probes.jsonl")
    table("natural questions (retriever=%s)" % api.cfg.retrieval_mode,
          score(api, natural, ks), ks)
    if probes:
        table("exact-token probes", score(api, probes, ks), ks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
