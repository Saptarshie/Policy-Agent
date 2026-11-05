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


def _norm(text: str) -> str:
    flat = re.sub(r"[§#]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", flat).strip(" .-")


def pack_sections(api, cases, ks, modes=MODES, breadcrumb: bool = True):
    """Rank the reference's own sections, per mode, at several k.

    A hit means: one of the top-k passages IS that section - either the chunk's own
    `section`, or (breadcrumb=True) a section on its breadcrumb path, which is how a
    reference to a parent heading ("8. Stacking and double funding") can be satisfied by
    the chunk that carries only the child ("8.1 Permitted stacking").
    """
    out = {}
    for mode in modes:
        hits = {k: 0 for k in ks}
        mrr = 0.0
        total = 0
        missing_at_max = []
        for c in cases:
            expected = [e.get("section") or "" for e in c.get("ground_truth_citations", [])]
            expected = [e for e in expected if e.strip()]
            if not expected:
                continue
            total += len(expected)
            top = api.rag.search(c["question"], top_k=max(ks), mode=mode)
            ranks = []
            for sec in expected:
                want = _norm(sec)
                rank = None
                for i, h in enumerate(top, 1):
                    own = _norm(h.section)
                    crumb = _norm(getattr(h, "breadcrumb", "") or "")
                    if own == want or (breadcrumb and want and f" {want} " in f" {crumb} "):
                        rank = i
                        break
                ranks.append(rank)
                if rank:
                    mrr += 1.0 / rank
                else:
                    missing_at_max.append((c["id"], sec))
            for k in ks:
                hits[k] += sum(1 for r in ranks if r and r <= k)
        out[mode] = (hits, total, mrr, missing_at_max)
    return out


def section_table(title, res, ks):
    print()
    print(title)
    head = "mode".ljust(9) + "".join(("sec@%d" % k).rjust(9) for k in ks) + "     MRR"
    print(head)
    print("-" * len(head))
    for mode, (hits, total, mrr, _missing) in res.items():
        line = mode.ljust(9)
        line += "".join(("%3d/%-5d" % (hits[k], total)) for k in ks)
        line += ("   %.3f" % (mrr / total if total else 0.0))
        print(line)
    for mode, (_hits, _total, _mrr, missing) in res.items():
        if missing:
            print(f"\n  {mode}: never found at k={max(ks)}")
            for qid, sec in missing:
                print(f"    {qid}: {sec}")


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
    ap.add_argument("--pack", default="",
                    help="score a benchmark pack's expected SECTIONS instead of the "
                         "eval sets (needs the pack's documents indexed in DATA_DIR)")
    ap.add_argument("--modes", default="vector,keyword,hybrid")
    ap.add_argument("--exact-sections", action="store_true",
                    help="match the chunk's own section only, not its breadcrumb path")
    args = ap.parse_args()
    ks = [int(x) for x in args.k.split(",")]

    api = AdminAPI()
    if not api.db.list_documents():
        print("no documents ingested - run run.py upload first")
        return 2

    if args.pack:
        pack = Path(args.pack)
        if not pack.is_absolute():
            pack = ROOT / pack
        cases = []
        for f in sorted(pack.glob("*.jsonl")):
            cases += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()
                      if l.strip()]
        modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
        print(f"pack {pack.name}: {len(cases)} questions, k={ks}, "
              f"weights v={api.cfg.hybrid_vector_weight} k={api.cfg.hybrid_keyword_weight}, "
              f"breadcrumb={'off' if args.exact_sections else 'on'}")
        section_table("expected sections retrieved",
                      pack_sections(api, cases, ks, modes,
                                    breadcrumb=not args.exact_sections), ks)
        return 0

    natural = load("eval_set.jsonl") + load("qa_extra.jsonl")
    probes = load("retrieval_probes.jsonl")
    table("natural questions (retriever=%s)" % api.cfg.retrieval_mode,
          score(api, natural, ks), ks)
    if probes:
        table("exact-token probes", score(api, probes, ks), ks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
