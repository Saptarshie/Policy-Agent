#!/usr/bin/env python3
"""Re-score stored ablation answers against the CURRENT case labels.

Why this exists: cases are labelled by hand, and reading an answer sometimes
shows the label was wrong (the answer quoted a clause that settles the question,
or a question is legitimately multi-hop). Re-running the agent would cost 20
minutes and change nothing; re-applying the new label to the stored answer is the
honest way to correct the table, provided every relabelling is written down (see
the `note` field on the case and ABLATION.md).

    python eval/rescore.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"

REFUSAL_RE = re.compile(
    r"(?i)ingested policy documents do not cover|"
    r"don'?t have enough information|do not have enough information")


def load_cases():
    cases = {}
    for name in ("eval_set.jsonl", "qa_extra.jsonl"):
        for line in (ROOT / "eval" / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = json.loads(line)
                cases[c["id"]] = c
    return cases


def load_rows():
    rows = {}
    for path in sorted(RESULTS.glob("ablation_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            rows[(row["variant"], row["id"])] = row
    return rows


def score(case, row):
    answer = row.get("answer") or ""
    citations = row.get("citations") or []
    is_refusal = bool(REFUSAL_RE.search(answer)) and not citations
    if case.get("expect_refusal"):
        pattern_ok = bool(REFUSAL_RE.search(answer))
    else:
        rx = case.get("expect_answer_regex")
        pattern_ok = bool(re.search(rx, answer, re.I)) if rx else True
        if pattern_ok and is_refusal and not case.get("allow_refusal"):
            pattern_ok = False
    # expect_doc may list alternatives ("travel_reimbursement|expense"): the old
    # literal substring test made those cases impossible to pass
    hints = [h.strip().lower() for h in (case.get("expect_doc") or "").split("|") if h.strip()]
    cited = " ".join(citations).lower()
    doc_ok = True
    if hints and citations:
        doc_ok = any(h in cited for h in hints)
    return pattern_ok and doc_ok and not row.get("error"), pattern_ok, doc_ok


def main() -> int:
    cases = load_cases()
    rows = load_rows()
    if not rows:
        print("no ablation rows")
        return 2
    by_var = defaultdict(list)
    for (variant, cid), row in rows.items():
        case = cases.get(cid)
        if not case:
            continue
        ok, pattern_ok, doc_ok = score(case, row)
        by_var[variant].append((case, row, ok, pattern_ok, doc_ok))

    print(f"{'var':4} {'n':>3} {'pass':>7} {'judgeFAIL':>10} {'p50s':>6}")
    print("-" * 34)
    for variant in sorted(by_var):
        items = by_var[variant]
        n = len(items)
        passed = sum(1 for *_, ok, _, _ in [(c, r, ok, p, d) for c, r, ok, p, d in items])
        passed = sum(1 for c, r, ok, p, d in items if ok)
        flagged = sum(1 for c, r, ok, p, d in items if r.get("faithful") is False)
        lat = sorted(r["elapsed_s"] for c, r, ok, p, d in items if r.get("elapsed_s"))
        p50 = lat[len(lat) // 2] if lat else 0
        print(f"{variant:4} {n:3} {100*passed/n:6.1f}% {flagged:>10} {p50:6.1f}")

    print("\nfailures under the current labels:")
    for variant in sorted(by_var):
        for case, row, ok, pattern_ok, doc_ok in by_var[variant]:
            if not ok:
                why = []
                if not pattern_ok:
                    why.append("expectation")
                if not doc_ok:
                    why.append("citations")
                if row.get("error"):
                    why.append("error")
                print(f"  {variant} {case['id']:5} [{','.join(why)}] "
                      f"{case['question'][:58]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
