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

from scoring import load_cases, load_rows, score_row

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"






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
        verdict = score_row(case, row)
        ok, pattern_ok, doc_ok = verdict["ok"], verdict["pattern_ok"], verdict["doc_ok"]
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
