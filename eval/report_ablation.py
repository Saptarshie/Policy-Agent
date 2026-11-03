#!/usr/bin/env python3
"""End-to-end ablation report: merge per-variant rows and compare.

Variants (see run_ablation.py):
  A  regex intent + agent prompt v1 + judge v1   (the original system)
  B  LLM intent   + agent prompt v1 + judge v1
  C  LLM intent   + agent prompt v2 + judge v2
  D  LLM intent   + agent prompt v2, judge OFF   (judge cost/benefit)

Every row is one (variant, case) result; a case may appear more than once if a
run was retried, so the newest row per key wins.

    python eval/report_ablation.py
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"


def load_rows():
    rows = {}
    for path in sorted(RESULTS.glob("ablation_*.jsonl")):
        if path.name == "ablation_runs.jsonl":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            rows[(row["variant"], row["id"])] = row
    # also fold in anything written to the legacy single-file log
    legacy = RESULTS / "ablation_runs.jsonl"
    if legacy.exists():
        for line in legacy.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            rows.setdefault((row["variant"], row["id"]), row)
    return list(rows.values())


def warn_list(row):
    """citation_warnings holds BOTH citation problems and judge verdicts; they
    mean different things and must be counted separately."""
    w = row.get("citation_warnings") or []
    if isinstance(w, int):
        return []
    return [str(x) for x in w]


def cit_warns(row) -> int:
    return sum(1 for w in warn_list(row) if "citation" in w.lower())


def faith_warns(row) -> int:
    return sum(1 for w in warn_list(row) if "faithfulness" in w.lower())


def warns(row) -> int:
    return len(warn_list(row))


def main() -> int:
    rows = load_rows()
    if not rows:
        print("no ablation rows yet")
        return 2
    by_var = defaultdict(list)
    for r in rows:
        by_var[r["variant"]].append(r)

    print(f"{'var':4} {'n':>3} {'pass':>6} {'citLet':>7} {'judgeFAIL':>10} "
          f"{'judge?':>7} {'refused':>8} {'p50s':>6} {'tools':>6} {'errors':>7}")
    print("-" * 74)
    for var in sorted(by_var):
        rs = by_var[var]
        n = len(rs)
        passed = sum(1 for r in rs if r.get("pattern_ok"))
        clean = sum(1 for r in rs if cit_warns(r) == 0)
        fail = sum(1 for r in rs if r.get("faithful") is False)
        judged = sum(1 for r in rs if r.get("faithful") is not None)
        refused = sum(1 for r in rs if r.get("refused"))
        lat = [r["elapsed_s"] for r in rs if r.get("elapsed_s")]
        tools = [r["tool_calls"] for r in rs if r.get("tool_calls") is not None]
        errors = sum(1 for r in rs if r.get("error"))
        print(f"{var:4} {n:3} {100*passed/n:5.1f}% {100*clean/n:8.1f}% "
              f"{100*fail/n:9.1f}% {judged:>7} {refused:>8} "
              f"{statistics.median(lat):6.1f} {statistics.mean(tools):6.1f} {errors:>7}")

    print()
    print("by case type (pass/n):")
    types = sorted({r.get("type", "?") for r in rows})
    print("var  " + "".join(t.rjust(12) for t in types))
    for var in sorted(by_var):
        line = var.ljust(5)
        for t in types:
            sel = [r for r in by_var[var] if r.get("type") == t]
            line += ("%d/%d" % (sum(1 for r in sel if r.get("pattern_ok")), len(sel))).rjust(12)
        print(line)

    print()
    print("per-case (P=pattern ok, C=clean citations, F=faithful):")
    ids = sorted({r["id"] for r in rows})
    print("id    " + "".join(v.rjust(8) for v in sorted(by_var)))
    for cid in ids:
        line = cid.ljust(6)
        for var in sorted(by_var):
            r = next((x for x in by_var[var] if x["id"] == cid), None)
            if not r:
                line += "-".rjust(8)
                continue
            marks = ""
            marks += "P" if r.get("pattern_ok") else "p"
            marks += "C" if cit_warns(r) == 0 else "c"
            marks += "F" if r.get("faithful") else ("?" if r.get("faithful") is None else "f")
            line += marks.rjust(8)
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
