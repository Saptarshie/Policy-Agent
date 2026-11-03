"""One scoring implementation for every ablation view.

`run_ablation.py` (writes rows), `report_ablation.py` (merged table) and
`rescore.py` (re-apply corrected labels) must agree, or the numbers in
`ABLATION.md` depend on which script you ran. Any change to the pass rule belongs
here and nowhere else.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# strict: the answer IS a refusal. Keep this narrow - the escalation note for
# sensitive cases also says "contact your HR business partner" and must not be
# mistaken for a refusal.
REFUSAL_RE = re.compile(
    r"(?i)ingested policy documents do not cover|"
    r"don'?t have enough information|do not have enough information")


def load_cases(directory: Path | None = None) -> dict:
    directory = directory or (ROOT / "eval")
    cases = {}
    for name in ("eval_set.jsonl", "qa_extra.jsonl"):
        path = directory / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                case = json.loads(line)
                cases[case["id"]] = case
    return cases


def score_row(case: dict, row: dict) -> dict:
    """Score one stored row (or a freshly produced one) against *case*.

    row needs: answer, citations, error. Returns ok/pattern_ok/doc_ok/refused.
    """
    answer = row.get("answer") or ""
    citations = row.get("citations") or []
    is_refusal = bool(REFUSAL_RE.search(answer)) and not citations

    if case.get("expect_refusal"):
        pattern_ok = bool(REFUSAL_RE.search(answer))
    else:
        rx = case.get("expect_answer_regex")
        pattern_ok = bool(re.search(rx, answer, re.I)) if rx else True
        if pattern_ok and is_refusal and not case.get("allow_refusal"):
            pattern_ok = False          # a refusal cannot satisfy a factual case

    # expect_doc may list alternatives ("travel_reimbursement|expense")
    hints = [h.strip().lower() for h in (case.get("expect_doc") or "").split("|")
             if h.strip()]
    cited = " ".join(citations).lower()
    doc_ok = True
    if hints and citations:
        doc_ok = any(h in cited for h in hints)

    return {
        "pattern_ok": pattern_ok,
        "doc_ok": doc_ok,
        "refused": is_refusal,
        "ok": pattern_ok and doc_ok and not row.get("error"),
    }


def load_rows(directory: Path | None = None) -> dict:
    """All stored rows, newest wins per (variant, case id)."""
    directory = directory or (ROOT / "eval" / "results")
    rows = {}
    for path in sorted(directory.glob("ablation_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            rows[(row["variant"], row["id"])] = row
    return rows
