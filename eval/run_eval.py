#!/usr/bin/env python3
"""Run the eval set against a LIVE stack (Ollama + LLM endpoint from .env).

Usage (from the project root):

    python eval/run_eval.py               # upload sample docs if empty, then score
    python eval/run_eval.py --no-upload   # score whatever is already ingested
    python eval/run_eval.py --verbose     # show every answer

Each case in eval_set.jsonl is asked through the full AdminAPI pipeline
(intent gate -> ReAct agent -> verifier). A case passes when:
  * expect_refusal=false and expect_answer_regex matches the answer, and
    (if set) expect_doc appears among the citations; or
  * expect_refusal=true and the answer matches the refusal regex.

Conflict-type cases (e.g. "3 vs 1 remote days" once v2 is uploaded) accept
either side of the conflict - they test that the agent finds a grounded
number, not which version is "right".
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

SAMPLE_DOCS = sorted(
    list((ROOT / "sample_docs").glob("*.txt"))
    + list((ROOT / "sample_docs").glob("*.md"))
    + list((ROOT / "sample_docs" / "amendments").glob("*.md"))
)


def load_cases(path: Path):
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cases.append(json.loads(line))
    return cases


def upload_samples(api: AdminAPI) -> None:
    have = {d["name"] for d in api.list_documents()}
    for p in SAMPLE_DOCS:
        if p.stem in have:
            continue
        print(f"  uploading {p.name} ...")
        api.upload_document(p, start_scan=False)
    # run the cascade once over everything (flags conflicts for review)
    api.scan(blocking=True)


def score(case: dict, result) -> tuple:
    ok = True
    notes = []
    answer = result.answer or ""
    if result.error:
        return False, [f"agent error: {result.error}"]
    if case.get("expect_refusal"):
        pat = case.get("expect_answer_regex") or "."
        if not re.search(pat, answer, re.I):
            ok, notes = False, notes + ["expected a refusal, answer did not match"]
    else:
        pat = case.get("expect_answer_regex")
        if pat and not re.search(pat, answer):
            ok = False
            notes.append(f"answer did not match /{pat}/")
        doc_hint = (case.get("expect_doc") or "").lower()
        if doc_hint and result.citations:
            joined = " ".join(result.citations).lower()
            if doc_hint not in joined:
                ok = False
                notes.append(f"expected a citation containing '{doc_hint}', "
                             f"got {result.citations}")
        if result.citation_warnings:
            notes.append(f"warnings: {result.citation_warnings}")
    return ok, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    api = AdminAPI()
    health = api.health()
    if not health["llm_configured"]:
        print("LLM not configured - set LLM_BASE_URL/LLM_MODEL in .env first.")
        return 2
    print(f"model={health['llm_model']}  embed={health['embed_model']}  "
          f"docs={health['documents']}")

    if not args.no_upload:
        upload_samples(api)

    cases = load_cases(ROOT / "eval" / "eval_set.jsonl")
    passed = 0
    for case in cases:
        result = api.ask(case["question"])
        ok, notes = score(case, result)
        passed += ok
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {case['id']} ({case['type']}) {case['question']}")
        if args.verbose or not ok:
            print(f"       answer: {(result.answer or '')[:200]}")
            print(f"       citations: {result.citations}")
            for n in notes:
                print(f"       - {n}")

    total = len(cases)
    print(f"\n{passed}/{total} passed ({100 * passed / max(1, total):.0f}%)")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
