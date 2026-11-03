#!/usr/bin/env python3
"""Three-minute demo script: upload -> index -> fact / table / off-policy / refusal.

Prints the structured JSON the API returns (answer + citations + intent +
verification), so a screen recording shows the whole contract, not just prose.

    python scripts/demo.py                 # uses ./data, ingests sample_docs if empty
    python scripts/demo.py --fresh         # wipe ./data first (destructive)

Requires a configured .env (LLM endpoint) and Ollama with the embed model.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.admin import AdminAPI  # noqa: E402

SCENARIOS = [
    ("FACT", "How many casual leave days can I carry forward?"),
    ("TABLE / TIER", "Does the Standard health tier cover dental implants?"),
    ("OFF-POLICY (excluded by the corpus)", "Can I expense a personal home gym?"),
    ("NOT IN CORPUS (must refuse)", "What is the sabbatical leave policy?"),
]

DOCS = [
    "sample_docs/leave_policy.md",
    "sample_docs/health_plans.md",
    "sample_docs/expense_policy.md",
    "sample_docs/amendments/leave_policy_2025_amendment.md",
]


def wait_ready(api: AdminAPI, doc_id: int, timeout: float = 180.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        doc = api.db.get_document(doc_id)
        if (doc or {}).get("status") in ("ready", "error"):
            return doc or {}
        time.sleep(0.5)
    return api.db.get_document(doc_id) or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="delete ./data first")
    args = ap.parse_args()

    if args.fresh:
        data = ROOT / "data"
        if data.exists():
            shutil.rmtree(data)

    api = AdminAPI()
    if not api.db.list_documents():
        print("· ingesting sample policies")
        for rel in DOCS:
            path = ROOT / rel
            if not path.exists():
                print(f"  ! missing {rel}")
                continue
            info = api.upload_document_async(path, name=path.stem)
            doc = wait_ready(api, info["doc_id"])
            print(f"  {path.name:44} -> {doc.get('status')}")

    for label, question in SCENARIOS:
        print("\n" + "=" * 78)
        print(f"{label}\nQ: {question}")
        print("=" * 78)
        res = api.ask(question)
        out = res.to_dict()
        print("answer:\n" + (out.get("answer") or "").strip())
        print("\nstructured result:")
        keys = ("intent", "intent_route", "citations", "citation_warnings",
                "faithful", "unsupported_claims", "tool_calls")
        print(json.dumps({k: out.get(k) for k in keys if k in out},
                         indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
