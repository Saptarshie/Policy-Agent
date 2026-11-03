#!/usr/bin/env python3
"""Judge calibration: does prompt v2 catch more injected claims, and does it
stay quiet on clean answers?

Each sample is a (question, answer, contexts) triple built from the ablation
log:
  * CLEAN    - the answer the agent actually produced for a case whose
               expected pattern matched (assumed correct).
  * TAMPERED - the same answer with one fabricated policy claim spliced in
               (a number/limit that appears nowhere in the contexts).

Ground truth: CLEAN must be judged faithful, TAMPERED must not. Reports TPR
(tampered caught), TNR (clean passed) and the disagreement rate between
prompt variants, so the judge is measured like a classifier instead of being
assumed to work.

    python eval/calibrate_judge.py --samples 10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.admin import AdminAPI  # noqa: E402
from app.agent.verify import AnswerVerifier  # noqa: E402
from app.prompts import FAITHFULNESS_SYSTEM, FAITHFULNESS_SYSTEM_V2  # noqa: E402

RUNS = ROOT / "eval" / "results" / "ablation_runs.jsonl"
OUT = ROOT / "eval" / "results" / "judge_calibration.jsonl"

# claims that do not exist anywhere in the sample corpus
FABRICATIONS = [
    " In addition, employees may carry forward up to 30 days of casual leave.",
    " The company also reimburses home gym equipment up to $2,000 per year.",
    " The standard tier covers dental implants at 100% with no waiting period.",
    " Employees receive 25 casual leave days per year in addition to sick leave.",
    " International per diem is $150 per day for all grades.",
    " Remote work is permitted 4 days per week without manager approval.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--variant", default="C", help="ablation variant to draw answers from")
    args = ap.parse_args()

    if not RUNS.exists():
        print("run eval/run_ablation.py first")
        return 2
    rows = [json.loads(l) for l in RUNS.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["variant"] == args.variant
            and r.get("pattern_ok") and not r.get("refused") and len(r.get("answer", "")) > 80]
    if not rows:
        print("no suitable answered rows in the ablation log")
        return 2
    random.seed(7)
    rows = rows[: args.samples]

    api = AdminAPI()
    judges = {
        "v1": AnswerVerifier(db=api.db, llm=api.llm, enabled=True,
                             system_prompt=FAITHFULNESS_SYSTEM),
        "v2": AnswerVerifier(db=api.db, llm=api.llm, enabled=True,
                             system_prompt=FAITHFULNESS_SYSTEM_V2),
    }

    results = {"v1": {"clean_ok": 0, "tamper_caught": 0},
               "v2": {"clean_ok": 0, "tamper_caught": 0}}
    n = len(rows)
    for r in rows:
        verdicts = {}
        question = r.get("de_noised_question") or r["question"]
        contexts = [h.text for h in api.rag.search(question, top_k=3)]
        if not contexts:
            continue
        tampered = r["answer"].rstrip()
        insert_at = tampered.rfind("Sources:")
        claim = random.choice(FABRICATIONS)
        tampered = (tampered[:insert_at] + claim + "\n\n" + tampered[insert_at:]
                    if insert_at > 0 else tampered + claim)
        for name, judge in judges.items():
            clean = judge.judge(question, r["answer"], contexts)
            bad = judge.judge(question, tampered, contexts)
            results[name]["clean_ok"] += int(clean.faithful is True)
            results[name]["tamper_caught"] += int(bad.faithful is False)
            verdicts[name] = {"clean": clean.faithful, "tampered": bad.faithful}
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": r["id"], "claim": claim.strip(),
                                 "verdicts": verdicts}, ensure_ascii=False) + "\n")

    print(f"\nsamples={n}  (judge = classifier: faithful vs not)")
    for name, m in results.items():
        print(f"  {name}: TNR(clean judged faithful) = {m['clean_ok']}/{n}"
              f"   TPR(tampered caught) = {m['tamper_caught']}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
