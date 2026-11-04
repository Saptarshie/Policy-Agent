"""Measure the answer-shape classifier on labelled answers.

The classifier decides what the faithfulness judge is asked to check, so it is
worth measuring rather than assuming. Every case in ``eval/shape_cases.jsonl``
is a realistic answer shape (answer / partial / refusal / chit-chat); this script
asks the live classifier and prints per-class accuracy plus the confusions.

    python eval/run_shape_eval.py              # decision model (DECISION_MODE)
    python eval/run_shape_eval.py --backend llm

No agent calls, no retrieval: one classifier request per case, cached by nothing
(the point is to see the model's actual behaviour, not a cached verdict).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_config                      # noqa: E402
from app.agent.answer_shape import (LLMShapeClassifier,  # noqa: E402
                                    SystemOneShapeClassifier)
from app.contradictions.gate import build_gate_from_config  # noqa: E402


def load_cases(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "eval" / "shape_cases.jsonl"))
    ap.add_argument("--backend", choices=("auto", "llm"), default="auto")
    args = ap.parse_args()

    cfg = get_config()
    classifier = None
    if args.backend == "auto":
        gate = build_gate_from_config(cfg)
        client = getattr(gate, "remote", None)
        if client is not None and hasattr(client, "ask"):
            classifier = SystemOneShapeClassifier(client)
            print(f"backend: decision model ({cfg.decision_model_name})")
    if classifier is None:
        from app.llm import OpenAICompatClient
        classifier = LLMShapeClassifier(OpenAICompatClient.from_config(cfg))
        print("backend: chat model (JSON mode)")

    cases = load_cases(Path(args.cases))
    rows, ok = [], 0
    for case in cases:
        t0 = time.time()
        shape = classifier.classify(case["question"], case["answer"])
        hit = shape.kind == case["shape"]
        ok += hit
        rows.append({"id": case["id"], "expected": case["shape"],
                     "got": shape.kind, "ok": hit,
                     "scores": shape.scores, "latency_s": round(time.time() - t0, 2)})
        print(f"  {case['id']}  expected={case['shape']:<9} got={shape.kind:<9} "
              f"{'ok' if hit else 'MISS'}  {rows[-1]['latency_s']}s")

    total = len(rows)
    print(f"\naccuracy: {ok}/{total} = {100.0 * ok / total:.1f}%")
    by_class = defaultdict(lambda: [0, 0])
    confusions = Counter()
    for r in rows:
        by_class[r["expected"]][1] += 1
        by_class[r["expected"]][0] += int(r["ok"])
        if not r["ok"]:
            confusions[(r["expected"], r["got"])] += 1
    for kind, (hits, n) in sorted(by_class.items()):
        print(f"  {kind:<9} {hits}/{n}")
    if confusions:
        print("confusions:", {f"{a}->{b}": c for (a, b), c in confusions.items()})
    lat = sorted(r["latency_s"] for r in rows)
    print(f"latency p50 {lat[len(lat) // 2]:.1f}s  max {lat[-1]:.1f}s")

    out = ROOT / "eval" / "results" / "shape_eval.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"rows -> {out}")
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
