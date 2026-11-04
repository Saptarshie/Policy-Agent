"""Repeat one question N times and show the answer head, to measure a prompt change.

A single pass on a volatile question proves nothing (MH-01 has answered both Rs 0 and
Rs 25,000 on the same code), so a change is judged on a hit rate over repeats, plus a
control set that must not regress.

    python eval/repeat_question.py MH-01 5
    python eval/repeat_question.py LC-08 3 --pack Benchmark/complex_contract_reasoning_pack --data-dir data_bench_legal
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", help="comma separated question ids")
    ap.add_argument("repeats", nargs="?", type=int, default=3)
    ap.add_argument("--pack", default=str(ROOT / "Benchmark" / "hr_rag_policy_pack_and_benchmark"))
    ap.add_argument("--data-dir", default=str(ROOT / "data_bench"))
    ap.add_argument("--head", type=int, default=180, help="chars of answer to show")
    ap.add_argument("--judge", action="store_true",
                    help="grade each answer with the pack's judge (same as run_benchmark)")
    args = ap.parse_args()

    os.environ["DATA_DIR"] = args.data_dir
    from app import config as cfg_mod        # noqa: E402
    cfg_mod._CACHE = None
    from app.admin import AdminAPI           # noqa: E402

    pack = Path(args.pack)
    rows = {r["id"]: r for r in
            (__import__("json").loads(line) for line in
             sorted(pack.glob("*.jsonl"))[0].read_text(encoding="utf-8").splitlines())}

    api = AdminAPI()
    judge = None
    if args.judge:
        sys.path.insert(0, str(ROOT / "eval"))
        from run_benchmark import judge as _judge      # noqa: E402
        judge = _judge
    for qid in args.ids.split(","):
        qid = qid.strip()
        print(f"\n=== {qid}: {rows[qid]['question']}")
        print(f"    REFERENCE: {rows[qid]['ground_truth_answer'][:200]}")
        verdicts = []
        for i in range(args.repeats):
            t0 = time.time()
            res = api.ask(rows[qid]["question"])
            dt = time.time() - t0
            head = (res.answer or "").replace("\n", " ")[:args.head]
            tag = ""
            if judge:
                g = judge(api, rows[qid], res.answer or "")
                verdicts.append(g["verdict"])
                tag = f"{g['verdict']:<10}"
            print(f"  [{i + 1}/{args.repeats}] {dt:5.1f}s {tag}tools={res.tool_calls}  {head}")
        if verdicts:
            good = sum(v == "correct" for v in verdicts)
            print(f"  ==> {good}/{len(verdicts)} correct  {verdicts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
