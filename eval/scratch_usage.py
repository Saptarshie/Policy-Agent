"""How much working memory does the agent actually use?

The scratchpad exists so the agent can hold intermediate findings across turns; the
prompt only suggests it ("if you kept notes, read them once before answering"). Nothing
measured whether it is used at all, so this asks a pack's questions and counts what the
agent did with its tools.

    python eval/scratch_usage.py --pack Benchmark/adversarial_policy_arbitrage_pack \
        --data-dir data_bench_adv --workers 10

Writes eval/results/scratch_<stamp>.json and prints a per-question table plus the
aggregate. Timing under --workers > 1 includes contention; the counts do not care.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCRATCH = ("scratchpad_write", "scratchpad_read", "scratchpad_edit",
           "scratchpad_delete", "scratchpad_list")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ids", default="", help="comma-separated subset")
    args = ap.parse_args()

    pack = Path(args.pack)
    if not pack.is_absolute():
        pack = ROOT / pack
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    os.environ["DATA_DIR"] = str(data_dir)

    from app import config as cfg_mod
    cfg_mod._CACHE = None
    from app.admin import AdminAPI

    rows = []
    for f in sorted(pack.glob("*.jsonl")):
        rows += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.ids:
        want = {i.strip() for i in args.ids.split(",") if i.strip()}
        rows = [r for r in rows if r["id"] in want]

    api = AdminAPI()
    print(f"pack {pack.name}: {len(rows)} questions, AGENT_MAX_STEPS={api.cfg.agent_max_steps}, "
          f"workers={args.workers}")

    def one(row: dict) -> dict:
        t0 = time.time()
        res = api.ask(row["question"])
        steps = res.steps or []
        tools = [s.get("tool") for s in steps if s.get("type") in ("tool_call", "tool_error")]
        scratch = [t for t in tools if t in SCRATCH]
        return {
            "id": row["id"],
            "turns": res.turns,
            "tool_calls": res.tool_calls,
            "scratch_calls": len(scratch),
            "scratch_by_tool": dict(Counter(scratch)),
            "scratch_notes_kept": res.scratch_notes,
            "tools_by_name": dict(Counter(t for t in tools if t)),
            "elapsed_s": round(time.time() - t0, 1),
        }

    out: list[dict] = []
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(rows))) as pool:
            futs = {pool.submit(one, r): r["id"] for r in rows}
            for f in as_completed(futs):
                out.append(f.result())
    else:
        for r in rows:
            out.append(one(r))

    out.sort(key=lambda r: r["id"])
    print(f"\n{'id':7}{'calls':>6}{'scratch':>9}{'notes':>7}{'turns':>7}{'elapsed':>9}  tools")
    for r in out:
        used = ", ".join(f"{k.replace('scratchpad_', 'sp_')}x{v}"
                         for k, v in sorted(r["scratch_by_tool"].items())) or "-"
        print(f"{r['id']:7}{r['tool_calls']:>6}{r['scratch_calls']:>9}"
              f"{r['scratch_notes_kept']:>7}{r['turns']:>7}{r['elapsed_s']:>8}s  {used}")

    total_calls = sum(r["tool_calls"] for r in out)
    total_scratch = sum(r["scratch_calls"] for r in out)
    users = [r for r in out if r["scratch_calls"]]
    writers = [r for r in out if r["scratch_notes_kept"]]
    by_tool = Counter()
    for r in out:
        by_tool.update(r["scratch_by_tool"])
    overall = Counter()
    for r in out:
        overall.update(r["tools_by_name"])

    print("\nSCRATCHPAD USE")
    print(f"  questions that touched it      {len(users)}/{len(out)}")
    print(f"  questions that kept notes      {len(writers)}/{len(out)}")
    print(f"  scratch calls                  {total_scratch} of {total_calls} tool calls "
          f"({100.0 * total_scratch / total_calls:.1f}%)" if total_calls else "  no calls")
    print(f"  by tool                        {dict(by_tool) or '-'}")
    if users:
        print(f"  scratch calls when used        mean {statistics.mean(r['scratch_calls'] for r in users):.1f}, "
              f"max {max(r['scratch_calls'] for r in users)}")
    print(f"  notes kept per question        mean {statistics.mean(r['scratch_notes_kept'] for r in out):.2f}, "
          f"max {max(r['scratch_notes_kept'] for r in out)}")
    print(f"\n  all tool calls                 {dict(overall.most_common())}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = ROOT / "eval" / "results" / f"scratch_{stamp}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "pack": pack.name, "data_dir": str(data_dir), "workers": args.workers,
        "questions": len(out),
        "totals": {"tool_calls": total_calls, "scratch_calls": total_scratch,
                   "questions_using_scratchpad": len(users),
                   "questions_keeping_notes": len(writers)},
        "by_tool": dict(by_tool), "all_tools": dict(overall), "rows": out,
    }, indent=1), encoding="utf-8")
    print(f"\n  saved: {dest}")


if __name__ == "__main__":
    main()
