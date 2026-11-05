"""Does the decision model distinguish a source from a coincidence?

Citation recovery asks the SystemOne model, once per candidate, "does this passage support
the answer?". Before choosing a threshold, this measures the scores against ground truth:
for every section the agent read, was it one the reference answer expects, and what did the
model score it?

    python eval/calibrate_citation_recovery.py --pack Benchmark/adversarial_policy_arbitrage_pack \
        --data-dir data_bench_adv --workers 10

Reports, per question, the score of every seen section split into expected / other, then
the aggregate: mean and median of each group, and the recall / false-positive counts at a
sweep of thresholds. Saves eval/results/citation_recovery_<stamp>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def norm(text: str) -> str:
    import re
    flat = re.sub(r"[§#]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", flat).strip(" .-")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cap", type=int, default=24)
    ap.add_argument("--snippet", type=int, default=400)
    args = ap.parse_args()

    pack = Path(args.pack)
    data_dir = Path(args.data_dir)
    if not pack.is_absolute():
        pack = ROOT / pack
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["CITATION_RECOVERY"] = "0"          # we score manually here

    from app import config as cfg_mod
    cfg_mod._CACHE = None
    from app.admin import AdminAPI
    from app.citation_recovery import (build_asker, build_candidates, build_questions,
                                       build_state)

    rows = []
    for f in sorted(pack.glob("*.jsonl")):
        rows += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]

    api = AdminAPI()
    # CITATION_RECOVERY is 0 here so api.ask does not spend its own request; the asker is
    # built directly because this harness scores every candidate itself
    ask = build_asker(api.cfg)
    if ask is None:
        raise SystemExit("no SystemOne decision model configured (DECISION_MODEL_*)")
    print(f"pack {pack.name}: {len(rows)} questions, cap {args.cap}, snippet {args.snippet}, "
          f"workers {args.workers}")

    def one(row: dict) -> dict:
        res = api.ask(row["question"])
        expected = {norm(c.get("document", "") + " " + c.get("section", ""))
                    for c in row.get("ground_truth_citations", [])}
        expected_sections = {norm(c.get("section", "")) for c in row.get("ground_truth_citations", [])}
        cited = {norm(c) for c in (res.citations or [])}
        cands = build_candidates(api.db, res.seen_chunk_ids, [],
                                 max_candidates=args.cap, snippet_chars=args.snippet)
        if not cands:
            return {"id": row["id"], "rows": [], "cited": sorted(cited)}
        scores = ask(build_state(res.answer, cands), build_questions(cands)) or {}
        out = []
        for k, c in enumerate(cands, start=1):
            try:
                p = float(scores.get(f"cite_{k}", 0.0))
            except (TypeError, ValueError):
                p = 0.0
            out.append({
                "citation": c["citation"],
                "section": c["section"],
                "score": round(p, 3),
                "is_expected": norm(c["section"]) in expected_sections
                               or norm(c["citation"]) in expected,
                "already_cited": norm(c["citation"]) in cited,
            })
        return {"id": row["id"], "rows": out, "cited": sorted(cited)}

    results = []
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(rows))) as pool:
            for f in as_completed({pool.submit(one, r): r["id"] for r in rows}):
                results.append(f.result())
    else:
        results = [one(r) for r in rows]
    results.sort(key=lambda r: r["id"])

    pos, neg, missing = [], [], []
    for r in results:
        got = set()
        for c in r["rows"]:
            (pos if c["is_expected"] else neg).append(c["score"])
            if c["is_expected"]:
                got.add(norm(c["section"]))
        for c in r["rows"]:
            if not c["is_expected"] and not c["already_cited"] and c["score"] >= 0.4:
                missing.append((r["id"], c["citation"], c["score"]))
        print(f"  {r['id']}: {len(r['rows'])} candidates, "
              f"{sum(1 for c in r['rows'] if c['is_expected'])} expected, "
              f"max score {max((c['score'] for c in r['rows']), default=0):.2f}")

    print(f"\nSCORE SEPARATION  expected sections n={len(pos)}, other sections n={len(neg)}")
    if pos:
        print(f"  expected   mean {statistics.mean(pos):.3f}  median {statistics.median(pos):.3f}"
              f"  max {max(pos):.3f}")
    if neg:
        print(f"  other      mean {statistics.mean(neg):.3f}  median {statistics.median(neg):.3f}"
              f"  max {max(neg):.3f}")

    print("\nTHRESHOLD SWEEP (on every seen section, cited ones included)")
    print(f"{'t':>5}{'added':>7}{'true':>6}{'false':>7}{'precision':>11}{'recall_of_expected':>21}")
    for t in THRESHOLDS:
        added = [c for r in results for c in r["rows"] if c["score"] >= t]
        tp = sum(1 for c in added if c["is_expected"])
        fp = len(added) - tp
        prec = (100.0 * tp / len(added)) if added else 0.0
        rec = (100.0 * tp / len(pos)) if pos else 0.0
        print(f"{t:>5}{len(added):>7}{tp:>6}{fp:>7}{prec:>10.1f}%{rec:>20.1f}%")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = ROOT / "eval" / "results" / f"citation_recovery_{stamp}.json"
    dest.write_text(json.dumps({"pack": pack.name, "cap": args.cap, "snippet": args.snippet,
                                "questions": len(results), "rows": results}, indent=1),
                    encoding="utf-8")
    print(f"\n  saved: {dest}")


if __name__ == "__main__":
    main()
