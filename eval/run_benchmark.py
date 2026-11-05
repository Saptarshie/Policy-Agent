"""Run the Northstar Dynamics HR RAG benchmark (20 multihop questions).

The pack ships four policy documents and a JSONL of questions with a reference
answer, expected behaviour (``answer`` / ``refuse``) and the ground-truth
citations. This script ingests the four documents into an isolated index, asks
every question through the real pipeline, grades each answer against the
reference with an LLM judge, checks citation recall, and reports scores with the
time each question took.

    python eval/run_benchmark.py                 # fresh index, all 20
    python eval/run_benchmark.py --limit 3       # smoke test
    python eval/run_benchmark.py --reuse         # keep the index, ask again
    python eval/run_benchmark.py --no-judge      # answers only, no grading

Output: eval/results/benchmark_<timestamp>.jsonl (one row per question) plus a
summary printed at the end and written to eval/results/benchmark_<ts>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                     # `python eval/run_benchmark.py`
    sys.path.insert(0, str(ROOT))
DEFAULT_PACK = ROOT / "Benchmark" / "hr_rag_policy_pack_and_benchmark"


def find_manifest(pack: Path) -> Path:
    """manifest.json, or the pack's only *manifest*.json (packs name it freely)."""
    exact = pack / "manifest.json"
    if exact.exists():
        return exact
    found = sorted(pack.glob("*manifest*.json"))
    if len(found) == 1:
        return found[0]
    raise SystemExit(f"{pack}: cannot pick a manifest from {[p.name for p in found]}")


def find_questions(pack: Path) -> Path:
    """The pack's single question file (benchmark_*.jsonl / *_benchmark_*.jsonl)."""
    found = sorted(pack.glob("*.jsonl"))
    if len(found) == 1:
        return found[0]
    named = [p for p in found if p.name.startswith("benchmark_")]
    if len(named) == 1:
        return named[0]
    raise SystemExit(f"{pack}: cannot pick a question file from {[p.name for p in found]}")


def is_refusal(row: dict) -> bool:
    """expected_behavior varies by pack: 'refuse', 'answer', 'answer_with_...'."""
    return str(row.get("expected_behavior", "")).lower().startswith("refuse")

JUDGE_SYSTEM = """You grade a policy assistant's answer against a reference answer.

Return ONE JSON object:
{"verdict": "correct|partial|incorrect", "reason": "<one sentence>",
 "missing": ["<key fact the answer lacks>", ...]}

Rules:
- Compare DECISIONS, AMOUNTS, CONDITIONS and WHO/WHEN, not wording. Different
  phrasing, extra correct detail, or added caveats are fine.
- "correct": every key fact in the reference is present and nothing contradicts it.
- "partial": the answer gets some key facts but is missing at least one, or is
  vaguer than the reference on a decisive point.
- "incorrect": it contradicts the reference, states a different amount/decision,
  or dodges an answerable question.
- If expected_behavior is "refuse": "correct" ONLY if the assistant declines
  because the policies do not cover it, WITHOUT inventing policy content.
  If expected_behavior is "answer": a refusal is "incorrect" - the documents do
  cover it.
- Judge only the assistant's answer text; ignore any trace or citations shown."""

JUDGE_USER = """QUESTION
{question}

REFERENCE ANSWER
{reference}

EXPECTED BEHAVIOUR
{behavior}

ASSISTANT ANSWER
{answer}"""


JUDGE_SYSTEM_RUBRIC = """You grade a contract-analysis assistant's answer against a
reference answer, and also score it on the pack's four dimensions.

Return ONE JSON object:
{"verdict": "correct|partial|incorrect", "reason": "<one sentence>",
 "missing": ["<key fact the answer lacks>", ...],
 "scores": {"answer_correctness": <0-4>, "reasoning": <0-3>,
            "citations": <0-2>, "uncertainty": <0-1>}}

Rules:
- Compare DECISIONS, AMOUNTS, CONDITIONS, DEADLINES and HIERARCHY, not wording.
- "correct": every decisive fact in the reference is present and nothing contradicts it.
- "partial": some key facts present, at least one missing or vague on a decisive point.
- "incorrect": contradicts the reference, states a different amount/decision, or dodges.
- Score honestly and independently of the verdict:
  * answer_correctness (0-4): the substantive conclusion AND the arithmetic.
  * reasoning (0-3): the right hierarchy/exception/condition was applied.
  * citations (0-2): the cited document AND section are the ones that govern.
  * uncertainty (0-1): does not overclaim where the facts are insufficient.
- Judge only the answer text; ignore any trace."""

JUDGE_USER_RUBRIC = """QUESTION
{question}

REFERENCE ANSWER
{reference}

EXPECTED BEHAVIOUR
{behavior}

ASSISTANT ANSWER
{answer}"""

RUBRIC_MAX = {"answer_correctness": 4, "reasoning": 3, "citations": 2, "uncertainty": 1}


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------- setup
def prepare_index(pack: Path, data_dir: Path, reuse: bool):
    """Ingest the pack's documents into an isolated DATA_DIR; returns (api, timings)."""
    os.environ["DATA_DIR"] = str(data_dir)          # must precede app imports
    if not reuse and data_dir.exists():
        shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    from app.admin import AdminAPI
    from app.config import get_config

    get_config.__globals__["_CACHE"] = None         # re-read DATA_DIR
    api = AdminAPI()
    api.reconcile_stale_jobs() if hasattr(api, "reconcile_stale_jobs") else None

    manifest = json.loads(find_manifest(pack).read_text(encoding="utf-8"))
    already = {d["name"] for d in api.db.list_documents()}
    timings = {}
    for doc_name in manifest["documents"]:
        stem = Path(doc_name).stem
        if stem in already:
            log(f"  index: '{stem}' already present")
            timings[stem] = 0.0
            continue
        t0 = time.time()
        result = api.upload_document(pack / doc_name, name=stem, start_scan=False)
        dt = time.time() - t0
        timings[stem] = dt
        log(f"  index: '{stem}' -> {result.get('chunks')} chunks in {dt:.1f}s")
    return api, timings


# ------------------------------------------------------------------ judging
def judge(api, row: dict, answer: str) -> dict:
    """Grade one answer. Packs that ship a `scoring` block get the rubric variant,
    which returns the same verdict plus per-dimension points."""
    rubric = isinstance(row.get("scoring"), dict) and bool(row["scoring"])
    system, user = ((JUDGE_SYSTEM_RUBRIC, JUDGE_USER_RUBRIC) if rubric
                    else (JUDGE_SYSTEM, JUDGE_USER))
    obj, err = api.llm.chat_json(
        system,
        user.format(question=row["question"],
                    reference=row["ground_truth_answer"],
                    behavior=row["expected_behavior"],
                    answer=answer or "(empty answer)"),
        temperature=0.0,
    )
    if not obj:
        return {"verdict": "unjudged", "reason": f"judge failed: {err}", "missing": []}
    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in ("correct", "partial", "incorrect"):
        verdict = "unjudged"
    scores = obj.get("scores") if rubric else None
    if isinstance(scores, dict):
        scores = {k: max(0, min(RUBRIC_MAX[k], int(scores.get(k, 0) or 0)))
                  for k in RUBRIC_MAX}
    else:
        scores = None
    return {"verdict": verdict,
            "reason": str(obj.get("reason", ""))[:300],
            "missing": obj.get("missing") or [],
            "scores": scores}


# ---------------------------------------------------------------- citations
def citation_hits(answer: str, expected: list) -> tuple[int, int]:
    """(document-level hits, section-level hits) out of len(expected)."""
    if not expected:
        return 0, 0
    text = (answer or "").lower()
    doc_hits = sec_hits = 0
    for cite in expected:
        stem = Path(cite["document"]).stem.lower()          # 03_workplace_...
        short = stem.split("_", 1)[-1] if "_" in stem else stem
        section = (cite.get("section") or "").lower()
        # the answer may name the file, its stem, or the doc's own title
        doc_named = stem in text or short in text
        sec_named = bool(section) and section in text
        if doc_named or sec_named:
            doc_hits += 1
        if doc_named and sec_named:
            sec_hits += 1
    return doc_hits, sec_hits


def pct(a: int, b: int) -> float:
    return round(100.0 * a / b, 1) if b else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(DEFAULT_PACK))
    ap.add_argument("--data-dir", default=str(ROOT / "data_bench"))
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--only", default="", help="comma separated ids, e.g. MH-01,MH-19")
    ap.add_argument("--reuse", action="store_true", help="reuse an existing index")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="questions in flight at once (1 = sequential; >1 trades "
                         "clean per-question latency for wall time)")
    ap.add_argument("--index-only", action="store_true",
                    help="ingest the documents and stop (timing the index)")
    args = ap.parse_args()

    pack = Path(args.pack)
    rows = [json.loads(line) for line in
            find_questions(pack).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if args.only:
        wanted = {x.strip().upper() for x in args.only.split(",")}
        rows = [r for r in rows if r["id"].upper() in wanted]
    if args.limit:
        rows = rows[: args.limit]

    started = time.time()
    log(f"benchmark: {len(rows)} question(s) from {pack.name}")
    log("indexing (no contradiction scan, so timings reflect the QA path):")
    api, index_times = prepare_index(pack, Path(args.data_dir), args.reuse)
    index_wall = time.time() - started
    docs = api.db.list_documents()
    log(f"  corpus: {len(docs)} document(s), "
        f"{sum(d['chunk_count'] for d in docs)} chunks")
    if args.index_only:
        per_doc = ", ".join(f"{n}={t:.1f}s" for n, t in index_times.items())
        log(f"  index wall: {index_wall:.1f}s" + (f"  ({per_doc})" if per_doc else ""))
        return 0
    log("")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"benchmark_{stamp}.jsonl"

    results = []

    def run_one(i: int, row: dict) -> dict:
        """Ask one question, judge it, and return the result row.

        Runs in a worker thread when --workers > 1. Every shared piece is safe
        for that: the DB serialises writes under its own lock, the intent cache
        locks and writes atomically, the agent holds no per-question state, and
        the LLM client opens a connection per request.
        """
        t0 = time.time()
        try:
            res = api.ask(row["question"])
            answer = res.answer or ""
            payload = res.to_dict()
            error = payload.get("error")
            task = time.time() - t0
        except Exception as e:                       # noqa: BLE001
            answer, payload, task = f"(agent failed: {e})", {}, time.time() - t0
            error = str(e)

        grading = {"verdict": "unjudged", "reason": "not judged", "missing": []}
        if not args.no_judge:
            grading = judge(api, row, answer)
            task_total = time.time() - t0
        else:
            task_total = task

        doc_hits, sec_hits = citation_hits(answer, row["ground_truth_citations"])
        out = {
            "id": row["id"],
            "question": row["question"],
            "expected_behavior": row["expected_behavior"],
            "answer": answer,
            "error": error,
            "verdict": grading["verdict"],
            "judge_reason": grading["reason"],
            "judge_missing": grading["missing"],
            "rubric_scores": grading.get("scores"),
            "rubric_points": (sum(grading["scores"].values())
                              if grading.get("scores") else None),
            "tool_calls": payload.get("tool_calls"),
            # turns = model calls = the thing AGENT_MAX_STEPS caps; step_entries is
            # thought + one entry per tool call + final, so a parallel turn adds several
            "turns": payload.get("turns"),
            "step_entries": len(payload.get("steps") or []),   # entries, not turns
            "forced_final": any(isinstance(s, dict) and
                                str(s.get("note", "")).startswith("forced")
                                for s in (payload.get("steps") or [])),
            "parallel_turns": payload.get("parallel_turns"),
            "elapsed_s": round(payload.get("elapsed_s") or 0, 2),
            "wall_s": round(task, 1),
            "wall_with_judge_s": round(task_total, 1),
            "citations": payload.get("citations") or [],
            "citation_warnings": payload.get("citation_warnings") or [],
            "expected_citation_docs": len(row["ground_truth_citations"]),
            "citation_doc_hits": doc_hits,
            "citation_section_hits": sec_hits,
        }
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
        log(f"  [{i}/{len(rows)}] {row['id']}  {grading['verdict']:<9} "
            f"{task_total:5.1f}s  tools={payload.get('tool_calls')}  "
            f"cite {doc_hits}/{len(row['ground_truth_citations'])}  "
            f"{(grading['reason'] or '')[:70]}")
        return out

    if args.workers > 1:
        log(f"  NOTE: {args.workers} questions in flight at once - per-question "
            f"timings include contention, so compare ACCURACY, not latency.")
        with ThreadPoolExecutor(max_workers=min(args.workers, len(rows)),
                                thread_name_prefix="bench") as pool:
            futures = {pool.submit(run_one, i, row): row["id"]
                       for i, row in enumerate(rows, 1)}
            for fut in as_completed(futures):
                results.append(fut.result())          # errors surface here
        order = {row["id"]: i for i, row in enumerate(rows)}
        results.sort(key=lambda r: order.get(r["id"], 0))
    else:
        for i, row in enumerate(rows, 1):
            results.append(run_one(i, row))

    # ------------------------------------------------------------------ scores
    total = len(results)
    correct = sum(1 for r in results if r["verdict"] == "correct")
    partial = sum(1 for r in results if r["verdict"] == "partial")
    incorrect = sum(1 for r in results if r["verdict"] == "incorrect")
    unjudged = sum(1 for r in results if r["verdict"] == "unjudged")

    refusals = [r for r in results if str(r["expected_behavior"]).lower().startswith("refuse")]
    answerable = [r for r in results if r not in refusals]
    exp_cites = sum(r["expected_citation_docs"] for r in results)
    doc_hits = sum(r["citation_doc_hits"] for r in results)
    sec_hits = sum(r["citation_section_hits"] for r in results)

    forced = [r for r in results if r.get("forced_final")]
    step_entries = [r["step_entries"] for r in results
                    if r.get("step_entries") is not None]
    turns = [r["turns"] for r in results if r.get("turns") is not None]

    scored = [r for r in results if r.get("rubric_points") is not None]
    rubric_summary = None
    if scored:
        max_points = sum(RUBRIC_MAX.values())
        by_dim = {k: round(statistics.mean([r["rubric_scores"][k] for r in scored]), 2)
                  for k in RUBRIC_MAX}
        rubric_summary = {
            "cases": len(scored),
            "max_points": max_points,
            "mean_points": round(statistics.mean([r["rubric_points"] for r in scored]), 2),
            "mean_pct": round(100.0 * statistics.mean(
                [r["rubric_points"] / max_points for r in scored]), 1),
            "by_dimension": by_dim,
            "by_dimension_max": RUBRIC_MAX,
        }

    times = [r["wall_s"] for r in results]
    walls = [r["wall_with_judge_s"] for r in results]
    summary = {
        "benchmark": pack.name,
        "workers": args.workers,
        "timing_note": ("sequential: per-question times are single-request latency"
                        if args.workers <= 1 else
                        f"{args.workers} concurrent: times include contention, not comparable"),
        "questions": total,
        "index_wall_s": round(index_wall, 1),
        "index_per_doc_s": {k: round(v, 1) for k, v in index_times.items()},
        "answer_accuracy_pct": pct(correct, total),
        "strict_or_partial_pct": pct(correct + partial, total),
        "verdicts": {"correct": correct, "partial": partial,
                     "incorrect": incorrect, "unjudged": unjudged},
        "answerable": {"n": len(answerable),
                       "correct": sum(1 for r in answerable if r["verdict"] == "correct"),
                       "accuracy_pct": pct(sum(1 for r in answerable
                                               if r["verdict"] == "correct"),
                                           len(answerable))},
        "refusals": {"n": len(refusals),
                     "correct": sum(1 for r in refusals if r["verdict"] == "correct"),
                     "accuracy_pct": pct(sum(1 for r in refusals
                                             if r["verdict"] == "correct"),
                                         len(refusals))},
        "citation_recall": {"expected": exp_cites,
                            "doc_level_pct": pct(doc_hits, exp_cites),
                            "section_level_pct": pct(sec_hits, exp_cites)},
        "citation_warnings_total": sum(len(r["citation_warnings"]) for r in results),
        "rubric": rubric_summary,
        "agent_steps": {
            "max_steps_env": os.getenv("AGENT_MAX_STEPS", ""),
            "forced_final": len(forced),
            "forced_ids": [r["id"] for r in forced],
            "max_entries": max(step_entries) if step_entries else None,
            "mean_entries": round(statistics.mean(step_entries), 1) if step_entries else None,
            "max_turns": max(turns) if turns else None,
            "mean_turns": round(statistics.mean(turns), 1) if turns else None,
        },
        "time_s": {"total_wall": round(time.time() - started, 1),
                   "qa_wall": round(sum(times), 1),
                   "mean": round(statistics.mean(times), 1) if times else 0,
                   "p50": round(statistics.median(times), 1) if times else 0,
                   "p95": round(sorted(times)[max(0, math.ceil(0.95 * len(times)) - 1)], 1)
                          if times else 0,
                   "max": round(max(times), 1) if times else 0,
                   "min": round(min(times), 1) if times else 0},
        "rows": str(jsonl_path),
    }
    (out_dir / f"benchmark_{stamp}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    log("\n" + "=" * 78)
    log(f"BENCHMARK {summary['benchmark']}  ({total} questions)")
    log("=" * 78)
    log(f"  answer accuracy (correct/total)   {correct}/{total}  "
        f"({summary['answer_accuracy_pct']}%)")
    log(f"  correct or partial                {correct + partial}/{total}  "
        f"({summary['strict_or_partial_pct']}%)")
    log(f"  verdicts                          correct {correct}, partial {partial}, "
        f"incorrect {incorrect}, unjudged {unjudged}")
    log(f"  answerable questions              {summary['answerable']['correct']}/"
        f"{len(answerable)}  ({summary['answerable']['accuracy_pct']}%)")
    if refusals:
        log(f"  refusals                          {summary['refusals']['correct']}/"
            f"{len(refusals)}  ({summary['refusals']['accuracy_pct']}%)")
    else:
        log("  refusals                          n/a (no refusal cases in this slice)")
    log(f"  citation recall (document)        {doc_hits}/{exp_cites}  "
        f"({summary['citation_recall']['doc_level_pct']}%)")
    log(f"  citation recall (doc + section)   {sec_hits}/{exp_cites}  "
        f"({summary['citation_recall']['section_level_pct']}%)")
    log(f"  citation warnings                 {summary['citation_warnings_total']}")
    steps = summary.get("agent_steps") or {}
    if steps:
        log(f"  agent turns                       max {steps.get('max_turns')}, "
            f"mean {steps.get('mean_turns')}, forced finals {steps['forced_final']}"
            + (f"  (cap AGENT_MAX_STEPS={steps['max_steps_env']})" if steps['max_steps_env'] else ""))
        log(f"  step entries                      max {steps['max_entries']}, "
            f"mean {steps['mean_entries']}  (thoughts + tool calls + final; not the cap)")
    rubric = summary.get("rubric")
    if rubric:
        log(f"  rubric (judge-scored)             {rubric['mean_points']}/"
            f"{rubric['max_points']}  ({rubric['mean_pct']}%)  over {rubric['cases']} cases")
        dims = ", ".join(f"{k} {v}/{rubric['by_dimension_max'][k]}"
                         for k, v in rubric["by_dimension"].items())
        log(f"    dimensions                      {dims}")
    t = summary["time_s"]
    log(f"  time: total wall {t['total_wall']}s (index {summary['index_wall_s']}s + "
        f"QA {t['qa_wall']}s)")
    log(f"        per question: mean {t['mean']}s, p50 {t['p50']}s, "
        f"p95 {t['p95']}s, max {t['max']}s, min {t['min']}s")
    log(f"  rows: {jsonl_path}")
    log(f"  summary: {out_dir / f'benchmark_{stamp}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
