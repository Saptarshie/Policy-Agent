"""Where does ingestion time actually go? Time the window-parse call.

Ingestion makes exactly two kinds of network call per document:
  1. one chat_json per rolling window (~6KB each), sequential and chained by the
     rolling summary,
  2. one batched embedding call for all chunks.
Everything else (chunking, SQLite writes) is local. This measures (1) with and
without reasoning, and reports the token usage, so the optimisation targets the
right thing.

    python eval/profile_ingest.py [--doc PATH]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chunker import split_into_windows          # noqa: E402
from app.config import get_config                   # noqa: E402
from app.llm import OpenAICompatClient              # noqa: E402
from app.prompts import WINDOW_PARSER_SYSTEM, WINDOW_PARSER_USER  # noqa: E402

DEFAULT_DOC = (ROOT / "Benchmark" / "hr_rag_policy_pack_and_benchmark"
               / "03_WORKPLACE_EXPENSE_AND_REMOTE_WORK_POLICY.md")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=str(DEFAULT_DOC))
    ap.add_argument("--effort", default="", help="e.g. off, low, medium")
    ap.add_argument("--runs", type=int, default=2)
    args = ap.parse_args()

    cfg = get_config()
    llm = OpenAICompatClient(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key,
                             model=cfg.llm_model,
                             max_output_tokens=cfg.max_output_tokens,
                             timeout=(20, 600))
    text = Path(args.doc).read_text(encoding="utf-8")
    windows = split_into_windows(text, cfg.ingest_window_chars)
    print(f"{Path(args.doc).name}: {len(text)} chars -> {len(windows)} windows "
          f"({cfg.ingest_window_chars} chars each)")
    print(f"model={cfg.llm_model}  reasoning_effort={args.effort or '(provider default)'}  "
          f"max_tokens={cfg.max_output_tokens}\n")

    times, tokens = [], []
    for run in range(args.runs):
        for i, window in enumerate(windows):
            user = WINDOW_PARSER_USER.format(
                prev_summary="(empty)", i=i + 1, n=len(windows),
                doc_name=Path(args.doc).stem, window=window)
            t0 = time.time()
            obj, err = llm.chat_json(
                WINDOW_PARSER_SYSTEM, user,
                reasoning_effort=(args.effort or None))
            dt = time.time() - t0
            times.append(dt)
            usage = getattr(llm, "last_usage", None)
            if usage:
                tokens.append(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            ok = "ok" if obj else f"FAILED ({err})"
            print(f"  run {run + 1} window {i + 1}/{len(windows)}: {dt:6.1f}s  {ok}"
                  + (f"  completion_tokens={tokens[-1]}" if tokens else ""))
    if times:
        print(f"\n  per call: mean {statistics.mean(times):.1f}s, "
              f"min {min(times):.1f}s, max {max(times):.1f}s  (n={len(times)})")
        print(f"  whole document would take: {statistics.mean(times) * len(windows):.1f}s "
              f"(sequential) vs {statistics.mean(times):.1f}s (windows in parallel)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
