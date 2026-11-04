"""Does parallel window parsing change what gets stored, byte for byte?

The claim in the docs is "performance only": chunk text must be identical whether
the windows are parsed one at a time (the original chain) or concurrently. This
ingests the same document twice through the real model with different settings
and compares the stored chunk text and the citation anchors.

    python eval/compare_parse_modes.py [--doc PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DOC = (ROOT / "Benchmark" / "hr_rag_policy_pack_and_benchmark"
               / "03_WORKPLACE_EXPENSE_AND_REMOTE_WORK_POLICY.md")


def ingest_once(doc: Path, workers: int, effort: str, tmp: Path) -> dict:
    """Ingest into a throwaway DB; returns chunks, texts and the summary."""
    os.environ["DATA_DIR"] = str(tmp)
    from app import config as config_mod
    config_mod._CACHE = None
    for k, v in (("INGEST_PARALLEL_WINDOWS", str(workers)),
                 ("INGEST_REASONING_EFFORT", effort)):
        os.environ[k] = v
    config_mod._CACHE = None

    from app.admin import AdminAPI
    api = AdminAPI()
    t0 = time.time()
    res = api.db  # keep the handle
    result = api.upload_document(doc, start_scan=False)      # blocking ingest
    wall = time.time() - t0
    doc_id = result["doc_id"]
    row = res.get_document(doc_id)
    chunks = res.list_chunks(doc_id)
    return {
        "seconds": wall,
        "status": row["status"],
        "summary": row["summary"] or "",
        "chunks": [(c["chunk_index"], c["section"], c["text"]) for c in chunks],
    }


def fingerprint(chunks) -> str:
    h = hashlib.sha256()
    for idx, section, text in chunks:
        h.update(f"{idx}\x1f{section}\x1f{text}\x1e".encode("utf-8"))
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=str(DEFAULT_DOC))
    args = ap.parse_args()
    doc = Path(args.doc)

    runs = {}
    for label, workers, effort in (("sequential + reasoning", 1, ""),
                                   ("parallel + reasoning off", 4, "off")):
        tmp = Path(tempfile.mkdtemp(prefix=f"parse_{workers}_"))
        try:
            runs[label] = ingest_once(doc, workers, effort, tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        r = runs[label]
        print(f"{label:26} {r['seconds']:6.1f}s  {len(r['chunks']):3} chunks  "
              f"sha={fingerprint(r['chunks'])[:16]}  status={r['status']}")

    a, b = list(runs.values())
    print()
    print(f"  chunk count identical : {len(a['chunks']) == len(b['chunks'])}")
    print(f"  chunk text identical  : {fingerprint(a['chunks']) == fingerprint(b['chunks'])}")
    print(f"  wall time             : {a['seconds']:.1f}s -> {b['seconds']:.1f}s "
          f"({a['seconds'] / max(b['seconds'], 0.1):.1f}x)")
    if fingerprint(a["chunks"]) != fingerprint(b["chunks"]):
        for (ia, sa, ta), (ib, sb, tb) in zip(a["chunks"], b["chunks"]):
            if ta != tb or sa != sb:
                print(f"  first difference at chunk {ia}: section {sa!r} vs {sb!r}")
                print(f"    A: {ta[:120]!r}")
                print(f"    B: {tb[:120]!r}")
                break
    # the summary is metadata: it legitimately differs (one merge call vs a chain)
    print(f"  summary A: {a['summary'][:120]!r}")
    print(f"  summary B: {b['summary'][:120]!r}")
    return 0 if fingerprint(a["chunks"]) == fingerprint(b["chunks"]) else 1


if __name__ == "__main__":
    sys.exit(main())
