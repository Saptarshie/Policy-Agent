"""Measure the contradiction scan: serial per-pair vs batched vs batched+threads.

Runs the FULL corpus three times against fresh copies of the database (so
`contradiction_pair_exists` never skips a pair), counting decision-model requests
and verifier requests, and checking that all three configurations flag the same
pairs.

    python eval/measure_scan.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SRC = Path(os.getenv("DATA_DIR", "./data")) / "policy_rag.db"
CONFIGS = [
    ("serial per-pair", dict(batch=1, workers=1)),
    ("batched (6)", dict(batch=6, workers=1)),
    ("batched + 4 threads", dict(batch=6, workers=4)),
]


def run(label: str, batch: int, workers: int, outdir: Path) -> dict:
    os.environ["DECISION_BATCH_SIZE"] = str(batch)
    os.environ["SCAN_WORKERS"] = str(workers)
    os.environ["DATA_DIR"] = str(outdir)

    import importlib
    import app.config as config
    importlib.reload(config)
    config._CACHE = None

    from app.admin import AdminAPI
    api = AdminAPI()

    counters = {"batch": 0, "single": 0, "verify": 0, "pairs_batched": 0}
    gate = api.tracker.gate
    remote = gate.remote

    orig_many, orig_one = remote.score_many, remote.score
    orig_verify = api.tracker.verifier.verify

    def score_many(pairs):
        counters["batch"] += 1
        counters["pairs_batched"] += len(pairs)
        return orig_many(pairs)

    def score(a, b, signals):
        counters["single"] += 1
        return orig_one(a, b, signals)

    def verify(**kw):
        counters["verify"] += 1
        return orig_verify(**kw)

    remote.score_many, remote.score = score_many, score
    api.tracker.verifier.verify = verify

    t0 = time.time()
    flagged = []
    for doc in api.db.list_documents():
        api.tracker.scan_document(doc["id"], log_fn=lambda *a, **k: None)
    elapsed = time.time() - t0
    for c in api.db.list_contradictions():
        flagged.append(f"{c['doc_a']}::{c['section_a'][:24]} <-> {c['doc_b']}::{c['section_b'][:24]}")
    requests = counters["batch"] + counters["single"]
    return {
        "config": label,
        "seconds": round(elapsed, 1),
        "requests_to_decision_model": requests,
        "batched_requests": counters["batch"],
        "single_requests": counters["single"],
        "pairs_judged": counters["pairs_batched"] + counters["single"],
        "verifier_calls": counters["verify"],
        "flagged": len(flagged),
        "flags": sorted(flagged),
    }


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"no database at {SRC}")
    results = []
    for label, cfg in CONFIGS:
        tmp = Path(os.getenv("BH_AGENT_WORKSPACE", ".")) / "_scan_measure" / label.replace(" ", "_").replace("+", "p")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC, tmp / "policy_rag.db")
        print(f"--- {label} ---", flush=True)
        res = run(label, cfg["batch"], cfg["workers"], tmp)
        print(json.dumps({k: v for k, v in res.items() if k != "flags"}, indent=2), flush=True)
        results.append(res)

    base = results[0]["seconds"]
    print("\nsummary")
    print(f"{'configuration':24}{'wall':>8}{'speedup':>9}{'requests':>10}{'verifier':>10}{'flags':>7}")
    for r in results:
        print(f"{r['config']:24}{r['seconds']:8.1f}{base / r['seconds']:8.1f}x"
              f"{r['requests_to_decision_model']:10}{r['verifier_calls']:10}{r['flagged']:7}")

    same = all(r["flags"] == results[0]["flags"] for r in results)
    print(f"\nsame flags in all configurations: {same}")
    if not same:
        for r in results:
            print(f"\n[{r['config']}]")
            for f in r["flags"]:
                print("   ", f)


if __name__ == "__main__":
    main()
