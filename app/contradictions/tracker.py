"""Stage 1 + orchestration of the cascade: the background tracker.

For a freshly ingested document:

  stage 1  RAG retrieval        - for each chunk, retrieve the most similar
                                 passages of the OTHER documents (cheap,
                                 embeddings only);
  stage 2  DECISION MODEL       - the gate scores every candidate pair and
                                 only pairs >= threshold survive. Pluggable
                                 per DECISION_MODE: in-code JEV heuristic
                                 (default, zero LLM), a remote decision-model
                                 endpoint, or hybrid (heuristic + remote
                                 refinement of borderline pairs);
  stage 3  LLM verifier         - escalated ONLY for gate-passing pairs;
  stage 4  admin flagging       - verified contradictions are written with
                                 full citations (doc + section + quoted text)
                                 and a human-readable explanation of WHY they
                                 conflict.

Pairs whose verdict could not be verified (LLM unavailable / unparseable)
but whose gate score is far above threshold are recorded as UNVERIFIED so
an admin can review them - nothing is silently dropped.

A `scan` job row tracks progress chunk-by-chunk (visible in CLI/web UI).
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, List, Optional

from ..db import Database
from ..rag import RAGEngine
from .gate import CascadeGate, GateDecision, JEVStyleGate
from .verify import ContradictionVerifier

log = logging.getLogger(__name__)

AlertFn = Callable[[dict], None]


class ContradictionTracker:
    def __init__(
        self,
        db: Database,
        rag: RAGEngine,
        gate,  # CascadeGate (any object with .evaluate()/.threshold, so a
               # bare JEVStyleGate also works - see tests)
        verifier: ContradictionVerifier,
        cross_doc_only: bool = True,
        min_similarity: float = 0.65,
        max_candidates: int = 200,
        workers: int = 4,
        on_alert: Optional[AlertFn] = None,
    ) -> None:
        self.db = db
        self.rag = rag
        self.gate = gate
        self.verifier = verifier
        self.cross_doc_only = cross_doc_only
        # candidate generation is similarity-gated, not count-gated: a conflict
        # can rank 41st among 60 similar chunks, so a top-k cut is the wrong
        # instrument. The only count left is a safety cap on total escalations.
        self.min_similarity = float(min_similarity)
        self.max_candidates = int(max_candidates)
        # the retrieval pool is NOT the pair cap: a small cap on escalations must
        # not also hide candidates from the pool (that would make the cap log
        # unreachable and the floor meaningless)
        self.search_pool = max(self.max_candidates, 64)
        #: concurrent verifier requests (each is one HTTP call)
        self.workers = max(1, int(workers))
        self.on_alert = on_alert

    # ------------------------------------------------------------------ public

    def scan_document_async(self, doc_id: int) -> Optional[int]:
        """Start a background scan; returns the job id immediately."""
        doc = self.db.get_document(doc_id)
        if not doc or doc.get("status") != "ready":
            log.warning("scan skipped: doc %s not ready", doc_id)
            return None
        chunks = self.db.list_chunks(doc_id)
        job_id = self.db.create_job("scan", doc_id, doc["name"], total=len(chunks))
        t = threading.Thread(target=self._safe_scan, args=(doc_id, job_id),
                             daemon=True, name=f"contradiction-scan-{doc_id}")
        t.start()
        return job_id

    def scan_document(self, doc_id: int, log_fn: Optional[Callable[..., None]] = None
                      ) -> dict:
        """Run the scan synchronously; returns {'flagged': n, 'job_id': id}."""
        doc = self.db.get_document(doc_id)
        if not doc:
            raise ValueError(f"unknown document id {doc_id}")
        chunks = self.db.list_chunks(doc_id)
        job_id = self.db.create_job("scan", doc_id, doc["name"], total=len(chunks))
        return self._scan(doc_id, job_id, log_fn=log_fn or (lambda *a, **k: None))

    # ----------------------------------------------------------------- private

    def _safe_scan(self, doc_id: int, job_id: int) -> None:
        try:
            self._scan(doc_id, job_id, log_fn=log.info)
        except Exception:
            log.exception("contradiction scan failed for doc %s", doc_id)

    def _scan(self, doc_id: int, job_id: int, log_fn=print) -> dict:
        doc = self.db.get_document(doc_id)
        if not doc:
            return {"flagged": 0, "job_id": job_id}
        doc_name = doc["name"]
        chunks = self.db.list_chunks(doc_id)
        flagged = 0

        # ---- stage 1: candidate pairs by SIMILARITY, not by count.
        # A conflict can rank 41st among 60 passages that are all similar; a
        # top-k cut would never show it to the gate, so the floor decides and
        # only a global cap guards total cost. Collected for the WHOLE document
        # so that batches fill across chunk boundaries instead of ending every
        # chunk with a half-empty request.
        candidates = []            # (chunk, hit) in scan order
        considered = 0
        capped = False
        for chunk in chunks:
            try:
                hits = self.rag.find_similar_chunks(
                    chunk["id"],
                    min_score=self.min_similarity,
                    max_candidates=self.search_pool,
                    exclude_doc_ids=[doc_id] if self.cross_doc_only else None,
                )
            except Exception as e:
                log_fn("  [scan] embedding search failed on chunk %s: %s",
                       chunk["id"], e)
                hits = []

            for hit in hits:
                if hit.chunk_id == chunk["id"]:
                    continue          # a passage cannot contradict itself
                if self.cross_doc_only and hit.doc_id == doc_id:
                    continue          # caller restricted the scan to cross-document
                                      # pairs; intra-document pairs are the default
                if considered >= self.max_candidates:
                    if not capped:
                        capped = True
                        log_fn("  [scan] candidate cap reached (%d pairs); raise "
                               "SCAN_MAX_CANDIDATES to look at the tail",
                               self.max_candidates)
                    break
                considered += 1
                if self.db.contradiction_pair_exists(chunk["id"], hit.chunk_id):
                    continue
                candidates.append((chunk, hit))

        if not candidates:
            log_fn("  [scan] no candidate pairs within %.2f of any passage in '%s'",
                   self.min_similarity, doc_name)
            self.db.update_job(job_id, status="done", stage="done")
            log_fn("[scan] finished '%s': 0 contradiction(s) flagged", doc_name)
            return {"flagged": 0, "job_id": job_id}

        self.db.update_job(job_id, stage=f"{len(candidates)} candidate pair(s)")

        # ---- stage 2: the decision model judges every candidate.
        # Batched (many pairs per request: the passages are the shared prefix)
        # and concurrent across batches; a pair near the threshold is re-asked
        # alone, so batching cannot quietly move a verdict.
        pairs = [(c["text"], h.text) for c, h in candidates]
        evaluate_many = getattr(self.gate, "evaluate_many", None)
        if evaluate_many is not None:
            decisions = evaluate_many(pairs)
        else:                              # custom gate without batching support
            decisions = [self.gate.evaluate(a, b) for a, b in pairs]

        if decisions and all(d.decided_by == "unavailable" for d in decisions):
            # no judge: say so instead of reporting "no conflicts"
            log_fn("  [scan] no decision model available (%s) - candidate pairs "
                   "cannot be judged; this scan is INCOMPLETE, not clean",
                   getattr(decisions[0], "error", None) or "?")
            self.db.update_job(job_id, status="done", stage="done")
            log_fn("  [scan] finished without a decision model: 0 flags means "
                   "'not examined', not 'no contradictions'")
            return {"flagged": 0, "job_id": job_id}

        survivors = [(c, h, d) for (c, h), d in zip(candidates, decisions)
                     if d.passed and d.decided_by != "unavailable"]
        for chunk, hit, decision in survivors:
            log_fn("  [scan] gate PASSED (score %.2f) %s | %s  -> escalating "
                   "to LLM verifier", decision.score, doc_name, hit.doc_name)

        # ---- stage 3: verification, concurrently (each call is one HTTP request)
        verdicts = self._verify_all(survivors, doc_name)
        self.db.update_job(job_id, stage=f"verified {len(verdicts)} pair(s)")

        # ---- stage 4: record, serially, in scan order (one SQLite connection)
        for (chunk, hit, decision), verdict in zip(survivors, verdicts):
            if verdict.verified and verdict.is_contradiction:
                cid = self.db.add_contradiction(
                    chunk_a_id=chunk["id"], doc_a=doc_name,
                    section_a=chunk.get("section") or "",
                    text_a=chunk["text"],
                    chunk_b_id=hit.chunk_id, doc_b=hit.doc_name,
                    section_b=hit.section or "", text_b=hit.text,
                    gate_score=decision.score, gate_signals=decision.signals,
                    verified=True, confidence=verdict.confidence,
                    conflict_type=verdict.conflict_type,
                    explanation=verdict.explanation, points=verdict.points,
                )
                if cid:
                    flagged += 1
                    self._alert(cid, doc_name, hit.doc_name, verdict.explanation)
                    log_fn("  [scan] CONTRADICTION #%s: %s vs %s - %s",
                           cid, doc_name, hit.doc_name, verdict.explanation)
                else:
                    # the pair is already flagged in the opposite scan direction
                    log_fn("  [scan] pair already flagged; duplicate flag "
                           "suppressed: %s | %s", doc_name, hit.doc_name)
            elif not verdict.verified and \
                    decision.score >= self.gate.threshold + 0.15:
                # the verifier could not be run but the gate is emphatic: record
                # it for review rather than dropping it silently
                cid = self.db.add_contradiction(
                    chunk_a_id=chunk["id"], doc_a=doc_name,
                    section_a=chunk.get("section") or "",
                    text_a=chunk["text"],
                    chunk_b_id=hit.chunk_id, doc_b=hit.doc_name,
                    section_b=hit.section or "", text_b=hit.text,
                    gate_score=decision.score, gate_signals=decision.signals,
                    verified=False, confidence=0.0, conflict_type="unverified",
                    explanation=(
                        "UNVERIFIED - the LLM verifier could not be run "
                        f"({verdict.error}); flagged by the decision model alone. "
                        "Signals: " + "; ".join(decision.signals)
                    ),
                    points=[],
                )
                if cid:
                    flagged += 1
                    self._alert(cid, doc_name, hit.doc_name,
                                "unverified decision-model flag")
                    log_fn("  [scan] UNVERIFIED flag #%s: %s vs %s",
                           cid, doc_name, hit.doc_name)
                else:
                    log_fn("  [scan] pair already flagged; duplicate "
                           "unverified flag suppressed: %s | %s",
                           doc_name, hit.doc_name)
            else:
                log_fn("  [scan] verifier rejected the pair (gate score %.2f)",
                       decision.score)

        self.db.update_job(job_id, status="done", stage="done",
                           done=len(chunks))
        log_fn("[scan] finished '%s': %d contradiction(s) flagged", doc_name, flagged)
        return {"flagged": flagged, "job_id": job_id}

    def _verify_all(self, survivors, doc_name: str):
        """Verify survivors concurrently; order preserved (writes stay serial)."""
        def one(item):
            chunk, hit, decision = item
            return self.verifier.verify(
                doc_a=doc_name, sec_a=chunk.get("section") or "",
                text_a=chunk["text"],
                doc_b=hit.doc_name, sec_b=hit.section or "",
                text_b=hit.text,
                gate_signals=decision.signals,
            )

        if not survivors:
            return []
        if self.workers <= 1 or len(survivors) == 1:
            return [one(s) for s in survivors]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(self.workers, len(survivors))) as pool:
            return list(pool.map(one, survivors))

    def _alert(self, cid: int, doc_a: str, doc_b: str, explanation: str) -> None:
        if self.on_alert:
            try:
                self.on_alert({"id": cid, "doc_a": doc_a, "doc_b": doc_b,
                               "explanation": explanation})
            except Exception:
                log.exception("alert callback failed")
