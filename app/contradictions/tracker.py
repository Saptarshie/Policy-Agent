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
        top_k: int = 4,
        max_pairs_per_chunk: int = 3,
        on_alert: Optional[AlertFn] = None,
    ) -> None:
        self.db = db
        self.rag = rag
        self.gate = gate
        self.verifier = verifier
        self.cross_doc_only = cross_doc_only
        self.top_k = max(1, top_k)
        self.max_pairs_per_chunk = max(1, max_pairs_per_chunk)
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

        for i, chunk in enumerate(chunks):
            # ---- stage 1: RAG retrieval of candidate pairs
            try:
                hits = self.rag.search(
                    chunk["text"], top_k=self.top_k,
                    exclude_doc_ids=[doc_id] if self.cross_doc_only else None,
                    # candidate pairs are "semantically adjacent passages";
                    # keyword overlap would surface repeated clauses instead of
                    # genuine conflicts, so the cascade stays on vectors
                    mode="vector",
                )
            except Exception as e:
                log_fn("  [scan] embedding search failed on chunk %s: %s",
                       chunk["id"], e)
                hits = []

            escalated = 0
            for hit in hits:
                if hit.doc_id == doc_id and self.cross_doc_only:
                    continue
                if escalated >= self.max_pairs_per_chunk:
                    break
                if self.db.contradiction_pair_exists(chunk["id"], hit.chunk_id):
                    continue

                # ---- stage 2: decision model gate (heuristic / remote /
                #      hybrid per DECISION_MODE; runs BEFORE any LLM call)
                decision = self.gate.evaluate(chunk["text"], hit.text)
                if not decision.passed:
                    continue

                # ---- stage 3: LLM verification (ONLY for gate-passing pairs)
                escalated += 1
                log_fn("  [scan] gate PASSED (score %.2f) %s | %s  -> escalating "
                       "to LLM verifier", decision.score, doc_name, hit.doc_name)
                verdict = self.verifier.verify(
                    doc_a=doc_name, sec_a=chunk.get("section") or "",
                    text_a=chunk["text"],
                    doc_b=hit.doc_name, sec_b=hit.section or "",
                    text_b=hit.text,
                    gate_signals=decision.signals,
                )

                # ---- stage 4: flag to the admin (with citations)
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
                        # the pair is already flagged in the opposite scan
                        # direction - do not count or re-alert it
                        log_fn("  [scan] pair already flagged; duplicate flag "
                               "suppressed: %s | %s", doc_name, hit.doc_name)
                elif not verdict.verified and \
                        decision.score >= self.gate.threshold + 0.15:
                    # LLM unavailable/unparseable but strong heuristic signal:
                    # record as UNVERIFIED for admin review (never drop).
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
                            f"({verdict.error}); flagged by the heuristic gate "
                            "alone. Heuristic signals: "
                            + "; ".join(decision.signals)
                        ),
                        points=[],
                    )
                    if cid:
                        flagged += 1
                        self._alert(cid, doc_name, hit.doc_name,
                                    "unverified heuristic flag (LLM unavailable)")
                        log_fn("  [scan] UNVERIFIED flag #%s: %s vs %s",
                               cid, doc_name, hit.doc_name)
                    else:
                        log_fn("  [scan] pair already flagged; duplicate "
                               "unverified flag suppressed: %s | %s",
                               doc_name, hit.doc_name)
                else:
                    log_fn("  [scan] verifier rejected the pair "
                           "(gate score %.2f)", decision.score)

            self.db.update_job(job_id, done=i + 1,
                               stage=f"scanned {i + 1}/{len(chunks)} chunks")

        self.db.update_job(job_id, status="done", stage="done")
        log_fn("[scan] finished '%s': %d contradiction(s) flagged", doc_name, flagged)
        return {"flagged": flagged, "job_id": job_id}

    def _alert(self, cid: int, doc_a: str, doc_b: str, explanation: str) -> None:
        if self.on_alert:
            try:
                self.on_alert({"id": cid, "doc_a": doc_a, "doc_b": doc_b,
                               "explanation": explanation})
            except Exception:
                log.exception("alert callback failed")
