"""RAG search over the chunk store.

Two retrievers, fused:

* **vector** - cosine similarity over stored embeddings. Good at paraphrase
  ("time off" ~ "leave"), blind to exact tokens such as a clause number, a tier
  name or a currency amount.
* **keyword** - SQLite FTS5 / BM25. Exact at those tokens, blind to paraphrase.

``mode="hybrid"`` fuses their rankings with **Reciprocal Rank Fusion**:
``score(d) = sum_r w_r / (k + rank_r(d))``. RRF is used instead of a weighted
sum of raw scores because cosine similarity and BM25 are not on a comparable
scale, and per-query normalisation is unstable when one retriever returns a
single strong hit. RRF only needs ranks, so it is parameter-light (k=60) and
cannot be dominated by whichever retriever produces larger absolute numbers.

The contradiction cascade deliberately stays on ``mode="vector"``: it asks
"which passages are semantically adjacent", where keyword overlap would surface
the same clause twice rather than a genuine conflict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .db import Database

log = logging.getLogger(__name__)

MODES = ("vector", "keyword", "hybrid")


@dataclass
class Hit:
    chunk_id: int
    doc_id: int
    doc_name: str
    section: str
    breadcrumb: str
    text: str
    score: float
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None


class RAGEngine:
    def __init__(self, db: Database, embedder, mode: str = "vector",
                 rrf_k: int = 60, vector_weight: float = 1.0,
                 keyword_weight: float = 1.0) -> None:
        self.db = db
        self.embedder = embedder
        self.mode = (mode or "vector").strip().lower()
        self.rrf_k = max(1, int(rrf_k))
        self.vector_weight = float(vector_weight)
        self.keyword_weight = float(keyword_weight)

    def search(self, query: str, top_k: Optional[int] = None,
               doc_ids: Optional[Iterable[int]] = None,
               exclude_doc_ids: Optional[Iterable[int]] = None,
               mode: Optional[str] = None,
               min_score: Optional[float] = None,
               max_candidates: Optional[int] = None) -> List[Hit]:
        """Return passages for *query* under ONE of two gating rules.

        ``top_k``      return the best k - a count-based cut, right when the
                       consumer wants a bounded amount of evidence.
        ``min_score``  return EVERYTHING at or above a similarity floor - a
                       relevance-based cut, right when a count-based cut could
                       drop the one passage that matters. Used by the
                       contradiction scanner: a conflict can rank 41st among 60
                       equally-similar chunks, and top_k=4 would never see it.

        ``max_candidates`` is a safety cap for the threshold mode (cost guard,
        not a relevance decision); when it truncates, a log line says so.
        """
        if not query or not query.strip():
            return []
        threshold_mode = min_score is not None
        top_k = int(top_k) if top_k else 6
        mode = (mode or self.mode).strip().lower()
        if mode not in MODES:
            mode = "vector"
        if threshold_mode:
            min_score = float(min_score)
            if mode != "vector":
                # an RRF score is a rank artefact, not a similarity: comparing it
                # to a cosine floor would be meaningless
                log.warning("min_score is a cosine floor; forcing mode=vector "
                            "(requested mode=%s)", mode)
                mode = "vector"
            # in threshold mode the pool IS the candidate set: it has to be
            # large enough that the floor decides, not the pool size
            pool = int(max_candidates or 200)
            top_k = pool

        # one pool for both retrievers: fusion can only rerank what it is given,
        # so over-fetch before truncating to top_k
        pool = max(top_k * 4, 20)

        vec_rows: List[dict] = []
        kw_rows: List[dict] = []
        if mode in ("vector", "hybrid"):
            qv = self.embedder.embed_query(query.strip())
            vec_rows = self.db.vector_search(qv, top_k=pool, doc_ids=doc_ids,
                                             exclude_doc_ids=exclude_doc_ids)
        if mode in ("keyword", "hybrid"):
            kw_rows = self.db.keyword_search(query.strip(), top_k=pool,
                                             doc_ids=doc_ids,
                                             exclude_doc_ids=exclude_doc_ids)

        if mode == "hybrid":
            ordered = self._fuse(vec_rows, kw_rows, top_k)
        else:
            ordered = [(r["chunk_id"], float(r["score"]), None, None)
                       for r in (vec_rows or kw_rows)[:top_k]]

        if threshold_mode:
            # threshold only makes sense on a real similarity, so the vector arm
            # is the only one that can honour it
            ordered = [(cid, sc, vr, kr) for cid, sc, vr, kr in ordered
                       if sc is not None and float(sc) >= min_score]
            if len(ordered) >= int(max_candidates or 200):
                log.warning("retrieval candidate cap reached (%d hits >= %.3f); "
                            "raise max_candidates to see the tail",
                            len(ordered), min_score)

        hits: List[Hit] = []
        for chunk_id, score, vrank, krank in ordered:
            chunk = self.db.get_chunk(chunk_id)
            if not chunk:
                continue
            doc = self.db.get_document(chunk["doc_id"])
            hits.append(Hit(
                chunk_id=int(chunk["id"]),
                doc_id=int(chunk["doc_id"]),
                doc_name=(doc or {}).get("name", "?"),
                section=chunk.get("section") or "",
                breadcrumb=chunk.get("breadcrumb") or "",
                text=chunk.get("text") or "",
                score=float(score),
                vector_rank=vrank,
                keyword_rank=krank,
            ))
        return hits

    def find_similar_chunks(self, chunk_id: int, min_score: float = 0.65,
                            max_candidates: int = 200,
                            exclude_doc_ids: Optional[Iterable[int]] = None
                            ) -> List[Hit]:
        """Every chunk whose stored vector is at or above *min_score* of *chunk_id*.

        Used by the contradiction scanner: a similarity floor, not a top-k cut,
        because a conflict can rank far down a long list of similar passages.
        """
        rows = self.db.similar_to_chunk(chunk_id, min_score=min_score,
                                        limit=max_candidates,
                                        exclude_doc_ids=exclude_doc_ids)
        if len(rows) >= int(max_candidates):
            log.warning("candidate cap reached (%d chunks within %.3f of chunk %s); "
                        "raise SCAN_MAX_CANDIDATES to see the tail",
                        len(rows), float(min_score), chunk_id)
        hits: List[Hit] = []
        for row in rows:
            chunk = self.db.get_chunk(row["chunk_id"])
            if not chunk:
                continue
            doc = self.db.get_document(chunk["doc_id"])
            hits.append(Hit(
                chunk_id=int(chunk["id"]),
                doc_id=int(chunk["doc_id"]),
                doc_name=(doc or {}).get("name", "?"),
                section=chunk.get("section") or "",
                breadcrumb=chunk.get("breadcrumb") or "",
                text=chunk.get("text") or "",
                score=float(row["score"]),
            ))
        return hits

    # ------------------------------------------------------------------ fusion

    def _fuse(self, vec_rows: List[dict], kw_rows: List[dict], top_k: int
              ) -> List[tuple]:
        """RRF over the two rankings. Returns [(chunk_id, score, vr, kr), ...]."""
        fused: Dict[int, float] = {}
        vrank: Dict[int, int] = {}
        krank: Dict[int, int] = {}
        for i, r in enumerate(vec_rows, start=1):
            cid = int(r["chunk_id"])
            vrank[cid] = i
            fused[cid] = fused.get(cid, 0.0) + self.vector_weight / (self.rrf_k + i)
        for i, r in enumerate(kw_rows, start=1):
            cid = int(r["chunk_id"])
            krank[cid] = i
            fused[cid] = fused.get(cid, 0.0) + self.keyword_weight / (self.rrf_k + i)
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(cid, score, vrank.get(cid), krank.get(cid))
                for cid, score in ranked[:top_k]]
