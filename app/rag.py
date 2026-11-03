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

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .db import Database

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
               mode: Optional[str] = None) -> List[Hit]:
        """Return the top passages for *query* using the configured retriever."""
        if not query or not query.strip():
            return []
        top_k = int(top_k) if top_k else 6
        mode = (mode or self.mode).strip().lower()
        if mode not in MODES:
            mode = "vector"

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
