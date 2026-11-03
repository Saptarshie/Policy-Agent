"""SQLite persistence: documents, chunks (+embeddings), contradictions, jobs.

A single connection guarded by a re-entrant lock keeps multi-threaded use
(web app background ingestion + scan threads) safe. Embeddings are stored
as float32 BLOBs; an in-memory matrix cache speeds up vector search and is
invalidated on every write.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT UNIQUE NOT NULL,
  filename TEXT DEFAULT '',
  format TEXT DEFAULT '',
  size_bytes INTEGER DEFAULT 0,
  status TEXT DEFAULT 'pending',       -- pending|ingesting|ready|error
  summary TEXT DEFAULT '',
  meta_json TEXT DEFAULT '{}',
  created_at TEXT DEFAULT '',
  updated_at TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id INTEGER NOT NULL,
  chunk_index INTEGER DEFAULT 0,
  section TEXT DEFAULT '',
  breadcrumb TEXT DEFAULT '',
  text TEXT DEFAULT '',
  char_count INTEGER DEFAULT 0,
  embedding BLOB,
  norm REAL DEFAULT 0,
  created_at TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
-- Keyword index for hybrid retrieval. External-content FTS5 keeps the text in
-- one place (chunks) and the index in another; the triggers below are what keep
-- them in sync, so an update/delete cannot leave a stale entry behind.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text,
  content='chunks',
  content_rowid='id',
  tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
  INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TABLE IF NOT EXISTS contradictions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_a_id INTEGER, doc_a TEXT, section_a TEXT, text_a TEXT,
  chunk_b_id INTEGER, doc_b TEXT, section_b TEXT, text_b TEXT,
  gate_score REAL DEFAULT 0,
  gate_signals TEXT DEFAULT '[]',
  verified INTEGER DEFAULT 0,          -- 1 = LLM-verified, 0 = heuristic-only
  confidence REAL DEFAULT 0,
  conflict_type TEXT DEFAULT '',
  explanation TEXT DEFAULT '',
  points_json TEXT DEFAULT '[]',
  status TEXT DEFAULT 'OPEN',          -- OPEN|resolved|dismissed
  detected_at TEXT DEFAULT ''
);
-- A chunk pair (a,b) and (b,a) is ONE conflict: flags may be written from
-- either scan direction, so uniqueness must be order-agnostic. Dedupe any
-- rows written before this constraint existed (keep the oldest), then make
-- the canonical pair key unique so a race cannot slip a duplicate in.
DELETE FROM contradictions WHERE chunk_a_id IS NOT NULL AND chunk_b_id IS NOT NULL
  AND id NOT IN (
    SELECT MIN(id) FROM contradictions
     WHERE chunk_a_id IS NOT NULL AND chunk_b_id IS NOT NULL
     GROUP BY min(chunk_a_id, chunk_b_id), max(chunk_a_id, chunk_b_id)
  );
CREATE UNIQUE INDEX IF NOT EXISTS idx_contradictions_pair
  ON contradictions(min(chunk_a_id, chunk_b_id), max(chunk_a_id, chunk_b_id));
CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT,                            -- ingest|scan
  doc_id INTEGER,
  doc_name TEXT DEFAULT '',
  status TEXT DEFAULT 'running',        -- running|done|error
  total INTEGER DEFAULT 0,
  done INTEGER DEFAULT 0,
  stage TEXT DEFAULT '',
  detail TEXT DEFAULT '{}',
  error TEXT DEFAULT '',
  created_at TEXT DEFAULT '',
  updated_at TEXT DEFAULT ''
);
-- small key/value table: embedding-model fingerprint and other store metadata.
-- Without a fingerprint, switching embedder (ollama 768-d <-> sentence
-- transformers 384-d) fails later as a numpy matmul shape error on query.
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY,
  value TEXT DEFAULT ''
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path) -> None:
        self.path = str(path)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()
            self._backfill_keyword_index()
        self._vec_cache: Optional[List[Any]] = None

    def _backfill_keyword_index(self) -> None:
        """Build/repair the FTS index so it covers every stored chunk (idempotent).

        ``count(*) FROM chunks_fts`` is useless here: for an external-content
        table it reads the *content* table, so it always equals the chunk count
        even when the index itself is empty. ``chunks_fts_docsize`` has one row
        per indexed document, which is the invariant we actually want
        (indexed == stored).
        """
        try:
            n_chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            n_indexed = self.conn.execute(
                "SELECT COUNT(*) FROM chunks_fts_docsize").fetchone()[0]
        except sqlite3.Error:                   # FTS5 not compiled in
            log.warning("FTS5 unavailable: keyword search disabled")
            return
        if n_chunks and n_indexed < n_chunks:
            self.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            self.conn.commit()
            log.info("keyword index rebuilt for %s chunks (was %s)",
                     n_chunks, n_indexed)

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    # ------------------------------------------------------------- documents

    def add_document(self, name: str, filename: str = "", fmt: str = "",
                     size: int = 0, status: str = "pending") -> int:
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO documents(name, filename, format, size_bytes, status,"
                " created_at, updated_at) VALUES(?,?,?,?,?,?,?)",
                (name, filename, fmt, int(size), status, _now(), _now()),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def update_document(self, doc_id: int, **fields: Any) -> None:
        if not fields:
            return
        fields = dict(fields)
        fields["updated_at"] = _now()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.lock:
            self.conn.execute(
                f"UPDATE documents SET {cols} WHERE id=?",
                (*fields.values(), doc_id),
            )
            self.conn.commit()

    def get_document(self, doc: Any) -> Optional[Dict[str, Any]]:
        with self.lock:
            if isinstance(doc, int) or (isinstance(doc, str) and doc.isdigit()):
                row = self.conn.execute(
                    "SELECT * FROM documents WHERE id=?", (int(doc),)
                ).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT * FROM documents WHERE name=?", (str(doc),)
                ).fetchone()
        return dict(row) if row else None

    def find_document_fuzzy(self, name: str) -> Optional[Dict[str, Any]]:
        doc = self.get_document(name)
        if doc:
            return doc
        low = (name or "").lower().strip()
        for d in self.list_documents():
            nl = d["name"].lower()
            if low and (low in nl or nl in low):
                return d
        return None

    def list_documents(self) -> List[Dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT d.*,
                  (SELECT COUNT(*) FROM chunks c WHERE c.doc_id=d.id) AS chunk_count,
                  (SELECT COUNT(*) FROM contradictions k
                     WHERE (k.doc_a=d.name OR k.doc_b=d.name) AND k.status='OPEN')
                    AS open_contradictions
                FROM documents d ORDER BY d.id
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: int) -> bool:
        with self.lock:
            row = self.conn.execute("SELECT id FROM documents WHERE id=?", (doc_id,)).fetchone()
            if not row:
                return False
            self.conn.execute(
                "DELETE FROM contradictions WHERE chunk_a_id IN"
                " (SELECT id FROM chunks WHERE doc_id=?) OR chunk_b_id IN"
                " (SELECT id FROM chunks WHERE doc_id=?)",
                (doc_id, doc_id),
            )
            self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self.conn.execute("DELETE FROM jobs WHERE doc_id=?", (doc_id,))
            self.conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            self.conn.commit()
        self._vec_cache = None
        return True

    def unique_name(self, base: str) -> str:
        base = (base or "document").strip() or "document"
        with self.lock:
            names = {r["name"] for r in self.conn.execute("SELECT name FROM documents")}
        if base not in names:
            return base
        i = 2
        while f"{base} ({i})" in names:
            i += 1
        return f"{base} ({i})"

    # ---------------------------------------------------------------- chunks

    def add_chunk(self, doc_id: int, index: int, section: str, breadcrumb: str,
                  text: str, embedding: Sequence[float]) -> int:
        vec = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vec)) if vec.size else 0.0
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO chunks(doc_id, chunk_index, section, breadcrumb, text,"
                " char_count, embedding, norm, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (doc_id, index, section, breadcrumb, text, len(text),
                 vec.tobytes(), norm, _now()),
            )
            self.conn.commit()
            cid = int(cur.lastrowid)
        self._vec_cache = None
        return cid

    def get_chunk(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        with self.lock:
            row = self.conn.execute("SELECT * FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        return dict(row) if row else None

    def list_chunks(self, doc_id: int) -> List[Dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM chunks WHERE doc_id=? ORDER BY chunk_index", (doc_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def count_chunks(self, doc_id: int) -> int:
        with self.lock:
            row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE doc_id=?", (doc_id,)
            ).fetchone()
        return int(row["n"])

    # ---------------------------------------------------------- vector search

    def get_meta(self, key: str, default: str = "") -> str:
        with self.lock:
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?,?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)),
            )
            self.conn.commit()

    def set_embed_fingerprint(self, model: str, dim: int) -> None:
        """Record which embedding model produced this store's vectors."""
        self.set_meta("embed_model", model or "unknown")
        self.set_meta("embed_dim", str(int(dim)))

    def embed_fingerprint(self):
        dim = self.get_meta("embed_dim")
        return (self.get_meta("embed_model", "unknown"),
                int(dim) if dim.isdigit() else 0)

    def _load_matrix(self):
        if self._vec_cache is None:
            with self.lock:
                rows = self.conn.execute(
                    "SELECT id, doc_id, embedding FROM chunks"
                ).fetchall()
            ids: List[int] = []
            doc_ids: List[int] = []
            vecs: List[np.ndarray] = []
            for r in rows:
                ids.append(int(r["id"]))
                doc_ids.append(int(r["doc_id"]))
                vecs.append(np.frombuffer(r["embedding"], dtype=np.float32))
            if vecs:
                mat = np.vstack(vecs)
                norms = np.linalg.norm(mat, axis=1)
            else:
                mat = np.zeros((0, 1), dtype=np.float32)
                norms = np.zeros(0, dtype=np.float32)
            self._vec_cache = [ids, doc_ids, mat, norms]
        return self._vec_cache

    def vector_search(self, query_vec: Sequence[float], top_k: int = 6,
                      doc_ids: Optional[Iterable[int]] = None,
                      exclude_doc_ids: Optional[Iterable[int]] = None
                      ) -> List[Dict[str, Any]]:
        """Cosine-similarity search. Returns [{'chunk_id','doc_id','score'}, ...]."""
        ids, doc_ids_all, mat, norms = self._load_matrix()
        if mat.shape[0] == 0:
            return []
        include = set(doc_ids) if doc_ids is not None else None
        exclude = set(exclude_doc_ids) if exclude_doc_ids is not None else None
        q = np.asarray(query_vec, dtype=np.float32)
        # fail with an actionable message instead of a numpy matmul error when
        # the store was built with a different embedding model
        expected_dim = mat.shape[1]
        if q.size and expected_dim and q.size != expected_dim:
            model, _ = self.embed_fingerprint()
            raise RuntimeError(
                f"embedding dimension mismatch: the store holds {expected_dim}-d "
                f"vectors (model: {model}) but the query is {q.size}-d. "
                "Re-ingest the documents with the configured embedder, or point "
                "OLLAMA_BASE_URL/EMBED_MODEL at the model that built this store."
            )
        qn = float(np.linalg.norm(q))
        if qn == 0:
            return []
        sims = (mat @ q) / (norms * qn + 1e-12)
        out: List[Dict[str, Any]] = []
        for i in np.argsort(-sims):
            d = doc_ids_all[int(i)]
            if include is not None and d not in include:
                continue
            if exclude is not None and d in exclude:
                continue
            out.append({"chunk_id": ids[int(i)], "doc_id": d, "score": float(sims[int(i)])})
            if len(out) >= top_k:
                break
        return out

    # ---------------------------------------------------------- keyword search

    # FTS5 MATCH has its own query grammar: a raw employee question can carry
    # quotes, hyphens, `*`, `NEAR`, `AND` - any of which raises OperationalError.
    # Tokenise defensively and quote every term instead of passing text through.
    _TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

    @classmethod
    def build_match_query(cls, text: str) -> str:
        tokens = [t.lower() for t in cls._TOKEN_RE.findall(text or "")]
        # drop single characters ("a", "I") - they match everything
        tokens = [t for t in tokens if len(t) > 1]
        # keep order, drop duplicates, cap the clause length
        seen, uniq = set(), []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        uniq = uniq[:24]
        return " OR ".join(f'"{t}"' for t in uniq)

    def keyword_search(self, query: str, top_k: int = 6,
                       doc_ids: Optional[Iterable[int]] = None,
                       exclude_doc_ids: Optional[Iterable[int]] = None
                       ) -> List[Dict[str, Any]]:
        """BM25 keyword search over chunk text (FTS5). Returns
        [{'chunk_id','doc_id','score'}, ...] with score = -bm25 (higher better)."""
        match = self.build_match_query(query)
        if not match:
            return []
        include = set(doc_ids) if doc_ids is not None else None
        exclude = set(exclude_doc_ids) if exclude_doc_ids is not None else None
        # fetch a wider window then filter, so document filters do not shrink
        # the result set below top_k for the common unfiltered case
        limit = max(top_k * 4, 20)
        try:
            with self.lock:
                rows = self.conn.execute(
                    "SELECT c.id AS chunk_id, c.doc_id AS doc_id, bm25(chunks_fts) AS rank"
                    " FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid"
                    " WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                    (match, limit),
                ).fetchall()
        except sqlite3.Error as e:              # FTS5 missing / malformed query
            log.warning("keyword search failed (%s); falling back to vectors only", e)
            return []
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = int(r["doc_id"])
            if include is not None and d not in include:
                continue
            if exclude is not None and d in exclude:
                continue
            # bm25() is lower-is-better and negative; flip so callers can sort desc
            out.append({"chunk_id": int(r["chunk_id"]), "doc_id": d,
                        "score": -float(r["rank"])})
            if len(out) >= top_k:
                break
        return out

    # --------------------------------------------------------- contradictions

    def add_contradiction(self, chunk_a_id: int, doc_a: str, section_a: str, text_a: str,
                          chunk_b_id: int, doc_b: str, section_b: str, text_b: str,
                          gate_score: float, gate_signals: List[str], verified: bool,
                          confidence: float, conflict_type: str, explanation: str,
                          points: List[Dict[str, Any]]) -> int:
        """Insert a contradiction flag, at most one per chunk pair.

        Uniqueness is order-agnostic: a flag for (chunk 2, chunk 1) and one
        for (chunk 1, chunk 2) are the same conflict, because the tracker
        writes whichever chunk belongs to the document being scanned.
        Returns the new row id, or ``0`` when this pair was already flagged.
        """
        with self.lock:
            cur = self.conn.execute(
                """INSERT OR IGNORE INTO contradictions(chunk_a_id, doc_a, section_a, text_a,
                     chunk_b_id, doc_b, section_b, text_b, gate_score, gate_signals,
                     verified, confidence, conflict_type, explanation, points_json,
                     status, detected_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'OPEN',?)""",
                (chunk_a_id, doc_a, section_a, text_a[:2500],
                 chunk_b_id, doc_b, section_b, text_b[:2500],
                 float(gate_score), json.dumps(gate_signals, ensure_ascii=False),
                 1 if verified else 0, float(confidence), conflict_type,
                 explanation, json.dumps(points, ensure_ascii=False), _now()),
            )
            self.conn.commit()
            if cur.rowcount:
                return int(cur.lastrowid)
            # INSERT OR IGNORE skipped: this chunk pair is already flagged
            # (stored in either order, or inserted by a concurrent scan).
            # Report 0 = "nothing created" so callers do not double-count or
            # re-alert an existing flag.
            return 0

    def contradiction_pair_exists(self, chunk_a_id: int, chunk_b_id: int) -> bool:
        """True when this chunk pair is already flagged, in EITHER order."""
        with self.lock:
            return self._find_pair_id(chunk_a_id, chunk_b_id) is not None

    def _find_pair_id(self, chunk_a_id: int, chunk_b_id: int) -> Optional[int]:
        """Id of the existing flag for this chunk pair, independent of the
        order the ids were stored in.  Caller must hold ``self.lock``."""
        a, b = sorted((int(chunk_a_id), int(chunk_b_id)))
        row = self.conn.execute(
            "SELECT id FROM contradictions WHERE"
            " (chunk_a_id=? AND chunk_b_id=?) OR (chunk_a_id=? AND chunk_b_id=?)"
            " ORDER BY id LIMIT 1",
            (a, b, b, a),
        ).fetchone()
        return int(row["id"]) if row else None

    def list_contradictions(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.lock:
            if status:
                rows = self.conn.execute(
                    "SELECT * FROM contradictions WHERE status=? ORDER BY id DESC",
                    (status,),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM contradictions ORDER BY id DESC"
                ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["gate_signals"] = json.loads(d.get("gate_signals") or "[]")
            except json.JSONDecodeError:
                d["gate_signals"] = []
            try:
                d["points"] = json.loads(d.get("points_json") or "[]")
            except json.JSONDecodeError:
                d["points"] = []
            out.append(d)
        return out

    def set_contradiction_status(self, cid: int, status: str) -> bool:
        if status not in ("OPEN", "resolved", "dismissed"):
            return False
        with self.lock:
            cur = self.conn.execute(
                "UPDATE contradictions SET status=? WHERE id=?", (status, cid)
            )
            self.conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------ jobs

    def create_job(self, kind: str, doc_id: int, doc_name: str, total: int) -> int:
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO jobs(kind, doc_id, doc_name, status, total, done,"
                " created_at, updated_at) VALUES(?,?,?,?,?,0,?,?)",
                (kind, doc_id, doc_name, "running", int(total), _now(), _now()),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def update_job(self, job_id: int, **fields: Any) -> None:
        if not fields:
            return
        fields = dict(fields)
        fields["updated_at"] = _now()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.lock:
            self.conn.execute(f"UPDATE jobs SET {cols} WHERE id=?",
                              (*fields.values(), job_id))
            self.conn.commit()

    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        with self.lock:
            row = self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["detail"] = json.loads(d.get("detail") or "{}")
        except json.JSONDecodeError:
            d["detail"] = {}
        return d

    def reconcile_stale_jobs(self, max_age_minutes: int = 30) -> List[Dict[str, Any]]:
        """Mark jobs abandoned by a dead process as failed.

        Ingestion runs on a daemon thread; if the process is killed the `jobs`
        row stays `running` forever and the document stays `ingesting`, so the UI
        polls a job that will never finish and the document is unusable. A job
        that has not been touched for *max_age_minutes* is treated as abandoned.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(1, max_age_minutes))
        fixed: List[Dict[str, Any]] = []
        with self.lock:
            rows = self.conn.execute(
                "SELECT id, doc_id, kind, stage, updated_at, created_at FROM jobs"
                " WHERE status='running'").fetchall()
            for row in rows:
                stamp = row["updated_at"] or row["created_at"] or ""
                try:
                    when = datetime.fromisoformat(stamp)
                except ValueError:
                    when = None
                if when is not None and when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                if when is not None and when >= cutoff:
                    continue
                self.conn.execute(
                    "UPDATE jobs SET status='error', error=?, updated_at=?"
                    " WHERE id=?",
                    ("interrupted: no progress recorded since "
                     f"{stamp or 'unknown'}", _now(), row["id"]))
                self.conn.execute(
                    "UPDATE documents SET status='error', updated_at=?"
                    " WHERE id=? AND status IN ('ingesting','pending')",
                    (_now(), row["doc_id"]))
                fixed.append({"job_id": row["id"], "doc_id": row["doc_id"],
                              "kind": row["kind"], "stage": row["stage"]})
            if fixed:
                self.conn.commit()
        return fixed

    def latest_job_for_doc(self, doc_id: int, kind: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self.lock:
            if kind:
                row = self.conn.execute(
                    "SELECT * FROM jobs WHERE doc_id=? AND kind=? ORDER BY id DESC LIMIT 1",
                    (doc_id, kind),
                ).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT * FROM jobs WHERE doc_id=? ORDER BY id DESC LIMIT 1",
                    (doc_id,),
                ).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["detail"] = json.loads(d.get("detail") or "{}")
        except json.JSONDecodeError:
            d["detail"] = {}
        return d
