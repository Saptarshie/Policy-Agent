"""Cascading rolling-window ingestion pipeline for (possibly very large) docs.

Never feeds a whole document to the LLM. Instead:

    read file -> windows (paragraph boundaries, ~INGEST_WINDOW_CHARS each)
      -> for each window, ONE bounded LLM call that extracts title / statements /
         key terms / numbers (metadata only - it never touches the text that is
         stored, and the windows are independent, so they run concurrently),
      -> ONE document brief: a bounded 3-sentence "what is this document about",
         built from the opening INGEST_BRIEF_WORDS words plus every heading. This
         is what the triage layer and the agent read to know what the corpus holds;
         it is NOT an input to chunking (chunk text stays verbatim),
      -> progress tracking: an `ingest` job row is updated after every window,
      -> structure-aware chunking of the full text (headings -> paragraphs ->
         sentences; breadcrumbs for citations),
      -> batched embedding of all chunks (Ollama embeddinggemma, document
         prefix), stored in SQLite.

Knobs, all off the synchronous answer path:

    INGEST_REASONING_EFFORT     effort for the window parser (default: off - this
                                is structured extraction, not reasoning; measured
                                40.5s -> 9.6s per call on the same window)
    INGEST_PARALLEL_WINDOWS     windows parsed at once (default 4). There is no
                                chain between windows any more: nothing a window
                                produces feeds the next one.
    INGEST_BRIEF_WORDS          opening words the document brief reads
    INGEST_HEURISTIC_MAX_CHARS  documents at or below this size skip the LLM
                                parse entirely (0 = never skip)

After success the caller starts the background contradiction scan.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .chunker import chunk_text, detect_headings, split_into_windows
from .db import Database
from .llm import OpenAICompatClient
from .prompts import (DOC_BRIEF_SYSTEM, DOC_BRIEF_USER,
                      WINDOW_PARSER_SYSTEM, WINDOW_PARSER_USER)
from .readers import read_document

log = logging.getLogger(__name__)

ProgressFn = Callable[[Dict], None]


class IngestPipeline:
    def __init__(
        self,
        db: Database,
        embedder,
        llm: Optional[OpenAICompatClient] = None,
        window_chars: int = 6000,
        use_llm: bool = True,
        chunk_target: int = 1400,
        chunk_max: int = 2200,
        on_progress: Optional[ProgressFn] = None,
        reasoning_effort: str = "",
        max_parse_tokens: int = 3000,
        parallel_windows: int = 1,
        heuristic_max_chars: int = 0,
        brief_words: int = 1000,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.llm = llm
        self.window_chars = window_chars
        self.use_llm = use_llm and llm is not None
        self.chunk_target = chunk_target
        self.chunk_max = chunk_max
        self.on_progress = on_progress
        # the window parser only extracts structure; extended reasoning there
        # costs ~4x wall time and changes nothing in the stored chunks
        self.reasoning_effort = (reasoning_effort or "").strip()
        self.max_parse_tokens = max(256, int(max_parse_tokens))
        #: windows parsed concurrently (1 = the original sequential chain)
        self.workers = max(1, int(parallel_windows))
        #: documents at or below this size skip the LLM parse (0 = never)
        self.heuristic_max_chars = max(0, int(heuristic_max_chars))
        #: opening words the document brief reads
        self.brief_words = max(200, int(brief_words))

    # ------------------------------------------------------------------ public

    def ingest(self, path: Path, name: str, doc_id: Optional[int] = None,
               on_progress: Optional[ProgressFn] = None) -> Dict:
        """Ingest one file. Returns dict with doc_id / job_id / chunks / warnings."""
        if on_progress is not None:
            self.on_progress = on_progress
        path = Path(path)
        text, fmt = read_document(path)
        size = path.stat().st_size if path.exists() else len(text)

        if doc_id is None:
            doc_id = self.db.add_document(name=name, filename=path.name, fmt=fmt,
                                          size=size, status="ingesting")
        else:
            self.db.update_document(doc_id, format=fmt, size_bytes=size,
                                    filename=path.name, status="ingesting")

        windows = split_into_windows(text, self.window_chars)
        job_id = self.db.create_job("ingest", doc_id, name, total=len(windows))
        self._emit(doc_id, job_id, 0, len(windows), "starting")

        warnings: List[str] = []
        meta: Dict = {"windows": len(windows), "sections": [], "key_terms": [],
                      "statements": [], "warnings": []}
        prev_summary = ""

        try:
            if self.use_llm and self.heuristic_max_chars \
                    and len(text) <= self.heuristic_max_chars:
                # small document: the LLM parse buys little and costs a call
                log.info("'%s' is %d chars <= INGEST_HEURISTIC_MAX_CHARS (%d): "
                         "heuristic parse", name, len(text), self.heuristic_max_chars)
                meta["parse"] = "heuristic (configured by document size)"
                use_llm = False
            else:
                meta["parse"] = "llm" if self.use_llm else "heuristic (INGEST_LLM_ENABLED=false)"
                use_llm = self.use_llm

            if use_llm:
                self._parse_windows(name, windows, doc_id, job_id, meta, warnings)
            else:
                for i in range(len(windows)):
                    self._progress(job_id, doc_id, i + 1, len(windows),
                                   f"window {i + 1}/{len(windows)} (heuristic)")

            # ---- the document brief: what the triage layer and the agent read
            self.db.update_job(job_id, stage="writing the document brief")
            brief = self._doc_brief(name, text, warnings, use_llm=use_llm)
            meta["brief"] = {"chars": len(brief),
                             "source": "llm" if use_llm else "opening"}

            # ---- structure-aware chunking of the FULL text
            self.db.update_job(job_id, stage="chunking document")
            chunks = chunk_text(text, name, self.chunk_target, self.chunk_max)
            if not chunks:
                raise ValueError("chunking produced no chunks (document too small?)")

            # ---- batched embedding + storage
            self.db.update_job(job_id, stage=f"embedding {len(chunks)} chunks")
            texts = [c.text for c in chunks]
            vectors = self.embedder.embed_documents(texts)
            if len(vectors) != len(chunks):
                raise ValueError(
                    f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
                )
            if vectors:
                # record which model produced these vectors so a later query
                # with a different embedder fails loudly instead of silently.
                # NOTE: this used to reuse the local name `name`, which shadowed the
                # document name for every log line and return value after this point.
                embed_name = getattr(self.embedder, "model_name", "")
                if not isinstance(embed_name, str) or not embed_name:
                    cand = getattr(self.embedder, "model", "")
                    embed_name = (cand if isinstance(cand, str)
                                  else type(self.embedder).__name__)
                self.db.set_embed_fingerprint(embed_name, len(vectors[0]))
            for c, v in zip(chunks, vectors):
                self.db.add_chunk(doc_id, c.index, c.section, c.breadcrumb, c.text, v)

            meta["key_terms"] = sorted(set(meta.get("key_terms", [])))[:60]
            self.db.update_document(
                doc_id, status="ready", summary=brief,
                meta_json=json.dumps(meta, ensure_ascii=False),
            )
            self.db.update_job(job_id, status="done", stage="done")
            self._emit(doc_id, job_id, len(windows), len(windows), "done")
            log.info("ingested '%s' (%d windows, %d chunks, %d warnings)",
                     name, len(windows), len(chunks), len(warnings))
            return {"doc_id": doc_id, "job_id": job_id, "chunks": len(chunks),
                    "windows": len(windows), "warnings": warnings,
                    "summary": brief}
        except Exception as e:
            self.db.update_document(doc_id, status="error")
            self.db.update_job(job_id, status="error", error=str(e)[:1000])
            self._emit(doc_id, job_id, 0, len(windows), f"error: {e}")
            raise

    # ----------------------------------------------------------------- private

    def _parse_windows(self, doc_name: str, windows: List[str], doc_id: int,
                       job_id: int, meta: Dict, warnings: List[str]) -> None:
        """Extract per-window metadata. Windows are independent.

        Nothing a window produces is used by another window (the old chained
        ``prev_summary`` only ever fed metadata), so the pool needs no ordering
        and the stored result cannot depend on which worker finished first.
        """
        n = len(windows)
        if n == 0:
            return
        done = 0
        with ThreadPoolExecutor(max_workers=min(self.workers, n),
                                thread_name_prefix="ingest-window") as pool:
            futures = {
                pool.submit(self._parse_window, doc_name, i, n, window): i
                for i, window in enumerate(windows)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    obj, err = fut.result()
                except Exception as e:                  # noqa: BLE001
                    obj, err = None, f"worker error: {e}"
                if obj is None:
                    w = (f"window {i + 1}/{n}: LLM parse failed ({err}); "
                         "metadata skipped")
                    warnings.append(w)
                    meta["warnings"].append(w)
                else:
                    self._merge_meta(meta, obj)
                done += 1
                self._progress(job_id, doc_id, done, n,
                               f"parsed {done}/{n} windows")

    def _doc_brief(self, doc_name: str, text: str, warnings: List[str],
                   use_llm: bool = True) -> str:
        """One bounded call: what is this document about?

        The opening gives scope, definitions and the first figures; the heading
        list gives everything the opening never reaches. Bounded on purpose - this
        text is put in front of the triage layer for every document on every
        question, so it has to stay small (two or three sentences).
        """
        headings = [h.title for h in detect_headings(text)]
        opening_words = text.split()[: self.brief_words]
        opening = " ".join(opening_words)
        if not (use_llm and self.use_llm):
            return self._brief_from_opening(text, headings)
        user = DOC_BRIEF_USER.format(
            doc_name=doc_name,
            headings=("\n".join(f"- {h}" for h in headings[:80]) or "(none detected)"),
            n_words=len(opening_words),
            opening=opening,
        )
        try:
            obj, err = self.llm.chat_json(
                DOC_BRIEF_SYSTEM, user, max_tokens=400,
                reasoning_effort=self.reasoning_effort or None)
        except Exception as e:                          # noqa: BLE001
            obj, err = None, f"brief call failed: {e}"
        brief = ""
        if isinstance(obj, dict) and isinstance(obj.get("brief"), str):
            brief = obj["brief"].strip()
        if brief:
            return brief[:600]
        w = f"document brief unavailable ({err}); used the opening sentences"
        warnings.append(w)
        return self._brief_from_opening(text, headings)

    @staticmethod
    def _brief_from_opening(text: str, headings: List[str]) -> str:
        """Deterministic fallback: the headings plus the first two sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
        parts = []
        if headings:
            parts.append("Sections: " + ", ".join(headings[:12]))
        if sentences and sentences[0].strip():
            parts.append(" ".join(sentences[:2])[:300])
        return " ".join(parts)[:600]

    def _progress(self, job_id: int, doc_id: int, done: int, total: int,
                  stage: str) -> None:
        self.db.update_job(job_id, done=done, stage=stage)
        self._emit(doc_id, job_id, done, total, stage)

    def _parse_window(self, doc_name: str, i: int, n: int, window: str):
        user = WINDOW_PARSER_USER.format(
            i=i + 1, n=n, doc_name=doc_name, window=window,
        )
        try:
            return self.llm.chat_json(
                WINDOW_PARSER_SYSTEM, user,
                max_tokens=self.max_parse_tokens,
                reasoning_effort=self.reasoning_effort or None)
        except Exception as e:  # LLM down mid-ingestion -> heuristic fallback
            return None, f"LLM error: {e}"

    @staticmethod
    def _merge_meta(meta: Dict, obj: Dict) -> None:
        title = obj.get("window_title")
        if isinstance(title, str) and title.strip():
            meta["sections"].append(title.strip()[:100])
        for key, cap in (("key_terms", 60), ("statements", 300)):
            vals = obj.get(key)
            if isinstance(vals, list):
                for v in vals:
                    if isinstance(v, str) and v.strip() and len(meta[key]) < cap:
                        meta[key].append(v.strip()[:400])
        nums = obj.get("numbers")
        if isinstance(nums, list):
            for v in nums[:20]:
                if isinstance(v, dict):
                    s = f"{v.get('value', '')} {v.get('unit', '')}".strip()
                    ctx = v.get("context", "")
                    if s:
                        meta["statements"].append(
                            f"[number] {s} ({ctx})"[:400]
                        )

    def _emit(self, doc_id: int, job_id: int, done: int, total: int, stage: str) -> None:
        if self.on_progress:
            try:
                self.on_progress({"doc_id": doc_id, "job_id": job_id, "done": done,
                                  "total": total, "stage": stage})
            except Exception:
                log.exception("progress callback failed")
