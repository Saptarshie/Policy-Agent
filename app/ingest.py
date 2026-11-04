"""Cascading rolling-window ingestion pipeline for (possibly very large) docs.

Never feeds a whole document to the LLM. Instead:

    read file -> windows (paragraph boundaries, ~INGEST_WINDOW_CHARS each)
      -> for each window, ONE bounded LLM call that does BOTH:
           (a) window parse  -> title / statements / key terms / numbers
           (b) rolling summary (prev summary + window -> new summary)
         with heuristic fallback + recorded warning if the LLM output is
         unparseable even after regex repair,
      -> progress tracking: an `ingest` job row is updated after every window
         (done/total/stage/rolling-summary checkpoint),
      -> structure-aware chunking of the full text (headings -> paragraphs ->
         sentences; breadcrumbs for citations),
      -> batched embedding of all chunks (Ollama embeddinggemma, document
         prefix), stored in SQLite.

Two knobs trade quality for wall time, both off the synchronous answer path:

    INGEST_REASONING_EFFORT   effort for the window parser (default: off - this
                              is structured extraction, not reasoning; measured
                              40.5s -> 9.6s per call on the same window)
    INGEST_PARALLEL_WINDOWS   parse N windows at once (default 4). The windows
                              are no longer chained through prev_summary, which
                              only fed metadata; the rolling summary is built in
                              ONE merge call afterwards.
    INGEST_HEURISTIC_MAX_CHARS  documents at or below this size skip the LLM
                              parse entirely (0 = never skip).

After success the caller starts the background contradiction scan.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .chunker import chunk_text, split_into_windows
from .db import Database
from .llm import OpenAICompatClient
from .prompts import WINDOW_PARSER_SYSTEM, WINDOW_PARSER_USER
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

            prev_summary = self._parse_windows(name, windows, doc_id, job_id,
                                               use_llm, meta, warnings)

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
                # with a different embedder fails loudly instead of silently
                name = getattr(self.embedder, "model_name", "")
                if not isinstance(name, str) or not name:
                    cand = getattr(self.embedder, "model", "")
                    name = cand if isinstance(cand, str) else type(self.embedder).__name__
                self.db.set_embed_fingerprint(name, len(vectors[0]))
            for c, v in zip(chunks, vectors):
                self.db.add_chunk(doc_id, c.index, c.section, c.breadcrumb, c.text, v)

            meta["key_terms"] = sorted(set(meta.get("key_terms", [])))[:60]
            self.db.update_document(
                doc_id, status="ready", summary=prev_summary,
                meta_json=json.dumps(meta, ensure_ascii=False),
            )
            self.db.update_job(job_id, status="done", stage="done")
            self._emit(doc_id, job_id, len(windows), len(windows), "done")
            log.info("ingested '%s' (%d windows, %d chunks, %d warnings)",
                     name, len(windows), len(chunks), len(warnings))
            return {"doc_id": doc_id, "job_id": job_id, "chunks": len(chunks),
                    "windows": len(windows), "warnings": warnings,
                    "summary": prev_summary}
        except Exception as e:
            self.db.update_document(doc_id, status="error")
            self.db.update_job(job_id, status="error", error=str(e)[:1000])
            self._emit(doc_id, job_id, 0, len(windows), f"error: {e}")
            raise

    # ----------------------------------------------------------------- private

    def _parse_windows(self, doc_name: str, windows: List[str], doc_id: int,
                       job_id: int, use_llm: bool, meta: Dict,
                       warnings: List[str]) -> str:
        """Parse every window; returns the document's rolling summary.

        Sequential (``workers == 1``) keeps the original behaviour: window i sees
        the summary of windows < i. With more workers the windows are parsed
        independently (the chain only fed METADATA, never the chunks) and the
        rolling summary is produced afterwards in one merge call.
        """
        n = len(windows)
        if not use_llm:
            prev = ""
            for i, window in enumerate(windows):
                prev = self._heuristic_summary(prev, window)
                self._progress(job_id, doc_id, i + 1, n, f"window {i + 1}/{n} (heuristic)")
            return prev

        if self.workers <= 1 or n <= 1:
            prev = ""
            for i, window in enumerate(windows):
                obj, err = self._parse_window(doc_name, i, n, window, prev)
                if obj is not None:
                    self._merge_meta(meta, obj)
                    rs = obj.get("rolling_summary")
                    if isinstance(rs, str) and rs.strip():
                        prev = rs.strip()[:1500]
                else:
                    w = (f"window {i + 1}/{n}: LLM parse failed ({err}); "
                         "used heuristic fallback")
                    warnings.append(w)
                    meta["warnings"].append(w)
                    prev = self._heuristic_summary(prev, window)
                self._progress(job_id, doc_id, i + 1, n, f"parsed window {i + 1}/{n}")
            return prev

        # ---- parallel: parse independently, keep the order for the merge
        results: List[Optional[Dict]] = [None] * n
        errors: List[Optional[str]] = [None] * n
        done = 0
        with ThreadPoolExecutor(max_workers=min(self.workers, n),
                                thread_name_prefix="ingest-window") as pool:
            futures = {
                pool.submit(self._parse_window, doc_name, i, n, window, ""): i
                for i, window in enumerate(windows)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    obj, err = fut.result()
                except Exception as e:                  # noqa: BLE001
                    obj, err = None, f"worker error: {e}"
                results[i], errors[i] = obj, err
                done += 1
                self._progress(job_id, doc_id, done, n,
                               f"parsed {done}/{n} windows (parallel)")

        summaries: List[str] = []
        for i, obj in enumerate(results):
            if obj is None:
                w = (f"window {i + 1}/{n}: LLM parse failed ({errors[i]}); "
                     "used heuristic fallback")
                warnings.append(w)
                meta["warnings"].append(w)
                summaries.append(self._heuristic_summary("", windows[i]))
                continue
            self._merge_meta(meta, obj)
            rs = obj.get("rolling_summary")
            summaries.append(rs.strip()[:1500] if isinstance(rs, str) and rs.strip()
                             else self._heuristic_summary("", windows[i]))

        self.db.update_job(job_id, stage="merging window summaries")
        return self._merge_summaries(summaries)

    def _merge_summaries(self, summaries: List[str]) -> str:
        """One call to fold per-window summaries into a document summary."""
        summaries = [s for s in summaries if s and s.strip()]
        if not summaries:
            return ""
        if len(summaries) == 1:
            return summaries[0][:1500]
        numbered = "\n\n".join(f"[{i + 1}] {s}" for i, s in enumerate(summaries))
        try:
            resp = self.llm.chat(
                [{"role": "system",
                  "content": ("Merge these per-window summaries of ONE policy "
                              "document into a single rolling summary. Keep "
                              "decisions, amounts, thresholds and who they apply "
                              "to. Reply with the summary text only, at most 1200 "
                              "characters, no preamble.")},
                 {"role": "user", "content": numbered}],
                max_tokens=600,
                reasoning_effort=self.reasoning_effort or None,
            )
            merged = (resp.content or "").strip()
            if merged:
                return merged[:1500]
        except Exception as e:                          # noqa: BLE001
            log.warning("summary merge failed (%s); concatenating instead", e)
        return " ".join(summaries)[:1500]

    def _progress(self, job_id: int, doc_id: int, done: int, total: int,
                  stage: str) -> None:
        self.db.update_job(job_id, done=done, stage=stage)
        self._emit(doc_id, job_id, done, total, stage)

    def _parse_window(self, doc_name: str, i: int, n: int, window: str,
                      prev_summary: str):
        user = WINDOW_PARSER_USER.format(
            prev_summary=prev_summary or "(empty)",
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

    @staticmethod
    def _heuristic_summary(prev: str, window: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", window.strip())
        add = " ".join(sentences[:2])
        s = (prev + " " + add).strip()
        return s[-1200:]

    def _emit(self, doc_id: int, job_id: int, done: int, total: int, stage: str) -> None:
        if self.on_progress:
            try:
                self.on_progress({"doc_id": doc_id, "job_id": job_id, "done": done,
                                  "total": total, "stage": stage})
            except Exception:
                log.exception("progress callback failed")
