"""Admin facade: one object wiring db + embedder + llm + agent + tracker.

Used by both the CLI and the FastAPI web app. Uploads run the rolling-window
ingestion pipeline (optionally in a background thread) and then trigger the
staged contradiction scan in another background thread.  ``ask()`` routes
through the deterministic intent gate first, runs the ReAct agent for real
policy questions, and finishes with the grounding verifier (evidence check
+ optional faithfulness judge).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .agent.intent import IntentPlan, IntentRouter
from .agent.react import AgentResult, ReactAgent
from .agent.tools import ToolRegistry
from .config import Config, get_config
from .contradictions.gate import CascadeGate, JEVStyleGate, build_gate_from_config
from .contradictions.tracker import ContradictionTracker
from .contradictions.verify import ContradictionVerifier
from .db import Database
from .embeddings import OllamaEmbedder
from .ingest import IngestPipeline
from .llm import OpenAICompatClient
from .prompts import (
    AGENT_SYSTEM, AGENT_SYSTEM_V2, ESCALATION_NOTE, FAITHFULNESS_SYSTEM,
    FAITHFULNESS_SYSTEM_V2, GREETING_REPLY, INTENT_SYSTEM, INTENT_SYSTEM_V2,
    REFUSAL_TEXT,
)
from .rag import RAGEngine

log = logging.getLogger(__name__)

_LOG_CONFIGURED = False


class AdminAPI:
    def __init__(self, config: Optional[Config] = None,
                 db: Optional[Database] = None,
                 embedder=None, llm=None,
                 intent_router: Optional[IntentRouter] = None,
                 agent_prompt: Optional[str] = None,
                 judge_prompt: Optional[str] = None,
                 gate=None) -> None:
        global _LOG_CONFIGURED
        self.cfg = config or get_config()
        if not _LOG_CONFIGURED:
            logging.basicConfig(
                level=getattr(logging, self.cfg.log_level.upper(), logging.INFO),
                format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            )
            _LOG_CONFIGURED = True
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = db or Database(self.cfg.db_path)
        # a job left 'running' by a killed process would keep its document
        # 'ingesting' forever; reconcile once per process start
        try:
            stale = self.db.reconcile_stale_jobs(self.cfg.stale_job_minutes)
            for item in stale:
                log.warning("abandoned %s job #%s for doc %s (stage '%s'); "
                            "document marked error - re-upload it",
                            item["kind"], item["job_id"], item["doc_id"],
                            item["stage"])
        except Exception as e:               # never block startup on this
            log.info("stale-job reconciliation skipped: %s", e)
        self._embedder = embedder
        self._llm = llm
        self._intent_router = intent_router
        # prompt variants are explicit so the eval harness can ablate them
        self._agent_prompt = agent_prompt
        self._judge_prompt = judge_prompt
        self._rag: Optional[RAGEngine] = None
        self._registry: Optional[ToolRegistry] = None
        self._agent: Optional[ReactAgent] = None
        self._tracker: Optional[ContradictionTracker] = None
        self._injected_gate = gate
        self._citation_asker: Any = False      # False = not built yet, None = unavailable


    # ------------------------------------------------------------- components

    @property
    def llm(self) -> OpenAICompatClient:
        if self._llm is None:
            self._llm = OpenAICompatClient(
                base_url=self.cfg.llm_base_url,
                api_key=self.cfg.llm_api_key,
                model=self.cfg.llm_model,
                max_output_tokens=self.cfg.max_output_tokens,
                temperature=self.cfg.temperature,
                reasoning_effort=self.cfg.llm_reasoning_effort,
            )
        return self._llm

    @property
    def embedder(self):
        if self._embedder is None:
            ollama = OllamaEmbedder(
                base_url=self.cfg.ollama_base_url,
                model=self.cfg.embed_model,
                query_prefix=self.cfg.embed_query_prefix,
                doc_prefix=self.cfg.embed_doc_prefix,
                batch_size=self.cfg.embed_batch_size,
            )
            if ollama.healthcheck():
                self._embedder = ollama
            else:
                try:
                    from .embeddings import SentenceTransformerEmbedder
                    self._embedder = SentenceTransformerEmbedder(batch_size=self.cfg.embed_batch_size)
                    log.info(
                        "Ollama unreachable at %s; using local SentenceTransformerEmbedder (all-MiniLM-L6-v2)",
                        self.cfg.ollama_base_url,
                    )
                except Exception as e:
                    log.warning("could not initialize SentenceTransformerEmbedder (%s); using OllamaEmbedder", e)
                    self._embedder = ollama
        return self._embedder

    @property
    def rag(self) -> RAGEngine:
        if self._rag is None:
            self._rag = RAGEngine(
                self.db, self.embedder,
                mode=self.cfg.retrieval_mode,
                rrf_k=self.cfg.rrf_k,
                vector_weight=self.cfg.hybrid_vector_weight,
                keyword_weight=self.cfg.hybrid_keyword_weight,
            )
        return self._rag

    @property
    def registry(self) -> ToolRegistry:
        if self._registry is None:
            self._registry = ToolRegistry(self.db, self.rag,
                                          max_obs_chars=self.cfg.agent_max_obs_chars,
                                          default_top_k=self.cfg.rag_top_k)
        return self._registry

    @property
    def intent_router(self) -> IntentRouter:
        if self._intent_router is None:
            cache_path = self.cfg.intent_cache_path or str(
                self.cfg.data_dir / "intent_cache.json")
            variant = self.cfg.prompt_intent or "v1"
            self._intent_router = IntentRouter(
                llm=self.llm if (self.cfg.intent_enabled
                                 and self.cfg.llm_configured) else None,
                model=self.cfg.intent_model or self.cfg.llm_model,
                cache_path=cache_path,
                prompt=INTENT_SYSTEM_V2 if variant == "v2" else INTENT_SYSTEM,
                prompt_version=variant,
                max_tokens=self.cfg.intent_max_tokens,
                reasoning_effort=self.cfg.intent_reasoning_effort,
            )
        return self._intent_router

    @property
    def agent(self) -> ReactAgent:
        if self._agent is None:
            variant = self._agent_prompt or self.cfg.prompt_agent
            prompt = AGENT_SYSTEM_V2 if variant == "v2" else AGENT_SYSTEM
            self._agent = ReactAgent(self.llm, self.registry,
                                     max_steps=self.cfg.agent_max_steps,
                                     system_prompt=prompt)
        return self._agent

    @property
    def tracker(self) -> ContradictionTracker:
        if self._tracker is None:
            self._tracker = ContradictionTracker(
                db=self.db,
                rag=self.rag,
                # stage-2 decision model: the only judge of a candidate pair
                gate=self._injected_gate or build_gate_from_config(self.cfg),
                verifier=ContradictionVerifier(self.llm),
                cross_doc_only=self.cfg.scan_cross_doc_only,
                min_similarity=self.cfg.scan_min_similarity,
                max_candidates=self.cfg.scan_max_candidates,
                workers=self.cfg.scan_workers,
            )
        return self._tracker

    def _pipeline(self, on_progress=None) -> IngestPipeline:
        return IngestPipeline(
            db=self.db,
            embedder=self.embedder,
            llm=self.llm if self.cfg.ingest_llm_enabled else None,
            window_chars=self.cfg.ingest_window_chars,
            use_llm=self.cfg.ingest_llm_enabled,
            chunk_target=self.cfg.chunk_target_chars,
            chunk_max=self.cfg.chunk_max_chars,
            on_progress=on_progress,
            reasoning_effort=getattr(self.cfg, "ingest_reasoning_effort", ""),
            max_parse_tokens=getattr(self.cfg, "ingest_max_tokens", 3000),
            parallel_windows=getattr(self.cfg, "ingest_parallel_windows", 1),
            heuristic_max_chars=getattr(self.cfg, "ingest_heuristic_max_chars", 0),
            brief_words=getattr(self.cfg, "ingest_brief_words", 1000),
        )

    # --------------------------------------------------------------- documents

    def upload_document(self, path, name: Optional[str] = None,
                        on_progress: Optional[Callable[[Dict], None]] = None,
                        start_scan: bool = True) -> Dict[str, Any]:
        """Blocking upload + ingestion; returns pipeline result dict."""
        path = Path(path)
        name = self.db.unique_name((name or path.stem).strip() or path.stem)
        result = self._pipeline(on_progress).ingest(path, name)
        if start_scan:
            # fire-and-forget background cascade (RAG -> gate -> verifier -> flags)
            self.tracker.scan_document_async(result["doc_id"])
        return result

    def upload_document_async(self, path, name: Optional[str] = None) -> Dict[str, Any]:
        """Non-blocking upload for the web UI: creates the doc row + a job row
        immediately, then ingests on a background thread."""
        path = Path(path)
        name = self.db.unique_name((name or path.stem).strip() or path.stem)
        doc_id = self.db.add_document(
            name=name, filename=path.name, fmt=path.suffix.lstrip(".").lower(),
            size=path.stat().st_size if path.exists() else 0, status="pending",
        )

        def work() -> None:
            try:
                result = self._pipeline().ingest(path, name, doc_id=doc_id)
                self.tracker.scan_document_async(result["doc_id"])
            except Exception:
                log.exception("background ingestion failed for '%s'", name)
                doc = self.db.get_document(doc_id)
                if doc and doc.get("status") not in ("error", "ready"):
                    self.db.update_document(doc_id, status="error")

        threading.Thread(target=work, daemon=True,
                         name=f"ingest-{doc_id}").start()
        return {"doc_id": doc_id, "doc_name": name}

    def delete_document(self, doc: Any) -> bool:
        d = self.db.get_document(doc)
        if not d and isinstance(doc, str):
            d = self.db.find_document_fuzzy(doc)
        if not d:
            return False
        ok = self.db.delete_document(d["id"])
        if ok:
            log.info("deleted document '%s' (+chunks, +contradictions)", d["name"])
            # the row is gone; leaving the upload behind leaks disk on a long run
            stored = (d.get("filename") or "").strip()
            if stored:
                path = self.cfg.data_dir / "uploads" / Path(stored).name
                try:
                    path.unlink(missing_ok=True)
                except OSError as e:
                    log.info("could not remove stored upload %s: %s", path, e)
        return ok

    def list_documents(self) -> List[Dict[str, Any]]:
        out = []
        for d in self.db.list_documents():
            d["latest_job"] = self.db.latest_job_for_doc(d["id"])
            out.append(d)
        return out

    # -------------------------------------------------------------------- ask

    def _document_listing(self) -> str:
        docs = self.db.list_documents()
        if not docs:
            return ("No policy documents have been ingested yet. An admin "
                    "can upload txt / md / pdf files from the web UI or "
                    "the CLI.")
        lines = [f"{len(docs)} policy document(s) are currently ingested:"]
        for d in docs:
            flags = d.get("open_contradictions") or 0
            note = f"  !! {flags} OPEN contradiction flag(s)" if flags else ""
            lines.append(f"- {d['name']} ({d.get('format') or '?'}, "
                         f"{d.get('chunk_count', 0)} passages, "
                         f"status: {d['status']}){note}")
        lines.append("Ask me anything about their contents and I will "
                     "answer with citations.")
        return "\n".join(lines)

    def corpus_briefs(self, limit: int = 8, chars: int = 240) -> List[Dict[str, str]]:
        """What the triage layer is allowed to know about the corpus.

        One line per ready document, bounded on both axes: a triage prompt that
        grows with the corpus is a triage prompt that eventually stops classifying.
        An empty brief is honest - the model is told the brief is missing rather
        than shown a blank line it would read as "this document covers nothing".
        """
        out: List[Dict[str, str]] = []
        for d in self.db.list_documents():
            if d.get("status") != "ready":
                continue
            brief = (d.get("summary") or "").strip()
            out.append({"name": d["name"], "brief": brief[:chars]})
            if len(out) >= limit:
                break
        return out

    def ask(self, question: str,
            history: Optional[List[Dict[str, str]]] = None,
            on_event=None) -> AgentResult:
        """Full ask pipeline: LLM triage -> route -> agent -> answer.

        There is no post-answer LLM verification in this path: at model parity
        a same-model judge added ~6s and flagged roughly one honest answer in
        five, so grounding is carried by the agent prompt (and by deterministic
        citation resolution against the store) instead. The faithfulness judge
        survives as an offline audit tool - see eval/calibrate_judge.py and
        dev_docs/DATAFLOW.md §"why there is no verifier in the path".

        The triage layer is the only component allowed to refuse *before*
        retrieval, and only for non-HR input (chit-chat, injections, empty
        strings). Off-policy workplace questions always reach the agent,
        because only the agent has read the corpus and only it may refuse
        for missing coverage.
        """
        def _emit(kind: str, payload: Dict[str, Any]) -> None:
            if on_event is None:
                return
            try:
                on_event(kind, payload)
            except Exception:
                log.debug("event listener failed", exc_info=True)

        _emit("stage", {"name": "triage"})
        plan = self.intent_router.classify(question, corpus=self.corpus_briefs())
        plan_dict = plan.to_dict()
        _emit("stage", {"name": "routed", "intent": plan.intent,
                        "route": plan.route,
                        "question": plan.de_noised_question or question})

        # ---------------------------------------------------- deterministic routes
        if plan.route == "FAST_REFUSE":
            if not (question or "").strip():
                answer = ("Please ask a question about the ingested policy "
                          "documents (for example: \"How many casual leave days "
                          "can I carry forward?\").")
            else:
                answer = REFUSAL_TEXT
            return AgentResult(
                answer=answer,
                steps=[{"type": "direct",
                        "content": f"intent gate: {plan.intent} -> FAST_REFUSE "
                                   f"({plan.refusal_reason})"}],
                intent=plan.intent,
                intent_route=plan.route,
                intent_plan=plan_dict,
            )

        if plan.intent == "GREETING":
            return AgentResult(
                answer=GREETING_REPLY,
                steps=[{"type": "direct", "content": "intent gate: GREETING"}],
                intent=plan.intent,
                intent_route=plan.route,
                intent_plan=plan_dict,
            )

        if plan.intent == "META_DOCUMENTS":
            return AgentResult(
                answer=self._document_listing(),
                steps=[{"type": "direct",
                        "content": "intent gate: META_DOCUMENTS -> listing from the store"}],
                intent=plan.intent,
                intent_route=plan.route,
                intent_plan=plan_dict,
            )

        # ------------------------------------------------------------- the agent
        result = self.agent.run(plan.de_noised_question or question,
                               history=history, plan=plan, on_event=on_event)

        # the answer cites what it quoted; this adds what it read and relied on
        self.recover_citations(result)
        result.intent = plan.intent
        result.intent_route = plan.route
        result.intent_plan = plan_dict

        if plan.intent == "SENSITIVE_HR_ESCALATION":
            result.answer = ESCALATION_NOTE + (result.answer or "")
        return result

    # ------------------------------------------------------- citation recovery

    def citation_asker(self):
        """The decision-model ask function, built once. None when unavailable."""
        if self._citation_asker is False:
            from .citation_recovery import build_asker
            self._citation_asker = (build_asker(self.cfg)
                                    if getattr(self.cfg, "citation_recovery", True) else None)
        return self._citation_asker

    def recover_citations(self, result: Any) -> Any:
        """Append the sources the agent read but the answer did not cite.

        Mutates *result* (citations plus an audit step) and returns it. Silent no-op when
        recovery is off, when the decision model is not configured, or when the agent read
        nothing. It can only add citations that resolve to real sections: candidates come
        from chunk ids the agent saw, never from the model's imagination.
        """
        if not getattr(self.cfg, "citation_recovery", True):
            return result
        ask = self.citation_asker()
        if ask is None:
            return result
        seen = getattr(result, "seen_chunk_ids", None) or []
        if not seen or not (result.answer or "").strip():
            return result
        from .citation_recovery import build_candidates, recover
        candidates = build_candidates(
            self.db, seen, result.citations or [],
            max_candidates=int(getattr(self.cfg, "citation_recovery_max_candidates", 12)),
            snippet_chars=int(getattr(self.cfg, "citation_recovery_snippet_chars", 700)),
        )
        if not candidates:
            return result
        added, scores = recover(
            ask, result.answer, candidates,
            threshold=float(getattr(self.cfg, "citation_recovery_threshold", 0.4)),
            max_added=int(getattr(self.cfg, "citation_recovery_max_added", 3)),
        )
        if added:
            result.citations = list(result.citations or []) + added
        result.steps.append({
            "type": "citations_recovered",
            "candidates": len(candidates),
            "added": added,
            "threshold": float(getattr(self.cfg, "citation_recovery_threshold", 0.4)),
            "scores": scores,
        })
        return result

    # ---------------------------------------------------------- contradictions

    def list_contradictions(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.db.list_contradictions(status)

    def set_contradiction_status(self, cid: int, status: str) -> bool:
        return self.db.set_contradiction_status(cid, status)

    def scan(self, doc: Optional[str] = None, blocking: bool = True) -> Any:
        """Re-run the contradiction cascade for one doc (by name/id) or all."""
        if doc is not None:
            d = self.db.get_document(doc) or self.db.find_document_fuzzy(str(doc))
            if not d:
                raise ValueError(f"unknown document: {doc}")
            if blocking:
                return self.tracker.scan_document(d["id"])
            job_id = self.tracker.scan_document_async(d["id"])
            return {"job_id": job_id}
        results = []
        for d in self.db.list_documents():
            if d["status"] == "ready":
                results.append(self.tracker.scan_document(d["id"]))
        return results

    # ------------------------------------------------------------------- misc

    def job_status(self, job_id: int) -> Optional[Dict[str, Any]]:
        return self.db.get_job(job_id)

    def health(self) -> Dict[str, Any]:
        docs = self.db.list_documents()
        open_flags = self.db.list_contradictions("OPEN")
        return {
            "llm_configured": self.cfg.llm_configured,
            "llm_model": self.cfg.llm_model,
            "llm_base_url": self.cfg.llm_base_url,
            "embed_model": self.cfg.embed_model,
            "ollama_base_url": self.cfg.ollama_base_url,
            "documents": len(docs),
            "ready_documents": sum(1 for d in docs if d["status"] == "ready"),
            "open_contradictions": len(open_flags),
            "jev_gate_threshold": self.cfg.jev_gate_threshold,
        }

    def resolve_citation(self, cite: str) -> Dict[str, Any]:
        """`[doc § section]` -> the stored passage, so a reader can verify the claim.

        Never invents a passage: an unmatched section comes back with the sections
        the document really has (app/citations.py).
        """
        from app.citations import CitationResolver
        return CitationResolver.resolve_citation(self, cite)

    def get_chunk(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        chunk = self.db.get_chunk(chunk_id)
        if not chunk:
            return None
        doc = self.db.get_document(chunk["doc_id"])
        chunk["doc_name"] = (doc or {}).get("name", "?")
        chunk.pop("embedding", None)   # raw float32 BLOB: not JSON-serialisable
        chunk.pop("norm", None)
        return chunk
