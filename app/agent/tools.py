"""Tool registry exposed to the ReAct agent (OpenAI function-calling specs)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..db import Database
from ..rag import RAGEngine

TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_policy_docs",
            "description": (
                "Semantic search over the ingested policy documents. ALWAYS call "
                "this before answering any policy question. Use a focused query "
                "with the key terms of the question; reformulate and search again "
                "if the results are weak."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, e.g. 'remote work allowance notice period'",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of passages to return (default: server setting)",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_policy_chunk",
            "description": (
                "Return the FULL text of one policy passage by its chunk id (as "
                "reported by search_policy_docs). Use when a snippet looks "
                "truncated or ambiguous."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "integer",
                        "description": "numeric chunk id from the search results",
                    },
                },
                "required": ["chunk_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_policy_documents",
            "description": (
                "List all ingested policy documents with status, chunk count and "
                "a short summary. Use when the user asks what documents exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_summary",
            "description": (
                "Return the rolling summary and key metadata (key terms, main "
                "statements) of one policy document, by document name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_name": {
                        "type": "string",
                        "description": "exact document name from list_policy_documents",
                    },
                },
                "required": ["doc_name"],
                "additionalProperties": False,
            },
        },
    },
]


class ToolRegistry:
    """Executes tool calls against the local store; observations are strings."""

    def __init__(self, db: Database, rag: RAGEngine, max_obs_chars: int = 4000,
                 default_top_k: int = 6) -> None:
        self.db = db
        self.rag = rag
        self.max_obs = max(500, max_obs_chars)
        self.default_top_k = max(1, int(default_top_k))
        self._handlers = {
            "search_policy_docs": self._search,
            "read_policy_chunk": self._read_chunk,
            "list_policy_documents": self._list_docs,
            "get_document_summary": self._doc_summary,
        }

    # ------------------------------------------------------------- properties

    @property
    def specs(self) -> List[Dict[str, Any]]:
        return TOOL_SPECS

    @property
    def names(self) -> List[str]:
        return [t["function"]["name"] for t in TOOL_SPECS]

    def get_spec(self, name: str) -> Optional[Dict[str, Any]]:
        for t in TOOL_SPECS:
            if t["function"]["name"] == name:
                return t["function"]
        return None

    def doc_names(self) -> List[str]:
        return [d["name"] for d in self.db.list_documents()]

    def chunk_exists(self, chunk_id: Any) -> bool:
        try:
            return self.db.get_chunk(int(chunk_id)) is not None
        except (TypeError, ValueError):
            return False

    # ---------------------------------------------------------------- execute

    def execute(self, name: str, args: Dict[str, Any]) -> str:
        handler = self._handlers.get(name)
        if handler is None:
            return f"ERROR: unknown tool '{name}' (available: {', '.join(self.names)})"
        try:
            out = str(handler(args or {}))
        except Exception as e:
            out = f"ERROR: tool '{name}' failed: {e}"
        if len(out) > self.max_obs:
            out = (
                out[: self.max_obs]
                + f"\n...[observation truncated to {self.max_obs} chars; "
                  "call read_policy_chunk for the full passage]"
            )
        return out

    # -------------------------------------------------------------- handlers

    def _search(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()
        top_k = args.get("top_k") or self.default_top_k
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = self.default_top_k
        hits = self.rag.search(query, top_k=max(1, min(top_k, 20)))
        if not hits:
            return (
                "No matching passages found in the ingested policy documents. "
                "Try a different query, or call list_policy_documents to see "
                "what is ingested."
            )
        lines = [f'Found {len(hits)} passages for query "{query}" '
                 f"(cosine similarity in brackets):"]
        for h in hits:
            snippet = h.text if len(h.text) <= 900 else h.text[:900] + " ..."
            lines.append(
                f"\n[chunk {h.chunk_id}] Doc: {h.doc_name} | "
                f"Section: {h.section or '-'} | (score {h.score:.3f})\n"
                f'"""\n{snippet}\n"""'
            )
        return "\n".join(lines)

    def _read_chunk(self, args: Dict[str, Any]) -> str:
        try:
            cid = int(args.get("chunk_id"))
        except (TypeError, ValueError):
            return "ERROR: chunk_id must be an integer."
        chunk = self.db.get_chunk(cid)
        if not chunk:
            return (f"ERROR: no chunk with id {cid}. Use the chunk ids exactly "
                    "as reported by search_policy_docs.")
        doc = self.db.get_document(chunk["doc_id"])
        doc_name = (doc or {}).get("name", "?")
        return (
            f"[chunk {cid}] Doc: {doc_name} | Section: {chunk['section'] or '-'} | "
            f"Breadcrumb: {chunk['breadcrumb'] or '-'}\n"
            f'"""\n{chunk["text"]}\n"""'
        )

    def _list_docs(self, args: Dict[str, Any]) -> str:
        docs = self.db.list_documents()
        if not docs:
            return "No documents have been ingested yet."
        lines = [f"{len(docs)} document(s) ingested:"]
        for d in docs:
            s = (d.get("summary") or "").strip()
            line = (f"- {d['name']} [{d['status']}] ({d.get('chunk_count', 0)} chunks, "
                    f"{d.get('format') or '?'}, {d.get('size_bytes', 0)} bytes)")
            if s:
                line += f" — {s[:200]}"
            lines.append(line)
        return "\n".join(lines)

    def _doc_summary(self, args: Dict[str, Any]) -> str:
        name = str(args.get("doc_name", "")).strip()
        doc = self.db.find_document_fuzzy(name)
        if not doc:
            avail = ", ".join(self.doc_names()) or "(none)"
            return f"ERROR: no document named '{name}' (available: {avail})"
        try:
            meta = json.loads(doc.get("meta_json") or "{}")
        except json.JSONDecodeError:
            meta = {}
        parts = [f"Document: {doc['name']} (status: {doc['status']}, "
                 f"{doc.get('chunk_count', 0)} chunks)",
                 f"Rolling summary: {doc.get('summary') or '(none)'}"]
        terms = meta.get("key_terms") or []
        if terms:
            parts.append("Key terms: " + ", ".join(terms[:30]))
        statements = meta.get("statements") or []
        if statements:
            parts.append("Main statements (extracted during ingestion):")
            parts += [f"  - {s[:300]}" for s in statements[:15]]
        return "\n".join(parts)
