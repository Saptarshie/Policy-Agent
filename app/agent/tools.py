"""Tool registry exposed to the ReAct agent (OpenAI function-calling specs)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..db import Database
from ..rag import RAGEngine
from .calc import calculate
from .scratchpad import Scratchpad

TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_policy_docs",
            "description": (
                "Search the ingested policy documents. ALWAYS call this before "
                "answering any policy question. Two ways to say how deep to go, "
                "and you may use AT MOST ONE per call: top_k (\"give me the best "
                "6\") or min_similarity (\"give me everything that scores at least "
                "0.55\"). Use min_similarity when the question is broad or you "
                "suspect several documents say something about it and a small k "
                "would hide one; use top_k for an ordinary focused lookup. "
                "Neither given = the server default."
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
                        "description": "Return at most this many passages (mutually exclusive with min_similarity)",
                    },
                    "min_similarity": {
                        "type": "number",
                        "description": "Return every passage with cosine similarity >= this (0-1; mutually exclusive with top_k)",
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
    {
        "type": "function",
        "function": {
            "name": "list_contradictions",
            "description": (
                "List the contradictions the scanner found between passages "
                "(verified ones include what each side says and why they "
                "conflict). Use this when the user asks whether the documents "
                "disagree, contradict, conflict or are inconsistent - it is the "
                "audit trail of exactly those pairs, so prefer it to guessing "
                "from search results. Returns nothing for a corpus with no "
                "flags, which means no conflict was found, not that none exists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "OPEN (default), resolved, dismissed, or all",
                        "enum": ["OPEN", "resolved", "dismissed", "all"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "maximum flags to return (default 10)",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate arithmetic exactly. Use this for ANY computed figure - "
                "pro-rated entitlements, totals, per-night or per-day costs, tax on "
                "a stated salary, percentage changes - instead of doing the maths in "
                "your head. Supply the numbers you read or were given as named "
                "variables, and state the calculation in the answer. "
                "Allowed: + - * / // % **, parentheses, abs, round, min, max."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic over the variable names, e.g. days * months / 12",
                    },
                    "variables": {
                        "type": "object",
                        "description": "Numbers the expression refers to, e.g. {days: 12, months: 7}",
                        "additionalProperties": True,
                    },
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scratchpad_write",
            "description": (
                "Save a finding to your working notes for this question. Use it when "
                "the work spans several documents or many steps: park the exact "
                "number/rule plus its citation, then keep reading. It is NOT evidence "
                "and NOT visible to the employee - write what a passage actually says, "
                "and re-read the notes with scratchpad_read before composing a long "
                "answer. For a single-fact question you do not need it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Short note name, e.g. leave_cap"},
                    "text": {"type": "string", "description": "The finding, with its citation"},
                    "append": {"type": "boolean",
                               "description": "Append to the note instead of replacing it (default false)"},
                },
                "required": ["key", "text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scratchpad_read",
            "description": (
                "Read your working notes back - one note by key, or all of them when "
                "key is omitted. Read before writing a long or multi-part answer so "
                "you use what you already found instead of re-deriving or guessing it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Note to read; omit for all notes"},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scratchpad_edit",
            "description": (
                "Correct one note in place: replace the first occurrence of find with "
                "replace. Use it when a later passage supersedes an earlier finding, "
                "e.g. an amendment changed the cap you wrote down."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "find": {"type": "string", "description": "Exact text to replace"},
                    "replace": {"type": "string", "description": "New text (may be empty)"},
                },
                "required": ["key", "find", "replace"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scratchpad_delete",
            "description": "Delete a note that is wrong or no longer relevant.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scratchpad_list",
            "description": "List your notes and their sizes (cheaper than reading them all).",
            "parameters": {"type": "object", "properties": {},
                           "required": [], "additionalProperties": False},
        },
    },
]


class ToolRegistry:
    """Executes tool calls against the local store; observations are strings."""

    def __init__(self, db: Database, rag: RAGEngine, max_obs_chars: int = 4000,
                 default_top_k: int = 6, max_candidates: int = 200) -> None:
        self.db = db
        self.rag = rag
        self.max_obs = max(500, max_obs_chars)
        self.default_top_k = max(1, int(default_top_k))
        # safety cap for min_similarity searches: the floor decides relevance,
        # this only stops a pathological corpus from flooding the transcript
        self.max_candidates = max(1, int(max_candidates))
        self._handlers = {
            "search_policy_docs": self._search,
            "read_policy_chunk": self._read_chunk,
            "list_policy_documents": self._list_docs,
            "get_document_summary": self._doc_summary,
            "list_contradictions": self._contradictions,
            "calculate": self._calculate,
            "scratchpad_write": self._scratch_write,
            "scratchpad_read": self._scratch_read,
            "scratchpad_edit": self._scratch_edit,
            "scratchpad_delete": self._scratch_delete,
            "scratchpad_list": self._scratch_list,
        }
        # set per ask() call: working notes belong to one question, so two
        # concurrent questions can never read each other's notes
        self.scratch: Optional[Scratchpad] = None

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

    def execute(self, name: str, args: Dict[str, Any],
                scratch: Optional[Scratchpad] = None) -> str:
        self.scratch = scratch
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
        raw_k, raw_sim = args.get("top_k"), args.get("min_similarity")
        if raw_k not in (None, "") and raw_sim not in (None, ""):
            return ("ERROR: top_k and min_similarity are mutually exclusive - "
                    "pick one: a count (top_k) or a relevance floor "
                    "(min_similarity). Default with neither is top_k=server default.")
        if raw_sim not in (None, ""):
            try:
                sim = float(raw_sim)
            except (TypeError, ValueError):
                return f"ERROR: min_similarity must be a number, got {raw_sim!r}"
            if not 0.0 < sim <= 1.0:
                return "ERROR: min_similarity must be in (0, 1]."
            hits = self.rag.search(query, min_score=sim,
                                   max_candidates=self.max_candidates,
                                   mode="vector")
            how = f"min_similarity={sim:g}"
        else:
            try:
                top_k = int(raw_k) if raw_k not in (None, "") else self.default_top_k
            except (TypeError, ValueError):
                top_k = self.default_top_k
            hits = self.rag.search(query, top_k=max(1, min(top_k, 40)))
            how = f"top_k={max(1, min(top_k, 40))}"
        if not hits:
            return (
                f"No matching passages found for {how}. "
                "Try a different query, widen it (lower min_similarity or raise "
                "top_k), or call list_policy_documents to see what is ingested."
            )
        lines = [f'Found {len(hits)} passages for query "{query}" ({how}; '
                 "score in brackets):"]
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

    # ------------------------------------------------- working memory + maths

    def _contradictions(self, args: Dict[str, Any]) -> str:
        status = str(args.get("status") or "OPEN").strip()
        limit = max(1, min(int(args.get("limit") or 10), 25))
        rows = self.db.list_contradictions(None if status.lower() == "all" else status)
        if not rows:
            return (f"No {status} contradictions are recorded. Treat this as "
                    "\"none found so far\", not as proof the documents agree: "
                    "the scanner only compares passages it can see, and flags "
                    "can also have been dismissed by an admin.")
        lines = [f"{len(rows)} {status} contradiction(s); showing up to {limit}:"]
        for f in rows[:limit]:
            tag = "VERIFIED" if f.get("verified") else "UNVERIFIED"
            lines.append(
                f"\nFlag {f['id']} - {tag}, {f.get('status')}, "
                f"confidence {float(f.get('confidence') or 0):.2f}, "
                f"gate {float(f.get('gate_score') or 0):.2f}")
            lines.append(f"  A: {f.get('doc_a')} § {f.get('section_a')} "
                         f"(chunk {f.get('chunk_a_id')})")
            lines.append(f"     \"{(f.get('text_a') or '')[:300]}\"")
            lines.append(f"  B: {f.get('doc_b')} § {f.get('section_b')} "
                         f"(chunk {f.get('chunk_b_id')})")
            lines.append(f"     \"{(f.get('text_b') or '')[:300]}\"")
            lines.append(f"  Why: {f.get('explanation') or '(no explanation)'}")
        if len(rows) > limit:
            lines.append(f"\n...{len(rows) - limit} more not shown")
        # the flag is the scanner's own analysis, not a source: citing it inside
        # brackets produces a citation that resolves to no document (and the
        # citation check flags it), so the passages are what must be cited
        lines.append(
            "\nCite the PASSAGES, not the flag: [doc_a § section_a] and "
            "[doc_b § section_b]. Square brackets are reserved for document "
            "citations - mention a flag in prose instead (\"the scanner flags "
            "this as contradiction 4\"), never as [contradiction 4].")
        return "\n".join(lines)

    def _need_scratch(self) -> Scratchpad:
        if self.scratch is None:
            raise RuntimeError("the scratchpad is not available in this context")
        return self.scratch

    def _calculate(self, args: Dict[str, Any]) -> str:
        result = calculate(str(args.get("expression", "")),
                           args.get("variables") or {})
        return (f"{result['value']}   (expression: {result['resolved']})\n"
                "State this figure with its inputs in the answer: a computed number "
                "must be traceable, and the documents do not state it.")

    def _scratch_write(self, args: Dict[str, Any]) -> str:
        pad = self._need_scratch()
        key, text = args.get("key", ""), args.get("text", "")
        return pad.append(key, text) if args.get("append") else pad.write(key, text)

    def _scratch_read(self, args: Dict[str, Any]) -> str:
        return self._need_scratch().read(str(args.get("key", "") or ""))

    def _scratch_edit(self, args: Dict[str, Any]) -> str:
        return self._need_scratch().edit(args.get("key", ""), args.get("find", ""),
                                         args.get("replace", ""))

    def _scratch_delete(self, args: Dict[str, Any]) -> str:
        return self._need_scratch().delete(args.get("key", ""))

    def _scratch_list(self, args: Dict[str, Any]) -> str:
        return self._need_scratch().list_keys()
