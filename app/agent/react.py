"""ReAct agent with a tool_call loop.

Per turn the agent may emit native tool_calls (OpenAI-style) or, on
endpoints without function calling, a text ``<tool_call>{...}</tool_call>``
block (also parsed). Every tool call passes through the regex-heuristic
schema repair + validation layer; when repair fails the agent receives a
precise ``TOOL_CALL_ERROR`` observation instead of a fabricated result, and
can re-emit a corrected call. The loop ends with a grounded final answer
whose citations are then verified against the store (anti-hallucination).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..llm import LLMError, LLMResponse, OpenAICompatClient
from ..naming import resolve_one
from ..prompts import AGENT_SYSTEM
from ..schema_repair import extract_tool_calls_from_text, make_tool_error_feedback, repair_json, validate_arguments
from .scratchpad import Scratchpad
from .tools import ToolRegistry

log = logging.getLogger(__name__)

_CITE_RE = re.compile(r"\[([^\[\]]+)\]")
_CHUNK_REF_RE = re.compile(r"\[chunk (\d+)\]")


@dataclass
class AgentResult:
    answer: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    citation_warnings: List[str] = field(default_factory=list)
    seen_chunk_ids: List[int] = field(default_factory=list)
    judge_context_ids: List[int] = field(default_factory=list)
    faithful: Optional[bool] = None
    answer_shape: str = ""                   # answer|partial|refusal|chitchat|unknown
    declined_parts: List[str] = field(default_factory=list)
    scratch_notes: int = 0                   # notes the agent kept while working
    parallel_turns: int = 0                  # turns that carried more than one tool call
    turns: int = 0                           # model calls = loop steps consumed (the cap)
    unsupported_claims: List[str] = field(default_factory=list)
    tool_calls: int = 0
    elapsed_s: float = 0.0
    error: Optional[str] = None
    intent: str = "POLICY_DIRECT_LOOKUP"     # triage category
    intent_route: str = "REACT_LOOP"         # REACT_LOOP | DIRECT_REPLY | FAST_REFUSE
    intent_plan: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "steps": self.steps,
            "citations": self.citations,
            "citation_warnings": self.citation_warnings,
            "seen_chunk_ids": self.seen_chunk_ids,
            "judge_context_ids": self.judge_context_ids,
            "faithful": self.faithful,
            "answer_shape": self.answer_shape,
            "scratch_notes": self.scratch_notes,
            "parallel_turns": self.parallel_turns,
            "turns": self.turns,
            "declined_parts": self.declined_parts,
            "unsupported_claims": self.unsupported_claims,
            "tool_calls": self.tool_calls,
            "elapsed_s": round(self.elapsed_s, 2),
            "error": self.error,
            "intent": self.intent,
            "intent_route": self.intent_route,
            "intent_plan": self.intent_plan,
        }


def _match_doc(part: str, names: List[str]) -> Optional[str]:
    """The document this reference names, or None.

    Delegates to :mod:`app.naming`, which requires a *unique* candidate: the old
    first-substring-wins loop resolved "Remote Work Policy v2" to
    `remote_work_policy_v1`, i.e. it attributed a citation of the amendment to
    the superseded policy.
    """
    return resolve_one(part, names)


def _looks_like_citation(ref: str) -> bool:
    """Only 'doc § section' / 'doc #id' shapes are citations.

    Everything else in square brackets (markdown links, footnote markers,
    [1], emphasis) is ignored instead of being reported as a broken
    citation - the old behaviour produced warnings for ordinary prose.
    """
    if "§" in ref:
        return True
    if re.search(r"#\s*\d+", ref):
        return True
    return False


def check_citations(answer: str, doc_names: List[str],
                     chunk_exists: Callable[[Any], bool]
                     ) -> tuple:  # (citations, warnings)
    citations: List[str] = []
    warnings: List[str] = []
    for ref in _CITE_RE.findall(answer or ""):
        doc_part = re.split(r"[§#]", ref, maxsplit=1)[0].strip().strip('"“”\'')
        if not doc_part:
            continue
        matched = _match_doc(doc_part, doc_names)
        if not _looks_like_citation(ref):
            # a bracket that names a document (e.g. "[leave_policy, section
            # 3.2]") is still worth resolving; anything else is prose
            if matched:
                citations.append(ref)
            continue
        if not matched:
            warnings.append(
                f"citation '{ref}' does not match any ingested document"
            )
            continue
        cid_m = re.search(r"#\s*(\d+)", ref)
        if cid_m and not chunk_exists(cid_m.group(1)):
            warnings.append(
                f"citation '{ref}' references unknown chunk id {cid_m.group(1)}"
            )
            continue
        citations.append(ref)
    # de-duplicate while keeping first-seen order
    seen = set()
    unique = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique, warnings


class ReactAgent:
    def __init__(
        self,
        llm: OpenAICompatClient,
        registry: ToolRegistry,
        max_steps: int = 12,
        system_prompt: str = AGENT_SYSTEM,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.max_steps = max(2, max_steps)
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------ public

    def run(self, question: str,
            history: Optional[List[Dict[str, str]]] = None,
            plan=None,
            on_event=None) -> AgentResult:
        """Run the ReAct loop.

        ``on_event(kind, payload)`` is called as the loop progresses
        (``turn`` / ``step`` / ``error``), so a caller can stream the trace.
        A raising callback must never break the answer, so its errors are
        swallowed here.

        ``plan`` is an optional :class:`app.agent.intent.IntentPlan`; its
        de-noised question is the one to answer, and its suggested searches
        are injected as a second system message (routing metadata, not user
        content - so it cannot be used for prompt injection).
        """
        t0 = time.time()
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        if plan is not None:
            block = plan.prompt_block() if hasattr(plan, "prompt_block") else ""
            if block:
                messages.append({"role": "system", "content": block})
        for h in history or []:
            role = h.get("role")
            content = (h.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": question})

        def _emit_step(step: Dict[str, Any], full_obs: Optional[str] = None) -> None:
            payload = step
            if full_obs is not None and step.get("observation") is not None:
                payload = {**step, "observation": full_obs[:1200]}
            _emit("step", payload)

        def _emit(kind: str, payload: Dict[str, Any]) -> None:
            if on_event is None:
                return
            try:
                on_event(kind, payload)
            except Exception:                     # a broken listener cannot
                log.debug("event listener failed", exc_info=True)   # stop an answer

        steps: List[Dict[str, Any]] = []
        n_tool_calls = 0
        parallel_turns = 0
        turns = 0
        # working memory for THIS question only (in memory, never persisted)
        scratch = Scratchpad()
        error: Optional[str] = None
        answer = ""
        seen_chunk_ids: List[int] = []

        def _remember_chunks(observation: str) -> None:
            for m in _CHUNK_REF_RE.finditer(observation or ""):
                cid = int(m.group(1))
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.append(cid)

        for _step in range(self.max_steps):
            turns += 1
            _emit("turn", {"n": _step + 1})
            try:
                resp = self.llm.chat(messages, tools=self.registry.specs)
            except LLMError as e:
                error = str(e)
                _emit("error", {"error": error})
                break

            # ---- native tool calls
            if resp.tool_calls:
                if resp.content and resp.content.strip():
                    steps.append({"type": "thought", "content": resp.content.strip()})
                    _emit("step", steps[-1])
                messages.append(self._native_assistant_msg(resp))
                if len(resp.tool_calls) > 1:
                    parallel_turns += 1
                for tc in resp.tool_calls:
                    obs, step = self._run_tool(tc.name, tc.arguments, scratch)
                    _remember_chunks(obs)
                    steps.append(step)
                    _emit_step(step, obs)
                    if step["type"] == "tool_call":
                        n_tool_calls += 1
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": obs})
                continue

            # ---- text-based tool calls (endpoints without function calling)
            text_calls = extract_tool_calls_from_text(
                resp.content or "", self.registry.names)
            if text_calls:
                messages.append(self._text_assistant_msg(resp, text_calls))
                if len(text_calls) > 1:
                    parallel_turns += 1
                for k, c in enumerate(text_calls):
                    raw = json.dumps(c.get("arguments") or {})
                    obs, step = self._run_tool(c.get("name", ""), raw, scratch)
                    _remember_chunks(obs)
                    steps.append(step)
                    _emit_step(step, obs)
                    if step["type"] == "tool_call":
                        n_tool_calls += 1
                    messages.append({"role": "tool",
                                     "tool_call_id": c.get("id") or f"textcall_{k}",
                                     "content": obs})
                continue

            # ---- final answer
            answer = (resp.content or "").strip()
            steps.append({"type": "final", "content": answer[:400]})
            _emit("step", steps[-1])
            break
        else:
            # reasoning budget exhausted -> force a grounded final answer
            messages.append({
                "role": "user",
                "content": (
                    "You have used all your reasoning steps. Provide your final "
                    "answer NOW, grounded strictly in the evidence you gathered. "
                    "If it does not support an answer, say the documents do not "
                    "cover it."
                ),
            })
            try:
                turns += 1          # the forced final is its own model call
                resp = self.llm.chat(messages)
                answer = (resp.content or "").strip()
                steps.append({"type": "final", "content": answer[:400],
                              "note": "forced after max steps"})
            except LLMError as e:
                error = str(e)

        citations, warnings = check_citations(
            answer, self.registry.doc_names(), self.registry.chunk_exists)
        return AgentResult(
            answer=answer,
            steps=steps,
            citations=citations,
            citation_warnings=warnings,
            seen_chunk_ids=seen_chunk_ids,
            tool_calls=n_tool_calls,
            scratch_notes=len(scratch.notes),
            parallel_turns=parallel_turns,
            turns=turns,
            elapsed_s=time.time() - t0,
            error=error,
        )

    # ----------------------------------------------------------------- private

    def _native_assistant_msg(self, resp: LLMResponse) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": resp.content or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in resp.tool_calls
            ],
        }

    @staticmethod
    def _text_assistant_msg(resp: LLMResponse, calls: List[Dict[str, Any]]
                            ) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": resp.content or None,
            "tool_calls": [
                {
                    "id": c.get("id") or f"textcall_{i}",
                    "type": "function",
                    "function": {"name": c["name"],
                                 "arguments": json.dumps(c.get("arguments") or {})},
                }
                for i, c in enumerate(calls)
            ],
        }

    def _run_tool(self, name: str, raw_args: str, scratch=None):
        """Repair -> validate -> execute. Returns (observation, step)."""
        spec = self.registry.get_spec(name)
        if spec is None:
            problems = [f"unknown tool '{name}' "
                        f"(available: {', '.join(self.registry.names)})"]
            obs = make_tool_error_feedback(name, problems, raw_args)
            return obs, {"type": "tool_error", "tool": name,
                         "problems": problems, "observation": obs[:300]}

        obj, notes = repair_json(raw_args)
        if obj is None:
            problems = [
                "arguments are not valid JSON even after heuristic regex repair "
                f"(raw: {str(raw_args)[:200]!r})"
            ]
            if notes:
                problems.append("fixes attempted: " + ", ".join(notes))
            obs = make_tool_error_feedback(name, problems, raw_args)
            return obs, {"type": "tool_error", "tool": name,
                         "problems": problems, "observation": obs[:300]}

        cleaned, errors = validate_arguments(obj, spec.get("parameters") or {})
        if errors:
            obs = make_tool_error_feedback(name, errors, raw_args)
            return obs, {"type": "tool_error", "tool": name,
                         "problems": errors, "observation": obs[:300]}

        obs = self.registry.execute(name, cleaned, scratch=scratch)
        step = {
            "type": "tool_call",
            "tool": name,
            "arguments": cleaned,
            "observation": obs[:300],
        }
        if notes:
            step["schema_repairs"] = notes
        return obs, step
