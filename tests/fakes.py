"""Test doubles: a scriptable fake LLM and a deterministic fake embedder.

No network access is required by any test.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from app.llm import LLMResponse, ToolCall
from app.schema_repair import repair_json


class FakeLLM:
    """Pops scripted LLMResponse objects; records every request it receives.

    chat_json mimics the real client: one retry with the parse error fed
    back before giving up."""

    def __init__(self, responses: List[Any]) -> None:
        self.responses = list(responses)
        self.received: List[Dict[str, Any]] = []      # chat() calls
        self.json_received: List[Dict[str, Any]] = []  # chat_json() calls

    def chat(self, messages, tools=None, temperature=None, max_tokens=None,
             response_format=None, reasoning_effort=None) -> LLMResponse:
        self.received.append({"messages": [dict(m) for m in messages], "tools": tools,
                              "reasoning_effort": reasoning_effort,
                              "max_tokens": max_tokens})
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def chat_json(self, system: str, user: str, temperature=None,
                  max_tokens=None, reasoning_effort=None
                  ) -> Tuple[Optional[dict], Optional[str]]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        self.json_received.append({"messages": [dict(m) for m in messages],
                                   "reasoning_effort": reasoning_effort,
                                   "max_tokens": max_tokens})
        for _attempt in range(2):
            if not self.responses:
                return None, "fake: no scripted responses left"
            r = self.responses.pop(0)
            self.received.append({"messages": [dict(m) for m in messages]})
            if isinstance(r, Exception):
                raise r
            obj, _ = repair_json(r.content)
            if isinstance(obj, dict):
                return obj, None
            messages = messages + [
                {"role": "assistant", "content": r.content},
                {"role": "user", "content": "invalid JSON; reply with strict JSON only"},
            ]
        return None, "reply was not valid JSON (fake)"


class StubLLM:
    """Function-driven fake LLM that never runs out of responses.

    ``chat_fn(messages, tools)`` must return an LLMResponse;
    ``chat_json_fn(system, user)`` must return ``(obj_or_None, err_or_None)``.
    Used by the integration tests (ingestion + ask + verification all share
    one client)."""

    def __init__(self, chat_fn=None, chat_json_fn=None) -> None:
        self.chat_fn = chat_fn or (lambda messages, tools=None: LLMResponse(
            content="(stub answer)"))
        self.chat_json_fn = chat_json_fn or (lambda system, user: (None, "stub"))
        self.received: List[Dict[str, Any]] = []

    def chat(self, messages, tools=None, temperature=None, max_tokens=None,
             response_format=None, reasoning_effort=None) -> LLMResponse:
        self.received.append({"kind": "chat",
                              "messages": [dict(m) for m in messages],
                              "tools": tools,
                              "reasoning_effort": reasoning_effort})
        return self.chat_fn(messages, tools)

    def chat_json(self, system: str, user: str, temperature=None,
                  max_tokens=None, reasoning_effort=None
                  ) -> Tuple[Optional[dict], Optional[str]]:
        self.received.append({"kind": "chat_json", "system": system[:80],
                              "user": user[:200],
                              "reasoning_effort": reasoning_effort})
        return self.chat_json_fn(system, user)


class FakeEmbedder:
    """Deterministic bag-of-words embedder: each distinct word gets its own
    dimension (per-instance vocabulary), so unrelated words never collide
    and similar texts -> similar vectors."""
    def __init__(self, dim: int = 4096) -> None:
        self.dim = dim
        self._vocab: Dict[str, int] = {}

    def _idx(self, w: str) -> int:
        i = self._vocab.get(w)
        if i is None:
            i = len(self._vocab) % self.dim
            self._vocab[w] = i
        return i

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z]+", text.lower()):
            v[self._idx(w)] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)


class StubIntentRouter:
    """Deterministic stand-in for ``IntentRouter`` in wiring/API tests.

    ``plans`` maps a lowercase substring of the question to an IntentPlan;
    anything unmatched gets ``default`` (or a plain policy lookup).
    """

    def __init__(self, plans=None, default=None) -> None:
        from app.agent.intent import IntentPlan
        self._IntentPlan = IntentPlan
        self.plans = plans or {}
        self.default = default
        self.calls: List[str] = []

    def classify(self, question: str):
        q = (question or "").strip()
        self.calls.append(q)
        for needle, plan in self.plans.items():
            if needle in q.lower():
                return plan
        if self.default is not None:
            return self.default
        return self._IntentPlan(
            intent="POLICY_DIRECT_LOOKUP", route="REACT_LOOP",
            de_noised_question=q, search_queries=[q], decided_by="stub",
        )


class StubRemote:
    """Decision-model stand-in for the contradiction cascade.

    ``scores`` maps (text_a, text_b) to a verdict probability; anything not in
    the map gets ``default``. Records every call as (a, b, signals).
    """

    model = "stub-remote"

    def __init__(self, scores=None, default: float = 0.9) -> None:
        self.scores = dict(scores or {})
        self.default = default
        self.calls = []

    def score(self, text_a, text_b, signals=None):
        from app.contradictions.gate import GateDecision
        self.calls.append((text_a, text_b, list(signals or [])))
        value = self.scores.get((text_a, text_b), self.default)
        return GateDecision(True, float(value), ["stub decision"],
                            decided_by="decision_model")
