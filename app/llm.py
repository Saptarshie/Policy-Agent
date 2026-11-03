"""Minimal OpenAI-compatible chat client (requests-based, no SDK lock-in).

Robustness features:
* retries with backoff on 429/5xx,
* automatic downgrade if the endpoint rejects ``tools`` / ``response_format``
  (falls back to text-based tool calling handled by the agent),
* automatic ``max_tokens`` halving if the endpoint caps output below the
  configured 64k,
* ``chat_json()`` helper that parses strict-JSON replies through the
  heuristic repair layer and feeds the parse error back for one retry.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from .schema_repair import repair_json

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM endpoint cannot be used."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string; may be malformed -> repaired later


@dataclass
class LLMResponse:
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    raw: Optional[dict] = None


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "",
        max_output_tokens: int = 65536,
        temperature: float = 0.1,
        timeout: Tuple[int, int] = (20, 600),
        max_retries: int = 3,
        reasoning_effort: str = "",
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        # Optional provider hint ("off"|"low"|"medium"|"high"|... ). Empty =
        # don't send the field at all (portable across OpenAI-compatible
        # endpoints; reasoning models default to their own budget).
        self.reasoning_effort = (reasoning_effort or "").strip().lower()
        # learned endpoint capabilities (None = unknown)
        self._supports_tools: Optional[bool] = None
        self._supports_response_format: Optional[bool] = None
        self._supports_reasoning_effort: Optional[bool] = None

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
        reasoning_effort: Optional[str] = None,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [dict(m) for m in messages],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_output_tokens if max_tokens is None else max_tokens,
        }
        effort = (reasoning_effort if reasoning_effort is not None
                  else self.reasoning_effort)
        effort = (effort or "").strip().lower()
        if effort and self._supports_reasoning_effort is not False:
            payload["reasoning_effort"] = effort
        if tools and self._supports_tools is not False:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format and self._supports_response_format is not False:
            payload["response_format"] = response_format
        data = self._post_with_fallbacks(payload, tools=tools, response_format=response_format)
        return self._parse(data)

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Tuple[Optional[dict], Optional[str]]:
        """Ask for a strict-JSON reply; repair heuristically; retry once with
        the parse error fed back. Returns ``(obj, None)`` or ``(None, error)``."""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_error = "unknown error"
        for _attempt in range(2):
            resp = self.chat(messages, temperature=temperature, max_tokens=max_tokens,
                             response_format={"type": "json_object"},
                             reasoning_effort=reasoning_effort)
            obj, _notes = repair_json(resp.content)
            if isinstance(obj, dict):
                return obj, None
            last_error = (
                f"reply was not valid JSON even after heuristic repair "
                f"(first 200 chars: {resp.content[:200]!r})"
            )
            messages = messages + [
                {"role": "assistant", "content": resp.content or "(empty reply)"},
                {
                    "role": "user",
                    "content": (
                        f"Your previous reply was invalid: {last_error}. "
                        "Reply again with ONLY a valid JSON object - no prose, "
                        "no markdown, no code fences."
                    ),
                },
            ]
        return None, last_error

    # ------------------------------------------------------------------ http

    def _post_with_fallbacks(self, payload: Dict[str, Any], tools, response_format) -> dict:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        attempt = 0
        while True:
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt < self.max_retries:
                    attempt += 1
                    time.sleep(min(15.0, 1.5 ** attempt))
                    continue
                raise LLMError(f"cannot reach LLM endpoint {url}: {e}") from e

            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError as e:
                    raise LLMError(f"endpoint returned invalid JSON: {e}") from e

            err_text = f"{r.status_code} {r.text[:400]}"

            if r.status_code == 400:
                # Drop unsupported features one at a time, then try to
                # satisfy max_tokens limits by halving.
                low = (r.text or "").lower()
                if "reasoning_effort" in low and self._supports_reasoning_effort is not False:
                    self._supports_reasoning_effort = False
                    payload.pop("reasoning_effort", None)
                    log.warning("endpoint rejected reasoning_effort; retrying without it")
                    continue
                if response_format is not None and self._supports_response_format is not False:
                    self._supports_response_format = False
                    payload.pop("response_format", None)
                    log.warning("endpoint rejected response_format; retrying without it")
                    continue
                if tools and self._supports_tools is not False:
                    self._supports_tools = False
                    payload.pop("tools", None)
                    payload.pop("tool_choice", None)
                    log.warning("endpoint rejected tools; falling back to text-based tool calling")
                    continue
                if any(t in low for t in ("max_tokens", "max output", "max completion",
                                          "maximum", "too large", "context length", "context_window")):
                    cur = int(payload.get("max_tokens", self.max_output_tokens))
                    if cur > 512:
                        payload["max_tokens"] = max(512, cur // 2)
                        log.warning("endpoint rejected max_tokens=%s; retrying with %s",
                                    cur, payload["max_tokens"])
                        continue
                raise LLMError(f"LLM endpoint error: {err_text}")

            if r.status_code in (401, 403, 404):
                raise LLMError(f"LLM endpoint error: {err_text}")

            if r.status_code in (429, 500, 502, 503, 504):
                if attempt < self.max_retries:
                    attempt += 1
                    time.sleep(min(30.0, 1.5 ** attempt))
                    continue
                raise LLMError(f"LLM endpoint error after retries: {err_text}")

            raise LLMError(f"LLM endpoint error: {err_text}")

    # ----------------------------------------------------------------- parse

    @staticmethod
    def _parse(data: dict) -> LLMResponse:
        try:
            choice = data["choices"][0]
            msg = choice.get("message") or {}
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape ({e}): {str(data)[:300]}") from e

        content = msg.get("content") or ""
        calls: List[ToolCall] = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            calls.append(ToolCall(
                id=tc.get("id") or f"call_{i}",
                name=fn.get("name") or "",
                arguments=fn.get("arguments") or "",
            ))
        if not calls and msg.get("function_call"):  # legacy completions API shape
            fc = msg["function_call"]
            calls.append(ToolCall(id="call_0", name=fc.get("name") or "",
                                  arguments=fc.get("arguments") or ""))
        return LLMResponse(
            content=content,
            tool_calls=calls,
            finish_reason=choice.get("finish_reason") or "",
            raw=data,
        )
