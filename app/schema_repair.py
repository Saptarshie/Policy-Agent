"""Heuristic, regex-based repair for malformed LLM tool calls and JSON.

This is the first line of defence against broken/hallucinated structured
output, in three layers:

1. **Repair** obviously-broken JSON emitted by a model: markdown code fences,
   surrounding prose, JS comments, Python literals (``True/False/None``),
   trailing commas, unquoted keys, single-quoted strings, unbalanced
   brackets, truncated tails.
2. **Validate** the repaired arguments against the tool's JSON schema
   (required params, basic types, unknown params).
3. **Feed back**: if repair or validation fails, build a precise error
   message that is returned to the agent as the tool observation so it can
   correct itself on the next turn - never silently "hallucinate" a result.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------- regexes

_LINE_COMMENT_RE = re.compile(r"(?m)^\s*//.*$")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
_TRAILING_COMMA_RE = re.compile(r",\s*(?=[}\]])")
_UNQUOTED_KEY_RE = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_\-.]*)(\s*:)')
_SINGLE_QUOTE_STR_RE = re.compile(r"'([^'\\\n]*)'")
_PY_TRUE_RE = re.compile(r"\bTrue\b")
_PY_FALSE_RE = re.compile(r"\bFalse\b")
_PY_NONE_RE = re.compile(r"\bNone\b")
_NAN_RE = re.compile(r"\b(NaN|Infinity|-Infinity)\b")

_TOOLCALL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_FENCED_JSON_RE = re.compile(r"```(?:json|javascript|js)?\s*(\{.*?\})\s*```", re.S)


# --------------------------------------------------------------------- helpers

def _first_json_span(s: str) -> Optional[str]:
    """Return the first complete top-level {...} or [...] inside *s*."""
    start = -1
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start < 0:
        return None
    stack: List[str] = []
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            if not stack:
                return s[start:i + 1]
    return None


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[A-Za-z0-9_+\-]*\s*", "", s)
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def _drop_truncated_tail(s: str) -> str:
    """Drop an unterminated string literal at the end (truncated output)."""
    in_str = False
    esc = False
    last_open = -1
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
                last_open = i
    if not in_str:
        return s
    s = s[:last_open].rstrip()
    s = re.sub(r",\s*$", "", s)
    s = re.sub(r",?\s*\"[^\"]*\"\s*:\s*$", "", s)
    return s


def _balance_brackets(s: str) -> str:
    stack: List[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    if not stack:
        return s
    return s + "".join("}" if c == "{" else "]" for c in reversed(stack))


# --------------------------------------------------------------------- repair

def repair_json(text: Optional[str]) -> Tuple[Optional[Any], List[str]]:
    """Best-effort repair + parse of a model-emitted JSON blob.

    Returns ``(obj, notes)``. ``obj`` is ``None`` on failure; ``notes``
    lists the heuristic fixes that were applied (for logging/feedback).
    """
    notes: List[str] = []
    if text is None:
        return None, ["empty input"]
    s = text.strip()
    if not s:
        return None, ["empty input"]

    if "```" in s:
        s = _strip_code_fences(s)
        notes.append("removed markdown code fences")

    span = _first_json_span(s)
    if span is not None and span != s:
        s = span
        notes.append("extracted JSON from surrounding prose")

    try:
        return json.loads(s), notes
    except json.JSONDecodeError:
        pass

    def _remove_comments(x: str) -> str:
        x = _BLOCK_COMMENT_RE.sub("", x)
        return _LINE_COMMENT_RE.sub("", x)

    def _fix_py_literals(x: str) -> str:
        x = _PY_TRUE_RE.sub("true", x)
        x = _PY_FALSE_RE.sub("false", x)
        x = _PY_NONE_RE.sub("null", x)
        return _NAN_RE.sub("null", x)

    def _fix_single_quotes(x: str) -> str:
        # Only safe when the whole blob is JSON-ish (starts with { or [).
        if not x.lstrip()[:1] in ("{", "["):
            return x

        def repl(m: "re.Match[str]") -> str:
            inner = m.group(1)
            if '"' in inner:
                return m.group(0)
            return '"' + inner + '"'

        return _SINGLE_QUOTE_STR_RE.sub(repl, x)

    fixers = [
        ("removed JS comments", _remove_comments),
        ("converted Python literals (True/False/None)", _fix_py_literals),
        ("removed trailing commas", lambda x: _TRAILING_COMMA_RE.sub("", x)),
        ("quoted unquoted keys", lambda x: _UNQUOTED_KEY_RE.sub(r'\1"\2"\3', x)),
        ("converted single quotes to double quotes", _fix_single_quotes),
        ("dropped truncated tail", _drop_truncated_tail),
    ]
    for name, fn in fixers:
        s2 = fn(s)
        if s2 == s:
            continue
        s = s2
        notes.append(name)
        try:
            return json.loads(s), notes
        except json.JSONDecodeError:
            continue

    s2 = _balance_brackets(s)
    if s2 != s:
        s = s2
        notes.append("balanced unbalanced brackets")
        try:
            return json.loads(s), notes
        except json.JSONDecodeError:
            pass
    return None, notes


# --------------------------------------------------------------------- validate

def _coerce(value: Any, spec: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str]]:
    """Loosely coerce *value* to the type demanded by *spec*."""
    if not isinstance(spec, dict):
        return value, None
    t = (spec.get("type") or "").lower()
    enum = spec.get("enum")
    if enum and isinstance(enum, list):
        if value not in enum and str(value) not in [str(e) for e in enum]:
            return None, f"value {value!r} not in allowed set {enum}"
        return value, None
    if t == "integer":
        if isinstance(value, bool):
            return None, "expected integer, got boolean"
        if isinstance(value, int):
            return value, None
        if isinstance(value, str):
            try:
                return int(value.strip()), None
            except ValueError:
                return None, "expected integer"
        if isinstance(value, float) and value.is_integer():
            return int(value), None
        return None, f"expected integer, got {type(value).__name__}"
    if t == "number":
        try:
            return float(value), None
        except (TypeError, ValueError):
            return None, "expected number"
    if t == "string":
        if isinstance(value, str):
            return value, None
        if isinstance(value, (int, float, bool)):
            return str(value).lower() if isinstance(value, bool) else str(value), None
        return None, f"expected string, got {type(value).__name__}"
    if t == "boolean":
        if isinstance(value, bool):
            return value, None
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true", None
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value), None
        return None, "expected boolean"
    if t == "array":
        if isinstance(value, list):
            return value, None
        return None, "expected array"
    if t == "object":
        if isinstance(value, dict):
            return value, None
        return None, "expected object"
    return value, None


def validate_arguments(args: Any, schema: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Validate *args* against an OpenAI-style function parameters schema.

    Returns ``(cleaned_args, errors)``. Unknown parameters are rejected so
    the model gets explicit feedback instead of being silently ignored.
    """
    errors: List[str] = []
    if args is None:
        return {}, ["arguments are empty"]
    if not isinstance(args, dict):
        return {}, ["arguments must be a JSON object"]

    props = schema.get("properties") or {}
    required = schema.get("required") or []
    cleaned: Dict[str, Any] = {}

    for key in required:
        if key not in args:
            errors.append(f"missing required parameter '{key}'")

    for key, value in args.items():
        if key not in props:
            errors.append(f"unknown parameter '{key}' (allowed: {sorted(props)})")
            continue
        coerced, err = _coerce(value, props.get(key) or {})
        if err:
            errors.append(f"parameter '{key}': {err}")
        else:
            cleaned[key] = coerced
    return cleaned, errors


def make_tool_error_feedback(tool_name: str, problems: List[str], raw_args: Any = None) -> str:
    """Build the observation fed back to the agent when a tool call is broken."""
    raw = repr(raw_args)[:400] if raw_args is not None else ""
    lines = [
        f"TOOL_CALL_ERROR: the call to '{tool_name}' could NOT be executed.",
        "Problems:",
    ]
    lines += [f"  - {p}" for p in problems]
    if raw:
        lines.append(f"Raw arguments received: {raw}")
    lines.append(
        "Re-emit the tool call with valid JSON arguments that match the tool "
        "schema exactly. Do NOT invent or assume a result for this call."
    )
    return "\n".join(lines)


# ------------------------------------------------------- text tool-call parsing

def _normalize_tool_obj(obj: Any) -> Optional[Dict[str, Any]]:
    """Normalize a parsed object like {"tool": ..., "arguments": {...}}."""
    if not isinstance(obj, dict):
        return None
    name = None
    for key in ("name", "tool", "function", "action"):
        v = obj.get(key)
        if isinstance(v, str) and v:
            name = v
            break
        if isinstance(v, dict) and isinstance(v.get("name"), str):
            name = v["name"]
            break
    if not name:
        return None
    args: Any = None
    for key in ("arguments", "parameters", "args", "input", "arguments_"):
        if key in obj:
            args = obj[key]
            break
    if isinstance(args, str):
        args, _ = repair_json(args)
    if args is None:
        args = {}
    if not isinstance(args, dict):
        args = {"_raw": args}
    return {"name": name, "arguments": args}


def extract_tool_calls_from_text(content: str, valid_names: List[str]) -> List[Dict[str, Any]]:
    """Extract tool calls from plain text (fallback for endpoints without
    native function calling). Recognises <tool_call>{...}</tool_call> tags,
    fenced JSON blocks that look like tool calls, and ACTION:-style lines."""
    if not content or not valid_names:
        return []
    calls: List[Dict[str, Any]] = []

    for m in _TOOLCALL_TAG_RE.finditer(content):
        obj, _ = repair_json(m.group(1))
        norm = _normalize_tool_obj(obj)
        if norm and norm["name"] in valid_names:
            calls.append(norm)

    if not calls:
        for m in _FENCED_JSON_RE.finditer(content):
            obj, _ = repair_json(m.group(1))
            norm = _normalize_tool_obj(obj)
            if norm and norm["name"] in valid_names:
                calls.append(norm)

    if not calls:
        m = re.search(r"ACTION\s*[:=]\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\{.*\})?", content, re.S)
        if m and m.group(1) in valid_names:
            args, _ = repair_json(m.group(2)) if m.group(2) else ({}, [])
            if isinstance(args, dict):
                calls.append({"name": m.group(1), "arguments": args})

    return calls
