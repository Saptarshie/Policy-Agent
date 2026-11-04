"""A per-answer scratchpad the agent can write to and read back.

Why: on a long, multi-document question the failure mode is not a missing passage,
it is holding too much in one context window and re-deriving what it already
found — or inventing the piece it lost. A small named-note store lets the model
park findings ("v1 cap = 6 days", "amendment effective 1 July 2025") and re-read
them before it answers, instead of trusting recall.

Scope: one scratchpad per `ask()` call, in memory. It is *working memory for this
question*, not a knowledge store and not evidence — a note is only ever as true as
the passage the agent copied into it, and the prompt says so. Nothing here is
retrieved, indexed, embedded or persisted.

Bounded on purpose: at most :data:`MAX_NOTES` notes, :data:`MAX_NOTE_CHARS` each,
:data:`MAX_TOTAL_CHARS` overall, so a runaway loop cannot grow the observation
budget by writing to itself.
"""
from __future__ import annotations

from typing import Dict, List

MAX_NOTES = 12
MAX_NOTE_CHARS = 4000
MAX_TOTAL_CHARS = 12000

_APPEND_MARK = "\n"


class Scratchpad:
    def __init__(self) -> None:
        self._notes: Dict[str, str] = {}
        self.writes = 0

    # ------------------------------------------------------------------ api
    def write(self, key: str, text: str) -> str:
        key = self._key(key)
        if key not in self._notes and len(self._notes) >= MAX_NOTES:
            raise ValueError(f"scratchpad is full ({MAX_NOTES} notes); delete one first "
                             f"(current: {sorted(self._notes)})")
        body = self._body(text, key)
        self._notes[key] = body
        self.writes += 1
        return f"note '{key}' written ({len(body)} chars). Notes now: {self.summary()}"

    def append(self, key: str, text: str) -> str:
        key = self._key(key)
        body = self._body(text)
        merged = (self._notes.get(key, "") + _APPEND_MARK + body).strip()
        return self.write(key, merged)

    def edit(self, key: str, find: str, replace: str) -> str:
        key = self._key(key)
        if key not in self._notes:
            raise KeyError(f"no note '{key}' (notes: {self.summary()})")
        if not find:
            raise ValueError("'find' must not be empty; use write to replace a note")
        body = self._notes[key]
        if find not in body:
            raise ValueError(f"'{find[:80]}' does not appear in note '{key}'")
        updated = body.replace(find, replace or "", 1)
        self._notes[key] = self._body(updated, key)
        self.writes += 1
        return f"note '{key}' edited ({len(self._notes[key])} chars)."

    def read(self, key: str = "") -> str:
        if not key:
            if not self._notes:
                return "scratchpad is empty."
            return "\n\n".join(f"=== {k} ===\n{v}" for k, v in self._notes.items())
        key = self._key(key)
        if key not in self._notes:
            raise KeyError(f"no note '{key}' (notes: {self.summary()})")
        return f"=== {key} ===\n{self._notes[key]}"

    def delete(self, key: str) -> str:
        key = self._key(key)
        if key not in self._notes:
            raise KeyError(f"no note '{key}' (notes: {self.summary()})")
        del self._notes[key]
        self.writes += 1
        return f"note '{key}' deleted. Notes now: {self.summary()}"

    def list_keys(self) -> str:
        if not self._notes:
            return "scratchpad is empty."
        rows = [f"- {k} ({len(v)} chars)" for k, v in self._notes.items()]
        return f"{len(self._notes)} note(s):\n" + "\n".join(rows)

    def summary(self) -> str:
        return ", ".join(f"{k}({len(v)})" for k, v in self._notes.items()) or "none"

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _key(key: str) -> str:
        k = " ".join(str(key or "").split())[:60]
        if not k:
            raise ValueError("note key must not be empty")
        return k

    def _body(self, text: str, key: str = "") -> str:
        body = str(text or "").strip()
        if not body:
            raise ValueError("note text must not be empty")
        if len(body) > MAX_NOTE_CHARS:
            raise ValueError(f"note text is {len(body)} chars; keep one note under "
                             f"{MAX_NOTE_CHARS} and split it into several notes")
        # space the note will occupy once written: everything except this key
        other = sum(len(v) for k, v in self._notes.items() if k != key)
        if other + len(body) > MAX_TOTAL_CHARS:
            raise ValueError(f"scratchpad would exceed {MAX_TOTAL_CHARS} chars total; "
                             "delete or shorten a note first")
        return body

    # for tests / introspection
    @property
    def notes(self) -> Dict[str, str]:
        return dict(self._notes)

    def keys(self) -> List[str]:
        return list(self._notes)
