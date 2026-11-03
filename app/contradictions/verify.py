"""Stage 3 of the cascade: LLM contradiction verifier.

Runs ONLY for pairs that passed the JEV-style heuristic gate. Given both
passages verbatim (with doc name + section) plus the gate's signals, it
returns a strict-JSON verdict. Output is parsed through the same regex
repair layer used everywhere else; if parsing fails after the built-in
retry, the verdict is marked ``verified=False`` so the tracker can record
it as "unverified - needs admin review" instead of silently dropping it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from ..llm import OpenAICompatClient
from ..prompts import CONTRADICTION_VERIFIER_SYSTEM, CONTRADICTION_VERIFIER_USER

log = logging.getLogger(__name__)


@dataclass
class Verdict:
    verified: bool                    # False = LLM could not be run / parsed
    is_contradiction: bool = False
    confidence: float = 0.0
    conflict_type: str = ""
    explanation: str = ""
    points: List[dict] = field(default_factory=list)
    error: Optional[str] = None


def _to_float(v, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return False


class ContradictionVerifier:
    def __init__(self, llm: Optional[OpenAICompatClient]) -> None:
        self.llm = llm

    def verify(self, doc_a: str, sec_a: str, text_a: str,
               doc_b: str, sec_b: str, text_b: str,
               gate_signals: Optional[List[str]] = None) -> Verdict:
        if self.llm is None:
            return Verdict(verified=False,
                           error="no LLM configured (heuristic-only mode)")
        signals = gate_signals or []
        signals_txt = "\n".join(f"  - {s}" for s in signals) or "  (none)"
        user = CONTRADICTION_VERIFIER_USER.format(
            doc_a=doc_a, sec_a=sec_a or "-", text_a=(text_a or "").strip()[:4000],
            doc_b=doc_b, sec_b=sec_b or "-", text_b=(text_b or "").strip()[:4000],
            signals=signals_txt,
        )
        try:
            obj, err = self.llm.chat_json(CONTRADICTION_VERIFIER_SYSTEM, user)
        except Exception as e:  # endpoint down -> unverified, never crash the scan
            log.warning("contradiction verifier LLM error: %s", e)
            return Verdict(verified=False, error=str(e))
        if obj is None:
            return Verdict(verified=False, error=err or "invalid JSON verdict")

        points = obj.get("conflicting_points")
        if not isinstance(points, list):
            points = []
        points = [p for p in points if isinstance(p, dict)][:10]
        return Verdict(
            verified=True,
            is_contradiction=_to_bool(obj.get("is_contradiction")),
            confidence=_to_float(obj.get("confidence"), 0.5),
            conflict_type=str(obj.get("conflict_type") or "")[:40],
            explanation=str(obj.get("explanation") or "").strip(),
            points=points,
        )
