"""What shape is this answer? — asked of the decision model, not of a keyword list.

The verifier needs to know whether an answer is a plain answer, a partial answer
(some of the question answered, some declined), a bare refusal, or chit-chat.
The old implementation tested a hardcoded list of refusal phrases
(``"do not cover"``, ``"couldn't find"`` …). That is a lexicographic classifier
for a natural-language property: it fails on rewording ("nothing on that in what
I've got"), it silently misfires on partial answers, and every fix adds another
phrase to a list that can never be complete.

This module asks the decision model (``DECISION_MODEL_NAME``, e.g.
``typesafe/jev``) four yes/no questions instead. The model returns a probability
per question, the shape is derived from those probabilities, and anything
ambiguous degrades to ``unknown`` — which means "judge it", never "skip it".

Two backends, same interface:

* :class:`SystemOneShapeClassifier` — the decision model. Cheap, ~2s, and it
  accepts a long state (the model's documented single-request context is 32K),
  so question + full answer + a context sample fit comfortably.
* :class:`LLMShapeClassifier` — a strict-JSON classification through the main
  chat model, for deployments that do not configure a decision model.

Failure is always ``kind="unknown"`` with ``error`` set: the verifier then runs
the faithfulness judge, because a classification outage must degrade to *more*
checking, not less.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

log = logging.getLogger(__name__)

KINDS = ("answer", "partial", "refusal", "chitchat", "unknown")

_QUESTIONS = {
    "declines_part": (
        "Does the ANSWER decline to answer some part of the question, saying the "
        "documents do not cover it, or that the information is not available?"
    ),
    "states_rules": (
        "Does the ANSWER state specific policy rules, numbers, limits, dates, "
        "tiers or obligations?"
    ),
    "is_only_refusal": (
        "Is the ANSWER only a refusal or a statement that the documents do not "
        "cover the question, with no policy rule, number, limit, date or "
        "obligation stated anywhere in it?"
    ),
    "is_chitchat": (
        "Is the ANSWER only chit-chat — a greeting, thanks, small talk or "
        "acknowledgement — with no policy content at all?"
    ),
}

_LLM_SYSTEM = """You classify the SHAPE of an answer produced by a policy assistant. \
You do not judge whether it is true; you only say what kind of answer it is.

Return STRICT JSON:
{"shape": "answer" | "partial" | "refusal" | "chitchat",
 "declined_parts": ["the part of the question the answer explicitly declines, quoted"],
 "reason": "one short sentence"}

Definitions:
- answer:   states policy rules/numbers/limits/dates/obligations.
- partial:  states some policy specifics AND explicitly declines some other part
            of the question because the documents do not cover it.
- refusal:  declines the question as a whole and states no policy specifics.
- chitchat: greeting, thanks or small talk, with no policy content."""


@dataclass
class AnswerShape:
    kind: str = "unknown"                 # one of KINDS
    declined_parts: List[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    decided_by: str = ""                  # "decision_model" | "llm" | "fallback"
    reason: str = ""
    error: str = ""

    @property
    def has_claims(self) -> bool:
        """Does this shape contain policy assertions worth verifying?"""
        return self.kind in ("answer", "partial", "unknown")

    @property
    def is_partial(self) -> bool:
        return self.kind == "partial"

    def as_dict(self) -> dict:
        return {"kind": self.kind, "declined_parts": self.declined_parts,
                "scores": {k: round(v, 3) for k, v in self.scores.items()},
                "decided_by": self.decided_by, "reason": self.reason,
                "error": self.error}


def shape_from_scores(scores: dict) -> AnswerShape:
    """Derive the shape from decision-model probabilities.

    Thresholds are 0.5 on each question; the ordering matters more than the
    exact value, and every ambiguous combination falls through to ``unknown``
    so that the answer still gets judged.
    """
    def p(name: str) -> float:
        try:
            return float(scores.get(name, 0.0))
        except (TypeError, ValueError):
            return 0.0

    declines, rules = p("declines_part"), p("states_rules")
    only_refusal, chitchat = p("is_only_refusal"), p("is_chitchat")

    if chitchat >= 0.5 and rules < 0.5:
        kind = "chitchat"
    elif declines >= 0.5 and rules >= 0.5:
        kind = "partial"
    elif only_refusal >= 0.5 and rules < 0.5:
        kind = "refusal"
    elif rules >= 0.5 and declines < 0.5:
        kind = "answer"
    elif only_refusal < 0.5 and rules < 0.5 and declines < 0.5 and chitchat < 0.5:
        kind = "unknown"
    else:
        # contradictory confidence (e.g. refuses and states rules without
        # admitting the split) -> treat as partial and let the judge decide
        kind = "partial" if declines >= 0.5 else "answer"
    return AnswerShape(kind=kind, scores=dict(scores), decided_by="decision_model")


def _state(question: str, answer: str, context_sample: Optional[str]) -> str:
    parts = [f"QUESTION:\n{question or ''}", f"ANSWER:\n{answer or ''}"]
    if context_sample:
        parts.append("PASSAGES AVAILABLE TO THE ANSWER:\n" + context_sample)
    return "\n\n".join(parts)


class SystemOneShapeClassifier:
    """Shape classification through the decision model (typesafe/jev)."""

    def __init__(self, client) -> None:      # SystemOneDecisionModel-like .ask()
        self.client = client

    def classify(self, question: str, answer: str,
                 context_sample: Optional[str] = None) -> AnswerShape:
        if not (answer or "").strip():
            return AnswerShape(kind="chitchat", decided_by="decision_model")
        scores = self.client.ask(_state(question, answer, context_sample), _QUESTIONS)
        if not scores:
            return AnswerShape(kind="unknown", decided_by="fallback",
                               error="decision model unavailable")
        return shape_from_scores(scores)


class LLMShapeClassifier:
    """Fallback: the same classification through the chat model's JSON mode."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def classify(self, question: str, answer: str,
                 context_sample: Optional[str] = None) -> AnswerShape:
        if not (answer or "").strip():
            return AnswerShape(kind="chitchat", decided_by="llm")
        if self.llm is None:
            return AnswerShape(kind="unknown", decided_by="fallback",
                               error="no classifier configured")
        try:
            obj, err = self.llm.chat_json(
                _LLM_SYSTEM,
                _state(question, answer, context_sample)[:8000],
                temperature=0.0,
            )
        except Exception as e:               # endpoint down -> judge the answer
            log.warning("answer-shape classifier failed (%s); judging anyway", e)
            return AnswerShape(kind="unknown", decided_by="fallback", error=str(e))
        if not isinstance(obj, dict):
            return AnswerShape(kind="unknown", decided_by="fallback",
                               error=err or "unparseable shape verdict")
        kind = str(obj.get("shape") or "").strip().lower()
        if kind not in KINDS:
            kind = "unknown"
        declined = obj.get("declined_parts")
        return AnswerShape(
            kind=kind,
            declined_parts=[str(x)[:300] for x in declined][:4]
            if isinstance(declined, list) else [],
            decided_by="llm",
            reason=str(obj.get("reason") or "")[:200],
        )


def build_shape_classifier(*, decision_client=None, llm=None):
    """Decision model first, chat model second, never a keyword list."""
    if decision_client is not None:
        return SystemOneShapeClassifier(decision_client)
    if llm is not None:
        return LLMShapeClassifier(llm)
    return None
