"""LEGACY regex intent gate - kept only as the ablation baseline.

Do not wire this into a request path. It exists so ``eval/run_intent_eval.py``
can report exactly what the keyword layer scored before the LLM triage layer
replaced it, on the same labelled cases.

Known failure modes (reproduced in tests/test_intent_legacy.py):
  * "What policies apply to parental leave?" -> treated as a listing request;
  * "Which policy covers dental implants?" -> same;
  * every implicit grievance, multi-hop, sensitive or adversarial input falls
    through to POLICY_DIRECT_LOOKUP (it only has three buckets);
  * greeting vocabulary is a closed whitelist ("hii", "namaste", "morning!"
    all miss).
"""

from __future__ import annotations

import re

from .intent import CATEGORY_TO_ROUTE, IntentPlan

_GREETING = re.compile(
    r"^\s*(hi|hey|hello|yo|good\s+(morning|afternoon|evening)|thanks|thank\s*you|"
    r"thx|ty|bye|goodbye|ok(?:ay)?|great|awesome|cool)"
    r"(\s+(there|everyone|team|all|folks|sir|madam|hr))?[\s!.?]*$",
    re.IGNORECASE,
)
_META_DOCS = re.compile(
    r"\b(what|which|list|show)\s+((all|me|us|the|available|uploaded|ingested|current)\s+)*"
    r"(policy|policies|documents?|docs?|files?)\b|"
    r"\b(uploaded|available|indexed|ingested)\s+(policies|documents?|docs?)\b|"
    r"\bwhat\s+do\s+you\s+know\s+about\b|\bwhat\s+can\s+you\s+help\s+with\b",
    re.IGNORECASE,
)


def classify_intent_legacy(question: str) -> IntentPlan:
    """Regex baseline, adapted to the ``IntentPlan`` contract."""
    q = (question or "").strip()
    if not q:
        return IntentPlan(intent="CHITCHAT_OR_OUT_OF_SCOPE", route="FAST_REFUSE",
                          refusal_reason="empty question", decided_by="legacy")

    if _GREETING.match(q):
        intent = "GREETING"
    elif _META_DOCS.search(q):
        intent = "META_DOCUMENTS"
    else:
        intent = "POLICY_DIRECT_LOOKUP"

    route = CATEGORY_TO_ROUTE[intent]
    return IntentPlan(
        intent=intent,
        route=route,
        de_noised_question=q if route == "REACT_LOOP" else "",
        search_queries=[q] if route == "REACT_LOOP" else [],
        decided_by="legacy",
        notes=["regex baseline"],
    )
