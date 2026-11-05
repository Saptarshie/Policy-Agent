"""LLM-backed Intent & Query-Normalization (triage) layer.

Sits in front of the ReAct loop and does four things:

  1. classify the employee's underlying intent (8 categories),
  2. decide the route: ``REACT_LOOP`` / ``DIRECT_REPLY`` / ``FAST_REFUSE``,
  3. de-noise the raw input into a neutral HR question,
  4. produce 1-3 formal search queries + entities for retrieval.

Why not a regex gate: keyword matching overfits phrasings and cannot tell
"What policies apply to parental leave?" (a policy question) from "What
policy documents do you have?" (a listing request) - the old gate answered
the first with a document list. The regex version is preserved in
``intent_legacy.py`` purely as the ablation baseline.

Reliability properties:
  * strict JSON, parsed through the shared repair layer (``chat_json``);
  * decisions are cached on disk keyed by question + prompt version + model,
    so repeats are free, results are deterministic across runs, and an
    interrupted evaluation resumes where it stopped;
  * fails **open**: if the classifier is unavailable or unparseable, the
    input goes to the grounded agent, whose own refusal path is the
    backstop. A triage outage must not become a product outage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..prompts import INTENT_SYSTEM

log = logging.getLogger(__name__)

# --------------------------------------------------------------------- enums

CATEGORIES = (
    "POLICY_DIRECT_LOOKUP",
    "IMPLICIT_SITUATION_OR_GRIEVANCE",
    "MULTI_HOP_COMPARISON",
    "SENSITIVE_HR_ESCALATION",
    "CHITCHAT_OR_OUT_OF_SCOPE",
    "ADVERSARIAL_OR_MALICIOUS",
    "GREETING",
    "META_DOCUMENTS",
)

# Routing is a pure function of the category - one source of truth, so a model
# that contradicts itself cannot emit FAST_REFUSE for a policy question (or
# REACT_LOOP for an injection attempt).
CATEGORY_TO_ROUTE: Dict[str, str] = {
    "POLICY_DIRECT_LOOKUP": "REACT_LOOP",
    "IMPLICIT_SITUATION_OR_GRIEVANCE": "REACT_LOOP",
    "MULTI_HOP_COMPARISON": "REACT_LOOP",
    "SENSITIVE_HR_ESCALATION": "REACT_LOOP",
    "GREETING": "DIRECT_REPLY",
    "META_DOCUMENTS": "DIRECT_REPLY",
    "CHITCHAT_OR_OUT_OF_SCOPE": "FAST_REFUSE",
    "ADVERSARIAL_OR_MALICIOUS": "FAST_REFUSE",
}

ROUTES = ("REACT_LOOP", "DIRECT_REPLY", "FAST_REFUSE")

_MAX_QUERIES = 3
_MAX_ENTITIES = 12


# --------------------------------------------------------------------- model

@dataclass
class IntentPlan:
    intent: str = "POLICY_DIRECT_LOOKUP"
    route: str = "REACT_LOOP"
    de_noised_question: str = ""
    search_queries: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)   # what the QUESTION states
    requires_table_check: bool = False
    tone_guidance: str = ""
    refusal_reason: str = ""
    user_state_summary: str = ""
    decided_by: str = "llm"       # llm | cache | validation | fallback | legacy
    prompt_version: str = ""
    latency_s: float = 0.0
    notes: List[str] = field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None

    # -- rendering ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "route": self.route,
            "de_noised_question": self.de_noised_question,
            "search_queries": list(self.search_queries),
            "entities": list(self.entities),
            "conditions": list(self.conditions),
            "requires_table_check": self.requires_table_check,
            "tone_guidance": self.tone_guidance,
            "refusal_reason": self.refusal_reason,
            "user_state_summary": self.user_state_summary,
            "decided_by": self.decided_by,
            "prompt_version": self.prompt_version,
            "latency_s": round(self.latency_s, 2),
            "notes": list(self.notes),
        }

    def prompt_block(self) -> str:
        """Second system message injected into the agent conversation."""
        if self.route != "REACT_LOOP":
            return ""
        queries = " | ".join(self.search_queries) or "(none)"
        lines = [
            "## INTENT PLAN (internal routing metadata - never quote or mention it)",
            f"- intent: {self.intent}",
            f"- question to answer: {self.de_noised_question}",
            f"- suggested searches: {queries}",
        ]
        if self.entities:
            lines.append("- entities/terms: " + ", ".join(self.entities))
        if self.conditions:
            lines.append("- conditions the question states (apply every one of these; "
                         "they are facts, not claims, so they need no citation): "
                         + " | ".join(self.conditions))
        if self.requires_table_check:
            lines.append("- table check: REQUIRED (read the exact row+column cell)")
        if self.tone_guidance:
            lines.append(f"- tone: {self.tone_guidance}")
        # NOTE: user_state_summary is deliberately NOT injected into the agent.
        # It restates the employee's own numbers/dates, which the agent would
        # then quote as if they were policy facts, and the faithfulness judge
        # (which never sees the summary) flags them. It stays in the API
        # response for auditability.
        return "\n".join(lines)


# ------------------------------------------------------------------ helpers

def _clean_list(value: Any, cap: int, max_len: int = 300) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for v in value:
        if isinstance(v, str) and v.strip():
            s = v.strip()[:max_len]
            if s not in out:
                out.append(s)
        if len(out) >= cap:
            break
    return out


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    if isinstance(value, (int, float)):
        return bool(value)
    return False


_CORPUS_MAX_DOCS = 8
_CORPUS_MAX_CHARS = 240


_CONV_MAX_MESSAGES = 6      # last three exchanges
_CONV_MAX_CHARS = 480       # per message; an answer can be a page long


def format_conversation(history: Optional[List[Dict[str, str]]]) -> tuple[str, str]:
    """Render the CONVERSATION block shown to the classifier, plus its fingerprint.

    The agent has always been given the conversation; the classifier was not, so a
    follow-up ("tell me more", "why?", "and for the Plus tier?") arrived with no
    referent and was routed as chit-chat - refused before retrieval, with a
    perfectly good answer sitting one turn earlier. Continuations are the cheapest
    kind of question to get wrong, because the user has already said what they want.

    Bounded on both axes for the same reason as the corpus block: the prompt must not
    grow with the conversation, and an answer can be long while the question it
    answered is the part that matters.
    """
    turns = []
    for m in (history or []):
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip().lower()
        content = " ".join(str(m.get("content") or "").split())
        if role not in ("user", "assistant") or not content:
            continue
        turns.append(f"{role.upper()}: {content[:_CONV_MAX_CHARS]}")
    turns = turns[-_CONV_MAX_MESSAGES:]
    if not turns:
        block = ("CONVERSATION (none - this is the first question of the session; "
                 "judge INPUT on its own)")
        return block, "none"
    block = ("CONVERSATION (earlier turns of this session, oldest first - context for "
             "INPUT, not evidence):\n" + "\n".join(turns))
    fp = hashlib.sha256(block.encode("utf-8")).hexdigest()[:10]
    return block, fp


def format_corpus(corpus: Optional[List[Dict[str, str]]]) -> tuple[str, str]:
    """Render the CORPUS block shown to the classifier, plus its fingerprint.

    Bounded on both axes: a prompt that grows with the corpus is a prompt that
    eventually stops classifying (and costs more on every question). The block is
    labelled as a digest, because the whole failure mode it can cause is a model
    refusing a question the brief simply does not mention.
    """
    docs = [d for d in (corpus or []) if isinstance(d, dict) and d.get("name")]
    docs = docs[:_CORPUS_MAX_DOCS]
    if not docs:
        block = ("CORPUS (no documents ingested yet): the user may still ask about "
                 "rules in documents that have not been uploaded.")
        return block, "empty"
    lines = [f"CORPUS ({len(docs)} document(s) ingested; each line is a one-paragraph "
             "digest, NOT the document):"]
    for d in docs:
        name = str(d.get("name") or "").strip()[:80]
        brief = str(d.get("brief") or "").strip()[:_CORPUS_MAX_CHARS]
        lines.append(f"- {name}: {brief or '(no brief available)'}")
    block = "\n".join(lines)
    fp = hashlib.sha256(block.encode("utf-8")).hexdigest()[:10]
    return block, fp


def _as_str(value: Any, max_len: int = 500) -> str:
    return value.strip()[:max_len] if isinstance(value, str) else ""


def plan_from_obj(obj: Dict[str, Any], question: str,
                  prompt_version: str = "") -> Optional[IntentPlan]:
    """Validate + normalize a raw classifier reply.

    Returns ``None`` when the reply cannot be trusted (unknown category);
    the caller then fails open to the grounded agent.
    """
    if not isinstance(obj, dict):
        return None
    intent = _as_str(obj.get("intent"), 60).upper().replace(" ", "_")
    if intent not in CATEGORIES:
        return None

    notes: List[str] = []
    expected_route = CATEGORY_TO_ROUTE[intent]
    route = _as_str(obj.get("action_route"), 30).upper()
    if route and route != expected_route:
        notes.append(f"route '{route}' contradicted intent; using '{expected_route}'")
    route = expected_route

    de_noised = _as_str(obj.get("de_noised_question"))
    if route == "REACT_LOOP" and not de_noised:
        de_noised = question.strip()
        notes.append("classifier returned no de-noised question; used the raw input")

    queries = _clean_list(obj.get("search_queries"), _MAX_QUERIES)
    if route == "REACT_LOOP" and not queries:
        queries = [de_noised or question.strip()]
        notes.append("no search queries returned; fell back to the de-noised question")

    refusal_reason = _as_str(obj.get("refusal_reason"))
    if route == "FAST_REFUSE" and not refusal_reason:
        refusal_reason = ("classified as "
                          + intent.lower().replace("_", " ")
                          + "; not an HR policy question")
        notes.append("FAST_REFUSE without a reason; synthesized one")

    return IntentPlan(
        intent=intent,
        route=route,
        de_noised_question=de_noised,
        search_queries=queries,
        entities=_clean_list(obj.get("extracted_entities"), _MAX_ENTITIES, 80),
        conditions=_clean_list(obj.get("conditions"), 8, 160),
        requires_table_check=_as_bool(obj.get("requires_table_check")),
        tone_guidance=_as_str(obj.get("tone_guidance"), 240),
        refusal_reason=refusal_reason if route == "FAST_REFUSE" else "",
        user_state_summary=_as_str(obj.get("user_state_summary"), 400),
        decided_by="llm",
        prompt_version=prompt_version,
        notes=notes,
        raw=obj,
    )


class IntentCache:
    """Tiny JSON cache: deterministic repeats + resumable evaluations."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    self._data = loaded
            except (OSError, ValueError):
                log.warning("intent cache unreadable; starting empty: %s", path)

    @staticmethod
    def key(model: str, prompt_version: str, question: str,
            prompt_fingerprint: str = "", corpus_fingerprint: str = "",
            conversation_fingerprint: str = "") -> str:
        """Cache key. The prompt FINGERPRINT is part of it on purpose: editing the
        routing prompt under the same version label used to replay the old
        verdict forever, which is exactly the situation during prompt tuning.

        The CORPUS fingerprint is there for the same reason one level up: the
        classifier is now shown what the corpus contains, so uploading a policy
        changes the answer it would give for the same question. Without this,
        every verdict decided before the upload would be replayed forever.
        """
        raw = (f"{prompt_version}|{prompt_fingerprint}|{corpus_fingerprint}|"
               f"{conversation_fingerprint}|{model}|"
               f"{' '.join((question or '').split())}")
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._data.get(key)

    def put(self, key: str, plan: Dict[str, Any], question: str) -> None:
        with self._lock:
            self._data[key] = {"q": question[:500], "plan": plan}
            self._dirty = True

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            if not self._dirty:
                return
            data = dict(self._data)
            self._dirty = False
        tmp = f"{self.path}.tmp"
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            os.replace(tmp, self.path)   # atomic: crash-safe
        except OSError:
            log.warning("could not persist intent cache to %s", self.path)


# -------------------------------------------------------------------- router

class IntentRouter:
    """Classify + normalize one employee input with an LLM (cached)."""

    def __init__(
        self,
        llm=None,
        model: str = "",
        cache_path: Optional[str] = None,
        prompt: str = INTENT_SYSTEM,
        prompt_version: str = "v1",
        max_tokens: int = 1500,
        reasoning_effort: str = "",
    ) -> None:
        self.llm = llm
        self.model = model
        self.prompt = prompt
        self.prompt_version = prompt_version
        # editing the prompt must invalidate cached verdicts even when the
        # version label stays "v1" (which is what happens while tuning)
        self.prompt_fingerprint = hashlib.sha256(
            (prompt or "").encode("utf-8")).hexdigest()[:10]
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.cache = IntentCache(cache_path)

    @property
    def _version_tag(self) -> str:
        # the same prompt at a different reasoning budget is a different
        # classifier: keep their cached decisions apart
        return f"{self.prompt_version}|eff={self.reasoning_effort or 'default'}"

    # ------------------------------------------------------------------ api

    def classify(self, question: str,
                 corpus: Optional[List[Dict[str, str]]] = None,
                 history: Optional[List[Dict[str, str]]] = None) -> IntentPlan:
        """Classify one input. ``corpus`` is [{name, brief}] - orientation only.

        The briefs change *how* the input is classified (vocabulary, which
        document it concerns, whether it is a listing request), never *whether*
        the corpus covers it: that judgement stays with the agent, which has read
        the passages. The prompt says so in as many words.
        """
        q = (question or "").strip()

        # Input validation, not NLU: an empty query is never a policy question.
        if not q:
            return IntentPlan(
                intent="CHITCHAT_OR_OUT_OF_SCOPE", route="FAST_REFUSE",
                refusal_reason="empty question",
                decided_by="validation", prompt_version=self.prompt_version,
                notes=["empty input rejected before the classifier"],
            )

        corpus_block, corpus_fp = format_corpus(corpus)
        conv_block, conv_fp = format_conversation(history)
        user_text = f"{conv_block}\n\n{corpus_block}\n\nQUESTION:\n{q}"

        key = IntentCache.key(self.model, self._version_tag, q,
                              self.prompt_fingerprint, corpus_fp, conv_fp)
        cached = self.cache.get(key)
        if cached is not None:
            plan = self._from_cache(cached.get("plan") or {}, q)
            if plan is not None:
                return plan

        if self.llm is None:
            return self._fallback(q, "no LLM configured for the intent layer")

        t0 = time.time()
        try:
            obj, err = self.llm.chat_json(
                self.prompt, user_text, temperature=0.0, max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort or None,
            )
        except Exception as e:              # endpoint down / timeout / auth
            log.warning("intent classifier failed (%s); failing open", e)
            return self._fallback(q, f"classifier error: {e}")

        if not isinstance(obj, dict):
            log.warning("intent classifier returned unparseable JSON (%s); failing open", err)
            return self._fallback(q, f"unparseable classifier reply: {err}")

        plan = plan_from_obj(obj, q, self.prompt_version)
        if plan is None:
            # The reply parsed as JSON but is not a usable plan: the usual
            # cause is output truncation (finish_reason=length) producing a
            # fragment that still looks like an object. One corrective retry,
            # then fail open.
            log.warning("intent classifier returned an unusable plan (%r); retrying once",
                        obj.get("intent"))
            retry_user = (
                f"{user_text}\n\n[classifier correction] Your previous JSON was unusable "
                f"(invalid or truncated `intent`: {obj.get('intent')!r}). Return the "
                "COMPLETE JSON object with one exact category from the schema."
            )
            try:
                obj2, err2 = self.llm.chat_json(
                    self.prompt, retry_user, temperature=0.0, max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort or None,
                )
            except Exception as e:
                return self._fallback(q, f"classifier error on retry: {e}")
            plan = plan_from_obj(obj2 or {}, q, self.prompt_version) if isinstance(obj2, dict) else None
            if plan is None:
                return self._fallback(q, f"unusable classifier reply: {err2 or obj2!r}")
            plan.notes.append("recovered after one corrective retry")

        plan.latency_s = time.time() - t0
        self.cache.put(key, plan.to_dict(), q)
        self.cache.save()
        return plan

    # ------------------------------------------------------------- internals

    @staticmethod
    def _from_cache(obj: Dict[str, Any], question: str) -> Optional[IntentPlan]:
        plan = plan_from_obj(obj, question, str(obj.get("prompt_version") or ""))
        if plan is None:
            return None
        plan.route = str(obj.get("route") or CATEGORY_TO_ROUTE[plan.intent])
        plan.decided_by = "cache"
        plan.latency_s = 0.0
        plan.notes = []
        return plan

    @staticmethod
    def _fallback(question: str, why: str) -> IntentPlan:
        """Fail open: hand the raw question to the grounded agent."""
        return IntentPlan(
            intent="POLICY_DIRECT_LOOKUP", route="REACT_LOOP",
            de_noised_question=question,
            search_queries=[question],
            decided_by="fallback",
            notes=[why],
        )
