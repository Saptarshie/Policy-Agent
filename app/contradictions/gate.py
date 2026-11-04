"""Stage 2 of the contradiction cascade: the JEV-style DECISION MODEL.

The decision model comes in three pluggable flavours (``DECISION_MODE`` in
``.env``):

* ``heuristic`` (default) - :class:`JEVStyleGate`: a cheap, deterministic,
  explainable lexical judge implemented IN CODE (regex lexicons + weighted
  signals + threshold). Zero LLM cost, zero latency - this is the classic
  "JEV-style" heuristic gate and needs NO endpoint.
* ``remote``              - :class:`RemoteDecisionModel`: score pairs with a
  dedicated, separately configured OpenAI-compatible model
  (``DECISION_MODEL_BASE_URL`` / ``DECISION_MODEL_API_KEY`` /
  ``DECISION_MODEL_NAME``) that returns strict JSON
  ``{"score": 0..1, "signals": [...]}``. Use it when you want a small
  served classifier/LLM as the decision model.
* ``hybrid``              - :class:`CascadeGate` in hybrid mode: the free
  heuristic runs first; only BORDERLINE pairs (score within
  ``DECISION_BORDERLINE_BAND`` of the threshold) are refined by the remote
  decision model. Clear pass / clear fail never touch the network - best
  precision-per-cost.

:class:`CascadeGate` is the facade the tracker uses; it exposes the same
``evaluate(text_a, text_b) -> GateDecision`` contract in every mode and
falls back to the heuristic when the remote model fails or is unconfigured,
so the cascade never breaks.

Heuristic signals (each weighted, all reported in the decision):

* topical-overlap gate  - pairs talking about different subjects are
  rejected outright (prevents "30 days" vs "3 days" false alarms on
  unrelated topics);
* normative flip        - requirement ("must/shall/required") in one
  passage vs prohibition ("must not/prohibited") in the other;
* permission flip       - permission ("may/allowed/optional") vs
  prohibition, or requirement vs optional;
* numeric conflicts     - same unit, different value, overlapping context
  ("15 days" vs "30 days" notice; "3 days" vs "1 day" remote work);
* antonym pairs         - increase/decrease, remote/onsite,
  permanent/temporary, ... split across the two passages.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple

import requests

from .. import prompts

log = logging.getLogger(__name__)

# --------------------------------------------------------------------- lexicons

_REQ_RE = re.compile(
    r"\b(?:must|shall|is\s+required\s+to|are\s+required\s+to|required\s+to|"
    r"mandatory|obligated\s+to|obliged\s+to|has\s+to|have\s+to)\b", re.I)
_PROHIB_RE = re.compile(
    r"\b(?:must\s+not|shall\s+not|may\s+not|can\s?not|is\s+not\s+allowed|"
    r"are\s+not\s+allowed|not\s+allowed|not\s+permitted|not\s+authorized|"
    r"prohibited|forbidden|disallowed|barred)\b", re.I)
_PERM_RE = re.compile(
    r"\b(?:may|can|is\s+allowed|are\s+allowed|allowed\s+to|permitted|"
    r"optional|authorized\s+to|entitled\s+to)\b", re.I)

_NUM_UNIT_RE = re.compile(
    r"(?<![\w.,$€₹])(\d{1,6}(?:\.\d+)?)\s*"
    r"(%|percent|per\s+cent|business\s+days?|days?|weeks?|months?|years?|"
    r"hours?|minutes?|times|USD|EUR|GBP|INR|usd|eur|gbp|inr|\$|€|₹)?")

ANTONYM_PAIRS: List[Tuple[str, str]] = [
    ("increase", "decrease"), ("raise", "reduce"), ("raise", "lower"),
    ("extend", "shorten"), ("include", "exclude"), ("internal", "external"),
    ("remote", "onsite"), ("remote", "in-office"), ("onsite", "offsite"),
    ("temporary", "permanent"), ("limited", "unlimited"),
    ("approve", "reject"), ("approval", "rejection"), ("refund", "charge"),
    ("before", "after"), ("above", "below"), ("minimum", "maximum"),
    ("more", "fewer"), ("more", "less"), ("earlier", "later"),
    ("mandatory", "voluntary"), ("paid", "unpaid"),
    ("renewable", "non-renewable"), ("eligible", "ineligible"),
    ("public", "private"), ("new", "existing"), ("start", "stop"),
    ("allow", "deny"), ("permitted", "prohibited"), ("allowed", "forbidden"),
    ("submit", "waive"), ("grant", "deny"), ("increase", "cap"),
]

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with",
    "by", "at", "as", "is", "are", "be", "been", "was", "were", "this",
    "that", "these", "those", "it", "its", "from", "will", "shall", "must",
    "may", "can", "could", "should", "would", "not", "no", "yes", "if",
    "then", "than", "so", "such", "any", "all", "each", "every", "who",
    "whom", "which", "what", "when", "where", "how", "their", "his", "her",
    "our", "your", "my", "we", "you", "they", "he", "she", "i", "us",
    "them", "also", "into", "within", "under", "over", "between", "after",
    "before", "during", "while", "about", "against", "out", "up", "down",
    "off", "again", "further", "once", "here", "there", "own", "same",
    "etc", "per", "via", "using", "use", "used", "only", "least", "least",
    "at", "least", "based", "following", "following",
}
# generic words that must NOT count as shared "topic" context for numeric
# conflicts (otherwise every pair mentioning "employees ... days" fires)
_GENERIC = {
    "employee", "employees", "staff", "company", "companies", "organization",
    "policy", "policies", "document", "documents", "section", "work",
    "works", "working", "day", "days", "week", "weeks", "month", "months",
    "year", "years", "time", "period", "acme", "corp", "corporation",
    "must", "may", "shall", "will", "can", "any", "all", "one", "two",
    "given", "made",
}


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z][a-z\-']{2,}", text.lower())


def _content_words(text: str) -> List[str]:
    return [w for w in _words(text) if w not in _STOPWORDS and len(w) > 2]


def _stem(w: str) -> str:
    for suf in ("ies", "es", "s", "ing", "ed", "ly", "ment", "tion"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: len(w) - len(suf)]
    return w


def _norm_unit(u: str) -> str:
    u = (u or "").strip().lower()
    if u in ("%", "percent", "per cent"):
        return "percent"
    if u.startswith("business day"):
        return "days"
    for stem in ("day", "week", "month", "year", "hour", "minute", "time"):
        if u.startswith(stem):
            return stem + "s"
    if u in ("usd", "$"):
        return "usd"
    if u in ("eur", "€"):
        return "eur"
    if u == "gbp":
        return "gbp"
    if u in ("inr", "₹"):
        return "inr"
    return u


def _ctx_stems(text: str, start: int, end: int) -> Set[str]:
    ctx = text[max(0, start - 70): end + 70].lower()
    return {_stem(w) for w in _content_words(ctx)
            if w not in _GENERIC and len(w) > 2}


def _extract_numbers(text: str) -> List[Tuple[str, str, Set[str]]]:
    out: List[Tuple[str, str, Set[str]]] = []
    for m in _NUM_UNIT_RE.finditer(text):
        val = m.group(1)
        unit = _norm_unit(m.group(2) or "")
        out.append((val, unit, _ctx_stems(text, m.start(), m.end())))
    return out


def _sets_overlap(a: Set[str], b: Set[str]) -> bool:
    if a & b:
        return True
    for x in a:
        for y in b:
            if len(x) >= 5 and len(y) >= 5 and x[:5] == y[:5]:
                return True
    return False


@dataclass
class GateDecision:
    passed: bool
    score: float
    signals: List[str] = field(default_factory=list)
    topical_overlap: float = 0.0
    decided_by: str = "heuristic"   # "heuristic" | "remote" | "hybrid"
    error: str = ""                 # non-empty when the remote model failed

    def as_dict(self) -> dict:
        return {
            "passed": self.passed, "score": round(self.score, 3),
            "signals": self.signals, "decided_by": self.decided_by,
            "error": self.error,
        }


class JEVStyleGate:
    """The heuristic decision model ("cheap judge") of the cascade."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)

    # ------------------------------------------------------------------ score

    def evaluate(self, text_a: str, text_b: str) -> GateDecision:
        wa = _content_words(text_a or "")
        wb = _content_words(text_b or "")
        if not wa or not wb:
            return GateDecision(False, 0.0, ["one passage has no content words"],
                                decided_by="heuristic")
        sa, sb = set(wa), set(wb)
        jac = len(sa & sb) / max(1, len(sa | sb))
        ba = set(zip(wa, wa[1:]))
        bb = set(zip(wb, wb[1:]))
        shared_bigram = bool(ba & bb)

        # ---- topical-overlap gate (precision guard)
        if not (jac >= 0.10 or shared_bigram):
            return GateDecision(
                False, 0.0,
                [f"topical overlap too low (jaccard={jac:.2f}, no shared bigram)"],
                jac, decided_by="heuristic",
            )

        signals: List[str] = []
        score = 0.0
        a_low, b_low = (text_a or "").lower(), (text_b or "").lower()
        a_req, a_pro, a_perm = (
            bool(_REQ_RE.search(a_low)), bool(_PROHIB_RE.search(a_low)),
            bool(_PERM_RE.search(a_low)))
        b_req, b_pro, b_perm = (
            bool(_REQ_RE.search(b_low)), bool(_PROHIB_RE.search(b_low)),
            bool(_PERM_RE.search(b_low)))

        if (a_req and b_pro) or (a_pro and b_req):
            score += 0.45
            signals.append("normative flip: requirement vs prohibition")
        if (a_perm and b_pro) or (a_pro and b_perm):
            score += 0.25
            signals.append("permission vs prohibition")
        if (a_req and b_perm) or (b_req and a_perm):
            score += 0.20
            signals.append("requirement vs permission/optional")

        numeric_evidence = self._numeric_conflicts(text_a or "", text_b or "")
        if numeric_evidence:
            score += 0.55
            signals.extend(numeric_evidence[:3])

        ant_evidence = self._antonym_hits(wa, wb)
        if ant_evidence:
            score += min(0.30, 0.15 * len(ant_evidence))
            signals.extend(ant_evidence[:2])

        if jac > 0.30:
            score += 0.10
            signals.append(f"high topical overlap (jaccard={jac:.2f})")

        score = min(1.0, score)
        passed = score >= self.threshold
        return GateDecision(passed, score, signals, jac, decided_by="heuristic")

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _numeric_conflicts(a: str, b: str) -> List[str]:
        ev: List[str] = []
        na, nb = _extract_numbers(a), _extract_numbers(b)
        for va, ua, ca in na:
            for vb, ub, cb in nb:
                if ua != ub:
                    continue
                try:
                    if float(va) == float(vb):
                        continue
                except ValueError:
                    continue
                if not _sets_overlap(ca, cb):
                    continue
                shared = ", ".join(sorted(ca & cb)[:4]) or "related wording"
                ev.append(
                    f'numeric conflict: "{va} {ua}" vs "{vb} {ua}" '
                    f"(shared context: {shared})"
                )
        # dedupe near-identical evidence lines
        seen, out = set(), []
        for e in ev:
            key = e.split("(shared")[0]
            if key not in seen:
                seen.add(key)
                out.append(e)
        return out

    @staticmethod
    def _antonym_hits(wa: List[str], wb: List[str]) -> List[str]:
        sa, sb = set(wa), set(wb)
        hits: List[str] = []
        for x, y in ANTONYM_PAIRS:
            if (x in sa and y in sb) or (y in sa and x in sb):
                hits.append(f"antonym pair across passages: {x} vs {y}")
        return hits


# ======================================================================
# Remote decision model (DECISION_MODE=remote)
# ======================================================================

class RemoteDecisionModel:
    """Score candidate pairs with a dedicated OpenAI-compatible model.

    Configured entirely from ``.env`` (``DECISION_MODEL_BASE_URL``,
    ``DECISION_MODEL_API_KEY``, ``DECISION_MODEL_NAME``), so the decision
    model can be a DIFFERENT, cheaper endpoint than the main LLM (e.g. a
    0.5b classifier served via Ollama, or a hosted small model).

    Anti-hallucination measures reused from the main agent:
      * the reply goes through the heuristic JSON repair layer
        (``chat_json`` -> ``repair_json``);
      * if parsing still fails, the parse error is fed back for one retry;
      * the ``score`` is clamped to [0, 1] and non-numeric scores are
        rejected (a model that cannot emit a number cannot gate anything);
      * any failure returns ``None`` so the caller can fall back to the
        free heuristic - the cascade never breaks.
    """

    def __init__(self, client) -> None:
        self.client = client  # OpenAICompatClient built with decision vars
        self.model = getattr(client, "model", "remote_model")

    def score(
        self, text_a: str, text_b: str,
        heuristic_signals: Optional[Sequence[str]] = None,
    ) -> Optional[GateDecision]:
        sig = list(heuristic_signals or []) or ["(no heuristic signals)"]
        user = prompts.DECISION_MODEL_USER.format(
            text_a=(text_a or "")[:4000], text_b=(text_b or "")[:4000],
            signals="; ".join(sig[:6]),
        )
        try:
            obj, err = self.client.chat_json(
                system=prompts.DECISION_MODEL_SYSTEM, user=user,
            )
        except Exception as e:  # endpoint down / timeout / auth
            log.warning("decision model endpoint failed: %s", e)
            return None

        if obj is None:
            log.warning("decision model returned unparseable JSON: %s", err)
            return None

        raw_score = obj.get("score")
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            log.warning("decision model score not numeric: %r", raw_score)
            return None
        if not 0.0 <= score <= 1.0:
            score = max(0.0, min(1.0, score))  # clamp stray values

        signals = obj.get("signals") or []
        if not isinstance(signals, list):
            signals = [str(signals)]
        signals = [f"remote decision model: {str(s)[:160]}" for s in signals][:6]
        # the pass/fail cut is applied by CascadeGate's threshold
        return GateDecision(
            passed=score >= 0.5, score=score, signals=signals,
            decided_by="remote",
        )


# ======================================================================
# SystemOne decision model (DECISION_MODEL_NAME=typesafe/jev)
# ======================================================================

class SystemOneDecisionModel:
    """Score candidate pairs using CommandCode's Typesafe / SystemOne API.

    API endpoint: POST <base_url>/systemone
    Payload format:
      {
        "model": "typesafe/jev",
        "state": "...",
        "questions": {
          "contradiction": {
            "type": "noul",
            "instructions": "Do Passage A and Passage B contain conflicting, incompatible, or contradictory rules or requirements?"
          },
          "numeric_conflict": {
            "type": "noul",
            "instructions": "Is there a conflict or discrepancy between numbers, days, time periods, or quantitative limits in the two passages?"
          },
          "normative_flip": {
            "type": "noul",
            "instructions": "Does one passage permit or require something that the other passage restricts, disallows, or prohibits?"
          }
        }
      }
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "typesafe/jev",
        timeout: float = 15.0,
        max_retries: int = 2,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        if not self.base_url.endswith("/systemone"):
            self.endpoint = f"{self.base_url}/systemone"
        else:
            self.endpoint = self.base_url
        self.api_key = api_key or ""
        self.model = model or "typesafe/jev"
        self.timeout = timeout
        self.max_retries = max_retries

    def ask(self, state: str, questions: dict) -> Optional[dict]:
        """Ask the decision model arbitrary yes/no questions about *state*.

        Returns ``{question_name: probability}`` or ``None`` when the endpoint
        is unavailable. ``questions`` is ``{name: instructions}``; each becomes
        a ``noul`` question, which is the model's probability that the statement
        holds.
        """
        payload = {
            "model": self.model,
            "state": (state or "")[:32000],
            "questions": {
                name: {"type": "noul", "instructions": instr}
                for name, instr in questions.items()
            },
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(self.endpoint, headers=headers,
                                     json=payload, timeout=self.timeout)
                if resp.status_code == 429:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                if resp.status_code != 200:
                    log.warning("systemone endpoint returned HTTP %s: %s",
                                resp.status_code, resp.text[:300])
                    return None
                data = resp.json()
            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                log.warning("systemone endpoint failed: %s", e)
                return None
            answers = (data or {}).get("answers") or {}
            out = {}
            for name in questions:
                try:
                    out[name] = float(answers.get(name, {}).get("noul", 0.0))
                except (TypeError, ValueError):
                    out[name] = 0.0
            return out
        return None

    def score(
        self,
        text_a: str,
        text_b: str,
        heuristic_signals: Optional[Sequence[str]] = None,
    ) -> Optional[GateDecision]:
        state_parts = [
            f"Passage A:\n{(text_a or '').strip()[:3500]}",
            f"Passage B:\n{(text_b or '').strip()[:3500]}",
        ]
        if heuristic_signals:
            sigs = [s for s in heuristic_signals if s and not s.startswith("(")]
            if sigs:
                state_parts.append(
                    "Preliminary signals detected:\n"
                    + "\n".join(f"- {s}" for s in sigs[:5])
                )
        state = "\n\n".join(state_parts)

        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                "contradiction": {
                    "type": "noul",
                    "instructions": (
                        "Do Passage A and Passage B contain conflicting, "
                        "incompatible, or contradictory rules or requirements?"
                    ),
                },
                "numeric_conflict": {
                    "type": "noul",
                    "instructions": (
                        "Is there a conflict or discrepancy between numbers, "
                        "days, time periods, or quantitative limits in the two passages?"
                    ),
                },
                "normative_flip": {
                    "type": "noul",
                    "instructions": (
                        "Does one passage permit or require something that the "
                        "other passage restricts, disallows, or prohibits?"
                    ),
                },
            },
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code == 429:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                if resp.status_code != 200:
                    log.warning(
                        "systemone endpoint returned HTTP %s: %s",
                        resp.status_code,
                        resp.text[:300],
                    )
                    return None
                data = resp.json()
                break
            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                log.warning("systemone endpoint failed: %s", e)
                return None

        if not data or not isinstance(data, dict):
            return None

        answers = data.get("answers") or {}
        try:
            c_score = float(answers.get("contradiction", {}).get("noul", 0.0))
        except (TypeError, ValueError):
            c_score = 0.0
        try:
            num_score = float(answers.get("numeric_conflict", {}).get("noul", 0.0))
        except (TypeError, ValueError):
            num_score = 0.0
        try:
            norm_score = float(answers.get("normative_flip", {}).get("noul", 0.0))
        except (TypeError, ValueError):
            norm_score = 0.0

        score = max(c_score, 0.9 * num_score, 0.9 * norm_score)
        score = max(0.0, min(1.0, score))

        signals = [
            f"systemone {self.model}: contradiction={c_score:.2f}, "
            f"numeric={num_score:.2f}, normative={norm_score:.2f}"
        ]
        if c_score >= 0.5:
            signals.append("systemone: direct rule conflict detected")
        if num_score >= 0.5:
            signals.append("systemone: quantitative/numeric discrepancy detected")
        if norm_score >= 0.5:
            signals.append("systemone: normative permission/prohibition clash detected")

        return GateDecision(
            passed=score >= 0.5,
            score=score,
            signals=signals,
            decided_by="remote",
        )


# ======================================================================
# Cascade gate: the facade used by the tracker (all three modes)
# ======================================================================

class CascadeGate:
    """The stage-2 decision model the :class:`ContradictionTracker` uses.

    There is one judge: the configured decision model (``DECISION_MODEL_*``,
    typically ``typesafe/jev``). The lexical gate is *not* a fallback verdict -
    its signals are passed to the model as evidence, because a conflict is a
    question about meaning and word overlap cannot answer it.

    If no decision model is configured the gate refuses to decide: every pair
    comes back ``passed=False``, ``decided_by="unavailable"`` with an explicit
    error, so a scan reports "no model" rather than "no conflicts".
    """

    def __init__(
        self,
        threshold: float = 0.5,
        remote: Optional[RemoteDecisionModel] = None,
    ) -> None:
        self.threshold = float(threshold)
        self.remote = remote
        self.heuristic = JEVStyleGate(threshold=self.threshold)
        if self.remote is None:
            log.warning("no decision model configured: contradiction detection "
                        "will pass nothing (set DECISION_MODEL_BASE_URL and "
                        "DECISION_MODEL_NAME)")

    # ------------------------------------------------------------------ api

    def evaluate(self, text_a: str, text_b: str) -> GateDecision:
        if self.remote is None:
            return GateDecision(
                False, 0.0, ["no decision model configured"],
                decided_by="unavailable",
                error="no decision model configured",
            )
        # the lexical pass still runs: its signals are evidence for the model
        heur = self.heuristic.evaluate(text_a, text_b)
        decision = self.remote.score(text_a, text_b, heur.signals)
        if decision is None:
            return GateDecision(
                False, 0.0, list(heur.signals or []) + ["decision model failed"],
                decided_by="unavailable",
                error="decision model returned no verdict",
            )
        decision.passed = decision.score >= self.threshold
        decision.decided_by = "decision_model"
        return decision

    @property
    def describe(self) -> str:
        if self.remote is None:
            return "no decision model configured"
        return f"decision model ({getattr(self.remote, 'model', '?')})"


def build_gate_from_config(cfg) -> CascadeGate:
    """Build the decision model from ``.env``.

    No mode selection any more: either the endpoint is configured and it judges,
    or the gate reports itself unavailable. ``cfg`` is an ``app.config.Config``
    (or any object with the same attributes).
    """
    base_url = (getattr(cfg, "decision_model_base_url", "") or "").strip()
    model_name = (getattr(cfg, "decision_model_name", "") or "").strip()
    remote = None
    if base_url and model_name:
        lowered = model_name.lower()
        if lowered == "typesafe/jev" or "systemone" in base_url.lower() \
                or "systemone" in lowered:
            remote = SystemOneDecisionModel(
                base_url=base_url,
                api_key=getattr(cfg, "decision_model_api_key", "") or "",
                model=model_name,
            )
        else:
            from ..llm import OpenAICompatClient
            client = OpenAICompatClient(
                base_url=base_url,
                api_key=getattr(cfg, "decision_model_api_key", "") or "",
                model=model_name,
                max_output_tokens=getattr(cfg, "max_output_tokens", 4096),
                temperature=getattr(cfg, "decision_temperature", 0.0),
            )
            remote = RemoteDecisionModel(client)
        log.info("decision model: %s @ %s", model_name, base_url)
    else:
        log.warning("DECISION_MODEL_BASE_URL/NAME are not set: contradiction "
                    "detection is disabled (no lexical fallback by design)")
    return CascadeGate(threshold=getattr(cfg, "jev_gate_threshold", 0.5),
                       remote=remote)

