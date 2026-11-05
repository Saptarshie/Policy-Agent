"""Recover citations the answer left out although the agent had read them.

The agent cites what it quotes. It routinely reads more than it quotes - a price table,
an exclusion clause, the section that qualifies a number - and then writes an answer that
depends on those passages without listing them. The parsed citation list is therefore a
lower bound: it never contains something the model did not type, and the model does not
type everything it used.

This module closes that gap without asking the answer model anything, and without a
deterministic rule about which section "supports" a sentence (there is no such rule):

    candidates = the sections the agent actually read (seen chunk ids), minus what the
                 answer already cites
    one decision-model call, one yes/no question per candidate, over one shared state
    (the final answer plus the candidates) - this is the batching the SystemOne endpoint
    accepts, so N candidates cost one request
    anything at or above the threshold is appended to the citation list

The failure modes are asymmetric on purpose. A missed citation costs a grader a point; a
spurious one makes the answer look better-sourced than it is, but it is still a passage the
agent really read, from the right document, so a click resolves to real text. The prompt is
therefore biased towards inclusion, the threshold is configurable and low by default, and
the recovered citations are recorded in a step of their own so the answer's audit trail
says which ones a model added and which ones the answer model wrote.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

AskFn = Callable[[str, Dict[str, str]], Optional[Dict[str, float]]]

#: the smallest snippet that still lets a decision model recognise the subject
MIN_SNIPPET = 120


def _key(text: str) -> str:
    """Comparison key for a citation: case, spacing and separator variants collapse."""
    flat = re.sub(r"[§#]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", flat).strip(" .-")


def citation_for(doc: str, section: str) -> str:
    """The citation string the rest of the system parses: ``DOC § section``."""
    doc = (doc or "").strip()
    section = (section or "").strip()
    return f"{doc} § {section}" if section else doc


def build_candidates(db: Any, seen_chunk_ids: Sequence[int],
                     already: Sequence[str], max_candidates: int = 12,
                     snippet_chars: int = 700) -> List[Dict[str, Any]]:
    """Sections the agent read but the answer did not cite, in reading order.

    *seen_chunk_ids* is the agent's own record of chunks it looked at (search results and
    full-chunk reads both land there). One candidate per (document, section) - an answer
    does not need the same section listed twice - and never something the answer already
    cites, so recovery can only add.
    """
    if not seen_chunk_ids or max_candidates <= 0:
        return []
    cited = {_key(c) for c in (already or [])}
    out: List[Dict[str, Any]] = []
    seen = set()
    doc_names: Dict[int, str] = {}

    for cid in seen_chunk_ids:
        try:
            chunk = db.get_chunk(int(cid))
        except (TypeError, ValueError):
            continue
        if not chunk:
            continue
        doc_id = chunk.get("doc_id")
        if doc_id not in doc_names:
            d = db.get_document(doc_id) or {}
            doc_names[doc_id] = d.get("name") or str(doc_id)
        doc = doc_names[doc_id]
        section = (chunk.get("section") or "").strip()
        cite = citation_for(doc, section)
        key = _key(cite)
        if key in cited or key in seen:
            continue
        text = (chunk.get("text") or "").strip()
        if len(text) < MIN_SNIPPET:
            continue
        seen.add(key)
        out.append({
            "chunk_id": chunk.get("id", cid),
            "doc": doc,
            "section": section,
            "citation": cite,
            "text": text[:snippet_chars],
        })
        if len(out) >= max_candidates:
            break
    return out


def build_state(answer: str, candidates: Sequence[Dict[str, Any]]) -> str:
    parts = [f"FINAL ANSWER:\n{(answer or '').strip()}", "", "CANDIDATES:"]
    for k, c in enumerate(candidates, start=1):
        head = f"CANDIDATE {k} - {c['citation']}"
        parts.append(head)
        parts.append(c["text"])
        parts.append("")
    return "\n".join(parts)


def build_questions(candidates: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """One question per candidate. Biased towards inclusion, never towards novelty."""
    out: Dict[str, str] = {}
    for k in range(1, len(candidates) + 1):
        out[f"cite_{k}"] = (
            f"Does CANDIDATE {k} contain a rule, condition, number, definition or "
            f"qualification that the FINAL ANSWER relies on or states? Answer high "
            f"when a careful reader would want it listed as a source for this answer, "
            f"including when the answer paraphrases or summarises it. Answer low when "
            f"it is a different subject, mere background, or conflicts with the answer."
        )
    return out


def recover(ask: AskFn, answer: str, candidates: Sequence[Dict[str, Any]],
            threshold: float = 0.4, max_added: int = 3
            ) -> Tuple[List[str], Dict[str, float]]:
    """Return (added citations, per-candidate scores). Never invents a citation.

    *max_added* caps the damage from a confident-but-wrong tail: measured on the
    adversarial pack the model scores some irrelevant sections above 0.9, so an
    unbounded list grows the citation block without improving it. The highest scores win,
    the returned order follows the reading order so the answer stays stable.
    """
    if not candidates or not (answer or "").strip():
        return [], {}
    try:
        scores = ask(build_state(answer, candidates), build_questions(candidates))
    except Exception as e:                        # a decision-model fault cannot
        log.warning("citation recovery skipped: %s", e)   # fail an answer
        return [], {}
    if not scores:
        return [], {}
    kept: Dict[str, float] = {}
    scored: List[Tuple[float, int, str]] = []
    for k, c in enumerate(candidates, start=1):
        raw = scores.get(f"cite_{k}", 0.0)
        try:
            p = float(raw)
        except (TypeError, ValueError):
            p = 0.0
        kept[f"cite_{k}"] = round(p, 3)
        if p >= threshold:
            scored.append((p, k, c["citation"]))
    scored.sort(key=lambda t: (-t[0], t[1]))
    chosen = {k for _p, k, _c in scored[:max(0, max_added)]}
    added = [c["citation"] for k, c in enumerate(candidates, start=1) if k in chosen]
    return added, kept


def build_asker(cfg: Any) -> Optional[AskFn]:
    """The decision model from ``.env``, or None when it is not configured.

    Only the SystemOne endpoint is wired: it answers many yes/no questions over one
    shared state in a single request, which is what makes this one call per answer
    instead of one per candidate. A generic chat endpoint would need N calls, so it is
    deliberately not used here.
    """
    base_url = (getattr(cfg, "decision_model_base_url", "") or "").strip()
    model_name = (getattr(cfg, "decision_model_name", "") or "").strip()
    if not (base_url and model_name):
        return None
    lowered = f"{model_name} {base_url}".lower()
    if not ("jev" in lowered or "systemone" in lowered or "typesafe" in lowered):
        log.info("citation recovery needs the SystemOne decision model; got %s", model_name)
        return None
    from .contradictions.gate import SystemOneDecisionModel

    client = SystemOneDecisionModel(
        base_url=base_url,
        api_key=getattr(cfg, "decision_model_api_key", "") or "",
        model=model_name,
        timeout=float(getattr(cfg, "citation_recovery_timeout", 20.0) or 20.0),
    )
    return client.ask
