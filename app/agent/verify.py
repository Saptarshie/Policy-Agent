"""Post-answer grounding verification (the last anti-hallucination gate).

Runs after the ReAct loop, at the ``AdminAPI.ask`` level:

1. **Evidence check (deterministic)** - the contexts the agent may rely on
   are the passages it actually saw (``result.seen_chunk_ids``, collected
   from tool observations during the run).  A non-refusal answer that was
   produced without ANY retrieved evidence gets an explicit warning.
2. **Faithfulness judge (optional LLM, default on)** - one strict-JSON call
   that compares the final answer against ONLY those retrieved passages and
   lists unsupported claims.  The result is surfaced as
   ``result.faithful`` / ``result.unsupported_claims`` and appended to
   ``citation_warnings`` - the answer itself is never silently rewritten,
   so the admin always sees what was flagged and why.

The judge degrades open: if the LLM is unavailable or returns unparseable
JSON, ``faithful`` stays ``None`` and no warning is invented.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..db import Database
from ..llm import OpenAICompatClient
from ..prompts import FAITHFULNESS_SYSTEM
from .answer_shape import AnswerShape, build_shape_classifier

log = logging.getLogger(__name__)

#: sentinel for "build a shape classifier from whatever client is available"
_AUTO = object()

_WS = re.compile(r"\s+")

def _norm(s: str) -> str:
    """Lowercase and collapse whitespace. Structural normalisation only."""
    return _WS.sub(" ", (s or "")).strip().lower()


def quote_matches(quote: str, text: str, threshold: float = 0.82) -> bool:
    """Fuzzy containment: *quote* should appear (approximately) in *text*."""
    q, t = _norm(quote), _norm(text)
    if not q:
        return False
    if q in t:
        return True
    if len(q) > 260:
        q = q[:260]
    win = max(int(len(q) * 1.4), 60)
    step = max(40, win // 4)
    best = 0.0
    for i in range(0, max(1, len(t) - len(q) + 1), step):
        seg = t[i:i + len(q)]          # length-matched window for a fair ratio
        r = difflib.SequenceMatcher(None, q, seg).ratio()
        if r > best:
            best = r
        if best >= threshold:
            return True
    # last resort: compare against the whole (short) text
    if not t:
        return False
    return difflib.SequenceMatcher(None, q, t[:len(q) + 60]).ratio() >= threshold


@dataclass
class FaithfulnessReport:
    ran: bool                       # did the LLM judge actually run?
    faithful: Optional[bool] = None
    unsupported_claims: List[str] = field(default_factory=list)
    reason: str = ""
    error: Optional[str] = None


def gather_contexts(db: Database, chunk_ids: Sequence[int],
                    max_chunks: int = 24, max_chars: int = 24000) -> List[str]:
    """Fetch the texts of the chunks the agent saw, bounded for the judge.

    The old bound (8 chunks / 9000 chars) was smaller than what a multi-hop or
    conflict answer actually reads, and the chunks were taken in *arrival*
    order - so the judge routinely never saw the passages the answer cited and
    reported them as invented. Callers should order *chunk_ids* citations-first
    (see ``AnswerVerifier._judge_context_ids``).
    """
    out: List[str] = []
    used = 0
    for cid in list(chunk_ids)[:max_chunks]:
        chunk = db.get_chunk(int(cid))
        if not chunk:
            continue
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        if used + len(text) > max_chars and out:
            break
        out.append(text)
        used += len(text)
    return out


def cited_contexts(db: Database, ids: Sequence[int],
                   max_chars: int = 24000) -> Tuple[List[str], List[int]]:
    """Judge inputs for the cited evidence, then as much of the rest as fits."""
    texts: List[str] = []
    used_ids: List[int] = []
    used = 0
    for cid in ids:
        chunk = db.get_chunk(int(cid))
        if not chunk:
            continue
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        if used + len(text) > max_chars and texts:
            break
        texts.append(text)
        used_ids.append(int(cid))
        used += len(text)
    return texts, used_ids


class AnswerVerifier:
    """Deterministic evidence check + optional LLM faithfulness judge."""

    def __init__(self, db: Database, llm: Optional[OpenAICompatClient],
                 enabled: bool = True,
                 system_prompt: str = FAITHFULNESS_SYSTEM,
                 cache_path: Optional[str] = None,
                 prompt_version: str = "",
                 shape_classifier=_AUTO) -> None:
        self.db = db
        self.llm = llm
        self.enabled = enabled
        self.system_prompt = system_prompt
        self.prompt_version = prompt_version
        # Verdicts vary run to run even at temperature 0 (provider-side), which
        # makes demos and regressions noisy. An identical (question, answer,
        # contexts, prompt) triple therefore replays its verdict from disk.
        # What kind of answer is this? Asked of the decision model, never of a
        # keyword list (see app/agent/answer_shape.py for why).
        # AUTO -> build one (decision model, else the chat model's JSON mode);
        # None -> classification disabled (the judge then runs on every answer)
        self.shape_classifier = (build_shape_classifier(llm=llm)
                                 if shape_classifier is _AUTO
                                 else shape_classifier)
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: Dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except ValueError:
                log.warning("judge cache unreadable; starting empty")

    # ---------------------------------------------------------------- cache

    def _cache_key(self, question: str, answer: str, contexts: Sequence[str]) -> str:
        raw = "".join([self.prompt_version or "", question or "", answer or "",
                            *contexts])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _cache_get(self, key: str) -> Optional[dict]:
        return self._cache.get(key)

    def _cache_put(self, key: str, report: "FaithfulnessReport") -> None:
        if not self.cache_path:
            return
        self._cache[key] = {"faithful": report.faithful,
                            "unsupported_claims": report.unsupported_claims[:8],
                            "reason": report.reason}
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False),
                                       encoding="utf-8")
        except OSError as e:                 # a cache miss is never fatal
            log.info("judge cache write failed: %s", e)

    # ------------------------------------------------------------------ check

    def check(self, result, question: str,
              user_message: Optional[str] = None) -> None:
        """Mutate *result* (an ``AgentResult``) in place with verification
        findings.  Cheap checks always run; the LLM judge only when enabled,
        configured, and there is an answer + evidence to compare.

        *user_message* is the employee's ORIGINAL wording. It is passed to the
        judge so that claims which merely restate the user's own facts (their
        numbers, dates, tiers) are not counted as unsupported policy claims.
        """
        if not self.enabled:
            return
        if result.error or not (result.answer or "").strip():
            return

        contexts = gather_contexts(self.db, result.seen_chunk_ids)
        judge_ids = self._judge_context_ids(result)

        # ---- what kind of answer is this? (decision model, not keywords)
        shape = self._classify_shape(question, result.answer, contexts)
        if shape is not None:
            result.answer_shape = shape.kind
            result.declined_parts = list(shape.declined_parts)

        # ---- deterministic evidence gate
        # A bare refusal or chit-chat may legitimately have no evidence behind
        # it; anything that states policy must have seen a passage.
        claims_possible = shape is None or shape.has_claims
        if not contexts and claims_possible:
            result.citation_warnings.append(
                "answer was produced without any retrieved evidence "
                "(no policy passages were seen during this run)"
            )

        # ---- LLM faithfulness judge
        # Skip ONLY when the classifier says the answer carries no policy
        # content (chit-chat). Partial answers are always judged: the part that
        # was answered has claims, and the part that was declined is itself a
        # checkable statement about coverage.
        if self.llm is None or not contexts:
            return
        if shape is not None and shape.kind == "chitchat":
            return
        judge_texts, used = cited_contexts(self.db, judge_ids)
        omitted = [cid for cid in judge_ids if cid not in used]
        if omitted:
            # never let a truncated evidence set turn into an accusation
            log.info("judge context truncated: %d of %d seen chunks omitted",
                     len(omitted), len(judge_ids))
        declined = list(shape.declined_parts) if shape else []
        if shape is not None and shape.is_partial and not declined:
            # the classifier saw a decline but could not quote it (the decision
            # model returns scores, not spans) - tell the judge to find it
            declined = ["(the answer refers to parts the documents do not cover - "
                        "identify them from the answer text)"]
        report = self.judge(question, result.answer, judge_texts or contexts,
                            user_message=user_message,
                            truncated=bool(omitted),
                            declined=declined)
        result.judge_context_ids = used
        if not report.ran:
            return
        result.faithful = report.faithful
        if report.faithful is False:
            result.unsupported_claims = report.unsupported_claims[:6]
            why = "; ".join(result.unsupported_claims) or report.reason
            result.citation_warnings.append(
                "faithfulness check FAILED - the answer contains claims not "
                f"supported by the retrieved passages: {why[:400]}"
            )

    def _classify_shape(self, question: str, answer: str,
                        contexts: Sequence[str]) -> Optional[AnswerShape]:
        """Shape of the answer, or None when no classifier is available.

        A classifier outage returns ``kind='unknown'`` (which the caller treats
        as "has claims, judge it"), so failure degrades toward more checking.
        """
        if self.shape_classifier is None:
            return None
        sample = "\n\n---\n\n".join(contexts)[:6000]
        try:
            shape = self.shape_classifier.classify(question, answer, sample)
        except Exception as e:               # never break verification on this
            log.warning("answer-shape classification failed (%s); judging anyway", e)
            return AnswerShape(kind="unknown", decided_by="fallback", error=str(e))
        log.info("answer shape: %s (%s)", shape.kind, shape.decided_by)
        return shape

    def _judge_context_ids(self, result) -> List[int]:
        """Chunks the judge should read: the ones the answer CITES first.

        A conflict or multi-hop answer cites passages it opened late in the run;
        feeding the judge only the earliest chunks makes correct answers look
        invented (observed on the remote-work conflict cases).
        """
        cited: List[int] = []
        for ref in result.citations or []:
            doc_part = re.split(r"[§#]", ref, maxsplit=1)[0].strip().strip('"“”\'')
            section = ref.split("§", 1)[1].strip() if "§" in ref else ""
            doc = self.db.find_document_fuzzy(doc_part) if doc_part else None
            if not doc:
                continue
            chunks = self.db.list_chunks(int(doc["id"]))
            target = _WS.sub(" ", section).strip().lower()
            exact: List[int] = []
            loose: List[tuple] = []
            for chunk in chunks:
                sec = _WS.sub(" ", (chunk.get("section") or "")).strip().lower()
                cid = int(chunk["id"])
                if not target:
                    exact.append(cid)
                elif sec == target:
                    exact.append(cid)
                elif sec and (target in sec):
                    # substring matching alone is ambiguous ("S1" is inside
                    # "S11"), so it is only a fallback and the longest wins
                    loose.append((len(sec), cid))
            picked = exact or ([max(loose)[1]] if loose else [])
            for cid in picked:
                if cid not in cited:
                    cited.append(cid)
            # a document's title / effective date / scope sit in its header
            # chunk, which a section citation does not cover
            if chunks:
                header = min(chunks, key=lambda c: (c.get("chunk_index") or 0, c["id"]))
                if int(header["id"]) not in cited:
                    cited.append(int(header["id"]))
        order = list(cited)
        for cid in result.seen_chunk_ids:
            if int(cid) not in order:
                order.append(int(cid))
        return order

    def _inventory(self) -> str:
        """Every ingested document plus its ingestion summary.

        Titles, effective dates and applicability scope live in a document's
        header, which retrieval rarely returns and a section citation never
        covers - so without this the judge flags correct statements about *which
        document says what* as invented (observed on all remote-work conflict
        cases). The ingestion summary is included because the agent may quote it
        via ``get_document_summary``: it is a legitimate evidence source that is
        not a chunk, so it never appears in the judge's passage window.
        """
        try:
            docs = self.db.list_documents()
        except Exception as e:               # never break verification on this
            log.info("document inventory unavailable: %s", e)
            return ""
        lines = []
        for doc in docs[:40]:
            name = str(doc.get("name") or "")
            if not name:
                continue
            summary = _WS.sub(" ", str(doc.get("summary") or "")).strip()
            lines.append(f"- {name}" + (f": {summary[:600]}" if summary else ""))
        return "\n".join(lines)

    # ------------------------------------------------------------------ judge

    def judge(self, question: str, answer: str,
              contexts: Sequence[str],
              user_message: Optional[str] = None,
              truncated: bool = False,
              declined: Optional[Sequence[str]] = None) -> FaithfulnessReport:
        # the verdict depends on what the judge was ASKED, not only on the answer:
        # the declined-parts block and the truncation note change the question
        key = self._cache_key(
            question, answer,
            [*contexts, f"|declined={list(declined or [])}|truncated={truncated}"])
        hit = self._cache_get(key)
        if hit is not None:
            return FaithfulnessReport(
                ran=True,
                faithful=hit.get("faithful"),
                unsupported_claims=list(hit.get("unsupported_claims") or []),
                reason=str(hit.get("reason") or ""),
            )
        ctx_block = "\n\n---\n\n".join(contexts)[:24000]
        user = (
            f"QUESTION: {question}\n\n"
            f"PROPOSED ANSWER:\n{answer}\n\n"
            f"ALLOWED CONTEXTS (the only sources the answer may rely on):\n"
            f"{ctx_block}"
        )
        if declined:
            listed = "\n".join(f"- {d}" for d in list(declined)[:4])
            user += ("\n\nDECLINED BY THE ANSWER (the answer says the documents do not cover these):\n"
                     f"{listed}\n\n"
                     "A PARTIAL ANSWER IS CHECKED IN TWO HALVES:\n"
                     "1. The part it answered: verify every rule, number, limit, date and "
                     "obligation against the contexts exactly as usual.\n"
                     "2. The part it declined: that is a claim ABOUT THE CORPUS (\"the documents "
                     "do not say\"). It is SUPPORTED only if no context above addresses it. If any "
                     "context above does address it - even partially - the decline is UNSUPPORTED: "
                     "list it as a claim to remove, and quote the passage that contradicts it.")
        if truncated:
            user += ("\n\nNOTE ON COVERAGE: this context set is a truncated subset of the passages the agent retrieved. A claim that merely does not "
                      "appear above MAY still have come from an omitted passage - only list it as unsupported if the text shown contradicts it, and "
                      "otherwise leave it out.")
        if user_message and user_message.strip() and user_message.strip() != question.strip():
            user = (
                f"USER MESSAGE (the employee's own words - context, not evidence):\n"
                f"{user_message.strip()[:2000]}\n\n" + user
            )
        inventory = self._inventory()
        if inventory:
            user += (
                "\n\nDOCUMENT INVENTORY (every ingested document, verified by "
                "listing rather than retrieval):\n"
                f"{inventory}\n\n"
                "SCOPE OF THIS CHECK: the inventory is verified metadata, not "
                "retrieval. Claims about which documents exist, and a document's "
                "own title, effective date and applicability scope as stated in "
                "its inventory line, are SUPPORTED by this inventory and must not "
                "be flagged merely because no passage repeats them. A conclusion "
                "that follows from comparing two passages is supported when both "
                "premises are present. Claims about policy CONTENT (numbers, "
                "rules, tiers, obligations) must appear in a context above."
            )
        try:
            obj, err = self.llm.chat_json(self.system_prompt, user)
        except Exception as e:  # endpoint down -> degrade open, never crash ask()
            log.warning("faithfulness judge LLM error: %s", e)
            return FaithfulnessReport(ran=False, error=str(e))
        report = self._parse_verdict(obj, err)
        if report.faithful is False and not report.unsupported_claims:
            # An accusation with no named claim cannot be acted on (and cannot
            # be shown to the employee). Ask once for the specific claim; if it
            # still cannot name one, the verdict is unsubstantiated -> unknown,
            # never a silent pass and never a false alarm.
            retry_user = user + (
                "\n\nYour previous verdict was \"faithful\": false but did not "
                "name a single unsupported claim. Name and quote the specific "
                "claim(s) that are absent from the contexts, or answer "
                "\"faithful\": true if there are none."
            )
            try:
                obj2, err2 = self.llm.chat_json(self.system_prompt, retry_user)
            except Exception as e:
                log.warning("faithfulness judge retry failed: %s", e)
                return FaithfulnessReport(ran=True, faithful=None,
                                          reason=report.reason,
                                          error=str(e))
            report2 = self._parse_verdict(obj2, err2)
            if report2.faithful is False and not report2.unsupported_claims:
                log.warning("judge verdict unsubstantiated (false, no claims)")
                return FaithfulnessReport(ran=True, faithful=None,
                                          reason=report2.reason or report.reason,
                                          error="unsubstantiated verdict")
            self._cache_put(key, report2)
            return report2
        self._cache_put(key, report)
        return report

    @staticmethod
    def _parse_verdict(obj, err) -> FaithfulnessReport:
        if not isinstance(obj, dict) or not isinstance(obj.get("faithful"), bool):
            return FaithfulnessReport(ran=False, error=err or "invalid JSON verdict")
        claims = obj.get("unsupported_claims")
        if not isinstance(claims, list):
            claims = []
        return FaithfulnessReport(
            ran=True,
            faithful=bool(obj["faithful"]),
            unsupported_claims=[str(c)[:300] for c in claims if isinstance(c, str)][:8],
            reason=str(obj.get("reason") or "")[:400],
        )
