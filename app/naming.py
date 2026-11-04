"""Resolve a document name the answer used to a document that exists.

Citations are written by a model, so the same document arrives in many shapes:
`remote_work_policy_v2`, `Remote Work Policy v2`, `remote-work-policy-v2.docx`.
Resolving them is structural: normalise the text, then require a *unique*
candidate. It is not a similarity contest.

The old implementation returned the first name that matched, in list order:

    for n in names:
        if p == nl or p in nl or nl in p: return n      # v1 wins for "v2"
    for n in names:
        if difflib(...) >= 0.72: return n                # also v1

With a corpus that contains `remote_work_policy_v1` and `remote_work_policy_v2`,
"Remote Work Policy v2" resolved to **v1** - a citation of the amended policy
silently attributed to the superseded one, and those two documents are the
contradiction pair the whole product exists to surface.

Resolution now has three passes, and every pass requires uniqueness:

    1. normalised equality
    2. whole-token subset either way (the smaller name inside the bigger one)
    3. fuzzy ratio >= 0.90 with a clear margin to the runner-up

Ambiguity is a first-class result (`reason="ambiguous"`), not a coin flip: the
caller turns it into a citation warning, so the reviewer sees it instead of a
wrong attribution.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

_EXT = re.compile(r"\.(md|markdown|txt|pdf|docx?|html?)$", re.I)
_NORM = re.compile(r"[^a-z0-9]+")

FUZZY_THRESHOLD = 0.90      # high on purpose: this is a last resort, not a vote
FUZZY_MARGIN = 0.08         # best must beat the runner-up by this much


def normalise(name: str) -> str:
    """Lowercase, drop a file extension, and collapse separators to spaces.

    `remote_work_policy_v2` / `Remote-Work-Policy-v2.md` / `Remote Work Policy v2`
    all become `remote work policy v2`.
    """
    s = (name or "").strip().lower()
    s = _EXT.sub("", s)
    s = _NORM.sub(" ", s)
    return " ".join(s.split())


def tokens(name: str) -> set:
    return set(normalise(name).split())


@dataclass
class Match:
    name: Optional[str] = None
    how: str = ""              # exact | subset | fuzzy | none | ambiguous
    reason: str = ""

    def __bool__(self) -> bool:
        return self.name is not None


def resolve(name: str, candidates: Iterable[str]) -> Match:
    """Resolve *name* to exactly one of *candidates* (or explain why not)."""
    wanted = normalise(name)
    if not wanted:
        return Match(how="none", reason="empty reference")
    cands: List[str] = [c for c in candidates if c]
    if not cands:
        return Match(how="none", reason="no documents to match against")

    # 1. normalised equality
    exact = [c for c in cands if normalise(c) == wanted]
    if len(exact) == 1:
        return Match(exact[0], "exact")
    if len(exact) > 1:
        return Match(how="ambiguous",
                     reason=f"{len(exact)} documents normalise to the same name: {exact}")

    # 2. whole-token subset, either direction, unique only
    wt = tokens(wanted)
    subset = [c for c in cands
              if wt and (wt <= tokens(c) or tokens(c) <= wt)]
    if len(subset) == 1:
        return Match(subset[0], "subset")
    if len(subset) > 1:
        return Match(how="ambiguous",
                     reason=f"'{name}' matches {len(subset)} documents: {subset[:4]}")

    # 3. fuzzy, unique and clearly ahead
    scored = sorted(((difflib.SequenceMatcher(None, wanted, normalise(c)).ratio(), c)
                     for c in cands), reverse=True)
    best_score, best = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if best_score >= FUZZY_THRESHOLD and (best_score - runner_up) >= FUZZY_MARGIN:
        return Match(best, "fuzzy")
    if best_score >= FUZZY_THRESHOLD:
        return Match(how="ambiguous",
                     reason=(f"'{name}' is {best_score:.2f} similar to {best} but also "
                             f"{runner_up:.2f} to {scored[1][1]}"))
    return Match(how="none", reason=f"no document matches '{name}'")


def resolve_one(name: str, candidates: Sequence[str]) -> Optional[str]:
    """Convenience wrapper: the resolved name, or None."""
    return resolve(name, candidates).name
