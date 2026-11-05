"""Intelligent, structure-aware chunking.

NOT fixed-length cutting. Boundaries are chosen at, in order of preference:

1. heading boundaries (markdown ``#``, numbered sections, ALL-CAPS titles,
   labelled sections, underlined titles) -> every chunk carries a breadcrumb
   ``doc > section > subsection`` used for embedding context and citations;
2. blank-line paragraph boundaries (soft target size);
3. line boundaries inside oversized paragraph blocks;
4. sentence boundaries inside oversized lines (with a hard-split fallback).

Windows (for the read+parse pipeline) are also built here, on
paragraph boundaries with a small carry-over so the LLM never sees the
whole large document at once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Heading:
    pos: int
    level: int
    title: str


@dataclass
class Chunk:
    text: str
    section: str
    breadcrumb: str
    char_count: int
    index: int = 0


# --------------------------------------------------------------------- regexes

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.{1,120})$")
_NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+([^\n]{3,90})$")
_ALLCAPS_RE = re.compile(r"^[A-Z][A-Z0-9\s.,&'/()\-]{3,78}$")
_LABELLED_RE = re.compile(
    r"^(section|article|policy|chapter|part|appendix|annex)\s+"
    r"[A-Za-z0-9IVXLC]+\b[.:)]?\s*[^\n]{0,80}$",
    re.I,
)
_UNDERLINE_RE = re.compile(r"^\s*(={3,}|-{3,})\s*$")


def _is_sentence_like(line: str) -> bool:
    return len(line.split()) >= 10 and line.rstrip().endswith(".")


def detect_headings(text: str) -> List[Heading]:
    headings: List[Heading] = []
    lines = text.split("\n")
    offset = 0
    prev_blank = True
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        pos = offset
        offset += len(ln) + 1
        if not stripped:
            prev_blank = True
            continue

        level: Optional[int] = None
        title: Optional[str] = None

        m = _MD_HEADING_RE.match(stripped)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip().rstrip("#").strip()
        elif _LABELLED_RE.match(stripped):
            level = 1
            title = stripped
        else:
            m = _NUMBERED_RE.match(stripped)
            if (m and prev_blank and len(stripped) <= 90
                    and len(stripped.split()) <= 12
                    and not stripped.endswith((";", ",", ":", "."))):
                depth = m.group(1).count(".") + 1
                level = min(depth, 4)
                title = stripped
            elif (_ALLCAPS_RE.match(stripped) and prev_blank
                  and len(stripped) > 4 and not _is_sentence_like(stripped)):
                level = 1
                title = stripped.title()
            elif (i + 1 < len(lines) and _UNDERLINE_RE.match(lines[i + 1])
                  and len(stripped) <= 90 and not _is_sentence_like(stripped)):
                level = 2 if lines[i + 1].strip().startswith("-") else 1
                title = stripped

        if level and title and not _is_sentence_like(stripped):
            headings.append(Heading(pos=pos, level=level, title=title[:100]))
        prev_blank = False
    return headings


def _sections(text: str) -> List[Tuple[List[str], str]]:
    """Split into (breadcrumb_parts, body) sections at heading boundaries."""
    headings = detect_headings(text)
    if not headings:
        return [([], text)]
    out: List[Tuple[List[str], str]] = []
    if headings[0].pos > 80:
        out.append(([], text[: headings[0].pos]))
    stack: List[Tuple[int, str]] = []
    for idx, h in enumerate(headings):
        while stack and stack[-1][0] >= h.level:
            stack.pop()
        crumb = [t for _, t in stack] + [h.title]
        end = headings[idx + 1].pos if idx + 1 < len(headings) else len(text)
        out.append((crumb, text[h.pos:end]))
        stack.append((h.level, h.title))
    return out


# ------------------------------------------------------------------- splitters

def _hard_split(s: str, max_chars: int) -> List[str]:
    out: List[str] = []
    for sep in ("; ", ", ", " "):
        parts = [p for p in s.split(sep) if p]
        cur = ""
        pieces: List[str] = []
        for p in parts:
            cand = cur + sep + p if cur else p
            if len(cand) > max_chars and cur:
                pieces.append(cur)
                cur = p
            else:
                cur = cand
        if cur:
            pieces.append(cur)
        if all(len(p) <= max_chars for p in pieces):
            return pieces
        out = pieces
    # last resort: fixed cut (only for pathological unbreakable strings)
    return [s[i:i + max_chars] for i in range(0, len(s), max_chars)]


def _sentence_split(text: str, max_chars: int) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out: List[str] = []
    cur = ""
    for p in parts:
        if len(p) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            out.extend(_hard_split(p, max_chars))
            continue
        if cur and len(cur) + len(p) + 1 > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}" if cur else p
    if cur:
        out.append(cur)
    return out


def _block_units(block: str, target: int, max_chars: int) -> List[str]:
    """Break one blank-line block into units no longer than *max_chars*."""
    if len(block) <= max_chars:
        return [block]
    units: List[str] = []
    if "\n" in block:
        cur = ""
        for ln in block.split("\n"):
            if not ln.strip():
                if cur:
                    units.append(cur)
                    cur = ""
                continue
            if cur and len(cur) + len(ln) + 1 > target:
                units.append(cur)
                cur = ln
            else:
                cur = f"{cur}\n{ln}" if cur else ln
        if cur:
            units.append(cur)
    else:
        units = [block]
    final: List[str] = []
    for u in units:
        if len(u) <= max_chars:
            final.append(u)
        else:
            final.extend(_sentence_split(u, max_chars))
    return final


# ------------------------------------------------------------------- chunking

def chunk_text(text: str, doc_name: str, target: int = 1400,
               max_chars: int = 2200) -> List[Chunk]:
    """Structure-aware chunking of a whole document."""
    target = max(120, target)
    max_chars = max(target, max_chars)
    raw_chunks: List[Tuple[str, str, str]] = []  # (text, section, breadcrumb)

    for crumb, body in _sections(text):
        if not body.strip():
            continue
        title = crumb[-1] if crumb else "Preamble"
        breadcrumb = " > ".join([doc_name] + [c[:60] for c in crumb]) if crumb else doc_name
        blocks = [b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()]
        units: List[str] = []
        for b in blocks:
            units.extend(_block_units(b, target, max_chars))

        cur = ""
        for u in units:
            if not u.strip():
                continue
            if cur and (
                len(cur) + len(u) + 2 > max_chars
                or (len(cur) + len(u) + 2 > target and len(cur) >= 200)
            ):
                raw_chunks.append((cur, title, breadcrumb))
                cur = u
            else:
                cur = f"{cur}\n\n{u}" if cur else u
        if cur.strip():
            raw_chunks.append((cur, title, breadcrumb))

    # merge a tiny trailing chunk into its predecessor
    merged: List[Tuple[str, str, str]] = []
    for rc in raw_chunks:
        if merged and len(rc[0]) < 150 and len(merged[-1][0]) + len(rc[0]) + 2 <= int(max_chars * 1.05):
            t, s, b = merged[-1]
            merged[-1] = (f"{t}\n\n{rc[0]}", s, b)
        else:
            merged.append(rc)

    return [
        Chunk(text=t, section=s, breadcrumb=b, char_count=len(t), index=i)
        for i, (t, s, b) in enumerate(merged)
    ]


# -------------------------------------------------------------------- windows

def _tail_chars(s: str, n: int) -> str:
    if n <= 0 or len(s) <= n:
        return ""
    tail = s[-n:]
    sp = tail.find(" ")
    return tail[sp + 1:] if sp >= 0 else tail


def split_into_windows(text: str, window_chars: int = 6000,
                       overlap: int = 200) -> List[str]:
    """Split a long document into consecutive windows on paragraph boundaries.

    Each window starts with a small carry-over tail of the previous window,
    so the LLM parser always has a bit of context - but never the whole
    document at once."""
    window_chars = max(500, window_chars)
    paras: List[str] = []
    for b in re.split(r"\n\s*\n", text):
        b = b.strip()
        if not b:
            continue
        if len(b) <= window_chars:
            paras.append(b)
        else:
            paras.extend(_block_units(b, window_chars, window_chars))

    windows: List[str] = []
    cur = ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > window_chars:
            windows.append(cur)
            tail = _tail_chars(cur, overlap)
            cur = f"{tail}\n\n{p}" if tail else p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur.strip():
        windows.append(cur)
    return windows
