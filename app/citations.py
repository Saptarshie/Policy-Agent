"""Resolve a citation printed in an answer back to the passage it points at.

The UI makes citations clickable so a reader can check the claim against the text.
The agent writes them as `[doc_name § Section Title]` or `[doc_name #chunk_id]`; the
match here is deliberately forgiving (case, punctuation, spacing, a shortened section
title) because the name came out of a model, not out of an index - but it never
invents a passage: when nothing matches it says so and lists the sections the
document actually has.
"""
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _flat(s: str) -> str:
    """lowercase, alphanumerics only - so '4.2 Dental implants' == '42 dental IMPLANTS'"""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def section_number(text: str) -> str:
    """Leading section number of a heading: "8.1 Permitted stacking" -> "8.1".

    Structure, not meaning. It exists so "8." can be recognised as the parent of "8.1"
    and "8.2" when a document only has subsections chunked.
    """
    m = re.match(r"\s*(\d+(?:\.\d+)*)\b", text or "")
    return m.group(1) if m else ""


def parse_citation(cite: str) -> Tuple[str, Optional[int], str]:
    """'[doc § section]' / 'doc #12' -> (doc part, chunk id or None, section part)."""
    text = (cite or "").strip().strip("[]").strip()
    m = re.search(r"#\s*(\d+)\s*$", text)
    if m:
        return text[: m.start()].strip(), int(m.group(1)), ""
    doc, sep, section = text.partition("§")
    return doc.strip(), None, (section.strip() if sep else "")


def section_satisfied(expected: str, cited: Sequence[str]) -> bool:
    """Does citing any of *cited* satisfy the reference's *expected* section?

    Two ways: the section is named outright (substring, case-insensitive), or the
    expected heading only has subsections and a subsection was cited. The second rule is
    not a fudge - a chunker stores "8.1 Permitted stacking", never "8. Stacking and double
    funding", so without it a correct citation looks like a miss.
    """
    want = re.sub(r"\s+", " ", (expected or "").lower()).strip()
    if not want:
        return False
    have = [re.sub(r"\s+", " ", (c or "").lower()).strip() for c in (cited or []) if c]
    if any(want in c for c in have):
        return True
    num = section_number(want)
    if not num:
        return False
    return any((section_number(c) or "").startswith(num + ".") for c in have)


class CitationResolver:
    """Mixin-friendly resolver: needs only `list_documents`, `list_chunks`, `get_chunk`."""

    def resolve_citation(self, cite: str) -> Dict[str, Any]:
        db = self.db                                    # type: ignore[attr-defined]
        doc_part, chunk_id, section = parse_citation(cite)
        out: Dict[str, Any] = {
            "found": False, "cite": (cite or "").strip().strip("[]"),
            "doc": None, "section": section or None, "chunk_id": None,
            "breadcrumb": None, "text": None, "how": None, "parts": [],
            "candidates": [], "reason": None,
        }
        if not (doc_part or chunk_id):
            out["reason"] = "empty"
            return out

        # ---- [doc #12]
        if chunk_id is not None:
            chunk = db.get_chunk(chunk_id)
            if chunk:
                out.update(found=True, chunk_id=chunk["id"], section=chunk["section"],
                           breadcrumb=chunk.get("breadcrumb"), text=chunk["text"],
                           parts=[{"chunk_id": chunk["id"], "text": chunk["text"]}],
                           how="chunk-id",
                           doc=db.get_document(chunk["doc_id"])["name"]
                           if db.get_document(chunk["doc_id"]) else doc_part)
                return out
            out["reason"] = "no-chunk"
            return out

        # ---- match the document name (longest contained match wins)
        flat_doc = _flat(doc_part)
        best: Optional[Dict[str, Any]] = None
        best_len = 0
        for d in db.list_documents():
            name_flat = _flat(d["name"])
            if name_flat and (name_flat in flat_doc or (len(flat_doc) >= 8 and flat_doc in name_flat)):
                if len(name_flat) > best_len:
                    best, best_len = d, len(name_flat)
        if best is None:
            out["reason"] = "no-document"
            return out
        out["doc"] = best["name"]

        chunks = db.list_chunks(best["id"])
        if not chunks:
            out["reason"] = "no-chunks"
            return out

        # ---- match the section, from strict to forgiving
        want = _flat(section)
        ranked: List[Tuple[int, Dict[str, Any]]] = []
        for ch in chunks:
            have = _flat(ch.get("section") or "")
            if not want:
                continue
            if have == want:
                ranked.append((0, ch))
            elif want and (have.startswith(want) or want.startswith(have)):
                ranked.append((1, ch))
            elif want and len(want) >= 6 and want in have:
                ranked.append((2, ch))
            elif want and len(have) >= 6 and have in want:
                ranked.append((3, ch))
        # a heading that only exists as subsections ("8. Stacking and double funding"
        # -> "8.1 Permitted stacking") is a real reference: a reader clicking it wants
        # the subsections, and a chunker never stores the parent as its own section.
        want_num = section_number(section)
        if want_num and not ranked:
            desc = [ch for ch in chunks
                    if (section_number(ch.get("section") or "") or "").startswith(want_num + ".")]
            if desc:
                desc.sort(key=lambda c: c.get("chunk_index") or 0)
                out.update(found=True, chunk_id=desc[0]["id"], section=section,
                           breadcrumb=desc[0].get("breadcrumb"), how="descendant",
                           parts=[{"chunk_id": c["id"], "text": c["text"]} for c in desc],
                           text="\n\n".join(c["text"] for c in desc))
                return out

        if ranked:
            ranked.sort(key=lambda p: (p[0], p[1].get("chunk_index") or 0))
            how = ["exact", "prefix", "contains", "contained-by"][ranked[0][0]]
            ch = ranked[0][1]
            # a section can be split across chunks (a long table, say). Showing only
            # the first would look like the policy stops mid-table, so return all
            # chunks of the section that matched, in document order.
            same = [c for c in chunks
                    if _flat(c.get("section")) == _flat(ch.get("section"))]
            same.sort(key=lambda c: c.get("chunk_index") or 0)
            out.update(found=True, chunk_id=ch["id"], section=ch.get("section"),
                       breadcrumb=ch.get("breadcrumb"), how=how,
                       parts=[{"chunk_id": c["id"], "text": c["text"]} for c in same],
                       text="\n\n".join(c["text"] for c in same))
            return out

        # ---- no section match: never invent a passage, name what exists
        seen: List[str] = []
        for ch in chunks:
            s = (ch.get("section") or "").strip()
            if s and s not in seen:
                seen.append(s)
        out["reason"] = "no-section"
        out["candidates"] = seen[:14]
        return out
