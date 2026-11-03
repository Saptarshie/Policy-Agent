"""File readers for policy documents (txt / md / pdf) + text normalisation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple


class ReaderError(RuntimeError):
    """Raised when a file cannot be read or parsed."""


_SUPPORTED = {
    ".txt": "txt",
    ".text": "txt",
    ".md": "md",
    ".markdown": "md",
    ".pdf": "pdf",
}


def read_document(path: Path) -> Tuple[str, str]:
    """Read *path* and return ``(clean_text, format)``."""
    path = Path(path)
    if not path.exists():
        raise ReaderError(f"file not found: {path}")
    ext = path.suffix.lower()
    fmt = _SUPPORTED.get(ext)
    if not fmt:
        raise ReaderError(
            f"unsupported file type '{ext or path.name}' (allowed: txt, md, pdf)"
        )
    if fmt == "pdf":
        raw = _read_pdf(path)
    else:
        raw = _read_text_any(path)
        if fmt == "md":
            raw = _clean_markdown(raw)
    text = clean_text(raw)
    if not text.strip():
        raise ReaderError(f"document appears to be empty: {path.name}")
    return text, fmt


def _read_text_any(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ReaderError(f"cannot decode text file: {path.name}")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ReaderError(
            "pypdf is required for PDF files (pip install pypdf)"
        ) from e
    try:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    except Exception as e:
        raise ReaderError(f"failed to parse PDF {path.name}: {e}") from e
    if not any(p.strip() for p in pages):
        raise ReaderError(
            f"PDF {path.name} contains no extractable text (scanned image PDF?)"
        )
    return "\n\n".join(pages)


def _clean_markdown(text: str) -> str:
    """Strip inline markdown decoration while keeping heading lines intact
    (headings are detected later by the chunker)."""
    text = re.sub(r"```[A-Za-z0-9_+\-]*\n?", "", text)          # fence markers
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)        # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)         # links
    text = re.sub(r"(\*\*|__|\*|~~)", "", text)                  # bold/italic/strike
    return text


def clean_text(text: str) -> str:
    """Normalise line endings, strip trailing spaces, collapse blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
