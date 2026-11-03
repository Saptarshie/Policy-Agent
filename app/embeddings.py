"""Ollama embedding client for embeddinggemma.

Uses the modern ``POST /api/embed`` endpoint (batched), falling back to the
legacy single-input ``POST /api/embeddings`` on older Ollama versions.

EmbeddingGemma is task-steered and Ollama does NOT add prompt prefixes on
your behalf, so query and document prefixes must be prepended manually
(see .env: EMBED_QUERY_PREFIX / EMBED_DOC_PREFIX).
"""

from __future__ import annotations

import logging
from typing import List

import requests

log = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced."""


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str = "embeddinggemma",
        query_prefix: str = "task: search result | query: ",
        doc_prefix: str = "title: none | text: ",
        batch_size: int = 32,
        timeout: float = 300.0,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.query_prefix = query_prefix or ""
        self.doc_prefix = doc_prefix or ""
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.dim: int = 0

    # ------------------------------------------------------------------ api

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed passages for indexing (uses the document prefix)."""
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [self.doc_prefix + t for t in texts[i:i + self.batch_size]]
            out.extend(self._embed_raw(batch))
        return out

    def embed_query(self, text: str) -> List[float]:
        """Embed a search query (uses the query prefix)."""
        return self._embed_raw([self.query_prefix + text])[0]

    def healthcheck(self) -> bool:
        try:
            vec = self._embed_raw(["healthcheck"])
            return bool(vec and vec[0])
        except Exception:  # pragma: no cover - requires a live Ollama
            return False

    # ----------------------------------------------------------------- http

    def _embed_raw(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            r = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise EmbeddingError(
                f"cannot reach Ollama at {self.base_url} (is it running? "
                f"did you 'ollama pull {self.model}'?): {e}"
            ) from e
        if r.status_code == 404:
            return self._embed_raw_legacy(texts)
        if r.status_code != 200:
            raise EmbeddingError(f"Ollama /api/embed error {r.status_code}: {r.text[:300]}")
        try:
            embs = r.json().get("embeddings")
        except ValueError as e:
            raise EmbeddingError(f"invalid JSON from Ollama: {e}") from e
        if not embs or len(embs) != len(texts):
            raise EmbeddingError(
                f"Ollama returned {len(embs or [])} embeddings for {len(texts)} inputs"
            )
        self.dim = len(embs[0])
        return [list(map(float, e)) for e in embs]

    def _embed_raw_legacy(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            try:
                r = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": t},
                    timeout=self.timeout,
                )
            except requests.RequestException as e:
                raise EmbeddingError(f"cannot reach Ollama at {self.base_url}: {e}") from e
            if r.status_code != 200:
                raise EmbeddingError(
                    f"Ollama /api/embeddings error {r.status_code}: {r.text[:300]}"
                )
            emb = r.json().get("embedding")
            if not emb:
                raise EmbeddingError("legacy Ollama response has no 'embedding' field")
            out.append(list(map(float, emb)))
        if out:
            self.dim = len(out[0])
        log.debug("used legacy /api/embeddings endpoint (batch=1)")
        return out


class SentenceTransformerEmbedder:
    """Local embeddings via sentence-transformers (free, offline, zero services)."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self._model = None
        self.dim: int = 384

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self.dim = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                raise EmbeddingError(
                    f"failed to load sentence-transformers model '{self.model_name}': {e}"
                ) from e
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [list(map(float, v)) for v in vecs]

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.dim
        vec = self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return list(map(float, vec))

    def healthcheck(self) -> bool:
        try:
            return bool(self.embed_query("healthcheck"))
        except Exception:
            return False
