"""ASGI entry point: ``uvicorn app.main:app``.

Builds the production FastAPI app (real AdminAPI: Ollama embeddinggemma +
OpenAI-compatible LLM from ``.env``).  Tests build their own app with
``app.api.routes.create_app(api=...)``.
"""

from __future__ import annotations

from .api.routes import create_app

app = create_app()
