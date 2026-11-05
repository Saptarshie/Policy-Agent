"""Policy RAG Agent.

Citation-grounded question answering over admin-managed policy documents,
with a windowed ingestion pipeline and a staged background
contradiction tracker (RAG retrieval -> JEV-style heuristic decision gate
-> LLM verifier -> admin alert with citations).

Package layout:
    app/agent/         ReAct loop, tools, intent gate, faithfulness verifier
    app/contradictions/ the staged contradiction cascade
    app/api/           FastAPI routes (web UI + JSON API)
    app/core modules   config, llm, embeddings, db, chunker, ingest, rag, ...
"""

__version__ = "1.1.0"
