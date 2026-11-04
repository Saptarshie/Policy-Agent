"""FastAPI routes: admin document management + chat + contradiction flags.

``create_app(api=None)`` builds a fresh app; tests inject an ``AdminAPI``
wired with fakes, production uses the real one (see ``app.main``).
"""

from __future__ import annotations

import json
import queue
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from ..admin import AdminAPI

_ALLOWED_EXT = {".txt", ".text", ".md", ".markdown", ".pdf"}
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class AskRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []


class StatusRequest(BaseModel):
    status: str


def create_app(api: Optional[AdminAPI] = None) -> FastAPI:
    app = FastAPI(title="Policy RAG Agent")
    _api = api or AdminAPI()

    # ------------------------------------------------------------------- ui

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return (_WEB_DIR / "index.html").read_text(encoding="utf-8")

    # ------------------------------------------------------------- health

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return _api.health()

    # ----------------------------------------------------------- documents

    @app.post("/api/documents")
    async def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:
        filename = file.filename or "upload.txt"
        suffix = Path(filename).suffix.lower()
        if suffix not in _ALLOWED_EXT:
            raise HTTPException(400, f"unsupported file type '{suffix}' "
                                     f"(allowed: txt, md, pdf)")
        uploads_dir = _api.cfg.data_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        target = uploads_dir / f"{uuid.uuid4().hex[:12]}{suffix}"
        # Stream instead of file.read(): an unbounded read buffers the whole
        # upload in memory, so one large PDF can take the process down before
        # validation ever runs.
        limit = max(1, int(_api.cfg.max_upload_mb)) * 1024 * 1024
        written = 0
        try:
            with open(target, "wb") as fh:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > limit:
                        raise HTTPException(
                            413, f"file exceeds the {_api.cfg.max_upload_mb} MB limit")
                    fh.write(chunk)
        except HTTPException:
            target.unlink(missing_ok=True)
            raise
        if not written:
            target.unlink(missing_ok=True)
            raise HTTPException(400, "empty file")
        return _api.upload_document_async(target, name=Path(filename).stem)

    @app.get("/api/documents")
    def list_documents() -> List[Dict[str, Any]]:
        return _api.list_documents()

    @app.delete("/api/documents/{doc_id}")
    def delete_document(doc_id: int) -> Dict[str, Any]:
        if not _api.delete_document(doc_id):
            raise HTTPException(404, f"document {doc_id} not found")
        return {"ok": True}

    # --------------------------------------------------------------- jobs

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: int) -> Dict[str, Any]:
        job = _api.job_status(job_id)
        if not job:
            raise HTTPException(404, f"job {job_id} not found")
        return job

    # ---------------------------------------------------------------- ask

    @app.post("/api/ask")
    def ask(req: AskRequest) -> Dict[str, Any]:
        if not req.question.strip():
            raise HTTPException(400, "empty question")
        try:
            result = _api.ask(req.question, history=req.history)
        except Exception as e:
            raise HTTPException(503, f"agent failed: {e}")
        return result.to_dict()

    @app.post("/api/ask/stream")
    def ask_stream(req: AskRequest) -> StreamingResponse:
        """Same pipeline as /api/ask, streamed as Server-Sent Events.

        Synchronous on purpose: the agent is a blocking tool-call loop, so it
        runs on a worker thread and its events are queued for this generator.
        A client that cannot read streams keeps working with /api/ask.
        """
        if not req.question.strip():
            raise HTTPException(400, "empty question")

        def frames():
            events: "queue.Queue" = queue.Queue()

            def on_event(kind: str, payload: dict) -> None:
                events.put((kind, payload))

            def work() -> None:
                try:
                    result = _api.ask(req.question, history=req.history,
                                      on_event=on_event)
                    events.put(("final", result.to_dict()))
                except Exception as e:               # noqa: BLE001 - reported, not raised
                    events.put(("error", {"error": str(e)}))
                finally:
                    events.put(("done", {}))

            threading.Thread(target=work, daemon=True, name="ask-stream").start()
            # an immediate comment opens the stream so the browser shows it live
            yield ": stream open\n\n"
            while True:
                try:
                    kind, payload = events.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"        # keeps proxies/UI from timing out
                    continue
                if kind == "done":
                    break
                yield f"event: {kind}\ndata: {json.dumps(payload, default=str)}\n\n"

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                     "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------ contradictions

    @app.get("/api/contradictions")
    def contradictions(status: Optional[str] = None) -> List[Dict[str, Any]]:
        return _api.list_contradictions(status)

    @app.post("/api/contradictions/{cid}/status")
    def set_contradiction_status(cid: int, req: StatusRequest) -> Dict[str, Any]:
        if req.status not in ("OPEN", "resolved", "dismissed"):
            raise HTTPException(400, "status must be OPEN|resolved|dismissed")
        if not _api.set_contradiction_status(cid, req.status):
            raise HTTPException(404, f"contradiction {cid} not found")
        return {"ok": True}

    # -------------------------------------------------------------- chunks

    @app.get("/api/citations/resolve")
    def resolve_citation(cite: str) -> Dict[str, Any]:
        """What the UI calls when a citation chip is clicked."""
        return _api.resolve_citation(cite)

    @app.get("/api/chunks/{chunk_id}")
    def get_chunk(chunk_id: int) -> Dict[str, Any]:
        chunk = _api.get_chunk(chunk_id)
        if not chunk:
            raise HTTPException(404, f"chunk {chunk_id} not found")
        return chunk

    @app.post("/api/scan/{doc_id}")
    def scan_doc(doc_id: int) -> Dict[str, Any]:
        job_id = _api.tracker.scan_document_async(doc_id)
        if job_id is None:
            raise HTTPException(400, f"document {doc_id} is not ready for scanning")
        return {"job_id": job_id}

    return app
