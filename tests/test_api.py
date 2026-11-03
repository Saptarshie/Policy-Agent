"""HTTP API integration test: the whole journey through TestClient.

Upload v1 -> upload the contradicting v2 (both ingested on background
threads with fake embedder + stub LLM) -> the cascade flags contradictions
-> ask a question -> grounded answer with citations -> resolve a flag ->
delete a document.  No network access anywhere.
"""
import json
import tempfile
import time
import unittest
from pathlib import Path

from app.admin import AdminAPI
from app.agent.intent import IntentPlan
from app.db import Database
from app.llm import LLMResponse
from tests.fakes import FakeEmbedder, StubIntentRouter, StubLLM
from tests.test_agent_admin import make_config

SAMPLES = Path(__file__).resolve().parent.parent / "sample_docs"
V1 = (SAMPLES / "remote_work_policy_v1.txt").read_text(encoding="utf-8")
V2 = (SAMPLES / "remote_work_policy_v2.md").read_text(encoding="utf-8")

WINDOW_JSON = ('{"window_title": "Allowance", "statements": ["remote days rule"], '
               '"key_terms": ["remote work"], "numbers": [], '
               '"rolling_summary": "remote work rules"}')
VERDICT_JSON = ('{"is_contradiction": true, "confidence": 0.9, '
                '"conflict_type": "direct", '
                '"explanation": "one policy allows 3 remote days, the other '
                'requires 4 office days - incompatible.", '
                '"conflicting_points": [{"topic": "remote days", '
                '"claim_a": "3 days", "claim_b": "1 day"}]}')
ANSWER = ("Employees may work remotely up to 3 days per calendar week "
          "[remote_work_policy_v1 § 2. Remote Work Allowance].")


def _stub_llm() -> StubLLM:
    """One stub serving ingestion, verification and the chat loop."""

    def chat(messages, tools=None):
        return LLMResponse(content=ANSWER)

    def chat_json(system, user):
        if "contradict" in system.lower():
            return json.loads(VERDICT_JSON), None
        return json.loads(WINDOW_JSON), None

    return StubLLM(chat_fn=chat, chat_json_fn=chat_json)


def _wait_for(predicate, timeout=15.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.cfg = make_config(self.dir / "data", verify_answers=False)
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.cfg.db_path)
        self.api = AdminAPI(config=self.cfg, db=self.db,
                            embedder=FakeEmbedder(), llm=_stub_llm(),
                            intent_router=StubIntentRouter(plans={
                                "hello": IntentPlan(intent="GREETING",
                                                    route="DIRECT_REPLY",
                                                    decided_by="stub")}))
        from app.api.routes import create_app
        from fastapi.testclient import TestClient
        self.client = TestClient(create_app(api=self.api))

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    # ------------------------------------------------------------------ helpers

    def _upload(self, name, content):
        r = self.client.post("/api/documents",
                             files={"file": (name, content, "text/plain")})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def _wait_ready(self, doc_name):
        ok = _wait_for(lambda: any(
            d["name"] == doc_name and d["status"] == "ready"
            for d in self.client.get("/api/documents").json()))
        self.assertTrue(ok, f"{doc_name} never became ready")

    # -------------------------------------------------------------------- tests

    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("llm_model", body)
        self.assertIn("jev_gate_threshold", body)

    def test_ui_served(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("<!DOCTYPE html>", r.text)

    def test_rejects_bad_file_type(self):
        r = self.client.post("/api/documents",
                             files={"file": ("x.docx", b"PK", "application/msword")})
        self.assertEqual(r.status_code, 400)
        self.assertIn("unsupported file type", r.text)

    def test_rejects_empty_file(self):
        r = self.client.post("/api/documents",
                             files={"file": ("empty.md", b"", "text/markdown")})
        self.assertEqual(r.status_code, 400)

    def test_rejects_oversized_file(self):
        """A big upload must be refused, not buffered into memory."""
        self.api.cfg = make_config(self.cfg.data_dir, max_upload_mb=1,
                                   verify_answers=False)
        # 1 MB + 1 byte of text
        payload = b"a" * (1024 * 1024 + 1)
        r = self.client.post("/api/documents",
                             files={"file": ("big.md", payload, "text/markdown")})
        self.assertEqual(r.status_code, 413, r.text)
        # nothing half-written is left behind for ingest to pick up
        uploads = list((self.cfg.data_dir / "uploads").glob("*"))
        self.assertEqual(uploads, [])

    def test_full_journey(self):
        # --- upload both versions (contradicting policies)
        self._upload("remote_work_policy_v1.txt", V1)
        self._wait_ready("remote_work_policy_v1")
        self._upload("remote_work_policy_v2.md", V2)
        self._wait_ready("remote_work_policy_v2")

        # --- the background cascade flags the conflict
        doc2 = self.db.get_document("remote_work_policy_v2")
        _wait_for(lambda: (
            (self.db.latest_job_for_doc(doc2["id"], kind="scan") or {}).get("status") == "done"
        ))
        flagged = _wait_for(lambda: self.client.get(
            "/api/contradictions?status=OPEN").json())
        self.assertTrue(flagged, "contradiction scan never flagged anything")
        flags = self.client.get("/api/contradictions?status=OPEN").json()
        f = flags[0]
        self.assertTrue(f["verified"])
        self.assertEqual(f["conflict_type"], "direct")
        self.assertEqual({f["doc_a"], f["doc_b"]},
                         {"remote_work_policy_v1", "remote_work_policy_v2"})
        self.assertTrue(f["explanation"])
        self.assertTrue(f["section_a"] and f["section_b"])   # citations intact

        # --- ask a question -> grounded answer with citations
        r = self.client.post("/api/ask", json={"question": "How many days can I work remotely?"})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn("3 days", body["answer"])
        self.assertTrue(body["citations"])
        self.assertEqual(body["citation_warnings"], [])
        self.assertGreaterEqual(body["tool_calls"], 0)

        # --- resolve every open flag (several sections of v1/v2 conflict:
        #     remote days, notice period, eligibility, core hours ...)
        open_flags = self.client.get("/api/contradictions?status=OPEN").json()
        self.assertTrue(open_flags)
        for f in open_flags:
            r = self.client.post(f"/api/contradictions/{f['id']}/status",
                                 json={"status": "resolved"})
            self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.get("/api/contradictions?status=OPEN").json(), [])
        self.assertEqual(len(self.client.get("/api/contradictions").json()),
                         len(open_flags))

        # --- chunks endpoint (used by the UI to expand citations)
        chunk_id = body["seen_chunk_ids"][0] if body["seen_chunk_ids"] else \
            self.db.list_chunks(self.db.get_document("remote_work_policy_v1")["id"])[0]["id"]
        r = self.client.get(f"/api/chunks/{chunk_id}")
        self.assertEqual(r.status_code, 200)
        self.assertIn("remote", r.json()["text"].lower())

        # --- delete one document
        doc_id = self.db.get_document("remote_work_policy_v2")["id"]
        r = self.client.delete(f"/api/documents/{doc_id}")
        self.assertEqual(r.status_code, 200)
        r = self.client.delete(f"/api/documents/{doc_id}")
        self.assertEqual(r.status_code, 404)
        names = [d["name"] for d in self.client.get("/api/documents").json()]
        self.assertNotIn("remote_work_policy_v2", names)

    def test_ask_empty_question_rejected(self):
        r = self.client.post("/api/ask", json={"question": "   "})
        self.assertEqual(r.status_code, 400)

    def test_intent_gate_over_http(self):
        r = self.client.post("/api/ask", json={"question": "hello"})
        body = r.json()
        self.assertEqual(body["intent"], "GREETING")
        self.assertEqual(body["intent_route"], "DIRECT_REPLY")
        self.assertIn("PolicyPilot", body["answer"])
        self.assertEqual(body["tool_calls"], 0)

    def test_job_status_404(self):
        self.assertEqual(self.client.get("/api/jobs/99999").status_code, 404)


if __name__ == "__main__":
    unittest.main()
