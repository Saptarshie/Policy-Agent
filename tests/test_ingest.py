import tempfile
import unittest
from pathlib import Path

from app.contradictions.gate import CascadeGate
from app.contradictions.tracker import ContradictionTracker
from app.contradictions.verify import ContradictionVerifier
from app.db import Database
from app.ingest import IngestPipeline
from app.llm import LLMResponse
from app.rag import RAGEngine
from tests.fakes import FakeEmbedder, FakeLLM, StubRemote

DOC = (
    "ACME Remote Work Policy\n\n"
    "1. Allowance\n\n"
    "Employees may work remotely for up to 3 days per calendar week. "
    "The remaining working days must be spent at the assigned office location.\n\n"
    "2. Notice\n\n"
    "Employees must give at least 15 days written notice before changing an "
    "approved remote work arrangement.\n\n"
    "3. Eligibility\n\n "
    "Employees are eligible after 6 months of continuous employment.\n\n"
    "4. Core hours\n\n "
    "Remote employees must be reachable between 10:00 and 16:00 local time.\n\n"
    "5. Security\n\n "
    "Employees must connect through the corporate VPN when working remotely. "
    "Personal printers may not be used for company documents.\n"
)

WINDOW_JSON = (
    '{"window_title": "Allowance", "statements": ["may work remotely 3 days"], '
    '"key_terms": ["remote work", "manager approval"], '
    '"numbers": [{"value": "3", "unit": "days", "context": "remote work"}], '
    '"rolling_summary": "ACME remote work policy: 3 remote days/week allowed '
    'with approval; 15 days notice; eligible after 6 months."}'
)


class TestIngestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.db = Database(self.dir / "t.db")
        self.embedder = FakeEmbedder()
        self.path = self.dir / "policy.txt"
        self.path.write_text(DOC, encoding="utf-8")

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def _pipeline(self, llm) -> IngestPipeline:
        return IngestPipeline(self.db, self.embedder, llm=llm,
                              window_chars=900, use_llm=llm is not None,
                              chunk_target=300, chunk_max=450)

    def test_ingest_with_llm(self):
        fake = FakeLLM([LLMResponse(content=WINDOW_JSON) for _ in range(20)])
        events = []
        result = self._pipeline(fake).ingest(
            self.path, "rw", on_progress=lambda p: events.append(p))

        doc = self.db.get_document(result["doc_id"])
        self.assertEqual(doc["status"], "ready")
        self.assertIn("3 remote days", doc["summary"])
        meta = __import__("json").loads(doc["meta_json"])
        self.assertIn("may work remotely 3 days", meta["statements"])
        self.assertGreater(result["chunks"], 1)
        self.assertEqual(self.db.count_chunks(result["doc_id"]), result["chunks"])
        job = self.db.get_job(result["job_id"])
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["done"], job["total"])
        self.assertGreaterEqual(len(events), job["total"] + 1)  # start + per-window + done
        # progress events are monotonic
        dones = [e["done"] for e in events]
        self.assertEqual(dones, sorted(dones))
        self.assertEqual(events[-1]["done"], events[-1]["total"])

    def test_ingest_heuristic_fallback_on_bad_json(self):
        # garbage from the LLM (twice, incl. the retry) -> heuristic fallback,
        # warning recorded, ingestion still completes
        fake = FakeLLM([LLMResponse(content="sorry, not json"),
                        LLMResponse(content="still not json")])
        result = self._pipeline(fake).ingest(self.path, "rw2")
        self.assertTrue(any("heuristic fallback" in w for w in result["warnings"]))
        doc = self.db.get_document(result["doc_id"])
        self.assertEqual(doc["status"], "ready")
        self.assertGreater(self.db.count_chunks(result["doc_id"]), 0)

    def test_ingest_without_llm(self):
        result = self._pipeline(None).ingest(self.path, "rw3")
        self.assertEqual(result["warnings"], [])
        self.assertEqual(self.db.get_document(result["doc_id"])["status"], "ready")


class TestTrackerCascade(unittest.TestCase):
    """End-to-end cascade test: RAG -> JEV gate -> (fake) LLM verifier -> flag."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "c.db")
        emb = FakeEmbedder()
        self.rag = RAGEngine(self.db, emb)
        # doc A (v1) and doc B (v2) with genuinely conflicting passages
        self.docs = {}
        for name, section, text in (
            ("remote_v1", "1. Allowance",
             "Employees may work remotely for up to 3 days per calendar week. "
             "The remaining working days must be spent at the office. Notice: "
             "employees must give 15 days written notice before changing the "
             "arrangement."),
            ("remote_v2", "2. Attendance",
             "Employees must work from the office at least 4 days per calendar "
             "week. Remote work is limited to 1 day per week. Changes require "
             "30 days advance notice through the HR portal."),
        ):
            did = self.db.add_document(name, f"{name}.txt", "txt", 10, "ready")
            vec = emb.embed_documents([text])[0]
            self.db.add_chunk(did, 0, section, f"{name} > {section}", text, vec)
            self.docs[name] = did

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def _tracker(self, fake_llm, gate_score: float = 0.9,
                 cross_doc_only: bool = True) -> ContradictionTracker:
        # the decision model is the only judge of a candidate pair now; a lexical
        # gate would be exactly the thing this design removed
        return ContradictionTracker(
            db=self.db, rag=self.rag,
            gate=CascadeGate(threshold=0.5, remote=StubRemote(default=gate_score)),
            verifier=ContradictionVerifier(fake_llm),
            cross_doc_only=cross_doc_only,
            min_similarity=0.0, max_candidates=100)

    def test_cascade_flags_verified_contradiction(self):
        verdict = ('{"is_contradiction": true, "confidence": 0.9, '
                   '"conflict_type": "direct", '
                   '"explanation": "3 remote days vs 1 remote day max.", '
                   '"conflicting_points": [{"topic": "remote days", '
                   '"claim_a": "3 days", "claim_b": "1 day"}]}')
        fake = FakeLLM([LLMResponse(content=verdict) for _ in range(10)])
        result = self._tracker(fake).scan_document(self.docs["remote_v2"])

        self.assertGreaterEqual(result["flagged"], 1)
        flags = self.db.list_contradictions("OPEN")
        self.assertTrue(flags)
        f = flags[0]
        self.assertTrue(f["verified"])
        self.assertEqual(f["conflict_type"], "direct")
        self.assertIn("3 remote days", f["explanation"])
        self.assertEqual(f["points"][0]["topic"], "remote days")
        # citations intact
        self.assertEqual({f["doc_a"], f["doc_b"]}, {"remote_v1", "remote_v2"})
        self.assertTrue(f["section_a"] and f["section_b"])
        self.assertEqual(f["status"], "OPEN")

    def test_gate_blocks_llm_when_no_conflict(self):
        # doc v1 only (no conflicting partner) -> gate never passes ->
        # the verifier LLM must never be called
        self.db.delete_document(self.docs["remote_v2"])
        fake = FakeLLM([])  # empty script: any call would raise
        result = self._tracker(fake).scan_document(self.docs["remote_v1"])
        self.assertEqual(result["flagged"], 0)
        self.assertEqual(fake.received, [])

    def test_unverified_strong_gate_records_flag(self):
        # verifier returns garbage twice -> chat_json fails -> unverified flag
        fake = FakeLLM([LLMResponse(content="garbage"),
                        LLMResponse(content="more garbage")])
        result = self._tracker(fake).scan_document(self.docs["remote_v2"])
        self.assertGreaterEqual(result["flagged"], 1)
        f = self.db.list_contradictions("OPEN")[0]
        self.assertFalse(f["verified"])
        self.assertIn("UNVERIFIED", f["explanation"])
        self.assertTrue(f["gate_signals"])

    # ---------------------------------------------------- dedupe regressions

    _VERDICT = ('{"is_contradiction": true, "confidence": 0.9, '
                '"conflict_type": "direct", '
                '"explanation": "3 remote days vs 1 remote day max.", '
                '"conflicting_points": []}')

    def test_rescanning_in_both_directions_flags_once(self):
        """Upload scans v2 (higher chunk ids), a later `run.py scan` scans v1
        and presents the same pair in the opposite order: one flag, one
        verifier call."""
        fake = FakeLLM([LLMResponse(content=self._VERDICT) for _ in range(10)])
        tracker = self._tracker(fake)

        first = tracker.scan_document(self.docs["remote_v2"])
        second = tracker.scan_document(self.docs["remote_v1"])

        self.assertEqual(first["flagged"], 1)
        self.assertEqual(second["flagged"], 0)
        self.assertEqual(len(self.db.list_contradictions()), 1)
        self.assertEqual(len(fake.received), 1, "verifier must run once per pair")

    def test_dismissed_flag_stays_dismissed_after_rescan(self):
        fake = FakeLLM([LLMResponse(content=self._VERDICT) for _ in range(10)])
        tracker = self._tracker(fake)
        tracker.scan_document(self.docs["remote_v2"])
        flag = self.db.list_contradictions()[0]
        self.assertTrue(self.db.set_contradiction_status(flag["id"], "dismissed"))

        tracker.scan_document(self.docs["remote_v1"])
        tracker.scan_document(self.docs["remote_v2"])

        self.assertEqual(self.db.list_contradictions("OPEN"), [])
        self.assertEqual(len(self.db.list_contradictions()), 1)


class TestVerifier(unittest.TestCase):
    def test_parses_valid_verdict(self):
        fake = FakeLLM([LLMResponse(content=(
            '{"is_contradiction": true, "confidence": 0.85, '
            '"conflict_type": "partial", "explanation": "different limits.", '
            '"conflicting_points": [{"topic": "x", "claim_a": "a", "claim_b": "b"}]}'))])
        v = ContradictionVerifier(fake).verify("dA", "s1", "text a", "dB", "s2",
                                               "text b", ["numeric conflict"])
        self.assertTrue(v.verified)
        self.assertTrue(v.is_contradiction)
        self.assertEqual(v.confidence, 0.85)
        self.assertEqual(v.points[0]["topic"], "x")

    def test_invalid_json_returns_unverified(self):
        fake = FakeLLM([LLMResponse(content="nope"),
                        LLMResponse(content="still nope")])
        v = ContradictionVerifier(fake).verify("dA", "s1", "a", "dB", "s2", "b")
        self.assertFalse(v.verified)
        self.assertTrue(v.error)

    def test_no_llm_configured(self):
        v = ContradictionVerifier(None).verify("a", "s", "a", "b", "s", "b")
        self.assertFalse(v.verified)


if __name__ == "__main__":
    unittest.main()
