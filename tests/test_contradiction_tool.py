"""The agent's view of the tracker: list_contradictions."""

import tempfile
import unittest
from pathlib import Path

from app.agent.tools import TOOL_SPECS, ToolRegistry
from app.db import Database
from app.rag import RAGEngine
from tests.fakes import FakeEmbedder


class TestListContradictions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "t.db")
        self.registry = ToolRegistry(self.db, RAGEngine(self.db, FakeEmbedder()))
        a = self.db.add_document("remote_v1", "a.md", "md", status="ready")
        b = self.db.add_document("remote_v2", "b.md", "md", status="ready")
        emb = FakeEmbedder()
        ta = "Remote work up to 3 days per week."
        tb = "Remote work is limited to 1 day per week."
        self.a = self.db.add_chunk(a, 0, "2. Allowance", "remote_v1", ta,
                                   emb.embed_query(ta))
        self.b = self.db.add_chunk(b, 0, "2. Attendance", "remote_v2", tb,
                                   emb.embed_query(tb))
        self.cid = self.db.add_contradiction(
            chunk_a_id=self.a, doc_a="remote_v1", section_a="2. Allowance",
            text_a="Remote work up to 3 days per week.",
            chunk_b_id=self.b, doc_b="remote_v2", section_b="2. Attendance",
            text_b="Remote work is limited to 1 day per week.",
            gate_score=0.94, gate_signals=["both use 'per week'"],
            verified=True, confidence=0.93, conflict_type="direct",
            explanation="3 days vs 1 day.", points=[])

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_tool_is_declared(self):
        names = [t["function"]["name"] for t in TOOL_SPECS]
        self.assertIn("list_contradictions", names)

    def test_open_flags_include_both_sides_and_the_reason(self):
        out = self.registry.execute("list_contradictions", {})
        self.assertIn(f"Flag {self.cid}", out)
        self.assertIn("remote_v1", out)
        self.assertIn("remote_v2", out)
        self.assertIn("3 days vs 1 day.", out)
        self.assertIn("VERIFIED", out)
        # the flag may not be cited as a source: the instruction must say so
        self.assertIn("Cite the PASSAGES, not the flag", out)
        self.assertIn("never as [contradiction", out)

    def test_empty_result_says_none_found_not_clean(self):
        self.db.set_contradiction_status(self.cid, "resolved")
        out = self.registry.execute("list_contradictions", {"status": "OPEN"})
        self.assertIn("none found so far", out)
        self.assertIn("not as proof", out)

    def test_status_filter_and_all(self):
        self.db.set_contradiction_status(self.cid, "dismissed")
        self.assertEqual(self.db.list_contradictions("OPEN"), [])
        out = self.registry.execute("list_contradictions", {"status": "all"})
        self.assertIn("dismissed", out)
        self.assertIn(f"Flag {self.cid}", out)

    def test_limit_is_respected_and_reported(self):
        out = self.registry.execute("list_contradictions", {"limit": 1})
        self.assertIn("showing up to 1", out)


if __name__ == "__main__":
    unittest.main()
