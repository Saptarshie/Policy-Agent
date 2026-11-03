"""The judge must be fed the evidence the answer actually cites."""
import tempfile
import unittest
from pathlib import Path

from app.agent.react import AgentResult
from app.agent.verify import AnswerVerifier
from app.db import Database
from app.prompts import FAITHFULNESS_SYSTEM
from tests.fakes import StubLLM


class TestJudgeContextSelection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "t.db")
        self.doc = self.db.add_document("remote_policy", "r.md", "md", 10, "ready")
        self.ids = []
        for i in range(12):
            cid = self.db.add_chunk(
                self.doc, i, "S%d" % i, "remote_policy > S%d" % i,
                "passage %d about remote work and leave" % i, [0.1] * 4)
            self.ids.append(cid)
        self.captured = {}

        def chat_json(system, user):
            self.captured["system"] = system
            self.captured["user"] = user
            return {"faithful": True, "unsupported_claims": [], "reason": "ok"}, None

        self.verifier = AnswerVerifier(self.db, StubLLM(chat_json_fn=chat_json),
                                       enabled=True, system_prompt=FAITHFULNESS_SYSTEM)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def _result(self, citations):
        r = AgentResult(answer="Remote work is capped at 3 days [remote_policy § S11].")
        r.seen_chunk_ids = list(self.ids)          # 12 chunks, the cited one last
        r.citations = citations
        return r

    def test_cited_chunk_is_put_first(self):
        r = self._result(["remote_policy § S11"])
        ids = self.verifier._judge_context_ids(r)
        self.assertEqual(ids[0], self.ids[11])

    def test_cited_chunk_from_late_in_the_run_reaches_the_judge(self):
        r = self._result(["remote_policy § S11"])
        self.verifier.check(r, "how many remote days?")
        self.assertIn("passage 11 about remote work", self.captured["user"])

    def test_all_seen_chunks_fit_in_default_judge_budget(self):
        r = self._result([])
        self.verifier.check(r, "how many remote days?")
        # nothing is dropped for a normal 12-chunk run, so no coverage note
        self.assertNotIn("NOTE ON COVERAGE", self.captured["user"])
        for i in range(12):
            self.assertIn("passage %d about" % i, self.captured["user"])

    def test_truncated_context_adds_a_coverage_note(self):
        # 400 long chunks cannot fit the char budget -> note tells the judge
        # not to treat omissions as fabrication
        for i in range(30):
            self.db.add_chunk(self.doc, 100 + i, "L%d" % i, "remote_policy > L%d" % i,
                              "x" * 1500, [0.1] * 4)
        r = AgentResult(answer="Remote work is capped at 3 days.")
        r.seen_chunk_ids = [c["id"] for c in self.db.list_chunks(self.doc)]
        self.verifier.check(r, "how many remote days?")
        self.assertIn("NOTE ON COVERAGE", self.captured["user"])

    def test_document_inventory_is_included(self):
        """Claims about WHICH documents exist must be checkable by the judge."""
        self.db.add_document("expense_policy", "e.md", "md", 10, "ready")
        r = self._result([])
        self.verifier.check(r, "how many remote days?")
        self.assertIn("DOCUMENT INVENTORY", self.captured["user"])
        self.assertIn("remote_policy", self.captured["user"])
        self.assertIn("expense_policy", self.captured["user"])

    def test_unresolvable_citation_does_not_crash(self):
        r = self._result(["ghost_doc § 9"])
        ids = self.verifier._judge_context_ids(r)
        self.assertEqual(ids, list(self.ids))     # falls back to seen order


if __name__ == "__main__":
    unittest.main()
