"""Citation recovery: it may only add sources the agent actually read.

Three contracts, in order of importance:

1. **It can never invent a citation.** Candidates come from chunk ids the agent saw; a
   decision model that hallucinates a name in its output cannot get one into the list.
2. **It never removes or reorders what the answer wrote.** Recovery appends.
3. **It is one request per answer** - one shared state, one question per candidate - and a
   decision-model failure leaves the answer exactly as it was.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.admin import AdminAPI
from app.agent.react import AgentResult
from app.citation_recovery import (build_candidates, build_questions, build_state,
                                   citation_for, recover)
from app.db import Database
from tests.fakes import FakeEmbedder

DOC = "03_SERVICE_LEVEL_AND_SCHEDULE"
TEXT = ("Scheduled Maintenance within the published window is excluded from both the "
        "numerator and denominator up to four hours per calendar month. Excess minutes "
        "count as Unavailable if the Covered Service is unavailable during them.")


class CandidateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "c.db")
        emb = FakeEmbedder()
        self.doc_id = self.db.add_document(DOC, DOC + ".md", "md", 10, "ready")
        self.ids = []
        for i, section in enumerate(["3. Excluded Minutes", "9. Planned Maintenance",
                                     "4. Availability Credits"]):
            self.ids.append(self.db.add_chunk(
                self.doc_id, i, section, f"{DOC} > {section}", TEXT + f" ({section})",
                emb.embed_documents([TEXT])[0]))

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_candidates_are_the_sections_the_agent_read_but_did_not_cite(self):
        cands = build_candidates(self.db, self.ids,
                                 [f"{DOC} § 3. Excluded Minutes"])
        sections = [c["section"] for c in cands]
        self.assertNotIn("3. Excluded Minutes", sections, "already cited, so not a candidate")
        self.assertEqual(sections, ["9. Planned Maintenance", "4. Availability Credits"])
        self.assertEqual(cands[0]["citation"], f"{DOC} § 9. Planned Maintenance")
        self.assertEqual(cands[0]["chunk_id"], self.ids[1])

    def test_a_cited_section_is_recognised_through_case_and_separator_noise(self):
        noisy = f"  {DOC.lower()}   #   3. excluded minutes  "
        cands = build_candidates(self.db, self.ids, [noisy])
        self.assertEqual([c["section"] for c in cands],
                         ["9. Planned Maintenance", "4. Availability Credits"])

    def test_one_candidate_per_section_even_if_the_agent_read_two_chunks_of_it(self):
        dup = self.db.add_chunk(self.doc_id, 9, "3. Excluded Minutes",
                                f"{DOC} > 3. Excluded Minutes", TEXT,
                                FakeEmbedder().embed_documents([TEXT])[0])
        cands = build_candidates(self.db, self.ids + [dup], [])
        keys = [c["citation"] for c in cands]
        self.assertEqual(len(keys), len(set(keys)))

    def test_the_candidate_list_is_bounded(self):
        cands = build_candidates(self.db, self.ids, [], max_candidates=2)
        self.assertEqual(len(cands), 2)

    def test_a_tiny_chunk_is_not_a_candidate(self):
        tiny = self.db.add_chunk(self.doc_id, 10, "Empty", f"{DOC} > Empty", "n/a",
                                 FakeEmbedder().embed_documents(["n/a"])[0])
        cands = build_candidates(self.db, [tiny], [])
        self.assertEqual(cands, [])

    def test_an_unknown_chunk_id_is_ignored_not_fatal(self):
        cands = build_candidates(self.db, [999999, "junk"], [])
        self.assertEqual(cands, [])

    def test_citation_for_handles_a_missing_section(self):
        self.assertEqual(citation_for("DOC", ""), "DOC")
        self.assertEqual(citation_for("DOC", " 4. Credits "), "DOC § 4. Credits")


class RecoverTest(unittest.TestCase):
    def setUp(self):
        self.cands = [
            {"citation": f"{DOC} § 9. Planned Maintenance", "text": TEXT, "section": "9"},
            {"citation": f"{DOC} § 4. Availability Credits", "text": TEXT, "section": "4"},
        ]

    def test_only_candidates_at_or_above_the_threshold_are_added(self):
        added, scores = recover(lambda state, qs: {"cite_1": 0.91, "cite_2": 0.05},
                                "Maintenance is excluded for four hours.", self.cands, 0.4)
        self.assertEqual(added, [f"{DOC} § 9. Planned Maintenance"])
        self.assertEqual(scores, {"cite_1": 0.91, "cite_2": 0.05})

    def test_the_state_and_questions_are_one_request_over_every_candidate(self):
        seen = {}

        def ask(state, questions):
            seen["state"] = state
            seen["questions"] = questions
            return {"cite_1": 0.0, "cite_2": 0.0}

        recover(ask, "Maintenance is excluded.", self.cands, 0.4)
        self.assertEqual(len(seen["questions"]), 2)
        self.assertIn("FINAL ANSWER:", seen["state"])
        self.assertIn("CANDIDATE 2 - ", seen["state"])
        self.assertIn("FINAL ANSWER", seen["questions"]["cite_1"])

    def test_a_hallucinated_name_cannot_become_a_citation(self):
        added, _ = recover(lambda s, q: {"cite_1": 0.99, "cite_9": 0.99, "ghost": 0.99},
                           "Answer.", self.cands, 0.4)
        self.assertEqual(added, [f"{DOC} § 9. Planned Maintenance"])

    def test_a_failing_endpoint_returns_nothing_and_does_not_raise(self):
        def boom(state, questions):
            raise RuntimeError("endpoint down")
        self.assertEqual(recover(boom, "Answer.", self.cands, 0.4), ([], {}))

    def test_no_reply_means_no_change(self):
        self.assertEqual(recover(lambda s, q: None, "Answer.", self.cands, 0.4), ([], {}))

    def test_empty_answer_or_no_candidates_is_a_no_op(self):
        self.assertEqual(recover(lambda s, q: {"cite_1": 1.0}, "", self.cands), ([], {}))
        self.assertEqual(recover(lambda s, q: {"cite_1": 1.0}, "Answer.", []), ([], {}))

    def test_the_number_added_is_capped_and_the_best_scores_win(self):
        cands = [{"citation": f"DOC § {i}", "text": TEXT, "section": str(i)} for i in range(1, 6)]
        scores = {"cite_1": 0.41, "cite_2": 0.95, "cite_3": 0.62, "cite_4": 0.99, "cite_5": 0.45}
        added, _ = recover(lambda s, q: scores, "Answer.", cands, 0.4, max_added=2)
        self.assertEqual(added, ["DOC § 2", "DOC § 4"], "top two by score, in reading order")
        added_all, _ = recover(lambda s, q: scores, "Answer.", cands, 0.4, max_added=10)
        self.assertEqual(len(added_all), 5)

    def test_a_junk_score_counts_as_zero(self):
        added, scores = recover(lambda s, q: {"cite_1": "high", "cite_2": None},
                                "Answer.", self.cands, 0.4)
        self.assertEqual(added, [])
        self.assertEqual(scores, {"cite_1": 0.0, "cite_2": 0.0})


class AdminWiringTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "w.db")
        emb = FakeEmbedder()
        did = self.db.add_document(DOC, DOC + ".md", "md", 10, "ready")
        self.cid = self.db.add_chunk(did, 0, "9. Planned Maintenance",
                                     f"{DOC} > 9. Planned Maintenance", TEXT,
                                     emb.embed_documents([TEXT])[0])
        # bypass __init__: recover_citations only needs cfg, db and the asker
        self.api = AdminAPI.__new__(AdminAPI)
        self.api.db = self.db
        self.api.cfg = SimpleNamespace(citation_recovery=True,
                                       citation_recovery_threshold=0.4,
                                       citation_recovery_max_candidates=12,
                                       citation_recovery_snippet_chars=700)
        self.api._citation_asker = lambda state, questions: {"cite_1": 0.88}

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_it_appends_and_records_an_audit_step(self):
        res = AgentResult(answer="Maintenance is excluded for four hours [x].\n\nSources:\n"
                                 f"- [x]", citations=[], seen_chunk_ids=[self.cid])
        out = self.api.recover_citations(res)
        self.assertEqual(out.citations, [f"{DOC} § 9. Planned Maintenance"])
        step = out.steps[-1]
        self.assertEqual(step["type"], "citations_recovered")
        self.assertEqual(step["added"], [f"{DOC} § 9. Planned Maintenance"])
        self.assertEqual(step["candidates"], 1)

    def test_what_the_answer_cited_comes_first_and_is_never_duplicated(self):
        cited = f"{DOC} § 9. Planned Maintenance"
        res = AgentResult(answer="Answer.", citations=[cited] * 2,
                          seen_chunk_ids=[self.cid])
        out = self.api.recover_citations(res)
        self.assertEqual(out.citations, [cited, cited], "no candidates left to add")

    def test_switching_it_off_makes_no_call(self):
        self.api.cfg.citation_recovery = False
        self.api._citation_asker = lambda s, q: (_ for _ in ()).throw(AssertionError("called"))
        res = AgentResult(answer="Answer.", citations=[], seen_chunk_ids=[self.cid])
        self.assertEqual(self.api.recover_citations(res).citations, [])

    def test_no_decision_model_configured_is_a_silent_no_op(self):
        self.api._citation_asker = None
        res = AgentResult(answer="Answer.", citations=[], seen_chunk_ids=[self.cid])
        out = self.api.recover_citations(res)
        self.assertEqual(out.citations, [])
        self.assertEqual(out.steps, [])


if __name__ == "__main__":
    unittest.main()
