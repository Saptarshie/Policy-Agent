"""Threshold-gated retrieval and intra-document contradiction scanning.

Three contracts:

1. ``search(min_score=...)`` returns EVERYTHING above the floor, and the floor
   decides - not the count. A conflict can rank 41st among 60 similar passages.
2. The search tool exposes ``top_k`` and ``min_similarity`` but never both.
3. The scanner looks inside a document as well as across documents, and its
   candidate set is similarity-gated (the old per-chunk top-k cap is gone).
"""
import tempfile
import unittest
from pathlib import Path

from app.agent.tools import ToolRegistry
from app.contradictions.gate import CascadeGate
from app.contradictions.tracker import ContradictionTracker
from app.contradictions.verify import ContradictionVerifier
from app.db import Database
from app.llm import LLMResponse
from app.rag import RAGEngine
from tests.fakes import FakeEmbedder, FakeLLM, StubRemote

EMB = FakeEmbedder()

# the same subject, deliberately spread over enough chunks that a top-k cut
# would hide the last one
SUBJECT = [
    "Casual leave accrues at 1 day per completed month of service.",
    "Casual leave may be carried forward up to 6 days into the next year.",
    "Any unused casual leave above 6 days lapses on 31 December.",
    "Carry-forward balance is capped at 6 days for all bands.",
    "Employees joining mid-year receive a pro-rated entitlement.",
    "Pro-rating uses completed months, not calendar days.",
    "Leave requests must be filed through the HR portal.",
    "Sick leave is separate from casual leave and does not lapse.",
    "Unused casual leave may be encashed at the manager's discretion.",
    "From 1 July 2025 the carry-forward cap becomes 15 days.",
]


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "t.db")
        self.emb = FakeEmbedder()
        self.rag = RAGEngine(self.db, self.emb)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def add_doc(self, name, texts):
        did = self.db.add_document(name, f"{name}.md", "md", status="ready")
        for i, text in enumerate(texts):
            self.db.add_chunk(did, i, f"S{i}. Section", name, text,
                              self.emb.embed_query(text))
        return did


class TestThresholdSearch(_Base):
    def test_threshold_returns_the_tail_that_top_k_would_cut(self):
        did = self.add_doc("leave", SUBJECT)
        query = SUBJECT[1]                     # "carried forward up to 6 days"
        top2 = self.rag.search(query, top_k=2, mode="vector")
        floor = self.rag.search(query, min_score=0.10, mode="vector",
                                max_candidates=100)
        self.assertEqual(len(top2), 2)
        self.assertGreater(len(floor), len(top2),
                           "a similarity floor must return more than a top-2 cut")
        self.assertTrue(all(h.score >= 0.10 for h in floor))
        self.assertIn(did, {h.doc_id for h in floor})

    def test_threshold_forces_vector_mode(self):
        self.add_doc("leave", SUBJECT)
        # an RRF score is not a similarity; the engine must not compare it to one
        hits = self.rag.search(SUBJECT[1], min_score=0.10, mode="hybrid",
                              max_candidates=50)
        # a real cosine, not an RRF artefact (~0.03): the floor is compared to a
        # similarity, so the engine must have used the vector arm
        self.assertTrue(hits)
        self.assertGreater(hits[0].score, 0.1)
        self.assertTrue(all(h.keyword_rank is None for h in hits))

    def test_cap_truncates_but_keeps_the_best(self):
        self.add_doc("leave", SUBJECT)
        hits = self.rag.search(SUBJECT[1], min_score=0.0, max_candidates=3,
                              mode="vector")
        self.assertEqual(len(hits), 3)
        scores = [h.score for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestSearchTool(_Base):
    def setUp(self):
        super().setUp()
        self.add_doc("leave", SUBJECT)
        self.registry = ToolRegistry(self.db, self.rag, default_top_k=3)

    def test_default_is_top_k(self):
        out = self.registry.execute("search_policy_docs", {"query": SUBJECT[1]})
        self.assertIn("top_k=3", out)

    def test_top_k_is_honoured(self):
        out = self.registry.execute("search_policy_docs",
                                    {"query": SUBJECT[1], "top_k": 1})
        self.assertIn("top_k=1", out)
        self.assertEqual(out.count("[chunk "), 1)

    def test_min_similarity_is_honoured(self):
        out = self.registry.execute("search_policy_docs",
                                    {"query": SUBJECT[1], "min_similarity": 0.05})
        self.assertIn("min_similarity=0.05", out)
        self.assertGreater(out.count("[chunk "), 1)

    def test_both_at_once_is_an_error_not_a_silent_choice(self):
        out = self.registry.execute("search_policy_docs",
                                    {"query": SUBJECT[1], "top_k": 2,
                                     "min_similarity": 0.5})
        self.assertIn("mutually exclusive", out)

    def test_out_of_range_threshold_is_rejected(self):
        for bad in (0, -0.2, 1.5):
            out = self.registry.execute("search_policy_docs",
                                        {"query": SUBJECT[1], "min_similarity": bad})
            self.assertIn("ERROR", out, bad)


class TestIntraDocumentScan(_Base):
    def _tracker(self, verdicts, gate_score=0.9, cross_doc_only=False):
        fake = FakeLLM([LLMResponse(content=v) for v in verdicts] * 3)
        return ContradictionTracker(
            db=self.db, rag=self.rag,
            gate=CascadeGate(threshold=0.5, remote=StubRemote(default=gate_score)),
            verifier=ContradictionVerifier(fake),
            cross_doc_only=cross_doc_only,
            min_similarity=0.10,      # the fixtures' short texts are all similar
            max_candidates=100)

    CONFLICT = ('{"is_contradiction": true, "confidence": 0.9, "conflict_type": '
                '"direct", "explanation": "6 days vs 15 days", '
                '"conflicting_points": []}')

    def test_same_document_pairs_are_scanned_by_default(self):
        did = self.add_doc("leave", SUBJECT)
        tracker = self._tracker([self.CONFLICT])
        result = tracker.scan_document(did)
        self.assertGreaterEqual(result["flagged"], 1,
                               "an intra-document conflict must be found")

    def test_cross_doc_only_still_supported(self):
        did = self.add_doc("leave", SUBJECT)
        tracker = self._tracker([self.CONFLICT], cross_doc_only=True)
        self.assertEqual(tracker.scan_document(did)["flagged"], 0)

    def test_candidate_cap_is_a_safety_valve_and_logged(self):
        did = self.add_doc("leave", SUBJECT)
        lines = []
        tracker = self._tracker([self.CONFLICT])
        tracker.max_candidates = 1
        tracker._scan(did, self.db.create_job("scan", did, "leave", total=1),
                      log_fn=lambda *a, **k: lines.append(a[0] % a[1:] if len(a) > 1 else a[0]))
        self.assertTrue(any("candidate cap reached" in ln for ln in lines))


class TestNoDecisionModelMeansNoScan(_Base):
    def test_scan_flags_nothing_and_says_why(self):
        did = self.add_doc("leave", SUBJECT)
        lines = []
        tracker = ContradictionTracker(
            db=self.db, rag=self.rag,
            gate=CascadeGate(threshold=0.5, remote=None),
            verifier=ContradictionVerifier(FakeLLM([])),
            cross_doc_only=False, min_similarity=0.10, max_candidates=50)
        result = tracker._scan(did, self.db.create_job("scan", did, "leave", total=1),
                               log_fn=lambda *a, **k: lines.append(a[0] % a[1:] if len(a) > 1 else a[0]))
        self.assertEqual(result["flagged"], 0)
        # and the run says the judge was missing rather than "no conflicts"
        joined = "\n".join(lines).lower()
        self.assertIn("no decision model", joined)
        self.assertIn("not examined", joined)


if __name__ == "__main__":
    unittest.main()
