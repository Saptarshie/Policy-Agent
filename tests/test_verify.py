import tempfile
import unittest
from pathlib import Path

from app.agent.react import AgentResult
from app.agent.verify import (AnswerVerifier, gather_contexts,
                              quote_matches)
from app.db import Database
from app.llm import LLMResponse
from tests.fakes import FakeEmbedder, FakeLLM, StubLLM

CTX = ("Employees may work remotely for up to 3 days per calendar week. "
       "The remaining working days must be spent at the office.")


class TestQuoteMatching(unittest.TestCase):
    def test_exact_substring(self):
        self.assertTrue(quote_matches("3 days per calendar week", CTX))

    def test_fuzzy_match_survives_minor_edits(self):
        self.assertTrue(quote_matches("employees may work remotly for up to 3 days",
                                      CTX))

    def test_absent_quote_fails(self):
        self.assertFalse(quote_matches("the office must be painted blue", CTX))

    def test_empty_quote_fails(self):
        self.assertFalse(quote_matches("", CTX))


class TestEvidenceGate(unittest.TestCase):
    """The verifier's own deterministic layer: no passages seen -> warn.

    Whether the answer *is* a refusal is no longer decided here by matching
    phrases - that is the decision model's job (tests/test_answer_shape.py).
    """

    def _verifier(self, db):
        return AnswerVerifier(db=db, llm=None, enabled=True)

    def test_no_evidence_and_policy_claim_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "g.db")
            shape = StubShape("answer")
            v = AnswerVerifier(db=db, llm=None, enabled=True,
                               shape_classifier=shape)
            r = AgentResult(answer="You get 20 days of casual leave.")
            v.check(r, "how many casual leave days?")
            db.close()
            self.assertTrue(any("without any retrieved evidence" in w
                                for w in r.citation_warnings))

    def test_no_evidence_for_chitchat_does_not_warn(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "g2.db")
            v = AnswerVerifier(db=db, llm=None, enabled=True,
                               shape_classifier=StubShape("chitchat"))
            r = AgentResult(answer="Hello! How can I help?")
            v.check(r, "hi")
            db.close()
            self.assertEqual(r.citation_warnings, [])


class StubShape:
    """Minimal stand-in for a shape classifier (real one: answer_shape.py)."""

    def __init__(self, kind, declined=None, error=""):
        self.kind = kind
        self.declined = declined or []
        self.error = error

    def classify(self, question, answer, context_sample=None):
        from app.agent.answer_shape import AnswerShape
        return AnswerShape(kind=self.kind, declined_parts=list(self.declined),
                           decided_by="stub", error=self.error)


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "v.db")
        emb = FakeEmbedder()
        self.doc = self.db.add_document("rw", "rw.txt", "txt", 10, "ready")
        self.cid = self.db.add_chunk(self.doc, 0, "2. Allowance", "rw > 2. Allowance",
                                     CTX, emb.embed_documents([CTX])[0])

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()


class TestGatherContexts(_Base):
    def test_fetches_seen_chunks(self):
        ctxs = gather_contexts(self.db, [self.cid])
        self.assertEqual(len(ctxs), 1)
        self.assertIn("3 days", ctxs[0])

    def test_missing_chunk_skipped(self):
        self.assertEqual(gather_contexts(self.db, [999]), [])


class TestJudgeGate(_Base):
    """The judge must run for substantive answers, and only then."""

    def _result(self, answer, citations):
        r = AgentResult(answer=answer)
        r.seen_chunk_ids = [self.cid]
        r.citations = citations
        return r

    def test_pure_refusal_skips_the_judge(self):
        fake = FakeLLM([])  # any call would raise
        v = AnswerVerifier(self.db, fake, enabled=True, shape_classifier=None)
        r = self._result("The ingested policy documents do not cover this.", [])
        v.check(r, "do the documents cover sabbaticals?")
        self.assertEqual(fake.received, [])
        self.assertIsNone(r.faithful)

    def test_refusal_that_also_asserts_a_fact_is_judged(self):
        verdict = ('{"faithful": false, "unsupported_claims": ["3 days"], '
                   '"reason": "invented"}')
        fake = FakeLLM([LLMResponse(content=verdict)])
        v = AnswerVerifier(self.db, fake, enabled=True, shape_classifier=None)
        r = self._result(
            "The policy does not cover sabbaticals, but casual leave carries "
            "3 days [rw § 2. Allowance].", ["rw § 2. Allowance"])
        v.check(r, "how much leave?")
        self.assertEqual(len(fake.received), 1)
        self.assertFalse(r.faithful)

    def test_user_message_is_forwarded_to_the_judge(self):
        captured = {}

        def chat_json(system, user):
            captured["system"] = system
            captured["user"] = user
            return {"faithful": True, "unsupported_claims": [], "reason": "ok"}, None

        v = AnswerVerifier(self.db, StubLLM(chat_json_fn=chat_json),
                           shape_classifier=None, enabled=True)
        r = self._result("3 remote days [rw § 2. Allowance].", ["rw § 2. Allowance"])
        v.check(r, "how many of my 8 days can I keep?",
                user_message="I have 8 unused days - how many can I keep?")
        self.assertTrue(r.faithful)
        self.assertIn("USER MESSAGE", captured["user"])
        self.assertIn("8 unused days", captured["user"])
        self.assertIn("Restating those facts", captured["system"])


class TestAnswerVerifier(_Base):
    def _result(self, answer, seen=None):
        return AgentResult(answer=answer,
                           seen_chunk_ids=[self.cid] if seen is None else seen)

    def test_faithful_answer_passes(self):
        fake = FakeLLM([LLMResponse(content=(
            '{"faithful": true, "unsupported_claims": [], "reason": "ok"}'))])
        v = AnswerVerifier(self.db, fake, enabled=True, shape_classifier=None)
        r = self._result("Employees may work remotely up to 3 days per week [rw #1].")
        v.check(r, "remote days?")
        self.assertTrue(r.faithful)
        self.assertEqual(r.unsupported_claims, [])

    def test_unfaithful_answer_flagged_with_claims(self):
        fake = FakeLLM([LLMResponse(content=(
            '{"faithful": false, "unsupported_claims": '
            '["the answer claims 5 remote days"], "reason": "number invented"}'))])
        v = AnswerVerifier(self.db, fake, enabled=True, shape_classifier=None)
        r = self._result("Employees get 5 remote days and a free monitor [rw #1].")
        v.check(r, "remote days?")
        self.assertFalse(r.faithful)
        self.assertEqual(r.unsupported_claims, ["the answer claims 5 remote days"])
        self.assertTrue(any("faithfulness check FAILED" in w
                            for w in r.citation_warnings))

    def test_no_evidence_answer_warns_without_llm(self):
        v = AnswerVerifier(self.db, None, enabled=True)
        r = self._result("Everyone gets 30 days of leave.", seen=[])
        v.check(r, "leave?")
        self.assertIsNone(r.faithful)
        self.assertTrue(any("without any retrieved evidence" in w
                            for w in r.citation_warnings))

    def test_refusal_answer_skips_judge(self):
        v = AnswerVerifier(self.db, FakeLLM([]), enabled=True)  # empty script
        r = self._result("The ingested policy documents do not cover this.")
        v.check(r, "dress code?")
        self.assertIsNone(r.faithful)  # judge never ran, no warning invented

    def test_disabled_verifier_is_a_noop(self):
        v = AnswerVerifier(self.db, FakeLLM([]), enabled=False)
        r = self._result("30 days of leave for everyone!", seen=[])
        v.check(r, "leave?")
        self.assertEqual(r.citation_warnings, [])

    def test_broken_judge_json_degrades_open(self):
        fake = FakeLLM([LLMResponse(content="not json"),
                        LLMResponse(content="still not json")])
        v = AnswerVerifier(self.db, fake, enabled=True, shape_classifier=None)
        r = self._result("3 remote days [rw #1].")
        v.check(r, "remote days?")
        self.assertIsNone(r.faithful)   # no verdict -> stays unknown
        self.assertEqual(r.unsupported_claims, [])
        self.assertFalse(any("faithfulness" in w for w in r.citation_warnings))

    def test_false_without_claims_is_retried_then_accepted(self):
        """An accusation must name a claim; an unsubstantiated one is unusable."""
        bad = '{"faithful": false, "unsupported_claims": [], "reason": "feels off"}'
        good = ('{"faithful": false, "unsupported_claims": ["carry-forward is '
                'unlimited"], "reason": "absent from contexts"}')
        fake = FakeLLM([LLMResponse(content=bad), LLMResponse(content=good)])
        v = AnswerVerifier(self.db, fake, enabled=True, shape_classifier=None)
        r = self._result("Employees get 5 remote days [rw #1].")
        r.citations = ["rw § 2. Allowance"]
        v.check(r, "how many days?")
        self.assertFalse(r.faithful)
        self.assertEqual(r.unsupported_claims, ["carry-forward is unlimited"])

    def test_false_without_claims_twice_degrades_to_unknown(self):
        bad = '{"faithful": false, "unsupported_claims": [], "reason": "feels off"}'
        fake = FakeLLM([LLMResponse(content=bad), LLMResponse(content=bad)])
        v = AnswerVerifier(self.db, fake, enabled=True, shape_classifier=None)
        r = self._result("Employees get 5 remote days [rw #1].")
        r.citations = ["rw § 2. Allowance"]
        v.check(r, "how many days?")
        self.assertIsNone(r.faithful)          # unknown, not a false alarm
        self.assertEqual(r.citation_warnings, [])

    def test_truthful_verdict_is_not_retried(self):
        good = '{"faithful": true, "unsupported_claims": [], "reason": "ok"}'
        fake = FakeLLM([LLMResponse(content=good)])
        v = AnswerVerifier(self.db, fake, enabled=True, shape_classifier=None)
        r = self._result("Employees get 3 remote days [rw #1].")
        r.citations = ["rw § 2. Allowance"]
        v.check(r, "how many days?")
        self.assertTrue(r.faithful)
        self.assertEqual(len(fake.received), 1)   # exactly one judge call, no retry


if __name__ == "__main__":
    unittest.main()
