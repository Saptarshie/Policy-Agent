"""The answer-shape classifier: decision model, not keyword list.

Three things are tested here:

1. ``shape_from_scores`` maps decision-model probabilities to a shape, and every
   ambiguous combination falls through to ``unknown`` (which means "judge it").
2. A classifier outage degrades to ``unknown``, never to "skip the check".
3. The verifier acts on the shape: chit-chat is skipped, a partial answer is
   always judged and the declined half is handed to the judge as a claim about
   the corpus.
"""
import tempfile
import unittest
from pathlib import Path

from app.agent.answer_shape import (AnswerShape, LLMShapeClassifier,
                                    SystemOneShapeClassifier, shape_from_scores)
from app.agent.react import AgentResult
from app.agent.verify import AnswerVerifier
from app.db import Database
from app.llm import LLMResponse
from tests.fakes import FakeEmbedder, StubLLM


class FakeDecisionModel:
    """Stands in for SystemOneDecisionModel: returns fixed probabilities."""

    def __init__(self, scores=None):
        self.scores = scores if scores is not None else {}
        self.calls = []

    def ask(self, state, questions):
        self.calls.append({"state": state, "questions": list(questions)})
        if self.scores is None:
            return None
        return dict(self.scores)


class TestShapeFromScores(unittest.TestCase):
    def test_bare_refusal(self):
        s = shape_from_scores({"declines_part": .9, "states_rules": .02,
                               "is_only_refusal": .95, "is_chitchat": .01})
        self.assertEqual(s.kind, "refusal")

    def test_partial_answer(self):
        s = shape_from_scores({"declines_part": .96, "states_rules": .86,
                               "is_only_refusal": .04, "is_chitchat": .01})
        self.assertEqual(s.kind, "partial")
        self.assertTrue(s.is_partial)
        self.assertTrue(s.has_claims)

    def test_full_answer(self):
        s = shape_from_scores({"declines_part": .03, "states_rules": .93,
                               "is_only_refusal": .02, "is_chitchat": .0})
        self.assertEqual(s.kind, "answer")

    def test_chitchat(self):
        s = shape_from_scores({"declines_part": .05, "states_rules": .04,
                               "is_only_refusal": .1, "is_chitchat": .92})
        self.assertEqual(s.kind, "chitchat")
        self.assertFalse(s.has_claims)

    def test_contradictory_confidence_never_skips(self):
        # declines + rules both high but "only a refusal" also high: nonsense
        s = shape_from_scores({"declines_part": .8, "states_rules": .8,
                               "is_only_refusal": .8, "is_chitchat": .8})
        self.assertEqual(s.kind, "partial")
        self.assertTrue(s.has_claims)

    def test_missing_scores_is_unknown(self):
        self.assertEqual(shape_from_scores({}).kind, "unknown")


class TestClassifierBackends(unittest.TestCase):
    def test_decision_model_is_asked_four_questions(self):
        dm = FakeDecisionModel({"declines_part": .1, "states_rules": .9,
                                "is_only_refusal": .0, "is_chitchat": .0})
        s = SystemOneShapeClassifier(dm).classify(
            "how many days?", "Six days [leave § 2].")
        self.assertEqual(s.kind, "answer")
        self.assertEqual(s.decided_by, "decision_model")
        self.assertEqual(len(dm.calls[0]["questions"]), 4)

    def test_decision_model_down_degrades_to_unknown(self):
        s = SystemOneShapeClassifier(FakeDecisionModel(None)).classify("q", "a")
        self.assertEqual(s.kind, "unknown")
        self.assertTrue(s.error)

    def test_llm_backend_parses_a_verdict(self):
        llm = StubLLM(chat_json_fn=lambda system, user: (
            {"shape": "partial", "declined_parts": ["transfer to another entity"],
             "reason": "covers resignation only"}, None))
        s = LLMShapeClassifier(llm).classify("resign or transfer?", "…")
        self.assertEqual(s.kind, "partial")
        self.assertEqual(s.declined_parts, ["transfer to another entity"])

    def test_llm_backend_garbage_is_unknown(self):
        llm = StubLLM(chat_json_fn=lambda system, user: (None, "not json"))
        self.assertEqual(LLMShapeClassifier(llm).classify("q", "a").kind,
                         "unknown")

    def test_llm_backend_exception_is_unknown(self):
        def boom(system, user):
            raise RuntimeError("endpoint down")
        llm = StubLLM(chat_json_fn=boom)
        s = LLMShapeClassifier(llm).classify("q", "a")
        self.assertEqual(s.kind, "unknown")
        self.assertIn("endpoint down", s.error)


class _VerifierBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "shape.db")
        self.doc = self.db.add_document("leave_policy.md", "leave_policy.md",
                                        "md", status="ready")
        self.chunk = self.db.add_chunk(
            self.doc, 0, "2. Casual Leave", "leave_policy.md",
            "Casual leave is 6 days per year.",
            FakeEmbedder().embed_query("casual leave"))

    def tearDown(self):
        try:
            self.db.close()
        except Exception:
            pass
        self.tmp.cleanup()

    def _verifier(self, shape, judge_obj=None, judge_calls=None):
        calls = judge_calls if judge_calls is not None else []

        def chat_json(system, user):
            calls.append(user)
            return (judge_obj if judge_obj is not None else
                    {"faithful": True, "unsupported_claims": [], "reason": "ok"},
                    None)

        llm = StubLLM(chat_json_fn=chat_json)
        return AnswerVerifier(db=self.db, llm=llm, enabled=True,
                              shape_classifier=shape, cache_path=None), calls


class TestVerifierUsesShape(_VerifierBase):
    PARTIAL = ("If you resign, unused casual leave lapses on your last working day "
               "[leave_policy § 2]. The documents do not say what happens to the "
               "balance if you transfer to another entity mid-year.")

    def test_partial_answer_is_always_judged(self):
        v, calls = self._verifier(StubShape("partial", declined=["transfer"]))
        r = AgentResult(answer=self.PARTIAL, seen_chunk_ids=[self.chunk],
                        citations=["leave_policy § 2"])
        v.check(r, "what happens on resignation or transfer?")
        self.assertEqual(r.answer_shape, "partial")
        self.assertEqual(len(calls), 1, "a partial answer must be judged")
        self.assertIn("DECLINED BY THE ANSWER", calls[0])
        self.assertIn("TWO HALVES", calls[0])

    def test_chitchat_skips_the_judge(self):
        v, calls = self._verifier(StubShape("chitchat"))
        r = AgentResult(answer="Hello! How can I help you today?",
                        seen_chunk_ids=[self.chunk])
        v.check(r, "hi")
        self.assertEqual(calls, [])
        self.assertIsNone(r.faithful)

    def test_refusal_is_judged_when_passages_exist(self):
        # the old code skipped every refusal on a substring; a refusal is a
        # claim about the corpus and must be checked when evidence exists
        v, calls = self._verifier(StubShape("refusal"))
        r = AgentResult(answer="The ingested policy documents do not cover this.",
                        seen_chunk_ids=[self.chunk])
        v.check(r, "what about sabbaticals?")
        self.assertEqual(len(calls), 1)

    def test_classifier_failure_still_judges(self):
        v, calls = self._verifier(StubShape("unknown", error="dm down"))
        r = AgentResult(answer=self.PARTIAL, seen_chunk_ids=[self.chunk],
                        citations=["leave_policy § 2"])
        v.check(r, "what happens on transfer?")
        self.assertEqual(len(calls), 1, "an outage must add checking, not remove it")

    def test_declined_part_reaches_the_judge_prompt(self):
        v, calls = self._verifier(
            StubShape("partial", declined=["what happens on internal transfer"]))
        r = AgentResult(answer=self.PARTIAL, seen_chunk_ids=[self.chunk],
                        citations=["leave_policy § 2"])
        v.check(r, "resign or transfer?")
        self.assertIn("what happens on internal transfer", calls[0])
        self.assertIn("claim ABOUT THE CORPUS", calls[0])


class StubShape:
    def __init__(self, kind, declined=None, error=""):
        self.kind = kind
        self.declined = declined or []
        self.error = error

    def classify(self, question, answer, context_sample=None):
        return AnswerShape(kind=self.kind, declined_parts=list(self.declined),
                           decided_by="stub", error=self.error)


if __name__ == "__main__":
    unittest.main()
