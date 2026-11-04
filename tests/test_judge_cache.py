"""Judge verdict cache: reproducible verdicts without paying for a repeat."""
import tempfile
import unittest
from pathlib import Path

from app.agent.verify import AnswerVerifier
from app.db import Database
from app.prompts import FAITHFULNESS_SYSTEM
from tests.fakes import StubLLM

VERDICT = {"faithful": False, "unsupported_claims": ["5 remote days"],
           "reason": "number invented"}


class TestJudgeCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "t.db")
        self.cache = str(Path(self.tmp.name) / "judge_cache.json")
        self.calls = {"n": 0}

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def _verifier(self, cache_path=None):
        def chat_json(system, user):
            self.calls["n"] += 1
            return dict(VERDICT), None

        return AnswerVerifier(self.db, StubLLM(chat_json_fn=chat_json),
                            shape_classifier=None,
                              enabled=True, system_prompt=FAITHFULNESS_SYSTEM,
                              cache_path=cache_path, prompt_version="v1")

    def test_identical_verdict_is_replayed_without_a_call(self):
        v = self._verifier(self.cache)
        q, a, ctx = "how many remote days?", "Five days.", ["three days per week"]
        first = v.judge(q, a, ctx)
        second = v.judge(q, a, ctx)
        self.assertEqual(self.calls["n"], 1)
        self.assertEqual(first.faithful, second.faithful)
        self.assertEqual(second.unsupported_claims, ["5 remote days"])

    def test_different_answer_or_context_misses(self):
        v = self._verifier(self.cache)
        v.judge("q", "Five days.", ["three days"])
        v.judge("q", "Six days.", ["three days"])
        v.judge("q", "Five days.", ["four days"])
        self.assertEqual(self.calls["n"], 3)

    def test_cache_survives_a_new_instance(self):
        self._verifier(self.cache).judge("q", "a", ["ctx"])
        again = self._verifier(self.cache).judge("q", "a", ["ctx"])
        self.assertEqual(self.calls["n"], 1)
        self.assertFalse(again.faithful)

    def test_no_cache_path_means_no_caching(self):
        v = self._verifier(None)
        v.judge("q", "a", ["ctx"])
        v.judge("q", "a", ["ctx"])
        self.assertEqual(self.calls["n"], 2)

    def test_prompt_version_participates_in_the_key(self):
        self._verifier(self.cache).judge("q", "a", ["ctx"])
        other = AnswerVerifier(self.db, StubLLM(
            chat_json_fn=lambda s, u: (dict(VERDICT), None)),
            enabled=True, system_prompt=FAITHFULNESS_SYSTEM,
            cache_path=self.cache, prompt_version="v2")
        other.judge("q", "a", ["ctx"])
        self.assertEqual(self.calls["n"], 1)   # v1 cached; v2 missed (own call path)


if __name__ == "__main__":
    unittest.main()
