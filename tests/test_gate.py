import unittest

from app.contradictions.gate import (
    CascadeGate,
    GateDecision,
    JEVStyleGate,
    RemoteDecisionModel,
    SystemOneDecisionModel,
    build_gate_from_config,
)
from tests.fakes import StubLLM, StubRemote


V1 = ("Employees may work remotely for up to 3 days per calendar week. "
      "The remaining working days must be spent at the assigned office location. "
      "A remote work arrangement must be approved in writing by the employee's "
      "direct manager before it starts.")

V2 = ("Employees must work from their assigned office at least 4 days per "
      "calendar week. Remote work is limited to a maximum of 1 day per week "
      "and requires written approval from the direct manager for each remote day.")

V1_NOTICE = ("Employees must give at least 15 days written notice to their "
             "manager before changing or terminating an approved remote work "
             "arrangement.")

V2_NOTICE = ("Employees must submit any change to their working-location "
             "pattern at least 30 days in advance through the HR portal.")

TRAVEL = ("Employees must submit all travel expense receipts within 30 days "
          "of completing the trip. Reimbursement is capped at 500 USD per "
          "domestic trip and requires director approval.")

SIMILAR_STANCE = ("Remote work is permitted for up to 3 days per week for "
                  "eligible employees, subject to manager approval.")


class TestJEVGate(unittest.TestCase):
    def setUp(self):
        self.gate = JEVStyleGate(threshold=0.5)

    def test_contradictory_pair_passes(self):
        d = self.gate.evaluate(V1, V2)
        self.assertTrue(d.passed, f"signals={d.signals} score={d.score}")
        self.assertGreaterEqual(d.score, 0.5)
        self.assertTrue(any("numeric conflict" in s for s in d.signals))
        self.assertTrue(d.signals)  # explainable

    def test_numeric_notice_conflict(self):
        d = self.gate.evaluate(V1_NOTICE, V2_NOTICE)
        self.assertTrue(d.passed, f"signals={d.signals} score={d.score}")
        self.assertTrue(any("15 days" in s and "30 days" in s for s in d.signals))

    def test_unrelated_pair_rejected(self):
        d = self.gate.evaluate(V1, TRAVEL)
        self.assertFalse(d.passed, f"signals={d.signals} score={d.score}")

    def test_same_stance_rejected(self):
        d = self.gate.evaluate(V1, SIMILAR_STANCE)
        self.assertFalse(d.passed, f"signals={d.signals} score={d.score}")

    def test_empty_input(self):
        self.assertFalse(self.gate.evaluate("", V1).passed)
        self.assertFalse(self.gate.evaluate(V1, "   ").passed)

    def test_threshold_is_configurable(self):
        strict = JEVStyleGate(threshold=0.99)
        d = strict.evaluate(V1, V2)
        self.assertFalse(d.passed)
        self.assertGreater(d.score, 0)

    def test_decision_is_deterministic(self):
        d1 = self.gate.evaluate(V1, V2)
        d2 = self.gate.evaluate(V1, V2)
        self.assertEqual((d1.passed, round(d1.score, 6)),
                         (d2.passed, round(d2.score, 6)))


class TestRemoteDecisionModel(unittest.TestCase):
    """DECISION_MODE=remote: pairs are scored by a dedicated model endpoint."""

    def _model(self, chat_json_fn):
        stub = StubLLM(chat_json_fn=chat_json_fn)
        return RemoteDecisionModel(stub), stub

    def test_valid_json_score(self):
        remote, _ = self._model(
            lambda s, u: ({"score": 0.9, "signals": ["3 vs 1 remote days"]}, None))
        d = remote.score(V1, V2, ["heuristic: numeric conflict"])
        self.assertIsNotNone(d)
        self.assertEqual(d.score, 0.9)
        self.assertEqual(d.decided_by, "remote")
        self.assertTrue(any("remote decision model" in s for s in d.signals))

    def test_score_clamped_to_unit_interval(self):
        remote, _ = self._model(lambda s, u: ({"score": 7, "signals": []}, None))
        d = remote.score(V1, V2)
        self.assertEqual(d.score, 1.0)

    def test_non_numeric_score_rejected(self):
        remote, _ = self._model(lambda s, u: ({"score": "high"}, None))
        self.assertIsNone(remote.score(V1, V2))

    def test_unparseable_reply_rejected(self):
        remote, _ = self._model(lambda s, u: (None, "not JSON"))
        self.assertIsNone(remote.score(V1, V2))

    def test_endpoint_exception_rejected(self):
        def boom(s, u):
            raise RuntimeError("endpoint down")
        remote, _ = self._model(boom)
        self.assertIsNone(remote.score(V1, V2))

    def test_signals_coerced_to_list(self):
        remote, _ = self._model(lambda s, u: ({"score": 0.6, "signals": "conflict"}, None))
        d = remote.score(V1, V2)
        self.assertTrue(isinstance(d.signals, list) and d.signals)


class TestCascadeGate(unittest.TestCase):
    """One judge: the decision model. No lexical verdicts."""

    def test_decision_model_judges(self):
        remote = StubRemote({(V1, V2): 0.9})
        gate = CascadeGate(threshold=0.5, remote=remote)
        d = gate.evaluate(V1, V2)
        self.assertTrue(d.passed)
        self.assertEqual(d.decided_by, "decision_model")

    def test_score_below_threshold_does_not_pass(self):
        gate = CascadeGate(threshold=0.5, remote=StubRemote(default=0.2))
        d = gate.evaluate(V1, V2)
        self.assertFalse(d.passed)

    def test_without_a_decision_model_nothing_passes_and_it_says_so(self):
        gate = CascadeGate(threshold=0.5, remote=None)
        d = gate.evaluate(V1, V2)
        self.assertFalse(d.passed)
        self.assertEqual(d.decided_by, "unavailable")
        self.assertIn("no decision model", d.error)
        self.assertIn("no decision model", gate.describe)

    def test_remote_failure_is_unavailable_never_a_lexical_verdict(self):
        class Boom:
            model = "boom"

            def score(self, a, b, signals=None):
                return None

        d = CascadeGate(threshold=0.5, remote=Boom()).evaluate(V1, V2)
        self.assertFalse(d.passed)
        self.assertEqual(d.decided_by, "unavailable")
        self.assertIn("no verdict", d.error)

    def test_lexical_signals_are_passed_as_evidence(self):
        remote = StubRemote(default=0.9)
        CascadeGate(threshold=0.5, remote=remote).evaluate(
            "Employees must give 15 days notice.",
            "Employees may give 15 days notice.")
        self.assertTrue(remote.calls, "the model must be asked")
        self.assertIsInstance(remote.calls[0][2], list)


class TestBuildGateFromConfig(unittest.TestCase):
    """Factory wires .env vars into the cascade gate - endpoint or nothing."""

    def _cfg(self, **kw):
        from types import SimpleNamespace
        base = dict(
            decision_model_base_url="",
            decision_model_api_key="",
            decision_model_name="",
            decision_temperature=0.0,
            jev_gate_threshold=0.5,
            max_output_tokens=65536,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def test_no_endpoint_means_no_judge(self):
        gate = build_gate_from_config(self._cfg())
        self.assertIsNone(gate.remote)
        self.assertIn("no decision model", gate.describe)
        self.assertFalse(gate.evaluate(V1, V2).passed)

    def test_systemone_endpoint_builds_the_decision_model(self):
        cfg = self._cfg(decision_model_base_url="https://api.example/v1",
                        decision_model_api_key="k",
                        decision_model_name="typesafe/jev")
        gate = build_gate_from_config(cfg)
        self.assertIsNotNone(gate.remote)
        self.assertIn("typesafe/jev", gate.describe)

    def test_openai_compatible_endpoint_builds_a_client(self):
        cfg = self._cfg(decision_model_base_url="http://localhost:11434/v1",
                        decision_model_api_key="ollama",
                        decision_model_name="qwen2.5:0.5b")
        gate = build_gate_from_config(cfg)
        self.assertIsNotNone(gate.remote)
        self.assertEqual(gate.remote.client.model, "qwen2.5:0.5b")


class TestSystemOneDecisionModel(unittest.TestCase):
    """Unit tests for SystemOneDecisionModel using mocks."""

    def test_score_success(self):
        from unittest.mock import MagicMock, patch

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model": "typesafe/jev",
            "answers": {
                "contradiction": {"type": "noul", "noul": 0.92},
                "numeric_conflict": {"type": "noul", "noul": 0.85},
                "normative_flip": {"type": "noul", "noul": 0.80},
            },
        }

        with patch("requests.post", return_value=mock_resp) as mock_post:
            model = SystemOneDecisionModel(
                base_url="https://api.commandcode.ai/provider/v1",
                api_key="key123",
                model="typesafe/jev",
            )
            d = model.score(V1, V2, ["numeric conflict"])
            self.assertIsNotNone(d)
            self.assertTrue(d.passed)
            self.assertAlmostEqual(d.score, 0.92)
            self.assertTrue(any("direct rule conflict" in s for s in d.signals))
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            self.assertEqual(kwargs["json"]["model"], "typesafe/jev")
            self.assertIn("contradiction", kwargs["json"]["questions"])

    def test_score_error_returns_none(self):
        from unittest.mock import MagicMock, patch

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("requests.post", return_value=mock_resp):
            model = SystemOneDecisionModel(
                base_url="https://api.commandcode.ai/provider/v1",
                api_key="key123",
            )
            d = model.score(V1, V2)
            self.assertIsNone(d)


if __name__ == "__main__":
    unittest.main()
