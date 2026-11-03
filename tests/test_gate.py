import unittest

from app.contradictions.gate import (
    CascadeGate,
    GateDecision,
    JEVStyleGate,
    RemoteDecisionModel,
    SystemOneDecisionModel,
    build_gate_from_config,
)
from tests.fakes import StubLLM


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


class TestCascadeGateModes(unittest.TestCase):
    """The facade the tracker uses: heuristic / remote / hybrid + fallback."""

    def test_heuristic_mode_uses_no_llm(self):
        called = []
        stub = StubLLM(chat_json_fn=lambda s, u: called.append(1) or ({"score": 1.0}, None))
        gate = CascadeGate(threshold=0.5, mode="heuristic", remote=RemoteDecisionModel(stub))
        d = gate.evaluate(V1, V2)
        self.assertTrue(d.passed)
        self.assertEqual(d.decided_by, "heuristic")
        self.assertEqual(called, [])  # zero endpoint calls

    def test_remote_mode_uses_endpoint_score(self):
        stub = StubLLM(chat_json_fn=lambda s, u: ({"score": 0.9, "signals": ["conflicting numbers"]}, None))
        gate = CascadeGate(threshold=0.5, mode="remote", remote=RemoteDecisionModel(stub))
        d = gate.evaluate(V1, V2)
        self.assertTrue(d.passed)
        self.assertEqual(d.decided_by, "remote")
        self.assertEqual(d.score, 0.9)

    def test_remote_mode_threshold_applied(self):
        # remote score below the threshold -> not escalated
        stub = StubLLM(chat_json_fn=lambda s, u: ({"score": 0.2, "signals": []}, None))
        gate = CascadeGate(threshold=0.5, mode="remote", remote=RemoteDecisionModel(stub))
        self.assertFalse(gate.evaluate(V1, V2).passed)

    def test_remote_failure_falls_back_to_heuristic(self):
        def boom(s, u):
            raise RuntimeError("endpoint down")
        gate = CascadeGate(threshold=0.5, mode="remote",
                           remote=RemoteDecisionModel(StubLLM(chat_json_fn=boom)))
        d = gate.evaluate(V1, V2)  # heuristically a strong conflict
        self.assertTrue(d.passed)
        self.assertEqual(d.decided_by, "remote->heuristic")
        self.assertTrue(d.error)

    def test_remote_failure_without_fallback_blocks(self):
        def boom(s, u):
            raise RuntimeError("endpoint down")
        gate = CascadeGate(threshold=0.5, mode="remote",
                           remote=RemoteDecisionModel(StubLLM(chat_json_fn=boom)),
                           fallback_heuristic=False)
        d = gate.evaluate(V1, V2)
        self.assertFalse(d.passed)
        self.assertEqual(d.decided_by, "remote")

    def test_hybrid_clear_pass_never_calls_endpoint(self):
        called = []
        stub = StubLLM(chat_json_fn=lambda s, u: called.append(1) or ({"score": 1.0}, None))
        gate = CascadeGate(threshold=0.5, mode="hybrid", remote=RemoteDecisionModel(stub),
                           borderline_band=0.15)
        d = gate.evaluate(V1, V2)  # heuristic score ~1.0 -> clear pass
        self.assertTrue(d.passed)
        self.assertEqual(d.decided_by, "hybrid(heuristic)")
        self.assertEqual(called, [])

    def test_hybrid_clear_fail_never_calls_endpoint(self):
        called = []
        stub = StubLLM(chat_json_fn=lambda s, u: called.append(1) or ({"score": 1.0}, None))
        gate = CascadeGate(threshold=0.5, mode="hybrid", remote=RemoteDecisionModel(stub),
                           borderline_band=0.15)
        d = gate.evaluate(V1, TRAVEL)  # unrelated -> score 0, clear fail
        self.assertFalse(d.passed)
        self.assertEqual(d.decided_by, "hybrid(heuristic)")
        self.assertEqual(called, [])

    def test_hybrid_borderline_refines_remotely(self):
        # Force a borderline heuristic score by choosing a pair whose
        # heuristic score sits inside [threshold-band, threshold+band):
        # SIMILAR_STANCE vs V2 gives a moderate score. Use a wide band so the
        # pair is guaranteed borderline, then let the remote model decide.
        heur = JEVStyleGate(threshold=0.5).evaluate(V2, SIMILAR_STANCE)
        band = max(0.2, heur.score + 0.05)  # ensure borderline
        stub = StubLLM(chat_json_fn=lambda s, u: ({"score": 0.85, "signals": ["1 vs 4 office days"]}, None))
        gate = CascadeGate(threshold=0.5, mode="hybrid", remote=RemoteDecisionModel(stub),
                           borderline_band=band)
        d = gate.evaluate(V2, SIMILAR_STANCE)
        self.assertTrue(d.passed)
        self.assertEqual(d.decided_by, "hybrid(remote)")
        self.assertTrue(any("borderline" in s for s in d.signals))

    def test_hybrid_remote_failure_keeps_heuristic_decision(self):
        def boom(s, u):
            raise RuntimeError("endpoint down")
        heur = JEVStyleGate(threshold=0.5).evaluate(V2, SIMILAR_STANCE)
        band = max(0.2, heur.score + 0.05)
        gate = CascadeGate(threshold=0.5, mode="hybrid",
                           remote=RemoteDecisionModel(StubLLM(chat_json_fn=boom)),
                           borderline_band=band)
        d = gate.evaluate(V2, SIMILAR_STANCE)
        self.assertEqual(d.decided_by, "hybrid(heuristic)")
        self.assertTrue(d.error)

    def test_mode_downgrades_without_remote(self):
        gate = CascadeGate(threshold=0.5, mode="remote", remote=None)
        self.assertEqual(gate.mode, "heuristic")  # graceful downgrade
        self.assertTrue(gate.evaluate(V1, V2).passed)


class TestBuildGateFromConfig(unittest.TestCase):
    """Factory wires .env vars into the cascade gate."""

    def _cfg(self, **kw):
        from types import SimpleNamespace
        base = dict(
            decision_mode="heuristic",
            decision_model_base_url="",
            decision_model_api_key="",
            decision_model_name="",
            decision_temperature=0.0,
            decision_borderline_band=0.15,
            decision_fallback_heuristic=True,
            jev_gate_threshold=0.5,
            # one global output ceiling for every LLM caller, the decision
            # model included: it is a ceiling, not a budget
            max_output_tokens=65536,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def test_default_is_pure_heuristic(self):
        gate = build_gate_from_config(self._cfg())
        self.assertEqual(gate.mode, "heuristic")
        self.assertIsNone(gate.remote)

    def test_remote_mode_without_endpoint_downgrades(self):
        cfg = self._cfg(decision_mode="remote")
        gate = build_gate_from_config(cfg)
        self.assertEqual(gate.mode, "heuristic")
        self.assertIsNone(gate.remote)

    def test_remote_mode_with_endpoint_builds_client(self):
        cfg = self._cfg(decision_mode="remote",
                        decision_model_base_url="http://localhost:11434/v1",
                        decision_model_api_key="ollama",
                        decision_model_name="qwen2.5:0.5b")
        gate = build_gate_from_config(cfg)
        self.assertEqual(gate.mode, "remote")
        self.assertIsNotNone(gate.remote)
        self.assertEqual(gate.remote.client.model, "qwen2.5:0.5b")
        self.assertEqual(gate.remote.client.base_url, "http://localhost:11434/v1")

    def test_hybrid_mode_with_endpoint(self):
        cfg = self._cfg(decision_mode="hybrid",
                        decision_model_base_url="http://x/v1",
                        decision_model_name="m")
        gate = build_gate_from_config(cfg)
        self.assertEqual(gate.mode, "hybrid")
        self.assertIsNotNone(gate.remote)

    def test_config_decision_mode_effective(self):
        from app.config import Config
        # re-parsing via the real Config dataclass
        fields = {f: getattr(self._cfg(), f) for f in
                  ("decision_mode", "decision_model_base_url", "decision_model_name")}
        ns = self._cfg(**fields)
        # heuristic mode stays heuristic
        self.assertEqual(build_gate_from_config(ns).mode, "heuristic")


    def test_remote_mode_with_systemone_builds_systemone_model(self):
        cfg = self._cfg(
            decision_mode="remote",
            decision_model_base_url="https://api.commandcode.ai/provider/v1",
            decision_model_api_key="test_key",
            decision_model_name="typesafe/jev",
        )
        gate = build_gate_from_config(cfg)
        self.assertEqual(gate.mode, "remote")
        self.assertIsInstance(gate.remote, SystemOneDecisionModel)
        self.assertEqual(gate.remote.endpoint, "https://api.commandcode.ai/provider/v1/systemone")
        self.assertEqual(gate.remote.model, "typesafe/jev")


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
