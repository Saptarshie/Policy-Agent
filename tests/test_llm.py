"""Client-level behaviour: optional provider fields must degrade, not crash."""
import unittest
from unittest import mock

from app.llm import LLMError, OpenAICompatClient


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


def _ok(content="ok"):
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {}}


class TestReasoningEffort(unittest.TestCase):
    """reasoning_effort is optional and degrades cleanly when unknown."""

    def _client(self, effort=""):
        from app.llm import OpenAICompatClient
        return OpenAICompatClient(base_url="http://x/v1", api_key="k",
                                  model="m", reasoning_effort=effort)

    def test_field_is_omitted_by_default(self):
        captured = {}

        def fake_post(payload, **kw):
            captured.update(payload)
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {}}

        c = self._client("")
        c._post_with_fallbacks = fake_post
        c.chat([{"role": "user", "content": "hi"}])
        self.assertNotIn("reasoning_effort", captured)

    def test_field_is_sent_when_configured(self):
        captured = {}

        def fake_post(payload, **kw):
            captured.update(payload)
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {}}

        c = self._client("off")
        c._post_with_fallbacks = fake_post
        c.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(captured["reasoning_effort"], "off")

    def test_per_call_override(self):
        captured = {}

        def fake_post(payload, **kw):
            captured.update(payload)
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {}}

        c = self._client("off")
        c._post_with_fallbacks = fake_post
        c.chat([{"role": "user", "content": "hi"}], reasoning_effort="high")
        self.assertEqual(captured["reasoning_effort"], "high")




class TestFeatureDegradation(unittest.TestCase):
    """An endpoint that does not know a field must not break the request."""

    def client(self, **kw):
        kw.setdefault("base_url", "http://x/v1")
        kw.setdefault("api_key", "k")
        kw.setdefault("model", "m")
        return OpenAICompatClient(**kw)

    def test_rejected_reasoning_effort_is_dropped_and_retried(self):
        payloads = []

        def fake_post(url, headers=None, json=None, timeout=None):
            payloads.append(dict(json))
            if len(payloads) == 1:
                return _Resp(400, {"error": {"message":
                                             "Invalid option: reasoning_effort"}})
            return _Resp(200, _ok("fine"))

        with mock.patch("app.llm.requests.post", side_effect=fake_post):
            r = self.client(reasoning_effort="off", max_retries=1).chat(
                [{"role": "user", "content": "hi"}])
        self.assertEqual(r.content, "fine")
        self.assertIn("reasoning_effort", payloads[0])
        self.assertNotIn("reasoning_effort", payloads[1])

    def test_oversized_max_tokens_is_halved(self):
        payloads = []

        def fake_post(url, headers=None, json=None, timeout=None):
            payloads.append(dict(json))
            if len(payloads) == 1:
                return _Resp(400, {"error": {"message": "max_tokens too large"}})
            return _Resp(200, _ok("fine"))

        with mock.patch("app.llm.requests.post", side_effect=fake_post):
            r = self.client(max_output_tokens=8192).chat(
                [{"role": "user", "content": "hi"}])
        self.assertEqual(r.content, "fine")
        self.assertEqual(payloads[1]["max_tokens"], 4096)

    def test_auth_error_is_not_retried(self):
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append(1)
            return _Resp(401, {"error": {"message": "bad key"}})

        with mock.patch("app.llm.requests.post", side_effect=fake_post):
            with self.assertRaises(LLMError):
                self.client().chat([{"role": "user", "content": "hi"}])
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
