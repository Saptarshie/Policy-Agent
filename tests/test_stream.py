"""Live trace: the agent's event hook and the SSE endpoint that exposes it."""

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.agent.react import ReactAgent
from app.agent.tools import ToolRegistry
from app.api.routes import create_app
from app.db import Database
from app.llm import LLMResponse, ToolCall
from app.rag import RAGEngine
from tests.fakes import FakeEmbedder, FakeLLM

V1 = ("Employees may work remotely for up to 3 days per calendar week.")
V2 = ("Employees must be onsite at least 4 days per calendar week.")


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "s.db")
        emb = FakeEmbedder()
        for doc, text in (("remote_work_v1", V1), ("remote_work_v2", V2)):
            did = self.db.add_document(doc, f"{doc}.txt", "txt", 10, "ready")
            self.db.add_chunk(did, 0, "2. Allowance", f"{doc} > 2. Allowance",
                              text, emb.embed_documents([text])[0])
        self.registry = ToolRegistry(self.db, RAGEngine(self.db, emb))

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()


class TestAgentEventHook(_Base):
    def _agent(self):
        fake = FakeLLM([
            LLMResponse(content="Let me search.",
                        tool_calls=[ToolCall(id="c1", name="search_policy_docs",
                                             arguments=json.dumps({"query": "remote work"}))]),
            LLMResponse(content="Up to 3 days [remote_work_v1 § 2. Allowance]."),
        ])
        return ReactAgent(llm=fake, registry=self.registry, max_steps=4), fake

    def test_events_arrive_in_order(self):
        agent, _ = self._agent()
        events = []
        result = agent.run("How many remote days?", on_event=lambda k, p: events.append((k, p)))

        kinds = [k for k, _ in events]
        self.assertEqual(kinds[0], "turn")
        self.assertIn("step", kinds)
        self.assertLess(kinds.index("turn"), kinds.index("step"))

        steps = [p for k, p in events if k == "step"]
        self.assertEqual(steps[0]["type"], "thought")
        tool_steps = [s for s in steps if s["type"] == "tool_call"]
        self.assertEqual(tool_steps[0]["tool"], "search_policy_docs")
        self.assertIn("observation", tool_steps[0])
        self.assertEqual(steps[-1]["type"], "final")
        # the stream and the final payload agree about what happened
        self.assertEqual(result.tool_calls, 1)

    def test_a_broken_listener_cannot_break_the_answer(self):
        agent, _ = self._agent()

        def boom(kind, payload):
            raise RuntimeError("listener exploded")

        result = agent.run("How many remote days?", on_event=boom)
        self.assertIn("3 days", result.answer)

    def test_no_listener_is_fine(self):
        agent, _ = self._agent()
        self.assertIn("3 days", agent.run("How many remote days?").answer)


class _FakeAPI:
    """Just enough AdminAPI for the route: streams a step, returns a result."""

    def ask(self, question, history=None, on_event=None):
        if on_event is None:          # the plain endpoint does not stream
            return self._result()
        on_event("stage", {"name": "triage"})
        on_event("stage", {"name": "routed", "intent": "POLICY_DIRECT_LOOKUP",
                           "route": "REACT_LOOP", "question": question})
        on_event("turn", {"n": 1})
        on_event("step", {"type": "thought", "content": "searching"})
        on_event("step", {"type": "tool_call", "tool": "search_policy_docs",
                          "arguments": {"query": "remote work"},
                          "observation": "[chunk 1] 3 days per week"})
        return self._result()

    class _Result:
        def to_dict(self):
            return {"answer": "Up to 3 days [remote_work_v1 § 2. Allowance].",
                    "tool_calls": 1, "elapsed_s": 2.4,
                    "citations": ["remote_work_v1 § 2. Allowance"],
                    "citation_warnings": [], "steps": []}

    def _result(self):
        return _FakeAPI._Result()


def _frames(text):
    """Parse SSE text into [(event, data_dict), ...]."""
    out = []
    for block in text.split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        event, data = "message", ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[7:].strip()
            elif line.startswith("data: "):
                data += line[6:]
        out.append((event, json.loads(data) if data else {}))
    return out


class TestStreamEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(api=_FakeAPI()))

    def test_frames_are_ordered_and_typed(self):
        r = self.client.post("/api/ask/stream",
                             json={"question": "How many remote days?", "history": []})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
        frames = _frames(r.text)
        events = [e for e, _ in frames]
        self.assertEqual(events[:2], ["stage", "stage"])
        self.assertIn("turn", events)
        steps = [d for e, d in frames if e == "step"]
        self.assertEqual(steps[0]["type"], "thought")
        self.assertEqual(steps[1]["tool"], "search_policy_docs")
        final = [d for e, d in frames if e == "final"][0]
        self.assertIn("3 days", final["answer"])
        self.assertEqual(final["tool_calls"], 1)
        # final is the last frame; the connection closes after it
        self.assertEqual(events[-1], "final")

    def test_empty_question_is_rejected_before_streaming(self):
        r = self.client.post("/api/ask/stream", json={"question": "  "})
        self.assertEqual(r.status_code, 400)

    def test_plain_endpoint_still_works(self):
        r = self.client.post("/api/ask", json={"question": "How many remote days?"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("3 days", r.json()["answer"])


if __name__ == "__main__":
    unittest.main()
