"""Calculator + scratchpad tools, and parallel tool calling.

Three contracts:

1. ``calculate`` is exact and safe: arithmetic only, variables supplied in the
   same call, and nothing that could execute code or reach an attribute.
2. The scratchpad is per-``ask()`` working memory: notes never leak between
   questions, and the tools fail loudly rather than silently dropping a write.
3. The ReAct loop executes every tool call in one assistant turn, so parallel
   calling works without any new machinery - the prompt is what has to permit it.
"""
import tempfile
import unittest
from pathlib import Path

from app.agent.calc import CalcError, calculate
from app.agent.react import ReactAgent
from app.agent.scratchpad import MAX_NOTES, Scratchpad
from app.agent.tools import TOOL_SPECS, ToolRegistry
from app.db import Database
from app.llm import LLMResponse, ToolCall
from app.prompts import AGENT_SYSTEM, AGENT_SYSTEM_V2
from app.rag import RAGEngine
from tests.fakes import FakeEmbedder, FakeLLM


class TestCalculate(unittest.TestCase):
    def test_pro_rata(self):
        r = calculate("days * months / 12", {"days": 12, "months": 7})
        self.assertEqual(r["value"], 7)
        self.assertEqual(r["resolved"], "12 * 7 / 12")

    def test_tax_from_stated_inputs(self):
        r = calculate("(gross - deductions) * rate",
                      {"gross": 100000, "deductions": 15000, "rate": 0.2})
        self.assertEqual(r["value"], 17000)

    def test_precedence_and_rounding(self):
        self.assertEqual(calculate("2 + 3 * 4")["value"], 14)
        self.assertEqual(calculate("round(10 / 3, 2)")["value"], 3.33)

    def test_percentages_and_units_are_the_caller_s_job(self):
        self.assertEqual(calculate("cap * 1.5", {"cap": 6})["value"], 9)

    def test_unknown_variable_names_the_known_ones(self):
        with self.assertRaises(CalcError) as cm:
            calculate("x + 1", {"y": 2})
        self.assertIn("unknown variable 'x'", str(cm.exception))
        self.assertIn("'y'", str(cm.exception))

    def test_code_is_not_reachable(self):
        for bad in ('__import__("os")', "a.b", "exec('x')", "[1,2]",
                    "lambda: 1", "open('f')", "'text'"):
            with self.assertRaises(CalcError):
                calculate(bad, {})

    def test_division_by_zero_and_runaway_powers(self):
        for bad in ("1/0", "1//0", "5%0", "10**99"):
            with self.assertRaises(CalcError):
                calculate(bad, {})

    def test_empty_expression(self):
        with self.assertRaises(CalcError):
            calculate("   ", {})


class TestScratchpad(unittest.TestCase):
    def test_write_read_roundtrip(self):
        pad = Scratchpad()
        pad.write("cap", "6 days [leave_policy § 3.2]")
        self.assertIn("6 days", pad.read("cap"))
        self.assertIn("cap", pad.list_keys())

    def test_append_keeps_earlier_findings(self):
        pad = Scratchpad()
        pad.write("cap", "v1: 6 days")
        pad.append("cap", "amendment: 15 days from 1 July 2025")
        body = pad.read("cap")
        self.assertIn("v1: 6 days", body)
        self.assertIn("15 days", body)

    def test_edit_replaces_first_occurrence_only(self):
        pad = Scratchpad()
        pad.write("cap", "6 days ... 6 days")
        pad.edit("cap", "6 days", "15 days")
        self.assertEqual(pad.read("cap").count("15 days"), 1)
        self.assertEqual(pad.read("cap").count("6 days"), 1)

    def test_edit_of_missing_text_fails_loudly(self):
        pad = Scratchpad()
        pad.write("cap", "6 days")
        with self.assertRaises(ValueError):
            pad.edit("cap", "9 days", "10 days")

    def test_delete_and_read_all(self):
        pad = Scratchpad()
        pad.write("a", "one")
        pad.write("b", "two")
        pad.delete("a")
        self.assertEqual(pad.keys(), ["b"])
        self.assertIn("=== b ===", pad.read())

    def test_unknown_key_is_an_error_not_an_empty_note(self):
        with self.assertRaises(KeyError):
            Scratchpad().read("ghost")

    def test_bounds(self):
        pad = Scratchpad()
        for i in range(MAX_NOTES):
            pad.write(f"n{i}", "x")
        with self.assertRaises(ValueError):
            pad.write("overflow", "y")

    def test_empty_text_rejected(self):
        with self.assertRaises(ValueError):
            Scratchpad().write("k", "   ")


class TestToolsWiring(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "t.db")
        self.registry = ToolRegistry(self.db, RAGEngine(self.db, FakeEmbedder()))

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_six_thinking_tools_are_exposed(self):
        names = [t["function"]["name"] for t in TOOL_SPECS]
        for name in ("calculate", "scratchpad_write", "scratchpad_read",
                     "scratchpad_edit", "scratchpad_delete", "scratchpad_list"):
            self.assertIn(name, names)
        self.assertEqual(len(names), 10)

    def test_calculate_through_the_registry(self):
        out = self.registry.execute("calculate",
                                    {"expression": "days * 2", "variables": {"days": 6}})
        self.assertIn("12", out)

    def test_scratchpad_tools_need_a_pad(self):
        out = self.registry.execute("scratchpad_list", {})
        self.assertIn("ERROR", out)          # fails loudly, does not silently no-op

    def test_scratchpad_tools_work_with_a_pad(self):
        pad = Scratchpad()
        self.assertIn("written", self.registry.execute(
            "scratchpad_write", {"key": "k", "text": "v"}, scratch=pad))
        self.assertIn("v", self.registry.execute(
            "scratchpad_read", {"key": "k"}, scratch=pad))
        self.assertEqual(pad.keys(), ["k"])


class TestParallelToolCalls(unittest.TestCase):
    """Two tool_calls in one assistant turn must both run, in order, with one
    tool message per call id - that is what makes batching work."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "p.db")
        self.doc = self.db.add_document("leave_policy", "leave_policy.md", "md",
                                        status="ready")
        emb = FakeEmbedder()
        self.c1 = self.db.add_chunk(self.doc, 0, "2. Casual Leave", "leave_policy",
                                    "Casual leave is 6 days per year.",
                                    emb.embed_query("casual leave"))
        self.c2 = self.db.add_chunk(self.doc, 1, "3.2 Carry-Forward", "leave_policy",
                                    "Up to 6 days may be carried forward.",
                                    emb.embed_query("carry forward"))
        self.registry = ToolRegistry(self.db, RAGEngine(self.db, emb))

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_both_calls_execute_in_one_turn(self):
        fake = FakeLLM([
            LLMResponse(content="", tool_calls=[
                ToolCall("a", "read_policy_chunk", '{"chunk_id": %d}' % self.c1),
                ToolCall("b", "read_policy_chunk", '{"chunk_id": %d}' % self.c2),
            ]),
            LLMResponse(content="6 days, and 6 may be carried forward "
                                "[leave_policy § 3.2 Carry-Forward]."),
        ])
        agent = ReactAgent(fake, self.registry, max_steps=4)
        result = agent.run("how much leave can I carry forward?")

        self.assertEqual(result.tool_calls, 2)
        self.assertEqual(result.error, None)
        tool_steps = [s for s in result.steps if s["type"] == "tool_call"]
        self.assertEqual([s["tool"] for s in tool_steps],
                         ["read_policy_chunk", "read_policy_chunk"])

        # one assistant message carrying BOTH calls, then one tool message per id
        roles = [m["role"] for m in fake.received[1]["messages"]]
        self.assertEqual(roles.count("tool"), 2)
        tool_msgs = [m for m in fake.received[1]["messages"] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["a", "b"])

        # every chunk it read is provenance, regardless of which call returned it
        self.assertEqual(set(result.seen_chunk_ids), {self.c1, self.c2})

    def test_notes_written_during_a_run_are_reported(self):
        fake = FakeLLM([
            LLMResponse(content="", tool_calls=[
                ToolCall("n1", "scratchpad_write",
                         '{"key": "cap", "text": "6 days [leave_policy 3.2]"}'),
            ]),
            LLMResponse(content="6 days [leave_policy § 3.2 Carry-Forward]."),
        ])
        result = ReactAgent(fake, self.registry, max_steps=4).run("carry forward?")
        self.assertEqual(result.scratch_notes, 1)
        self.assertIn("written", [s for s in result.steps
                                  if s["type"] == "tool_call"][0]["observation"])

    def test_notes_from_one_question_never_reach_the_next(self):
        write = LLMResponse(content="", tool_calls=[
            ToolCall("n1", "scratchpad_write", '{"key": "secret", "text": "x"}')])
        read = LLMResponse(content="", tool_calls=[
            ToolCall("n2", "scratchpad_read", "{}")])
        fake = FakeLLM([write, LLMResponse(content="done"),
                        read, LLMResponse(content="done")])
        agent = ReactAgent(fake, self.registry, max_steps=4)
        agent.run("first")
        second = agent.run("second")
        observation = [s for s in second.steps if s["type"] == "tool_call"][0]["observation"]
        self.assertIn("scratchpad is empty", observation)


class TestPromptsPermitIt(unittest.TestCase):
    def _assert_guidance(self, prompt: str) -> None:
        low = prompt.lower()
        self.assertIn("scratchpad_write", low)
        self.assertIn("calculate", low)
        self.assertIn("parallel", low)
        self.assertIn("scratchpad_read once", low)
        # and it must stay optional: the prompt has to say when NOT to use it
        self.assertTrue("do not write notes" in low or "needs no notes" in low
                        or "no notes" in low)

    def test_v1_agent_prompt(self):
        self._assert_guidance(AGENT_SYSTEM)

    def test_v2_agent_prompt(self):
        self._assert_guidance(AGENT_SYSTEM_V2)


if __name__ == "__main__":
    unittest.main()
