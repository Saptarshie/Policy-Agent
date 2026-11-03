import tempfile
import unittest
from pathlib import Path

from app.agent.react import ReactAgent
from app.db import Database
from app.llm import LLMResponse, ToolCall
from app.rag import RAGEngine
from app.agent.tools import ToolRegistry
from tests.fakes import FakeEmbedder, FakeLLM

V1_CHUNK = ("Employees may work remotely for up to 3 days per calendar week. "
            "The remaining days must be spent at the office.")
V2_CHUNK = ("Employees must be onsite at least 4 days per calendar week. "
            "Remote work is limited to 1 day per week.")


class AgentTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "a.db")
        emb = FakeEmbedder()
        for doc, text in (("remote_work_v1", V1_CHUNK), ("remote_work_v2", V2_CHUNK)):
            did = self.db.add_document(doc, f"{doc}.txt", "txt", 10, "ready")
            self.db.add_chunk(did, 0, "2. Allowance", f"{doc} > 2. Allowance",
                              text, emb.embed_documents([text])[0])
        self.registry = ToolRegistry(self.db, RAGEngine(self.db, emb))

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def tool_messages(self, fake: FakeLLM):
        """Tool-role observations accumulated in the LAST request the agent sent."""
        if not fake.received:
            return []
        return [m.get("content", "") for m in fake.received[-1]["messages"]
                if m.get("role") == "tool"]


class TestReActLoop(AgentTestBase):
    def test_repair_feedback_and_citations(self):
        fake = FakeLLM([
            # 1. malformed args (single quotes + trailing comma) -> must be repaired
            LLMResponse(content="", tool_calls=[ToolCall(
                "c1", "search_policy_docs",
                "{'query': 'remote work days per week', 'top_k': 3,}")]),
            # 2. invalid args (missing required param) -> error feedback
            LLMResponse(content="", tool_calls=[ToolCall(
                "c2", "search_policy_docs", '{"top_k": 3}')]),
            # 3. final grounded answer with a valid citation
            LLMResponse(content=(
                "Employees may work remotely up to 3 days per week "
                "[remote_work_v1 § 2. Allowance].\n\nSources:\n"
                "[remote_work_v1 § 2. Allowance]")),
        ])
        agent = ReactAgent(fake, self.registry, max_steps=6)
        result = agent.run("How many days can I work remotely?")

        self.assertIn("3 days", result.answer)
        self.assertEqual(result.tool_calls, 1)  # second call failed validation
        self.assertTrue(result.citations)
        self.assertEqual(result.citation_warnings, [])

        tool_msgs = self.tool_messages(fake)
        self.assertEqual(len(tool_msgs), 2)
        self.assertIn("[chunk ", tool_msgs[0])            # repaired call executed
        self.assertIn("TOOL_CALL_ERROR", tool_msgs[1])    # broken call fed back
        self.assertIn("missing required parameter 'query'", tool_msgs[1])

        # schema repair must be recorded in the trace
        repairs = [s.get("schema_repairs") for s in result.steps
                   if s["type"] == "tool_call"]
        self.assertTrue(repairs and repairs[0])

    def test_unknown_tool_feedback(self):
        fake = FakeLLM([
            LLMResponse(content="", tool_calls=[ToolCall("c1", "destroy_database", "{}")]),
            LLMResponse(content="I only have read-only tools available."),
        ])
        result = ReactAgent(fake, self.registry).run("delete everything")
        self.assertIn("unknown tool 'destroy_database'",
                      self.tool_messages(fake)[0])
        self.assertEqual(result.tool_calls, 0)

    def test_text_fallback_tool_calling(self):
        fake = FakeLLM([
            LLMResponse(content=(
                'Let me search.\n<tool_call>{"name": "search_policy_docs", '
                '"arguments": {"query": "onsite requirement"}}</tool_call>')),
            LLMResponse(content=(
                "Employees must be onsite at least 4 days per week "
                "[remote_work_v2 § 2. Allowance].")),
        ])
        result = ReactAgent(fake, self.registry).run("office requirement?")
        self.assertEqual(result.tool_calls, 1)
        self.assertIn("4 days", result.answer)
        self.assertTrue(result.citations)

    def test_citation_warning_for_hallucinated_doc(self):
        fake = FakeLLM([
            LLMResponse(content=(
                "Per the handbook everyone gets 42 days of leave "
                "[mystery_handbook § 9].")),
        ])
        result = ReactAgent(fake, self.registry).run("leave days?")
        self.assertEqual(result.citation_warnings,
                         ["citation 'mystery_handbook § 9' does not match any "
                          "ingested document"])

    def test_citation_warning_for_bad_chunk_id(self):
        fake = FakeLLM([
            LLMResponse(content="Remote work is allowed 3 days "
                                "[remote_work_v1 #99999]."),
        ])
        result = ReactAgent(fake, self.registry).run("remote?")
        self.assertTrue(any("unknown chunk id 99999" in w
                            for w in result.citation_warnings))

    def test_llm_error_surfaces(self):
        from app.llm import LLMError
        fake = FakeLLM([LLMError("endpoint down")])
        result = ReactAgent(fake, self.registry).run("anything?")
        self.assertEqual(result.error, "endpoint down")
        self.assertEqual(result.answer, "")

    def test_registry_default_top_k_follows_config(self):
        emb = FakeEmbedder()
        reg = ToolRegistry(self.db, RAGEngine(self.db, emb), default_top_k=1)
        out = reg.execute("search_policy_docs", {"query": "remote work"})
        self.assertIn("Found 1 passages", out)


class TestCitationParsing(unittest.TestCase):
    """Bracketed prose is not a citation."""

    def test_markdown_links_and_footnotes_are_ignored(self):
        from app.agent.react import check_citations
        cits, warns = check_citations(
            "See [the portal](http://x) and [1] and [2].", ["policy_a"],
            lambda cid: True)
        self.assertEqual(cits, [])
        self.assertEqual(warns, [])

    def test_real_citations_still_resolve_and_warn(self):
        from app.agent.react import check_citations
        cits, warns = check_citations(
            "[policy_a § 2. Leave] and [ghost_doc § 9]", ["policy_a"],
            lambda cid: True)
        self.assertEqual(cits, ["policy_a § 2. Leave"])
        self.assertEqual(len(warns), 1)
        self.assertIn("ghost_doc", warns[0])

    def test_repeated_citations_are_deduplicated(self):
        from app.agent.react import check_citations
        cits, _ = check_citations("[policy_a § 2] then [policy_a § 2]",
                                  ["policy_a"], lambda cid: True)
        self.assertEqual(cits, ["policy_a § 2"])


if __name__ == "__main__":
    unittest.main()
