"""The document brief: what it is for, and the two things it must never do.

A brief exists so the triage layer (and the agent) can tell which documents exist
and what vocabulary they use. It must NOT become an input to chunking, and it must
NOT become a reason to refuse: a digest omits almost everything, so "the brief does
not mention X" says nothing about whether the document covers X.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.agent.intent import IntentCache, IntentRouter, format_corpus
from app.db import Database
from app.ingest import IngestPipeline
from app.llm import LLMResponse
from app.prompts import INTENT_SYSTEM, INTENT_SYSTEM_V2
from tests.fakes import FakeEmbedder, FakeLLM

DOC = ("ACME Remote Work Policy\n\n"
       "1. Allowance\n\nEmployees may work remotely for up to 3 days per calendar "
       "week.\n\n2. Notice\n\nEmployees must give at least 15 days written notice.\n")

WINDOW_JSON = json.dumps({"window_title": "Allowance",
                          "statements": ["may work remotely 3 days"],
                          "key_terms": ["remote work"],
                          "numbers": [{"value": "3", "unit": "days",
                                       "context": "remote work"}]})


class TestCorpusBlock(unittest.TestCase):
    def test_block_is_bounded_and_labelled_as_a_digest(self):
        corpus = [{"name": f"DOC_{i}", "brief": "x" * 1000} for i in range(20)]
        block, fp = format_corpus(corpus)
        self.assertLess(len(block), 8 * 260 + 200)
        self.assertEqual(block.count("\n- "), 8, block)      # eight documents max
        self.assertIn("NOT the document", block)
        self.assertTrue(fp)

    def test_a_missing_brief_is_said_out_loud(self):
        block, _ = format_corpus([{"name": "A", "brief": ""}])
        self.assertIn("(no brief available)", block)

    def test_empty_corpus_still_routes_document_questions(self):
        block, fp = format_corpus([])
        self.assertIn("no documents ingested", block)
        self.assertEqual(fp, "empty")

    def test_prompt_forbids_refusing_on_a_brief(self):
        for prompt in (INTENT_SYSTEM, INTENT_SYSTEM_V2):
            self.assertIn("NEVER refuse because a brief does not mention the topic", prompt)
            self.assertIn("NEVER answer from a brief", prompt)

    def test_uploading_a_document_invalidates_cached_verdicts(self):
        # the classifier is shown the corpus, so the same question can be classified
        # differently after a document is added: the cache key must move with it
        _, before = format_corpus([{"name": "A", "brief": "leave rules"}])
        _, after = format_corpus([{"name": "A", "brief": "leave rules"},
                                  {"name": "B", "brief": "service credits"}])
        k1 = IntentCache.key("m", "v1", "What is the cap?", "fp", before)
        k2 = IntentCache.key("m", "v1", "What is the cap?", "fp", after)
        self.assertNotEqual(k1, k2)


class _BriefLLM:
    """Answers both the window parse and the brief call, records the prompts."""

    def __init__(self, brief):
        self.brief = brief
        self.brief_prompts = []

    def chat(self, *a, **kw):
        return LLMResponse(content="")

    def chat_json(self, system, user, **kw):
        if "one-paragraph briefs" in system:
            self.brief_prompts.append(user)
            return ({"brief": self.brief} if self.brief is not None else None), None
        return json.loads(WINDOW_JSON), None


class TestBriefIsNotChunkInput(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "p.txt"
        self.path.write_text(DOC, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def _ingest(self, brief):
        db = Database(self.dir / f"{abs(hash(brief))}.db")
        llm = _BriefLLM(brief)
        pipe = IngestPipeline(db, FakeEmbedder(), llm=llm, window_chars=400,
                              use_llm=True, chunk_target=300, chunk_max=450)
        res = pipe.ingest(self.path, "p")
        chunks = [c["text"] for c in db.list_chunks(res["doc_id"])]
        doc = db.get_document(res["doc_id"])
        db.close()
        return chunks, doc["summary"]

    def test_chunk_text_is_identical_whatever_the_brief_says(self):
        a, sum_a = self._ingest("Leave, notice and remote-work allowances.")
        b, sum_b = self._ingest("Completely different wording about travel credits.")
        self.assertEqual(a, b, "the brief must not influence chunk text")
        self.assertIn("Leave", sum_a)
        self.assertIn("travel", sum_b)

    def test_brief_prompt_carries_headings_and_the_opening(self):
        db = Database(self.dir / "c.db")
        llm = _BriefLLM("x")
        pipe = IngestPipeline(db, FakeEmbedder(), llm=llm, window_chars=400,
                              use_llm=True, chunk_target=300, chunk_max=450,
                              brief_words=50)
        pipe.ingest(self.path, "p")
        prompt = llm.brief_prompts[0]
        self.assertIn("1. Allowance", prompt)
        self.assertIn("2. Notice", prompt)
        self.assertIn("ACME Remote Work Policy", prompt)
        self.assertNotIn("15 days written notice. 2. Notice", prompt[:0])  # noop
        db.close()

    def test_failed_brief_falls_back_to_the_opening_and_warns(self):
        db = Database(self.dir / "d.db")
        pipe = IngestPipeline(db, FakeEmbedder(), llm=_BriefLLM(None), window_chars=400,
                              use_llm=True, chunk_target=300, chunk_max=450)
        res = pipe.ingest(self.path, "p")
        doc = db.get_document(res["doc_id"])
        self.assertIn("ACME Remote Work Policy", doc["summary"])
        self.assertTrue(any("brief unavailable" in w for w in res["warnings"]),
                        res["warnings"])
        db.close()


class TestRouterSendsTheCorpus(unittest.TestCase):
    def test_classify_puts_the_block_in_front_of_the_question(self):
        seen = {}

        class _LLM:
            def chat_json(self, system, user, **kw):
                seen["user"] = user
                return {"intent": "POLICY_DIRECT_LOOKUP", "de_noised_question": "q",
                        "search_queries": ["q"]}, None

        with tempfile.TemporaryDirectory() as tmp:
            r = IntentRouter(llm=_LLM(), cache_path=str(Path(tmp) / "c.json"))
            r.classify("How many remote days?",
                       corpus=[{"name": "POLICY", "brief": "remote work, 3 days"}])
        self.assertIn("CORPUS (1 document(s) ingested", seen["user"])
        self.assertIn("POLICY: remote work, 3 days", seen["user"])
        self.assertIn("QUESTION:", seen["user"])


if __name__ == "__main__":
    unittest.main()
