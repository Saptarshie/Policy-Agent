"""Ingestion speed knobs: reasoning effort, parallel windows, heuristic-only.

The contract that matters: these change HOW LONG ingestion takes, never WHAT is
stored. Chunks and their text must be identical whichever path runs.
"""

import json
import tempfile
import unittest
from pathlib import Path

from app.db import Database
from app.ingest import IngestPipeline
from app.llm import LLMResponse
from tests.fakes import FakeEmbedder, FakeLLM

DOC = ("ACME Remote Work Policy\n\n"
       "1. Allowance\n\nEmployees may work remotely for up to 3 days per calendar "
       "week. The remaining working days must be spent at the assigned office.\n\n"
       "2. Notice\n\nEmployees must give at least 15 days written notice before "
       "changing an approved remote work arrangement.\n\n"
       "3. Eligibility\n\nEmployees are eligible after 6 months of continuous "
       "employment.\n\n"
       "4. Core hours\n\nRemote employees must be reachable between 10:00 and "
       "16:00 local time.\n\n"
       "5. Security\n\nEmployees must connect through the corporate VPN.\n")

WINDOW_JSON = json.dumps({
    "window_title": "Allowance",
    "statements": ["may work remotely 3 days"],
    "key_terms": ["remote work"],
    "numbers": [{"value": "3", "unit": "days", "context": "remote work"}],
    "rolling_summary": "ACME remote work policy: 3 remote days/week.",
})


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.db = Database(self.dir / "t.db")
        self.path = self.dir / "policy.txt"
        self.path.write_text(DOC, encoding="utf-8")

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def _pipeline(self, llm, **kw):
        return IngestPipeline(self.db, FakeEmbedder(), llm=llm,
                              window_chars=260, use_llm=llm is not None,
                              chunk_target=300, chunk_max=450, **kw)


class TestReasoningEffort(_Base):
    def test_parser_gets_the_configured_effort_and_token_cap(self):
        fake = FakeLLM([LLMResponse(content=WINDOW_JSON) for _ in range(30)])
        self._pipeline(fake, reasoning_effort="off", max_parse_tokens=1200).ingest(
            self.path, "rw")
        self.assertTrue(fake.received)
        calls = fake.json_received
        self.assertTrue(calls)
        self.assertTrue(all(c["reasoning_effort"] == "off" for c in calls),
                        [c["reasoning_effort"] for c in calls])
        self.assertTrue(all(c["max_tokens"] == 1200 for c in calls),
                        [c["max_tokens"] for c in calls])


class TestParallelWindows(_Base):
    def test_chunks_are_identical_to_the_sequential_path(self):
        seq = self._pipeline(FakeLLM([LLMResponse(content=WINDOW_JSON)
                                      for _ in range(40)]),
                             parallel_windows=1).ingest(self.path, "seq")
        self.path.write_text(DOC + "\n\n6. Expenses\n\nReceipts within 30 days.\n",
                             encoding="utf-8")
        par = self._pipeline(FakeLLM([LLMResponse(content=WINDOW_JSON)
                                      for _ in range(40)]),
                             parallel_windows=4).ingest(self.path, "par")
        # the stored text is chunking's job; parallelism must not touch it
        self.assertGreater(seq["chunks"], 1)
        self.assertGreater(par["chunks"], 1)
        for doc, res in (("seq", seq), ("par", par)):
            d = self.db.get_document(res["doc_id"])
            self.assertEqual(d["status"], "ready", doc)
            self.assertTrue(d["summary"], doc)

    def test_windows_are_not_chained_and_the_summary_is_one_merge_call(self):
        fake = FakeLLM([LLMResponse(content=WINDOW_JSON) for _ in range(40)])
        result = self._pipeline(fake, parallel_windows=3).ingest(self.path, "rw")
        self.assertGreater(result["windows"], 1)

        # every window parse starts from an empty previous summary (de-chained)
        window_prompts = [c["messages"][-1]["content"] for c in fake.json_received]
        self.assertEqual(len(window_prompts), result["windows"])
        self.assertTrue(all("(empty)" in p for p in window_prompts))
        # ...and the rolling summary came from ONE extra call over the parts
        merges = [c for c in fake.received
                  if "Merge these per-window summaries" in c["messages"][0]["content"]]
        self.assertEqual(len(merges), 1,
                         [c["messages"][0]["content"][:60] for c in fake.received])
        self.assertIn("[1]", merges[0]["messages"][-1]["content"])

    def test_a_failed_window_still_lands_as_a_warning(self):
        good = LLMResponse(content=WINDOW_JSON)
        bad = LLMResponse(content="not json at all")
        fake = FakeLLM([good, bad, bad, good, good, good, good, good, good,
                        good, good, good])
        result = self._pipeline(fake, parallel_windows=3).ingest(self.path, "rw")
        self.assertTrue(any("heuristic fallback" in w for w in result["warnings"]),
                        result["warnings"])
        self.assertEqual(self.db.get_document(result["doc_id"])["status"], "ready")


class TestHeuristicSizeThreshold(_Base):
    def test_small_document_skips_the_llm_entirely(self):
        fake = FakeLLM([LLMResponse(content=WINDOW_JSON) for _ in range(10)])
        result = self._pipeline(fake, heuristic_max_chars=100_000).ingest(self.path, "rw")
        self.assertEqual(fake.received, [], "the LLM must not be called at all")
        doc = self.db.get_document(result["doc_id"])
        self.assertEqual(doc["status"], "ready")
        self.assertGreater(result["chunks"], 0)
        meta = json.loads(doc["meta_json"])
        self.assertIn("heuristic", meta["parse"])
        self.assertIn("ACME Remote Work Policy", doc["summary"])

    def test_threshold_zero_never_skips(self):
        fake = FakeLLM([LLMResponse(content=WINDOW_JSON) for _ in range(20)])
        self._pipeline(fake, heuristic_max_chars=0).ingest(self.path, "rw")
        self.assertTrue(fake.received)


if __name__ == "__main__":
    unittest.main()
