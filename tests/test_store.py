import tempfile
import unittest
from pathlib import Path

from app.db import Database
from app.rag import RAGEngine
from tests.fakes import FakeEmbedder


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "test.db")
        self.embedder = FakeEmbedder()
        self.rag = RAGEngine(self.db, self.embedder)
        self.doc_a = self.db.add_document("policy_a", "a.txt", "txt", 100, "ready")
        self.doc_b = self.db.add_document("policy_b", "b.md", "md", 100, "ready")
        for i, text in enumerate([
            "Employees may work remotely three days per week with approval",
            "Expense receipts must be submitted within thirty days",
        ]):
            self.db.add_chunk(self.doc_a, i, "S1", "policy_a > S1", text,
                              self.embedder.embed_documents([text])[0])
        for i, text in enumerate([
            "Employees must be onsite four days per week minimum",
        ]):
            self.db.add_chunk(self.doc_b, i, "S2", "policy_b > S2", text,
                              self.embedder.embed_documents([text])[0])

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_vector_search_ranks_relevant_first(self):
        hits = self.rag.search("remote work days per week", top_k=3)
        self.assertEqual(len(hits), 3)
        self.assertEqual(hits[0].doc_name, "policy_a")
        self.assertIn("remote", hits[0].text.lower())

    def test_doc_filters(self):
        only_b = self.rag.search("onsite office", top_k=5, doc_ids=[self.doc_b])
        self.assertTrue(only_b)
        self.assertTrue(all(h.doc_id == self.doc_b for h in only_b))
        excl = self.rag.search("anything", top_k=5, exclude_doc_ids=[self.doc_a])
        self.assertTrue(all(h.doc_id == self.doc_b for h in excl))

    def test_unique_names(self):
        self.assertEqual(self.db.unique_name("policy_a"), "policy_a (2)")
        self.assertEqual(self.db.unique_name("new_doc"), "new_doc")

    def test_contradictions_crud(self):
        cid = self.db.add_contradiction(
            1, "policy_a", "S1", "text a", 2, "policy_b", "S2", "text b",
            gate_score=0.7, gate_signals=["numeric conflict"], verified=True,
            confidence=0.9, conflict_type="direct", explanation="why",
            points=[{"topic": "t"}])
        self.assertTrue(self.db.contradiction_pair_exists(1, 2))
        self.assertTrue(self.db.contradiction_pair_exists(2, 1))
        flags = self.db.list_contradictions("OPEN")
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["gate_signals"], ["numeric conflict"])
        self.assertEqual(flags[0]["points"], [{"topic": "t"}])
        self.assertTrue(self.db.set_contradiction_status(cid, "resolved"))
        self.assertEqual(self.db.list_contradictions("OPEN"), [])
        self.assertEqual(len(self.db.list_contradictions()), 1)

    def test_pair_dedup_is_order_agnostic(self):
        """A flag stored as (bigger, smaller) - which is what scanning the
        newer document produces - must still block the reversed re-scan."""
        cid = self.db.add_contradiction(
            3, "policy_b", "S2", "text b", 1, "policy_a", "S1", "text a",
            gate_score=0.7, gate_signals=["numeric conflict"], verified=True,
            confidence=0.9, conflict_type="direct", explanation="why", points=[])
        self.assertTrue(self.db.contradiction_pair_exists(1, 3))
        self.assertTrue(self.db.contradiction_pair_exists(3, 1))
        again = self.db.add_contradiction(
            1, "policy_a", "S1", "text a", 3, "policy_b", "S2", "text b",
            gate_score=0.7, gate_signals=[], verified=True,
            confidence=0.9, conflict_type="direct", explanation="dupe", points=[])
        self.assertEqual(again, 0, "reverse-order insert must not create a flag")
        flags = self.db.list_contradictions()
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["id"], cid)
        self.assertEqual(flags[0]["explanation"], "why")

    def test_legacy_duplicate_flags_are_migrated(self):
        """A database written before the canonical-pair index existed is
        deduped (oldest row kept) when it is opened."""
        import sqlite3
        legacy = Path(self.tmp.name) / "legacy.db"
        conn = sqlite3.connect(legacy)
        conn.executescript(
            "CREATE TABLE contradictions("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " chunk_a_id INTEGER, doc_a TEXT, section_a TEXT, text_a TEXT,"
            " chunk_b_id INTEGER, doc_b TEXT, section_b TEXT, text_b TEXT,"
            " gate_score REAL DEFAULT 0, gate_signals TEXT DEFAULT '[]',"
            " verified INTEGER DEFAULT 0, confidence REAL DEFAULT 0,"
            " conflict_type TEXT DEFAULT '', explanation TEXT DEFAULT '',"
            " points_json TEXT DEFAULT '[]', status TEXT DEFAULT 'OPEN',"
            " detected_at TEXT DEFAULT '');"
            "INSERT INTO contradictions(chunk_a_id, chunk_b_id)"
            " VALUES (2,1),(1,2),(5,9);"
        )
        conn.commit()
        conn.close()

        db = Database(legacy)  # opening runs the migration
        try:
            rows = db.list_contradictions()
            self.assertEqual(len(rows), 2)
            self.assertEqual({(r["chunk_a_id"], r["chunk_b_id"]) for r in rows},
                             {(2, 1), (5, 9)})
            self.assertEqual(db.add_contradiction(
                1, "a", "s", "t", 2, "b", "s", "t", 0.6, [], True, 0.9,
                "direct", "dupe", []), 0)
            self.assertEqual(len(db.list_contradictions()), 2)
        finally:
            db.close()

    def test_delete_cascades(self):
        self.db.add_contradiction(
            1, "policy_a", "S1", "a", 3, "policy_b", "S2", "b",
            0.6, ["sig"], True, 0.8, "direct", "why", [])
        self.assertTrue(self.db.delete_document(self.doc_a))
        self.assertEqual(self.db.count_chunks(self.doc_a), 0)
        self.assertEqual(self.db.list_contradictions(), [])
        self.assertIsNone(self.db.get_document("policy_a"))
        # search still works over remaining doc
        hits = self.rag.search("onsite", top_k=5)
        self.assertTrue(all(h.doc_name == "policy_b" for h in hits))

    def test_jobs_progress(self):
        jid = self.db.create_job("ingest", self.doc_a, "policy_a", total=5)
        self.db.update_job(jid, done=2, stage="parsed window 2/5")
        job = self.db.get_job(jid)
        self.assertEqual(job["done"], 2)
        self.assertEqual(job["status"], "running")
        self.db.update_job(jid, status="done")
        self.assertEqual(self.db.get_job(jid)["status"], "done")
        self.assertEqual(self.db.latest_job_for_doc(self.doc_a)["id"], jid)

    def test_embedding_dim_mismatch_is_actionable(self):
        """Switching embedder must fail with a message, not a matmul error."""
        self.db.set_embed_fingerprint("embeddinggemma", 4096)
        with self.assertRaises(RuntimeError) as ctx:
            self.db.vector_search([0.1] * 384, top_k=1)
        msg = str(ctx.exception)
        self.assertIn("dimension mismatch", msg)
        self.assertIn("embeddinggemma", msg)

    def test_matching_dimension_still_searches(self):
        self.db.set_embed_fingerprint("fake", 4096)
        hits = self.db.vector_search([0.1] * 4096, top_k=2)
        self.assertEqual(len(hits), 2)

    def test_metadata_roundtrip(self):
        self.db.set_meta("k", "v1")
        self.db.set_meta("k", "v2")
        self.assertEqual(self.db.get_meta("k"), "v2")
        self.assertEqual(self.db.get_meta("missing", "fallback"), "fallback")


if __name__ == "__main__":
    unittest.main()
