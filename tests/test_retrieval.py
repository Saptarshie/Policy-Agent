"""Retrieval: FTS5 keyword index, RRF fusion, and the safety rails around them."""
import tempfile
import unittest
from pathlib import Path

from app.db import Database
from app.rag import MODES, RAGEngine
from tests.fakes import FakeEmbedder


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "t.db"
        self.db = Database(self.path)
        self.emb = FakeEmbedder()
        self.doc_a = self.db.add_document("health_plans", "h.md", "md", 10, "ready")
        self.doc_b = self.db.add_document("travel_policy", "t.md", "md", 10, "ready")
        self.texts = {
            "tier_table": ("health_plans", "Out-of-pocket maximum $6,000 $5,000 $4,000"),
            "per_diem": ("travel_policy", "The per diem for domestic travel is 60 USD per day"),
        }
        self.ids = {}
        for i, (key, (doc_name, text)) in enumerate(self.texts.items()):
            doc_id = self.doc_a if doc_name == "health_plans" else self.doc_b
            self.ids[key] = self.db.add_chunk(
                doc_id, i, "S1", "%s > S1" % doc_name, text,
                self.emb.embed_documents([text])[0])

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()


class TestKeywordIndex(_Base):
    def test_exact_token_is_found(self):
        hits = self.db.keyword_search("$4,000 out-of-pocket maximum", top_k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["chunk_id"], self.ids["tier_table"])

    def test_scores_are_sorted_desc(self):
        hits = self.db.keyword_search("per diem travel", top_k=5)
        scores = [h["score"] for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_hostile_query_does_not_raise(self):
        # FTS5 MATCH has its own grammar; these used to be able to blow up
        for q in ['"unbalanced', "AND OR NOT", "a * b", "NEAR(x y)", "(parens)",
                  "100%", "!!!", "", "  "]:
            self.assertIsInstance(self.db.keyword_search(q, top_k=3), list)

    def test_single_character_tokens_are_dropped(self):
        # FTS5 has no stopword list, so "a"/"I" would match nearly every chunk
        self.assertEqual(Database.build_match_query("a I"), "")
        self.assertEqual(self.db.keyword_search("a I", top_k=3), [])

    def test_absent_terms_return_nothing(self):
        self.assertEqual(self.db.keyword_search("zzzqqq wwwvvv", top_k=3), [])

    def test_index_follows_delete(self):
        self.assertFalse(self.db.keyword_search("zzzuniqueword", top_k=3))
        cid = self.db.add_chunk(self.doc_a, 9, "S9", "b", "zzzuniqueword appears here",
                                self.emb.embed_documents(["x"])[0])
        self.assertTrue(self.db.keyword_search("zzzuniqueword", top_k=3))
        self.db.delete_document(self.doc_a) if hasattr(self.db, "delete_document") else None
        self.db.conn.execute("DELETE FROM chunks WHERE id=?", (cid,))
        self.db.conn.commit()
        self.assertEqual(self.db.keyword_search("zzzuniqueword", top_k=3), [])

    def test_index_follows_update(self):
        cid = self.ids["per_diem"]
        self.db.conn.execute("UPDATE chunks SET text=? WHERE id=?",
                             ("The replacement text mentions kryptonite", cid))
        self.db.conn.commit()
        self.assertTrue(self.db.keyword_search("kryptonite", top_k=3))
        self.assertEqual(self.db.keyword_search("diem", top_k=3), [])


class TestBackfill(_Base):
    def test_empty_index_is_rebuilt_on_open(self):
        # simulate a store written before the FTS table existed
        self.db.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
        self.db.conn.commit()
        self.assertEqual(self.db.keyword_search("kryptonitex", top_k=3), [])
        self.assertEqual(
            self.db.conn.execute("SELECT COUNT(*) FROM chunks_fts_docsize").fetchone()[0], 0)
        self.db.close()
        self.db = Database(self.path)          # reopen -> backfill runs
        self.assertGreater(
            self.db.conn.execute("SELECT COUNT(*) FROM chunks_fts_docsize").fetchone()[0], 0)
        self.assertTrue(self.db.keyword_search("diem", top_k=3))


class TestModes(_Base):
    def test_keyword_mode_needs_no_embedder(self):
        class Boom:
            def embed_query(self, q):
                raise AssertionError("keyword mode must not embed the query")

        hits = RAGEngine(self.db, Boom(), mode="keyword").search("per diem", top_k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk_id, self.ids["per_diem"])

    def test_hybrid_returns_both_ranks(self):
        hits = RAGEngine(self.db, self.emb, mode="hybrid").search("per diem", top_k=2)
        self.assertTrue(hits)
        top = hits[0]
        self.assertEqual(top.chunk_id, self.ids["per_diem"])
        self.assertIsNotNone(top.vector_rank)
        self.assertIsNotNone(top.keyword_rank)

    def test_fusion_prefers_agreement(self):
        eng = RAGEngine(self.db, self.emb, mode="hybrid")
        # chunk 1 is 1st for both retrievers, chunk 2 is 1st for one and last
        # for the other -> agreement must win
        fused = eng._fuse([{"chunk_id": 1}, {"chunk_id": 2}],
                          [{"chunk_id": 1}, {"chunk_id": 3}], top_k=3)
        self.assertEqual(fused[0][0], 1)
        self.assertEqual({c for c, *_ in fused}, {1, 2, 3})

    def test_unknown_mode_falls_back_to_vector(self):
        eng = RAGEngine(self.db, self.emb, mode="nonsense")
        self.assertIn("vector", MODES)
        hits = eng.search("per diem", top_k=1)
        self.assertTrue(hits)

    def test_empty_query_returns_nothing_in_every_mode(self):
        for m in MODES:
            self.assertEqual(RAGEngine(self.db, self.emb, mode=m).search("   "), [])


if __name__ == "__main__":
    unittest.main()
