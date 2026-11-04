"""Process restart: an abandoned ingestion must not leave a document stuck."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.admin import AdminAPI
from app.db import Database
from tests.fakes import FakeEmbedder
from tests.test_agent_admin import make_config


class TestStaleJobReconciliation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.cfg = make_config(self.dir / "data")

    def tearDown(self):
        for db in getattr(self, "_dbs", []):
            try:
                db.close()
            except Exception:
                pass
        self.tmp.cleanup()

    def _track(self, db):
        self._dbs = getattr(self, "_dbs", []) + [db]
        return db

    def _db_with_job(self, age_minutes: int):
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        db = Database(self.cfg.db_path)
        doc_id = db.add_document("stuck_policy", "f.md", "md", 10, "ingesting")
        job_id = db.create_job("ingest", doc_id, "stuck_policy", total=5)
        stamp = (datetime.now(timezone.utc)
                 - timedelta(minutes=age_minutes)).isoformat(timespec="seconds")
        db.conn.execute("UPDATE jobs SET status='running', updated_at=?, created_at=?"
                        " WHERE id=?", (stamp, stamp, job_id))
        db.conn.commit()
        self._track(db)
        return db, doc_id, job_id

    def test_abandoned_job_is_marked_error_at_startup(self):
        db, doc_id, job_id = self._db_with_job(age_minutes=180)
        api = AdminAPI(config=self.cfg, db=db, embedder=FakeEmbedder(), llm=None)
        job = api.job_status(job_id)
        self.assertEqual(job["status"], "error")
        self.assertIn("interrupted", job["error"])
        self.assertEqual(api.db.get_document(doc_id)["status"], "error")

    def test_a_live_job_is_left_alone(self):
        db, doc_id, job_id = self._db_with_job(age_minutes=1)
        api = AdminAPI(config=self.cfg, db=db, embedder=FakeEmbedder(), llm=None)
        self.assertEqual(api.job_status(job_id)["status"], "running")
        self.assertEqual(api.db.get_document(doc_id)["status"], "ingesting")

    def test_reconciliation_is_idempotent(self):
        db, doc_id, job_id = self._db_with_job(age_minutes=180)
        first = db.reconcile_stale_jobs(30)
        second = db.reconcile_stale_jobs(30)
        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])


if __name__ == "__main__":
    unittest.main()
