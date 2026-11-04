"""Document-name resolution: unique candidate or nothing.

The regression these tests exist for: with `remote_work_policy_v1` and
`remote_work_policy_v2` both stored, a citation of "Remote Work Policy v2"
resolved to **v1** - first substring/difflib match in list order. Those two
documents are the corpus's contradiction pair, so a wrong attribution is not a
cosmetic error.
"""
import tempfile
import unittest
from pathlib import Path

from app.agent.react import check_citations
from app.db import Database
from app.naming import Match, resolve, resolve_one

NAMES = ["remote_work_policy_v1", "remote_work_policy_v2",
         "leave_policy", "leave_policy_2025_amendment",
         "expense_policy", "health_plans"]


class TestResolution(unittest.TestCase):
    def test_human_readable_version_resolves_to_the_right_document(self):
        self.assertEqual(resolve_one("Remote Work Policy v2", NAMES),
                         "remote_work_policy_v2")
        self.assertEqual(resolve_one("Remote Work Policy v1", NAMES),
                         "remote_work_policy_v1")

    def test_separators_case_and_extension_are_noise(self):
        for ref in ("remote_work_policy_v2", "REMOTE-WORK-POLICY-V2",
                    "Remote Work Policy V2.md", "remote.work.policy.v2.txt"):
            self.assertEqual(resolve_one(ref, NAMES), "remote_work_policy_v2", ref)

    def test_family_reference_is_ambiguous_not_a_coin_flip(self):
        match = resolve("Remote Work Policy", NAMES)
        self.assertFalse(match)
        self.assertEqual(match.how, "ambiguous")
        self.assertIn("remote_work_policy_v1", match.reason)

    def test_unique_subset_still_resolves(self):
        self.assertEqual(resolve_one("expense", NAMES), "expense_policy")

    def test_unrelated_reference_is_none(self):
        match = resolve("ghost_policy", NAMES)
        self.assertFalse(match)
        self.assertEqual(match.how, "none")

    def test_fuzzy_only_with_a_clear_margin(self):
        # one typo, no competing candidate -> fuzzy is acceptable
        self.assertEqual(resolve_one("expense_polcy", NAMES), "expense_policy")
        # two equally plausible candidates -> ambiguous, never the first one
        match = resolve("remote work polic", ["remote_work_policy_a",
                                             "remote_work_policy_b"])
        self.assertFalse(match)
        self.assertEqual(match.how, "ambiguous")

    def test_empty_input(self):
        self.assertFalse(resolve("", NAMES))
        self.assertFalse(resolve("leave", []))


class TestCitationIntegration(unittest.TestCase):
    def test_v2_citation_is_resolved_as_v2(self):
        answer = "Band 3 gets 2 days [Remote Work Policy v2 § 2. Remote Work Allowance]."
        cites, warns = check_citations(answer, NAMES, lambda cid: True)
        self.assertEqual(cites, ["Remote Work Policy v2 § 2. Remote Work Allowance"])
        self.assertEqual(warns, [])

    def test_ambiguous_citation_is_warned_not_attributed(self):
        answer = "Remote days differ [Remote Work Policy § 2. Allowance]."
        cites, warns = check_citations(answer, NAMES, lambda cid: True)
        self.assertEqual(cites, [])
        self.assertTrue(any("does not match any ingested document" in w for w in warns))


class TestDatabaseLookup(unittest.TestCase):
    def test_find_document_fuzzy_prefers_the_named_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "n.db")
            v1 = db.add_document("remote_work_policy_v1", "v1.txt", "txt", status="ready")
            v2 = db.add_document("remote_work_policy_v2", "v2.txt", "txt", status="ready")
            self.assertEqual(db.find_document_fuzzy("Remote Work Policy v2")["id"], v2)
            self.assertEqual(db.find_document_fuzzy("remote_work_policy_v1")["id"], v1)
            self.assertIsNone(db.find_document_fuzzy("Remote Work Policy"))
            db.close()


if __name__ == "__main__":
    unittest.main()
