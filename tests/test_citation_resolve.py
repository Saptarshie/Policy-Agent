"""Citations have to resolve to real text, or say why they do not.

A chip that opens the wrong passage is worse than a chip that does nothing: it looks
like verification. These tests pin the contract - forgiving on the surface form the
model produces, never inventing a passage - and the endpoint the UI calls.
"""

import unittest

from app.citations import CitationResolver, parse_citation
from app.db import Database

DOC = ("# Northstar Benefits\n\n## 4.2 Dental implants\n\n"
       "| Tier | Cover |\n|---|---|\n| Enhanced | 50% up to Rs 25,000 per implant |\n\n"
       "Pre-authorization is required for every implant.\n\n"
       "## Example A: Enhanced dental implant\n\n"
       "Enhanced covers 50%, which is Rs 35,000, but the per-implant maximum is "
       "Rs 25,000.\n")


class TestParse(unittest.TestCase):
    def test_section_form(self):
        doc, chunk, section = parse_citation("[02_BENEFITS § 4.2 Dental implants]")
        self.assertEqual(doc, "02_BENEFITS")
        self.assertIsNone(chunk)
        self.assertEqual(section, "4.2 Dental implants")

    def test_chunk_id_form(self):
        doc, chunk, section = parse_citation("02_BENEFITS #17")
        self.assertEqual((doc, chunk, section), ("02_BENEFITS", 17, ""))


class _Base(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(f"{self.tmp.name}/t.db")
        self.doc_id = self.db.add_document(name="02_HEALTH_AND_WELLBEING_BENEFITS",
                                          fmt="md", status="ready")
        for i, (section, text) in enumerate([
                ("4.2 Dental implants", "Enhanced covers 50% up to Rs 25,000."),
                ("Example A: Enhanced dental implant", "The payable amount is Rs 25,000.")]):
            self.db.add_chunk(doc_id=self.doc_id, index=i, section=section,
                              breadcrumb=f"doc > {section}", text=text,
                              embedding=[0.0] * 8)
        # the resolver needs only a .db - no model, no config, no second handle
        class _Shim:
            db = self.db

        self.api = _Shim()
        self.api.resolve_citation = lambda cite: CitationResolver.resolve_citation(
            self.api, cite)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()


class TestResolve(_Base):
    def test_exact_citation_returns_its_passage(self):
        r = self.api.resolve_citation("[02_HEALTH_AND_WELLBEING_BENEFITS § 4.2 Dental implants]")
        self.assertTrue(r["found"], r)
        self.assertEqual(r["section"], "4.2 Dental implants")
        self.assertIn("Rs 25,000", r["text"])
        self.assertEqual(r["how"], "exact")

    def test_surface_form_is_forgiven(self):
        # the model's casing/punctuation differs from the stored heading
        for cite in ("02_health_and_wellbeing_benefits § dental implants",
                     "02 Health and Wellbeing Benefits § 4.2 Dental Implants",
                     "02_HEALTH_AND_WELLBEING_BENEFITS § 42 dental implants"):
            r = self.api.resolve_citation(cite)
            self.assertTrue(r["found"], (cite, r))
            self.assertIn("Rs 25,000", r["text"])

    def test_unknown_section_lists_what_the_document_has(self):
        r = self.api.resolve_citation("02_HEALTH_AND_WELLBEING_BENEFITS § 9.9 Made up")
        self.assertFalse(r["found"])
        self.assertEqual(r["reason"], "no-section")
        self.assertIn("4.2 Dental implants", r["candidates"])
        self.assertIsNone(r["text"])          # never invents a passage

    def test_unknown_document_says_so(self):
        r = self.api.resolve_citation("[99_NONEXISTENT § 1. Scope]")
        self.assertFalse(r["found"])
        self.assertEqual(r["reason"], "no-document")

    def test_chunk_id_form_resolves(self):
        chunk = self.db.list_chunks(self.doc_id)[0]
        r = self.api.resolve_citation(f"[02_HEALTH_AND_WELLBEING_BENEFITS #{chunk['id']}]")
        self.assertTrue(r["found"], r)
        self.assertEqual(r["chunk_id"], chunk["id"])
        self.assertEqual(r["how"], "chunk-id")

    def test_empty_cite_is_not_an_error(self):
        r = self.api.resolve_citation("[ ]")
        self.assertFalse(r["found"])
        self.assertEqual(r["reason"], "empty")


if __name__ == "__main__":
    unittest.main()
