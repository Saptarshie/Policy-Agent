"""Citations have to resolve to real text, or say why they do not.

A chip that opens the wrong passage is worse than a chip that does nothing: it looks
like verification. These tests pin the contract - forgiving on the surface form the
model produces, never inventing a passage - and the endpoint the UI calls.
"""

import unittest

from app.citations import (CitationResolver, parse_citation, section_number,
                           section_satisfied)
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

    def test_a_section_split_across_chunks_returns_every_part(self):
        # a long table can be cut across chunks; showing only the first looks like the
        # policy stops mid-table, which is exactly what a verifier must not be shown
        self.db.add_chunk(doc_id=self.doc_id, index=2, section="4.2 Dental implants",
                          breadcrumb="doc > 4.2 Dental implants",
                          text="Continuation: the second implant is not covered.",
                          embedding=[0.0] * 8)
        r = self.api.resolve_citation("02_HEALTH_AND_WELLBEING_BENEFITS § 4.2 Dental implants")
        self.assertTrue(r["found"])
        self.assertEqual(len(r["parts"]), 2, r["parts"])
        self.assertIn("second implant", r["parts"][1]["text"])
        self.assertLess(r["parts"][0]["chunk_id"], r["parts"][1]["chunk_id"])
        self.assertIn("second implant", r["text"])      # text is the joined view

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


class TestParentHeadings(unittest.TestCase):
    """A heading that only has subsections must still resolve, and must still count.

    Chunkers store "8.1 Permitted stacking"; the rubric (and a reader) says
    "8. Stacking and double funding". Treating those as different things made correct
    citations look like misses - in the UI panel and in the benchmark alike.
    """

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(f"{self.tmp.name}/p.db")
        self.doc_id = self.db.add_document(name="02_INCENTIVES_ACT", fmt="md", status="ready")
        for i, (section, text) in enumerate([
                ("8.1 Permitted stacking", "Programs may stack only if they fund distinct costs."),
                ("8.2 Prohibited duplication", "One cost cannot support two benefits."),
                ("9. Recovery", "Recovery applies to overpayments.")]):
            self.db.add_chunk(doc_id=self.doc_id, index=i, section=section,
                              breadcrumb=f"02_INCENTIVES_ACT > 8. Stacking and double funding > {section}"
                              if section.startswith("8") else f"02_INCENTIVES_ACT > {section}",
                              text=text, embedding=[0.0] * 8)
        # CitationResolver is a mixin: it reads self.db from whoever implements the store
        self.r = type("_Store", (CitationResolver,), {"__init__": lambda self, db: setattr(self, "db", db)})(self.db)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_a_parent_heading_returns_its_subsections(self):
        out = self.r.resolve_citation("[02_INCENTIVES_ACT § 8. Stacking and double funding]")
        self.assertTrue(out["found"])
        self.assertEqual(out["how"], "descendant")
        self.assertEqual([p["chunk_id"] for p in out["parts"]],
                         [c["id"] for c in self.db.list_chunks(self.doc_id)][:2])
        self.assertIn("distinct costs", out["text"])
        self.assertIn("two benefits", out["text"], "both subsections, in document order")

    def test_an_unrelated_number_is_not_a_parent(self):
        out = self.r.resolve_citation("[02_INCENTIVES_ACT § 10. Nothing here]")
        self.assertFalse(out["found"])
        self.assertEqual(out["reason"], "no-section")

    def test_section_numbers_are_structural(self):
        self.assertEqual(section_number("8.1 Permitted stacking"), "8.1")
        self.assertEqual(section_number("  Section 4"), "")
        self.assertEqual(section_number("Example A: x"), "")

    def test_a_subsection_citation_satisfies_the_parent_reference(self):
        self.assertTrue(section_satisfied("8. Stacking and double funding",
                                          ["8.1 Permitted stacking"]))
        self.assertTrue(section_satisfied("8.1 Permitted stacking",
                                          ["8.1 Permitted stacking"]))
        self.assertFalse(section_satisfied("8. Stacking and double funding", ["9. Recovery"]))
        self.assertFalse(section_satisfied("8. Stacking and double funding", []))
        # a sibling is not a parent
        self.assertFalse(section_satisfied("8.2 Prohibited duplication",
                                           ["8.1 Permitted stacking"]))


if __name__ == "__main__":
    unittest.main()
