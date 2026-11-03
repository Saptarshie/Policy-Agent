import unittest

from app.chunker import chunk_text, split_into_windows, detect_headings


LONG_PARA = ("Employees must submit all expense claims through the corporate portal. "
             "Claims submitted outside the portal will be rejected without review. "
             "Each claim must include itemised receipts. ") * 6  # ~1400+ chars

SAMPLE = """# Leave Policy

This document governs all leave taken by employees of the company.

## 1. Annual Leave

Employees receive 25 days of paid annual leave per calendar year.

""" + LONG_PARA + """

## 2. Sick Leave

Sick leave is granted at full pay for up to 90 days per year.

A doctor's note is required for absences longer than three consecutive days.
"""


class TestDetectHeadings(unittest.TestCase):
    def test_markdown_headings(self):
        hs = detect_headings(SAMPLE)
        titles = [h.title for h in hs]
        self.assertIn("Leave Policy", titles)
        self.assertIn("1. Annual Leave", titles)
        self.assertIn("2. Sick Leave", titles)

    def test_numbered_txt_headings(self):
        text = "Intro line here.\n\n1. Purpose\nThis is the purpose.\n\n2. Scope\nThe scope is wide."
        hs = detect_headings(text)
        titles = [h.title for h in hs]
        self.assertIn("1. Purpose", titles)
        self.assertIn("2. Scope", titles)

    def test_allcaps_heading(self):
        hs = detect_headings("ACME CORPORATION\n\nSome body text follows here.")
        self.assertTrue(any("Acme Corporation" == h.title for h in hs))


class TestChunkText(unittest.TestCase):
    def test_basic_chunking(self):
        chunks = chunk_text(SAMPLE, "leave_policy", target=400, max_chars=700)
        self.assertGreaterEqual(len(chunks), 3)
        for c in chunks:
            self.assertLessEqual(c.char_count, 700)
            self.assertTrue(c.text.strip())
            self.assertTrue(c.breadcrumb.startswith("leave_policy"))
            self.assertTrue(c.section)

    def test_section_separation(self):
        chunks = chunk_text(SAMPLE, "leave_policy", target=400, max_chars=700)
        sick = [c for c in chunks if "Sick Leave" in c.section]
        annual = [c for c in chunks if "Annual Leave" in c.section]
        self.assertTrue(sick and annual)
        for c in sick:
            self.assertNotIn("25 days of paid annual leave", c.text)

    def test_long_paragraph_sentence_split(self):
        chunks = chunk_text("Plain doc\n\n" + LONG_PARA * 3, "plain", target=300, max_chars=500)
        self.assertGreater(len(chunks), 2)
        for c in chunks:
            self.assertLessEqual(c.char_count, 500)

    def test_single_newline_document(self):
        text = "\n".join(f"Rule number {i} applies to all staff members equally." for i in range(60))
        chunks = chunk_text(text, "rules", target=300, max_chars=500)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c.char_count, 500)

    def test_tiny_doc_single_chunk(self):
        chunks = chunk_text("Short doc\n\nOne small paragraph.", "tiny")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section, "Preamble")


class TestWindows(unittest.TestCase):
    def test_windows_cover_text_and_respect_size(self):
        text = "\n\n".join(f"Paragraph {i}: " + "word " * 60 for i in range(30))
        windows = split_into_windows(text, window_chars=2000, overlap=200)
        self.assertGreater(len(windows), 1)
        for w in windows:
            self.assertLessEqual(len(w), 2000 + 250)  # window + carried overlap
        # every paragraph number appears somewhere
        joined = "\n".join(windows)
        for i in (0, 10, 29):
            self.assertIn(f"Paragraph {i}:", joined)

    def test_short_doc_single_window(self):
        windows = split_into_windows("one small doc", window_chars=2000)
        self.assertEqual(len(windows), 1)


if __name__ == "__main__":
    unittest.main()
