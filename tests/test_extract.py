import tempfile
import unittest
from pathlib import Path

from app.readers import ReaderError, read_document

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class TestReaders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, content, binary=False):
        p = self.dir / name
        if binary:
            p.write_bytes(content)
        else:
            p.write_text(content, encoding="utf-8")
        return p

    def test_txt_roundtrip_and_normalisation(self):
        p = self._write("a.txt", "Line one.  \r\n\r\n\r\nLine two.\x0c")
        text, fmt = read_document(p)
        self.assertEqual(fmt, "txt")
        self.assertNotIn("\r", text)
        self.assertNotIn("\x0c", text)
        self.assertNotIn("\n\n\n", text)

    def test_markdown_decoration_stripped(self):
        p = self._write(
            "b.md",
            "# Title\n\nSome **bold** and *italic* text with a "
            "[link](https://example.com) and an image ![alt](img.png).\n")
        text, fmt = read_document(p)
        self.assertEqual(fmt, "md")
        self.assertIn("# Title", text)          # headings survive (chunker needs them)
        self.assertIn("Some bold and italic text", text)
        self.assertNotIn("**", text)
        self.assertNotIn("[link]", text)
        self.assertNotIn("!(", text)

    def test_pdf_fixture(self):
        p = FIXTURES / "sample_policy.pdf"
        if not p.exists():  # regenerated via scripts/make_pdf_fixture.py
            self.skipTest("pdf fixture missing")
        text, fmt = read_document(p)
        self.assertEqual(fmt, "pdf")
        self.assertIn("Remote Work Allowance Policy", text)
        self.assertIn("internet stipend", text)
        self.assertIn("Bronze", text)           # table content extracted

    def test_unsupported_extension(self):
        p = self._write("c.docx", b"PK\x03\x04", binary=True)
        with self.assertRaises(ReaderError):
            read_document(p)

    def test_missing_file(self):
        with self.assertRaises(ReaderError):
            read_document(self.dir / "nope.txt")

    def test_empty_file(self):
        p = self._write("empty.txt", "   \n\n  ")
        with self.assertRaises(ReaderError):
            read_document(p)

    def test_latin1_fallback(self):
        p = self._write("latin.txt", "caf\xe9 policy".encode("latin-1"), binary=True)
        text, _ = read_document(p)
        self.assertIn("policy", text)


if __name__ == "__main__":
    unittest.main()
