"""Exercise the UI's markdown renderer off-browser, with real agent answers.

    python eval/verify_renderer.py
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
HTML = ROOT / "app" / "web" / "index.html"
RESULTS = ROOT / "eval" / "results" / "ui_verify"


STUBS = """
// minimal DOM so the page script can be loaded in node: every element is a
// no-op object, and the paint/poll entry points are disabled
const _el = new Proxy({}, { get: (t, k) => (k === 'style' || k === 'classList' || k === 'dataset')
  ? new Proxy({}, { get: () => '', set: () => true })
  : (typeof k === 'string' && k.startsWith('add') ? () => {} : () => ''), set: () => true });
globalThis.document = {
  getElementById: () => _el,
  querySelector: () => null,
  querySelectorAll: () => [],
  createElement: () => _el,
  addEventListener: () => {},
};
globalThis.setInterval = () => 0;
globalThis.alert = () => {};
globalThis.confirm = () => false;
globalThis.fetch = () => Promise.reject(new Error('offline'));
process.on('unhandledRejection', () => {});
"""


def extract_js() -> str:
    html = HTML.read_text(encoding="utf-8")
    src = re.search(r"<script>(.*)</script>", html, re.S).group(1)
    return STUBS + src


def run(cases: list[dict]) -> list[dict]:
    js = extract_js() + """
const cases = JSON.parse(process.argv[2]);
const out = cases.map(c => ({ name: c.name, html: citeHtml(md(c.text)) }));
process.stdout.write(JSON.stringify(out));
"""
    tmp = pathlib.Path(tempfile.gettempdir()) / "renderer_check.js"
    tmp.write_text(js, encoding="utf-8")
    payload = json.dumps(cases)
    r = subprocess.run(["node", str(tmp), payload], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"node failed: {r.stderr[-800:]}")
    return json.loads(r.stdout)


def main() -> int:
    cases = []
    for f in sorted(RESULTS.glob("ask_*.json")):
        try:
            cases.append({"name": f.name, "text": json.loads(f.read_text(encoding="utf-8"))["answer"]})
        except Exception:
            pass
    cases += [
        {"name": "xss-script", "text": "Here is <script>alert(1)</script> and <img src=x onerror=alert(2)>."},
        {"name": "xss-attr", "text": "<a href=\"javascript:alert(1)\">click</a>"},
        {"name": "headings", "text": "## Direct answer\n**bold** and *italic* and `code`"},
        {"name": "list", "text": "- one\n- two\n\n1. first\n2. second"},
        {"name": "table", "text": "| Document | Limit |\n|---|---|\n| v1 | 3 days |\n| v2 | 1 day |"},
        {"name": "brackets", "text": "A [Q1] note and a citation [leave_policy § 3.2] and [42]."},
        {"name": "blockquote", "text": "> quoted rule\n\nplain"},
    ]
    rendered = run(cases)
    fails = []
    for r in rendered:
        name, h = r["name"], r["html"]
        if name.startswith("ask_"):
            if "## " in h or "**" in h:
                fails.append(f"{name}: raw markdown left in output")
            if "§" in h and "class=\"cite\" data-cite=" not in h:
                fails.append(f"{name}: citations not chipped")
        if name.startswith("xss"):
            # only real tags can hurt: the renderer is the only source of markup,
            # so any raw '<tag' from the input means escaping failed
            if re.search(r"<(script|img|a\s+href=[\"']?javascript)", h):
                fails.append(f"{name}: unescaped markup survived: {h[:200]}")
            if "&lt;" not in h:
                fails.append(f"{name}: input was not escaped at all")
        if name == "headings":
            for probe in ("<h4>Direct answer</h4>", "<strong>bold</strong>",
                          "<em>italic</em>", "<code>code</code>"):
                if probe not in h:
                    fails.append(f"headings: missing {probe}")
        if name == "list":
            if "<ul>" not in h or "<ol>" not in h or h.count("<li>") != 4:
                fails.append(f"list: unexpected output {h[:200]}")
        if name == "table":
            if "<table>" not in h or h.count("<td>") != 4 or "---" in h.replace("&lt;", ""):
                fails.append(f"table: unexpected output {h[:300]}")
        if name == "brackets":
            # citations must be CLICKABLE (a button carrying the cite text), while
            # anything that is not a citation stays literal
            if 'data-cite="leave_policy § 3.2"' not in h:
                fails.append(f"brackets: citation is not a clickable chip {h[:240]}")
            if ">[Q1]</button>" in h:
                fails.append("brackets: [Q1] should stay plain text")
        if name == "blockquote" and "<blockquote>" not in h:
            fails.append("blockquote: not rendered")

    print(f"{len(rendered) - 0} cases rendered, {len(fails)} problem(s)")
    for f in fails:
        print("  FAIL:", f)
    for r in rendered[:2]:
        print(f"\n--- {r['name']} (first 400 chars of HTML) ---\n{r['html'][:400]}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
