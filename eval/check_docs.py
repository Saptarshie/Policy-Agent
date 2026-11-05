"""Documentation consistency check: docs vs. code.

Catches the failure mode that matters here - README/DESIGN describing settings,
tools or behaviours that no longer exist (or new ones nobody documented).

    python eval/check_docs.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
DESIGN = (ROOT / "DESIGN.md").read_text(encoding="utf-8")
CONFIG = (ROOT / "app" / "config.py").read_text(encoding="utf-8")
ENV_EXAMPLE = (ROOT / ".env.example").read_text(encoding="utf-8")
TOOLS = (ROOT / "app" / "agent" / "tools.py").read_text(encoding="utf-8")

problems: list[str] = []
notes: list[str] = []


def check(ok: bool, message: str) -> None:
    if not ok:
        problems.append(message)


# ------------------------------------------------- 1. env keys read by the code
# the call may wrap across lines, so allow whitespace after the paren
read_keys = set(re.findall(r'_(?:str|int|float|bool)\(\s*"([A-Z0-9_]+)"', CONFIG))
env_keys = set(re.findall(r"^([A-Z0-9_]+)=", ENV_EXAMPLE, re.M))
missing_env = sorted(read_keys - env_keys)
check(not missing_env, f".env.example is missing keys the code reads: {missing_env}")
notes.append(f"{len(read_keys)} env keys read, {len(env_keys)} documented in .env.example")

# --------------------------------------------- 2. env keys documented in README
undocumented = sorted(k for k in env_keys if k not in README)
check(not undocumented, f"README does not mention: {undocumented}")
notes.append(f"{len(env_keys) - len(undocumented)}/{len(env_keys)} env keys appear in README")

# ----------------------------------------------------------- 3. tool inventory
tool_names = re.findall(r'"name": "([a-z_]+)"', TOOLS)
check(len(tool_names) == 11, f"expected 11 tools, code declares {len(tool_names)}")
for name in tool_names:
    # scratchpad_write/read/... may be documented as the family "scratchpad_*"
    alt = "scratchpad_" if name.startswith("scratchpad_") else name
    check(alt in README, f"README never mentions the tool '{name}'")
    check(alt in DESIGN, f"DESIGN never mentions the tool '{name}'")
notes.append(f"tools documented: {len(tool_names)}")

# --------------------------------- 4. removed features must not read as current
# word-ish matching: DECISION_MODE must not fire on DECISION_MODEL_*
REMOVED = [r"DECISION_MODE(?!_)", r"CONTRA_TOP_K",
           # the old architecture labels: the answer path has no judge, and the
           # lexical gate is no longer a decision maker
           r"GROUNDING VERIFIER", r"ZERO LLM",
           r"SCAN_MAX_PAIRS_PER_CHUNK", r"DECISION_BORDERLINE_BAND",
           r"DECISION_FALLBACK_HEURISTIC", r"VERIFY_ANSWERS",
           # the ingestion pipeline: no cascade, no rolling carry, no merge call
           r"rolling-window", r"rolling summary", r"merge call"]
HISTORICAL = ("gone", "first version", "removed", "reversal", "used to",
              "was a", "no longer", "any more", "replaced", "old design")
for doc, text in (("README.md", README), ("DESIGN.md", DESIGN)):
    lines = text.splitlines()
    for pattern in REMOVED:
        for i, line in enumerate(lines, 1):
            if not re.search(pattern, line):
                continue
            # a sentence about the removal may wrap: look one line back too
            # a sentence about the removal may wrap either way
            context = " ".join(lines[max(0, i - 2):i + 1]).lower()
            if any(h in context for h in HISTORICAL):
                continue
            problems.append(f"{doc}:{i} still describes '{pattern}' as current: "
                            f"{line.strip()[:100]}")

# ---------------------------------------- 5. settings the docs must explain now
REQUIRED = ["SCAN_MIN_SIMILARITY", "SCAN_MAX_CANDIDATES", "SCAN_CROSS_DOC_ONLY",
            "SCAN_WORKERS", "DECISION_BATCH_SIZE", "DECISION_RECHECK_BAND",
            "JEV_GATE_THRESHOLD", "min_similarity", "list_contradictions"]
for key in REQUIRED:
    check(key in README, f"README does not document '{key}'")
check("SCAN_MIN_SIMILARITY" in DESIGN and "SCAN_WORKERS" in DESIGN,
      "DESIGN does not explain the scan knobs")

# ------------------------------------------------ 6. the project layout is true
for path in ["app/agent/calc.py", "app/agent/scratchpad.py", "app/naming.py",
             "app/agent/answer_shape.py", "app/contradictions/tracker.py",
             "eval/measure_scan.py", "eval/verify_api.py"]:
    check((ROOT / path).exists(), f"documented path missing: {path}")
README_TREE = re.search(r"```\n(.*?)```", README[README.index("## Project layout"):], re.S)
if README_TREE:
    for name in ["calc.py", "scratchpad.py", "naming.py", "answer_shape.py"]:
        check(name in README_TREE.group(1), f"project layout omits {name}")

# ------------------------------------------------------- 7. stale counts/claims
test_funcs = sum(
    len(re.findall(r"^\s*def test_", f.read_text(encoding="utf-8"), re.M))
    for f in (ROOT / "tests").glob("test_*.py"))
m = re.search(r"(\d+)\s+tests", README)
check(bool(m), "README does not state the suite size")
if m:
    check(int(m.group(1)) == test_funcs,
          f"README says {m.group(1)} tests, tests/ defines {test_funcs}")
notes.append(f"suite size: {test_funcs} test functions")
check(not re.search(r"DECISION_MODE\b(?!_)", README),
      "README still has a DECISION_MODE mode table")

# ------------------------------------------- 8. every endpoint is documented
ROUTES_SRC = (ROOT / "app" / "api" / "routes.py").read_text(encoding="utf-8")
routes = re.findall(r'@app\.(?:get|post|put|delete|patch)\("([^"]+)"', ROUTES_SRC)
# prose may name the placeholder differently ({id} vs {doc_id}); compare shapes
def _shape(text: str) -> str:
    return re.sub(r"\{[^}]*\}", "{}", text)
for r in routes:
    if r == "/":                      # the UI page itself, documented in prose
        continue
    check(_shape(r) in _shape(README), f"README does not document the endpoint {r}")
    check(_shape(r) in _shape(DESIGN), f"DESIGN does not document the endpoint {r}")
notes.append(f"endpoints documented: {len(routes) - 1}")

# --------------------------------- 9. every code path the docs name must exist
named = set()
for text in (README, DESIGN):
    named |= set(re.findall(r"\b((?:app|eval|tests|scripts)/[A-Za-z0-9_/.]+\.py)", text))
for rel in sorted(named):
    check((ROOT / rel).exists(), f"docs name a path that does not exist: {rel}")
notes.append(f"{len(named)} code paths named in the docs all exist")
# README lists eval/run_benchmark.py etc.; a script renamed on disk must not
# linger in the README as a command nobody can run
for rel in sorted(r for r in named if r.startswith("eval/")):
    check(f"eval/{Path(rel).name}" in README, f"README never mentions {rel}")

# ------------------------- 10. every module in app/ is listed in the layout tree
for f in sorted((ROOT / "app").rglob("*.py")):
    if f.name == "__init__.py":
        continue
    check(f.name in README, f"app module never documented: {f.relative_to(ROOT)}")
for d in ("Benchmark", "dev_docs", "eval", "tests", "scripts"):
    check(f"{d}/" in README, f"README's layout omits the {d}/ directory")

# ------------------------------------- 11. one config-table row per env key
for key in sorted(env_keys):
    rows = len(re.findall(rf"^\|\s*`{key}`", README, re.M))
    check(rows <= 1, f"README has {rows} config-table rows for {key}")

print("doc consistency:")
for n in notes:
    print("  -", n)
if problems:
    print(f"\n{len(problems)} problem(s):")
    for p in problems:
        print("  FAIL:", p)
    sys.exit(1)
print("\nall documentation checks pass")
