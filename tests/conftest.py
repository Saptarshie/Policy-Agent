"""Shared test bootstrap: hermetic env + import path, before any app import."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_tmp = tempfile.mkdtemp(prefix="policy_rag_test_")
os.environ["DATA_DIR"] = _tmp
# keep unit tests hermetic: the faithfulness judge is tested explicitly in
# test_verify.py / test_agent_admin.py, not as a side effect of every ask()
os.environ.setdefault("LLM_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("LLM_MODEL", "test-model")
