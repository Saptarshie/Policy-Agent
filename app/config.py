"""Central configuration, loaded from environment variables / .env.

All tunables live here so behaviour can be changed without touching code.
See ``.env.example`` for documentation of every variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    _HAS_DOTENV = True
except ImportError:  # pragma: no cover - dotenv is in requirements
    _HAS_DOTENV = False

_LOADED = False


def _load_env() -> None:
    global _LOADED
    if _LOADED:
        return
    if _HAS_DOTENV:
        for candidate in (Path.cwd() / ".env", _ROOT / ".env"):
            if candidate.exists():
                load_dotenv(candidate)
                break
    _LOADED = True


def _str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v


def _int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    s = v.strip().lower()
    if s in ("1", "true", "yes", "on", "y"):
        return True
    if s in ("0", "false", "no", "off", "n"):
        return False
    return default


@dataclass(frozen=True)
class Config:
    # --- LLM (any OpenAI-compatible endpoint) ---
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    max_output_tokens: int
    temperature: float

    # --- Agent ---
    agent_max_steps: int
    agent_max_obs_chars: int

    # --- Embeddings (Ollama / embeddinggemma) ---
    ollama_base_url: str
    embed_model: str
    embed_query_prefix: str
    embed_doc_prefix: str
    embed_batch_size: int
    rag_top_k: int

    # --- Ingestion ---
    ingest_window_chars: int
    ingest_llm_enabled: bool
    chunk_target_chars: int
    chunk_max_chars: int

    # --- Contradiction tracker ---
    jev_gate_threshold: float
    scan_min_similarity: float    # candidate pairs: cosine floor, not a top-k
    scan_max_candidates: int      # safety cap on pairs per document
    scan_cross_doc_only: bool

    # --- Decision model (stage-2 gate: JEV heuristic / remote / hybrid) ---
    decision_model_base_url: str
    decision_model_api_key: str
    decision_model_name: str
    decision_temperature: float

    # --- Storage / server ---
    data_dir: Path
    app_host: str
    app_port: int
    log_level: str

    # --- Intent / triage layer + prompt variants (defaults keep older
    #     constructions of Config valid) ---
    intent_enabled: bool = True
    intent_model: str = ""              # empty -> llm_model
    # Provider reasoning budget. "" = don't send the field (portable default).
    # The classifier is deterministic JSON extraction and measured identical at
    # "off", so it runs with reasoning disabled by default (~25% less latency).
    llm_reasoning_effort: str = ""
    intent_reasoning_effort: str = "off"
    # verdict cache: the same (question, answer, contexts, prompt) replays
    # its verdict from disk, which makes demos and regressions reproducible
    judge_cache: bool = True
    max_upload_mb: int = 20
    # a job untouched for this long is treated as abandoned by a dead
    # process at startup, so its document cannot stay "ingesting" forever
    stale_job_minutes: int = 30
    # Retrieval: "hybrid" fuses dense (cosine) and sparse (FTS5/BM25) rankings
    # with Reciprocal Rank Fusion; "vector" and "keyword" isolate one retriever
    # (used by eval/measure_retrieval.py to justify the default).
    retrieval_mode: str = "hybrid"
    rrf_k: int = 60
    hybrid_vector_weight: float = 1.0
    hybrid_keyword_weight: float = 1.0
    intent_max_tokens: int = 1500
    intent_cache_path: str = ""         # empty -> data_dir/intent_cache.json
    prompt_agent: str = "v1"            # v1 | v2
    prompt_judge: str = "v1"            # v1 | v2
    prompt_intent: str = "v1"           # v1 | v2
    # performance tuning only - defaults are the measured values, so a
    # fixture that does not care about scanning need not set them
    # ingestion tuning (performance only; the stored chunks do not depend on it)
    ingest_reasoning_effort: str = "off"   # window parser: off | low | medium | high
    ingest_max_tokens: int = 3000          # the parser's output is small
    ingest_parallel_windows: int = 4       # 1 = the sequential chained parse
    ingest_heuristic_max_chars: int = 0    # >0: documents this small skip the LLM
    scan_workers: int = 4             # concurrent model/verifier requests
    decision_batch_size: int = 6      # pairs per batched request
    decision_recheck_band: float = 0.15  # re-ask pairs this near the threshold

    @property
    def db_path(self) -> Path:
        return self.data_dir / "policy_rag.db"

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)

    @property
    def decision_model_configured(self) -> bool:
        """True when a remote decision model endpoint is fully specified."""
        return bool(self.decision_model_base_url and self.decision_model_name)

    @property
    def decision_mode_effective(self) -> str:
        """Report which judge the contradiction cascade will use.

        There is no mode selection any more - the decision model judges or
        nothing does - so this is a status string for logs and the CLI, not a
        switch: "remote" when the endpoint is configured, "unconfigured" when it
        is not (in which case the gate passes no pair rather than guessing).
        """
        return "remote" if self.decision_model_configured else "unconfigured"


_CACHE: Optional[Config] = None


def get_config() -> Config:
    """Build (once) and return the process-wide Config."""
    global _CACHE
    if _CACHE is None:
        _load_env()
        _CACHE = Config(
            llm_base_url=_str("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/"),
            llm_api_key=_str("LLM_API_KEY", ""),
            llm_model=_str("LLM_MODEL", "qwen2.5:7b-instruct"),
            max_output_tokens=_int("MAX_OUTPUT_TOKENS", 65536),
            llm_reasoning_effort=_str("LLM_REASONING_EFFORT", "").strip().lower(),
            intent_reasoning_effort=_str("INTENT_REASONING_EFFORT", "off").strip().lower(),
            judge_cache=_bool("JUDGE_CACHE", True),
            temperature=_float("TEMPERATURE", 0.1),
            agent_max_steps=_int("AGENT_MAX_STEPS", 12),
            agent_max_obs_chars=_int("AGENT_MAX_OBS_CHARS", 4000),
            ollama_base_url=_str("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            embed_model=_str("EMBED_MODEL", "embeddinggemma"),
            embed_query_prefix=_str("EMBED_QUERY_PREFIX", "task: search result | query: "),
            embed_doc_prefix=_str("EMBED_DOC_PREFIX", "title: none | text: "),
            embed_batch_size=_int("EMBED_BATCH_SIZE", 32),
            rag_top_k=_int("RAG_TOP_K", 6),
            max_upload_mb=_int("MAX_UPLOAD_MB", 20),
            stale_job_minutes=_int("STALE_JOB_MINUTES", 30),
            retrieval_mode=_str("RETRIEVAL_MODE", "hybrid").strip().lower(),
            rrf_k=_int("RRF_K", 60),
            hybrid_vector_weight=_float("HYBRID_VECTOR_WEIGHT", 1.0),
            hybrid_keyword_weight=_float("HYBRID_KEYWORD_WEIGHT", 1.0),
            ingest_window_chars=_int("INGEST_WINDOW_CHARS", 6000),
            ingest_llm_enabled=_bool("INGEST_LLM_ENABLED", True),
            chunk_target_chars=_int("CHUNK_TARGET_CHARS", 1400),
            chunk_max_chars=_int("CHUNK_MAX_CHARS", 2200),
            jev_gate_threshold=_float("JEV_GATE_THRESHOLD", 0.5),
            # 0.65 measured on the sample corpus: the real leave-cap conflict
            # (v1 vs the 2025 amendment) sits at 0.678, i.e. rank 3 under the old
            # top_k=4 - inside the cut by luck, not by design
            scan_min_similarity=_float("SCAN_MIN_SIMILARITY", 0.65),
            scan_max_candidates=_int("SCAN_MAX_CANDIDATES", 200),
            scan_cross_doc_only=_bool("SCAN_CROSS_DOC_ONLY", False),
            decision_model_base_url=_str(
                "DECISION_MODEL_BASE_URL", "").rstrip("/"),
            decision_model_api_key=_str("DECISION_MODEL_API_KEY", ""),
            decision_model_name=_str("DECISION_MODEL_NAME", ""),
            decision_temperature=_float("DECISION_TEMPERATURE", 0.0),
            ingest_reasoning_effort=_str("INGEST_REASONING_EFFORT", "off").strip().lower(),
            ingest_max_tokens=_int("INGEST_MAX_TOKENS", 3000),
            ingest_parallel_windows=_int("INGEST_PARALLEL_WINDOWS", 4),
            ingest_heuristic_max_chars=_int("INGEST_HEURISTIC_MAX_CHARS", 0),
            scan_workers=_int("SCAN_WORKERS", 4),
            decision_batch_size=_int("DECISION_BATCH_SIZE", 6),
            decision_recheck_band=_float("DECISION_RECHECK_BAND", 0.15),
            data_dir=Path(_str("DATA_DIR", str(_ROOT / "data"))).expanduser().resolve(),
            app_host=_str("APP_HOST", "127.0.0.1"),
            app_port=_int("APP_PORT", 8000),
            log_level=_str("LOG_LEVEL", "INFO"),
            intent_enabled=_bool("INTENT_ENABLED", True),
            intent_model=_str("INTENT_MODEL", ""),
            intent_max_tokens=_int("INTENT_MAX_TOKENS", 1500),
            intent_cache_path=_str("INTENT_CACHE_PATH", ""),
            prompt_agent=_str("PROMPT_AGENT", "v1").strip().lower(),
            prompt_judge=_str("PROMPT_JUDGE", "v1").strip().lower(),
            prompt_intent=_str("PROMPT_INTENT", "v1").strip().lower(),
        )
    return _CACHE
