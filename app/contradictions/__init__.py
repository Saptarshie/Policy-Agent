"""Background contradiction tracker (staged cascade).

Stage 1: RAG retrieval of topically overlapping passages (cheap).
Stage 2: DECISION MODEL - decides which pairs are worth escalating. Pluggable
         via DECISION_MODE in .env:
           * "heuristic" (default) - JEVStyleGate: deterministic, explainable,
             zero LLM cost, implemented in code (no endpoint needed);
           * "remote"  - a dedicated OpenAI-compatible decision model endpoint
             (DECISION_MODEL_BASE_URL / _API_KEY / _NAME) scores each pair;
           * "hybrid"  - heuristic first, remote refinement only for borderline
             pairs.
         THE DECISION MODEL RUNS FIRST; it decides what is worth escalating.
Stage 3: LLM contradiction verifier - runs ONLY for gate-passing pairs.
Stage 4: Verified contradictions are flagged to the admin with citations
         and an explanation of WHY they contradict.
"""

from .gate import (  # noqa: F401
    JEVStyleGate,
    GateDecision,
    RemoteDecisionModel,
    SystemOneDecisionModel,
    CascadeGate,
    build_gate_from_config,
)
from .verify import ContradictionVerifier, Verdict  # noqa: F401
from .tracker import ContradictionTracker  # noqa: F401
