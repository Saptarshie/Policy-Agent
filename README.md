# Policy RAG Agent 📜

A complete, self-contained **RAG-based policy document QA agent** with:

- **ReAct agent + tool_call loop** — the agent thinks step by step, calls tools
  (`search_policy_docs`, `read_policy_chunk`, …), uses the result of the last
  call to decide the next one, then synthesises a final answer.
- **Citation-grounded answering** — every policy claim must cite
  `[doc_name § Section]` / `[doc_name #chunk_id]`; citations are verified
  after generation and warnings are surfaced when they don't resolve.
- **Rolling-window ingestion for large documents** — never feeds a whole doc
  to the parser LLM: cascading windows → per-window parse + **rolling
  summary** + **progress tracking** (resumable checkpoints in a job table).
- **Intelligent chunking** — heading-aware, paragraph/sentence-boundary
  chunking with breadcrumbs (no random fixed-size cuts).
- **Background contradiction tracker** — staged cascade (see below) that
  flags conflicting policies to the admin **with citations and reasons**.
- **Anti-hallucination armour** — deterministic intent gate, heuristic
  regex repair of malformed tool calls, error feedback to the agent when
  repair fails, strict grounding prompts, citation verification, an
  evidence + faithfulness judge, precision-biased contradiction verifier,
  conflict surfacing.
- **Admin interface** — CLI + web UI to dynamically upload/delete
  `txt` / `md` / `pdf` policy documents.

Models: **Ollama `embeddinggemma`** for embeddings, **any OpenAI-compatible
endpoint** for the LLM (URL + key from `.env`), `MAX_OUTPUT_TOKENS = 65536`.

---

## The contradiction cascade (in execution order)

```
admin uploads doc ──▶ rolling-window ingestion ──▶ chunks + embeddings
                                │
                                ▼  (background)
             ┌──────────────────────────────────────────────────┐
             │ 1. RAG RETRIEVAL        for each new chunk, pull │
             │                          the most similar passage │
             │                          of the other documents   │
             │                          (embeddings only, cheap) │
             ├──────────────────────────────────────────────────┤
             │ 2. DECISION MODEL - pluggable via DECISION_MODE  │
             │    heuristic (default): JEV-style judge IN CODE  │
             │    topical-overlap guard, normative flips        │
             │    (must vs must-not), permission flips,         │
             │    numeric conflicts (15 vs 30 days), antonyms   │
             │    remote: dedicated DECISION_MODEL_* endpoint   │
             │    scores every pair (strict JSON, repaired)     │
             │    hybrid: heuristic first; remote refines       │
             │    only BORDERLINE pairs                         │
             │    ── score >= JEV_GATE_THRESHOLD? ──▶ NO: drop  │
             ├──────────────────────────────────────────────────┤
             │ 3. LLM CONTRADICTION VERIFIER (escalation)       │
             │    runs ONLY for gate-passing pairs; strict JSON │
             │    verdict; precision-biased prompt              │
             ├──────────────────────────────────────────────────┤
             │ 4. ADMIN FLAG   citation (doc § section + quote) │
             │                + explanation of WHY they clash   │
             │                + gate signals for auditability   │
             └──────────────────────────────────────────────────┘
```

The **decision model runs first** so the expensive LLM verifier only ever
sees a small, pre-filtered set of plausible conflicts. By default the
decision model is the in-code JEV heuristic — deterministic, explainable and
**zero LLM cost**, which is why a default `.env` has no decision-model
endpoint vars. Prefer a served model as the judge? Set `DECISION_MODE=remote`
(or `hybrid`) and fill in `DECISION_MODEL_BASE_URL` / `DECISION_MODEL_API_KEY`
/ `DECISION_MODEL_NAME` — a separate, cheaper OpenAI-compatible endpoint
(e.g. `qwen2.5:0.5b` on Ollama) that scores each pair with a strict-JSON
reply (repaired + retried like every other LLM call, falling back to the
heuristic on failure). If the verifier can't run (endpoint down, unparseable
output) but the gate score is far above threshold, the pair is still recorded
as **UNVERIFIED** for admin review — nothing is silently dropped.

---

## Quickstart

```bash
cd policy-rag-agent
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt

# 1. embeddings backend
ollama pull embeddinggemma          # https://ollama.com/library/embeddinggemma

# 2. configure
cp .env.example .env                # then edit LLM_BASE_URL / LLM_API_KEY / LLM_MODEL

# 3. sanity check (offline unit tests, no LLM/Ollama needed)
python run.py selftest

# 4. ingest the demo policies (v1 and v2 contradict on purpose)
python run.py upload sample_docs/remote_work_policy_v1.txt
python run.py upload sample_docs/remote_work_policy_v2.md
python run.py upload sample_docs/travel_reimbursement_policy.txt

# 5. watch the cascade fire
python run.py contradictions

# 6. ask grounded questions
python run.py ask "How many days per week can I work remotely?" --trace

# 7. or use the web UI (upload / delete / chat / conflict board)
python run.py serve           # → http://127.0.0.1:8000
```

The first upload of `remote_work_policy_v2.md` should flag conflicts against
`remote_work_policy_v1.txt`: 3 vs 1 remote days, 15 vs 30 days notice,
6 vs 12 months eligibility, 10:00–16:00 vs 09:00–17:00 core hours.
Uploading `sample_docs/amendments/leave_policy_2025_amendment.md` next to
`leave_policy.md` similarly flags the 6 vs 15 carry-forward-day conflict.

---

## CLI reference

| command | what it does |
|---|---|
| `python run.py upload <files...> [--name N]` | ingest txt/md/pdf (rolling-window pipeline, live progress bar) |
| `python run.py list` | documents + status + chunk counts + open conflict counts |
| `python run.py delete <name\|id>` | remove a document and all its chunks/flags |
| `python run.py ask "<question>" [--trace]` | one-shot grounded answer; `--trace` shows the ReAct steps |
| `python run.py chat` | interactive session (`/trace`, `/docs`, `exit`) |
| `python run.py scan [--doc NAME]` | re-run the contradiction cascade |
| `python run.py contradictions [--status OPEN]` | show flags: citations, quotes, why they conflict, gate signals |
| `python run.py resolve <id> <resolved\|dismissed\|OPEN>` | triage a flag |
| `python run.py status <job_id>` | ingestion/scan job progress |
| `python run.py serve [--host H] [--port P]` | launch the web UI |
| `python run.py selftest` | run the offline test suite |

## Web UI

`python run.py serve` gives you a three-pane console: **Documents** (upload,
progress bars, delete, rescan), **Chat** (grounded answers, rendered
citations, collapsible ReAct trace, citation-check warnings) and
**Contradiction tracker** (flags with both quoted passages, the LLM's
explanation, the gate's heuristic signals, resolve/dismiss buttons).

---

## Anti-hallucination measures

1. **Triage before retrieval (one LLM call)** — the raw input is classified
   into one of 8 categories and the route is derived from the category:
   greetings and document listings are answered directly, non-HR input and
   prompt injection are refused, everything about workplace policy goes to the
   grounded agent. The layer **fails open** and **never refuses a workplace
   question** (it has not read the corpus), so it cannot become a coverage
   outage. It also de-noises the question and proposes the search queries.
2. **Grounding prompt** — answer only from tool output; explicit
   "the documents don't cover this" escape hatch; conflicts must be surfaced,
   never silently resolved.
3. **Forced citations + post-hoc verification** — citations that don't
   resolve to a real document/chunk produce visible warnings.
4. **Tool-call schema repair (heuristic regex)** — code fences, single
   quotes, trailing commas, unquoted keys, Python `True/False/None`,
   unbalanced brackets, truncated tails, prose around JSON.
5. **Repair-failure feedback** — when repair/validation fails, the agent
   receives a `TOOL_CALL_ERROR` observation describing exactly what was
   wrong and is told to re-emit a corrected call — it never receives a
   fabricated result.
6. **Fallback tool calling** — endpoints without native function calling
   can emit `<tool_call>{...}</tool_call>`; same repair/validation path.
7. **Evidence + faithfulness verification** — the judge reads the passages
   the answer **cites**, then the rest of what the agent saw (24 chunks /
   24k chars), plus the ingested-document inventory so statements about which
   policy exists are checkable. It flags unsupported claims
   (`faithful: false` + the quoted claims in `citation_warnings`); a verdict
   that names no claim is retried once and then degraded to `unknown`, never
   reported as a failure. Truncated evidence adds an explicit coverage note so
   an omission can never turn into an accusation.
7b. **Answer shape is classified, not pattern-matched** — before judging, the
   decision model (`typesafe/jev`, the same model the contradiction gate uses)
   answers four questions about the answer: does it decline part of the
   question, does it state policy rules, is it only a refusal, is it only
   chit-chat. That decides *what* gets verified — chit-chat is skipped, **a
   partial answer is always judged**, and a refusal is judged too, because
   "the documents do not cover this" is a claim about the corpus. A classifier
   outage degrades to `unknown`, which means the answer is judged anyway: the
   failure mode is more checking, never less. Nothing here is decided by
   matching phrases.
8. **Precision-biased verifier** — different scope / complementary rules /
   topical similarity only are explicitly *not* contradictions; uncertain →
   false. Unverifiable-but-strong gate hits are marked UNVERIFIED, not
   dropped and not silently confirmed.
9. **Explainable gate** — the decision model's verdict is fully auditable:
   `decided_by` records which judge made the call (`heuristic` / `remote` /
   `hybrid(remote)` / `remote->heuristic`) and its signals are stored with
   every flag.

## Configuration (`.env`)

| variable | default | meaning |
|---|---|---|
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | ollama/v1 | any OpenAI-compatible endpoint |
| `MAX_OUTPUT_TOKENS` | `65536` | output ceiling for every LLM call (agent, judge, decision model); a ceiling only - it never lengthens a reply, it prevents truncation. Auto-halved if an endpoint caps it |
| `TEMPERATURE` | `0.1` | low temperature for grounded answers |
| `OLLAMA_BASE_URL` / `EMBED_MODEL` | `…:11434` / `embeddinggemma` | embeddings backend |
| `EMBED_QUERY_PREFIX` / `EMBED_DOC_PREFIX` | EmbeddingGemma task prefixes | keep quoted! Ollama does not add them itself |
| `INGEST_WINDOW_CHARS` | `6000` | rolling-window size for read+parse |
| `INGEST_LLM_ENABLED` | `true` | set `false` for heuristic-only ingestion |
| `CHUNK_TARGET_CHARS` / `CHUNK_MAX_CHARS` | `1400` / `2200` | soft chunk bounds |
| `JEV_GATE_THRESHOLD` | `0.5` | decision-model escalation threshold |
| `DECISION_MODE` | `heuristic` | stage-2 decision model: `heuristic` (in-code JEV, zero cost) / `remote` (dedicated endpoint scores every pair) / `hybrid` (heuristic first, remote refines borderline pairs) |
| `DECISION_MODEL_BASE_URL` | *(empty)* | OpenAI-compatible base URL of the decision model (only for `remote`/`hybrid`; e.g. `http://localhost:11434/v1`) |
| `DECISION_MODEL_API_KEY` | *(empty)* | API key for the decision-model endpoint (e.g. `ollama`) |
| `DECISION_MODEL_NAME` | *(empty)* | model name at the decision endpoint (e.g. `qwen2.5:0.5b`) |
| `DECISION_TEMPERATURE` | `0.0` | keep the judge deterministic |
| `DECISION_BORDERLINE_BAND` | `0.15` | hybrid mode: heuristic scores within ±band of the threshold are refined remotely |
| `DECISION_FALLBACK_HEURISTIC` | `true` | fall back to the in-code heuristic when the remote decision model fails |
| `CONTRA_TOP_K` | `4` | RAG neighbours per chunk during a scan |
| `SCAN_MAX_PAIRS_PER_CHUNK` | `3` | max LLM verifications per chunk (cost cap) |
| `SCAN_CROSS_DOC_ONLY` | `true` | only compare across documents |
| `AGENT_MAX_STEPS` | `12` | ReAct loop budget |
| `VERIFY_ANSWERS` | `true` | post-answer faithfulness judge (one extra LLM call) |
| `DECISION_MODE` | `remote` | also selects the answer-shape classifier: `remote`/`hybrid` use the decision model, `heuristic` falls back to the chat model's JSON mode |
| `INTENT_ENABLED` | `true` | LLM triage in front of the agent (one extra cheap call) |
| `INTENT_MODEL` | *(empty)* | model for the classifier; empty = `LLM_MODEL` |
| `INTENT_MAX_TOKENS` | `65536` | ceiling for the classifier's JSON reply (a ceiling, not a budget) |
| `INTENT_REASONING_EFFORT` | `off` | reasoning budget for the classifier; JSON extraction needs none |
| `INTENT_CACHE_PATH` | *(empty)* | decision cache; empty = `DATA_DIR/intent_cache.json` |
| `LLM_REASONING_EFFORT` | *(empty)* | provider reasoning budget for agent/judge; empty leaves the endpoint default |
| `JUDGE_CACHE` | `true` | replay a verdict for an identical (question, answer, contexts, prompt) — verdicts vary run to run even at temperature 0 |
| `PROMPT_INTENT` / `PROMPT_AGENT` / `PROMPT_JUDGE` | `v1` / `v1` / `v1` | prompt variants; v1 and v2 tie end-to-end (see `eval/ABLATION.md`) |
| `RAG_TOP_K` | `6` | passages per `search_policy_docs` call |
| `RETRIEVAL_MODE` | `hybrid` | `vector` / `keyword` / `hybrid` (RRF fusion of cosine + FTS5/BM25) |
| `RRF_K` | `60` | RRF damping constant (higher = flatter rank weights) |
| `HYBRID_VECTOR_WEIGHT` / `HYBRID_KEYWORD_WEIGHT` | `1.0` / `1.0` | per-retriever weight inside the fusion |
| `MAX_UPLOAD_MB` | `20` | upload ceiling; larger files get HTTP 413 |
| `STALE_JOB_MINUTES` | `30` | a job untouched this long is marked `error` at startup (killed process), so its document cannot stay `ingesting` |
| `DATA_DIR` | `./data` | SQLite DB + uploads |

## Project layout

```
policy-rag-agent/
├── run.py                      # CLI entry point (upload/list/ask/chat/scan/serve/selftest)
├── DESIGN.md                   # architecture & design decisions (read this!)
├── .env.example                # every knob, documented
├── requirements.txt            # runtime dependencies
├── requirements-dev.txt        # + pytest / httpx / reportlab
├── pytest.ini
├── app/
│   ├── main.py                 # ASGI entry (uvicorn app.main:app)
│   ├── config.py               # .env -> typed config
│   ├── llm.py                  # OpenAI-compatible client (+fallbacks)
│   ├── embeddings.py           # Ollama embeddinggemma client
│   ├── db.py                   # SQLite: docs, chunks(+vectors), flags, jobs
│   ├── schema_repair.py        # regex JSON repair + validation + feedback
│   ├── prompts.py              # all prompts (grounding, parse, verify, judge)
│   ├── readers.py              # txt / md / pdf readers
│   ├── chunker.py              # heading-aware intelligent chunking + windows
│   ├── ingest.py               # rolling-window read+parse + summary + progress
│   ├── rag.py                  # dense + sparse retrieval, RRF fusion
│   ├── admin.py                # admin facade: intent gate -> agent -> verifier
│   ├── cli.py                  # command line interface
│   ├── agent/
│   │   ├── react.py            # ReAct loop + citation checking
│   │   ├── tools.py            # agent tool registry
│   │   ├── intent.py           # LLM intent & triage layer (cache, fail-open)
│   │   ├── intent_legacy.py    # the old regex gate - ablation baseline only
│   │   ├── answer_shape.py     # answer|partial|refusal|chitchat via the decision model
│   │   └── verify.py           # evidence check + faithfulness judge
│   ├── contradictions/
│   │   ├── gate.py             # ★ decision model (stage 2): JEV heuristic +
│   │   │                       #   remote/hybrid CascadeGate (DECISION_MODE)
│   │   ├── verify.py           # ★ LLM contradiction verifier (stage 3)
│   │   └── tracker.py          # ★ cascade orchestration + admin flagging
│   ├── api/
│   │   └── routes.py           # FastAPI routes (factory: create_app(api=...))
│   └── web/index.html          # single-file admin/chat UI
├── tests/                      # offline suite: pytest OR python run.py selftest
│   ├── fakes.py                # FakeLLM / StubLLM / FakeEmbedder
│   ├── conftest.py             # hermetic env bootstrap
│   └── fixtures/sample_policy.pdf
├── eval/
│   ├── eval_set.jsonl          # 18 scored cases (facts/conflicts/refusals/meta)
│   └── run_eval.py             # runner against a live stack
├── scripts/
│   └── make_pdf_fixture.py     # regenerates the PDF test fixture
└── sample_docs/                # demo corpus incl. deliberate conflicts
    ├── remote_work_policy_v1.txt       ┐ contradict each other
    ├── remote_work_policy_v2.md        ┘ (3 vs 1 remote days, 15 vs 30 notice ...)
    ├── travel_reimbursement_policy.txt
    ├── leave_policy.md                 ┐ contradict each other
    └── amendments/
        └── leave_policy_2025_amendment.md  ┘ (6 vs 15 carry-forward days)
```

## Testing & evaluation

```bash
pip install -r requirements-dev.txt
pytest                              # full offline suite (no LLM, no Ollama)
python run.py selftest              # same suite via unittest
python scripts/demo.py              # fact / table / off-policy / refusal, as JSON
python eval/run_eval.py             # 18-case eval against a live stack
python eval/run_intent_eval.py --variant legacy,v1,v2 --hard   # triage ablation
python eval/run_ablation.py --variant A,B,C,D                  # end-to-end prompts
python eval/report_ablation.py      # merged comparison table
python eval/measure_retrieval.py    # vector vs keyword vs hybrid (no LLM)
python eval/calibrate_judge.py      # judge TPR/TNR on injected claims
python scripts/make_pdf_fixture.py  # regenerate tests/fixtures/sample_policy.pdf
```

The suite covers: schema repair + feedback, the ReAct loop (native and
text-based tool calling, citation verification), chunking/windows, the
ingestion pipeline with heuristic fallback, the store, the JEV gate
(pass/reject/determinism), the full cascade (flag / gate-blocks-LLM /
unverified fallback), the triage layer (cache, fail-open, corrective retry,
plan injection), retrieval (FTS5 keyword index, RRF fusion, mode fallbacks),
the faithfulness verifier (shape classification, gating, judge context
selection, unsubstantiated verdicts, partial answers) and the HTTP API end-to-end.

### What the measurements changed

| finding | change |
|---|---|
| regex triage: 74.6% route accuracy, 15 of 22 adversarial inputs sent to the agent | replaced by an LLM triage layer: **100%** route accuracy, 1 miss, 0 false refusals |
| end-to-end, 23 labelled cases | A (regex) **87%** → B/C (triage) **100%**; A fails exactly the injection, the harassment escalation and the joke |
| classifier truncated at 700 output tokens | cap raised + corrective retry; `reasoning_effort=off` gives the same decisions at **4.1s vs 5.2s** |
| judge accused correct conflict answers of inventing facts | judge now sees the cited chunks, the document inventory and each cited document's header — 5 false alarms → **0** |
| refusal vs non-refusal was a hardcoded phrase list (`"do not cover"`, `"couldn't find"`) — a lexical classifier for a language property, and blind to partial answers | replaced by the decision model: four questions, probabilities, five shapes; partial answers always judged, refusals judged too, chit-chat skipped; failure degrades to *judge*, not to skip |
| keyword retrieval returned 0/17 | FTS backfill fixed (it was counting the content table); hybrid RRF is now the default |
| dense vs sparse vs hybrid | tied at k≥3 on a 7-document corpus; sparse wins one top-1 exact-token case. Reported as a tie, not a win |

Full tables, the relabelled cases and the threat list: `eval/ABLATION.md`.

## Troubleshooting

- **`cannot reach Ollama`** — start Ollama, `ollama pull embeddinggemma`,
  check `OLLAMA_BASE_URL`.
- **Endpoint rejects `max_tokens`** — the client halves it automatically
  (64k exceeds many endpoints' limits); set `MAX_OUTPUT_TOKENS` to your
  endpoint's real cap for predictable behaviour.
- **Endpoint doesn't support tools / JSON mode** — both are auto-detected and
  degraded: text `<tool_call>` protocol, JSON repaired from raw content.
- **Scanned PDFs** — PDFs without a text layer can't be parsed (use OCR first).
- **EmbeddingGemma prefixes** — `task: search result | query: ` for queries
  and `title: none | text: ` for documents are required by the model card;
  they're configurable in `.env` (keep the quotes so trailing spaces survive).
