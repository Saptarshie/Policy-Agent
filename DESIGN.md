# DESIGN.md — Policy RAG Agent ("PolicyPilot")

A citation-grounded Retrieval-Augmented Generation **agent** over admin-uploaded
policy documents (`.txt` / `.md` / `.pdf`). Employees ask questions in natural
language; a ReAct agent searches, reads and answers **only** from the ingested
policies, with verifiable inline citations — and refuses when the corpus is
silent. When an admin uploads a policy that **contradicts** an existing one, a
background **staged cascade** (RAG retrieval → cheap heuristic decision model →
LLM verifier) flags the conflict with citations and an explanation of *why*.

**Models**

| Role | Provider (default) | Why |
|---|---|---|
| Embeddings | Ollama `embeddinggemma` | local, free, private (policy data!); task-steered prefixes applied client-side |
| Chat / reasoning / tool calls | any **OpenAI-compatible endpoint** (Ollama, LM Studio, vLLM, Groq, OpenRouter, OpenAI …) via `.env` | the agent needs decent tool-calling + JSON; `MAX_OUTPUT_TOKENS=65536` with automatic halving if the endpoint caps lower |

Everything is env-configurable (`.env`, see `.env.example`). Every unit test
and the offline smoke run use deterministic fakes — **no key needed to develop
or run CI**.

---

## 1. Architecture

```
            ADMIN                                       EMPLOYEE
              │ upload txt/md/pdf                        │ question (natural language)
              ▼                                          ▼
   ┌─────────────────────┐                     ┌──────────────────────────┐
   │ FastAPI /api/       │                     │ FastAPI /api/ask         │
   │  documents (upload/ │                     └────────────┬─────────────┘
   │  delete) 202 async  │                                  ▼
   └────────┬────────────┘                     ┌──────────────────────────┐
            ▼                                  │ INTENT & TRIAGE (1 call) │
   ┌──────────────────────────────┐            │  8 categories → route    │
   │ INGESTION (bg thread)        │            │  de-noise + query plan   │
   │  read (txt/md/pypdf)         │            └────────────┬─────────────┘
   │  rolling windows (~6k ch)    │                         │ tool_call loop
   │   ├─ per-window LLM parse    │                         ▼
   │   ├─ rolling summary         │            ┌──────────────────────────┐
   │   └─ progress → jobs table   │            │ TOOLS: search_policy_docs│
   │  structure-aware chunking    │            │  read_policy_chunk       │
   │  batch embed (embeddinggemma)│            │  list_policy_documents   │
   └────────┬─────────────────────┘            │  get_document_summary    │
            ▼                                  └────────────┬─────────────┘
   ┌──────────────────────────────┐                         ▼
   │ SQLite (docs, chunks+vec,    │            ┌──────────────────────────┐
   │ contradictions, jobs)        │            │ GROUNDING VERIFIER       │
   └────────┬─────────────────────┘            │  citation check (store)  │
            │ scan (bg thread)                 │  evidence check (seen)   │
            ▼                                  │  LLM faithfulness judge  │
   ┌──────────────────────────────────────────────────────────┐           │
   │ CONTRADICTION CASCADE (background)                        │◀──────────┘
   │  1. RAG retrieval of neighbour passages (embeddings only) │
   │  2. JEV-style DECISION MODEL — heuristic gate, ZERO LLM   │
   │     score >= threshold?  ── no ──▶ drop (logged)          │
   │  3. LLM CONTRADICTION VERIFIER — ONLY for survivors       │
   │  4. ADMIN FLAG with citations + why + gate signals        │
   └──────────────────────────────────────────────────────────┘
```

**Data flow — upload → index.** The API validates type/size, stores the file
and returns `200` immediately with a `doc_id`; a daemon thread runs the
rolling-window pipeline (§2), persisting progress into a `jobs` row
(`stage`, `window i/n`, rolling-summary checkpoint) that the CLI and web UI
poll. On success a second daemon thread starts the contradiction cascade
(§4) against every other ingested document.

**Data flow — query → answer.** One triage call classifies the raw input and
decides the route (§3.1) — a greeting or a document listing is answered
directly, non-HR input and prompt injection are refused, everything about
workplace policy goes to the ReAct loop. The agent receives the *de-noised*
question plus the triage query plan as a second system message, thinks, emits
one tool call, **reads the observation and decides the next action** (refine
the query, open a full chunk, or answer) — up to `AGENT_MAX_STEPS`, then a
forced grounded final answer. Citations are checked against the store; the
faithfulness judge compares the answer against only the passages actually
seen, and against the employee's original wording (so restating the
employee's own numbers is not mistaken for hallucination).

## 2. Chunking & rolling-window ingestion (big documents)

Documents can be huge; **nothing is ever fed whole** to the parser LLM.

* **Reading**: `txt`/`md` decoded with UTF-8 → latin-1 fallback; markdown
  decoration stripped but headings kept; `pdf` via `pypdf` page-by-page
  (image-only PDFs are rejected with a clear error).
* **Windows**: the text is cut on *paragraph* boundaries into ~`INGEST_WINDOW_CHARS`
  windows with a ~200-char carry-over tail. Per window **one bounded LLM call**
  does both (a) structure extraction (title, normative statements, key terms,
  numbers+units+context) and (b) **rolling summary** — `prev_summary + window →
  new summary (≤120 words)` — the "cascade": each window is parsed in the
  context of everything before it, like recursive/rolling-document LLMs.
  If the LLM reply is unparseable even after regex repair, a heuristic
  fallback summarizes and a warning is recorded — ingestion never dies.
* **Progress tracking**: an `ingest` job row is updated after every window
  (`done/total/stage` + rolling-summary checkpoint in `detail`).
* **Chunking** (after windows): structure-aware, never fixed-size —
  boundaries in order of preference: headings (markdown `#`, numbered
  sections, ALL-CAPS titles, `Section/Article/Policy…` labels, underlined
  titles) → blank-line paragraphs → lines → sentences, with a hard-split
  last resort. Every chunk carries a breadcrumb `doc > section > subsection`
  and soft target ~1,400 chars / hard cap ~2,200 (comfortably inside
  embeddinggemma's context). Tiny trailing chunks merge into their
  predecessor.
* **Embedding**: batches of `EMBED_BATCH_SIZE` through Ollama's `/api/embed`
  (legacy `/api/embeddings` fallback), with embeddinggemma's task prefixes
  (`task: search result | query: …` / `title: none | text: …`) applied
  client-side — Ollama does **not** add them for you.

### 2.1 Retrieval: dense, sparse, and the fusion between them

Two retrievers run over the same chunks (`app/rag.py`, `app/db.py`):

* **dense** — cosine over the stored embeddinggemma vectors. It handles
  paraphrase ("time off" ≈ "leave", "can they do that?" ≈ "forfeiture rule"),
  which is most of what employees type.
* **sparse** — SQLite **FTS5 + BM25** over the same chunk text, kept in sync by
  triggers on `chunks` (external-content table, so the text is stored once).
  It is exact where dense vectors are weakest: policy numbers
  (`HR-BEN-2025-002`), tier names, clause numbers and amounts (`$4,000`).
  Queries are tokenised and every term quoted (`Database.build_match_query`) —
  FTS5 has its own grammar and a raw question containing `"`, `*` or `NEAR(` is
  an `OperationalError`, not a search.

`RETRIEVAL_MODE=hybrid` (default) fuses the two **rankings** with **Reciprocal
Rank Fusion**, `score(d) = Σ w_r / (k + rank_r(d))`, k=60. RRF is chosen over a
weighted sum of raw scores because cosine similarity and BM25 are not on a
comparable scale and per-query normalisation is unstable when one retriever
returns a single strong hit; RRF only needs ranks, so neither retriever can
dominate through the size of its numbers. `RETRIEVAL_MODE=vector|keyword`
isolates one retriever, which is what makes the comparison below possible.
If FTS5 is unavailable (it is compiled into CPython's Windows/macOS builds, but
not guaranteed), hybrid degrades to dense-only with a warning.

Measured with `eval/measure_retrieval.py` (label-only, no LLM calls):
`doc@k` = the expected document is in the top-k, `ans@k` = the top-k context
also contains the expected answer text, i.e. retrieval actually handed the model
something it can quote.

```
natural questions (17 labelled), k=1/3/6
mode        doc@1   doc@3   doc@6   ans@1   ans@3   ans@6
vector      14/17   15/17   15/17   14/17   15/17   15/17
keyword     14/17   15/17   15/17   13/17   15/17   15/17
hybrid      14/17   15/17   15/17   13/17   15/17   15/17

exact-token probes (10: policy numbers, tier amounts, clause dates)
mode        doc@1   doc@3   doc@6   ans@1   ans@3   ans@6
vector       8/10    10/10   10/10    8/10   10/10   10/10
keyword      9/10    10/10   10/10    9/10   10/10   10/10
hybrid       9/10    10/10   10/10    9/10   10/10   10/10
```

Read honestly: on a **7-document corpus the comparison is saturated at k=3-6** —
everything wins, so the numbers cannot prove hybrid superiority. The only
separation is top-1 on exact-token probes, where sparse retrieval gains one case
(9/10 vs 8/10) and loses none, and top-1 on natural questions, where hybrid
loses one (13 vs 14) because a keyword hit can outrank a better paraphrase match.
That is the expected trade-off, and it is why the default is hybrid **at the
agent's operating point (k=6) where the three tie**, while the probe result
covers the failure mode a policy assistant cannot afford (a wrong clause number
stated confidently). With a realistically sized corpus (hundreds of documents)
the k=1 separation is the one that will grow, and this harness is how to check it
before flipping the default.

The **contradiction cascade deliberately stays dense-only** (it passes
`mode="vector"` explicitly): it asks "which passages are semantically adjacent",
and keyword overlap there would surface the same clause twice instead of a
genuine conflict.



## 3. The ReAct agent & anti-hallucination stack

Cheap layers first, expensive last:

1. **Intent & triage layer** (`app/agent/intent.py`, one LLM call, no tools)
   — employees do not speak in policy language: "my 12 leaves vanish on Dec
   31st, this is BS" embeds badly against formal policy text. Triage
   classifies the raw input into one of eight categories, derives the route
   from the category (`REACT_LOOP` / `DIRECT_REPLY` / `FAST_REFUSE`), rewrites
   the emotional input into a neutral question and proposes 1-3 formal search
   queries plus entities and a table/tier hint. **It refuses only non-HR input
   and prompt injection — never a workplace question**, because it has not read
   the corpus; coverage refusals belong to the grounded agent. Decisions are
   cached on disk (question + prompt version + model + reasoning effort), so
   repeats are free and an interrupted evaluation resumes where it stopped,
   and the layer **fails open**: any error routes to the agent, because a
   triage outage must not become a coverage outage. The old regex gate is kept
   as `app/agent/intent_legacy.py` purely as the ablation baseline.
   Measured on 67 labelled cases (45 canonical + 22 adversarial/ambiguous):
   **route accuracy 74.6% → 100%**, missed refusals **15 → 1**, false refusals 0
   in both, at the cost of one cached call (p50 5.2s, 4.1s with reasoning
   disabled). The regex gate is 0s and wrong about one in four real questions,
   which is the trade the numbers support.
2. **Prompt contract** (`app/prompts.py`) — grounding, search-first, verify
   truncated snippets, mandatory citation format, exact "documents do not
   cover this" refusal text, "present both sides on conflicts", scope limits.
3. **Tool-call schema repair + error feedback** (`app/schema_repair.py`) —
   malformed JSON arguments go through a regex repair cascade (code fences,
   prose around JSON, JS comments, Python literals, trailing commas, unquoted
   keys, single quotes, truncated tails, unbalanced brackets) then schema
   validation (required/type/enum/unknown params, with coercion). **If repair
   fails the error itself becomes the tool observation** — the agent re-emits
   a corrected call; it never sees a fabricated result. Endpoints without
   native function calling are handled by parsing text tool-calls
   (`{"name": …, "arguments": …}`, `ACTION:` lines).
4. **Citation verification (deterministic)** — every `[doc § section]` /
   `[doc #chunk_id]` in the final answer is resolved against the store
   (fuzzy doc-name match, chunk-id existence); failures surface as
   `citation_warnings`, never silently.
5. **Answer shape** (`app/agent/answer_shape.py`) — before anything is judged,
   the **decision model** answers four questions about the answer (does it
   decline part of the question / does it state policy rules / is it only a
   refusal / is it only chit-chat) and the shape is derived from those
   probabilities: `answer`, `partial`, `refusal`, `chitchat`, `unknown`.
   This replaced a hardcoded refusal-phrase list, which was a lexical
   classifier for a language property and could not see a partial answer at
   all. Ambiguous scores, a classifier outage or an unparseable verdict all
   yield `unknown`, which the verifier treats as *judge it* — the failure mode
   is more checking, never less.
6. **Evidence + faithfulness check** (`app/agent/verify.py`) — the agent can
   only have relied on passages it *saw*: chunk ids are collected from tool
   observations during the run. An answer produced with zero retrieved
   evidence, while stating policy, gets an explicit warning. Then (optional,
   default on, `VERIFY_ANSWERS`) one strict-JSON **LLM judge** compares the
   answer against only those passages and lists unsupported claims →
   `faithful=false` + `unsupported_claims[]` + warning. Chit-chat is skipped;
   everything else is judged, including a **partial answer**, whose two halves
   are checked differently: the answered half against the passages, the
   declined half as a claim *about the corpus* (a decline is unsupported if any
   passage addresses it). The judge degrades open
   (unavailable/unparseable ⇒ `faithful=null`), and the answer is
   *flagged, never silently rewritten*.
7. **Endpoint hardening** (`app/llm.py`) — retries with backoff on 429/5xx;
   automatic feature downgrade when an endpoint rejects `tools`,
   `response_format` or `reasoning_effort`; automatic `max_tokens` halving when
   the configured 64k exceeds the endpoint cap; `chat_json()` parses every
   reply through the same repair layer and feeds the parse error back for one
   retry. `MAX_OUTPUT_TOKENS` is one global *ceiling*, not a budget — billing
   follows actual output, so a large ceiling costs nothing and small
   per-caller caps only create truncation risk (reasoning tokens are billed
   against it). `REASONING_EFFORT` is the real cost lever: the classifier runs
   with reasoning off, the agent leaves the provider default in place.

## 4. The contradiction cascade (staged, cost-aware)

Runs in the background after every successful ingestion, per chunk of the new
document against all *other* ready documents (`SCAN_CROSS_DOC_ONLY`):

| Stage | Component | Cost | What it does |
|---|---|---|---|
| 1 | **RAG retrieval** | embeddings only | pull the `CONTRA_TOP_K` most similar passages of the other docs for each new chunk |
| 2 | **Decision model** (`contradictions/gate.py`) | heuristic: **zero LLM**; remote/hybrid: bounded calls to a *separate, cheaper* endpoint | scores contradiction-likelihood and **gates** escalation |
| 3 | **LLM verifier** (`contradictions/verify.py`) | 1 bounded call **only for gate-passing pairs** | strict-JSON verdict: is it really a contradiction? |
| 4 | **Admin flag** (`contradictions/tracker.py`) | — | writes the conflict with full citations + why + gate signals |

### 4.1 The decision model — pluggable (`DECISION_MODE`)

The decision model is the stage-2 judge that decides WHICH candidate pairs
are worth escalating. It is pluggable, configured entirely from `.env`:

| Mode | Endpoint vars used | Behaviour |
|---|---|---|
| `heuristic` **(default)** | *none* | `JEVStyleGate`: deterministic, explainable lexical judge implemented **in code** (regex lexicons + weighted signals + threshold). Zero LLM cost — which is exactly why no decision-model endpoint appears in a default `.env`. |
| `remote` | `DECISION_MODEL_BASE_URL` / `_API_KEY` / `_NAME` | `RemoteDecisionModel`: every candidate pair is scored by a **dedicated OpenAI-compatible model** — a different, cheaper endpoint than the main LLM (e.g. `qwen2.5:0.5b` on Ollama, or a hosted mini model). It must reply strict JSON `{"score": 0..1, "signals": [...]}`; replies go through the same regex repair + parse-error-feedback retry as the main agent; the score is clamped/rejected if non-numeric. |
| `hybrid` | same as `remote` | Heuristic scores first for free; pairs **clearly** above/below the threshold (outside `DECISION_BORDERLINE_BAND`) are decided without any network call; only **borderline** pairs are refined by the remote decision model. Best precision-per-cost. |

The cascade never breaks: if `remote`/`hybrid` is selected but the decision
endpoint is missing or fails, the gate **degrades to the in-code heuristic**
(logged; or, with `DECISION_FALLBACK_HEURISTIC=false`, blocks escalation
instead of guessing). Every decision records `decided_by`
(`heuristic` / `remote` / `hybrid(remote)` / `remote->heuristic`) plus its
signals, so an admin can always audit *which judge* made each call.

**The decision model runs FIRST** — it is the whole point of the cascade: the
expensive LLM only ever sees a small, pre-filtered set of plausible conflicts.

The in-code JEV heuristic's signals (each weighted, all logged for auditability):

* **topical-overlap gate** — pairs about different subjects are rejected
  outright (kills "30 days" vs "3 days" false alarms on unrelated topics);
* **normative flip** — requirement (*must/shall/required*) vs prohibition
  (*must not/prohibited*);
* **permission flip** — *may/allowed/optional* vs prohibition, or requirement
  vs optional;
* **numeric conflicts** — same unit, different value, overlapping context
  ("15 days" vs "30 days" notice; "3 days" vs "1 day" remote work);
* **antonym pairs** — increase/decrease, remote/onsite, mandatory/voluntary…;
* high topical overlap bonus.

Score ≥ `JEV_GATE_THRESHOLD` (default 0.5) ⇒ escalate to the verifier, at
most `SCAN_MAX_PAIRS_PER_CHUNK` pairs per chunk (cost cap). The verifier
prompt is **precision-biased**: different scope, complementary rules, mere
topical overlap and explicit supersession are *not* contradictions.

**Nothing is silently dropped.** If the verifier can't run (endpoint down,
unparseable) but the gate score is well above threshold, the pair is recorded
as **UNVERIFIED** for admin review. Duplicate pairs are skipped; deleting a
document purges its flags; admins resolve/dismiss flags from CLI or web UI.

## 5. Storage & API

One SQLite file (`data/policy_rag.db`, WAL) + float32 embedding BLOBs with an
in-memory numpy matrix cache (invalidated on write) for sub-ms brute-force
cosine search — at thousands (not millions) of chunks this beats running a
vector DB service. Tables: `documents`, `chunks`, `contradictions`, `jobs`.
A single connection guarded by an `RLock` keeps the background ingest/scan
threads safe. A chunk pair is unique in **either order** (a `min/max`
expression index), because a scan writes whichever chunk belongs to the
document it started from — the same conflict reached from the other
direction must not become a second flag, and a dismissed flag must stay
dismissed after a re-scan.

`POST /api/ask` returns the full structured trace:

```json
{
  "answer": "Employees may work remotely up to 3 days … [remote_work_policy_v1 § 2. Remote Work Allowance].",
  "citations": ["remote_work_policy_v1 § 2. Remote Work Allowance"],
  "citation_warnings": [], "faithful": true, "unsupported_claims": [],
  "steps": [{"type": "tool_call", "tool": "search_policy_docs", "arguments": {"query": "…"},
             "observation": "[chunk 7] Doc: …", "schema_repairs": ["converted single quotes…"]}],
  "seen_chunk_ids": [7], "tool_calls": 1, "elapsed_s": 4.2, "intent": "policy_question"
}
```

Admin endpoints: `POST /api/documents` (upload, 202-style async), `GET /api/documents`
(status + live job progress + open-flag counts), `DELETE /api/documents/{id}`
(purges chunks + its flags), `GET /api/jobs/{id}`, `GET /api/contradictions`,
`POST /api/contradictions/{id}/status` (`OPEN|resolved|dismissed`),
`POST /api/scan/{doc_id}` (re-run the cascade), `GET /api/chunks/{id}`,
`GET /api/health`. The same operations exist in the CLI
(`python run.py upload|list|delete|ask|chat|scan|contradictions|resolve|status|serve|selftest`).

## 6. Trade-offs (chosen ✂ rejected)

1. **SQLite + numpy cosine ✂ Chroma/pgvector** — one inspectable file, zero
   extra services; brute-force cosine over a cached matrix is sub-ms at this
   scale.
2. **Own thin ReAct loop ✂ LangChain/LlamaIndex** — ~200 lines, every failure
   mode (malformed args → repair → feedback → forced answer) is inspectable
   and unit-tested; a framework would hide exactly the mechanics that matter
   here.
3. **Decision-model gate before the LLM verifier ✂ LLM-on-every-pair** — the
   stage-2 gate makes the cascade cheap: the verifier only sees pairs with
   real conflicting signals. The gate is pluggable (`heuristic`/`remote`/
   `hybrid`) so a deployment can trade a little endpoint cost for gate
   precision without touching the cascade code; the default in-code JEV
   heuristic stays deterministic, explainable and unit-testable without any
   model.
4. **Regex repair + error-as-feedback ✂ strict schema rejection** — strict
   rejection wastes a whole turn on trivially repairable output; pure
   leniency lets garbage flow downstream. Repair first, stay strict at the
   verification layer.
5. **Flag, don't rewrite ✂ auto-refusal on failed faithfulness** — a failed
   judge appends warnings and `faithful=false` instead of destroying the
   answer; the admin sees exactly what was flagged. (A stricter
   auto-refusal mode is a one-line change in `AdminAPI.ask`.)
6. **Sync requests client ✂ async httpx** — the agent loop is naturally
   sequential (next call depends on the last observation); a sync client
   keeps the code simple and thread-safe for background ingestion.
7. **In-process background threads ✂ Celery/RQ** — no broker dependency;
   job rows in SQLite already give resumable progress tracking.
8. **LLM triage ✂ regex/keyword gate** — a lexical gate cannot separate "what
   policies apply to parental leave?" from "what policy documents do you
   have?", and it cannot see a grievance with no policy vocabulary. Measured:
   route accuracy 74.6% (regex) vs 100% (LLM), 15 vs 1 missed refusals, at the
   cost of one cached call (~4.1s, reasoning disabled). The regex gate stays in
   the repo as the ablation baseline, not as a fallback.
9. **RRF over score fusion ✂ weighted sum of cosine + BM25** — the two scores
   live on incomparable scales and per-query normalisation flips ranking when
   one retriever returns a single strong hit. RRF needs ranks only, so it
   cannot be dominated by whichever retriever produces bigger numbers; the
   weights stay available for tuning (`HYBRID_*_WEIGHT`).
10. **One global output ceiling ✂ per-caller `max_tokens` budgets** — a cap is
    not a budget: billing follows actual output, so a 64k ceiling on a 40-token
    verdict costs nothing, while a small cap truncates a reasoning model's JSON
    (this happened: the classifier's verdict was cut mid-object at 700 tokens).
    `MAX_OUTPUT_TOKENS` is therefore global; `reasoning_effort` is the lever
    that actually controls cost/latency per caller.

## 7. If I had two more weeks

1. **Eval harness in CI** — the sets and runners exist (`eval/run_ablation.py`,
   `measure_retrieval.py`, `run_intent_eval.py`, `calibrate_judge.py`); what is
   missing is a `make eval` gate on a frozen corpus so a retrieval or prompt
   change cannot silently regress. The label-only retrieval check needs no LLM,
   so it can run on every commit for free.
2. **Reranker** — small cross-encoder over the top-20 hits (bge-reranker via
   Ollama); usually the single biggest precision jump once the corpus is big
   enough for the fusion step to matter.
3. **Judge verdict cache** — verdicts vary run to run at temperature 0; caching
   them keyed by `sha(question + answer + contexts)` makes demos reproducible and
   re-runs free (the triage layer already does exactly this).
4. **Table-aware extraction** — row-level digests for tables (PDF tables via
   pdfplumber with bbox detection), so "does tier X cover Y?" hits the exact
   cell.
5. **Auth & multi-tenancy** — real roles (admin/employee), per-company
   corpora, an `ADMIN_TOKEN` guard on the admin routes.
6. **Streaming** — SSE token streaming + a live step ticker in the web UI.
7. **Version pinning** — detect "supersedes policy vN" relationships so
   version-vs-version conflicts can be auto-classified as informational.

## 8. Limitations (honest notes)

* `embeddinggemma` has a ~2K-token context; the chunk budget respects it,
  but whole-section embeddings are impossible by design.
* The faithfulness judge adds one LLM call per answered question
  (`VERIFY_ANSWERS=false` trades the safety layer for latency). Verdicts are
  cached per (question, answer, contexts, prompt), so repeats are free and demos
  reproducible; a cached verdict is a replay, not a second opinion.
* The judge's precision is finite (measured TPR 10/10, TNR 8/10 on injected
  claims): a flag is a prompt to look, not a conclusion.
* Scanned/image PDFs are rejected with a clear error rather than OCR'd.
* The heuristic gate is lexical (English-oriented lexicons); it will miss
  contradictions phrased with synonyms it does not know — the verifier
  prompt is the precision backstop, and gate threshold/signals are logged.
* `check_citations` matches doc names fuzzily; a deliberately adversarial
  doc name could false-positive a citation match (warnings, not silent
  acceptance, are the failure mode).
