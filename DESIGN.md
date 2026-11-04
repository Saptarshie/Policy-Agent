# DESIGN.md — Policy RAG Agent ("PolicyPilot")

A citation-grounded Retrieval-Augmented Generation **agent** over admin-uploaded
policy documents (`.txt` / `.md` / `.pdf`). Employees ask questions in natural
language; a ReAct agent searches, reads and answers **only** from the ingested
policies, with verifiable inline citations — and refuses when the corpus is
silent. When an admin uploads a policy that **contradicts** an existing one, a
background **staged cascade** (similarity-gated candidates → a single
decision-model judge → LLM verifier) flags the conflict with citations and an
explanation of *why*. The scan looks inside a document as well as across
documents: an amendment that contradicts its own policy is still a conflict.

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
   │   └─ progress → jobs table   │            │ TOOLS (11)               │
   │  structure-aware chunking    │            │  search_policy_docs      │
   │  batch embed (embeddinggemma)│            │   (top_k XOR min_sim)    │
   └────────┬─────────────────────┘            │  read_policy_chunk       │
            ▼                                  │  list_policy_documents   │
   ┌──────────────────────────────┐            │  get_document_summary    │
   │ SQLite (docs, chunks+vec,    │            │  list_contradictions     │
   │ contradictions, jobs)        │            │  calculate               │
   └────────┬─────────────────────┘            │  scratchpad_* (5)        │
            │ scan (bg thread)                 └────────────┬─────────────┘
            ▼                                              ▼
   ┌──────────────────────────────────────────┐  ┌────────────────────────┐
   │ CONTRADICTION CASCADE (background)        │  │ ANSWER PATH (no judge) │
   │  1. candidates: every passage >=          │  │  citation resolution   │
   │     SCAN_MIN_SIMILARITY, intra- AND        │  │  against the store;    │
   │     cross-document                        │  │  warnings never silent │
   │  2. DECISION MODEL — the only judge,      │  └────────────────────────┘
   │     batched (6/request) + threaded        │
   │     below threshold ──▶ drop (logged)     │
   │  3. LLM VERIFIER — only for survivors     │
   │  4. ADMIN FLAG: citations + why + score   │
   └──────────────────────────────────────────┘
```

**Data flow — upload → index.** The API validates type/size, stores the file
and returns `200` immediately with a `doc_id`; a daemon thread runs the
rolling-window pipeline (§2), persisting progress into a `jobs` row
(`stage`, `window i/n`, rolling-summary checkpoint) that the CLI and web UI
poll. On success a second daemon thread starts the contradiction cascade
(§4) against every other ingested document.

**Data flow — query → answer.** One triage call classifies the raw input and
decides the route (§3.1) — a greeting or a document listing is answered
directly, input that is not a question about the ingested documents and prompt
injection are refused, everything about the documents goes to the ReAct loop. The agent receives the *de-noised*
question plus the triage query plan as a second system message, thinks, emits one
or more tool calls (parallel when they do not depend on each other), **reads the
observations and decides the next action** (refine the query, open a full chunk,
or answer) — up to `AGENT_MAX_STEPS`, then a forced grounded final answer.
Citations are resolved against the store deterministically (`citation_warnings`,
never silent) and the same resolver is what the UI opens when a reader clicks a
chip; there is **no post-answer judge in this path** — grounding is the prompt
contract plus that check (§3.7). The judge runs offline, in the ablation harness
only.

## 2. Chunking & rolling-window ingestion (big documents)

Documents can be huge; **nothing is ever fed whole** to the parser LLM.

* **Reading**: `txt`/`md` decoded with UTF-8 → latin-1 fallback; markdown
  decoration stripped but headings kept; `pdf` via `pypdf` page-by-page
  (image-only PDFs are rejected with a clear error).
* **Windows**: the text is cut on *paragraph* boundaries into ~`INGEST_WINDOW_CHARS`
  windows with a ~200-char carry-over tail. Per window **one bounded LLM call**
  does (a) structure extraction (title, normative statements, key terms,
  numbers+units+context) and (b) a summary of that window. Two shapes of that
  loop, both keeping the chunks identical:
  * `INGEST_PARALLEL_WINDOWS=1` — the original **cascade**: `prev_summary +
    window → new summary (≤120 words)`, each window parsed in the context of
    everything before it (recursive/rolling-document LLMs).
  * `INGEST_PARALLEL_WINDOWS>1` (default) — windows are parsed **concurrently**
    and independently (the chain only ever fed *metadata*, never the chunks),
    then one extra call folds the per-window summaries into the rolling summary.
    Measured 501.7s → 78.2s for the four-document benchmark pack.

  The parser runs at `INGEST_REASONING_EFFORT=off` by default: it extracts, it
  does not reason, and extended reasoning cost 4.2× wall time for the same
  chunk text (see `eval/ABLATION.md` §10).
  If the LLM reply is unparseable even after regex repair, a heuristic
  fallback summarizes and a warning is recorded — ingestion never dies.
  `INGEST_HEURISTIC_MAX_CHARS` skips the LLM parse entirely for documents at or
  below that size.
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
   queries plus entities and a table/tier hint. `META_DOCUMENTS` means
   **inventory only** (which documents exist); questions about what the
   documents *say* — including whether they contradict each other — are policy
   questions and must reach the agent, because the direct reply path can only
   list documents. **It refuses only input that is not a document question
   (world knowledge, code, personal advice) and prompt injection — never a
   question about the corpus**, whatever its subject (HR policy or contract
   clause), because it has not read the corpus; coverage refusals belong to the
   grounded agent. Decisions are
   cached on disk (question + prompt version + a **fingerprint of the prompt
   text** + model + reasoning effort, so editing the prompt invalidates its
   cached verdicts instead of replaying them), so
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
4. **Working memory and arithmetic as tools** — `calculate` evaluates an
   AST-whitelisted arithmetic expression over named variables (never `eval()`:
   the expression is model output whose context contains ingested documents, and
   `eval` there reads files and environment variables), and `scratchpad_*` gives
   one question its own notes (12 notes, 4k chars each, never persisted, never
   evidence). Neither is mandatory: the prompt names the agent's own
   `MULTI_HOP_COMPARISON` as the trigger, so a simple lookup stays two tool calls
   (~10s) while the complex question spends 18 calls and a minute of work. The
   loop already executes every call in a turn, so parallel tool calls needed only
   a prompt rule - the complex question now batches 7 turns.
5. **Citation verification (deterministic)** — every `[doc § section]` /
   `[doc #chunk_id]` in the final answer is resolved against the store
   (fuzzy doc-name match, chunk-id existence); failures surface as
   `citation_warnings`, never silently.
6. **Citations resolved deterministically, never guessed** (`app/citations.py`) —
   normalised exact match, unique token subset, then fuzzy with a margin; an
   ambiguous reference becomes a warning. The old first-match loop attributed
   "Remote Work Policy v2" to `remote_work_policy_v1`. The *same* resolver answers
   `GET /api/citations/resolve`, so what the check believes and what a reader sees
   cannot drift apart: the UI opens the passage behind a chip, rendered from the
   stored markdown (raw toggle for the exact bytes) with its breadcrumb and chunk
   id, or names the sections the document really has when the reference does not
   exist. A section split over several chunks returns **all** its parts (`parts[]`),
   because showing only the first chunk made a continuing table look truncated —
   the one way a verification panel could mislead the person verifying.
7. **No post-answer verifier** — a faithfulness judge was built, calibrated
   (TPR 10/10, TNR 8/10 on injected claims) and measured end to end, then removed
   from the answer path: at model parity it added ~6s p50, changed no pass rate,
   and flagged about one honest answer in five. Grounding now rests on the prompt
   contract plus the deterministic citation check above; the judge survives as an
   offline audit (`eval/calibrate_judge.py`). The full measurement is in
   `dev_docs/DATAFLOW.md` §6 and `eval/ABLATION.md`.
8. **The trace is streamed, not replayed** (`POST /api/ask/stream`) — the agent
   accepts an `on_event(kind, payload)` callback and emits `turn`, `step`
   (thought / tool_call / tool_error / final), `stage` (triage) and `error` as it
   works; the route bridges that callback to Server-Sent Events (worker thread →
   queue → generator, 15s keep-alive comments), and the UI appends a collapsible row
   per step. Streaming is additive: `POST /api/ask` still returns the whole result,
   and the UI falls back to it if a proxy buffers the stream. Editing a step's
   observation for the stream (1.2k chars) does not change the stored result
   (300 chars), so the payload shape the evals read is untouched. The UI treats a
   stream that ends without a `final` frame, an `error` frame and an unreachable
   server as three distinct terminal states — a request can stop, and the page says
   which way it stopped rather than spinning.
9. **Endpoint hardening** (`app/llm.py`) — retries with backoff on 429/5xx;
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

Runs in the background after every successful ingestion, for each chunk of the
new document — **against the rest of the corpus, including the same document**
(`SCAN_CROSS_DOC_ONLY=false` is the default; an amendment contradicting its own
policy is a real conflict).

| Stage | Component | Cost | What it does |
|---|---|---|---|
| 1 | **Candidate selection** (`rag.find_similar_chunks`) | embeddings already stored | every passage whose **stored** vector is at or above `SCAN_MIN_SIMILARITY` (0.65); `SCAN_MAX_CANDIDATES` (200) is a logged cost valve |
| 2 | **Decision model** (`contradictions/gate.py`) | batched calls to a *separate* endpoint, `DECISION_BATCH_SIZE` pairs each | scores each pair and **gates** escalation; the only judge |
| 3 | **LLM verifier** (`contradictions/verify.py`) | 1 call per gate-passing pair, `SCAN_WORKERS` in flight | strict-JSON verdict: is it really a contradiction? |
| 4 | **Admin flag** (`contradictions/tracker.py`) | — | writes the conflict with both passages, why, score and signals |

### 4.1 Candidates are chosen by similarity, not by count

The first version pulled the `CONTRA_TOP_K` most similar passages and escalated at
most `SCAN_MAX_PAIRS_PER_CHUNK` per chunk. That is the wrong instrument for
conflicts: on this corpus the real leave-cap conflict ranks **3rd of 4** at 0.678
— inside the cut by luck — and in a 60-chunk document a contradicting passage can
rank 41st among passages that are all similar. The floor now decides, and the cap
only bounds cost (it logs when it truncates).

One subtle defect is worth recording because it defeated the floor silently: the
scan embedded the chunk text as a **query** and compared it to stored vectors.
With a model that prefixes queries and documents differently (embeddinggemma
does), that measures lower — the same pair scores **0.601 query-vs-stored vs
0.678 stored-vs-stored** — so a 0.65 floor excluded exactly the conflict it had
been calibrated on. `db.similar_to_chunk()` compares stored vectors to stored
vectors, and is also one embedding call cheaper per chunk.

### 4.2 One judge, and it is the model

`DECISION_MODE` is gone. The decision model (`DECISION_MODEL_BASE_URL` /
`_API_KEY` / `_NAME`, any OpenAI-compatible endpoint) is the only thing that may
pass or drop a pair.

The in-code lexical judge (`JEVStyleGate`: topical overlap, normative and
permission flips, numeric conflicts, antonym pairs) still runs, but its output is
a **signal list handed to the model as evidence** — never a verdict. This was a
deliberate reversal: a lexical gate cannot see that "staff may work from home two
days" and "employees are expected on site three days a week" conflict, and
running it as a decider meant a corpus could look clean because word overlap was
low.

With no endpoint configured the gate answers `decided_by="unavailable"` and the
scan prints **INCOMPLETE, not clean**. A scan that judged nothing must never look
like a clean corpus.

### 4.3 Batched and threaded, without moving the verdicts

| configuration | wall | decision-model requests | verifier calls | flags |
|---|---|---|---|---|
| serial, one request per pair | 216.2s | 96 | 16 | 6 |
| batched (6 pairs/request) | 52.7s | 20 | 7 | 6 |
| batched + 4 threads | **29.8s** | 21 | 6 | 6 |

The endpoint takes many **questions over one shared state**, not many states (a
list for `state` returns 200 but concatenates). So a batch is one state listing
`PAIR 1..n` with one question per pair, and the passages become the shared
prefix: 810 input tokens instead of 2029 for six pairs.

Batching alone moves verdicts — scores shifted up to 0.24 and 2 of 6 decisions
flipped across the threshold in a controlled check — so a batched score is only a
first pass: any pair within `DECISION_RECHECK_BAND` (0.15) of the threshold is
re-asked in its own request. That gave 6/6 agreement with per-pair scoring at
7.3× the speed (`eval/measure_scan.py`, `dev_docs/OVERNIGHT_LOG.md` §10).

Both stages are HTTP calls, so `SCAN_WORKERS` runs batches and verifications
concurrently; SQLite writes stay on the calling thread and in scan order, which
is why the flags are identical.

Score ≥ `JEV_GATE_THRESHOLD` (default 0.5) ⇒ escalate to the verifier, whose
prompt is **precision-biased**: different scope, complementary rules, mere topical
overlap and explicit supersession are *not* contradictions.

**Nothing is silently dropped.** If the verifier can't run (endpoint down,
unparseable) but the score is well above threshold, the pair is recorded as
**UNVERIFIED** for admin review. Duplicate pairs are skipped; deleting a document
purges its flags; admins resolve/dismiss flags from CLI or web UI, and the agent
can read them with `list_contradictions`.

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

`POST /api/ask` returns the full structured run:

```json
{
  "answer": "Employees may work remotely up to 3 days … [remote_work_policy_v1 § 2. Remote Work Allowance].",
  "citations": ["remote_work_policy_v1 § 2. Remote Work Allowance"],
  "citation_warnings": [], "faithful": null, "answer_shape": "answer",
  "steps": [{"type": "tool_call", "tool": "search_policy_docs", "arguments": {"query": "…"},
             "observation": "[chunk 7] Doc: …", "schema_repairs": ["converted single quotes…"]}],
  "seen_chunk_ids": [7], "judge_context_ids": [],
  "scratch_notes": 0, "parallel_turns": 0, "declined_parts": [], "unsupported_claims": [],
  "tool_calls": 1, "elapsed_s": 4.2, "error": null,
  "intent": "POLICY_DIRECT_LOOKUP", "intent_route": "REACT_LOOP",
  "intent_plan": {"question": "…", "queries": ["…"], "entities": []}
}
```

`faithful` is `null` on this path by design — nothing judges the answer (item 7
above); the field is populated only when the offline audit harness runs, so a
client must not read `null` as "unfaithful". `intent_route` is the triage
decision, which is how an answer with zero tool calls can be legitimate
(`DIRECT_REPLY` for a document listing, `FAST_REFUSE` for a non-document request).

Endpoints: `POST /api/documents` (upload, returns immediately and ingests in the
background), `GET /api/documents` (status + live job progress + open-flag counts),
`DELETE /api/documents/{id}` (purges chunks, flags and the stored upload),
`GET /api/jobs/{id}`, `POST /api/ask`, `POST /api/ask/stream` (the same run as SSE:
`stage` / `turn` / `step` / `final` / `error`, 15 s keep-alive comments),
`GET /api/citations/resolve?cite=…` (the passage behind a citation, or `reason` +
`candidates`), `GET /api/chunks/{id}`, `GET /api/contradictions`,
`POST /api/contradictions/{id}/status` (`OPEN|resolved|dismissed`),
`POST /api/scan/{doc_id}` (re-run the cascade), `GET /api/health`. The same
operations exist in the CLI
(`python run.py upload|list|delete|ask|chat|scan|contradictions|resolve|status|serve|selftest`).

## 6. Benchmarks — what is measured, on what, and what it misses

Three external packs (two scored, one not yet), each a directory of documents plus
questions with expected behaviour. `eval/run_benchmark.py` indexes the pack into its own `DATA_DIR`, asks
every question through `AdminAPI.ask` (the same path the UI uses — no test-only
route, nothing from the pack embedded in a prompt), judges each answer, resolves
every citation it produced, and writes JSONL rows plus a summary JSON under
`eval/results/`.

| pack | docs | questions | what it exercises |
|---|---|---|---|
| `hr_rag_policy_pack_and_benchmark` | 4 | 20 | multi-hop HR policy: caps that bind, eligibility windows, worked examples, out-of-corpus refusals |
| `complex_contract_reasoning_pack` | 4 | 20 | commercial contract / SLA arithmetic: credits, caps, proration, definitions that change a clause's scope |
| `adversarial_policy_arbitrage_pack` | 4 | 20 | questions whose obvious reading is wrong once a definition elsewhere in the pack is applied — present, not yet scored |

Run of record, both sequential on the same code
(`benchmark_20261105_012006`, `…20261105_010648`):

| pack | correct | + partial | citation recall (doc / doc+section) | mean / p50 / p95 | QA wall |
|---|---|---|---|---|---|
| HR policy | 19/20 | 20/20 | 29/29 · 29/29 | 17.0 / 14.9 / 31.5 s | 340.0 s (67.8 s at `--workers 10`) |
| contract | 16/20 | 18/20 | 41/42 · 39/42 | 28.5 / 23.4 / 55.8 s | 570.8 s |

Contract pack rubric (answer /4, reasoning /3, citations /2, uncertainty /1):
**9.30 / 10** = 3.55 · 2.80 · 2.00 · 0.95. The rubric is reported separately from
"correct" on purpose: four answers are right in shape and incomplete, which a binary
verdict hides and a 3.55 shows. One contract question hit the step budget (89 s, 20
tool calls) and still graded partial, which is the honest cost of a fixed
`AGENT_MAX_STEPS`.

What these numbers do **not** establish:

* **One run is one sample.** The single miss moves between runs (MH-12, MH-19,
  MH-01 over six repeats of the same code); `eval/repeat_question.py` reports a hit
  rate (`MH-01`: 4/6) because the honest statement is "usually right", not "right".
* **20 questions = 5 points per question.** A 95% and a 90% are one answer apart;
  the packs support "there is no systematic failure", not fine-grained ranking.
* **The judge is a model.** It reads the answer against a reference and a rubric;
  it scores citation *format and resolution* well and truthfulness less well, so a
  confident, well-cited, wrong answer can still pass.
* **Latency is the endpoint's.** p50 15–28 s is dominated by 4–7 model round trips
  on a hosted endpoint at `reasoning_effort` default — the same contract pack
  measured p50 18.9 s concurrent and 23.4 s sequential — so the wall column
  describes a run, not a service level.
* **Nothing is measured on scale.** Every pack is 4 documents; retrieval
  comparisons on this corpus saturate at k=6 (§2.1), which is stated rather than
  papered over.

## 7. Trade-offs (chosen ✂ rejected)

1. **SQLite + numpy cosine ✂ Chroma/pgvector** — one inspectable file, zero
   extra services; brute-force cosine over a cached matrix is sub-ms at this
   scale.
2. **Own thin ReAct loop ✂ LangChain/LlamaIndex** — ~200 lines, every failure
   mode (malformed args → repair → feedback → forced answer) is inspectable
   and unit-tested; a framework would hide exactly the mechanics that matter
   here.
3. **Decision-model gate before the LLM verifier ✂ LLM-on-every-pair** — the
   stage-2 gate makes the cascade cheap: the verifier only sees scores the model
   considers conflict-shaped. The judge is a *separate, cheaper* endpoint than the
   answer model, and the in-code lexical gate survives as a signal provider
   (explainable, unit-testable, free) rather than as a decider — a lexical
   verdict is a language decision made by word overlap, which is exactly the
   class of shortcut this codebase removes.
4. **Regex repair + error-as-feedback ✂ strict schema rejection** — strict
   rejection wastes a whole turn on trivially repairable output; pure
   leniency lets garbage flow downstream. Repair first, stay strict at the
   verification layer.
5. **Warn, don't rewrite ✂ auto-refusal on unverified output** — the answer is
   never destroyed after the fact. What the system *can* check it does check
   (citation resolution, chunk ids, the calculator's arithmetic), and what it
   cannot check it says so with `citation_warnings`; a second model was measured
   as an unreliable arbiter, so it does not get to veto an answer
   (item 7 in §3).
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

## 8. If I had two more weeks

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
6. **Token-level streaming** — the *steps* already stream (§3.8) and the UI shows
   elapsed time and silence, so the gap is only the answer text: stream deltas into
   the bubble instead of printing it on `final`. Worth doing for perceived
   latency, not for correctness — and it needs a per-turn SSE fan-in that the
   current one-event-per-step bridge does not have.
7. **Version pinning** — detect "supersedes policy vN" relationships so
   version-vs-version conflicts can be auto-classified as informational.

## 9. Limitations (honest notes)

* `embeddinggemma` has a ~2K-token context; the chunk budget respects it,
  but whole-section embeddings are impossible by design.
* The faithfulness judge is **not on the answer path**. It runs only in the offline
  audit (`eval/calibrate_judge.py`), one LLM call per answered question, and its
  verdicts are cached per (question, answer, contexts, prompt) so repeats are free
  and demos reproducible; a cached verdict is a replay, not a second opinion.
* The judge's precision is finite (measured TPR 10/10, TNR 8/10 on injected
  claims): a flag is a prompt to look, not a conclusion. That finite precision is
  why it is an audit tool and not a gate.
* Citation resolution proves a reference *exists*, not that the passage *supports*
  the sentence. A well-cited wrong claim passes every deterministic check; the
  clickable panel exists so a reader can settle that in one click instead of
  trusting the system's own word for it.
* Benchmark numbers are single samples from a sampling model: the one miss moves
  between runs, so per-question hit rates (`eval/repeat_question.py`) are the honest
  unit, not one run's accuracy (§6).
* Scanned/image PDFs are rejected with a clear error rather than OCR'd.
* The lexical signals are English-oriented and can *bias* the decision model's
  score (they no longer decide anything). A pair the heuristics find boring still
  reaches the model, which reads the passages themselves.
* Scan cost grows with the number of pairs above the similarity floor, not with
  the document count; `SCAN_MAX_CANDIDATES` truncates (and says so) rather than
  growing without bound. Incremental "only pairs involving changed chunks"
  scanning is not implemented.
* Batching perturbs decision-model scores by up to ~0.25; the re-check band
  contains that, but a pair whose batched score lands far on the wrong side of
  the threshold is not re-asked.
* `check_citations` matches doc names fuzzily; a deliberately adversarial
  doc name could false-positive a citation match (warnings, not silent
  acceptance, are the failure mode).
