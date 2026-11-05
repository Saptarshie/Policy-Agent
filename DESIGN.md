# DESIGN.md — Policy RAG Agent ("PolicyPilot")

A citation-grounded Retrieval-Augmented Generation **agent** over admin-uploaded
policy documents (`.txt` / `.md` / `.pdf`). Employees ask questions in natural
language; a ReAct agent searches, reads and answers **only** from the ingested
policies, with verifiable inline citations — and refuses when the corpus is
silent. When an admin uploads a policy that **contradicts** an existing one, a
background **staged cascade** (similarity-gated candidates → a single
decision-model judge → LLM verifier) flags the conflict with citations and an
explanation of _why_. The scan looks inside a document as well as across
documents: an amendment that contradicts its own policy is still a conflict.

**Models**

| Role                          | Provider (default)                                                                                  | Why                                                                                                                     |
| ----------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Embeddings                    | Ollama `embeddinggemma`                                                                             | local, free, private (policy data!); task-steered prefixes applied client-side                                          |
| Chat / reasoning / tool calls | any **OpenAI-compatible endpoint** (Ollama, LM Studio, vLLM, Groq, OpenRouter, OpenAI …) via `.env` | the agent needs decent tool-calling + JSON; `MAX_OUTPUT_TOKENS=65536` with automatic halving if the endpoint caps lower |

Everything is env-configurable (`.env`, see `.env.example`). Every unit test
and the offline smoke run use deterministic fakes — **no key needed to develop
or run CI**.

**Stack, and why it is not the suggested one.** Python 3.13 + FastAPI + SQLite
(vectors in float32 BLOBs, keyword search through FTS5) + one HTML/JS file for the
UI. Embeddings run locally through Ollama; the chat model is any
OpenAI-compatible endpoint, so a free tier or a laptop model is enough. The brief
suggests LangChain/LlamaIndex plus a vector store; both were rejected on purpose
(§7.1, §7.2): the agent loop here is ~200 lines whose every failure mode is
unit-tested, and brute-force cosine over a cached matrix is sub-millisecond at this
size, so a framework would hide the mechanics the review is about, and a vector
service would add an install step to a ten-minute README. Hardware limits only the
local embedding model (a CPU is fine); nothing here needs a GPU.

**What this document covers.** The brief asks for six things in `DESIGN.md`; each
has a section below, and the parts that go past the ask are what pushed the page
count past the suggested 2–4.

| the brief asks for                                                                                     | section here        | what is beyond the ask                                                                 |
| ------------------------------------------------------------------------------------------------------ | ------------------- | -------------------------------------------------------------------------------------- |
| architecture, data flow upload → index → query → answer, a diagram                                     | §1                  | failure paths, the SSE trace, citation resolution                                      |
| chunking & retrieval: how documents are split, what metadata is stored, how top chunks are chosen, why | §2, §2.1            | a measured dense-vs-sparse-vs-hybrid comparison, including the cell where hybrid loses |
| grounding, and what happens when retrieval is weak                                                     | §3.1–3.7, **§3.10** | no post-answer judge (measured away), refusal text pinned by a test                    |
| schema & APIs, and why that schema                                                                     | §5                  | error codes, `parts[]`, why `faithful` is `null`                                       |
| trade-offs, at least two rejected                                                                      | §7                  | ten, each with the measurement that decided it                                         |
| what to harden in two more weeks                                                                       | §8                  | benchmark-in-CI is the headline                                                        |

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
   │  windows (~6k ch, parallel)  │                         │ tool_call loop
   │   ├─ per-window LLM parse    │                         ▼
   │   ├─ ONE document brief      │            ┌──────────────────────────┐
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
            ▼                                               ▼
   ┌──────────────────────────────────────────┐  ┌────────────────────────┐
   │ CONTRADICTION CASCADE (background)       │  │ ANSWER PATH (no judge) │
   │  1. candidates: every passage >=         │  │  citation resolution   │
   │     SCAN_MIN_SIMILARITY, intra- AND      │  │  against the store;    │
   │     cross-document                       │  │  warnings never silent │
   │  2. DECISION MODEL — the only judge,     │  └────────────────────────┘
   │     batched (6/request) + threaded       │
   │     below threshold ──▶ drop (logged)    │
   │  3. LLM VERIFIER — only for survivors    │
   │  4. ADMIN FLAG: citations + why + score  │
   └──────────────────────────────────────────┘
```

**Data flow — upload → index.** The API validates type/size, stores the file
and returns `200` immediately with a `doc_id`; a daemon thread runs the
windowed ingestion pipeline (§2), persisting progress into a `jobs` row
(`stage`, `window i/n`) that the CLI and web UI
poll. On success a second daemon thread starts the contradiction cascade
(§4) against every other ingested document.

Nothing about a document is known in advance: document names, section headings,
citation strings and the corpus inventory all come from the store at query time,
so a policy uploaded during a review is answerable as soon as its job says `done`.
The two benchmark packs in §6 are the evidence — neither existed when the pipeline
was written, and nothing in a prompt names a document.

**Data flow — query → answer.** One triage call classifies the raw input and
decides the route (§3.1) — a greeting or a document listing is answered
directly, input that is not a question about the ingested documents and prompt
injection are refused, everything about the documents goes to the ReAct loop. The agent receives the _de-noised_
question plus the triage query plan as a second system message, thinks, emits one
or more tool calls (parallel when they do not depend on each other), **reads the
observations and decides the next action** (refine the query, open a full chunk,
or answer) — up to `AGENT_MAX_STEPS` **turns**, then a forced grounded final answer. A
turn is one model call and may carry several parallel tool calls, so the step budget
and the tool-call count are different numbers: benchmarks report both (`turns`, and
`step_entries` = thoughts + one entry per tool call + final) because reporting only
entries made a 37-call question look like it had used 38 steps against a cap of 20.
Citations are resolved against the store deterministically (`citation_warnings`,
never silent) and the same resolver is what the UI opens when a reader clicks a
chip; there is **no post-answer judge in this path** — grounding is the prompt
contract plus that check (§3.7). The judge runs offline, in the ablation harness
only.

## 2. Chunking & windowed ingestion (big documents)

Documents can be huge; **nothing is ever fed whole** to the parser LLM.

**What runs, in order — and what is _not_ an agent.** Indexing is a deterministic
pipeline on a daemon thread (`IngestPipeline.ingest`, `app/ingest.py:88`). There is
no tool loop and no model deciding what to do next; the only model call is one
bounded JSON extraction per window, and it returns _metadata_:

```
read_document(path)              # txt/md/pdf -> text           readers.py
split_into_windows(text, 6000)   # paragraph windows + 200-char carry tail
_parse_windows(...)              # 1 chat_json call per window, N in parallel
                                 #   -> {window_title, statements, key_terms,
                                 #       numbers}             (metadata only)
_doc_brief(text, headings)       # 1 call per document -> documents.summary
chunk_text(text, 1400, 2200)     # <-- pure Python, on the FULL text, no model
embed_documents(chunk_texts)     # batches of EMBED_BATCH_SIZE
add_chunk(...) per chunk         # SQLite row + FTS5 trigger from the same INSERT
```

The arrow that matters: **`chunk_text` never sees the parser's output.** Chunk
boundaries come from the document's own structure (headings, blank lines, sentences),
so a chunk is a verbatim slice of the uploaded file — which is what makes a citation
provable, and what makes re-ingestion reproducible (the parallel and sequential parse
modes produce byte-identical chunks, `sha256 f0d72b17…`, `eval/compare_parse_modes.py`).
An LLM boundary-finder would be a semantic chunker: it would have to re-emit the text
to store it, and a citation would then quote a model's paraphrase instead of the
policy.

**The document brief — what the triage layer and the agent know about a document.**
One bounded call per document reads its opening `INGEST_BRIEF_WORDS` words (default 1000) plus its complete heading list, and writes at most three sentences (≤70 words):
what the document covers, who it applies to, and what it can answer. It is stored on
`documents.summary` and mirrored into `docs/meta_json` as `brief`.

There is no rolling summary any more. The old design chained window _i_'s summary
into window _i+1_'s parser prompt and then made one merge call; nothing in that chain
ever reached a chunk, and the triage layer never saw the result, so it cost calls to
produce a paragraph that only `get_document_summary` could quote. Windows now parse
independently (no ordering at all), and the one summary-shaped call per document is
the brief — which is small, bounded, and consumed by something that matters.

What the brief is and is not:

- **Not chunk input.** Chunk text stays a verbatim slice of the file: the same
  document indexed with two different briefs produces byte-identical chunks
  (`tests/test_corpus_brief.py` asserts exactly that).
- **Not embedded, not searched.** Retrieval sees chunk vectors and chunk text.
- **Consumed at four points**: the triage layer (all ready documents, at most 8, 300
  chars each, as a `CORPUS` block — §3.1), `get_document_summary`, `python run.py list`
  and the UI's document cards, and the offline judge's inventory, so the judge does
  not flag a correct statement about _which documents exist_ as unsupported.
- **Bounded twice, deliberately** — words read on the way in, sentences written on
  the way out — because this text goes into the triage prompt for every document on
  every question. A prompt that grows with the corpus is a prompt that eventually
  stops classifying.
- **If the call fails**, the fallback is deterministic and honest: the heading list
  plus the document's first two sentences, with a warning recorded on the job. A
  missing brief is shown as `(no brief available)` rather than as an empty line a
  model would read as "this document covers nothing".

- **Reading**: `txt`/`md` decoded with UTF-8 → latin-1 fallback; markdown
  decoration stripped but headings kept; `pdf` via `pypdf` page-by-page
  (image-only PDFs are rejected with a clear error).
- **Windows**: the text is cut on _paragraph_ boundaries into ~`INGEST_WINDOW_CHARS`
  windows with a ~200-char carry-over tail. Per window **one bounded LLM call**
  does structure extraction for that window: title, normative statements, key
  terms, and numbers with units and context. Windows are parsed **concurrently and
  independently** (`INGEST_PARALLEL_WINDOWS` is a worker count, nothing more): no
  window ever sees another window's output, because nothing a window produces is
  allowed near the chunk text, and the chain that used to link them fed only
  metadata. Measured 501.7s → 78.2s for the four-document benchmark pack.

  **The document brief.** One further call per document reads the opening
  `INGEST_BRIEF_WORDS` words (default 1000) plus every heading, and writes at most
  three sentences about what the document covers. It is stored on
  `documents.summary` and is what the triage layer, `get_document_summary`, the
  CLI/UI document cards and the offline judge's inventory read. The rolling summary
  it replaced was reachable only through `get_document_summary`, cost a merge call
  per document, and told the classifier nothing; this one is bounded on both ends
  (words in, sentences out) because it is injected into a prompt that runs for every
  question. If the call fails, the fallback is the heading list plus the opening
  sentences, with a warning on the job - never an empty summary.

  The parser runs at `INGEST_REASONING_EFFORT=off` by default: it extracts, it
  does not reason, and extended reasoning cost 4.2× wall time for the same
  chunk text (see `eval/ABLATION.md` §10).
  If the LLM reply is unparseable even after regex repair, a heuristic
  fallback summarizes and a warning is recorded — ingestion never dies.
  `INGEST_HEURISTIC_MAX_CHARS` skips the LLM parse entirely for documents at or
  below that size.

- **Progress tracking**: an `ingest` job row is updated after every window
  (`done/total/stage` in `detail`).
- **Chunking** (after windows): structure-aware, never fixed-size —
  boundaries in order of preference: headings (markdown `#`, numbered
  sections, ALL-CAPS titles, `Section/Article/Policy…` labels, underlined
  titles) → blank-line paragraphs → lines → sentences, with a hard-split
  last resort. Every chunk carries a breadcrumb `doc > section > subsection`
  and soft target ~1,400 chars / hard cap ~2,200 (comfortably inside
  embeddinggemma's context). Tiny trailing chunks merge into their
  predecessor.
- **What is stored per chunk** (the `chunks` table, and what retrieval and
  citation actually use): `doc_id`, `chunk_index`, `section` (the heading the
  chunk starts under), `breadcrumb` (`doc > section > subsection`), `text`,
  `char_count`, the `embedding` as a float32 BLOB with its `norm` precomputed
  (so cosine is a dot product), and `created_at`. The same text is mirrored into
  an FTS5 external-content table for keyword search (§2.1) — one copy of the
  text, two indexes. `section` is what a citation points at; `chunk_index`
  orders a section's parts when a section spans chunks, which is exactly what
  `GET /api/citations/resolve` returns.
- **Embedding**: batches of `EMBED_BATCH_SIZE` through Ollama's `/api/embed`
  (legacy `/api/embeddings` fallback), with embeddinggemma's task prefixes
  (`task: search result | query: …` / `title: none | text: …`) applied
  client-side — Ollama does **not** add them for you.

### 2.1 Retrieval: dense, sparse, and the fusion between them

Two retrievers run over the same chunks (`app/rag.py`, `app/db.py`):

- **dense** — cosine over the stored embeddinggemma vectors. It handles
  paraphrase ("time off" ≈ "leave", "can they do that?" ≈ "forfeiture rule"),
  which is most of what employees type.
- **sparse** — SQLite **FTS5 + BM25** over the same chunk text, kept in sync by
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

**Depth was the real ceiling, not the fusion.** On the adversarial pack the
reference's own sections sit at vector ranks 5-32 for the question that needs them, and
the tool asked for k=6 — so a correct section could exist in the index and never reach the
observation. `eval/measure_retrieval.py --pack` scores the reference sections directly
(no LLM), which is how this was measured rather than argued:

```
expected sections retrieved (adversarial pack, 42 expected sections over 20 questions)
mode        sec@6    sec@12   sec@24    MRR
vector      28/42    35/42    39/42     0.559
keyword     27/42    32/42    36/42     0.484
hybrid      28/42    35/42    39/42     0.561      (vector weight 1.5, keyword 0.5)
```

Two knobs changed because of that table: `RAG_TOP_K` 6 → **12**, and the fusion weights
from equal to **vector 1.5 / keyword 0.5**. The weighting matters for a specific failure:
with equal weights a keyword list that does not contain the section demotes a strong
vector hit (vector rank 5 → hybrid rank 12; vector 8 → hybrid 26 on two questions), and
equal-weight hybrid scored *worse* than vector alone at every k. Weighted, hybrid matches
vector at k=12 and never loses to it.

Three sections still never appear at k=24 (`1.3 Related persons`, `9.2 Indicators`,
`8. Programme look-through`): their wording and the question's wording share almost
nothing, which is a vocabulary problem, not a fusion problem — and the honest candidate
for it is a section-heading index, not a graph database (the numbers behind that judgement
are in ABLATION §18).

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
   documents _say_ — including whether they contradict each other — are policy
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
   **It is shown what the corpus contains** — the bounded `CORPUS` block above, one
   line per ready document. That is orientation, not evidence, and the prompt says so
   in as many words: never refuse because a brief does not mention the topic (a digest
   omits almost everything by design), never answer from a brief, and `FAST_REFUSE`
   only for input that is not a document question or is an injection attempt. The
   verdict cache keys on a fingerprint of the corpus block, so uploading a policy
   invalidates classifications instead of replaying verdicts decided against a
   different corpus. What this bought is measured in §6.
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
   "Remote Work Policy v2" to `remote_work_policy_v1`. The _same_ resolver answers
   `GET /api/citations/resolve`, so what the check believes and what a reader sees
   cannot drift apart: the UI opens the passage behind a chip, rendered from the
   stored markdown (raw toggle for the exact bytes) with its breadcrumb and chunk
   id, or names the sections the document really has when the reference does not.
   A reference to a *parent* heading ("8. Stacking and double funding" when the chunker
   stored "8.1 Permitted stacking") resolves to every subsection, in document order,
   because a heading that only exists as subsections is still a real target; the panel
   labels that case `descendant` rather than pretending it matched exactly
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
   retry. `MAX_OUTPUT_TOKENS` is one global _ceiling_, not a budget — billing
   follows actual output, so a large ceiling costs nothing and small
   per-caller caps only create truncation risk (reasoning tokens are billed
   against it). `REASONING_EFFORT` is the real cost lever: the classifier runs
   with reasoning off, the agent leaves the provider default in place.

10. **Citation recovery — the sources the answer forgot to mention** — the parsed
   citation list is a lower bound. The agent reads more than it quotes: a price table, an
   exclusion clause, the section that qualifies a number. After the answer is parsed, the
   sections the agent actually *read* (its own `seen_chunk_ids`) that the answer did not
   cite are offered to the decision model, together with the final answer, as one request:
   a shared state listing the answer and up to `CITATION_RECOVERY_MAX_CANDIDATES`
   candidates, and one yes/no question per candidate ("does this passage state a rule,
   condition, number or qualification the answer relies on?"). Candidates scoring at or
   above `CITATION_RECOVERY_THRESHOLD` are appended to `citations`, at most
   `CITATION_RECOVERY_MAX_ADDED` of them, highest first; the answer text is never edited.
   The recovered ones are recorded in a `citations_recovered` step with every score, so an
   audit can tell what the answer model wrote from what a second model added.

   Three properties make it safe to run on every answer: it **cannot invent** a citation
   (candidates come from chunk ids the agent saw, so a hallucinated name in the model's
   reply matches nothing), it **cannot remove or reorder** what the answer wrote (it only
   appends), and a decision-model failure is a no-op rather than an error. It is also
   bounded: 24 candidates × 400 chars, one request, ~20s timeout.

   What it measured, honestly (`eval/ABLATION.md` §17): the decision model separates real
   sources from coincidences in the mean (0.87 for the reference's sections vs 0.16 for the
   rest) but its tail is noisy — some irrelevant sections score 0.98 — which is why the
   addition is capped. End to end on the adversarial pack it added **7 citations across 6
   of 20 answers, none of them a section the reference lists**: real sections from the right
   documents, defensible as sources, simply not what the rubric wanted. Section recall moved
   32/42 → 34/42 (unjudged run) and 32/42 → 37/42 (judged run), but that movement is the
   answers themselves, not the additions.

   The premise it was built for did not hold on this pack: of 42 expected sections the agent
   read 38, cited 37 of them unaided, and the four misses were sections it **never read**.
   Those are retrieval misses, and no post-processing can cite a passage that was never
   retrieved. So recovery stays as insurance — it cannot hurt an answer, it costs one
   bounded request — but the honest lever for the metric is search (§2.1), not a second pass
   over the answer. §9 says so too.

### 3.10 When retrieval is weak

"Grounded" is only tested by the case where nothing relevant exists, so that case
is designed rather than left emergent:

- **No result at all.** `search_policy_docs` returns
  `No matching passages found for top_k=… Try a different query, widen it … or
call list_policy_documents` — the agent sees an observation it can act on, not a
  blank. The prompt then requires the one refusal sentence (`REFUSAL_TEXT`:
  _"The ingested documents do not cover this question. Please contact the document
  owner for guidance — for HR policies, that is your HR team."_), which is a
  single constant quoted verbatim by both agent prompts and pinned by
  `tests/test_prompts_invariants.py`, so the text the model produces, the text the
  code returns on a deterministic refusal, and the text a reader sees cannot
  drift apart.
- **Weak but non-empty results.** The model gets those passages and nothing else.
  The prompt forbids filling the gap with plausible policy text, and requires a
  partial answer to _name_ the uncovered part instead of refusing the whole
  question. `min_similarity` exists so the agent can raise its own bar when it
  suspects the question is off-corpus.
- **There is no similarity threshold that auto-refuses.** A hard cut would refuse
  an answer that happens to sit just under it, and "is this covered?" is precisely
  the judgement a model can make with the candidate passages in front of it —
  replacing that judgement with a number is the class of shortcut this codebase
  removes elsewhere (§7.8). Coverage refusal belongs to the component that has
  read the passages; triage never refuses on subject, only on the shape of the
  request (§3.1).
- **A fabricated citation cannot hide.** Deterministic resolution turns it into a
  `citation_warning`, and the reader can click it and be told it does not resolve
  (§3.6).
- **Nothing vetoes after the fact.** No judge, no second model, no auto-refusal —
  the measurement behind that choice is §3.7.

## 4. The contradiction cascade (staged, cost-aware)

Runs in the background after every successful ingestion, for each chunk of the
new document — **against the rest of the corpus, including the same document**
(`SCAN_CROSS_DOC_ONLY=false` is the default; an amendment contradicting its own
policy is a real conflict).

| Stage | Component                                           | Cost                                                                     | What it does                                                                                                                          |
| ----- | --------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | **Candidate selection** (`rag.find_similar_chunks`) | embeddings already stored                                                | every passage whose **stored** vector is at or above `SCAN_MIN_SIMILARITY` (0.65); `SCAN_MAX_CANDIDATES` (200) is a logged cost valve |
| 2     | **Decision model** (`contradictions/gate.py`)       | batched calls to a _separate_ endpoint, `DECISION_BATCH_SIZE` pairs each | scores each pair and **gates** escalation; the only judge                                                                             |
| 3     | **LLM verifier** (`contradictions/verify.py`)       | 1 call per gate-passing pair, `SCAN_WORKERS` in flight                   | strict-JSON verdict: is it really a contradiction?                                                                                    |
| 4     | **Admin flag** (`contradictions/tracker.py`)        | —                                                                        | writes the conflict with both passages, why, score and signals                                                                        |

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

| configuration                | wall      | decision-model requests | verifier calls | flags |
| ---------------------------- | --------- | ----------------------- | -------------- | ----- |
| serial, one request per pair | 216.2s    | 96                      | 16             | 6     |
| batched (6 pairs/request)    | 52.7s     | 20                      | 7              | 6     |
| batched + 4 threads          | **29.8s** | 21                      | 6              | 6     |

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
overlap and explicit supersession are _not_ contradictions.

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
  "citation_warnings": [],
  "faithful": null,
  "answer_shape": "answer",
  "steps": [
    {
      "type": "tool_call",
      "tool": "search_policy_docs",
      "arguments": { "query": "…" },
      "observation": "[chunk 7] Doc: …",
      "schema_repairs": ["converted single quotes…"]
    }
  ],
  "seen_chunk_ids": [7],
  "judge_context_ids": [],
  "scratch_notes": 0,
  "parallel_turns": 0,
  "declined_parts": [],
  "unsupported_claims": [],
  "tool_calls": 1,
  "elapsed_s": 4.2,
  "error": null,
  "intent": "POLICY_DIRECT_LOOKUP",
  "intent_route": "REACT_LOOP",
  "intent_plan": { "question": "…", "queries": ["…"], "entities": [] }
}
```

`faithful` is `null` on this path by design — nothing judges the answer (item 7
above); the field is populated only when the offline audit harness runs, so a
client must not read `null` as "unfaithful". `intent_route` is the triage
decision, which is how an answer with zero tool calls can be legitimate
(`DIRECT_REPLY` for a document listing, `FAST_REFUSE` for a non-document request).

Requests are small on purpose:

```json
{
  "question": "Does the Standard tier cover dental implants?",
  "history": [
    { "role": "user", "content": "…" },
    { "role": "assistant", "content": "…" }
  ]
}
```

`history` is the client's own transcript, used so a follow-up ("and for Premier?")
resolves; it is never treated as evidence. Uploads are `multipart/form-data` with a
single `file` field.

**Why this schema.** `answer` is prose because a person reads it. `citations` is a
list of strings in the same `[doc § section]` form that appears inline, so a client
can display them without re-parsing prose, and `GET /api/citations/resolve` turns
any one of them into the passage behind it — verification does not depend on the
client's own text munging. `steps` is the audit trail (tool, arguments,
observation) that makes "why did it answer that?" answerable afterwards, which is
also what the UI renders live. `seen_chunk_ids` is the set the answer was _allowed_
to use, which is what the offline judge reads instead of trusting the citations.
`intent_route` explains the one result shape that would otherwise look broken (zero
tool calls). `faithful` is `null` rather than `false` because "nobody judged this"
and "judged unfaithful" are different facts, and collapsing them would be a lie in a
convenient direction.

**Failure paths** (the brief's "handle empty queries and unreadable files without
crashing"):

| input                                             | behaviour                                                                                                                                                     |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| empty / whitespace question                       | `400` over HTTP; through the CLI the same input gets "please ask a question about the ingested documents …" instead of an empty answer                        |
| unsupported extension                             | `400 unsupported file type '.docx' (allowed: txt, md, pdf)`                                                                                                   |
| empty file                                        | `400 empty file`; the temporary upload is deleted                                                                                                             |
| file over `MAX_UPLOAD_MB`                         | `413`; the body is streamed in 1 MB windows, so the process cannot be OOM'd before the check                                                                  |
| unreadable PDF (image-only, encrypted, truncated) | `ReaderError` → ingest job `error` **and** document `error`, with the reader's message in `jobs.error`; the API never 500s and the thread never dies silently |
| job killed mid-run (process restart)              | the job is reclaimed as `error` at startup after `STALE_JOB_MINUTES`, so a document cannot stay `ingesting` forever                                           |
| model endpoint down or timing out                 | `503 agent failed: …` on `/api/ask`; an `error` frame on the stream, which the UI renders as a terminal state                                                 |
| unknown document / job / chunk id                 | `404`                                                                                                                                                         |

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

| pack                                | docs | questions | what it exercises                                                                                                                                             |
| ----------------------------------- | ---- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hr_rag_policy_pack_and_benchmark`  | 4    | 20        | multi-hop HR policy: caps that bind, eligibility windows, worked examples, out-of-corpus refusals                                                             |
| `complex_contract_reasoning_pack`   | 4    | 20        | commercial contract / SLA arithmetic: credits, caps, proration, definitions that change a clause's scope                                                      |
| `adversarial_policy_arbitrage_pack` | 4    | 20        | questions whose obvious reading is wrong once a definition elsewhere in the pack is applied, in a fictional jurisdiction — scored 16/20, section recall 76.2% |

Run of record, both sequential, both indexed and answered on the code with the
document brief (`benchmark_20261105_111219`, `…20261105_112329`):

| pack        | correct | + partial | citation recall (doc / doc+section) | mean / p50 / p95      | QA wall                            |
| ----------- | ------- | --------- | ----------------------------------- | --------------------- | ---------------------------------- |
| HR policy   | 19/20   | 20/20     | 28/29 · 27/29                       | 22.1 / 21.8 / 30.5 s  | 441.0 s (67.8 s at `--workers 10`) |
| contract    | 16/20   | 19/20     | 41/42 · 39/42                       | 30.6 / 28.0 / 45.0 s  | 611.9 s                            |
| adversarial | 16/20   | 19/20     | 41/42 · 35/42 (38/42 parent)        | 36.0 / 24.6 / 103.8 s | 170.3 s at `--workers 10`          |

Contract pack rubric (answer /4, reasoning /3, citations /2, uncertainty /1):
**9.40 / 10** = 3.60 · 2.85 · 2.00 · 0.95. The rubric is reported separately from
"correct" on purpose: four answers are right in shape and incomplete, which a binary
verdict hides and a 3.60 shows. The previous run of the same pack had one forced
final (a question that spent 89 s and 20 tool calls); with the brief it has none,
which is a difference in where the agent goes looking, not in the budget.

What these numbers do **not** establish:

- **One run is one sample.** The single miss moves between runs (MH-12, MH-19,
  MH-01 over six repeats of the same code); `eval/repeat_question.py` reports a hit
  rate (`MH-01`: 4/6) because the honest statement is "usually right", not "right".
- **20 questions = 5 points per question.** A 95% and a 90% are one answer apart;
  the packs support "there is no systematic failure", not fine-grained ranking.
- **The judge is a model.** It reads the answer against a reference and a rubric;
  it scores citation _format and resolution_ well and truthfulness less well, so a
  confident, well-cited, wrong answer can still pass.
- **Latency is the endpoint's.** p50 15–28 s is dominated by 4–7 model round trips
  on a hosted endpoint at `reasoning_effort` default — the same contract pack
  measured p50 18.9 s concurrent and 23.4 s sequential — so the wall column
  describes a run, not a service level.
- **Nothing is measured on scale.** Every pack is 4 documents; retrieval
  comparisons on this corpus saturate at k=6 (§2.1), which is stated rather than
  papered over.

## 7. What we rejected, and on what evidence

Two kinds of decision live in this repo and they are kept apart: ones a measurement
settled, and ones argued from structure. A number appears only where a run exists —
"argued" below means there is no run, not that a run agreed.

### 7.1 Rejected because the numbers said so

| rejected                                                   | what the code does instead                                    | the measurement that decided it                                                                                                                                             | where                             |
| ---------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| regex/keyword intent gate                                  | one LLM triage call, route derived from the category          | route accuracy **74.6% → 100%**, missed refusals **15 → 1**, false refusals 0 — for one _cached_ call (4.1s, reasoning off)                                                 | §3.1, `eval/ABLATION.md`          |
| a post-answer faithfulness judge                           | prompt contract + deterministic citation resolution           | **+6s p50, no pass-rate change** on 23 labelled cases, and ~**1 honest answer in 5** flagged at model parity                                                                | §3.7, `dev_docs/DATAFLOW.md` §6   |
| contradiction candidates by `top_k`                        | every passage above a similarity floor                        | the real leave-cap conflict ranks **3rd of 4** there and would rank **41st among 60** similar chunks                                                                        | §4.1                              |
| embedding a chunk _as a query_ to find its neighbours      | stored-vector vs stored-vector                                | the same pair scores **0.601 query-vs-stored** vs **0.678 stored-vs-stored**, so the 0.65 floor excluded the conflict it was calibrated on                                  | §4.1                              |
| the lexical gate as the decider                            | the decision model decides; lexical signals are evidence      | word overlap cannot see "may work from home two days" vs "expected on site three days"; with no endpoint the scan now says INCOMPLETE instead of "clean"                    | §4.2                              |
| one decision-model request per pair                        | batches of 6 + 4 threads, near-threshold pairs re-asked alone | **216.2s → 29.8s**, same 6 flags; batching alone shifted scores ≤0.24 and flipped 2 of 6, which is why the 0.15 re-check band exists                                        | §4.3                              |
| sequential window parsing at default reasoning             | parallel windows, `INGEST_REASONING_EFFORT=off`               | **501.7s → 78.2s** per four-document pack, chunks identical (`sha256`)                                                                                                      | §2, `eval/compare_parse_modes.py` |
| benchmark questions in parallel by default                 | `--workers` is opt-in and stamps a `timing_note`              | 373.9s → **67.8s** wall, but per-question times then include contention — comparable only within the same setting                                                           | §6                                |
| a general "background is part of the question" prompt rule | reverted                                                      | MH-01 **4/6 → 3/6** with the rule, 4/6 with a worked example, whole pack **19/20 → 18/20**; the model learned to _notice_ the condition and still led with the wrong figure | `eval/ABLATION.md` §12            |

The last row is the one we would most want a reviewer to read: a plausible prompt
improvement that was implemented, measured, and thrown away. It stays in the
document because the negative result is the thing that stopped it shipping.

Two of these deserve a paragraph rather than a cell.

**The judge.** A faithfulness judge was built (and calibrated: TPR 10/10, TNR 8/10 on
injected claims) because "verify the answer with a second model" looks obviously
right. Measured at model parity it added ~6s to p50, changed no pass rate, and
flagged about one honest answer in five — the cost was paid on every question and the
benefit was not measurable. It is not deleted: it runs offline as an audit tool
(`eval/calibrate_judge.py`), which is where a finite-precision second opinion is
worth having.

**The gate.** The contradiction scan started with a lexical gate that decided, was
inverted to "heuristic signals handed to a decision model", and then made batched +
threaded. The reversal is the interesting part: a lexical verdict is a language
decision made by word overlap, so the corpus could look clean simply because a
conflict used different words. Words are still useful — as evidence for a model that
reads the passages.

### 7.2 Rejected on structure, with no number behind it

| rejected                                 | why                                                                                                                                                                                                                                                  | what would change our mind                                   |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| LangChain / LlamaIndex                   | hides exactly the failure modes this repo is about (malformed args → repair → feedback → forced answer) behind a framework API                                                                                                                       | multi-agent graphs or tracing we do not have                 |
| Chroma / pgvector / a vector service     | brute-force cosine over a cached matrix is sub-millisecond at 10³–10⁴ chunks; a service adds an install step to a ten-minute README                                                                                                                  | ~10⁵ chunks, or more than one writer process                 |
| `eval()` for `calculate`                 | the expression is model output whose context contains ingested documents; `eval` there reads files and environment variables. The AST whitelist rejects `__import__(...)`, attribute access, `exec` and comprehensions — each attempt is a test case | never                                                        |
| strict schema rejection without repair   | burns a whole turn on trivially repairable JSON                                                                                                                                                                                                      | an endpoint that never emits malformed arguments             |
| a similarity threshold that auto-refuses | it would refuse an answer that sits just under the cut; "is this covered?" is a judgement made with the passages in front of the model                                                                                                               | never — it is the same shortcut class as a regex intent gate |
| async httpx client                       | the agent loop is sequential by construction (next call depends on the last observation); sync keeps it thread-safe beside background ingestion and scanning                                                                                         | token-level streaming inside a turn                          |
| Celery / RQ                              | the `jobs` table already gives resumable progress for a single-node process                                                                                                                                                                          | more than one worker host                                    |
| per-caller `max_tokens` budgets          | billing follows actual output; a small cap truncated the classifier's JSON mid-object at 700 tokens                                                                                                                                                  | an endpoint that bills reserved tokens                       |
| weighted sum of cosine + BM25            | the scales are incomparable and per-query normalisation flips ranking; RRF needs ranks only. Measured: a **tie** at k≥3, sparse wins one exact-token case, hybrid loses one paraphrase case                                                          | a corpus where the k=1 gap matters (§2.1 says how to check)  |
| markdown links or footnotes in answers   | a URL can be fabricated; `[doc § section]` resolves deterministically and is clickable                                                                                                                                                               | a citation syntax that can be validated as cheaply           |
| hosted embeddings                        | policy text stays local, no key, no per-query bill; only the chat call leaves the machine                                                                                                                                                            | a corpus where local embedding is the bottleneck             |
| fixed-size token chunks                  | a citation must name a section, and a fixed cut lands mid-clause; heading boundaries are what make `[doc § 4.2 Dental implants]` possible                                                                                                            | a corpus with no headings at all                             |
| table-aware PDF extraction               | not built: the reader flattens tables, so a wide benefits table survives as word order rather than cells                                                                                                                                             | it is §8.4 — the first thing to build next                   |

### 7.3 Refused on scope, not on merit

Auth/roles, multi-tenant corpora, OCR for scanned PDFs, a version/supersede graph,
per-answer feedback capture, token-level streaming, a cross-encoder reranker and a
CI eval gate are each named in §8 or §9 with what it would take. They are out of
scope by the brief and by cost — not because they are wrong.

## 8. If I had two more weeks

1. **Eval harness in CI** — the sets and runners exist (`eval/run_ablation.py`,
   `measure_retrieval.py`, `run_intent_eval.py`, `calibrate_judge.py`); what is
   missing is a `make eval` gate on a frozen corpus so a retrieval or prompt
   change cannot silently regress. The label-only retrieval check needs no LLM,
   so it can run on every commit for free.
2. **Reranker** — small cross-encoder over the top-20 hits (bge-reranker via
   Ollama); usually the single biggest precision jump once the corpus is big
   enough for the fusion step to matter.
3. **A feedback loop** — the brief's thumbs up/down, stored and later joined to the
   benchmark: right now quality is measured by a label set run offline, so a real
   question that gets a bad answer leaves no trace. The cheapest version is to
   store `{question, answer, citation_warnings, verdict}` from the UI and sample it
   weekly; the interesting part is turning thumbs into _cases_, not into a number.
4. **Table-aware extraction** — row-level digests for tables (PDF tables via
   pdfplumber with bbox detection), so "does tier X cover Y?" hits the exact
   cell.
5. **Auth & multi-tenancy** — real roles (admin/employee), per-company
   corpora, an `ADMIN_TOKEN` guard on the admin routes.
6. **Token-level streaming** — the _steps_ already stream (§3.8) and the UI shows
   elapsed time and silence, so the gap is only the answer text: stream deltas into
   the bubble instead of printing it on `final`. Worth doing for perceived
   latency, not for correctness — and it needs a per-turn SSE fan-in that the
   current one-event-per-step bridge does not have.
7. **Version pinning** — detect "supersedes policy vN" relationships so
   version-vs-version conflicts can be auto-classified as informational.

## 9. Limitations (honest notes)

- `embeddinggemma` has a ~2K-token context; the chunk budget respects it,
  but whole-section embeddings are impossible by design.
- The faithfulness judge is **not on the answer path**. It runs only in the offline
  audit (`eval/calibrate_judge.py`), one LLM call per answered question, and its
  verdicts are cached per (question, answer, contexts, prompt) so repeats are free
  and demos reproducible; a cached verdict is a replay, not a second opinion.
- The judge's precision is finite (measured TPR 10/10, TNR 8/10 on injected
  claims): a flag is a prompt to look, not a conclusion. That finite precision is
  why it is an audit tool and not a gate.
- Citation resolution proves a reference _exists_, not that the passage _supports_
  the sentence. A well-cited wrong claim passes every deterministic check; the
  clickable panel exists so a reader can settle that in one click instead of
  trusting the system's own word for it.
- Benchmark numbers are single samples from a sampling model: the one miss moves
  between runs, so per-question hit rates (`eval/repeat_question.py`) are the honest
  unit, not one run's accuracy (§6).
- Of the brief's stretch ideas, hybrid search is implemented and measured (§2.1) and
  table extraction is not: PDF tables are flattened to text by the reader, so a
  wide benefits table survives as word order rather than cells (§8.4 names the fix).
  Versioning ("this policy supersedes v2") is not implemented either — a superseded
  document stays in the corpus and can produce a contradiction flag instead of a
  replacement (§8.7). Both are stated rather than half-built.
- The extracted `statements`/`key_terms`/`numbers` are still **only** used by
  `get_document_summary`, `run.py list` and the offline judge's inventory: retrieval
  never sees them, and nothing measures how often the agent asks for them. The brief
  that replaced the rolling summary has a measured consumer (the triage layer: intent
  accuracy 98.5% → 100% on 67 labelled cases, `eval/ABLATION.md` §14) but no measured
  effect end to end, so it is a small, defensible cost rather than a proven win.
- The **scratchpad is used zero times** in 200 tool calls on the adversarial pack
  (`eval/ABLATION.md` §16). Five tool schemas ride in every prompt for a facility the
  agent does not reach for; the honest options are an explicit instruction to use it, or
  deleting it. It is kept for now because the instruction experiment has not been run.
- **Citation recovery is insurance, not a fix.** On the pack where it was measured it added
  seven citations and zero of them were sections the reference expects, because the answer
  already cited what it had read and the misses were retrieval misses (§17.3). It also
  cannot verify that a recovered section *supports* a sentence — the decision model scores
  relevance, and its tail is noisy (some irrelevant sections score 0.98), which is why the
  list is capped at three additions. Turning it off is one env key.
- No auth, no multi-tenancy: one process, one corpus, one operator. The admin and
  employee surfaces are the same HTTP API with no role check (§8.5).
- Scanned/image PDFs are rejected with a clear error rather than OCR'd.
- The lexical signals are English-oriented and can _bias_ the decision model's
  score (they no longer decide anything). A pair the heuristics find boring still
  reaches the model, which reads the passages themselves.
- Scan cost grows with the number of pairs above the similarity floor, not with
  the document count; `SCAN_MAX_CANDIDATES` truncates (and says so) rather than
  growing without bound. Incremental "only pairs involving changed chunks"
  scanning is not implemented.
- Batching perturbs decision-model scores by up to ~0.25; the re-check band
  contains that, but a pair whose batched score lands far on the wrong side of
  the threshold is not re-asked.
- `check_citations` matches doc names fuzzily; a deliberately adversarial
  doc name could false-positive a citation match (warnings, not silent
  acceptance, are the failure mode).
