# Policy RAG Agent 📜

A complete, self-contained **RAG-based policy document QA agent** with:

- **ReAct agent + tool_call loop** — 11 tools (`search_policy_docs`,
  `read_policy_chunk`, `list_policy_documents`, `get_document_summary`,
  `list_contradictions`, `calculate`, `scratchpad_write/read/edit/delete/list`),
  batched into one turn when the calls are independent. `search_policy_docs` takes
  **`top_k` OR `min_similarity`, never both** — a count for an ordinary lookup, a
  similarity floor when several documents may touch the question, neither for the
  `RAG_TOP_K` default. The scratchpad is per-question working memory (never
  evidence); `calculate` is an AST-whitelisted arithmetic evaluator, so a derived
  number has a traceable derivation. Both are optional by prompt rule: a
  single-passage question is answered without them.
- **Citation-grounded answering that a reader can verify** — every policy claim
  cites `[doc_name § Section]` / `[doc_name #chunk_id]`. Citations are resolved
  against the store after generation (unresolved ones are warnings, never silent),
  and **the citations are clickable**: the UI asks `GET /api/citations/resolve` and
  renders the passage it points at — or says which sections the document really
  has when it does not resolve.
- **Rolling-window ingestion, now parallel and cheap** — a document is never fed
  whole to the parser LLM: paragraph windows → per-window structure extraction →
  structure-aware chunking. Windows are parsed **concurrently**
  (`INGEST_PARALLEL_WINDOWS`) at `reasoning_effort=off`: **501.7s → 78.2s** for the
  four-document benchmark pack, with byte-identical chunks
  (`eval/compare_parse_modes.py`). Each document also gets one **bounded brief**
  (opening words + every heading → ≤3 sentences) which is what the triage layer and
  the agent read to know what the corpus holds; the brief never enters chunk text.
- **Intelligent chunking** — heading-aware, paragraph/sentence-boundary chunking
  with breadcrumbs (no random fixed-size cuts).
- **Hybrid retrieval** — cosine over stored vectors fused with SQLite FTS5/BM25 by
  Reciprocal Rank Fusion (`RETRIEVAL_MODE=hybrid`).
- **Background contradiction tracker** — similarity floor, not a top-k cut → one
  decision-model judge (batched + threaded) → precision-biased verifier → admin
  flag with both passages and the reason.
- **Anti-hallucination without a second opinion** — an **LLM triage layer** in
  front of the agent (the old regex gate is kept only as an ablation baseline),
  a strict grounding prompt contract, tool-call schema repair with error feedback,
  and deterministic citation resolution. There is **no post-answer judge**: it was
  built, calibrated and measured, then removed because at model parity it cost ~6s
  p50, changed no pass rate and flagged roughly one honest answer in five
  (`DESIGN.md` §3.7).
- **Admin interface** — CLI + web UI to upload and delete `txt` / `md` / `pdf`,
  watch ingestion progress, read the live agent trace over SSE, open citations and
  triage conflicts.
- **Measured, with a harness for every claim** — two scored benchmark packs
  (multi-hop HR policy, commercial contract reasoning) plus an adversarial pack
  (now scored: 16/20)
  awaiting its first run, a rubric judge with a per-dimension breakdown, and a
  labelled ablation set; every number in this README comes from a JSON run committed
  under `eval/results/`.

Models: **Ollama `embeddinggemma`** for embeddings, **any OpenAI-compatible
endpoint** for the LLM (URL + key from `.env`), `MAX_OUTPUT_TOKENS = 65536`. The
numbers in this README were produced with `deepseek-v4.1-flash` over a hosted
OpenAI-compatible endpoint and `embeddinggemma` running locally; nothing in the code
depends on either choice, and a free tier or a laptop model works (slower, and the
agent's tool-calling discipline is what suffers first on a small model).

---

## The contradiction cascade (in execution order)

```
admin uploads doc ──▶ rolling-window ingestion ──▶ chunks + embeddings
                                │
                                ▼  (background)
        ┌──────────────────────────────────────────────────────────────┐
        │ 1. CANDIDATES BY SIMILARITY, NOT BY COUNT                    │
        │    EVERY passage within SCAN_MIN_SIMILARITY (0.65) of a      │
        │    chunk - inside the same document as well as across        │
        │    documents (SCAN_CROSS_DOC_ONLY=false). A top-k cut hides  │
        │    the conflict that ranks 41st among 60 similar passages;   │
        │    SCAN_MAX_CANDIDATES is only a logged cost valve.          │
        ├──────────────────────────────────────────────────────────────┤
        │ 2. DECISION MODEL — the ONLY judge (DECISION_MODEL_*)        │
        │    batched: 6 pairs per request (DECISION_BATCH_SIZE), the   │
        │    passages are the shared prefix; a pair within             │
        │    DECISION_RECHECK_BAND of the threshold is re-asked alone  │
        │    threaded: SCAN_WORKERS requests in flight                 │
        │    ── no endpoint? the scan reports INCOMPLETE, not "clean"  │
        ├──────────────────────────────────────────────────────────────┤
        │ 3. LLM CONTRADICTION VERIFIER — only for gate-passing pairs  │
        ├──────────────────────────────────────────────────────────────┤
        │ 4. ADMIN FLAG — both passages + why they clash + gate score  │
        └──────────────────────────────────────────────────────────────┘
```

Two properties matter more than the stages:

* **The floor decides, not a count.** Candidates are every passage at or above
  `SCAN_MIN_SIMILARITY`, measured chunk-to-chunk on stored vectors (embedding
  the text as a *query* measures lower — 0.601 vs 0.678 on the corpus's own
  leave-cap conflict — and silently excluded it).
* **There is exactly one judge.** The in-code lexical gate still runs, but only
  to hand its signals to the decision model as evidence; it can no longer pass
  or drop a pair, and with no `DECISION_MODEL_*` configured nothing is judged
  (the scan says so instead of reporting zero conflicts).

Batching and threading were measured on the sample corpus: **216s → 30s** for a
full scan with identical flags (`eval/measure_scan.py`). Batching alone shifts
scores by up to ~0.25, which is why near-threshold pairs are re-asked in their
own request. If the verifier cannot run but the gate is emphatic, the pair is
still recorded as **UNVERIFIED** - nothing is silently dropped.

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

## Benchmarks — what it actually scores

Three external packs sit in `Benchmark/`; two are scored below. All are asked
through the same `AdminAPI.ask` the UI calls (no special
path, nothing from the pack is baked into a prompt):

| pack | docs | questions | what it tests |
|---|---|---|---|
| `Benchmark/hr_rag_policy_pack_and_benchmark` | 4 | 20 | multi-hop HR policy: caps, eligibility windows, worked examples, plus out-of-corpus refusals |
| `Benchmark/complex_contract_reasoning_pack` | 4 | 20 | commercial contract / SLA arithmetic: credits, caps, proration, and definitions that change a clause's scope |
| `Benchmark/adversarial_policy_arbitrage_pack` | 4 | 20 | adversarial arbitrage: questions whose obvious reading is the wrong one (a definition elsewhere in the pack flips it), a fictional jurisdiction, and asks whose honest answer is a qualified one. Scored: **16/20**, 19/20 with partials, section recall 76.2% (see ABLATION 15 and `eval/FAILURES.md`) |

```bash
python eval/run_benchmark.py --pack Benchmark/hr_rag_policy_pack_and_benchmark --data-dir data_bench
python eval/run_benchmark.py --pack Benchmark/complex_contract_reasoning_pack --data-dir data_bench_legal
#   --limit N  --reuse (skip re-index)  --index-only  --no-judge  --workers N
```

Run of record, sequential, both packs indexed and answered on the code with the
document brief (2026-11-05): `eval/results/benchmark_20261105_111219` (HR) and
`…20261105_112329` (contract).

| pack | correct | + partial | citation recall (doc / doc+section) | mean / p50 / p95 | QA wall |
|---|---|---|---|---|---|
| HR policy | **19/20 (95%)** | **20/20 (100%)** | 28/29 (97%) · 27/29 (93%) | 22.1 / 21.8 / 30.5 s | 441.0 s; **67.8 s** at `--workers 10` |
| contract | **16/20 (80%)** | 19/20 (95%) | 41/42 (98%) · 39/42 (93%) | 30.6 / 28.0 / 45.0 s | 611.9 s |
| adversarial | **16/20 (80%)** | 19/20 (95%) | 41/42 (98%) · 32/42 (76%) | 41.7 / 27.3 / 113.1 s | 222.1 s at `--workers 10` (834.2 s of agent time) |

Contract pack rubric (answer /4, reasoning /3, citations /2, uncertainty /1):
**9.40 / 10** = 3.60 · 2.85 · 2.00 · 0.95. The 3.60 is the honest number — four
answers are right in shape but incomplete, usually a clause that conditions the cap
the reader would otherwise take whole.

Every answer that was not fully correct is analysed one at a time in
[`eval/FAILURES.md`](eval/FAILURES.md) — nine answers across the three runs, five
mechanisms, with the decisive sentence quoted from each answer and from the reference. It
also records the one case where the benchmark, not the agent, is wrong.

Latency is a property of the run, not of the system: the same contract pack has
measured p50 18.9 s to 28.0 s across runs while the answers barely moved. Treat the
wall column as "what it cost that evening".


Two caveats the harness records itself:

* Concurrency trades per-request latency for wall time. `--workers N` stamps every
  summary with a `timing_note` so contaminated timings are never compared to
  sequential ones.
* These are single runs of a sampling model: the 1/20 miss *moves* between runs
  (MH-12, MH-19, MH-01 over six repeats), which is why `eval/repeat_question.py`
  reports hit rates (`MH-01`: 4/6) instead of one verdict. The same contract pack
  scored 15/20 an hour earlier on the same code, and a prompt experiment scored the
  HR pack 18/20 before it was reverted — `eval/ABLATION.md` §12.

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

`python run.py serve` gives you a three-pane console:

* **Documents** — upload (progress bars that show `window i/n`), delete, rescan.
* **Chat** — grounded answers rendered from the agent's markdown (headings, tables,
  lists, inline code) with citation chips and citation-check warnings. **Citations
  are clickable**: clicking `[doc § section]` calls `GET /api/citations/resolve` and
  shows the passage it points at, rendered from the stored markdown with a `raw`
  toggle for the exact bytes, plus the breadcrumb and chunk id — a reader can check
  the claim instead of trusting it. A citation that does not resolve says why, lists
  the sections the document really has, and marks the chip red.
* **Contradiction tracker** — flags with both quoted passages, why they clash, the
  decision model's signals, and resolve/dismiss buttons.

The ReAct trace is **live**: triage, each turn, each thought, every tool call with
its arguments and a 1.2k-character slice of the observation the agent actually read
appears as it happens over SSE (`POST /api/ask/stream`), so a 20-second answer shows
its work instead of a spinner. Each step is its own collapsible row and the whole
trace is collapsible too; folding it while the agent is still working keeps it
folded, and it folds itself only when the answer arrives.

The waiting state is honest about time: it counts seconds, and because keep-alives
are excluded it measures *output* silence — 15 s of silence is named, 45 s turns the
dot amber, 4 minutes aborts with an explanation. An `error` frame ends the run in
place, a stream that closes without an answer says so, and a dead server reads
"could not reach the server" rather than "Failed to fetch". Nothing is left spinning.

**＋ New conversation** clears the pane and the client-side history the agent is given
(and aborts an answer still streaming, so it cannot land in the new conversation) —
the corpus, the flags and the server state are untouched.

Controls are wired through `data-*` attributes and delegated listeners rather
than inline handlers, so a document named `Bob's "Q1" policy` cannot break the
markup. The agent can read the flag table itself (`list_contradictions`), which
is how "do the policies contradict each other?" gets answered from the audit
trail instead of a fresh guess. Defects found by driving the real page — and how
each was verified — are in `dev_docs/UI_VERIFICATION.md`.

---

## Anti-hallucination measures

1. **Triage before retrieval (one LLM call)** — the raw input is classified
   into one of 8 categories and the route is derived from the category:
   greetings and document listings are answered directly, input that is not a
   question about the ingested documents (world knowledge, code, personal
   advice) and prompt injection are refused, everything about the documents
   goes to the grounded agent. The layer **fails open** and **never refuses a
   question about document content** (it has not read the corpus), so it cannot
   become a coverage outage. The corpus is whatever was uploaded - HR policies
   or contracts; the classifier refuses by *shape of request*, never by topic
   (see `eval/ABLATION.md` §11, where an HR-flavoured triage refused three
   commercial questions before retrieval). It also de-noises the question and proposes the search queries.
2. **Grounding prompt** — answer only from tool output; one canonical refusal
   sentence ("the ingested documents do not cover this question" — and who to ask:
   the document owner, HR for HR policies) quoted verbatim by both agent prompts and
   pinned by a test, so the model's refusal and the code's refusal are the same
   string; conflicts must be surfaced, never silently resolved.
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
7. **Grounding without a second model** — the answer path has **no post-answer
   verifier**. At model parity (the judge was the same `deepseek-v4.1-flash` as
   the agent) it cost +6s p50, changed no pass rate on 23 labelled cases, and
   flagged roughly one honest answer in five. Grounding is carried by the prompt
   contract (grounding, search-first, citation format, refusal, weak evidence,
   injection, and "show your arithmetic for any derived number") plus the
   deterministic checks below. The measured judge is kept as an offline audit
   tool (`eval/calibrate_judge.py`), not as a gate on answers. Numbers and the
   reasoning: `dev_docs/DATAFLOW.md` §6.
8. **Citation resolution is deterministic and refuses to guess** — every
   `[doc § section]` / `[doc #chunk_id]` in the answer is resolved against the
   store: normalised exact match, then a *unique* token subset, then fuzzy with a
   margin. An ambiguous reference ("Remote Work Policy" when v1 and v2 exist) is
   reported as a warning instead of being attributed to the first match — the
   earlier first-match loop resolved "Remote Work Policy v2" to **v1**, i.e. a
   citation of the amended policy recorded against the superseded one.
9. **A citation is a thing the reader can open** (`app/citations.py`,
   `GET /api/citations/resolve`) — the same resolver backs the UI, so verification
   is not a claim the system makes about itself: clicking `[doc § section]` in an
   answer shows the passage it points at, rendered from the stored markdown (with a
   `raw` toggle for the exact bytes), the breadcrumb and the chunk id. When it does
   not resolve the panel says which of the three things went wrong and lists the
   sections the document really has — it never substitutes a nearby passage. A
   section split across chunks returns **all** its parts, so a continuing table
   does not look truncated (`tests/test_citation_resolve.py`).
10. **Precision-biased verifier** — different scope / complementary rules /
   topical similarity only are explicitly *not* contradictions; uncertain →
   false. Unverifiable-but-strong gate hits are marked UNVERIFIED, not
   dropped and not silently confirmed.
11. **Explainable gate** — the decision model's verdict is fully auditable:
   `decided_by` records which judge made the call (`heuristic` / `remote` /
   `hybrid(remote)` / `remote->heuristic`) and its signals are stored with
   every flag.

## HTTP API

`POST /api/ask` returns the whole structured run (this is what the evals read):

```json
{ "answer": "… [03_WORKPLACE_EXPENSE… § 3.1 Eligibility and amount]",
  "citations": ["03_WORKPLACE_EXPENSE… § 3.1 Eligibility and amount"],
  "citation_warnings": [], "faithful": null, "answer_shape": "answer",
  "steps": [{"type": "tool_call", "tool": "search_policy_docs",
             "arguments": {"query": "…"}, "observation": "[chunk 53] Doc: …"}],
  "seen_chunk_ids": [53], "scratch_notes": 3, "parallel_turns": 2,
  "declined_parts": [], "unsupported_claims": [],
  "tool_calls": 6, "elapsed_s": 23.7, "error": null,
  "intent": "MULTI_HOP_COMPARISON", "intent_route": "REACT_LOOP",
  "intent_plan": {"queries": ["…"], "entities": []} }
```

`faithful` is `null` because nothing judges the answer on this path (see §7 above);
it is populated only by the offline audit harness, so a client must not read a
`null` as "unfaithful".

| method | path | what it does |
|---|---|---|
| `GET` | `/` | the single-page UI |
| `GET` | `/api/health` | liveness |
| `POST` | `/api/documents` | upload `txt`/`md`/`pdf`; returns `doc_id` and starts background ingestion |
| `GET` | `/api/documents` | documents + status + live job progress + open-flag counts |
| `DELETE` | `/api/documents/{doc_id}` | purge a document with its chunks, flags and stored upload |
| `GET` | `/api/jobs/{job_id}` | ingestion / scan progress (`stage`, `window i/n`) |
| `POST` | `/api/ask` | grounded answer + trace (blocking) |
| `POST` | `/api/ask/stream` | the same answer as Server-Sent Events: `stage` / `turn` / `step` / `final` / `error`, 15 s keep-alives |
| `GET` | `/api/citations/resolve?cite=…` | resolve one citation to its passage (`parts[]`, breadcrumb, `how`, or `reason` + `candidates`) |
| `GET` | `/api/chunks/{chunk_id}` | one stored chunk |
| `GET` | `/api/contradictions` | flags with both passages, the reason and the gate's signals |
| `POST` | `/api/contradictions/{cid}/status` | `OPEN` / `resolved` / `dismissed` |
| `POST` | `/api/scan/{doc_id}` | re-run the contradiction cascade for one document |

## Configuration (`.env`)

| variable | default | meaning |
|---|---|---|
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | ollama/v1 | any OpenAI-compatible endpoint |
| `MAX_OUTPUT_TOKENS` | `65536` | output ceiling for every LLM call (agent, triage, parser, offline judge); a ceiling only - it never lengthens a reply, it prevents truncation. Auto-halved if an endpoint caps it |
| `TEMPERATURE` | `0.1` | low temperature for grounded answers |
| `LLM_REASONING_EFFORT` / `INTENT_REASONING_EFFORT` | *(default)* / `off` | the real cost lever: the classifier runs with reasoning off (4.1s vs 5.2s per call, same decisions) |
| `AGENT_MAX_STEPS` / `AGENT_MAX_OBS_CHARS` | `12` / `4000` | loop turns per question (one model call each - a turn may carry several parallel tool calls), and the observation truncation point |
| `EMBED_BATCH_SIZE` | `32` | passages embedded per Ollama batch |
| `OLLAMA_BASE_URL` / `EMBED_MODEL` | `…:11434` / `embeddinggemma` | embeddings backend |
| `EMBED_QUERY_PREFIX` / `EMBED_DOC_PREFIX` | EmbeddingGemma task prefixes | keep quoted! Ollama does not add them itself |
| `INGEST_WINDOW_CHARS` | `6000` | rolling-window size for read+parse |
| `INGEST_LLM_ENABLED` | `true` | set `false` for heuristic-only ingestion |
| `INGEST_HEURISTIC_MAX_CHARS` | `0` | documents at or below this size skip the LLM parse anyway (`0` = never); metadata only |
| `INGEST_BRIEF_WORDS` | `1000` | opening words the per-document brief reads (the brief is the triage layer's and the agent's view of what a document is about; at most 3 sentences out) |
| `INGEST_REASONING_EFFORT` | `off` | effort for the window parser — structured extraction, measured 4.2× faster at `off` with the same metadata shape |
| `INGEST_MAX_TOKENS` | `3000` | output cap for one window-parse call |
| `INGEST_PARALLEL_WINDOWS` | `4` | windows parsed at once; `1` restores the sequential chained parse. Stored chunks are identical either way |
| `CHUNK_TARGET_CHARS` / `CHUNK_MAX_CHARS` | `1400` / `2200` | soft chunk bounds |
| `JEV_GATE_THRESHOLD` | `0.5` | decision-model escalation threshold |
| `DECISION_MODEL_NAME` | *(empty)* | stage-2 judge of the cascade. There is one judge: the decision model, or none (a scan then reports itself INCOMPLETE rather than "no contradictions") |
| `DECISION_MODEL_BASE_URL` | *(empty)* | OpenAI-compatible base URL of the decision model (e.g. `http://localhost:11434/v1`, or the hosted `typesafe/jev`) |
| `DECISION_MODEL_API_KEY` | *(empty)* | API key for the decision-model endpoint (e.g. `ollama`) |
| `DECISION_TEMPERATURE` | `0.0` | keep the judge deterministic |
| `SCAN_MIN_SIMILARITY` | `0.65` | candidate pairs: every passage at or above this cosine floor (not a top-k cut — a conflict can rank 41st among 60 similar chunks) |
| `DECISION_BATCH_SIZE` | `6` | pairs judged per decision-model request (the endpoint takes many questions over one shared state; 1 disables batching) |
| `DECISION_RECHECK_BAND` | `0.15` | a pair this close to the threshold is re-asked in its own request, because batching perturbs scores by up to ~0.25 |
| `SCAN_WORKERS` | `4` | concurrent decision-model / verifier requests during a scan |
| `SCAN_MAX_CANDIDATES` | `200` | safety cap on pairs examined per document (logged when it truncates) |
| `SCAN_CROSS_DOC_ONLY` | `false` | `false` scans inside a document as well as across documents |
| `INTENT_ENABLED` | `true` | LLM triage in front of the agent (one extra cheap call) |
| `INTENT_MODEL` | *(empty)* | model for the classifier; empty = `LLM_MODEL` |
| `INTENT_MAX_TOKENS` | `65536` | ceiling for the classifier's JSON reply (a ceiling, not a budget) |
| `INTENT_REASONING_EFFORT` | `off` | reasoning budget for the classifier; JSON extraction needs none |
| `INTENT_CACHE_PATH` | *(empty)* | decision cache; empty = `DATA_DIR/intent_cache.json` |
| `JUDGE_CACHE` | `true` | replay a verdict for an identical (question, answer, contexts, prompt) — verdicts vary run to run even at temperature 0 |
| `PROMPT_INTENT` / `PROMPT_AGENT` / `PROMPT_JUDGE` | `v1` / `v1` / `v1` | prompt variants; v1 and v2 tie end-to-end (see `eval/ABLATION.md`) |
| `RAG_TOP_K` | `6` | passages per `search_policy_docs` call |
| `RETRIEVAL_MODE` | `hybrid` | `vector` / `keyword` / `hybrid` (RRF fusion of cosine + FTS5/BM25) |
| `RRF_K` | `60` | RRF damping constant (higher = flatter rank weights) |
| `HYBRID_VECTOR_WEIGHT` / `HYBRID_KEYWORD_WEIGHT` | `1.0` / `1.0` | per-retriever weight inside the fusion |
| `MAX_UPLOAD_MB` | `20` | upload ceiling; larger files get HTTP 413 |
| `STALE_JOB_MINUTES` | `30` | a job untouched this long is marked `error` at startup (killed process), so its document cannot stay `ingesting` |
| `DATA_DIR` | `./data` | SQLite DB + uploads |
| `APP_HOST` / `APP_PORT` | `127.0.0.1` / `8000` | `run.py serve` bind address |
| `LOG_LEVEL` | `INFO` | python logging level |

## Project layout

```
policy-rag-agent/
├── run.py                      # CLI entry (upload/list/ask/chat/scan/serve/selftest)
├── DESIGN.md                   # architecture & design decisions (read this!)
├── .env.example                # every knob, documented
├── requirements.txt            # runtime dependencies
├── requirements-dev.txt        # + pytest / httpx / reportlab
├── pytest.ini
├── app/
│   ├── main.py                 # ASGI entry (uvicorn app.main:app)
│   ├── config.py               # .env -> typed config (one field per knob)
│   ├── llm.py                  # OpenAI-compatible client, retries, feature downgrade
│   ├── embeddings.py           # Ollama embeddinggemma client (task prefixes)
│   ├── db.py                   # SQLite: documents, chunks(+vectors), FTS5, flags, jobs
│   ├── schema_repair.py        # regex JSON repair + validation + feedback
│   ├── prompts.py              # every prompt (agent, window parser, triage, judge)
│   ├── readers.py              # txt / md / pdf readers
│   ├── chunker.py              # heading-aware chunking + paragraph windows
│   ├── ingest.py               # window parse (parallel pool / chained), metadata, chunks
│   ├── rag.py                  # dense + sparse retrieval, RRF fusion, threshold search
│   ├── naming.py               # document-name resolution (unique match or nothing)
│   ├── citations.py            # parse_citation + CitationResolver (for warnings AND the UI)
│   ├── admin.py                # facade: triage -> agent -> citation check (no judge)
│   ├── cli.py                  # command line interface
│   ├── agent/
│   │   ├── react.py            # ReAct loop, on_event trace, citation checking
│   │   ├── tools.py            # 11 tool schemas + implementations
│   │   ├── calc.py             # arithmetic over an AST whitelist (no eval)
│   │   ├── scratchpad.py       # per-question working memory (bounded, in memory)
│   │   ├── intent.py           # LLM triage layer (disk cache, fail-open)
│   │   ├── intent_legacy.py    # the old regex gate - ablation baseline only
│   │   ├── answer_shape.py     # answer|partial|refusal|chitchat via the decision model
│   │   └── verify.py           # evidence check + faithfulness judge (offline audit)
│   ├── contradictions/
│   │   ├── gate.py             # decision model (batched, threaded); lexical signals only
│   │   ├── verify.py           # precision-biased contradiction verifier
│   │   └── tracker.py          # candidates -> gate -> verify -> admin flag
│   ├── api/routes.py           # FastAPI routes (factory: create_app(api=...))
│   └── web/index.html          # single-file UI: docs, chat, live SSE trace, citation panel
├── tests/                      # 292 offline tests: pytest OR python run.py selftest
│   ├── fakes.py                # FakeLLM / StubLLM / FakeEmbedder
│   ├── conftest.py             # hermetic env bootstrap
│   └── fixtures/sample_policy.pdf
├── eval/
│   ├── run_benchmark.py        # any pack: index, ask, judge, score, JSON + JSONL
│   ├── repeat_question.py      # hit rate of one question over N repeats (--judge)
│   ├── run_ablation.py / report_ablation.py      # prompt variants A-D on labelled cases
│   ├── run_intent_eval.py      # triage ablation (legacy regex vs v1 vs v2)
│   ├── measure_retrieval.py / measure_scan.py    # label-only, no LLM calls
│   ├── compare_parse_modes.py  # sequential vs parallel ingest -> identical chunks?
│   ├── scratch_usage.py       # does the agent actually use its working memory?
│   ├── profile_ingest.py       # per-call timing of the window parser
│   ├── calibrate_judge.py      # offline judge TPR/TNR on injected claims
│   ├── verify_api.py           # 42 end-to-end API checks (needs a live server)
│   ├── verify_renderer.py      # chat renderer contract, incl. XSS inputs
│   ├── verify_upload_cleanup.py# uploads disappear with their document
│   ├── watch_stream.py         # time every SSE frame of a live answer
│   ├── trace_steps.py          # dump one question's full tool trace
│   ├── scoring.py              # rubric judging + citation recall
│   ├── check_docs.py           # README/DESIGN vs code (env keys, tools, routes, scripts)
│   ├── results/                # committed benchmark runs (JSON + JSONL)
│   ├── ABLATION.md             # every measurement, incl. the ones that failed
│   └── FAILURES.md             # every answer that was not correct, and why
├── Benchmark/                  # external question packs (docs + questions + rubric)
│   ├── hr_rag_policy_pack_and_benchmark/
│   ├── complex_contract_reasoning_pack/
│   └── adversarial_policy_arbitrage_pack/
├── dev_docs/                   # DATAFLOW, HEURISTIC_AUDIT, OVERNIGHT_LOG, CODE_REVIEW,
│                               #   UI_VERIFICATION (defects found by driving the page)
├── scripts/
│   ├── demo.py                 # fact / table / off-policy / refusal, as JSON
│   └── make_pdf_fixture.py     # regenerates tests/fixtures/sample_policy.pdf
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

# benchmarks (need a live LLM; each indexes its pack into its own DATA_DIR)
python eval/run_benchmark.py --pack Benchmark/hr_rag_policy_pack_and_benchmark
python eval/run_benchmark.py --pack Benchmark/complex_contract_reasoning_pack --data-dir data_bench_legal
python eval/repeat_question.py MH-01 6 --judge        # hit rate, not one verdict
python eval/compare_parse_modes.py --doc <path.md>    # sequential vs parallel ingest

# ablations
python eval/run_eval.py             # 18-case eval against a live stack (legacy set)
python eval/run_intent_eval.py --variant legacy,v1,v2 --hard   # triage ablation
python eval/run_intent_eval.py --variant v1 --hard --briefs-from data_bench  # + CORPUS block
python eval/run_ablation.py --variant A,B,C,D                  # end-to-end prompts
python eval/report_ablation.py      # merged comparison table

# label-only / offline measurements (no LLM calls)
python eval/measure_retrieval.py    # vector vs keyword vs hybrid
python eval/measure_scan.py         # serial vs batched vs threaded scan
python eval/profile_ingest.py       # per-call cost of the window parser

# verification harnesses
python eval/verify_api.py           # 42 end-to-end API checks (needs a live server)
python eval/verify_renderer.py      # chat markdown renderer, incl. XSS inputs
python eval/verify_upload_cleanup.py# an upload disappears with its document
python eval/check_docs.py           # README/DESIGN vs code (env keys, tools, routes, scripts)
python eval/watch_stream.py "question"  # time every SSE frame of a live answer
python eval/trace_steps.py "question"   # one question's full tool trace
python scripts/make_pdf_fixture.py  # regenerate tests/fixtures/sample_policy.pdf
```

The suite (292 tests in 28 files, offline, no keys) covers: schema repair + feedback,
the ReAct loop (native and text-based tool calling, citation verification, the
calculator and scratchpad tools, parallel tool calls), chunking/windows, the
ingestion pipeline (including the parallel-window pool and its merge call), the
store, the decision-model gate (batching, the re-threshold re-check, "unavailable"
semantics), the full cascade (flag / gate-blocks-LLM / unverified fallback),
document-name resolution (uniqueness required: v1 vs v2), citation parsing and
resolution (multi-chunk sections, unknown section/document), the triage layer (cache
keyed on the prompt fingerprint, fail-open, corrective retry, plan injection),
retrieval (FTS5 keyword index, RRF fusion, threshold search, chunk-to-chunk
similarity), the offline faithfulness judge (shape classification, judge context
selection, partial answers), the SSE stream (frame order, a listener that raises, the
non-streaming fallback) and the HTTP API end-to-end.

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
| contradiction candidates chosen by `top_k`/per-chunk cap | the real leave-cap conflict ranks 3rd of 4 (0.678) - inside the cut by luck, and invisible among 60 similar chunks. Now: every passage above the similarity floor, intra-document included |
| the scan compared a *query* embed of a chunk to stored vectors | it measures lower than chunk-to-chunk (0.601 vs 0.678), so the 0.65 floor excluded the conflict it was calibrated on. `db.similar_to_chunk()` compares stored vectors |
| a full scan took 216s (96 model requests) | batched (6 pairs/request) + 4 threads: **29.8s**, 21 requests, **same six flags**; 0.15 re-check band keeps verdicts identical to per-pair scoring |
| UI: delete/flag buttons dead for names with a quote; answers shown as raw markdown | `data-*` + delegated listeners; escape-first markdown renderer. Found by driving the real page (`dev_docs/UI_VERIFICATION.md`) |
| indexing 4 documents took 501.7s, almost all of it the window parser | `reasoning_effort=off` (40.5s → 9.6s per call, 4.2×) plus `INGEST_PARALLEL_WINDOWS=4`: **78.2s** per pack, chunks byte-identical (`sha256 f0d72b17…`) |
| 20 benchmark questions took 373.9s of wall clock | `--workers 10`: **67.8s**, same 19/20; the summary stamps `workers` + `timing_note`, because those timings include contention |
| the triage prompt called the corpus "HR policy", so the contract pack got canned HR refusals in ~8s with **zero** tool calls (13/20) | refuse by *shape of request*, never by subject: **15/20**, false refusals 0, and the three refused contract questions now compute (LC-01 → ₹28,000) |
| one benchmark miss, chased over six runs | the miss **moves** (MH-12, MH-19, MH-01) — it is sampling, not a broken router — so `eval/repeat_question.py` reports hit rates (`MH-01`: 4/6), never one verdict |
| the triage layer had to classify without knowing what the corpus holds | it is shown a bounded `CORPUS` block (one 300-char brief per ready document): intent accuracy **98.5% → 100%**, route 100%, false refusals still 0 — the one remaining misclassification was a question whose vocabulary only the document brief supplies. The brief never enters chunk text (`tests/test_corpus_brief.py`) |
| a general "background is part of the question" prompt rule | measured and **reverted**: MH-01 4/6 → 3/6 with the rule, 4/6 with a worked example, pack 19/20 → 18/20. The model learned to *notice* the condition and still led with the wrong figure (`eval/ABLATION.md` §12) |
| citation chips looked clickable and did nothing; the panel then dumped raw markdown (`##` headings, rows of pipe characters) | chips resolve against the store (`GET /api/citations/resolve`) and the passage is rendered with a `raw` toggle; a miss names the sections the document really has, and a section split across chunks returns all its parts |
| "working — triage…" could spin forever: a stalled call is a 10-minute read timeout and a dead stream showed nothing | the waiting state counts *output* silence (15s names it, 45s turns amber, 4 min aborts), and error / closed stream / dead server each end the bubble in a terminal state |
| a row rendered as the literal text `\n\n` in the citation panel | the escapes had been double-escaped through a heredoc; the panel now takes its text from the resolver, and `eval/verify_renderer.py` checks the renderer instead of the source |

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
