# Dataflow — end to end, for review

Everything below is a real call path in this tree. Line numbers are given so a
reviewer can jump to the code instead of trusting prose; when they drift, the
function names are what to search for.

Two flows matter: **ingest** (documents in) and **ask** (question in, answer out).
They meet only in SQLite.

---

## 0. The pieces

```
app/
  config.py        Config: every knob, read once from env/.env
  db.py            SQLite: documents, chunks, chunks_fts, contradictions, jobs, meta
  readers.py       md / txt / pdf -> plain text
  chunker.py       text -> sections -> chunks (structural: heading shapes + sizes)
  ingest.py        the pipeline: parse -> chunk -> embed -> store (+ job progress)
  embeddings.py    OllamaEmbedder (embeddinggemma) / sentence-transformers fallback
  rag.py           RAGEngine.search: dense cosine + FTS5/BM25, fused with RRF
  llm.py           OpenAICompatClient: chat, chat_json, tool-call downgrade, 4xx retry
  prompts.py       every prompt string (intent, agent, judge, contradiction)
  naming.py        document-name resolution (unique candidate or ambiguous)
  admin.py         AdminAPI: the façade every entry point talks to
  api/routes.py    FastAPI: /api/ask, /api/documents, /api/jobs, /api/contradictions
  cli.py           same façade, terminal
  agent/
    intent.py        triage: one LLM call -> IntentPlan (category, route, queries)
    intent_legacy.py regex triage, ablation baseline only — never wired in
    tools.py         the four retrieval tools + observation formatting
    react.py         the ReAct loop, answer extraction, citation resolution
    verify.py        faithfulness judge — OFFLINE AUDIT ONLY (see §6)
    answer_shape.py  answer-shape classifier — OFFLINE AUDIT ONLY
  contradictions/
    gate.py          cascade gate; JEVStyleGate is the lexical prefilter
    verify.py        stage-3 contradiction verifier (LLM, per candidate pair)
    tracker.py       scan orchestration, duplicate suppression, dismissal state
```

---

## 1. INGEST — a document becomes searchable

Entry points: `AdminAPI.start_upload` / `POST /api/documents`.

```
upload bytes
  │
  ├─ size check            cfg.max_upload_mb            (admin.py)
  ├─ save to disk          DATA_DIR/uploads/<stamp>_<name>
  ├─ db.add_document(name, filename, fmt, status="pending")
  ├─ thread start          AdminAPI._run_ingest  -> IngestPipeline.ingest (ingest.py:63)
  │    job row in `jobs` mirrors progress: done/total/stage
  │
  └─ IngestPipeline.ingest(path, name, doc_id):
       1. readers.extract(path)                  -> text
       2. windows = split by cfg.ingest_window_chars (6000)
          for each window: WINDOW_PARSER_SYSTEM -> {window_title, statements,
                            key_terms, numbers, rolling_summary}
          (structure hints only; the text itself is never rewritten)
       3. chunker.split(text)                    -> [{section, breadcrumb, text}]
          heading detection = md `#`, numbered `3.2 Title`, ALL CAPS, labelled
          lines, underlined — structural shapes, no semantic guessing
       4. embeddings.embed_documents(texts)       -> vectors (768-d embeddinggemma)
          stored in `meta`: embed model + fingerprint + dimension
          a dimension change later raises a clear RuntimeError (db.py guard)
       5. db.add_chunk(...) per chunk            -> chunks table (embedding BLOB)
       6. db.update_job(stage="embedding N chunks" -> "done")
       7. document status: pending -> ingesting -> ready
```

FTS5: `chunks_fts` is kept in sync as a contentless index. If it is ever behind
(`chunks_fts_docsize` ≠ chunk count) it is rebuilt by
`INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')` — this was silently broken
once because the check counted the *content* table instead of the index.

Crash behaviour: a killed process leaves the job `running` and the document
`ingesting`; `AdminAPI.__init__` calls `db.reconcile_stale_jobs(cfg.stale_job_minutes)`
which marks both as error and logs that the document must be re-uploaded. It does
not auto-resume: re-ingesting costs embedding calls and a half-written document is
not obviously safe.

---

## 2. ASK — a question becomes an answer

Entry points: `AdminAPI.ask` / `POST /api/ask` / `cli ask`. **This is the path to
review: there is no post-answer verification in it.**

```
question
  │
  ├─ 1. TRIAGE  (agent/intent.py, one LLM call, reasoning_effort=off)
  │      input : the raw question only (no corpus access, no tools)
  │      output: IntentPlan {intent, route, de_noised_question,
  │                           search_queries[], entities[], requires_table_check,
  │                           refusal_reason}
  │      cache  : DATA_DIR/intent_cache.json keyed on
  │               question + prompt version + model + reasoning effort
  │      fail   : classifier error / unparseable JSON -> PLAN_FALLBACK, route
  │               REACT_LOOP (fail open: never lose coverage to a triage outage)
  │
  ├─ 2. ROUTE  (admin.py, deterministic dispatch on the plan)
  │      FAST_REFUSE            -> REFUSAL_TEXT (or the empty-question text)
  │      GREETING               -> GREETING_REPLY
  │      META_DOCUMENTS         -> _document_listing(): one line per stored doc
  │      everything else        -> REACT_LOOP
  │      (SENSITIVE_HR_ESCALATION goes to the agent, then gets ESCALATION_NOTE
  │       prepended to the answer - it is never refused before retrieval)
  │
  ├─ 3. REACT LOOP  (agent/react.py:186)
  │      system prompt: AGENT_SYSTEM or AGENT_SYSTEM_V2 (cfg.prompt_agent)
  │      user message : de_noised_question + the plan block
  │      per turn, up to cfg.agent_max_steps (12):
  │        a. llm.chat(messages, tools=TOOL_SPECS)
  │        b. native tool_calls? use them. else extract <tool_call>{...}</tool_call>
  │           from the text; else the turn is the final answer
  │        c. schema_repair: validate arguments; on failure send a
  │           TOOL_CALL_ERROR observation and let the model correct it -
  │           never a fabricated tool result
  │        d. tools.execute(name, args) -> observation string (capped)
  │           search_policy_docs | read_policy_chunk |
  │           list_policy_documents | get_document_summary
  │        e. append observation, record seen_chunk_ids (provenance of what the
  │           agent actually read)
  │
  ├─ 4. ANSWER + CITATIONS  (react.py:251)
  │      answer text comes from the model, citations are inline in that text.
  │      check_citations(answer, doc_names, chunk_exists):
  │        _CITE_RE finds every [ ... ]
  │        _looks_like_citation keeps only `doc § section` / `doc #id` shapes
  │        naming.resolve picks the document: normalised exact -> unique token
  │        subset -> fuzzy >= 0.90 with a margin. AMBIGUOUS -> None + warning
  │        `doc #id` must be a chunk that exists
  │      result.citations  = resolved references (deduped, first-seen order)
  │      result.citation_warnings = unresolvable / ambiguous references
  │
  └─ 5. RESPONSE
         AgentResult {answer, steps, citations, citation_warnings,
                      seen_chunk_ids, tool_calls, elapsed_s, intent, intent_plan}
         faithful / unsupported_claims stay None / [] - nothing judges them
```

Time budget for one policy question: triage ~4s, agent 2-4 turns ~10-15s.
The citation step is free (no model call).

---

## 3. RETRIEVAL — what a tool call actually does

`ToolRegistry.execute("search_policy_docs", {...})` -> `RAGEngine.search(query, top_k, mode)`
(rag.py:56):

```
query
 ├─ embed_query (Ollama, 768-d)             vector arm
 ├─ FTS5 MATCH (bm25)                        keyword arm    (mode=keyword: only this)
 │
 ├─ pool = max(top_k*4, 20)  over-fetch so fusion has something to rerank
 ├─ RRF: score(d) = Σ_r w_r / (k + rank_r(d)),  k=60, weights per arm
 │   mode=hybrid (default) | vector | keyword
 └─ top_k hits: {chunk_id, doc_id, doc_name, section, text, score}
```

The agent sees formatted hits (`[chunk 33] doc § section …`), which is exactly
what makes `[doc § section]` and `[doc #33]` citations resolvable afterwards.

---

## 4. STORAGE — what the answer is checked against

```
documents(id, name, filename, format, size_bytes, status, created_at, updated_at)
chunks(id, doc_id, chunk_index, section, breadcrumb, text, embedding BLOB, …)
chunks_fts(contentless FTS5 index over chunk text)
contradictions(id, a_chunk, b_chunk, status, decided_by, signals, …)
jobs(id, kind, doc_id, status, stage, done, total, …)
meta(key, value)        -- embed fingerprint, dimensions
```

`citations` are never stored: they are derived from the answer text each time,
which is why a citation can only ever name a document that exists.

---

## 5. CONTRADICTION SCAN — separate pipeline, not part of `ask`

`ContradictionTracker.scan_document` (contradictions/tracker.py) →
`POST /api/scan/{doc_id}`:

```
new chunks
 ├─ candidate pairs: same/similar topic, capped by
 │    cfg.contra_top_k per chunk, cfg.scan_max_pairs_per_chunk, cross-doc only
 ├─ stage 2: CascadeGate (gate.py)
 │    heuristic  -> JEVStyleGate only        (lexical: jaccard + modality regex)
 │    remote     -> decision model scores each pair
 │    hybrid     -> heuristic first, remote only for borderline pairs
 ├─ stage 3: ContradictionVerifier (LLM) for gate-passing pairs
 └─ db.add_contradiction(...)  order-agnostic (min,max) unique index,
                               INSERT OR IGNORE, dismissal survives rescans
```

This is deliberately *not* triggered by `ask`: the question flow never spends
scan money, and the scan never blocks an answer.

---

## 6. Why there is no verifier in the answer path

There used to be one: an LLM judge that read the answer plus the passages and
reported unsupported claims, gated by a keyword refusal list that was later
replaced by a decision-model classifier. It was removed from `ask()` after
measurement:

| measurement | value |
|---|---|
| pass rate with the judge (C) vs without (D) | 100% vs 100% — no difference on 23 labelled cases |
| latency | +6s p50 (18.9s vs 12.3s) |
| false alarms on honest answers | TNR 8/10 → ~1 in 5 answers flagged |
| stability of the flag count | 0 then 3 (variant B), 3 then 1 (variant C), same code |
| real catches | a wrong chunk-id citation (e01), a disputed transition rule (q18) |

At model parity — the judge is the same `deepseek-v4.1-flash` as the agent — the
only thing the judge adds is independence of context, and it was buying that with
a sixth of the latency and a fifth of the honest answers. Grounding is now carried
by the prompt contract (rules 1-8/1-11: grounding, search-first, citations,
refusal, weak evidence, injection, derived numbers) plus the deterministic
citation resolution above.

The measurement apparatus is kept, not deleted:
`app/agent/verify.py`, `app/agent/answer_shape.py`,
`eval/calibrate_judge.py` (TPR/TNR on injected claims),
`eval/run_shape_eval.py`. Nothing in `ask()` imports them.

---

## 7. Failure behaviour, point by point

| failure | what happens |
|---|---|
| triage LLM down / bad JSON | plan falls back, question still reaches the agent (fail open) |
| agent LLM error mid-loop | `result.error` set, `answer` empty, API returns 502-shaped error |
| endpoint rejects `tools` | `llm.py` drops the field and retries; agent uses `<tool_call>` text |
| endpoint rejects `response_format` / `reasoning_effort` | same: field dropped, retried |
| 429 / 5xx | backoff retries, then a clear error |
| chunk id cited that does not exist | citation warning, citation excluded |
| document name ambiguous ("Remote Work Policy") | warning, citation excluded — never guessed |
| embedding model changed | `RuntimeError` naming the model and dimension |
| process killed mid-ingest | next start reconciles the job to error, document flagged |
| upload too large | rejected before anything is stored |

---

## 8. Where a reviewer should look first

1. `app/admin.py:ask` — the only place routing decisions are made.
2. `app/agent/react.py:186-260` — the loop, the answer, the citations.
3. `app/naming.py` — document resolution (the v1/v2 bug lived here).
4. `app/prompts.py` — the entire grounding contract is prompt text now; read
   AGENT_SYSTEM rule by rule, that is what replaced the verifier.
5. `dev_docs/HEURISTIC_AUDIT.md` — every remaining place where something
   semantic is decided without a model.
