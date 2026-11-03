# Code review — defects, smells, and the decisions behind them

Written after the intent/prompt/retrieval work; every item below comes from
reading the code and, where possible, from a measurement recorded in
`eval/ABLATION.md` or `dev_docs/OVERNIGHT_LOG.md`.

Legend: **FIXED** = changed in this work · **OPEN** = real, not fixed (why
below) · **BY DESIGN** = looks like a defect, is not.

---

## 1. Fixed in this work

| area | defect | why it mattered |
|---|---|---|
| intent front door | regex keyword gate | answered "What policies apply to parental leave?" with a document *listing*; overfits phrasings by construction. Replaced with an LLM triage layer; regex kept only as the ablation baseline. **FIXED** |
| intent JSON | 700-token cap truncated the classifier mid-object and the repair layer accepted the fragment | silently produced a wrong category on a grievance case. Cap raised, corrective retry added, test added. **FIXED** |
| judge gating | judge skipped any answer containing a refusal phrase | a real answer that also said "the policy does not cover X" bypassed verification entirely. **FIXED** |
| judge evidence | judge received only the first 8 seen chunks (9k chars), in arrival order | correct conflict answers cite late passages, so the judge accused them of inventing numbers that were in the corpus. Now the cited chunks are resolved first and the budget is 24 chunks / 24k chars; truncation adds an explicit coverage note. **FIXED** |
| judge verdicts | "faithful": false with no named claim was accepted as a failure | unexplainable warning in the UI, indistinguishable from a real hallucination. One corrective retry, then degrade to unknown. **FIXED** |
| retrieval | `RAG_TOP_K` existed but the tool hardcoded 6 | config was a lie. **FIXED** |
| retrieval | keyword search did not exist; embeddings only | policy numbers / tier amounts are exactly where dense retrieval is weakest. Added FTS5 + BM25 and RRF fusion (`RETRIEVAL_MODE`), with a comparison harness. **FIXED** |
| store | embedding-model swap produced a raw numpy matmul error on query | now a stored model/dimension fingerprint and an actionable message. **FIXED** |
| store | FTS backfill counted the content table, so an empty index looked complete | caught while measuring: keyword search returned 0/17. Fixed to check the index's own docsize table. **FIXED** |
| citations | markdown links and `[1]` were verified as citations | spurious warnings on ordinary prose; duplicates repeated. **FIXED** |
| upload | no size limit; the whole file was read into memory | a large PDF could take the process down before validation. Now streamed with a `MAX_UPLOAD_MB` cap and 413. **FIXED** |
| contradictions | order-sensitive pair dedupe (see log §2) | duplicate flags, dismissed flags resurrected. **FIXED** |
| client | no `reasoning_effort` support → no way to cut classifier latency | measured ~25% p50 reduction at "off" with identical accuracy. **FIXED** |
| config | `DECISION_MAX_OUTPUT_TOKENS` | a per-caller token cap is not a budget; a small cap truncates reasoning-classifier JSON. Removed. **FIXED** |
| embeddings | `OLLAMA_BASE_URL=localhost` on Windows | ~2s per embedding call of IPv6-fallback latency. Default is `127.0.0.1`. **FIXED** |

---

## 2. Open, ranked

### O1 — FIXED — an interrupted ingestion left a document stuck forever
`AdminAPI.upload_document_async` ingests on a daemon thread. If the process
dies, the `documents` row stays `ingesting` and the `jobs` row stays `running`;
nothing reconciles them on the next start, and the UI polls a job that will never
finish. Fix: at startup, mark `running` jobs older than N minutes as `error` with
reason "interrupted", and offer a re-ingest command. Left open because it needs a
startup hook plus a policy decision about auto-resume (re-ingesting costs
embedding calls).

### O2 (medium) — the API leaks exception text
`app/api/routes.py::ask` wraps every failure in
`HTTPException(503, f"agent failed: {e}")`. That is how an internal endpoint URL
or a provider error body reaches a browser. Fix: log the exception with a request
id, return the id only. Also distinguish "model unavailable" (503) from "bad
request" (422) instead of collapsing both.

### O3 (medium) — `requires_table_check` is a hint with no consumer
Triage flags tabular/tiered questions (precision 86%, recall 100% on 17 labelled
cases at prompt v2), and the flag currently only shapes the generated search
queries. There is no `search_tables` tool or metadata filter, so the flag is
close to decorative. Either give the tool a `tables_only` filter (chunks whose
text contains a pipe table) or delete the field — a calibration number nobody
acts on is worse than no number.

### O4 — FIXED — judge verdicts were not reproducible across runs
Same variant, same case, temperature 0: variant A produced 4 vs 5 "unfaithful"
verdicts across two runs. A verdict cache keyed by
`sha(prompt_version + question + answer + contexts)` now replays from
`DATA_DIR/judge_cache.json` (disable with `JUDGE_CACHE=false`). `_inventory()` is
deliberately *not* part of the key: it decorates the prompt but does not change
which passage text was judged.

### O5 (medium) — PDF tables lose their structure
`app/readers.py` uses pypdf's text layer, so a PDF table arrives as runs of
numbers with no column information. The sample corpus is Markdown/text, where
tables survive as pipe tables and the "does the Standard tier cover dental
implants" case works, but a real PDF benefit table would degrade. Fix:
pdfplumber table extraction rendered back to Markdown at read time, so chunking
sees the same shape it sees today for .md.

### O6 (low) — ingestion cost is unbounded per window
`INGEST_WINDOW_CHARS=6000` windows each get an LLM parse; a 100-page policy is
dozens of calls with no cache. Fix: cache the parse keyed by a hash of the window
text (re-uploads become free) — the same trick the intent layer uses.

### O7 — FIXED — uploads were never pruned
Deleting a document removed rows but not the file under `data/uploads/`.
`AdminAPI.delete_document` now unlinks the stored file (best-effort: an OSError
is logged, never raised, because the row is already gone).

### O8 (low) — the scan is O(chunks × top_k) per document, with no prefilter
Fine for the demo corpus, not for a 100-document library. Fix: prefilter
candidate pairs by a cheap lexical overlap score before the embedding search, or
scan incrementally by content hash.

### O9 (low) — no auth, no rate limit
Explicitly out of scope in the assignment, recorded so it is a *decision*:
`/api/ask` is unauthenticated and costs an LLM call per request. A demo-only
system should not be exposed to the internet without a token.

### O10 (low) — the eval harness is not wired to CI
`eval/run_ablation.py`, `measure_retrieval.py`, `run_intent_eval.py` are manual.
A `make eval` target (or a pytest marker that runs the label-only retrieval
check) would catch retrieval regressions for free, since that one needs no LLM.

---

## 3. Looks like a defect, is deliberate

* **Fail-open triage.** If the classifier errors out, the question goes to the
  agent instead of a refusal. Rationale: the agent is the grounded component and
  owns refusals; a triage outage must not become a coverage outage. The cost is
  that a broken classifier degrades quality, never availability.
* **Refusals never happen in triage for workplace questions.** The triage layer
  has not read the corpus, so it cannot know that a policy is missing. Only
  non-HR input and injection are refused before retrieval.
* **Sticky capability flags per client.** Once a provider rejects `tools`,
  `response_format` or `reasoning_effort`, the client stops sending that field
  for its lifetime rather than retrying it on every call.
* **One shared SQLite connection behind an RLock.** Simple, correct for one
  process, and it keeps the vector matrix cache coherent. A connection pool is
  the next step, not the first.
* **`DECISION_MODE=remote` is expensive.** Every candidate pair becomes a remote
  HTTP call with a 15s timeout. That is the price of a pluggable decision model;
  `heuristic` is the demo default for a reason.
* **The contradiction cascade stays dense-only** even with hybrid retrieval on,
  because "semantically adjacent" is the question it asks, and keyword overlap
  finds repeated clauses rather than conflicts.
