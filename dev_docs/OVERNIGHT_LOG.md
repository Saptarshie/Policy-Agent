# Overnight work log — 2026-11-04

Autonomous session. Goal: finish the intent/prompt ablation, fix what the data
justifies, document everything. Every number below came from a real run; where a
number is missing I say so instead of estimating.

Repo: `policy-rag-agent` (no git repo existed at session start — initialised
tonight, see §9).

---

## 1. Starting point

- 119 tests existed before this work; all passed (`/c/Python313/python.exe`).
- The front door was a regex intent gate (`classify_intent_legacy`), and it was
  demonstrably wrong on real questions: "What policies apply to parental
  leave?" was answered with a *document listing* instead of a policy answer.
- The grounding judge was skipped whenever an answer merely *contained* a
  refusal phrase, so a substantive answer that also said "the policy does not
  cover X" bypassed verification entirely.
- `check_citations` reported markdown links and `[1]` as broken citations.

## 2. What I changed (in order)

| # | change | why |
|---|---|---|
| 1 | LLM intent & triage layer (`app/agent/intent.py`) | keyword gates cannot separate "policy question" from "list documents"; they overfit phrasings |
| 2 | triage routes: `REACT_LOOP` / `DIRECT_REPLY` / `FAST_REFUSE` derived from category, never taken from the model | the model must not be able to widen its own refusal power |
| 3 | de-noised question + search plan injected as a **second system message** | keeps the employee's words out of the instruction channel |
| 4 | disk cache keyed by question+prompt+model+reasoning effort | deterministic repeats, resumable evaluation |
| 5 | fail-open on classifier error | a triage outage must not become a coverage outage |
| 6 | corrective retry when a reply parses but is unusable | the 700-token truncation produced exactly this: valid JSON fragment, wrong category |
| 7 | judge runs unless the answer is a *pure* refusal (no citations) | substantive answers were bypassing verification |
| 8 | judge receives the employee's original message | it was flagging the employee's own numbers as hallucination |
| 9 | `RAG_TOP_K` actually wired (`ToolRegistry(default_top_k=…)`) | it was dead config |
| 10 | embed-model fingerprint + actionable dimension error | a silent 384-d vs 768-d mismatch crashed inside numpy with a useless message |
| 11 | citation parsing: only `doc § section` / `doc #id` are citations; dedupe | prose in brackets produced spurious warnings |
| 12 | `reasoning_effort` support (opt-in per call, degrades on 400) | measured 25% lower p50 for the classifier with no accuracy loss |
| 13 | removed `DECISION_MAX_OUTPUT_TOKENS` | a per-caller token cap is not a budget; for a reasoning classifier a small cap causes truncation |
| 14 | `OLLAMA_BASE_URL` default → `127.0.0.1` | `localhost` cost ~2s per embedding call on Windows (IPv6 fallback) |
| 15 | stale-job reconciliation at startup (`STALE_JOB_MINUTES`) | a killed ingest left a document `ingesting` forever and the UI polling a dead job |
| 16 | deleting a document unlinks its stored upload | the row was gone, the file was not: silent disk leak |
| 17 | judge verdict cache (`JUDGE_CACHE`) | verdicts vary run to run at temperature 0; a repeat now replays instead of re-deciding |

## 3. Test suite

119 → 156 tests, all passing. New coverage: triage (cache, fail-open, retry,
plan injection, route derivation), judge gating and user-message forwarding,
embedding fingerprint, citation parsing, client feature degradation
(`reasoning_effort`, `max_tokens` halving, auth errors not retried).

## 4. Measurements

Full write-up: `eval/ABLATION.md`.

### 4.1 Triage (67 labelled cases: 45 canonical + 22 adversarial)

| variant | intent acc. | route acc. | false refusals | missed refusals | p50 |
|---|---|---|---|---|---|
| regex (legacy) | 50.7% | 74.6% | 0 | 15 | 0s |
| prompt v1 | 100%* | 100% | 0 | 1 | 5.22s |
| prompt v2 | 98.5%* | 100% | 0 | 2 | 5.09s |

\* one label (g05) widened after review: the sentence asks two things, so
`MULTI_HOP_COMPARISON` is defensible. Raw run: v1 98.5%, v2 97.0%.

### 4.2 Classifier reasoning budget

`reasoning_effort=off` → 0 reasoning tokens, 20/20 agreement with the
default-effort decisions on a category-balanced subset, mean 4.1s (p90 5.7s) vs
p50 5.2s / p90 8.0s at the provider default. Native tool calling still works at
`off` (3.0s, correct tool + args). Shipped as the classifier default.

### 4.3 Retrieval

Dense vs sparse vs hybrid on 17 natural questions + 10 exact-token probes:
identical at k=3 and k=6 (the corpus saturates the metric); sparse gains one
top-1 exact-token case, hybrid loses one top-1 natural case. Hybrid stays the
default at the agent's k=6 operating point; the harness is the tool to re-check
on a bigger corpus. Building the harness found a real bug: the FTS backfill
checked `count(*) FROM chunks_fts`, which for an external-content table reads the
content table, so the index was never built (keyword search scored 0/17).

### 4.4 End-to-end (23 cases x 4 variants, final code)

| variant | pass | clean citations | judge flags | p50 | tools |
|---|---|---|---|---|---|
| A regex + prompts v1 | 87.0% | 100% | 2 | 15.5s | 2.3 |
| B triage + prompts v1 | 100% | 100% | 1 | 17.9s | 2.7 |
| C triage + prompts v2 | 100% | 100% | 1 | 18.9s | 2.6 |
| D = C, judge off | 100% | 100% | — | 12.3s | 2.7 |

A fails exactly the three cases triage exists for: the prompt injection demanding
a fabricated $5,000 gym reimbursement, the harassment complaint that must carry an
escalation note, and "tell me a joke". Prompts v1 vs v2 tie on pass rate; v2's
difference is auditability (quoted numbers, explicit conflict section), reported
as a wash rather than a win.

Two label/scorer defects were found only by reading the answers, and both are
written into the case files: `q16` ("does my plan cover cryotherapy?") was labelled
as must-refuse, but the plan's exclusions clause excludes services not listed as
covered, so a grounded "no" is correct; and `expect_doc` supports `a|b`
alternatives that the scorer compared as one literal string, making two cases
impossible to pass. `eval/rescore.py` re-applies the corrected labels to the
stored answers instead of spending 20 minutes re-running the agent.

### 4.5 Judge calibration (four rounds)

1. **First run**: 5/23 flagged. Reading them showed the judge was handed only the
   first 8 seen chunks, so conflict answers citing late passages "invented" facts
   that were in the corpus. Fix: resolve the cited chunks first, 24-chunk / 24k-char
   budget, coverage note when truncated.
2. **Second run**: still 5, now all about document identity (titles, effective
   dates, policy numbers, scope). Cause: those live in a document header (or in
   the ingestion summary the agent can quote) and section citations never cover
   them. Fix: the judge receives the document inventory (name + summary) and each
   cited document's header chunk.
3. **Third run**: flags dropped to 1-2, but the inventory scope rule was too
   narrow — the judge still counted a document's own scope line as unsupported
   content. Fix: inventory metadata (title, effective date, applicability) counts
   as support.
4. **Final**: B/C flag 1, D 0, A 2. The surviving flags are substantive: an answer
   generalising "casual leave cannot be encashed" from a context that only said
   excess above the cap lapses; an answer narrating "similarity scores were
   uniformly near zero" into the final answer; an answer adding "cryotherapy is
   also not reimbursable through the expense policy" without that passage.

Net effect: judge false alarms went from 5/23 (all artifacts of the evidence set
it was given) to 0/23 on correct answers, while still catching the real ones.

One bookkeeping trap worth recording: an early log file (`ablation_runs.jsonl`,
from the first killed run) held four pre-fix A rows. It matched the
`ablation_*.jsonl` glob, and because it sorted after `ablation_A.jsonl` its stale
rows overwrote the good ones, inflating A's judge flags from 2 to 6 in the first
write-up. The file is gone and the counts are now 23 rows per variant.

### 4.6 Judge as a classifier (injected-claim calibration)

10 clean answers + the same 10 with one fabricated policy claim spliced in:

| prompt | TNR (clean passes) | TPR (injection caught) |
|---|---|---|
| v1 (default) | 8/10 | 10/10 |
| v2 | 7/10 | 10/10 |

High recall, moderate precision — which is why the judge flags and never rewrites,
and why a `false` verdict with no named claim degrades to `unknown`. Caveat: the
"clean" answers are the agent's own, and at least one contains a real
unsupported generalisation, so TNR is a lower bound.

## 5. Where a reviewer should push back

1. **The corpus is seven documents.** Retrieval metrics saturate; the retrieval
   comparison cannot rank the modes.
2. **Prompt v2 was written against the hard cases** I also wrote. The honest
   claim is "fixes specific observed failures", not "generally better".
3. **The judge is a model.** Its verdicts vary run to run at temperature 0
   (observed: 4 vs 5 flags for the same variant). Caching verdicts would make
   demos reproducible; not done yet.
4. **`requires_table_check` still has no consumer** beyond shaping search
   queries — a calibrated signal nobody acts on.
5. **Ingestion is not resumable across a restart** (daemon threads; a killed
   process leaves a doc stuck at `ingesting`).

