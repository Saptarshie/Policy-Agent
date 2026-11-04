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

| #   | change                                                                                                         | why                                                                                              |
| --- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1   | LLM intent & triage layer (`app/agent/intent.py`)                                                              | keyword gates cannot separate "policy question" from "list documents"; they overfit phrasings    |
| 2   | triage routes: `REACT_LOOP` / `DIRECT_REPLY` / `FAST_REFUSE` derived from category, never taken from the model | the model must not be able to widen its own refusal power                                        |
| 3   | de-noised question + search plan injected as a **second system message**                                       | keeps the employee's words out of the instruction channel                                        |
| 4   | disk cache keyed by question+prompt+model+reasoning effort                                                     | deterministic repeats, resumable evaluation                                                      |
| 5   | fail-open on classifier error                                                                                  | a triage outage must not become a coverage outage                                                |
| 6   | corrective retry when a reply parses but is unusable                                                           | the 700-token truncation produced exactly this: valid JSON fragment, wrong category              |
| 7   | judge runs unless the answer is a *pure* refusal (no citations)                                                | substantive answers were bypassing verification                                                  |
| 8   | judge receives the employee's original message                                                                 | it was flagging the employee's own numbers as hallucination                                      |
| 9   | `RAG_TOP_K` actually wired (`ToolRegistry(default_top_k=…)`)                                                   | it was dead config                                                                               |
| 10  | embed-model fingerprint + actionable dimension error                                                           | a silent 384-d vs 768-d mismatch crashed inside numpy with a useless message                     |
| 11  | citation parsing: only `doc § section` / `doc #id` are citations; dedupe                                       | prose in brackets produced spurious warnings                                                     |
| 12  | `reasoning_effort` support (opt-in per call, degrades on 400)                                                  | measured 25% lower p50 for the classifier with no accuracy loss                                  |
| 13  | removed `DECISION_MAX_OUTPUT_TOKENS`                                                                           | a per-caller token cap is not a budget; for a reasoning classifier a small cap causes truncation |
| 14  | `OLLAMA_BASE_URL` default → `127.0.0.1`                                                                        | `localhost` cost ~2s per embedding call on Windows (IPv6 fallback)                               |
| 15  | stale-job reconciliation at startup (`STALE_JOB_MINUTES`)                                                      | a killed ingest left a document `ingesting` forever and the UI polling a dead job                |
| 16  | deleting a document unlinks its stored upload                                                                  | the row was gone, the file was not: silent disk leak                                             |
| 17  | judge verdict cache (`JUDGE_CACHE`)                                                                            | verdicts vary run to run at temperature 0; a repeat now replays instead of re-deciding           |

## 3. Test suite

119 → 156 tests, all passing. New coverage: triage (cache, fail-open, retry,
plan injection, route derivation), judge gating and user-message forwarding,
embedding fingerprint, citation parsing, client feature degradation
(`reasoning_effort`, `max_tokens` halving, auth errors not retried).

## 4. Measurements

Full write-up: `eval/ABLATION.md`.

### 4.1 Triage (67 labelled cases: 45 canonical + 22 adversarial)

| variant        | intent acc. | route acc. | false refusals | missed refusals | p50   |
| -------------- | ----------- | ---------- | -------------- | --------------- | ----- |
| regex (legacy) | 50.7%       | 74.6%      | 0              | 15              | 0s    |
| prompt v1      | 100%*       | 100%       | 0              | 1               | 5.22s |
| prompt v2      | 98.5%*      | 100%       | 0              | 2               | 5.09s |

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

| variant               | pass  | clean citations | judge flags | p50   | tools |
| --------------------- | ----- | --------------- | ----------- | ----- | ----- |
| A regex + prompts v1  | 87.0% | 100%            | 2           | 15.5s | 2.3   |
| B triage + prompts v1 | 100%  | 100%            | 1           | 17.9s | 2.7   |
| C triage + prompts v2 | 100%  | 100%            | 1           | 18.9s | 2.6   |
| D = C, judge off      | 100%  | 100%            | —           | 12.3s | 2.7   |

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

| prompt       | TNR (clean passes) | TPR (injection caught) |
| ------------ | ------------------ | ---------------------- |
| v1 (default) | 8/10               | 10/10                  |
| v2           | 7/10               | 10/10                  |

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
   (observed: 4 vs 5 flags for the same variant). Verdicts are cached per
   (prompt version, question, answer, contexts) in `DATA_DIR/judge_cache.json`,
   so a re-demo is reproducible; a fresh corpus or a prompt change is not.
4. **`requires_table_check` still has no consumer** beyond shaping search
   queries — a calibrated signal nobody acts on.
5. **Ingestion is not resumable across a restart** (daemon threads; a killed
   process leaves a doc stuck at `ingesting`).

## 6. Correction: the verifier was classifying language with a phrase list

Recorded because I made the mistake, defended it in writing, and the user called
it out — the note is here so the next reader does not re-introduce it.

### 6.1 What was wrong

```python
_REFUSAL_MARKERS = ("do not cover", "does not cover", "don't have enough information",
                    "do not have enough information", "not covered by the ingested",
                    "no information", "cannot find", "couldn't find", "not mention")
looks_like_refusal(answer) = any(m in answer.lower() for m in _REFUSAL_MARKERS)
```

That decided whether the faithfulness judge ran at all. Three failures, all real:

1. **Rewording breaks it.** "There's nothing on that in what I've got" is a
   refusal; the list does not know it.
2. **It cannot see a partial answer.** The important shape is *"X and Y happen if
   you resign, but the documents do not say what happens if you transfer"* — a
   substantive answer plus a corpus claim. The list classifies that by whichever
   phrase happens to be present.
3. **Every fix is another phrase.** A lexical list for a natural-language
   property never converges; it just overfits the cases you happened to test.

The rule this violates: language decisions belong to a model, not to a string
list. Regex stays for structural parsing (splitting a citation on `§`, collapsing
whitespace) — where the input grammar really is deterministic.

### 6.2 What replaced it

`app/agent/answer_shape.py`, built on the decision model the corpus already
uses for contradiction screening (`typesafe/jev`, `DECISION_MODE`). Four yes/no
questions, probabilities back, shape derived from them:

```
declines_part   does the answer decline part of the question, saying the documents don't cover it?
states_rules    does it state specific rules, numbers, limits, dates, tiers or obligations?
is_only_refusal is it ONLY a refusal, with no policy specifics anywhere?
is_chitchat     is it only greeting / thanks / small talk?
```

    probabilities → answer | partial | refusal | chitchat | unknown

Measured on the live endpoint, single request, 435 input tokens, 2.1s:

```
"If you resign, unused casual leave lapses on your last working day ... but my
 policy documents do not say what happens to your carry-forward balance if you
 transfer to another entity mid-year."
    declines_part 0.96   states_rules 0.86   is_only_refusal 0.04   is_chitchat 0.01
    → partial
```

Design rules, all of them there to make failure boring:

* `DECISION_MODE=remote|hybrid` → decision model. `heuristic` (no endpoint) →
  the chat model's strict-JSON classification, same questions. Neither available
  → `unknown`.
* **`unknown` means "judge it".** A classifier outage, a timeout, an unparseable
  reply and contradictory probabilities all land there, so the failure direction
  is more checking, never less.
* The model accepts a long state (documented 32K for a single request), so the
  classifier sees the question, the full answer and a 6k-char passage sample —
  enough to tell a refusal apart from an answer that *contains* a refusal.
* Scores are stored on the result (`answer_shape`, `declined_parts`), so a
  reviewer can see which shape decided the routing.

### 6.3 Partial answers are the normal case, not an exception

Both agent prompts gained an explicit rule (v1 `5b`, v2 `6b`): answer the parts
the documents cover, then name precisely the part they do not — never refuse the
whole question because one part is uncovered, never present an uncovered part as
documented.

On the verifier side a partial answer is **always judged**, and the judge is told
the declined parts and given two rules:

1. the answered half: verify numbers/rules/dates against the passages as usual;
2. the declined half: it is a claim *about the corpus*. Supported only if no
   passage addresses it. If a passage does address it — even partially — the
   decline is unsupported, and the judge is asked to quote the passage that
   contradicts it.

Both judge prompts also state the converse, so a correct decline is not punished:
a decline is faithful when the contexts genuinely do not cover the topic.

### 6.4 What changed in behaviour

| answer | before | now |
|---|---|---|
| "Hello! How can I help?" | judged (wasted call) | skipped (`chitchat`) |
| canonical refusal, passages exist | **skipped** — a wrong refusal could not be caught | judged: the decline is checked against the passages |
| "X lapses on resignation … documents do not say about transfer" | classified by whichever phrase matched; often skipped | `partial`: both halves verified, declined part named in the payload |
| refusal worded outside the list ("nothing on that in my docs") | skipped silently | judged as a refusal |
| classifier down | n/a (no classifier) | `unknown` → judged |

### 6.5 Cost

One extra call per answered question when no decision model is configured
(chat-model fallback), and ~2s when one is (the JEV request above). With
`DECISION_MODE=remote` this is cheaper than the judge it guards, and it is the
same endpoint the contradiction cascade already pays for. Ablation B/C were
re-run after the change so the published numbers include it.

### 6.6 Re-run and what it showed

B and C re-run on the final code, decision model as classifier:

```
B  100% pass   3 judge flags   p50 25.1s (2-way concurrent)
C  100% pass   1 judge flag    p50 25.8s
```

Two things worth keeping:

1. **Flag counts are not stable.** B measured 0 flags, then 3; C measured 3,
   then 1 — same code, same cases, temperature 0. At n=23 one flipped verdict is
   4.3%, and the judge's TNR is 8/10, so a prompt cannot be ranked by flag
   count. The pass rate reproduced exactly. That is the metric to trust.
2. **The exhaustive-list rule made one flag stricter, and the judge was right.**
   q16 ("is cryotherapy covered?") answers "no, it is not in the covered-benefits
   table". The new rule says a conclusion from an *exhaustive* list is supported;
   the judge then noticed the table calls itself a *summary of key benefits* and
   flagged the inference from absence. That is a better reading than my earlier
   relabel of the case, so the disagreement is recorded in `eval/ABLATION.md`
   instead of being papered over by another prompt tweak.

Reporting fix in the same pass: `clean citations` was counting faithfulness
warnings, so a grounding failure was being reported as a citation-resolution
failure. The two are now counted separately.

## 7. Reversal: the verifier is out of the answer path

Section 6 above describes building the answer-shape classifier so the
faithfulness judge could run on more answers. Days later the judge is gone from
`ask()` entirely, so the trail is worth reading in order.

Why it was removed:

| measurement | value |
|---|---|
| pass rate with the judge (C) vs without (D) | 100% vs 100% (23 labelled cases) |
| latency | +6s p50 (18.9s vs 12.3s) |
| precision on honest answers | TNR 8/10 - roughly one answer in five flagged |
| flag-count stability | 0 then 3 (variant B), 3 then 1 (variant C), identical code |
| genuine catches | one wrong chunk-id citation, one disputed transition rule |

The judge was the same `deepseek-v4.1-flash` as the agent, so it was never a
second opinion from a stronger model - only a second context. That has some value
(it caught e01), but not six seconds and a fifth of the honest answers, and it
made the product's most visible signal ("faithfulness check FAILED") the least
reliable thing in it.

What replaced it: the prompt contract (grounding, search-first, citation format,
refusal, weak evidence, injection) plus two new prompt rules about derived
numbers, and deterministic citation resolution that refuses to guess - see
`dev_docs/HEURISTIC_AUDIT.md`.

What was kept: `app/agent/verify.py`, `app/agent/answer_shape.py`,
`eval/calibrate_judge.py`, `eval/run_shape_eval.py`, and the ablation harness,
which now applies the judge itself for variants that ask for it. The measurement
apparatus outlives the feature, which is the only reason the removal can be
justified with numbers instead of taste.

Also caught by the audit while doing this: citation resolution attributed
"Remote Work Policy v2" to `remote_work_policy_v1` (first substring/fuzzy match
wins, 0.72 threshold, the two names are 0.97 similar). Fixed with unique-candidate
resolution; ambiguous references now warn instead of resolving. `tests/test_naming.py`.

## 8. Working memory and arithmetic for the agent

Added two capabilities to the ReAct toolset, both *optional by prompt rule* - no
complexity detector decides anything:

* `calculate` - arithmetic over named variables, AST-whitelisted (no `eval`, no
  attributes, no imports), returning the value and the resolved expression. This
  is also what makes a derived number auditable: the prompt requires computed
  figures to name their inputs and the calculation, so a reviewer can re-check
  12 x 7 / 12 instead of trusting prose.
* `scratchpad_write / read / edit / delete / list` - per-question working memory,
  in memory, bounded (12 notes, 4k chars each, 12k total), never persisted and
  never treated as evidence. It exists for long, multi-document questions where
  the failure mode is reconstructing a number read twenty turns ago.

Measured on the live stack (deepseek-v4.1-flash, 7-document corpus):

| question | calls | notes | parallel turns | latency |
|---|---|---|---|---|
| "How many casual leave days do I get per year?" | 2 | 0 | 1 | 9.7s |
| "What is the notice period to change my remote arrangement?" | 4 | 0 | 2 | 16s |
| compare v1/v2 carry-forward + pro-rate a resignation | 18 | 2 written, 1 read back | 7 | 60s |

The first prompt version produced **zero** notes on the complex question: the
model judged itself able to hold six passages. Two wording changes fixed it, and
both are prompt-only: (1) name the trigger the model already sees in its own
INTENT PLAN ("if the intent is MULTI_HOP_COMPARISON ..."), and (2) put the
read-back step in the final-answer section ("if you wrote notes, read them before
answering"), because otherwise it writes notes and never reads them - the first
tuned run wrote two notes and read none.

Cost, stated plainly: the complex question went from 34s / 8 calls to 60s / 18
calls. That is the price of the notes-plus-arithmetic path, and it is paid only
on multi-hop questions; the simple cases did not get slower (9.7s vs 14.7s before
the change, because batching cut a round trip).

Parallel tool calling needed no code: the loop already executed every
`tool_calls` entry in a turn and answered with one tool message per call id. It
was never *used* because nothing said it was allowed. One prompt rule (plus
reconciling "one action per turn" with it) produced 7 batched turns on the
complex question and 1-2 on the simple ones; `parallel_turns` on `AgentResult`
makes it observable.

## 9. Contradiction detection: similarity-gated candidates, intra-document, one judge

Four changes, all of them correcting a count-based or lexical shortcut.

**1. Candidates are gated by similarity, not by top_k.** The scanner used to take
the best `CONTRA_TOP_K=4` neighbours per chunk and escalate at most
`SCAN_MAX_PAIRS_PER_CHUNK=3`. Measured on the sample corpus: the real leave-cap
conflict (`leave_policy §3.2` vs the 2025 amendment) sits at **0.678, rank 3 of 4**
- inside the old cut by luck. With 60 similarly-worded chunks it would rank far
below any small k and never be seen. Now: every passage at or above
`SCAN_MIN_SIMILARITY` is a candidate; `SCAN_MAX_CANDIDATES` (200) is a logged
safety valve, not a relevance decision.

**2. A calibration bug found while testing the above.** The first threshold I
shipped (0.65) was measured stored-vs-stored, but the scanner was calling
`search(chunk_text)`, which embeds the text as a QUERY. With a model that
prefixes queries and documents differently, that is a lower number: the same pair
measures **0.601** query-vs-stored - below my own floor, i.e. the threshold would
have excluded exactly the conflict it was calibrated for. Fixed by adding
`find_similar_chunks()` / `db.similar_to_chunk()`, which compare stored vectors to
stored vectors (also one embedding call cheaper per chunk). After the fix the pair
is a candidate at 0.678 and the scan flags it: gate score 0.94, flagged as
contradiction #2.

**3. Intra-document scanning is on by default.** `SCAN_CROSS_DOC_ONLY=false` now
means "same document as well as across documents"; the code previously could only
exclude same-document pairs. The sample corpus has no intra-document conflict
beyond what the gate rejects, so this is a capability change, not a new finding -
and the fixtures prove the path works (`tests/test_threshold_scan.py`).

**4. One judge.** `DECISION_MODE` is gone. The lexical gate (`JEVStyleGate`) still
runs, but its output is a *signal list handed to the decision model as evidence*;
it can no longer pass or drop a pair. With no `DECISION_MODEL_*` configured the
gate returns `decided_by="unavailable"` and the scan logs "INCOMPLETE, not clean"
instead of reporting zero contradictions. That distinction matters: a scan that
could not judge anything must not look like a clean corpus.

**5. The agent's search tool can now do both depths.** `search_policy_docs` takes
`top_k` OR `min_similarity`, never both (asking for both returns a tool error
naming the conflict). Neither = `RAG_TOP_K`. The prompt explains the trade-off so
the model can choose: a count for an ordinary lookup, a floor when it suspects
several documents touch the question.

Cost: on 52 chunks, a full scan now examines 52 pairs at the 0.65 floor instead of
`chunks x 3`. Each candidate costs one decision-model call (~1.5s); only
gate-passing pairs reach the verifier.

## 10. Batching and threading the scan: 216s -> 30s, same flags

Two cheaper shapes of the same work, both measured with `eval/measure_scan.py`
(full corpus, 7 documents / 52 chunks, fresh database copies per configuration so
no pair is skipped, real decision model + real verifier).

**Batching is a questions-per-request trick, not a states-per-request one.** The
endpoint accepts a LIST for `state` and returns HTTP 200 - but it is not a batch:
`[A, B]` with A (same subject) and B (different subject) returns ONE answer of
0.29, i.e. it concatenates. What it does support is many *questions* over one
state, so a batch of passage pairs becomes one state listing `PAIR 1..n` and one
question per pair. The passages are the shared prefix: 6 pairs cost 810 input
tokens batched vs 2029 one at a time.

**Batching alone moves verdicts.** On 6 hand-built pairs the batched scores
differed from the per-pair scores by up to 0.24, and two decisions flipped across
the 0.5 threshold (0.66 -> 0.42, 0.43 -> 0.62). So the batch result is a first
pass only: every pair within `DECISION_RECHECK_BAND` (0.15) of the threshold is
re-asked in its own request, where the context is a single pair again. Measured
6/6 agreement with per-pair scoring after the re-check.

**Both stages are I/O, so both are threaded.** `SCAN_WORKERS` (4) runs batches
concurrently in the gate and verifications concurrently in the tracker; SQLite
writes stay on the calling thread and in scan order, so the flags are identical.

| configuration | wall | speedup | decision-model requests | verifier calls | flags |
|---|---|---|---|---|---|
| serial, one request per pair | 216.2s | 1.0x | 96 | 16 | 6 |
| batched (6 pairs/request) | 52.7s | 4.1x | 20 | 7 | 6 |
| batched + 4 threads | **29.8s** | **7.3x** | 21 | 6 | 6 |

Same six flags in all three runs. Note the verifier column: batching sent the
verifier 7 pairs instead of 16, and it still confirmed the same 6 - the nine pairs
that stopped reaching it were ones it had rejected anyway. That is the honest
reading of "same flags": the confirmed conflicts score far above the threshold and
survive batching; the pairs sitting near it are the ones whose scores move, which
is exactly what the re-check band is for.

Also settled en route: `state` as a list, `states`, `batch`, `items`, `/batch` and a
`prefix` field are all either ignored or rejected - there is no hidden batch route,
and the re-check band is the price of the one that exists.

## 11. UI verification: five defects found by using it

Ran the real server (real LLM, real decision model) on a copy of the corpus and
clicked through the page. 42/42 API checks, 14/14 renderer cases, no console
errors - after five fixes. Full detail in `dev_docs/UI_VERIFICATION.md`.

1. **Dead buttons.** Document cards built inline handlers with user data
   (`onclick="delDoc(11, 'Bob's travel policy')"`). `esc()` writes `&#39;`, the
   HTML parser decodes it back to `'` before the JS parser sees the attribute, so
   the handler is a syntax error and Delete silently does nothing. Now
   `data-*` attributes plus delegated listeners; no user data reaches a script
   context.
2. **Raw markdown.** `## headings`, `**bold**` and `|---|` tables were shown
   literally in the chat. Added an escape-then-format renderer.
3. **Inventory answered instead of content.** "Do the policy documents contradict
   each other about remote work?" went to META_DOCUMENTS / DIRECT_REPLY and got
   the document list in 0.0s. META_DOCUMENTS is now inventory-only in both intent
   prompts; content questions about the corpus (conflicts, staleness, gaps) must
   reach the agent.
4. **The agent could not see the flags.** It had no way to read what the cascade
   had already verified. New tool `list_contradictions`; on that question the
   agent now calls it first and answers from the audit trail with citations.
5. **Stale intent cache.** The cache key was `version|model|question`, so editing
   a prompt under the same "v1" label replayed pre-edit verdicts - which is why
   bug 3 looked unfixed right after the patch. The key now carries a fingerprint
   of the prompt text.

## 12. Live trace in the web UI (SSE)

The chat pane used to show a spinner and then dump the whole ReAct trace at the
end, so a 60-second multi-hop answer looked like a hang. The trace is now
streamed: `POST /api/ask/stream` emits `stage` / `turn` / `step` / `final` frames
as the agent works, each rendered as its own collapsible row.

Mechanics: `ReactAgent.run(..., on_event=...)` and `AdminAPI.ask(..., on_event=...)`
take a callback; the route runs the (blocking) ask on a worker thread and bridges
its events into an SSE generator through a queue, with 15-second keep-alive
comments so a buffering proxy cannot make the UI look dead. `POST /api/ask` is
unchanged, and the UI falls back to it if the stream fails, so nothing that
worked before depends on the new path.

Measured on the running app (`eval/watch_stream.py`, deepseek-v4.1-flash):

```
+  0.0s  stage  triage
+  3.6s  stage  routed: POLICY_DIRECT_LOOKUP -> REACT_LOOP
+  5.6s  turn   1
+  8.7s  step   thought
+  9.1s  step   tool_call search_policy_docs({"query": "...", "top_k": 6})
+  9.4s  step   tool_call search_policy_docs({"query": "..."})     <- parallel turn
+  9.4s  turn   2
+ 13.1s  step   tool_call list_contradictions({"status": "all"})
+ 20.9s  step   final
+ 20.9s  FINAL  4 tools, 15.3s, 5 citations, 0 warnings
```

In the browser (Chromium, 8 steps): the first three rows were on screen at 4.5s
with the answer still empty, folding the trace at 4.5s kept it folded as four
more steps arrived, and the final answer landed at 16.6s with the summary
"ReAct trace - 5 tool call(s), 15.6s, 8 step(s)". Tests: `tests/test_stream.py`
(frame order, a listener that raises, empty question, the plain endpoint).

### New conversation

The chat pane had no way to drop context: `history` accumulated for the life of
the tab, so a follow-up three questions later was still read against the first
one. The chat header now carries **＋ New conversation**: it aborts an in-flight
stream (`AbortController`), empties the client-side `history` the agent is given,
restores the welcome bubble, clears the box and focuses it. It only clears the
conversation - corpus, flags and server state are untouched.

Verified in the browser: after two turns `history` was 4 and the pane held 5
bubbles; one click gave `history` 0 and 1 bubble with focus in the textarea.
Mid-stream, a click at 4s (2 steps rendered) dropped the pane to the welcome
message and the aborted answer never appeared - checked again 20s later.
