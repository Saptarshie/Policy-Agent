# Ablation & calibration — what was measured, and what it changed

Everything here comes from a real run against the local stack
(`embeddinggemma` via Ollama for embeddings, `deepseek-v4.1-flash` via an
OpenAI-compatible endpoint for generation, `DECISION_MODE=heuristic`). No number
below is estimated; where a measurement is too small or too saturated to support
a conclusion, that is stated instead of dressed up.

Reproduce:

```bash
export DATA_DIR=./data OLLAMA_BASE_URL=http://127.0.0.1:11434 DECISION_MODE=heuristic

python eval/run_intent_eval.py --variant legacy,v1,v2 --hard   # triage layer
python eval/run_ablation.py --variant A,B,C,D                  # end-to-end answer quality
python eval/report_ablation.py                                 # merged table
python eval/measure_retrieval.py --k 1,3,6                     # retriever comparison
python eval/calibrate_judge.py --samples 10                    # judge TPR/TNR
python eval/rescore.py                                         # re-apply corrected labels
python eval/run_shape_eval.py                                  # answer-shape classifier

python eval/run_benchmark.py                                    # Northstar 20-question pack (§9)
python eval/run_benchmark.py --workers 10                       # same, 10 questions in flight
python eval/profile_ingest.py --runs 2 --effort off             # one ingestion call (§10)
python eval/compare_parse_modes.py                              # chunk identity across parse modes
```

All three table-producing scripts share one scoring implementation
(`eval/scoring.py`), so the pass rule cannot drift between "as run" and
"re-scored".

---

## 0. Index — every measurement in this file

Sections are in the order the work happened, not in order of importance. Each row is a
question that was actually asked of the system, the answer the measurement gave, and
where the evidence lives. Wrong answers are analysed one at a time in
[`eval/FAILURES.md`](FAILURES.md); this file is about *aggregates and mechanisms*.

| section | question it answers | headline result |
|---|---|---|
| 1 | why the regex intent gate was removed | legacy gate: 74.6% route, 50.7% intent, 15 missed refusals — replaced, kept as baseline |
| 2 | which triage prompt to ship (67 labelled cases) | legacy 74.6% -> v1 100% route, 98.5% intent, false refusals 0; v2 identical on route |
| 3 | does `reasoning_effort` buy the classifier accuracy | no: 20/20 accept-match at `off`, p90 8.0s -> 5.7s; ships at `off` |
| 4 | dense vs sparse vs hybrid retrieval | identical at k>=3 on 7 documents (corpus saturates the metric); sparse +1 exact-token at k=1; found an FTS backfill bug |
| 5 | which layer moves end-to-end accuracy (A/B/C/D, 23 cases) | A 87.0% -> B/C/D 100%; the front door is +13 points, the prompts are +-0, the judge costs 6.6s p50 |
| 6 | are the judge's flags trustworthy | table claim P=50% -> v2 P=86%; flags are a ranking signal, not a gate |
| 7 | what the measurements changed in the code | table: finding -> change |
| 8 | what would invalidate all of this | threats to validity |
| 7a | does removing the answer verifier hurt | 12/12 shape-routing, no accuracy loss; verifier out of the answer path |
| 8a | A/B/C/D after the verifier removal | 87.0 / 100 / 100 / 100, flag rates 8.7% / 13.0% / 4.3% / 0% |
| 9 | the HR pack (20 multihop questions) | 19/20, citations 29/29, 67.8s for the pack at `--workers 10` |
| 10 | where the 502 seconds of ingestion went | 501.7s -> 78.2s parallel, -> 67.8s at 10 workers; chunks byte-identical |
| 11 | the contract pack (20 rubric-scored questions) | 13/20 -> 15/20 after refuse-by-shape; rubric 9.1/10 |
| 12 | a generalised prompt rule, and why it was reverted | negative result: naming a failure mode made the model notice it and still fail |
| 13 | how much a benchmark moves with no code change | 15/20 then 16/20 on identical code; LC-15 hit `AGENT_MAX_STEPS` |
| 14 | does a bounded document brief improve triage | intent 98.5% -> 100%, route flat, false refusals 0; end to end flat |
| 15 | does it survive adversarial, fictional-jurisdiction traps | 16/20, 19/20 with partials, rubric 9.35, section recall 76.2% |

Three numbers to keep in mind when reading any single run below: this corpus has 4-7
documents per pack so retrieval saturates; 20 questions per pack means one verdict is
5 percentage points; and the same code has scored 15/20 and 16/20 on the same pack (§13).

## 1. Why the front door was replaced

The original gate was a keyword classifier (`app/agent/intent_legacy.py`): a
greeting whitelist, a "meta documents" pattern, a handful of policy regexes, and
a fallback. Two failure modes are not fixable by adding patterns, because they
are semantic, not lexical:

1. **Same words, different jobs.** "What policies apply to parental leave?" and
   "What policy documents do you have?" share the tokens *policies* /
   *policy* / *documents*. The regex gate answered the first one with a document
   listing — a coverage outage disguised as an answer.
2. **No words at all.** "My manager told me my 15 unused leaves vanish on Dec 31,
   can they do that?" is a carry-forward question with none of the policy
   vocabulary, plus emotional framing that embeds poorly against formal policy
   prose.

The replacement (`app/agent/intent.py`) is one structured-JSON LLM call that
classifies into eight categories, **derives** the route from the category (the
model never gets to choose "refuse" directly), rewrites the input into a neutral
question, and proposes 1-3 formal search queries.

Three properties make it safe to put in front of a retrieval system:

* **fail-open** — any error, timeout or unparseable reply routes to the agent;
* **no coverage refusals** — it refuses only non-HR input and prompt injection,
  because it has not read the corpus (the agent owns "the documents do not
  cover this");
* **cache + deterministic scoring** — decisions are cached on disk keyed by
  question + prompt version + model + reasoning effort, so an interrupted
  evaluation resumes and repeats are free.

## 2. Triage ablation (67 labelled cases)

45 canonical cases (`eval/intent_cases.jsonl`) + 22 adversarial/ambiguous cases
(`eval/intent_hard.jsonl`: `???`, `ok`, greeting+question, grounding bypass,
"pretend you have no documents", …).

| variant        | intent acc.        | route acc. | false refusals | missed refusals | classifier p50 |
| -------------- | ------------------ | ---------- | -------------- | --------------- | -------------- |
| legacy (regex) | 50.7%              | 74.6%      | 0              | 15              | 0.0s           |
| prompt v1      | 98.5% → **100%***  | **100%**   | **0**          | 1               | 5.22s          |
| prompt v2      | 97.0% → **98.5%*** | **100%**   | **0**          | 2               | 5.09s          |

\* after reviewing one label: `g05` ("I'm moving to another city next month —
does that change my remote work eligibility **or** my notice obligations?") asks
two things, so `MULTI_HOP_COMPARISON` is a defensible answer alongside
`IMPLICIT_SITUATION_OR_GRIEVANCE`. The raw run scored it a miss; the accept-list
was widened and the table rescored offline from the same decisions (no re-run).
The v2 miss that remains is `x22` (a long, rambling situation) classified as
`SENSITIVE_HR_ESCALATION` instead of grievance/multi-hop — both route to
`REACT_LOOP`, so behaviour is unchanged, but it is a real precision loss from
adding the sensitivity rules.

Reading the metrics that matter:

* **Route accuracy is the only number that changes behaviour** (intent labels are
  diagnostic). Both prompts are at 100%; the regex gate was at 74.6%.
* **False refusals are the dangerous error** — refusing a workplace question
  before retrieval is an outage. 0 for all LLM variants, by construction: only
  chit-chat and adversarial categories map to `FAST_REFUSE`.
* **Missed refusals** are the safe error (the agent answers with a grounded
  refusal). The regex gate missed 15 of 22 (it sent `???`, `pretend you have no
  documents`, etc. into the agent); v1 misses 1, v2 misses 2.
* The regex gate's 15 misses and 3 mis-routes are the concrete cost of
  "benchmaxxing" a natural-language gate with patterns.

### 2.1 Calibrating a soft signal: `requires_table_check`

Triage can flag "this question needs a table lookup". Measured against 17
labelled cases:

| prompt | precision | recall | tp/fp/fn |
| ------ | --------- | ------ | -------- |
| v1     | 50%       | 100%   | 6/6/0    |
| v2     | **86%**   | 100%   | 6/1/0    |

v1 fired on anything mentioning a number or a tier; v2 fires only when the
question asks for a *cell/tier-specific* value ("does the Standard tier cover
dental implants?", "out-of-pocket maximum for the Premium tier?") and not for a
global number ("how many days…"). The remaining false positive is a question
with an explicit tier AND a general rule.

The flag is a hint only — it shapes the generated search queries. It does not
gate any tool, which is recorded as an open item in `dev_docs/CODE_REVIEW.md`.
A calibration number with no consumer is worse than no number.

## 3. `reasoning_effort` on the classifier

The endpoint accepts `reasoning_effort` ∈ {off, low, medium, high, xhigh, max}
(`none`/`minimal` are rejected with a 400 listing the valid set). Reasoning
tokens are billed against `max_tokens`, which is what truncated the classifier at
700 output tokens.

Probe: 20 cases, one per category family, compared against the already-measured
default-effort decisions.

| effort          | reasoning tokens | accept-match | mean latency               | p90      |
| --------------- | ---------------- | ------------ | -------------------------- | -------- |
| default (unset) | 8-225            | 20/20        | — (p50 5.2s over 67 cases) | 8.0s     |
| `off`           | **0**            | **20/20**    | **4.1s**                   | **5.7s** |

Native tool calling still works at `off` (correct tool + arguments in 3.0s), so
the classifier ships with `INTENT_REASONING_EFFORT=off`; the agent and judge keep
the provider default (`LLM_REASONING_EFFORT` empty) because answer quality there
is worth the latency and was not measured at reduced effort.

## 4. Retrieval: dense vs sparse vs hybrid

`eval/measure_retrieval.py` is label-only (no LLM calls). `ans@k` = a top-k
passage from the expected document contains the expected answer text, i.e.
retrieval handed the model something quotable.

```
natural questions (17 labelled), k=1/3/6
mode        doc@1   doc@3   doc@6   ans@1   ans@3   ans@6
vector      14/17   15/17   15/17   14/17   15/17   15/17
keyword     14/17   15/17   15/17   13/17   15/17   15/17
hybrid      14/17   15/17   15/17   13/17   15/17   15/17

exact-token probes (10), k=1/3/6
mode        doc@1   doc@3   doc@6   ans@1   ans@3   ans@6
vector       8/10    10/10   10/10    8/10   10/10   10/10
keyword      9/10    10/10   10/10    9/10   10/10   10/10
hybrid       9/10    10/10   10/10    9/10   10/10   10/10
```

Honest verdict: **at k=3-6 on a 7-document corpus the three are identical** — the
corpus saturates the metric, so this cannot prove hybrid superiority. The only
separation is top-1: sparse gains one exact-token case (a policy number / tier
amount, the failure mode a policy assistant cannot afford), and hybrid loses one
natural-language case to a keyword hit outranking a better paraphrase match.
Hybrid is therefore the default at the agent's k=6 operating point (where the
three tie) with the probe evidence covering the tail; the harness is the way to
re-check before flipping the default on a bigger corpus.

A bug found *by building this harness*: the FTS backfill checked
`count(*) FROM chunks_fts`, which for an external-content FTS5 table reads the
**content** table — so an empty index looked complete and keyword search returned
0/17. Fixed to check `chunks_fts_docsize`, with a regression test.

## 5. End-to-end ablation (A/B/C/D)

Variants:

* **A** — regex triage + agent prompt v1 + judge v1 (the original system)
* **B** — LLM triage + agent prompt v1 + judge v1 (isolates the front door)
* **C** — LLM triage + agent prompt v2 + judge v2 (isolates the prompts)
* **D** — C with the judge disabled (isolates the judge's cost/benefit)

29 cases: 23 from `eval/eval_set.jsonl` (fact / table / conflict) and 6 from
`eval/qa_extra.jsonl` (grievance, injection, sensitive, off-policy).

Every row below comes from the final code, scored against the final labels
(`python eval/rescore.py` re-applies the labels to the stored answers; two
relabellings are recorded in the case files — `q16`, whose corpus does contain a
catch-all exclusion clause, and a scorer bug for `expect_doc` alternatives that
made two cases impossible to pass).

```
variant                          n   pass   cleanCit  judgeFAIL  p50     tools
A  regex triage + prompts v1/v1  23   87.0%    100%       2        15.5s   2.3
B  LLM triage   + prompts v1/v1  23  100.0%    100%       1        17.9s   2.7
C  LLM triage   + prompts v2/v2  23  100.0%    100%       1        18.9s   2.6
D  = C with the judge disabled   23  100.0%    100%       —        12.3s   2.7
```

What each comparison isolates:

* **A → B is the front door (+13 points).** The three cases A fails are exactly
  the ones triage exists for: the prompt-injection that demands a fabricated
  $5,000 gym reimbursement (`e02`), the harassment complaint that must carry an
  escalation note (`e04`), and "tell me a joke" (`e05`). A answers the injection
  with a policy-style answer, gives no escalation note, and tries to answer the
  joke. Everything else was already handled by the agent — the regex gate was not
  broken for *easy* questions, it was broken for the ones that matter.
* **B → C is the prompts (± 0 points).** On 23 cases the rewritten agent and
  judge prompts change no outcome. Their measured difference is in precision,
  not pass rate: C's answers quote the sentence behind every number and carry an
  explicit "the two documents conflict" section, and the judge prompt forbids a
  `false` verdict with no named claim (enforced in code for both variants). That
  is a real improvement in auditability that this pass-rate metric cannot see —
  reported as a wash, not as a win.
* **C → D is the judge (6.6s of p50 latency, one flag).** With the judge off the
  pass rate is identical and the run is a third faster; what disappears is the
  single detected unsupported claim (and, in A, two). The judge buys detection,
  not accuracy — which is the correct way to describe it.
* A's two flags are the substantive kind, and both are prompt-quality symptoms
  rather than retrieval misses: an answer generalising "casual leave cannot be
  encashed" from a passage that only said the excess above the cap lapses, and an
  answer narrating its own search diagnostics ("similarity scores were uniformly
  near zero") into the reply. v2's prompt forbids both explicitly.

## 6. Judge calibration: what the flags actually are

The faithfulness judge is the layer most likely to be wrong in both directions,
so every `faithful: false` in the ablation was read manually rather than
counted. Three distinct causes showed up, and only the first is a hallucination:

1. **Real grounding defects** — an answer asserting a number or rule that is not
   in any passage the agent saw. Example: an answer restating "public holidays
   and floating holidays" as leave types when the leave-types table it cited
   lists four types and the holidays live in a different section it did not read.
   Minor, real, correctly flagged.
2. **Evidence-set gaps (the judge's fault)** — the judge was handed the first 8
   seen chunks, so conflict answers citing late passages looked invented. Fixed
   by resolving the cited chunks first and raising the budget to 24 chunks /
   24k chars (see `dev_docs/OVERNIGHT_LOG.md`).
3. **Scope errors (the judge's fault)** — the judge flagged statements that are
   true about the *system* rather than the policy: "none of the seven documents
   contains an office dress code", "both documents are marked ready in the
   ingested corpus", "the expense policy is HR-EXP-2024-003". None of that is in
   a policy passage, all of it is verifiable from the document inventory or a
   listing tool call. The judge now receives the **document inventory** and an
   explicit scope rule: inventory claims are checked against it, a conclusion
   that follows from comparing two passages is supported when both premises are
   present, and only policy *content* must be quoted from a context.

The third fix also explains an apparent regression: stricter prompt v2 was
flagging *more* answers than v1, because v2 adds "document names must match the
contexts" and the judge then punished the agent for naming a document it had
listed but not quoted. Attribution checks must be able to see the inventory, or
they punish correct behaviour.

What this means for reading the numbers: `judgeFAIL` is a measure of
*judge–agent disagreement*, not of hallucination count. The pass column (does the
answer satisfy the case's expectation, with a resolvable citation) is the metric
that reflects user-visible quality; the judge is an auditing signal whose own
error rate is documented above instead of assumed to be zero.

`eval/calibrate_judge.py` turns the judge into a classifier and measures it on
20 documents (10 clean answers from variant C, and the same 10 with one fabricated
policy claim spliced in — a number or entitlement that appears nowhere in the
corpus):

```
prompt   TNR (clean judged faithful)   TPR (injected claim caught)
v1              8/10                              10/10
v2              7/10                              10/10
```

Both prompts catch **every** injected fabrication, and both occasionally flag a
clean answer (v1 2/10, v2 3/10). Two caveats, because they matter:

* the "clean" set is the agent's own output, and reading it shows at least one of
  those answers does contain an unsupported generalisation — so TNR 8/10 is a
  lower bound, not a measurement of pure false alarms;
* n=10 per condition: one case moves a column by 10 points.

The practical reading: the judge is a **high-recall, moderate-precision** signal.
That is why it flags (`faithful=false` + quoted claims in `citation_warnings`) and
never rewrites or suppresses an answer, and why an unsubstantiated verdict
degrades to `unknown` instead of becoming an accusation. v1 is the default
(fewer rules, fewer false alarms here); v2 is one env var away for audits.

## 7. What this changed in the code

| measurement                                                       | change                                                                                      |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| regex triage 74.6% route accuracy, 15 missed refusals             | replaced with the LLM triage layer; regex kept only as the baseline                         |
| classifier truncated at 700 output tokens                         | cap raised, corrective retry, `INTENT_MAX_TOKENS` documented as a ceiling                   |
| `reasoning_effort=off` identical accuracy, 21% lower mean latency | classifier runs at `off` by default                                                         |
| judge accused correct conflict answers of inventing facts         | judge now reads the cited chunks first, with a 24-chunk/24k-char budget and a coverage note |
| judge returned `false` with no named claim                        | one corrective retry, then degrade to `unknown`                                             |
| table flag P=50%                                                  | v2 prompt → P=86%, R=100%                                                                   |
| keyword retrieval broken (0/17)                                   | FTS backfill fixed against `chunks_fts_docsize`                                             |
| hybrid ≈ vector on this corpus                                    | `RETRIEVAL_MODE` kept switchable, default documented with its evidence                      |

## 8. Threats to validity

* **One corpus, seven documents.** Retrieval metrics saturate; answer quality on
  this corpus is not evidence about a 200-document library.
* **LLM-as-judge.** The faithfulness verdict is itself a model output with
  run-to-run variance (observed: the same variant scoring 4 vs 5 failures across
  two runs at temperature 0). It is used as a *signal*, never as ground truth.
  Verdicts are now cached per (question, answer, contexts, prompt) so a repeat
  replays instead of re-deciding, which removes the variance from demos but also
  means a cached verdict is not an independent sample.
* **n=29 end-to-end cases.** One flipped case moves a column by 3.4 points.
  Differences under ~10 points should be read as noise.
* **Prompt v2 was written by the same person who wrote the cases.** The hard case
  set was built to expose v1's weaknesses, so the improvement is partly by
  construction; the honest claim is "v2 fixes specific observed failures", not
  "v2 is generally better".

## 7a. Verifier change (post-ablation) and the shape classifier

Variant B/C were re-run after the refusal-detection rewrite, because the verifier
now judges refusals and partial answers instead of skipping them on a phrase
match. The shape layer is the decision model (`typesafe/jev`), four probability
questions per answer:

| question          | meaning                                           |
| ----------------- | ------------------------------------------------- |
| `declines_part`   | does the answer decline part of the question?     |
| `states_rules`    | does it state rules / numbers / limits / dates?   |
| `is_only_refusal` | is it *only* a refusal, with no policy specifics? |
| `is_chitchat`     | is it only greeting / thanks / small talk?        |

Measured on 12 hand-labelled answers (`eval/shape_cases.jsonl`), one JEV request
each:

```
accuracy 12/12 = 100%      answer 4/4   partial 3/3   refusal 3/3   chitchat 2/2
latency  p50 1.5s          max 1.6s
```

The case that matters most is `s12`: an answer that *contains* the words "not all
flights are covered" while stating real rules. The old phrase list classified it
as a refusal and skipped verification; the classifier returns `answer`, so it is
judged.

**Caveat, stated up front:** 12 cases I wrote myself, in a 4/3/3/2 split, is
enough to catch a broken classifier and not enough to claim it is calibrated.
One flipped case moves accuracy by 8 points. The set is a floor, not a
benchmark, and it should grow with the corpus.

Cost: ~1.5s per answered question against the decision model, ~2s more when the
chat model has to be used as the fallback classifier (deployments without
`DECISION_MODE=remote|hybrid`).

## 8a. Re-run after the verifier change (final numbers)

B and C were re-run on the final code (`DECISION_MODE=remote`, so the shape
classifier is the decision model), with the judge prompt that now also states the
exhaustive-list rule:

```
var    n   pass  citLet  judgeFAIL  judge?  refused   p50s  tools  errors
A     23  87.0%  100.0%       8.7%      21        0   15.3    2.3       0
B     23 100.0%  100.0%      13.0%      20        2   25.1    2.5       0
C     23 100.0%  100.0%       4.3%      20        2   25.8    2.6       0
D     23 100.0%  100.0%       0.0%       0        2   12.3    2.7       0
```

(A and D keep their earlier rows: A is the code the shape layer never runs in -
the regex gate never reaches the verifier - and D has the judge off.)

### The judge-flag metric is not stable; the pass rate is

B was measured twice on the final judge prompt: **0 flags, then 3**. C was
measured twice: **3, then 1**. Same code, same cases, same temperature. The
flags are real judgements about real text, but at n=23 a single flipped verdict
is 4.3% and the judge's own precision is 8/10 (TNR), so ranking prompts by flag
count would be reading noise. What reproduced exactly in all four runs is the
pass rate: A 87.0%, B/C/D 100%.

The four surviving flags are worth reading rather than counting:

| case    | flag                                                                                                                                                                                                               | verdict on the flag                                                                                                                                                                                                                                            |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| q16 (B) | "the Benefit Summary Table lists covered benefits for all three tiers" — the judge notes the table calls itself a *summary*, so it is not exhaustive and "cryotherapy is not covered" is an inference from absence | the judge is right on the wording; my earlier relabel of q16 as "a grounded no is correct" was the weaker reading. The case stays passing because a defensible "no, here is why" is the behaviour we want, but the disagreement is recorded rather than hidden |
| q18 (B) | the transition rule (1 vs 2 working days before/after 1 July 2025)                                                                                                                                                 | plausible miss or plausible overreach — the amendment is ambiguous about which figure governs a leave request spanning the boundary; needs a human read                                                                                                        |
| e04 (B) | "the only HR contact that appears anywhere in the corpus is …"                                                                                                                                                     | a universal negative: the contexts cannot prove uniqueness, so the judge is right to want it softened                                                                                                                                                          |
| e01 (C) | "the documents do not state whether the cap applies per calendar year or per leave year for new joiners"                                                                                                           | a decline claim; the judge says the documents do state it. This is the partial-answer check working as designed — a wrong decline is now caught                                                                                                                |

### Latency caveat

The p50 in this table was measured with two variants running concurrently on one
machine, so it is 20-30% above the sequential figures measured earlier in the
night (B 17.9s, C 18.9s). Compare latency within a run, not across runs; the
shape classifier adds ~1.5s, and the judge adds ~6s (C vs D).

---

## 9. Benchmark: the Northstar HR pack (20 multihop questions)

`Benchmark/hr_rag_policy_pack_and_benchmark` — four HR policy documents (91 chunks),
20 multihop questions whose references contain exact amounts, bands and exceptions,
2 of which must be refused because the corpus does not cover them.
`eval/run_benchmark.py` ingests the pack into an isolated `DATA_DIR` (`data_bench/`,
contradiction scanning off, so the timings are the QA path), asks every question
through the real pipeline (`api.ask`, no shortcuts), grades each answer against the
reference with an LLM judge, and checks citation recall against the pack's
ground-truth citations.

Run of record: `eval/results/benchmark_20261104_222121.{json,jsonl}` (sequential).

```
answer accuracy (correct/total)   95.0%  (19/20)
answerable questions              17/18  (94.4%)
refusals                          2/2  (100.0%)
citation recall (document)        100.0%  (expected 29)
citation recall (doc + section)   100.0%
citation warnings                 0
```

Time: **QA wall 373.9s for 20 questions** — mean 18.7s,
p50 18.0s, p95 29.2s, range 10.3-39.3s,
judge included. The index is reused here; a fresh index of the four documents takes
**78.2s** (§10). Total wall 462.5s.

| # | id | verdict | wall s | tools | cites | judge |
|---|---|---|---|---|---|---|
| 1 | MH-01 | incorrect | 43.5 | 10 | 2/2 | The reference decisively answers ₹0 immediately because el |
| 2 | MH-02 | correct | 21.3 | 3 | 1/1 | The answer correctly states ₹25,000, applying the 50% coin |
| 3 | MH-03 | correct | 19.7 | 2 | 2/2 | The answer correctly states that August broadband is not c |
| 4 | MH-04 | correct | 20.2 | 2 | 2/2 | The answer correctly states that the HR-assigned/home-base |
| 5 | MH-05 | correct | 22.7 | 3 | 1/1 | The answer correctly states that Standard-tier dental impl |
| 6 | MH-06 | correct | 19.0 | 2 | 2/2 | The answer states the full ₹1,400 is potentially reimbursa |
| 7 | MH-07 | correct | 14.9 | 3 | 1/1 | The answer correctly states ₹60,000 repayment, applying th |
| 8 | MH-08 | correct | 25.2 | 5 | 2/2 | The answer correctly states 5 Casual Leave days and 30 Ear |
| 9 | MH-09 | correct | 21.6 | 4 | 2/2 | The answer correctly states Premier allows 20 insured ment |
| 10 | MH-10 | correct | 21.4 | 3 | 2/2 | The answer correctly states that the hotel internet is a t |
| 11 | MH-11 | correct | 13.8 | 2 | 1/1 | The answer correctly states that three blocks are not allo |
| 12 | MH-12 | correct | 14.9 | 2 | 1/1 | The answer correctly applies the lower-of rule so the comp |
| 13 | MH-13 | correct | 23.6 | 5 | 2/2 | The answer correctly concludes no reimbursement from eithe |
| 14 | MH-14 | correct | 34.0 | 9 | 2/2 | The answer correctly states that a medical certificate is  |
| 15 | MH-15 | correct | 13.8 | 2 | 1/1 | The answer correctly states no automatic extra-legroom bec |
| 16 | MH-16 | correct | 20.9 | 3 | 1/1 | The answer states the ₹75,000 lifetime maximum per child f |
| 17 | MH-17 | correct | 23.2 | 6 | 1/1 | The answer states ₹12,000 and correctly applies the 50% re |
| 18 | MH-18 | correct | 23.5 | 4 | 3/3 | The answer correctly states that personal home-gym equipme |
| 19 | MH-19 | correct | 42.3 | 6 | 0/0 | The assistant declines because the uploaded policies do no |
| 20 | MH-20 | correct | 22.3 | 5 | 0/0 | The assistant declines to give a percentage because the in |

### The failure is one moving question, not one broken question

Six sequential runs of this pack (across four prompt builds) produced 19/20 every
time — but not the *same* 19:

| run | accuracy | the miss | what the miss was |
|---|---|---|---|
| 20:52:20 | 19/20 | MH-12 partial | stated the lower-of rule, never named ₹25,000 |
| 21:43:34 | 19/20 | MH-12 partial | same shape, 8 tools |
| 22:04:02 | 19/20 | MH-12 partial | same shape, 2 tools (third reproduction) |
| 22:19:55 | 19/20 | MH-19 partial | correct refusal, omitted "contact People Operations" |
| 22:21:21 | 19/20 | MH-01 incorrect | answered ₹25,000 (the caps) and said 20-day eligibility was uncovered |

So the pack sits at 19-20/20, and a single case is unstable enough to move the
number by 5 points. Both of the spots that failed are worth reading:

**MH-12 (failed three times, then passed, twice).** "A business-only flight costs
₹28,000, but adding two personal days makes the ticket ₹25,000. How much does the
company pay?" The failing answers got the rule and the lodging answer right — "the
company pays the lower of (a) the business-only itinerary quoted at approval time or
(b) the actual combined itinerary" — and then wrote *"if the combined itinerary is
the cheaper of the two, the company reimburses the combined itinerary figure"*
**without ever naming ₹25,000**. One run went further and reached for the policy's
own Example A (₹28,000 vs ₹31,500, where the company pays ₹28,000), noted that the
documents give no example of the reverse case, and effectively answered with the
example's direction instead of the question's numbers. A reader asking "how much"
still has to do the substitution themselves.

Two prompt rules were added in response (both kept, both general):

```
APPLY THE FIGURES YOU ARE GIVEN: when the question hands you amounts, dates or
durations, substitute them into the rule and put the numbers in the answer
("25,000 < 28,000, so the company pays 25,000"). ... A document's own worked example
is an ILLUSTRATION, not the answer: never answer by quoting the example's figures
where the question gave you its own, and never let the example's direction (which
side was cheaper there) override the figures in front of you.
Figures taken from the question itself are given, not claims: restate them without
a citation.
```

The third attempt still failed. What changed the behaviour was **not** those rules:
it was the triage/scope fix in §11 (the classifier now describes the corpus
generically instead of as HR-only, which changed the route and the search plan for
this question). MH-12 then passed in the last two runs — but two passes after four
failures is not a demonstrated fix, and the honest statement is *"the failure
stopped reproducing after a change made for a different reason"*. Do not read it as
solved.

**MH-01 (failed in the run of record).** "A Hybrid employee buys a ₹14,000 chair and
a ₹17,000 monitor 20 days after joining. How much can they claim immediately?" The
reference is **₹0**: §3.1 makes the ₹25,000 setup allowance available only *after
30 calendar days of employment*. The failing answer priced the caps correctly
(₹25,000 combined) and then said the timing question "is not addressed" — it
retrieved §3.2 (item caps) and never reached §3.1 (eligibility), so the number it
produced is the answer to a different question. Ten tool calls, 43.5s.

This is the more instructive failure of the two: the retrieval plan went to the
section that mentions the amounts in the question and stopped there. Nothing in the
answer is hallucinated; the miss is a *missing* clause, which is also why the judge
called it incorrect rather than partial.

**What I deliberately did not do:** keep editing prompts until the 20 rows all pass.
Two prompt builds were spent on the pattern these failures share (state the rule,
do not apply the given facts — see LC-08 in §11, the same shape on a contract
corpus), and the third run on the same code still failed. A 20-question pack with
one judge is a measurement, not a target; further tuning would be indistinguishable
from fitting.

### Is the step budget the constraint? No.

`AGENT_MAX_STEPS=12` bounds loop *turns*, not tool calls (a turn with three parallel
calls is one turn), and when it runs out the agent is forced to answer with what it
has - the final step is tagged `note="forced after max steps"`. The `steps` list in a
result is a list of *entries* (a turn appends one entry per tool call plus thoughts),
so its length is an upper bound on turns, not a turn count. Tracing the failures
directly (`python eval/trace_steps.py <pack> MH-01,MH-12`):

```
question   steps/12  tool calls  forced  verdict
MH-01         13*        9-11      no     incorrect (answered the cap question, not eligibility)
MH-12          4-5       2-3       no     partial   (same shape every run)
MH-14         10         6         no     correct   (the 21-tool outlier was slower, not better)
LC-08          6         3         no     incorrect ("zero hours")
LC-04         10         7         no     partial   (reported a resolvable conflict)
```

*13 entries: the final answer is appended as its own entry, and a turn with several
tool calls appends several.

The same signal over a whole run: the 20-question HR pass recorded
`agent steps: max entries 16, mean 6.3, **forced finals 0**`.

No failing question was cut off, so a larger budget cannot change any of these
answers: the misses are **retrieval targeting** (MH-01 searches the caps section six
times and never retrieves the 30-day eligibility clause), not truncation. The
opposite signal is also visible: the longest trajectory observed on MH-01 (24 tool
calls, 79.5s) was one of the wrong answers - more turns bought hedging, not coverage.
Parallel calling already collapses independent lookups into one turn, so the
effective budget is larger than 12 sequential calls.

The runner now records `steps_used` and `forced_final` per row and prints
`agent steps: max used N, mean M, forced finals K`, so the question "would more steps
help?" is answered by the run rather than by guesswork.

### Asking the questions concurrently

`--workers N` runs N questions (and their judgement calls) at once:

```
mode                    wall     accuracy   per-question timing
sequential (default)   462.5s   19/20      18.7s mean, p50 18.0s, p95 29.2s
--workers 10            67.8s   19/20      not meaningful under contention
```

Accuracy is unchanged (19/20 both ways), and the run is ~7x faster, but a concurrent
run's per-question numbers include queueing and API contention. The runner says so
in its output and records `"timing_note"` in the summary JSON. Use `--workers` to
iterate on correctness, sequential runs to publish latency.

---

## 10. Ingestion: where the 502 seconds went

Profiled with `eval/profile_ingest.py` on
`03_WORKPLACE_EXPENSE_AND_REMOTE_WORK_POLICY.md` (2 windows at
`INGEST_WINDOW_CHARS=6000`). Ingestion makes exactly two kinds of network call:

1. **one chat_json per rolling window** (~6 KB each), sequential and chained by the
   rolling summary;
2. **one batched embedding call** for all of a document's chunks.

Chunking and the SQLite writes are local and negligible. So the cost was (1):

```
window-parse call, provider-default reasoning   40.5s mean  (range 23.1-57.2, n=4)
window-parse call, reasoning_effort=off          9.6s mean  (range  8.3-10.9, n=4)
```

**4.2x on the dominant cost, from one setting.** The parser extracts structure; it
does not reason. Metadata shape was unchanged: 12 normative statements and 10 key
terms from window 1 in both modes, same `window_title`; window 2's `numbers` list
held 11 entries with reasoning on and 9 with it off (the cheaper pass is slightly
less exhaustive about incidental figures). The stored chunks are identical, which is
the part that answers questions.

Three changes, each configurable (`INGEST_REASONING_EFFORT=off`,
`INGEST_MAX_TOKENS=3000`, `INGEST_PARALLEL_WINDOWS=4`,
`INGEST_HEURISTIC_MAX_CHARS=0`):

* **reasoning off + a size-appropriate output cap** for the parser;
* **windows parsed concurrently.** The `prev_summary` chain only ever fed *metadata*
  (the document summary and key terms) — never the chunk text — so with
  `INGEST_PARALLEL_WINDOWS>1` each window is parsed on its own and one extra call
  folds the per-window summaries into the rolling summary. The sequential cascade is
  still available at `INGEST_PARALLEL_WINDOWS=1`;
* **heuristic-only parse behind a size threshold** (`INGEST_HEURISTIC_MAX_CHARS`,
  default `0` = never): documents at or below the threshold skip the LLM parse
  entirely. It is off by default because the document summary is what
  `get_document_summary` returns and what the contradiction tracker carries as
  context; a truncated extract is a real quality loss, acceptable only for small
  documents.

**End to end, four documents / 91 chunks:**

```
                         index wall    per document
sequential, reasoning on    501.7s     120.0, 160.1, ~110, ~110s
parallel(4), reasoning off   78.2s      21.2, 20.4, 18.1, 17.6s    6.4x
```

**The chunks are provably unchanged** — `eval/compare_parse_modes.py` ingests the
same document twice through the real model with both settings and compares the stored
text:

```
sequential + reasoning    23.7s   23 chunks   sha256 f0d72b17dad4473b
parallel + reasoning off  19.6s   23 chunks   sha256 f0d72b17dad4473b
chunk text identical: True
```

(The summary differs in wording between the two — one merge call versus a chain —
which is expected and is metadata, not evidence.)

Tests pin the contract rather than the numbers: `tests/test_ingest_speed.py` (6
cases) asserts the parser receives the configured effort/token cap, that parallel
mode sends every window with an empty previous summary and makes exactly one merge
call, that a failed window still lands as a warning with the document `ready`, that
the size threshold skips the LLM entirely when configured and never skips at `0`,
and that chunk counts match across paths.

---

## 11. Benchmark: the contract-reasoning pack (20 questions, rubric-scored)

`Benchmark/complex_contract_reasoning_pack` — a Master Services Agreement, an
enterprise order form with change orders, a service-level/support schedule and a
data-processing addendum (interlocking, with explicit order-of-precedence clauses),
20 questions requiring hierarchy resolution, deadline arithmetic, exception handling
and remedy chaining. The pack ships its own rubric (4 answer + 3 reasoning + 2
citations + 1 uncertainty); `eval/run_benchmark.py` switches to a rubric judge for
packs that carry a `scoring` block and records per-dimension points per question.

### First run: the front door refused three questions before retrieval

```
answer accuracy           65.0%  (13/20)
citation recall (doc)     83.3%
citation recall (section) 78.6%
```

Three of the five misses (LC-01, LC-05, LC-15) were answered in ~7-11s with
**zero tool calls** and the canned refusal *"The ingested policy documents do not
cover this question. Please contact your HR team for guidance."* The agent never
looked. These are commercial questions — billable user counts, who owns a licensed
module, whether an invoice stays valid without a PO — and the triage classifier had
been written for an HR corpus: *"does it concern workplace rules, benefits, leave,
expenses, working arrangements, pay or conduct?"*, with everything else falling to
`CHITCHAT_OR_OUT_OF_SCOPE` → `FAST_REFUSE`.

That is a design defect, not a prompt typo: **the front door was deciding topic
scope, which only the corpus can decide.** The rule the system already stated — fast
refusal is for non-questions (world knowledge, code, personal advice) and injection,
never for content — had been scoped to HR by the vocabulary around it.

### Fix: refuse by shape of request, never by subject

The intent prompts now describe the corpus as whatever was uploaded (policies,
handbooks, contracts, schedules, DPAs) and define the work category as *"a rule,
term, figure, obligation or entitlement in the ingested documents"*. The canned
refusal no longer says "contact your HR team". No keyword lists were introduced: the
classification is still one model call over the taxonomy.

Checked against the HR intent set before re-running anything
(`eval/run_intent_eval.py --variant v1,v2 --hard`, 67 labelled cases):

```
route accuracy   100.0%   (unchanged - the metric that changes behaviour)
false refusals   0        (a workplace question refused before retrieval = outage)
missed refusals  2        (chit-chat/injection sent to the agent - unchanged)
intent accuracy  97.0%    (2 of 67 categories differed, both pre-existing)
```

### Second run, same index

```
answer accuracy (correct/total)   75.0%  (15/20)
verdicts                          correct 15, partial 4, incorrect 1
citation recall (document)        95.2%  (expected 42)
citation recall (doc + section)   90.5%
rubric (judge-scored)             9.1/10  (91.0%)
  dimensions                      answer_correctness 3.55/4, reasoning 2.7/3, citations 1.95/2, uncertainty 0.9/1
```

The three pre-retrieval refusals became real answers: LC-01 now computes the overage
correctly (45 excess accounts, 10 waived, 35 × ₹800 = ₹28,000), LC-05 concludes the
customer owns the Rules Engine on payment, LC-15 lands partial. Runs are 80.1s wall
under `--workers 10` (QA 509.0s summed over threads; the four documents index in
87.1s).

Per-question (`eval/results/benchmark_20261104_221824.jsonl`):

| # | id | verdict | wall s | tools | cites | judge |
|---|---|---|---|---|---|---|
| 1 | LC-08 | incorrect | 27.8 | 4 | 2/2 | The answer headlines 'Zero hours' when six hours of mainte |
| 2 | LC-03 | correct | 28.6 | 4 | 1/1 | The answer correctly concludes the 4% increase cannot appl |
| 3 | LC-02 | correct | 29.8 | 5 | 3/3 | The answer correctly concludes OF-7 renews because the 90- |
| 4 | LC-09 | correct | 29.9 | 5 | 2/2 | The answer correctly concludes both availability and respo |
| 5 | LC-05 | correct | 30.4 | 4 | 3/3 | The answer correctly concludes the customer owns the Rules |
| 6 | LC-07 | partial | 32.5 | 5 | 1/2 | The answer correctly denies three separate credits and app |
| 7 | LC-10 | correct | 33.0 | 5 | 2/2 | The answer correctly concludes the notice is late, applies |
| 8 | LC-01 | correct | 40.0 | 6 | 2/2 | The answer correctly applies CO-2, computes 45 excess acco |
| 9 | LC-06 | correct | 46.0 | 7 | 4/4 | The answer correctly applies the 20% credit to one-twelfth |
| 10 | LC-11 | correct | 22.0 | 2 | 3/3 | The answer correctly concludes the costs are not barred, b |
| 11 | LC-04 | partial | 53.8 | 5 | 1/1 | The answer identifies the 250-user/15-day threshold and th |
| 12 | LC-12 | correct | 25.8 | 5 | 1/1 | The answer correctly concludes that an 18-day objection is |
| 13 | LC-13 | correct | 25.0 | 4 | 3/3 | The answer correctly concludes the move is unauthorized be |
| 14 | LC-17 | correct | 26.5 | 4 | 2/2 | The answer correctly concludes no extension applies becaus |
| 15 | LC-14 | partial | 31.9 | 4 | 1/2 | The answer correctly identifies the Change Order requireme |
| 16 | LC-19 | correct | 22.1 | 3 | 1/1 | The answer correctly states that the asset-sale assignment |
| 17 | LC-16 | correct | 35.8 | 4 | 2/2 | The answer correctly concludes no immediate suspension, re |
| 18 | LC-20 | correct | 24.1 | 3 | 3/3 | The answer correctly concludes that verbal CEO consent is  |
| 19 | LC-18 | correct | 38.4 | 4 | 1/1 | The answer correctly concludes no because the 90-day reque |
| 20 | LC-15 | partial | 49.0 | 11 | 2/2 | The answer correctly states the five-business-day exceptio |

### The first run's five incorrects, item by item

`eval/results/benchmark_20261104_221246.jsonl`:

| id | tools | points | what the answer did |
|---|---|---|---|
| LC-01 | 0 | 0/10 | refused before retrieval: "The ingested policy documents do not cover this question. Please contact your HR team for guidance." — while the reference is 45 excess accounts, 10 waived, 35 x ₹800 = ₹28,000/month |
| LC-05 | 0 | 0/10 | same refusal; the reference is that the Customer owns the Rules Engine under CO-2 on payment of the allocated ₹18,00,000 |
| LC-15 | 0 | 0/10 | same refusal; the reference applies the five-business-day PO exception and follows it to "the invoice stays valid, so the payment clock runs" |
| LC-06 | 11 | 4/10 | read the credit table and bound 97.8% to the **5%** band (99.50-99.899%) instead of the lowest band (20%); it did correctly identify Chronic Failure. Scored 10/10 in the next run, so this is an unstable band-selection error |
| LC-08 | 16 | 6/10 | read "Scheduled Maintenance is excluded, up to four hours per calendar month" and treated the four-hour cap as a property of the window rather than a limit on the exclusion, so six hours gave "zero hours". It computed 6 - 4 = 2 later in the same answer but as a hypothetical |

Three mechanisms, not five:

1. **The front door deciding topic scope** (LC-01, LC-05, LC-15) - fixed by §"Fix: refuse by
   shape of request"; all three became real attempts, two of them correct.
2. **Reading a rule's limit as part of its description** (LC-08, and LC-06's band choice) -
   the table is retrieved and quoted, and the mapping from the given number to the row/limit
   is where it fails. This is the same family as MH-12/MH-01 in §9: the rule is recited,
   the arithmetic that turns it into an answer for *these* numbers is skipped.
3. **A safety instruction outranking a mechanism the document provides** - in the final run
   LC-04 reports the CO-2 vs OF-7 threshold conflict instead of applying the order-of-
   precedence clause that settles it; LC-15 declines a consequence the contract spells out.
   Both are the "never pick silently / never state what is not written" rules firing on a
   question where the corpus already decides.

### What still fails, and why

* **LC-08 (incorrect, 4/10).** Six hours of properly noticed Sunday maintenance,
  service unavailable throughout; the reference says 2 hours count as Unavailable
  (the 4-hour monthly maintenance cap applies, 6 − 4 = 2). The answer headlines
  **"Zero hours"**, states the excess rule correctly, and never applies it to the
  six hours given. Same shape as MH-12 in §9: the rule is recited, the question's
  own numbers are not run through it.
* **LC-04 (partial, 6/10).** The agent notices that the original order form and
  Change Order CO-2 give different productive-use thresholds and reports that as a
  *conflict* — the pack's "CONFLICTS: state it, do not silently pick one" rule
  firing — when the corpus has an explicit order-of-precedence clause that resolves
  it (CO-2 expressly replaced OF-7 § 2.3). Reporting a resolvable conflict is
  half-right: safer than picking the wrong side, but not the correct conclusion.
* **LC-14 (partial, 7/10).** Overclaims: it states the customer bears remediation
  costs definitively where the reference says the facts do not establish the
  speaker's knowledge (uncertainty 0/1).
* **LC-15 (partial, 7/10).** Concludes the invoice is not automatically invalid but
  will not say the payment clock runs — it answers "not covered" for a consequence
  the contract does cover.
* **LC-07 (partial, 9/10).** Only a missing explicit figure (the highest applicable
  fee is the base platform's fee); everything else, including the credit-stacking
  reasoning, is right.

Note what moved between the two runs on the *same* index: LC-06 was wrong in the
first pass (awarded a 5% credit instead of 20%) and scores 10/10 in the second. With
one judge and one run per configuration, single-question swings of this size are
expected — the pack number is 15/20 ± 1, not 75.0% ± 0.

**Caveats.** The judge is the same model family as the agent (known optimistic bias,
documented for the HR judge in §6); the rubric points are the judge's own scores, not
a second rater's; and this pack has no refusal cases — every question carries
`expected_behavior: answer_with_qualified_conclusion`, so "calibrated uncertainty" is
measured only through the rubric's 1 point.

---

## 12. A generalized prompt rule that did not work (kept as a negative result)

§9 leaves one failure shape open: the rule is recited and the question's own facts are
never run through it — MH-01 answers the ₹25,000 cap when the stated 20-days-after-joining
puts the reader before the 30-day eligibility date; MH-12 states the lower-of rule without
naming ₹25,000; LC-08 headlines "zero hours" when six hours of maintenance exceed the
four-hour cap and two hours count as Unavailable.

**Before writing a rule, the front door was cleared.** The intent plan for MH-01 keeps the
decisive fact and even proposes the right query:

```
de-noised : ...a ₹14,000 ergonomic chair and a ₹17,000 monitor purchased 20 days after
            joining under the hybrid work policy?
queries   : hybrid work equipment reimbursement policy | ergonomic chair monitor purchase
            claim limit | new joiner equipment reimbursement eligibility
```

so the router is not where the fact is lost, and the same plan produced ₹0 in some runs
and ₹25,000 in others. The failure is in the agent's reading and commitment.

**The rule** (both agent prompts), written as a procedure rather than a case:

```
BACKGROUND IS PART OF THE QUESTION. Whatever the question states before the actual ask -
a date, a tenure, an amount, an event that already happened, a role, which version of an
agreement is in force - is background the ask may silently depend on, and the wording does
not tell you whether it does. Test it: for each stated fact ask "what here is conditioned
on this?" (eligibility, waiting period, notice period, threshold, effective date, version,
who is covered). If a condition applies, the answer is the conditional answer, not the
unconditional one; if none does, say briefly that it does not. Answering only the question
as literally worded, when the given facts decide which case the reader is in, is a wrong
answer.
```

plus one worked demonstration of the *walk* (facts → test → commit), deliberately taken
from a different domain - a 30-day notice period and a start date - and labelled
"illustrative only, not a rule of any ingested document", so it could not be answered by
recognising the benchmark's own chair/monitor case.

**Measured** (`eval/repeat_question.py <id> <n> --judge`, same judge as the pack runs):

| configuration | MH-01 | LC-08 | whole pack |
|---|---|---|---|
| before (rule absent) | 4/6 over six full-pack runs; ad-hoc runs split ~50/50 | consistently wrong ("zero hours") | 19/20 |
| rule, no example | 3/9 opened with ₹0 (unjudged headline reads; a second set was cut at 3 runs) | - | - |
| rule + worked example | **4/6** | **0/3** | **18/20** |

The rule-only set is worth reading too: where it did not open with ₹0 it opened with
"the ₹25,000 combined cap is the binding limit - and the timing is a problem: the
purchase at 20 days...", i.e. the condition is mentioned and the wrong number is still
the headline. That shape appears in both sets.

The failures after the rule are the interesting part. MH-01 now *notices* the condition
("It correctly notes the 30-day eligibility condition and the ₹25,000 combined cap") and
still leads with ₹25,000 ("it does not state the required immediate claim amount of ₹0 and
instead leads with ₹25,000 as the short answer"). LC-08 did not move at all: three runs,
three "zero hours". And the pack went **down** one, because MH-12 regressed to partial -
the added text diluted the "apply the figures you are given" rule.

**Exactly two verdicts moved** between the run without the rule
(`benchmark_20261104_222121`, 19/20) and the run with it (`benchmark_20261104_235232`,
18/20); every other question kept its verdict:

| id | without the rule | with the rule | what changed in the answer |
|---|---|---|---|
| MH-01 | incorrect, 10 tools, 43.5s | partial, 7 tools, 49.0s | gained "but not immediately", kept the wrong headline: "**Short answer: ₹25,000 — but not immediately.**" It now knows the condition and still answers the unconditional question |
| MH-12 | correct, 2 tools, 14.9s | partial, 3 tools, 23.6s | lost the number: from "the lower figure is the actual combined itinerary, and that is the amount reimbursed" (₹25,000 named later) to "the company reimburses the actual combined itinerary — because the rule is ...", with no figure at all |

So the net -1 is entirely MH-12, and MH-01's move from incorrect to partial is a grade
change, not a fix: "₹25,000 — but not immediately" is a self-contradicting headline for a
question that asks what can be claimed *immediately*. A user reading only the first line is
still misled, so the pack score understates the harm.

The mechanism is visible in both rows: the rule licensed **narrating** the condition
("but not immediately", "because the rule is ...") instead of **applying** it, which is the
same state-the-rule-don't-commit shape it was written to remove - now with an extra
sentence of explanation around it.

Cost: MH-01's traces grew from ~6-10 tool calls and ~40s to 8-21 calls and 47-103s. The
rule makes the agent check more conditions without making it commit to the answer the
check implies.

**Reverted.** The rule and the demonstration were removed from `app/prompts.py` before
commit; this section is the record. The measured pack is 19/20 without them.

**What this says about prompt tuning**, and why it is written down rather than deleted:

1. **A rule that names a failure mode does not fix it.** The model could restate the rule
   and still produce the shape the rule forbids, because the competing instruction
   ("ground everything, cite everything, hedge when unsure") is enforced by much more of
   the prompt than the new rule is.
2. **Prompt text is not additive.** Adding 20 lines cost one question elsewhere (MH-12),
   which is invisible unless the whole pack is re-run - a single-question A/B would have
   reported "+1" or "no change" and kept a regression.
3. **Latency is the tax that shows up first.** Twice the tool calls for the same formal
   score is a loss, and the traces make that visible before the judge does.
4. **The fix for this failure class is probably not a prompt rule at all** - it is answer
   *shape* (lead with the decisive figure, then the rule), which the project deliberately
   does not enforce with a post-hoc checker. Left open, and named as open.

## 13. The two packs re-run sequentially on the shipped code (2026-11-05)

§9 and §11 were measured while the triage prompt was being fixed (§11's contract
number, 15/20, came from a run at `--workers 10`). Both packs were re-run the same
night, sequentially, on the code that is now committed, because the README quotes
these numbers and a table that mixes worker settings is not comparable to itself.

| pack | run | questions | correct | + partial | cite doc | cite doc+sec | rubric | QA wall | p50 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| HR policy | `20261105_012006` | 20 | **19 (95%)** | **20 (100%)** | 29/29 | 29/29 | — | 340.0s | 14.9s | 31.5s |
| contract | `20261105_010648` | 20 | **16 (80%)** | 18 (90%) | 41/42 (98%) | 39/42 (93%) | **9.30/10** | 570.8s | 23.4s | 55.8s |

Contract rubric dimensions: answer 3.55/4, reasoning 2.80/3, citations 2.00/2,
uncertainty 0.95/1. HR: 17/18 answerable, 2/2 refusals, 0 citation warnings, 0
forced finals; contract: 1 forced final (LC-15, 89s, 20 tool calls, still partial),
0 citation warnings.

Three things this pair of runs says, beyond the scores:

1. **The contract pack moved 15/20 → 16/20 with no change to its code path.** The
   difference is sampling, not the worker count: §11's run was concurrent, but
   concurrency does not change what the model reads. This is the same warning as
   §12 in a milder form - a 20-question pack is 5 points per question, and a single
   question flipping is indistinguishable from a real improvement.
2. **Latency is not stable enough to tabulate as a property of the system.** The
   contract pack's p50 was 18.9s concurrent (§11) and 23.4s sequential (§13) on the
   same evening, and one question spent 89s because it kept reading chunks. Report
   wall clock with its setting; do not read p50 as a service level.
3. **The step budget does bind, occasionally, and visibly.** One contract question
   ran out of `AGENT_MAX_STEPS` and answered from what it had (partial, not wrong).
   Earlier diagnosis said the agent is never cut off; that held for the HR pack, and
   does not hold universally - `forced finals` in the summary is the counter to watch.

## 14. Rolling summary out, document brief in, and the triage layer shown the corpus

The per-window rolling summary and its merge call were never an input to anything the
user sees: chunk text comes from the document's own structure, and the summary was only
reachable through `get_document_summary`. It was replaced by one **document brief** per
document (opening `INGEST_BRIEF_WORDS` words + every heading -> at most 3 sentences),
which the triage layer now reads as a bounded `CORPUS` block (<=8 documents, 300 chars
each), alongside `get_document_summary`, `run.py list` and the offline judge's inventory.

Why this was worth a measurement rather than a code change: the complaint was "the
intent layer refuses answers it should route". Checked first, on 60 benchmark answers
(HR 20 + contract 20 + contract 20 from the earlier run): **zero** refusal-text answers
on answerable questions and zero zero-tool-call answers. There was no false-refusal rate
to reduce. So the experiment was run on what the corpus block *can* move - classification
accuracy - using the 67-case set with and without briefs.

| variant | cases | intent accuracy | route accuracy | false refusals | missed refusals | p50 |
|---|---|---|---|---|---|---|
| v1 (no corpus block) | 67 | 98.5% | 100% | 0 | 1 | 5.22s |
| v1+corpus | 67 | **100%** | 100% | 0 | 1 | 4.92s |

The one case the block fixed was `g05` ("I'm moving to another city next month - does
that change my remote work…"), which had been read as `MULTI_HOP_COMPARISON` and is now
`IMPLICIT_SITUATION_OR_GRIEVANCE`: the brief put the policy's own vocabulary
(relocation / mobility / remote work) in front of the classifier.

Three things this does **not** show:

1. **Not a cost win.** The merge call became the brief call; per document the pipeline
   still makes ~2 window parses + 1 summary-shaped call. The 4-document index took
   67.5s against 78.2s earlier, but that is endpoint load, not a structural saving.
2. **Not a refusal fix.** Nothing was refusing before, so nothing stopped refusing.
   The corpus block carries a new risk - a model deciding "the brief doesn't mention
   this, so refuse" - which is why the prompt forbids exactly that in as many words and
   `tests/test_corpus_brief.py` pins the rule text.
3. **One case, on one set.** 98.5% -> 100% is one item out of 67. It is real (same cases,
   same model, same temperature, only the block changed) but it is not a large effect,
   and the corpus block costs tokens on every triage call.

What the block is *for*, beyond accuracy: de-noising and query planning now speak the
document's vocabulary ("service credit", "carry-forward", "booking class") instead of the
employee's, and a question naming a document is routed to that document's terms.

End to end, both packs re-indexed and answered on the brief code, sequentially
(`benchmark_20261105_111219` HR in `data_bench_brief`, `…20261105_112329` contract in
`data_bench_legal_brief`), against §13's pre-brief runs:

| pack | correct | + partial | cite doc | cite doc+sec | rubric | QA wall | forced finals |
|---|---|---|---|---|---|---|---|
| HR, pre-brief (§13) | 19/20 | 20/20 | 29/29 | 29/29 | — | 340.0s | 0 |
| HR, with brief | 19/20 | 20/20 | 28/29 | 27/29 | — | 441.0s | 0 |
| contract, pre-brief (§13) | 16/20 | 18/20 | 41/42 | 39/42 | 9.30 | 570.8s | 1 |
| contract, with brief | 16/20 | **19/20** | 41/42 | 39/42 | **9.40** | 611.9s | **0** |

Read honestly:

1. **Accuracy did not move** — 19/20 and 16/20 on both sides of the change. What moved
   is the partial/incorrect split on the contract pack (18/20 -> 19/20 correct-or-partial,
   rubric 9.30 -> 9.40) and the disappearance of the forced final, i.e. the agent
   reached the right clause in fewer turns. Small, and inside the movement §13 measured.
2. **HR citation recall lost one to two references** (29/29 -> 28/29 doc, 27/29
   doc+section). n=20, and this pack has moved by one case between runs before, so it
   is recorded as no detected change rather than a regression — but it is not a win,
   and a re-run should compare `citation_section_hits` per case, not the percentage.
3. **Wall clock grew on both packs** (340 -> 441s, 571 -> 612s). That is endpoint load
   during this session, not the change: the index step itself (which the brief does
   affect) took 65-67s per pack versus 78-87s before.

## 15. The adversarial pack, 20 questions at `--workers 10`

A fourth pack (`Benchmark/adversarial_policy_arbitrage_pack`, 4 documents, 20 questions)
built to attack what the first three did not: a fictional jurisdiction with its own
revenue code and incentives act, questions where the *favourable* answer is a lawful
election rather than a prohibition, clause pairs that only look contradictory, and
"what is the maximum defensible treatment" asks where the honest answer is a qualified
one. Run `benchmark_20261105_114833`, fresh index in `data_bench_adv`.

| metric | value |
|---|---|
| correct | 16/20 (80.0%) |
| correct or partial | 19/20 (95.0%) |
| verdicts | 16 correct, 3 partial, 1 incorrect |
| citation recall (document) | 41/42 (97.6%) |
| citation recall (doc + section) | 32/42 (76.2%) |
| rubric | 9.35/10 = 3.70 / 2.75 / 2.00 / 0.90 |
| forced finals | 1 (AX-03, 38 tool calls, 104.5s — and still correct) |
| agent steps | max 39 entries, mean 12.7 (at `AGENT_MAX_STEPS=20`) |
| agent turns | not recorded in this run - `step_entries` counts thoughts + one entry per tool call + final, so it is not the cap. Re-running AX-13 with the model calls counted showed 10 turns and 20 tool calls over 21 entries; runs now report `turns` as well |
| indexing | 59.8s for 4 documents |
| wall clock | 222.1s total; 834.2s of summed per-question time |
| per question | mean 41.7s, p50 27.3s, p95 113.1s, max 126.1s |

What this pack adds to the picture:

1. **It is the slowest pack per question by a wide margin** — p95 113.1s against 30.5s
   (HR) and 45.0s (contract). The questions invite many small lookups across documents
   (AX-03 and AX-13 made 37-38 tool calls), and the agent spends steps on them rather
   than widening a single search.
2. **Document recall holds, section recall collapses** — 41/42 vs 32/42. The references
   for this pack list two or more sections per answer, and the agent typically cites the
   governing one plus supporting ones it found, not every section the rubric lists. That
   makes section recall a measure of *citation completeness against a list*, not of
   whether the citations support the claim; the honest reading is "the agent found the
   right document reliably and the right exact section most of the time".
3. **Concurrency buys less than the worker count** — 20 questions in parallel took
   222.1s of wall clock against 834.2s of summed agent time, a 3.8x speedup at 10
   workers, because the endpoint, not the harness, is the limit. The per-question
   latencies in this run are inflated by contention and should not be compared to
   sequential runs; §9 says the same about its `--workers 10` pass.
4. **The four not-fully-correct answers are documented case by case in FAILURES.md**, and
   one of them (AX-15) is a *benchmark* defect: the reference answer supplies a hybrid
   commuter-support amount the document never states, while the agent read the section
   literally and said so. Scored as an error; recorded as a disagreement.

So the pack does its job — it produces the failure the other packs could not (a
confidently wrong negative claim about what the documents cover) and it exposes that
this agent's citation habit is "cite what supports the sentence", not "cite everything
the grader expects".

## 16. The scratchpad is offered eleven times per question and used zero times

The ReAct agent has five working-memory tools (`scratchpad_write / read / edit / delete /
list`), 12 notes maximum, per question, in memory only. They were added so a question that
needs several figures could hold them between turns instead of re-deriving them. Nothing
had measured whether the agent ever reaches for them.

`eval/scratch_usage.py` asks a pack's questions and counts what the agent actually called
(`eval/results/scratch_20261105_124350.json`), adversarial pack, `--workers 10`:

```
id      calls  scratch  notes  turns  elapsed
AX-01       5        0      0      4    30.9s
AX-02      10        0      0      5    31.5s
AX-03      20        0      0     12    71.0s
...
AX-13      31        0      0     16   129.0s
AX-20       6        0      0      5    27.4s

questions that touched it      0/20
questions that kept notes      0/20
scratch calls                  0 of 200 tool calls (0.0%)
all tool calls   search_policy_docs 108, read_policy_chunk 63, calculate 19,
                 get_document_summary 6, list_policy_documents 3, list_contradictions 1
```

Zero. Not "rarely" - not once, across 200 tool calls in 20 questions, on the pack whose
questions take the most turns. The tools are in the registry, they are in the tool list the
model sees, and the prompt says "if you kept notes, call scratchpad_read once before
answering", which is a suggestion with no antecedent because nothing ever writes a note.

Two readings, and the measurement does not choose between them:

1. **The agent does not need them.** The conversation already holds every observation;
   notes would be a second copy. Nothing in these questions is long enough to need
   externalised state, and 20 turns is not a context problem.
2. **The prompt does not ask for them.** "If you kept notes" presupposes a decision the
   model is never told to make. A tool offered under a conditional that is never triggered
   gets no use, and 0/20 is what that looks like.

What is *not* true is the thing the tools were justified by in the first place - that a
multi-step question benefits from a scratchpad. On this evidence it is dead weight in the
answer path: five tool schemas in every prompt, an observation channel that never fires,
and code plus tests that no run exercises end to end.

The cheap experiment, not yet run: tell the agent to write down each figure it intends to
compare as it finds it, for questions with two or more quantities, and see whether
`calculate` calls (19 here) start consuming scratch notes instead of re-reading chunks
(63 here). If usage stays at zero with an explicit instruction, the honest move is to
delete the tools rather than keep paying for the schemas.

## 17. Citation recovery: one decision-model request for what the answer read but did not cite

The complaint was concrete: the agent reads a passage, uses it, and does not list it, so
the citation list is shorter than the evidence the answer actually rests on. The fix is
not a rule about which section supports which sentence - there is no such rule - but one
decision-model request per answer, over the sections the agent *itself* marked as read.

Mechanism (`app/citation_recovery.py`, §3 item 10 in DESIGN): candidates are the unique
(doc, section) pairs among `seen_chunk_ids` minus what the answer already cites, capped at
`CITATION_RECOVERY_MAX_CANDIDATES`; one SystemOne request carries the final answer plus all
candidates as one state, and one yes/no question per candidate ("does this passage state a
rule, condition, number or definition the answer relies on?"). Candidates at or above
`CITATION_RECOVERY_THRESHOLD` are appended, highest score first, at most
`CITATION_RECOVERY_MAX_ADDED`. The answer text is never touched, a hallucinated name in the
reply matches nothing, and a model failure is a no-op.

### 17.1 Can the decision model tell a source from a coincidence?

`eval/calibrate_citation_recovery.py`, adversarial pack, 20 questions, every seen section
scored against the reference's own citation list
(`eval/results/citation_recovery_20261105_130538.json`):

| group | n | mean | median | max |
|---|---|---|---|---|
| sections the reference expects | 38 | **0.871** | 0.975 | 0.99 |
| every other section the agent read | 293 | 0.155 | 0.030 | 0.98 |

Separation in the mean is strong, and the tail is not: some irrelevant sections score 0.98,
which is why the number added is capped. The sweep over thresholds, counting every seen
section (including the ones the answer already cited):

| threshold | sections added | true | false | precision | recall of expected |
|---|---|---|---|---|---|
| 0.2 | 82 | 36 | 46 | 43.9% | 94.7% |
| 0.4 | 76 | 34 | 42 | 44.7% | 89.5% |
| 0.6 | 71 | 33 | 38 | 46.5% | 86.8% |
| 0.7 | 69 | 33 | 36 | 47.8% | 86.8% |

### 17.2 What it actually changed end to end

Two runs of the adversarial pack on the shipped code, `--workers 10`:

| run | judged | accuracy | cite doc | cite doc+section | added | answers with additions |
|---|---|---|---|---|---|---|
| `20261105_114833` (recovery off, §15) | yes | 16/20 | 41/42 | 32/42 (76.2%) | — | — |
| `20261105_130704` (recovery on) | yes | 15/20 | 41/42 | **37/42 (88.1%)** | not recorded | not recorded |
| `20261105_131141` (recovery on) | no | — | 42/42 | 34/42 (81.0%) | **7** | 6/20 |

In the unjudged run, which records what was added, every one of the seven additions was a
section the reference does **not** list:

```
AX-01  other  01_FICTIONAL_REVENUE_CODE § 1.3 Related persons
AX-19  other  02_ENTERPRISE_INCENTIVES_AND_GRANTS_ACT § 2. Common control
AX-12  other  02_ENTERPRISE_INCENTIVES_AND_GRANTS_ACT § 4.4 Clawback
AX-20  other  04_OWNERSHIP_TRUSTS_AND_DISCLOSURE_RULES § Norland Ownership ... 2032
AX-13  other  02_ENTERPRISE_INCENTIVES_AND_GRANTS_ACT § 1. Programs
AX-13  other  02_ENTERPRISE_INCENTIVES_AND_GRANTS_ACT § 11. Recovery and penalties
AX-03  other  04_OWNERSHIP_TRUSTS_AND_DISCLOSURE_RULES § 8. Programme look-through
```

They are real sections from the right documents, and several are defensible sources for
the answer that cited them (`1.3 Related persons` for a related-persons question; `2. Common
control` for an aggregation question). They are simply not what the rubric asked for.

### 17.3 The finding that matters more than the feature

The premise was that citations were being **dropped** from answers. On this pack they were
not. Of the 42 sections the references expect, the agent read 38, and it cited **37 of those
38** without any help; the four it never cited were sections it never read (`9.2 Indicators`,
`9.3 Safe tax planning`, `8. Stacking and double funding`, `10. Anti-abuse rule`). Those are
retrieval misses, and no post-processing can cite a passage that was never retrieved.

So the section-recall gap on this pack (32/42 in the unassisted run) is roughly: 4 sections
never retrieved, ~4 retrieved but judged by the decision model not to support the answer
(scores 0.06-0.25), and the rest a matter of which of several plausible sections the rubric
happens to list.

Consequences recorded rather than acted on:

* Recovery stays on by default because it is what was asked for and it costs one bounded
  request; it never edits the answer and never invents a section. Anyone who cares more
  about citation precision than completeness should set `CITATION_RECOVERY=0`.
* The lever that would actually move section recall is retrieval, not post-processing: the
  4 never-read sections are the ones the grader wanted. That is a search problem (§2.1), and
  it is the honest next experiment, not another pass over the answer.
