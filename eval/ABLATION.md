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
```

---

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

| variant | intent acc. | route acc. | false refusals | missed refusals | classifier p50 |
|---|---|---|---|---|---|
| legacy (regex) | 50.7% | 74.6% | 0 | 15 | 0.0s |
| prompt v1 | 98.5% → **100%*** | **100%** | **0** | 1 | 5.22s |
| prompt v2 | 97.0% → **98.5%*** | **100%** | **0** | 2 | 5.09s |

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
|---|---|---|---|
| v1 | 50% | 100% | 6/6/0 |
| v2 | **86%** | 100% | 6/1/0 |

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

| effort | reasoning tokens | accept-match | mean latency | p90 |
|---|---|---|---|---|
| default (unset) | 8-225 | 20/20 | — (p50 5.2s over 67 cases) | 8.0s |
| `off` | **0** | **20/20** | **4.1s** | **5.7s** |

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

<!-- END2END_TABLE -->

## 6. Judge calibration on injected claims

<!-- JUDGE_CALIBRATION -->

## 7. What this changed in the code

| measurement | change |
|---|---|
| regex triage 74.6% route accuracy, 15 missed refusals | replaced with the LLM triage layer; regex kept only as the baseline |
| classifier truncated at 700 output tokens | cap raised, corrective retry, `INTENT_MAX_TOKENS` documented as a ceiling |
| `reasoning_effort=off` identical accuracy, 21% lower mean latency | classifier runs at `off` by default |
| judge accused correct conflict answers of inventing facts | judge now reads the cited chunks first, with a 24-chunk/24k-char budget and a coverage note |
| judge returned `false` with no named claim | one corrective retry, then degrade to `unknown` |
| table flag P=50% | v2 prompt → P=86%, R=100% |
| keyword retrieval broken (0/17) | FTS backfill fixed against `chunks_fts_docsize` |
| hybrid ≈ vector on this corpus | `RETRIEVAL_MODE` kept switchable, default documented with its evidence |

## 8. Threats to validity

* **One corpus, seven documents.** Retrieval metrics saturate; answer quality on
  this corpus is not evidence about a 200-document library.
* **LLM-as-judge.** The faithfulness verdict is itself a model output with
  run-to-run variance (observed: the same variant scoring 4 vs 5 failures across
  two runs at temperature 0). It is used as a *signal*, never as ground truth.
* **n=29 end-to-end cases.** One flipped case moves a column by 3.4 points.
  Differences under ~10 points should be read as noise.
* **Prompt v2 was written by the same person who wrote the cases.** The hard case
  set was built to expose v1's weaknesses, so the improvement is partly by
  construction; the honest claim is "v2 fixes specific observed failures", not
  "v2 is generally better".
