# Audit: where a semantic question is decided without a model

Written after two of these mistakes were caught in review. The rule they violate:

> A language question is a model's job. Code may decide *structure* — is this a
> citation-shaped reference, does this chunk id exist, is this number in this
> passage. It may not decide *meaning* — is this a refusal, which document is
> this, is this a contradiction — by matching strings or comparing scores.

Each entry below is a real decision point, what it decides, and its verdict.

Verdicts: **STRUCTURAL** (keep — the input genuinely has a grammar) ·
**FIXED** · **OPEN** (documented, not fixed) · **OFFLINE** (not in the answer path) ·
**BASELINE** (kept only for the ablation).

---

## 1. Fixed in this pass

### 1.1 `app/agent/react.py:_match_doc` → `app/naming.py:resolve` — FIXED

Decided *which stored document* a citation names, by first match in list order:

```python
for n in names:
    if p == nl or p in nl or nl in p: return n      # substring, first wins
for n in names:
    if difflib(...) >= 0.72: return n                # fuzzy, first wins
```

With `remote_work_policy_v1` and `remote_work_policy_v2` in the store:

```
'Remote Work Policy v2'  ->  'remote_work_policy_v1'      # WRONG, silently
```

An answer citing the amended policy was recorded as citing the superseded one —
and those two documents are the contradiction pair the product exists to find.
The 0.72 threshold made it worse: the two names are ~0.97 similar, so it cannot
separate them, and returning the first match hid the failure.

Now (`app/naming.py`): normalise (case, separators, extension) → exact equality →
unique whole-token subset → fuzzy ≥ 0.90 **with** a 0.08 margin to the runner-up.
Anything ambiguous returns `Match(how="ambiguous")`, which becomes a citation
warning: `citation 'Remote Work Policy § 2' does not match any ingested document`.
Guessing is not an option the resolver has any more. Tests: `tests/test_naming.py`.

### 1.2 `app/db.py:find_document_fuzzy` — FIXED

Same first-substring-wins loop, used by the judge's context builder and by any
caller resolving a human-typed document name. Now delegates to `naming.resolve`.

### 1.3 `app/agent/verify.py` refusal detection — FIXED (then removed, see §3)

`_REFUSAL_MARKERS` was a hardcoded phrase list deciding whether the faithfulness
judge ran. Replaced by the decision model (`answer_shape.py`): four probability
questions, shape derived, ambiguity → `unknown` → judge anyway. It is out of the
answer path entirely now, but the lesson stands: a phrase list for a language
property is a losing game.

### 1.4 Numbers: containment is not verification — FIXED (design)

An earlier proposal in this repo's notes was "every number with a unit must appear
in a cited passage". That is wrong: a policy assistant *derives* numbers (pro-rated
entitlement, totals, per-diem × nights, tax on a stated salary). The correct rule
is provenance — a number is admissible if it is (a) in a passage, (b) in the
employee's own message, or (c) produced by a computation over (a)/(b). Rejected the
containment idea; the prompt now states the weaker, correct contract: a derived
figure must show its inputs and arithmetic (AGENT_SYSTEM rule 8, AGENT_SYSTEM_V2
rule 11), so a reviewer can check the derivation instead of the tool pretending to.

---

## 2. Open — semantic decisions still made lexically

### 2.1 `app/contradictions/gate.py:JEVStyleGate.evaluate` — OPEN (the big one)

Decides **whether two passages contradict**, using:

* topical overlap: content-word Jaccard ≥ 0.10, or a shared bigram;
* modality regexes (`must` / `shall not` / `may`) scored +0.45 / +0.25 / +0.20;
* numeric conflicts (different numbers under similar phrasing) × 0.9;
* total ≥ `JEV_GATE_THRESHOLD` (0.5) → contradiction.

This is a language judgement made by word overlap and regex lexicons. In
`DECISION_MODE=hybrid` it is a *prefilter* and the decision model refines the
borderline band; in `hybrid` the clearly-above/below pairs are still decided
lexically. In `DECISION_MODE=remote` the decision model decides every pair.

Why it survives: with no endpoint configured there is nothing else to call, and
the zero-cost default is a real requirement for a handed-out repo. Why it is a
compromise, not a principle: a pair can be a genuine contradiction with low
lexical overlap ("staff may work from home two days" vs "employees are expected
on site three days a week") and be dropped before any model sees it.

Recommendation: any deployment with an endpoint sets `DECISION_MODE=remote` (every
measurement in `eval/` does). A principled version would let the heuristic only
*rank*, never *dismiss* — i.e. below-threshold pairs are `UNVERIFIED`, not
"not a contradiction" — at the cost of many more pairs to review.

### 2.2 `_REQ_RE` / `_PROHIB_RE` / `_PERM_RE` (same file) — OPEN

Modality detection (`must`, `shall not`, `is required to`, `may`) by regex, feeding
the score above. Legitimate as a *signal* inside a prefilter, illegitimate as a
decider — which is what it is in heuristic mode.

### 2.3 `app/agent/intent_legacy.py` — BASELINE

Regex/keyword triage: the original overfitting intent gate, kept solely so the
ablation can report 74.6% route accuracy as the baseline. Must never be wired into
`AdminAPI`. (Grepped: nothing outside `eval/` imports it.)

### 2.4 `requires_table_check` — OPEN (calibration, not classification)

Produced by the LLM (fine) but over-fires (16/67 in the calibration set) and has no
consumer. A correct signal that nothing acts on is still a defect in the pipeline;
see CODE_REVIEW O3.

---

## 3. Offline / evidence only — not in the answer path

| site | decides | why it is acceptable |
|---|---|---|
| `agent/verify.py:quote_matches` (fuzzy 0.82) | is this quote present in this passage | offline audit only; a similarity threshold for "did the model quote this" is a measurement, not a product decision |
| `agent/answer_shape.py` (0.5 cutoffs) | answer / partial / refusal / chitchat | thresholds on the decision model's own probabilities; the model does the meaning, the thresholds only bucket it — and it is not in `ask()` |
| `agent/verify.py:judge` | faithfulness | same-model judge, measured at TNR 8/10 and +6s; removed from the path, kept for calibration |

`eval/scoring.py:REFUSAL_RE` decides whether an answer "is a refusal" **for
scoring the benchmark**. Acceptable because it scores evidence, never the product:
a wrong label shows up as a wrong pass rate, not as a wrong answer to an employee.
It must stay out of `app/`.

---

## 4. Genuinely structural — reviewed and kept

| site | decides | why it is structural |
|---|---|---|
| `agent/react.py:_looks_like_citation` | is this `[ ... ]` a citation | the citation grammar is fixed by contract (`doc § section`, `doc #id`) |
| `agent/react.py:_CITE_RE` | split a citation into doc / section | parsing a format we defined |
| `naming.normalise` | case, separators, extension | text normalisation, no meaning |
| `chunker.py` heading regexes | section boundaries | shapes (`#`, `3.2 Title`, ALL CAPS, underlined), not semantics |
| `rag.py` RRF constants (k=60, weights) | ranking order | ordering, not truth; recalibrate with `eval/measure_retrieval.py` |
| `db.reconcile_stale_jobs` (30 min) | is this job abandoned | operational timeout |
| `ingest` window/chunk sizes | where to cut long text | size budget |
| `admin._document_listing` | render the document list | formatting |

---

## 5. The pattern, in one line each

1. **Parsing a format we defined** → code. Correct.
2. **Identity of a named entity** ("which document is this") → code, but only with
   an explicit *ambiguous* outcome. Guessing is the bug.
3. **Judgement** ("is this a contradiction", "is this a refusal", "is this claim
   supported") → a model, or nothing. Never a threshold on surface features.
4. **Numbers** → provenance (passage, user message, or a shown computation), never
   containment: derived values legitimately appear nowhere in the corpus.
