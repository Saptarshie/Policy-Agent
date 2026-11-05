# Failure analysis — every answer that was not fully correct, and why

This is the companion to `eval/ABLATION.md`. ABLATION records what each change did to
the *aggregate* numbers; this file records what the system actually got wrong, one
answer at a time, and what the mechanism of each error was.

## Current state: four answers not fully correct, on the shipped prompt (v3)

Three runs at `--workers 5 --reuse`, prompt v3, k=12, contract index scanned:
`benchmark_20261105_190013` (adversarial), `_190344` (contract), `_190713` (HR).
60 questions: **56 correct, 3 partial, 1 incorrect**.

| case | pack | verdict | right | still missing |
|---|---|---|---|---|
| AX-01 | adversarial | partial | conditional "no automatic attribution"; the conjunctive test and its carve-out applied | leaves the owner-skill threshold unsettled, where the reference treats that condition as satisfied |
| AX-19 | adversarial | partial | transfer not ignored solely for an eligibility motive; genuine-transfer factors applied | the caveat: programme-specific common-control / related-person rules may still aggregate the parties |
| AX-15 | adversarial | incorrect | N$4,000 remote-work support and the either/or rule read correctly | the reference asserts a hybrid commuter amount the policy never states (mechanism D - a benchmark defect, not an agent error) |
| LC-15 | contract | partial | the five-business-day saving provision applied; invoice remains valid | the opening "Yes" contradicts its own analysis, and the reference's assumption that the other invoice fields were present is unstated |

How the five mechanisms stand now:

- **A (rule restated, scenario not instantiated) — closed.** Prompt v3's conditions list
  took LC-08 from 0/4 to 4/4 and MH-12 from 0/4 to 4/4 across repeats, with LC-10, LC-01 and
  AX-20 unchanged (`eval/ABLATION.md` §21).
- **B (sub-fact unresolved) — down to AX-13**, which is 1/3 correct now (from 0/3): its
  missing fact is an *unstated* eligibility precondition, which a list of the conditions the
  question states cannot supply.
- **C (calibration) — one case left**, AX-01 hedging on a threshold the document leaves to
  judgment. LC-14 now concludes.
- **D (contested ground truth) — unchanged**, AX-15, by construction.
- **E (missing caveat) — now the dominant residual**: AX-19 and LC-15, both answers that are
  substantively right and do not say the extra thing the reference counts as part of the
  answer. That is a different problem from A: nothing was misapplied, something was left unsaid.

So the honest summary of the current state: the failures are no longer "did not apply the
condition" - they are "answered the question and omitted an assumption, a caveat, or a
judgment the reference expects". One is a benchmark defect. Lever 5 below is the one written
for the remaining family, and it is still unbuilt and unmeasured.

---

## The run this file was originally written from (pre-v3, kept for the mechanisms)

Collected from three runs on the shipped code (document brief, no rolling summary,
triage shown the corpus block), 60 questions in total. The adversarial row is the run
before citation recovery was switched on; a later run of the same pack (`ABLATION` §17)
swapped three verdicts and produced no new failure mechanism, which is the sampling
already documented in §13 — so this file stays tied to the run it analyses:

| pack | run | correct | partial | incorrect |
|---|---|---|---|---|
| HR policy (20) | `benchmark_20261105_111219` | 19 | 1 | 0 |
| contract reasoning (20) | `benchmark_20261105_112329` | 16 | 3 | 1 |
| adversarial arbitrage (20) | `benchmark_20261105_114833` | 16 | 3 | 1 |

Nine answers are not fully correct. They are not nine independent accidents: they fall
into five mechanisms, and three of the nine share one.

A verdict is produced by an LLM judge that is given the reference answer and the
question, and returns correct / partial / incorrect plus the specific points the answer
missed (`judge_missing`). The judge is fallible; where its verdict is contestable, that
is said so below rather than smoothed over.

---

## The five mechanisms

| # | mechanism | cases | what goes wrong |
|---|---|---|---|
| A | **rule restated, scenario not instantiated** | MH-12, LC-08, LC-10 | the answer recites the governing rule correctly and never substitutes the question's own values into it. In LC-08 a *retrieved* value replaces a *given* one. |
| B | **deciding sub-fact left unresolved or assumed** | LC-07, LC-14, AX-13 | the rule is applied, but the one further fact that decides the answer (which fee is highest; whether the speaker had authority; whether the cost is eligible at all) is neither looked up nor flagged as missing. |
| C | **calibration** | AX-01, (LC-14) | the answer hedges on an element the facts settle (AX-01), or concludes firmly where the facts do not settle it (LC-14). Same axis, opposite errors. |
| D | **contested ground truth** | AX-15 | the reference answer asserts something the document does not say. The agent's reading is the literal one. |
| E | **missing "and also" caveat** | AX-19 | the direct question is answered correctly, and the caveat that the reference treats as part of the answer is absent. |

A and B together are six of the nine, and both are the same underlying weakness: the
model is good at *retrieving and quoting* a rule and weak at *running it*.

---

## A. Rule restated, scenario not instantiated

### MH-12 — HR pack, partial, both runs (11.4s / 11.6s, 2 tool calls)

> A business-only flight costs ₹28,000, but adding two personal days makes the ticket
> ₹25,000. How much does the company pay, and who pays personal lodging?

Reference: the company pays **₹25,000** (the lower of the two), the employee pays
personal lodging.

What it answered:

> "The company reimburses the lower of the two fares — so if the combined ticket is
> cheaper, it pays the actual combined itinerary amount — and the employee bears all
> personal lodging costs."

Then, in the second run, it went further and defended *not* giving the number:

> "This is an illustration only; the rule to apply is the lower-of comparison above,
> using your own quoted and actual amounts."

It had both amounts in the question. It performed the comparison in prose ("if the
combined ticket is cheaper") without ever doing it, and it spent the rest of the answer
describing the policy's worked Example A — which is the *opposite* case — instead of
applying the rule to the numbers it was given. The number ₹25,000 appears nowhere.

Mechanism: the answer treats "state the rule" as the task. Two tool calls: it found the
section and stopped. No calculation was needed beyond a comparison, and no calculator
was used.

### LC-08 — contract pack, **incorrect**, both runs (15.4s / 22.0s)

> Provider performed six hours of properly noticed Sunday Scheduled Maintenance, and the
> service was unavailable throughout. How many hours can count as Unavailable?

Reference: **two hours** — the four-hour monthly exclusion is exceeded by two, and the
excess counts as Unavailable because the service was down.

What it answered, both runs:

> "**Zero hours** ... the window is 4 hours (00:00–04:00) and the monthly exclusion cap
> is 4 hours, so 4 − 4 = 0 hours count as Unavailable."

This is the most instructive failure in the suite. The answer quotes the decisive
sentence almost verbatim —

> "Scheduled Maintenance exceeding four hours remains excluded for the first four hours;
> excess minutes count as Unavailable if the Covered Service is unavailable"

— and then reasons `4 − 4 = 0` anyway. It replaced the question's **six hours** with the
policy's **four-hour window** and computed on the substituted number. The correct
arithmetic is `6 − 4 = 2`, and the sentence it quoted says exactly that.

Mechanism: a retrieved constant overwrote a given one. Not a retrieval failure (right
section, right quote), not a reasoning failure in the abstract — a binding failure. The
same answer in two runs with different step counts says it is systematic, not sampling.

### LC-10 — contract pack, partial (21.8s, 4 tool calls)

> Provider confirms unauthorized access to Customer Personal Data at 10:00 Monday but
> waits until it has complete forensic findings at 16:00 Tuesday to notify Customer. Is
> the notice timely?

Reference: **not timely** — 16:00 Tuesday is 30 hours after confirmation, 6 hours past
the 24-hour deadline of 10:00 Tuesday.

What it answered:

> "The documents do not state the confirmation time, so the assessment turns on it...
> For example, confirmation at 09:00 on day 1 gives a deadline of 09:00 on day 2, so a
> 16:00 notice is 7 hours past the deadline."

The question states the confirmation time in its first clause ("confirms ... at 10:00
Monday"). The answer declared it unknown, then did the arithmetic correctly on an
invented 09:00. So the calculation machinery works; what failed is reading its own
inputs, the same family as LC-08 — the scenario's parameters never made it into the
computation.

---

## B. The deciding sub-fact is unresolved or assumed

### LC-07 — contract pack, partial (13.0s, 4 tool calls)

> The base platform and both modules were unavailable in the same incident. Can Customer
> claim three separate availability credits?

Reference: no — one credit, calculated on the **highest applicable monthly recurring
fee, which is the base platform's fee on the stated price table**.

The answer got the no-duplication rule, quoted "highest applicable monthly recurring fee
among the affected services", then reproduced the availability-credit percentage table
and the claim process. What it never did is identify **which** fee is the highest, which
requires reading the price table. It cited §12, §4, §7 and §5 — not the price table. So
either the table was not retrieved, or it was retrieved and not recognised as the thing
that decides the sentence the model had just quoted.

### LC-14 — contract pack, partial (31.9s, 8 tool calls)

> Customer uploaded health information after support personnel said in a ticket that it
> was acceptable. Who bears remediation costs when the upload is discovered?

Reference: the upload was unauthorised (a ticket cannot amend the DPA); the Customer
**ordinarily** bears the costs **unless** Provider knowingly encouraged it; and because
the facts do not establish the speaker's knowledge or authority, **cost allocation
cannot be conclusively determined**.

The answer:

> "**Short answer: the Customer bears the remediation costs — unless the Provider
> *knowingly encouraged* the upload" ... "Whether a support statement rises to 'knowingly
> encouraged' is a factual question the documents do not further define."

Everything it says is defensible, and the last sentence is the right observation. The
failure is one of **commitment and ordering**: it led with a definite allocation and
demoted the indeterminacy to a footnote about definition, when the indeterminate
conclusion *is* the answer. A reader who stops after the bolded line gets "Customer
pays", which is not what the reference says.

### AX-13 — adversarial pack, partial (121.8s, 37 tool calls — longest answer in the suite)

> A company receives a Green Equipment Rebate for a machine and also includes the
> machine's full invoice as eligible research expenditure for the Innovation Grant
> because researchers use it 40% of the time. What is the maximum defensible treatment?

Reference: the rebate-supported cost must be removed, **and** the equipment cost must
itself be eligible research expenditure rather than a capital purchase or pre-existing
technology — which the facts do not establish. So the maximum defensible claim is a
properly allocated, otherwise eligible portion, not a computed entitlement.

The answer did the stacking arithmetic correctly and thoroughly (worked example on an
invented N$1,000,000 invoice, four steps, correct deductions) but **assumed the property
the question never establishes**: that 40% research use makes a machine purchase eligible
research expenditure at all. It imported the question's framing as a premise and then
computed on it — mechanism B, with a side of A.

---

## C. Calibration

### AX-01 — adversarial pack, partial (40.7s, 9 tool calls)

> Mira owns a consulting company with one major client generating 78% of revenue. She is
> the sole consultant ... two unrelated engineers perform substantive delivery work. Is
> all company income automatically attributed to Mira?

Reference: two conditions apparently satisfied (concentration, and her skill or labour as
the principal source), one not; the protective test is partly established; the answer is
conditional.

The answer called the skill/labour condition **"undetermined"** because "the documents
state this condition but give no test for weighing it". That is a category error: the
documents do not need to supply a test for a fact the facts supply. A sole consultant
whose one client is 78% of revenue satisfies "the individual's skill or labor is the
principal source". It also softened the asset element to "points this way" where the
reference says "appears to bear commercial capacity".

Mechanism C, over-hedging: absence of a *threshold definition* treated as absence of a
*fact*. This is the mirror of LC-14, where the answer was too definite where the
documents were silent.

---

## D. Contested ground truth

### AX-15 — adversarial pack, **incorrect** (109.0s, 29 tool calls)

> A Hybrid employee works remotely eight days and commutes on twelve days in a month.
> Public transport cost is N$5,500. Which benefit may the employee choose?

Reference: remote-work support of N$4,000 **or commuter support based on actual cost of
N$5,500**; the policy lets the employee choose the more favourable after month-end, so
the answer is N$5,500.

The document (`03_EMPLOYEE_BENEFITS_AND_EQUITY_POLICY`, §4) says, in full:

> "Office employees receive actual public-transport cost up to N$6,000 monthly. Hybrid
> employees receive either commuter support or remote-work support for a given month,
> not both. Remote-work support is N$4,000 per month where the employee works remotely at
> least eight scheduled days."

The amount for **hybrid** commuter support is never stated. The reference answer supplies
it by extending the office-employee rule. The agent refused to do that:

> "The section ... does not state the amount or cap of commuter support for a *hybrid*
> employee. I checked the whole document (all 17 chunks) and there is no table setting
> out a hybrid commuter-support cell ... For illustration only, and flagged as an
> assumption rather than policy: *if* hybrid commuter support followed the office rule,
> N$5,500 ... a difference of N$1,500."

On the text, the agent is right and the reference is inferring. This is recorded as a
benchmark defect, not an agent failure — but the *behaviour* around it is worth keeping:
the answer asserts "I checked the whole document (all 17 chunks)", a claim about its own
tool coverage that no tool actually verifies. An honest version would say "I found no
section stating it", which is what it can know.

---

## E. The missing caveat

### AX-19 — adversarial pack, partial (24.8s, 6 tool calls)

> A founder transfers 30% of a company outright to an adult sibling ... Must the transfer
> be ignored solely because benefit eligibility motivated it?

Reference: no — motive alone does not invalidate a genuine transfer; **and** programme-
specific common-control or related-person rules may still aggregate the parties for a
separate purpose, and crossing the registry threshold does not guarantee grant
eligibility.

The answer nails the first half and stops. It cites §9 anti-circumvention and §5
temporary transfers and reasons from substance and control, which is exactly right, but
the two caveats that keep the answer from being read as "the plan works" are absent. The
user's real question is "does this get me eligibility?", and the answer as written
implies yes.

---

## Fixed since the earlier runs (and one that was not a fix)

The 2026-11-05 01:0x runs of the same packs had LC-01, LC-04 and LC-15 failing. All three
pass in the current runs. Only one of those can be attributed to a change:

* **LC-04** (productive use vs CO-2 threshold) failed because the answer declared the two
  clauses an unresolved conflict. The mechanism is now pinned (ABLATION §19): the benchmark
  indexes **without** running the contradiction scan, so `list_contradictions` returned
  nothing, and the agent said so in its answer ("the contradiction scanner records no
  verified flag for this pair"). Given an empty flag list and a prompt that forbids picking
  sides between conflicting documents, abstaining was the instructed behaviour — the index
  was stale, not the reasoning. With the scan run, both precedence pairs in that pack are
  flagged correctly (2 flags, 0 spurious, 22.3s), which removes `list_contradictions` as the
  explanation for the current-run pass: nothing targeted precedence, so treat the recovery as
  sampling.
* **LC-01** (labelling 45 vs 35 billable accounts) and **LC-15** (the payment clock,
  which spent 20 tool calls and hit the step cap) are sampling — §13 already showed the
  same pack scoring 15/20 and 16/20 on identical code.

So: three of the earlier failures are gone and no code change was aimed at two of them.
Treat single-run deltas in this file as noise unless the mechanism is named.

---

## What has *not* been tried

These are the levers the mechanisms suggest. None of them has been measured, and they are
listed as proposals precisely so nobody later mistakes them for findings.

1. **An instantiation step, as prompt text** — before concluding, restate the values the
   question gives and substitute them into the rule; if the rule's own number (a four-hour
   window) differs from the scenario's number (six hours), the scenario's number governs
   and the difference is the answer; if the answer is an amount, write the amount. This
   targets A directly (MH-12, LC-08, LC-10) and was the highest-value single change
   available. **Built and measured as prompt v3** (`eval/ABLATION.md` §21): LC-08 and MH-12
   went 0/4 -> 4/4, controls flat, packs 52/60 -> 56/60. The verbosity risk was watched for
   and did not materialise - `reasoning` rose (2.85 -> 3/3 on contract) without
   `answer_correctness` falling.
2. **"Resolve the sentence's subject before quoting it"** — when a rule says "the highest
   applicable fee", find which fee is highest before moving on. Targets LC-07.
3. **Indeterminacy first** — when a required fact is not established (authority, elapsed
   period, eligibility), the leading sentence should be the indeterminate conclusion, not
   the default rule. Targets LC-14, AX-01 (the inverse), and AX-13.
4. **Eligibility before arithmetic** — before computing a benefit, confirm the cost is
   eligible at all. Targets AX-13.
5. **"Answer the question asked, then the question behind it"** — for eligibility-shaped
   asks, state the caveats that stop the answer being read as approval. Targets AX-19.

A and B are prompt-shaped problems, which is why they are listed as prompt levers. If
they survive a prompt change, the next honest step is not another prompt: it is a
calculator-forced step on questions whose reference answer contains a number — and that
already exists as a tool the agent simply did not reach for on MH-12.

---

## How to reproduce

    /c/Python313/python.exe eval/run_benchmark.py --pack Benchmark/hr_rag_policy_pack_and_benchmark --data-dir data_bench_brief --workers 5 --reuse
    /c/Python313/python.exe eval/run_benchmark.py --pack Benchmark/complex_contract_reasoning_pack --data-dir data_bench_legal_scan --workers 5 --reuse
    /c/Python313/python.exe eval/run_benchmark.py --pack Benchmark/adversarial_policy_arbitrage_pack --data-dir data_bench_adv --workers 5 --reuse

Per-question rows (answer, judge reason, missed points, citations, timings) are in
`eval/results/benchmark_<stamp>.jsonl`. `python eval/trace_steps.py <pack> <ids> <data_dir>`
prints the step trace for one question when the question is *why* it went where it went.
