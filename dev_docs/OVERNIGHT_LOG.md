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

## 3. Test suite

119 → 156 tests, all passing. New coverage: triage (cache, fail-open, retry,
plan injection, route derivation), judge gating and user-message forwarding,
embedding fingerprint, citation parsing, client feature degradation
(`reasoning_effort`, `max_tokens` halving, auth errors not retried).

## 4. Measurements

(filled in as runs complete — see `eval/ABLATION.md` for the full write-up)

