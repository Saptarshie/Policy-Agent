# UI and API verification

Driven against a live server (`uvicorn app.main:app`) on a **copy** of the sample
database, real `deepseek-v4.1-flash` for the agent, real `typesafe/jev` for the
contradiction cascade, `embeddinggemma` via Ollama. Nothing here is a mock.

```
DATA_DIR=<scratch>/ui_verify  uvicorn app.main:app --port 8077
BASE=http://127.0.0.1:8077 python eval/verify_api.py     # 42 checks
python eval/verify_renderer.py                           # renderer, off-browser
```

Final state on the fixed tree: **42/42 API checks**, **14/14 renderer cases**,
**0 citation warnings across all seven archived answers**, upload cleanup verified,
and every UI control clicked in a real browser (`window.__errs` empty throughout).
The last full pass (`eval/results/ui_final/`) ran against a fresh copy of the
database on the committed tree, not on the intermediate state.

## What was exercised

| Area | Checks |
|---|---|
| Health | model/embedder names, document + flag counts |
| Index | page served (15 KB), three panels present, title |
| Upload | `.md`, `.txt`, `.pdf` accepted and ingested; `.docx` -> 400 with the allowed list; empty file -> 400; upload returns `doc_id`; document reaches `ready` with chunks; unique naming (`sample_policy (2)`) |
| Ingest cascade | an uploaded 5-day remote policy is auto-scanned after ingest and flagged against `remote_work_policy_v2` (gate 0.95, confidence 0.99) with both passages and an explanation |
| Ask | answerable question cites 7 passages; out-of-corpus question refuses; greeting answered in 3.2s with 0 tools; partial/premise question 41s/11 tools; arithmetic on the user's own numbers; contradiction question |
| Contradictions | list all/open, status change, invalid status -> 400, unknown id -> 404 |
| Chunks | lookup by id, unknown id -> 404 |
| Scan | manual rescan queued + finished, unknown document -> 400 |
| Delete | deletes doc+chunks+flags, second delete -> 404 |
| UI controls | upload via the real file input, ask via the textarea, resolve a flag (3 -> 2 cards, badge follows), rescan, delete a document whose name contains an apostrophe, **＋ New conversation** (history 4 -> 0, pane 5 -> 1 bubble, focus restored, in-flight stream aborted and never lands) |

## Bugs found (all fixed in this pass)

**1. Delete/flag buttons were dead for documents whose name contains a quote.**
Cards built inline handlers with user data:

```html
<button onclick="delDoc(11, 'Bob's travel policy')">
```

`esc()` turns `'` into `&#39;`; the HTML parser decodes it back to `'` *before*
the JS parser sees the attribute, so the handler is a syntax error and the button
does nothing. Reproduced with a real upload named `Bob's travel policy`:

```
verdict: SYNTAX ERROR: missing ) after argument list
```

Fixed by rendering `data-del` / `data-rescan` / `data-flag` attributes and
delegating from the two lists, so no user data ever reaches a script context.
Verified: click deleted the document, zero `[onclick]` attributes remain.

**2. Answers were rendered as raw markdown.** Every bubble showed `## Direct
answer`, `**bold**`, `|---|` tables and `-` bullets literally. Added a small
renderer that escapes first and then handles headings, bold, italics, inline
code, lists, tables, blockquotes and rules — so it cannot introduce markup.
Verified on the real answers (14 cases, including `<script>` and
`javascript:` inputs, which come out inert).

**3. A content question about the corpus was answered with the inventory.**
"Do any of the policy documents contradict each other about remote work?" was
classified `META_DOCUMENTS` / `DIRECT_REPLY` and answered with the document list
in 0.0s, 0 tools. The classifier had understood the question (its own
`de_noised_question` said "do the policy documents contain contradictions") but
the intent is too coarse. Both intent prompts now say META_DOCUMENTS is
**inventory only**, and that questions about what the documents *say* — including
contradictions, staleness, overlap or gaps — are policy questions that must reach
the agent. Verified: same question now routes `REACT_LOOP` /
`POLICY_DIRECT_LOOKUP`, uses 12 tools, answers with 7 citations.

**4. The agent could not see the tracker's verdicts.** Even correctly routed, it
could only rediscover conflicts by re-reading text, although the cascade had
already verified and explained those pairs. Added an 11th tool,
`list_contradictions` (status filter, cap, both sides, why, and an instruction to
cite normally). Its empty result says "none found so far", not "the documents
agree". On the contradiction question the agent now calls it **first**.

**5. The intent cache replayed verdicts from before the prompt edit.** The key was
`prompt_version|model|question`, so editing a prompt under the same label ("v1")
never invalidated anything — exactly the situation during tuning, and the reason
bug 3 appeared "unfixed" after the patch. The key now includes a fingerprint of
the prompt text.

**Minor:** `<title>` said "Policy RAG Agent" while the header says PolicyPilot —
aligned.

## Notes, not bugs

- `POST /api/documents` returns `{doc_id, doc_name}` and **no job id**; the UI
  polls the document list, which carries `latest_job`. Verified by polling the
  document instead.
- A document's progress bar shows its *latest* job, so once ingestion finishes the
  scan job replaces it — informative, but it is the scan's progress, not ingest's.
- Uploaded files are stored under `data/uploads/<uuid><ext>` and deleting the
  document **does** remove the stored file (`eval/verify_upload_cleanup.py`:
  upload -> delete -> the file is gone).
- The chat keeps the last 10 turns as history and sends them with each question;
  the trace panel shows tool calls, observations and elapsed time.
- Citation chips are a display heuristic: `[doc § section]` and `[doc_name]` are
  chipped, `[Q1]`/`[42]` are left alone.

## Evidence

- `eval/verify_api.py` — the 42 checks, re-runnable against any running server
- `eval/verify_upload_cleanup.py` — upload, delete, confirm the stored file is gone
- `eval/verify_renderer.py` — loads the page script in node with a stub DOM and
  renders the archived answers, including XSS inputs
- `eval/results/ui_verify/api_verification.json` — per-check pass/fail
- `eval/results/ui_verify/ask_*.json` — full agent responses (answers, citations,
  steps, intent plan) for every question asked
- `eval/results/ui_verify/ask_contradiction.json` — the routing fix, end to end

## 2026-11-05 — citations are verifiable, and a silent stream stops spinning

Two UI defects, both reported from the running console.

**1. A citation chip looked like a link and did nothing.** The chip was a `<span>`;
a reader who wanted to check "is this really in § 4.2?" had to retype the section
into a search. Chips are now `<button class="cite" data-cite="…">`, and clicking one
calls `GET /api/citations/resolve?cite=…` (`app/citations.py`) which returns the
stored passage, its breadcrumb and its chunk id — or, when nothing matches, the
reason plus the sections the document actually has. The chip turns red on a miss.
The match is forgiving about the surface form a model produces (case, punctuation,
"4.2" vs "42", a shortened title) and strict about never inventing text:
`tests/test_citation_resolve.py` pins exact/prefix/contains matching, the
`[doc #chunk_id]` form, an unknown section (candidates, `text: null`) and an
unknown document.

Verified live on the dental question: 4 chips, the first resolving to
`02_HEALTH_AND_WELLBEING_BENEFITS § 4.2 Dental implants · chunk 33` with the tier
table in the panel; a fabricated section gives "✗ that section is not in this
document" plus the real headings, a fabricated document gives "✗ that document is
not in the store", and Escape closes the panel.

**2. "working — triage…" could sit there forever.** Nothing in the UI distinguished
"the model is thinking" from "the request is dead": a stalled model call is a
10-minute HTTP read timeout, and a stream that closed without a `final` frame (server
restart, killed worker) left the spinner and the yellow dot up with no error. Now:

* the waiting state shows elapsed seconds and, after 15s of silence, names it
  ("· 42s (no output 27s)"), because keep-alive comments are excluded from the
  timer — the clock measures *output*, not connection health;
* 45s of silence turns the dot amber; 4 minutes aborts the request and says so;
* an `error` frame ends the run in place (red dot, "failed", the error in the
  answer area) instead of only appending a step;
* a stream that closes with no answer renders "the stream ended before an answer
  arrived — the model may have timed out. Ask again.";
* a dead server reads "could not reach the server — it may have stopped or
  restarted" rather than "Failed to fetch";
* the elapsed timer clears when the answer lands, so the trace records the work,
  not the waiting.

Verified by killing the server mid-answer: the bubble ended at `dot done` /
`failed` / "⚠ could not reach the server", with no spinner left running.
`eval/verify_renderer.py` follows the new contract (chips must be clickable
buttons; `[Q1]` and `[42]` stay literal). 276 tests pass; documentation check clean.
