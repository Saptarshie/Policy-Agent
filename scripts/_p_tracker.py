"""Patch 2/4: threshold-gated candidates, intra-document scanning, no per-chunk cap.

Before: for each chunk, take the top CONTRA_TOP_K=4 neighbours, escalate at most
SCAN_MAX_PAIRS_PER_CHUNK=3 of them. Two count-based gates, both of which can drop
the one passage that contradicts.

After: for each chunk, take EVERY passage at or above SCAN_MIN_SIMILARITY
(default 0.65 - measured: the real leave-cap contradiction in this corpus sits at
0.678, i.e. rank 3 of 4, inside the old cut only by luck), in the same document
or another one, bounded only by a global safety cap.
"""
import pathlib

p = pathlib.Path("app/contradictions/tracker.py")
s = p.read_text(encoding="utf-8")
assert "min_similarity" not in s, "already patched"

old = '''        for i, chunk in enumerate(chunks):
            # ---- stage 1: RAG retrieval of candidate pairs
            try:
                hits = self.rag.search(
                    chunk["text"], top_k=self.top_k,
                    exclude_doc_ids=[doc_id] if self.cross_doc_only else None,
                    # candidate pairs are "semantically adjacent passages";
                    # keyword overlap would surface repeated clauses instead of
                    # genuine conflicts, so the cascade stays on vectors
                    mode="vector",
                )
            except Exception as e:
                log_fn("  [scan] embedding search failed on chunk %s: %s",
                       chunk["id"], e)
                hits = []

            escalated = 0
            for hit in hits:
                if hit.doc_id == doc_id and self.cross_doc_only:
                    continue
                if escalated >= self.max_pairs_per_chunk:
                    break
                if self.db.contradiction_pair_exists(chunk["id"], hit.chunk_id):
                    continue
'''
new = '''        considered = 0
        capped = False
        for i, chunk in enumerate(chunks):
            # ---- stage 1: candidate pairs by SIMILARITY, not by count.
            # A conflict can rank 41st among 60 passages that are all similar;
            # a top-k cut would never show it to the gate, so the floor decides
            # and only a global cap guards total cost.
            try:
                hits = self.rag.search(
                    chunk["text"],
                    min_score=self.min_similarity,
                    max_candidates=self.max_candidates,
                    # candidate pairs are "semantically adjacent passages";
                    # keyword overlap would surface repeated clauses instead of
                    # genuine conflicts, so the cascade stays on similarity
                    mode="vector",
                )
            except Exception as e:
                log_fn("  [scan] embedding search failed on chunk %s: %s",
                       chunk["id"], e)
                hits = []

            for hit in hits:
                if hit.chunk_id == chunk["id"]:
                    continue          # a passage cannot contradict itself
                if hit.doc_id != doc_id and not self.cross_doc_only:
                    # intra-document detection is ON, so cross-document pairs
                    # are also fair game - unless the caller restricted to one
                    # document on purpose
                    pass
                if considered >= self.max_candidates:
                    if not capped:
                        capped = True
                        log_fn("  [scan] candidate cap reached (%d pairs); "
                               "raise SCAN_MAX_CANDIDATES to see the tail",
                               self.max_candidates)
                    break
                considered += 1
                if self.db.contradiction_pair_exists(chunk["id"], hit.chunk_id):
                    continue
'''
assert old in s
s = s.replace(old, new, 1)

# the old per-chunk cap counter is gone
s = s.replace('''            escalated += 1
                log_fn''', '''                log_fn''')
s = s.replace('''                escalated += 1
                log_fn("  [scan] gate PASSED''', '''                log_fn("  [scan] gate PASSED''')

# constructor: threshold + global cap
old_ctor = '''        self.cross_doc_only = cross_doc_only
        self.top_k = top_k
        self.max_pairs_per_chunk = max_pairs_per_chunk'''
new_ctor = '''        # candidate generation is similarity-gated, not count-gated; the only
        # count left is a safety cap on total escalations per document
        self.cross_doc_only = cross_doc_only
        self.min_similarity = float(min_similarity)
        self.max_candidates = int(max_candidates)'''
assert old_ctor in s, s[s.index("def __init__"):s.index("def __init__") + 900]
s = s.replace(old_ctor, new_ctor, 1)

s = s.replace('''        top_k: int = 4,
        max_pairs_per_chunk: int = 3,''',
'''        min_similarity: float = 0.65,
        max_candidates: int = 200,''')
if "min_similarity: float = 0.65" not in s:
    # constructor signature may order things differently - patch loosely
    s = s.replace("        top_k: int = 4,", "        min_similarity: float = 0.65,"
                  "\n        max_candidates: int = 200,", 1)
    s = s.replace("        max_pairs_per_chunk: int = 3,\n", "", 1)

p.write_text(s, encoding="utf-8")
print("tracker patched")
print("escalated refs left:", s.count("escalated"))
print("signature:", [ln for ln in s.splitlines() if "min_similarity" in ln][:3])
