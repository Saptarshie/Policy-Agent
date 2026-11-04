"""Batching the decision model: speed without moving the verdicts.

Measured against the hosted decision model on 6 hand-built pairs:
  * one request per pair: 8.7s, 2029 input tokens
  * one request for all  : 1.5s,  810 input tokens
  * ...but the batch perturbed scores by up to 0.24 and flipped 2 of 6 decisions.

Hence two tiers: batch the clear pairs, re-ask the ambiguous ones on their own.
These tests pin the tier logic, not the model's numbers.
"""
import unittest

from app.contradictions.gate import CascadeGate, GateDecision


class RecordingRemote:
    """Scores by lookup; records every call so the tiers are visible."""

    def __init__(self, singles, batch_scores=None):
        self.singles = singles                 # (a, b) -> score
        self.batch_scores = batch_scores or {}  # index-in-batch -> score
        self.batch_calls = []
        self.single_calls = []

    def score(self, text_a, text_b, signals):
        self.single_calls.append((text_a, text_b))
        score = self.singles.get((text_a, text_b), 0.0)
        return GateDecision(score >= 0.5, score, list(signals or []),
                            decided_by="decision_model")

    def score_many(self, pairs):
        self.batch_calls.append(list(pairs))
        if not self.batch_scores:
            return None
        # a batch score of None means "no opinion", handled as a fallback
        return [self.batch_scores.get(i) for i in range(len(pairs))]


P = [(f"a{i}", f"b{i}") for i in range(6)]


class TestBatchTier(unittest.TestCase):
    def test_clear_pairs_cost_one_request(self):
        remote = RecordingRemote({}, batch_scores={0: 0.05, 1: 0.95, 2: 0.02,
                                                   3: 0.9, 4: 0.06, 5: 0.97})
        gate = CascadeGate(threshold=0.5, remote=remote, batch_size=6,
                           recheck_band=0.15)
        decisions = gate.evaluate_many(P)
        self.assertEqual(len(remote.batch_calls), 1)
        self.assertEqual(remote.single_calls, [],
                         "nothing near the threshold should be re-asked")
        self.assertEqual([d.passed for d in decisions],
                         [False, True, False, True, False, True])
        self.assertTrue(all(d.decided_by == "decision_model(batch)" for d in decisions))

    def test_borderline_pairs_are_reasked_alone(self):
        remote = RecordingRemote({P[2]: 0.62, P[4]: 0.41},
                                 batch_scores={0: 0.05, 1: 0.95, 2: 0.44,
                                               3: 0.9, 4: 0.58, 5: 0.97})
        gate = CascadeGate(threshold=0.5, remote=remote, batch_size=6,
                           recheck_band=0.15)
        decisions = gate.evaluate_many(P)
        self.assertEqual(remote.single_calls, [P[2], P[4]],
                         "only the two near-threshold pairs are re-asked")
        # the re-ask is what decides: 0.44 -> 0.62 passes, 0.58 -> 0.41 does not
        self.assertTrue(decisions[2].passed)
        self.assertFalse(decisions[4].passed)
        self.assertEqual(decisions[2].decided_by, "decision_model")

    def test_batch_failure_falls_back_to_per_pair(self):
        remote = RecordingRemote({}, batch_scores=None)   # score_many -> None
        gate = CascadeGate(threshold=0.5, remote=remote, batch_size=6)
        decisions = gate.evaluate_many(P)
        self.assertEqual(len(remote.single_calls), len(P))
        self.assertEqual(len(decisions), len(P))

    def test_batching_off_is_the_old_path(self):
        remote = RecordingRemote({}, batch_scores={0: 0.9})
        gate = CascadeGate(threshold=0.5, remote=remote, batch_size=1)
        gate.evaluate_many(P[:2])
        self.assertEqual(remote.batch_calls, [])
        self.assertEqual(len(remote.single_calls), 2)

    def test_batches_split_at_batch_size(self):
        remote = RecordingRemote({}, batch_scores={0: 0.02, 1: 0.97})
        gate = CascadeGate(threshold=0.5, remote=remote, batch_size=2,
                           recheck_band=0.0)
        gate.evaluate_many(P)
        self.assertEqual([len(c) for c in remote.batch_calls], [2, 2, 2])

    def test_no_remote_is_unavailable_not_clean(self):
        gate = CascadeGate(threshold=0.5, remote=None)
        decisions = gate.evaluate_many(P[:2])
        self.assertEqual([d.decided_by for d in decisions],
                         ["unavailable", "unavailable"])


if __name__ == "__main__":
    unittest.main()
