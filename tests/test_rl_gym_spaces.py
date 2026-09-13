"""Unit tests for src/rl_gym/spaces.py (P1.5).

Stdlib unittest, so no new dependency: `python -m unittest discover tests`. pytest
collects unittest classes unchanged if it is ever added to the environment.

tests/test_rl_gym.py is reserved for P1.10's rollout gate.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rl_gym.spaces import (  # noqa: E402
    build_action_mask,
    candidate_slot_validity,
    make_action_space,
    reject_action,
    truncate_candidates,
)

K = 8


def candidates(n, delta_start=0.0):
    """n (vid, vehplan, delta_cfv) tuples, ascending by delta_cfv as insertion returns them."""
    return [(f"vid{i}", f"plan{i}", delta_start + i) for i in range(n)]


class TestActionSpace(unittest.TestCase):

    def test_size_is_k_plus_one(self):
        self.assertEqual(make_action_space(K).n, K + 1)
        self.assertEqual(make_action_space(1).n, 2)

    def test_reject_is_the_last_slot(self):
        space = make_action_space(K)
        self.assertEqual(reject_action(K), K)
        self.assertTrue(space.contains(reject_action(K)))
        self.assertFalse(space.contains(K + 1))

    def test_rejects_bad_k(self):
        for bad in (0, -1, -8):
            with self.assertRaises(ValueError):
                make_action_space(bad)
        for bad in (1.5, "8", None, True):
            with self.assertRaises(TypeError):
                make_action_space(bad)


class TestTruncation(unittest.TestCase):

    def test_preserves_order_exactly(self):
        cands = candidates(5)
        self.assertEqual(truncate_candidates(cands, K), cands)

    def test_prefix_when_longer_than_k(self):
        cands = candidates(11)
        out = truncate_candidates(cands, K)
        self.assertEqual(len(out), K)
        self.assertEqual(out, cands[:K])
        # the dropped candidates are the most expensive ones, never the cheapest
        self.assertEqual([c[2] for c in out], [float(i) for i in range(K)])

    def test_k_larger_than_candidate_count_leaves_list_unchanged(self):
        cands = candidates(3)
        self.assertEqual(truncate_candidates(cands, K), cands)
        self.assertEqual(len(truncate_candidates(cands, 100)), 3)

    def test_exactly_k_candidates(self):
        cands = candidates(K)
        self.assertEqual(truncate_candidates(cands, K), cands)

    def test_empty_list(self):
        self.assertEqual(truncate_candidates([], K), [])

    def test_does_not_sort_or_tie_break(self):
        # tied delta_cfv in an order no sort would produce: upstream order within ties comes
        # from a backwards Dijkstra, not from vid, so any re-sort breaks the byte gate (D3)
        tied = [("vid7", "p7", 5.0), ("vid2", "p2", 5.0), ("vid9", "p9", 5.0)]
        self.assertEqual(truncate_candidates(tied, K), tied)
        # descending input must come back descending -- proof no sort is applied
        descending = [("a", "p", 9.0), ("b", "p", 4.0), ("c", "p", 1.0)]
        self.assertEqual(truncate_candidates(descending, K), descending)

    def test_returns_a_new_list(self):
        cands = candidates(3)
        out = truncate_candidates(cands, K)
        self.assertIsNot(out, cands)
        out.append(("extra", "plan", 99.0))
        self.assertEqual(len(cands), 3, "mutating the result must not touch the input")

    def test_rejects_bad_k(self):
        with self.assertRaises(ValueError):
            truncate_candidates(candidates(3), 0)
        with self.assertRaises(TypeError):
            truncate_candidates(candidates(3), 2.5)


class TestSlotValidity(unittest.TestCase):

    def test_fewer_than_k(self):
        v = candidate_slot_validity(3, K)
        self.assertEqual(v.dtype, np.bool_)
        self.assertEqual(len(v), K)
        np.testing.assert_array_equal(v, [True] * 3 + [False] * 5)

    def test_exactly_k(self):
        np.testing.assert_array_equal(candidate_slot_validity(K, K), [True] * K)

    def test_more_than_k_saturates(self):
        np.testing.assert_array_equal(candidate_slot_validity(11, K), [True] * K)
        np.testing.assert_array_equal(candidate_slot_validity(1000, K), [True] * K)

    def test_zero_candidates(self):
        np.testing.assert_array_equal(candidate_slot_validity(0, K), [False] * K)

    def test_agrees_with_truncation_length(self):
        # the invariant P1.6 depends on: exactly the slots truncation fills are valid
        for n in range(0, 15):
            with self.subTest(n=n):
                kept = truncate_candidates(candidates(n), K)
                v = candidate_slot_validity(n, K)
                self.assertEqual(int(v.sum()), len(kept))
                self.assertTrue(v[:len(kept)].all())
                self.assertFalse(v[len(kept):].any())

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            candidate_slot_validity(-1, K)
        with self.assertRaises(ValueError):
            candidate_slot_validity(3, 0)
        with self.assertRaises(TypeError):
            candidate_slot_validity(3.0, K)


class TestActionMask(unittest.TestCase):

    def test_shape_and_dtype(self):
        m = build_action_mask(3, K)
        self.assertEqual(m.shape, (K + 1,))
        self.assertEqual(m.dtype, np.bool_)

    def test_masks_only_the_surplus_slots(self):
        m = build_action_mask(3, K)
        np.testing.assert_array_equal(m, [True] * 3 + [False] * 5 + [True])

    def test_full_candidate_list(self):
        np.testing.assert_array_equal(build_action_mask(K, K), [True] * (K + 1))

    def test_more_candidates_than_k(self):
        np.testing.assert_array_equal(build_action_mask(11, K), [True] * (K + 1))

    def test_reject_always_legal(self):
        for n in range(0, 15):
            with self.subTest(n=n):
                self.assertTrue(build_action_mask(n, K)[reject_action(K)])

    def test_never_all_false(self):
        for n in range(0, 15):
            for k in (1, 2, K, 20):
                with self.subTest(n=n, k=k):
                    self.assertTrue(build_action_mask(n, k).any())

    def test_zero_candidates_leaves_reject_as_only_option(self):
        m = build_action_mask(0, K)
        self.assertEqual(int(m.sum()), 1)
        self.assertTrue(m[reject_action(K)])

    def test_mask_matches_slot_validity(self):
        # trap 12: the mask's candidate half must be exactly candidate_slot_validity
        for n in range(0, 15):
            with self.subTest(n=n):
                np.testing.assert_array_equal(
                    build_action_mask(n, K)[:K], candidate_slot_validity(n, K)
                )

    def test_every_unmasked_candidate_slot_is_a_legal_commit_index(self):
        # an unmasked slot k < K must be a valid index into the truncated candidate list,
        # which is what RLPoolingIRSOnly.commit_assignment_choice asserts
        for n in range(0, 15):
            kept = truncate_candidates(candidates(n), K)
            m = build_action_mask(n, K)
            for k in range(K):
                if m[k]:
                    with self.subTest(n=n, k=k):
                        self.assertLess(k, len(kept))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            build_action_mask(3, 0)
        with self.assertRaises(ValueError):
            build_action_mask(-1, K)

    def test_mask_length_matches_action_space_size(self):
        # MaskablePPO requires one mask entry per action; a mismatch here is the kind of
        # off-by-one that would silently shift which slot the policy thinks it is choosing
        for k in (1, 2, K, 20):
            with self.subTest(k=k):
                self.assertEqual(len(build_action_mask(3, k)), make_action_space(k).n)

    def test_observed_candidate_lengths_from_the_reference_day(self):
        # the real distribution P1.3 measured: min 1, max 11, with 13 lists over K=8
        for n in (1, 4, 5, 8, 9, 11):
            with self.subTest(n_candidates=n):
                m = build_action_mask(n, K)
                self.assertEqual(int(m[:K].sum()), min(n, K))
                self.assertTrue(m[reject_action(K)])
                self.assertEqual(len(truncate_candidates(candidates(n), K)), min(n, K))


if __name__ == "__main__":
    unittest.main(verbosity=2)
