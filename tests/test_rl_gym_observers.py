"""Unit tests for src/rl_gym/observers.py (P1.6).

Stdlib unittest: `python -m unittest discover tests`. Stub doubles rather than a real
simulation, so these stay fast; the full-run checks (no NaN/inf across 436 real decisions,
slot 0 always the argmin) live in P1.6's verification script and later in P1.10.
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.misc.globals import (  # noqa: E402
    G_OP_MAX_WT,
    G_SIM_END_TIME,
    G_SIM_START_TIME,
    VRL_STATES,
)
from src.rl_gym.observers import (  # noqa: E402
    CandidateObserver,
    GlobalStateObserver,
    build_observation_layout,
    signed_log1p,
)
from src.rl_gym.policies import greedy_action, greedy_choice  # noqa: E402
from src.rl_gym.spaces import build_action_mask, candidate_slot_validity  # noqa: E402

K = 8
MAX_WT = 1500.0
START, END = 25200.0, 70000.0

SCENARIO = {G_OP_MAX_WT: MAX_WT, G_SIM_START_TIME: START, G_SIM_END_TIME: END}


class StubPlan:
    def __init__(self, pax_info):
        self.pax_info = pax_info


class StubPrq:
    def __init__(self, rq_time):
        self.rq_time = rq_time


class StubCtx:
    """Stands in for PendingDecision."""

    def __init__(self, candidates, rq_time=30000.0, sim_time=30000.0, rid_struct=7):
        self.candidates = candidates
        self.prq = StubPrq(rq_time)
        self.sim_time = sim_time
        self.rid_struct = rid_struct


class StubVehicle:
    def __init__(self, status):
        self.status = status


class StubFleetCtrl:
    def __init__(self, statuses):
        self.sim_vehicles = [StubVehicle(s) for s in statuses]


def make_candidates(waits, cfvs, rq_time=30000.0, rid_struct=7):
    """Candidates whose plans quote the given waits (seconds) and insertion costs."""
    return [
        (f"vid{i}", StubPlan({rid_struct: [rq_time + w, rq_time + w + 600.0]}), c)
        for i, (w, c) in enumerate(zip(waits, cfvs))
    ]


class TestSignedLog1p(unittest.TestCase):

    def test_zero_maps_to_zero(self):
        self.assertEqual(float(signed_log1p(0.0)), 0.0)

    def test_sign_preserved(self):
        self.assertGreater(float(signed_log1p(5.0)), 0.0)
        self.assertLess(float(signed_log1p(-5.0)), 0.0)

    def test_is_odd(self):
        for x in (0.5, 3.0, 1e4):
            self.assertAlmostEqual(float(signed_log1p(-x)), -float(signed_log1p(x)))

    def test_strictly_monotonic(self):
        # P1.10 takes the argmin over this transform and must land on the same candidate as an
        # argmin over raw delta_cfv, which requires monotonicity
        xs = np.array([-1e5, -1e3, -7.5, -1.0, 0.0, 1.0, 7.5, 1e3, 1e5])
        ys = signed_log1p(xs)
        self.assertTrue(np.all(np.diff(ys) > 0))

    def test_argmin_agrees_with_raw(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            raw = rng.normal(0, 500, size=8)
            self.assertEqual(int(np.argmin(signed_log1p(raw))), int(np.argmin(raw)))

    def test_compresses_range(self):
        self.assertLess(abs(float(signed_log1p(1e6))), 20.0)


class TestCandidateObserver(unittest.TestCase):

    def setUp(self):
        self.obs = CandidateObserver(K, SCENARIO)

    def test_output_length_is_three_k(self):
        self.assertEqual(self.obs.output_length, 3 * K)
        self.assertEqual(CandidateObserver(3, SCENARIO).output_length, 9)

    def test_shape_and_dtype(self):
        ctx = StubCtx(make_candidates([100.0, 200.0], [1.0, 2.0]))
        vec = self.obs.observe_vector(None, ctx)
        self.assertEqual(vec.shape, (3 * K,))
        self.assertEqual(vec.dtype, np.float32)

    def test_is_valid_comes_from_shared_function(self):
        for n in range(0, 12):
            with self.subTest(n=n):
                ctx = StubCtx(make_candidates([100.0] * n, list(range(n))))
                parts = self.obs.observe(None, ctx)
                np.testing.assert_array_equal(
                    parts["is_valid"], candidate_slot_validity(n, K).astype(np.float32)
                )

    def test_padded_slots_are_all_zero(self):
        ctx = StubCtx(make_candidates([100.0, 200.0], [5.0, 9.0]))
        parts = self.obs.observe(None, ctx)
        for name in ("is_valid", "delta_cfv", "offered_wait"):
            np.testing.assert_array_equal(parts[name][2:], np.zeros(K - 2, dtype=np.float32))

    def test_delta_cfv_transformed_not_raw(self):
        ctx = StubCtx(make_candidates([100.0], [100.0]))
        parts = self.obs.observe(None, ctx)
        self.assertAlmostEqual(float(parts["delta_cfv"][0]), math.log1p(100.0), places=5)

    def test_negative_delta_cfv(self):
        ctx = StubCtx(make_candidates([100.0], [-100.0]))
        parts = self.obs.observe(None, ctx)
        self.assertAlmostEqual(float(parts["delta_cfv"][0]), -math.log1p(100.0), places=5)

    def test_offered_wait_matches_the_operator_formula(self):
        # exactly what _create_user_offer quotes: pax_info[rid][0] - prq.rq_time
        rq_time = 30000.0
        ctx = StubCtx(make_candidates([300.0, 750.0], [1.0, 2.0], rq_time=rq_time),
                      rq_time=rq_time)
        parts = self.obs.observe(None, ctx)
        self.assertAlmostEqual(float(parts["offered_wait"][0]), 300.0 / MAX_WT, places=6)
        self.assertAlmostEqual(float(parts["offered_wait"][1]), 750.0 / MAX_WT, places=6)

    def test_normalizer_read_from_scenario_parameters_not_hardcoded(self):
        # trap 8: the two constant configs disagree on op_max_wait_time (1500 vs 4200)
        other = dict(SCENARIO, **{G_OP_MAX_WT: 4200.0})
        ctx = StubCtx(make_candidates([600.0], [1.0]))
        self.assertAlmostEqual(
            float(CandidateObserver(K, other).observe(None, ctx)["offered_wait"][0]),
            600.0 / 4200.0, places=6)

    def test_truncates_to_k_preserving_order(self):
        cfvs = [float(i) for i in range(11)]
        ctx = StubCtx(make_candidates([100.0] * 11, cfvs))
        parts = self.obs.observe(None, ctx)
        np.testing.assert_allclose(
            parts["delta_cfv"], signed_log1p(np.array(cfvs[:K])).astype(np.float32), atol=1e-6)
        np.testing.assert_array_equal(parts["is_valid"], np.ones(K, dtype=np.float32))

    def test_slot_zero_is_the_minimum_for_a_sorted_list(self):
        cfvs = [1.0, 4.0, 9.0, 50.0]
        ctx = StubCtx(make_candidates([100.0] * 4, cfvs))
        d = self.obs.observe(None, ctx)["delta_cfv"]
        valid = self.obs.observe(None, ctx)["is_valid"] > 0
        self.assertEqual(int(np.argmin(d[valid])), 0)

    def test_missing_pax_info_yields_zero_wait_not_a_crash(self):
        # defensive: a plan that does not carry this rid must not raise
        ctx = StubCtx([("vid0", StubPlan({}), 1.0)])
        parts = self.obs.observe(None, ctx)
        self.assertEqual(float(parts["offered_wait"][0]), 0.0)
        self.assertEqual(float(parts["is_valid"][0]), 1.0)

    def test_empty_candidate_list(self):
        parts = self.obs.observe(None, StubCtx([]))
        for name in ("is_valid", "delta_cfv", "offered_wait"):
            np.testing.assert_array_equal(parts[name], np.zeros(K, dtype=np.float32))

    def test_rejects_bad_config(self):
        with self.assertRaises(ValueError):
            CandidateObserver(0, SCENARIO)
        with self.assertRaises(ValueError):
            CandidateObserver(K, dict(SCENARIO, **{G_OP_MAX_WT: 0.0}))
        with self.assertRaises(KeyError):
            CandidateObserver(K, {})

    def test_no_nan_or_inf(self):
        ctx = StubCtx(make_candidates([0.0, 1500.0, 3000.0], [-1e6, 0.0, 1e6]))
        vec = self.obs.observe_vector(None, ctx)
        self.assertTrue(np.all(np.isfinite(vec)))


class TestGlobalStateObserver(unittest.TestCase):

    def setUp(self):
        self.obs = GlobalStateObserver(SCENARIO)

    def test_output_length_is_four(self):
        self.assertEqual(self.obs.output_length, 4)

    def test_shape_and_dtype(self):
        fc = StubFleetCtrl([VRL_STATES.IDLE] * 3)
        vec = self.obs.observe_vector(fc, StubCtx([], sim_time=START))
        self.assertEqual(vec.shape, (4,))
        self.assertEqual(vec.dtype, np.float32)

    def test_time_of_day_is_absolute_clock_not_episode_fraction(self):
        fc = StubFleetCtrl([VRL_STATES.IDLE])
        # 06:00 -> t_of_day = 0.25 -> sin = 1, cos = 0
        parts = self.obs.observe(fc, StubCtx([], sim_time=21600.0))
        self.assertAlmostEqual(float(parts["time_sin"]), 1.0, places=5)
        self.assertAlmostEqual(float(parts["time_cos"]), 0.0, places=5)
        # midnight -> sin = 0, cos = 1
        parts = self.obs.observe(fc, StubCtx([], sim_time=0.0))
        self.assertAlmostEqual(float(parts["time_sin"]), 0.0, places=5)
        self.assertAlmostEqual(float(parts["time_cos"]), 1.0, places=5)

    def test_time_of_day_wraps_at_midnight(self):
        fc = StubFleetCtrl([VRL_STATES.IDLE])
        a = self.obs.observe(fc, StubCtx([], sim_time=3600.0))
        b = self.obs.observe(fc, StubCtx([], sim_time=3600.0 + 86400.0))
        self.assertAlmostEqual(float(a["time_sin"]), float(b["time_sin"]), places=6)
        self.assertAlmostEqual(float(a["time_cos"]), float(b["time_cos"]), places=6)

    def test_sin_cos_differ_from_episode_progress(self):
        # the two are distinct quantities; this is the check that they were not conflated
        fc = StubFleetCtrl([VRL_STATES.IDLE])
        parts = self.obs.observe(fc, StubCtx([], sim_time=START))
        self.assertAlmostEqual(float(parts["day_progress"]), 0.0, places=6)
        self.assertNotAlmostEqual(float(parts["time_sin"]), 0.0, places=3)

    def test_episode_progress_endpoints_and_midpoint(self):
        fc = StubFleetCtrl([VRL_STATES.IDLE])
        for t, want in ((START, 0.0), ((START + END) / 2, 0.5), (END, 1.0)):
            with self.subTest(sim_time=t):
                parts = self.obs.observe(fc, StubCtx([], sim_time=t))
                self.assertAlmostEqual(float(parts["day_progress"]), want, places=6)

    def test_episode_progress_clipped_past_end_time(self):
        # record_remaining_assignments advances past end_time
        fc = StubFleetCtrl([VRL_STATES.IDLE])
        parts = self.obs.observe(fc, StubCtx([], sim_time=END + 14400.0))
        self.assertEqual(float(parts["day_progress"]), 1.0)
        parts = self.obs.observe(fc, StubCtx([], sim_time=START - 100.0))
        self.assertEqual(float(parts["day_progress"]), 0.0)

    def test_idle_fraction_excludes_out_of_service_from_denominator(self):
        fc = StubFleetCtrl([VRL_STATES.IDLE, VRL_STATES.IDLE,
                            VRL_STATES.ROUTE, VRL_STATES.ROUTE,
                            VRL_STATES.OUT_OF_SERVICE, VRL_STATES.OUT_OF_SERVICE])
        parts = self.obs.observe(fc, StubCtx([], sim_time=START))
        self.assertAlmostEqual(float(parts["idle_fraction"]), 0.5, places=6)

    def test_only_idle_counts_as_idle(self):
        # WAITING, PLANNED_STOP and REPO_TARGET are active but not idle (§5.1)
        for status in (VRL_STATES.WAITING, VRL_STATES.PLANNED_STOP, VRL_STATES.REPO_TARGET):
            with self.subTest(status=status):
                fc = StubFleetCtrl([VRL_STATES.IDLE, status])
                parts = self.obs.observe(fc, StubCtx([], sim_time=START))
                self.assertAlmostEqual(float(parts["idle_fraction"]), 0.5, places=6)

    def test_all_out_of_service_guards_the_divide(self):
        fc = StubFleetCtrl([VRL_STATES.OUT_OF_SERVICE] * 4)
        parts = self.obs.observe(fc, StubCtx([], sim_time=START))
        self.assertEqual(float(parts["idle_fraction"]), 0.0)

    def test_empty_fleet_guards_the_divide(self):
        parts = self.obs.observe(StubFleetCtrl([]), StubCtx([], sim_time=START))
        self.assertEqual(float(parts["idle_fraction"]), 0.0)

    def test_all_idle_and_none_idle(self):
        fc = StubFleetCtrl([VRL_STATES.IDLE] * 3)
        self.assertAlmostEqual(
            float(self.obs.observe(fc, StubCtx([], sim_time=START))["idle_fraction"]), 1.0)
        fc = StubFleetCtrl([VRL_STATES.ROUTE] * 3)
        self.assertAlmostEqual(
            float(self.obs.observe(fc, StubCtx([], sim_time=START))["idle_fraction"]), 0.0)

    def test_enum_status_never_compared_against_int(self):
        # the upstream `== 5` bug: a vehicle whose status is the enum must still be counted
        # out of service. If the observer used `== 5` this would report 1.0, not 0.0.
        fc = StubFleetCtrl([VRL_STATES.OUT_OF_SERVICE, VRL_STATES.IDLE])
        self.assertFalse(VRL_STATES.OUT_OF_SERVICE == 5)  # documents why
        parts = self.obs.observe(fc, StubCtx([], sim_time=START))
        self.assertAlmostEqual(float(parts["idle_fraction"]), 1.0, places=6)

    def test_rejects_bad_config(self):
        with self.assertRaises(ValueError):
            GlobalStateObserver({G_SIM_START_TIME: 70000.0, G_SIM_END_TIME: 25200.0})
        with self.assertRaises(ValueError):
            GlobalStateObserver({G_SIM_START_TIME: 100.0, G_SIM_END_TIME: 100.0})
        with self.assertRaises(KeyError):
            GlobalStateObserver({})


class TestPaddingMustNotCompeteInArgmin(unittest.TestCase):
    """Pins why the scripted greedy driver must restrict its argmin to `is_valid` slots.

    On the reference scenario real `delta_cfv` is negative, so 0.0 padding is the largest value
    in the slot and an unrestricted argmin coincidentally also returns slot 0 -- which means
    P1.10's byte-for-byte gate would pass even with the restriction removed. The sign is a
    property of this objective function, not of the design, and could change.

    These tests supply the counter-example the real scenario does not: positive `delta_cfv` in
    the valid slots, where an unrestricted argmin returns a padded slot and is wrong. If the
    restriction is ever dropped, `test_unrestricted_argmin_is_wrong_under_positive_costs`
    documents the failure and `test_restricted_argmin_is_correct_under_positive_costs` is the
    behaviour that must hold.
    """

    def setUp(self):
        self.obs = CandidateObserver(K, SCENARIO)
        # two valid candidates, both POSITIVE, sorted ascending as insertion returns them
        self.ctx = StubCtx(make_candidates([100.0, 200.0], [3.0, 7.0]))
        parts = self.obs.observe(None, self.ctx)
        self.delta = parts["delta_cfv"]
        self.valid = parts["is_valid"] > 0

    def test_padding_is_the_minimum_when_real_costs_are_positive(self):
        # the premise: with positive costs, a padded 0.0 slot is strictly below every real slot
        self.assertTrue(np.all(self.delta[self.valid] > 0.0))
        self.assertTrue(np.all(self.delta[~self.valid] == 0.0))
        self.assertLess(float(self.delta[~self.valid].min()), float(self.delta[self.valid].min()))

    def test_unrestricted_argmin_is_wrong_under_positive_costs(self):
        # what a driver that forgot the restriction would pick: a padded slot, not slot 0
        picked = int(np.argmin(self.delta))
        self.assertFalse(bool(self.valid[picked]),
                         "unrestricted argmin should land on a padded slot here")
        self.assertNotEqual(picked, 0)

    def test_restricted_argmin_is_correct_under_positive_costs(self):
        # the required behaviour: argmin over valid slots only, which is slot 0
        valid_idx = np.flatnonzero(self.valid)
        picked = int(valid_idx[np.argmin(self.delta[valid_idx])])
        self.assertEqual(picked, 0)

    def test_restricted_argmin_is_slot_zero_for_any_positive_sorted_list(self):
        for n in range(1, K + 1):
            with self.subTest(n_candidates=n):
                cfvs = [float(i + 1) * 10.0 for i in range(n)]
                ctx = StubCtx(make_candidates([100.0] * n, cfvs))
                parts = self.obs.observe(None, ctx)
                valid_idx = np.flatnonzero(parts["is_valid"] > 0)
                picked = int(valid_idx[np.argmin(parts["delta_cfv"][valid_idx])])
                self.assertEqual(picked, 0)
                if n < K:  # a padded slot exists, so the unrestricted form must disagree
                    self.assertNotEqual(int(np.argmin(parts["delta_cfv"])), picked)

    def test_restricted_argmin_also_correct_under_negative_costs(self):
        # the real scenario's sign: restriction must be harmless, not just necessary
        cfvs = [-500.0, -200.0, -10.0]
        ctx = StubCtx(make_candidates([100.0] * 3, cfvs))
        parts = self.obs.observe(None, ctx)
        valid_idx = np.flatnonzero(parts["is_valid"] > 0)
        self.assertEqual(int(valid_idx[np.argmin(parts["delta_cfv"][valid_idx])]), 0)
        self.assertEqual(int(np.argmin(parts["delta_cfv"])), 0)  # coincides here

    def test_single_valid_slot_with_positive_cost(self):
        ctx = StubCtx(make_candidates([100.0], [42.0]))
        parts = self.obs.observe(None, ctx)
        valid_idx = np.flatnonzero(parts["is_valid"] > 0)
        self.assertEqual(int(valid_idx[np.argmin(parts["delta_cfv"][valid_idx])]), 0)


class TestGreedyActionHelper(unittest.TestCase):
    """`policies.greedy_action` — the helper P1.10 drives with and P2.7's baseline imports.

    These are the tests that fail if the `is_valid` restriction is removed. The byte-for-byte
    gate cannot catch that, because real `delta_cfv` on this scenario is negative and an
    unrestricted argmin coincidentally agrees; the positive-cost cases below are where the two
    diverge.
    """

    def setUp(self):
        self.obs_builder = CandidateObserver(K, SCENARIO)
        self.glob = GlobalStateObserver(SCENARIO)
        # the same map the env publishes, built from the same specs in the same order (P2.8)
        self.layout = build_observation_layout([self.obs_builder, self.glob])

    def _full_obs(self, ctx):
        fc = StubFleetCtrl([VRL_STATES.IDLE])
        return np.concatenate([self.obs_builder.observe_vector(None, ctx),
                               self.glob.observe_vector(fc, ctx)])

    def test_picks_slot_zero_on_a_sorted_negative_list(self):
        ctx = StubCtx(make_candidates([100.0] * 4, [-500.0, -200.0, -50.0, -1.0]))
        obs = self._full_obs(ctx)
        mask = build_action_mask(4, K)
        self.assertEqual(greedy_action(obs, mask, self.layout), 0)

    def test_picks_slot_zero_on_a_sorted_positive_list(self):
        # the case that separates restricted from unrestricted
        ctx = StubCtx(make_candidates([100.0] * 3, [3.0, 7.0, 11.0]))
        obs = self._full_obs(ctx)
        mask = build_action_mask(3, K)
        self.assertEqual(greedy_action(obs, mask, self.layout), 0)

    def test_unrestricted_argmin_would_disagree_under_positive_costs(self):
        # documents precisely what a dropped restriction would do
        ctx = StubCtx(make_candidates([100.0] * 3, [3.0, 7.0, 11.0]))
        obs = self._full_obs(ctx)
        mask = build_action_mask(3, K)
        naive = int(np.argmin(obs[self.layout["delta_cfv"]]))   # no is_valid restriction
        self.assertNotEqual(naive, greedy_action(obs, mask, self.layout))
        self.assertFalse(mask[naive], "the naive pick lands on a padded slot")

    def test_selects_by_value_not_by_assuming_index_zero(self):
        # an unsorted list must still give the true minimum, so the helper does not silently
        # depend on insertion.py's ordering
        ctx = StubCtx(make_candidates([100.0] * 4, [9.0, 2.0, 40.0, 5.0]))
        obs = self._full_obs(ctx)
        mask = build_action_mask(4, K)
        self.assertEqual(greedy_action(obs, mask, self.layout), 1)

    def test_rejects_when_no_candidate_is_valid(self):
        ctx = StubCtx([])
        obs = self._full_obs(ctx)
        mask = build_action_mask(0, K)
        self.assertEqual(greedy_action(obs, mask, self.layout), K)

    def test_returned_action_is_always_legal_under_the_mask(self):
        for n in range(0, 12):
            with self.subTest(n_candidates=n):
                ctx = StubCtx(make_candidates([100.0] * n, [float(i) for i in range(n)]))
                obs = self._full_obs(ctx)
                mask = build_action_mask(n, K)
                a = greedy_action(obs, mask, self.layout)
                self.assertTrue(mask[a], f"greedy returned masked action {a}")

    def test_greedy_choice_maps_reject_to_none(self):
        ctx = StubCtx([])
        obs = self._full_obs(ctx)
        self.assertIsNone(greedy_choice(obs, build_action_mask(0, K), self.layout))
        ctx = StubCtx(make_candidates([100.0], [1.0]))
        obs = self._full_obs(ctx)
        self.assertEqual(greedy_choice(obs, build_action_mask(1, K), self.layout), 0)

    def test_a_wrong_length_mask_raises(self):
        ctx = StubCtx(make_candidates([100.0], [1.0]))
        obs = self._full_obs(ctx)
        with self.assertRaises(ValueError):
            greedy_action(obs, build_action_mask(1, K)[:-1], self.layout)
        with self.assertRaises(ValueError):
            greedy_action(obs, np.ones(K + 2, dtype=bool), self.layout)

    def test_an_observation_that_does_not_cover_the_layout_raises(self):
        ctx = StubCtx(make_candidates([100.0], [1.0]))
        obs = self._full_obs(ctx)
        with self.assertRaises(ValueError):
            greedy_action(obs[:K], build_action_mask(1, K), self.layout)

    def test_the_total_length_is_deliberately_no_longer_the_contract(self):
        # P2.8 removes the hardcoded (3K + 4,) check: the layout's slices are what must be
        # covered, so trailing features greedy does not read may be absent. This is the point
        # of the item -- P2.9-P2.11 widen the vector and greedy keeps working untouched.
        ctx = StubCtx(make_candidates([100.0], [1.0]))
        obs = self._full_obs(ctx)
        trailing_dropped = obs[:self.layout["offered_wait"].stop]
        self.assertEqual(greedy_action(trailing_dropped, build_action_mask(1, K), self.layout),
                         greedy_action(obs, build_action_mask(1, K), self.layout))

    def test_reads_delta_cfv_through_the_layout_not_a_hardcoded_slice(self):
        # the layout is now the contract; if CandidateObserver reorders its features the map
        # follows, and greedy still selects on delta_cfv rather than on is_valid or
        # offered_wait
        ctx = StubCtx(make_candidates([100.0] * 3, [5.0, 1.0, 9.0]))
        parts = self.obs_builder.observe(None, ctx)
        obs = self._full_obs(ctx)
        np.testing.assert_allclose(obs[self.layout["delta_cfv"]], parts["delta_cfv"], atol=0)
        np.testing.assert_allclose(obs[self.layout["is_valid"]], parts["is_valid"], atol=0)
        self.assertEqual(greedy_action(obs, build_action_mask(3, K), self.layout), 1)


class TestCombinedLength(unittest.TestCase):

    def test_sum_of_observer_lengths_is_three_k_plus_four(self):
        total = CandidateObserver(K, SCENARIO).output_length + \
            GlobalStateObserver(SCENARIO).output_length
        self.assertEqual(total, 3 * K + 4)
        self.assertEqual(total, 28)

    def test_scales_with_k_without_hardcoding(self):
        for k in (1, 4, 8, 16):
            with self.subTest(k=k):
                total = CandidateObserver(k, SCENARIO).output_length + \
                    GlobalStateObserver(SCENARIO).output_length
                self.assertEqual(total, 3 * k + 4)

    def test_concatenated_vector_shape_and_dtype(self):
        cand = CandidateObserver(K, SCENARIO)
        glob = GlobalStateObserver(SCENARIO)
        ctx = StubCtx(make_candidates([100.0, 200.0], [1.0, 2.0]))
        fc = StubFleetCtrl([VRL_STATES.IDLE, VRL_STATES.ROUTE])
        vec = np.concatenate([cand.observe_vector(fc, ctx), glob.observe_vector(fc, ctx)])
        self.assertEqual(vec.shape, (3 * K + 4,))
        self.assertEqual(vec.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(vec)))


class TestObservationLayout(unittest.TestCase):
    """The name -> slice map (P2.8), built from the static specs."""

    def setUp(self):
        self.cand = CandidateObserver(K, SCENARIO)
        self.glob = GlobalStateObserver(SCENARIO)
        self.observers = [self.cand, self.glob]
        self.layout = build_observation_layout(self.observers)
        self.obs_len = sum(o.output_length for o in self.observers)

    def test_slices_tile_the_vector_with_no_gap_or_overlap(self):
        slices = list(self.layout.values())
        self.assertEqual(slices[0].start, 0)
        self.assertEqual(slices[-1].stop, self.obs_len)
        for a, b in zip(slices, slices[1:]):
            self.assertEqual(a.stop, b.start, f"{a} then {b} leaves a gap or overlaps")
        covered = sum(sl.stop - sl.start for sl in slices)
        self.assertEqual(covered, self.obs_len)

    def test_every_index_is_covered_exactly_once(self):
        seen = [0] * self.obs_len
        for sl in self.layout.values():
            for i in range(sl.start, sl.stop):
                seen[i] += 1
        self.assertEqual(seen, [1] * self.obs_len)

    def test_names_and_order_match_the_declarations(self):
        declared = [n for o in self.observers for n, _ in o.feature_spec()]
        self.assertEqual(list(self.layout), declared)

    def test_phase_one_feature_positions(self):
        self.assertEqual(self.layout["is_valid"], slice(0, K))
        self.assertEqual(self.layout["delta_cfv"], slice(K, 2 * K))
        self.assertEqual(self.layout["offered_wait"], slice(2 * K, 3 * K))
        self.assertEqual(self.layout["idle_fraction"], slice(3 * K + 3, 3 * K + 4))

    def test_output_length_is_derived_from_the_spec(self):
        for o in self.observers:
            self.assertEqual(o.output_length,
                             sum(length for _n, length in o.feature_spec()))

    def test_observe_returns_exactly_the_declared_names_in_order(self):
        ctx = StubCtx(make_candidates([100.0, 200.0], [1.0, 2.0]))
        fc = StubFleetCtrl([VRL_STATES.IDLE])
        for o in self.observers:
            parts = o.observe(fc, ctx)
            self.assertEqual(list(parts.keys()), [n for n, _ in o.feature_spec()],
                             type(o).__name__)
            for name, length in o.feature_spec():
                self.assertEqual(np.size(parts[name]), length, f"{type(o).__name__}.{name}")

    def test_the_layout_addresses_the_right_values_in_a_real_vector(self):
        ctx = StubCtx(make_candidates([100.0, 200.0, 300.0], [5.0, 1.0, 9.0]))
        fc = StubFleetCtrl([VRL_STATES.IDLE])
        vec = np.concatenate([o.observe_vector(fc, ctx) for o in self.observers])
        parts = {**self.cand.observe(fc, ctx), **self.glob.observe(fc, ctx)}
        for name, sl in self.layout.items():
            np.testing.assert_allclose(vec[sl], np.ravel(parts[name]), atol=0,
                                       err_msg=f"layout points {name} at the wrong values")

    def test_a_duplicate_feature_name_raises(self):
        # P2.9 and P2.10 both add blocks; a collision would leave one feature unreachable by
        # name while the vector still had exactly the right length
        with self.assertRaises(ValueError) as cm:
            build_observation_layout([self.cand, CandidateObserver(K, SCENARIO)])
        self.assertIn("duplicate", str(cm.exception).lower())
        self.assertIn("is_valid", str(cm.exception))

    def test_a_non_positive_declared_length_raises(self):
        class Bad(CandidateObserver):
            def feature_spec(self):
                return [("broken", 0)]
        with self.assertRaises(ValueError):
            build_observation_layout([Bad(K, SCENARIO)])

    def test_an_observe_that_disagrees_with_its_spec_raises(self):
        class Renamed(GlobalStateObserver):
            def observe(self, fleetpy_module, ctx):
                parts = super().observe(fleetpy_module, ctx)
                parts["time_sine"] = parts.pop("time_sin")       # a typo'd rename
                return parts
        ctx = StubCtx(make_candidates([100.0], [1.0]))
        with self.assertRaises(ValueError) as cm:
            Renamed(SCENARIO).observe_vector(StubFleetCtrl([VRL_STATES.IDLE]), ctx)
        self.assertIn("feature_spec", str(cm.exception))


class TestGreedyEnforcesIsValidAgainstTheMask(unittest.TestCase):
    """`greedy_action` reads `is_valid` by name and requires it to equal the mask (P2.8).

    Two failures in one check: trap 12 drift between the mask and the observation, and a
    `VecNormalize`'d observation being fed to greedy selection (P2.12). The second is the
    dangerous one — normalization scales each `delta_cfv` slot by its own running mean and
    variance, so an argmin across slots would compare z-scores from different distributions
    rather than costs, returning plausible actions that are quietly not greedy.
    """

    def setUp(self):
        self.cand = CandidateObserver(K, SCENARIO)
        self.glob = GlobalStateObserver(SCENARIO)
        self.layout = build_observation_layout([self.cand, self.glob])

    def _obs(self, n, cfvs=None):
        cfvs = cfvs if cfvs is not None else [float(i + 1) for i in range(n)]
        ctx = StubCtx(make_candidates([100.0] * n, cfvs))
        fc = StubFleetCtrl([VRL_STATES.IDLE])
        return np.concatenate([self.cand.observe_vector(fc, ctx),
                               self.glob.observe_vector(fc, ctx)])

    def test_agreeing_mask_and_is_valid_are_accepted(self):
        for n in range(0, K + 1):
            with self.subTest(n=n):
                obs = self._obs(n)
                greedy_action(obs, build_action_mask(n, K), self.layout)

    def test_a_mask_that_disagrees_with_is_valid_raises(self):
        obs = self._obs(3)                               # is_valid = 1,1,1,0,...
        for wrong_n in (2, 4, 0, K):
            with self.subTest(mask_n=wrong_n):
                with self.assertRaises(ValueError) as cm:
                    greedy_action(obs, build_action_mask(wrong_n, K), self.layout)
                self.assertIn("trap 12", str(cm.exception))

    def test_a_normalized_observation_raises(self):
        # what VecNormalize actually produces: is_valid is no longer 0/1
        obs = self._obs(3)
        normalized = obs.copy()
        sl = self.layout["is_valid"]
        normalized[sl] = (obs[sl] - 0.375) / 0.484      # a plausible running mean and std
        with self.assertRaises(ValueError) as cm:
            greedy_action(normalized, build_action_mask(3, K), self.layout)
        msg = str(cm.exception)
        self.assertIn("VecNormalize", msg)
        self.assertIn("0.0 or 1.0", msg)

    def test_a_wholly_normalized_vector_raises_too(self):
        obs = self._obs(4)
        normalized = (obs - obs.mean()) / (obs.std() + 1e-8)
        with self.assertRaises(ValueError):
            greedy_action(normalized, build_action_mask(4, K), self.layout)

    def test_near_one_is_not_close_enough(self):
        # the check is exact on purpose: 0.999 is not a validity flag
        obs = self._obs(3)
        obs[self.layout["is_valid"]] = np.array([0.999, 1.0, 1.0] + [0.0] * (K - 3),
                                                dtype=np.float32)
        with self.assertRaises(ValueError):
            greedy_action(obs, build_action_mask(3, K), self.layout)

    def test_a_layout_without_delta_cfv_raises_rather_than_guessing(self):
        bad = {k: v for k, v in self.layout.items() if k != "delta_cfv"}
        with self.assertRaises(ValueError) as cm:
            greedy_action(self._obs(3), build_action_mask(3, K), bad)
        self.assertIn("delta_cfv", str(cm.exception))

    def test_a_layout_without_is_valid_raises(self):
        bad = {k: v for k, v in self.layout.items() if k != "is_valid"}
        with self.assertRaises(ValueError) as cm:
            greedy_action(self._obs(3), build_action_mask(3, K), bad)
        self.assertIn("is_valid", str(cm.exception))

    def test_a_truncated_observation_raises(self):
        obs = self._obs(3)
        with self.assertRaises(ValueError):
            greedy_action(obs[:K], build_action_mask(3, K), self.layout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
