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
    signed_log1p,
)
from src.rl_gym.spaces import candidate_slot_validity  # noqa: E402

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
