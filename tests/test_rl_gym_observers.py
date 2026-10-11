"""Unit tests for src/rl_gym/observers.py (P1.6).

Stdlib unittest: `python -m unittest discover tests`. Stub doubles rather than a real
simulation, so these stay fast; the full-run checks (no NaN/inf across 436 real decisions,
slot 0 always the argmin) live in P1.6's verification script and later in P1.10.
"""

import math
import os
import sys
import tempfile
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
    NodeCoordinates,
    RequestObserver,
    build_observation_layout,
    signed_log1p,
)
from src.rl_gym.zones import ZoneTable  # noqa: E402
from src.rl_gym.policies import greedy_action, greedy_choice  # noqa: E402
from src.rl_gym.spaces import build_action_mask, candidate_slot_validity  # noqa: E402

K = 8
MAX_WT = 1500.0
START, END = 25200.0, 70000.0

SCENARIO = {G_OP_MAX_WT: MAX_WT, G_SIM_START_TIME: START, G_SIM_END_TIME: END}


class StubStop:
    """A plan stop: a position plus the boarding/alighting lists keyed +1 / -1.

    `copy()` reproduces `PlanStop.copy()`'s **shallow** `boarding_dict.copy()` deliberately
    (§5.2 exception a): the inner lists stay shared, which is the one way a re-timed copy can
    reach the stored plan.
    """

    def __init__(self, pos, boarding=(), alighting=()):
        self.pos = pos
        self.boarding_dict = {1: list(boarding), -1: list(alighting)}

    def get_pos(self):
        return self.pos

    def get_list_boarding_rids(self):
        return list(self.boarding_dict.get(1, []))

    def get_list_alighting_rids(self):
        return list(self.boarding_dict.get(-1, []))

    def copy(self):
        cp = StubStop(self.pos)
        cp.boarding_dict = self.boarding_dict.copy()   # shallow, like the real one
        return cp


class StubPlan:
    """A vehicle plan.

    `update_tt_and_check_plan` installs `retimed_pax_info` rather than re-deriving times from
    the network: these tests pin the observer's arithmetic and its copy discipline, not FleetPy's
    re-timer. The re-timer itself is exercised by the hand computations on a real episode and by
    the P1.10 gate — reimplementing it here would only test the reimplementation.
    """

    def __init__(self, pax_info, stops=(), utility=None, retimed_pax_info=None):
        self.pax_info = dict(pax_info)
        self.list_plan_stops = list(stops)
        self.utility = utility
        self.retimed_pax_info = retimed_pax_info
        self.retimed_on_self = 0        # a re-time on the stored plan is a trap-14 violation
        self.copies_retimed = 0
        self._origin = None

    def copy(self):
        cp = StubPlan({k: list(v) for k, v in self.pax_info.items()},
                      [s.copy() for s in self.list_plan_stops],
                      self.utility, self.retimed_pax_info)
        cp._origin = self
        return cp

    def update_tt_and_check_plan(self, veh_obj, sim_time, routing_engine,
                                 init_plan_state=None, keep_feasible=False, rq_dict_prq=None):
        if self._origin is None:
            self.retimed_on_self += 1
        else:
            self._origin.copies_retimed += 1
        if self.retimed_pax_info is not None:
            self.pax_info = {k: list(v) for k, v in self.retimed_pax_info.items()}
        return True


class StubOffer:
    def __init__(self, offered_waiting_time):
        self.offered_waiting_time = offered_waiting_time


class StubPrq:
    def __init__(self, rq_time, o_pos=(1, None, None), d_pos=(2, None, None),
                 nr_pax=1, offer=None):
        self.rq_time = rq_time
        self.o_pos = o_pos
        self.d_pos = d_pos
        self.nr_pax = nr_pax
        self.offer = offer


class StubCtx:
    """Stands in for PendingDecision, with the stub operator the observer reads through."""

    def __init__(self, candidates, rq_time=30000.0, sim_time=30000.0, rid_struct=7,
                 o_pos=(1, None, None), d_pos=(2, None, None), op=None):
        self.candidates = candidates
        self.prq = StubPrq(rq_time, o_pos=o_pos, d_pos=d_pos)
        self.sim_time = sim_time
        self.rid_struct = rid_struct
        self.op = op if op is not None else StubOperator(candidates)


class StubVehicle:
    def __init__(self, status=None, pos=(1, None, None), pax=(), max_pax=4,
                 assigned_route=()):
        self.status = status
        self.pos = pos
        self.pax = list(pax)
        self.max_pax = max_pax
        self.assigned_route = list(assigned_route)


class StubLeg:
    def __init__(self, destination_pos):
        self.destination_pos = destination_pos


class StubRider:
    """A traveler object as `veh_obj.pax` holds them."""

    def __init__(self, rid, nr_pax=1):
        self.rid = rid
        self.nr_pax = nr_pax

    def get_rid_struct(self):
        return self.rid


class StubCoords:
    """Hull-scaled node coordinates, linear in the node index so hand computation is trivial."""

    def scaled(self, node):
        n = float(node)
        return (n * 0.1 - 1.0, 1.0 - n * 0.1)


class StubNetwork:
    """Just enough routing engine, with `NetworkBasicWithStore`'s cache semantics.

    `travel_time_infos` is written **only when the key is absent**, exactly as
    `_add_to_database` does, because that is what makes the observer's snapshot-and-delete
    restore exact. A trivial route (same position) is resolved without touching the cache, as
    `test_and_get_trivial_route_tt_and_dis` does.
    """

    def __init__(self, coords=None, with_cache=True):
        #: node -> (x, y) in metres
        self.coords = coords or {i: (float(i) * 100.0, 0.0) for i in range(20)}
        self.queries = []
        if with_cache:
            self.travel_time_infos = {}

    def return_node_coordinates(self, node):
        return self.coords[node]

    def return_position_coordinates(self, pos):
        if pos[1] is None:
            return self.return_node_coordinates(pos[0])
        c0 = np.array(self.coords[pos[0]], dtype=float)
        c1 = np.array(self.coords[pos[1]], dtype=float)
        c = pos[2] * c1 + (1.0 - pos[2]) * c0
        return (float(c[0]), float(c[1]))

    def return_travel_costs_1to1(self, o_pos, d_pos, customized_section_cost_function=None):
        self.queries.append((o_pos, d_pos))
        if o_pos == d_pos:
            return (0.0, 0.0, 0.0)
        a = np.array(self.return_position_coordinates(o_pos), dtype=float)
        b = np.array(self.return_position_coordinates(d_pos), dtype=float)
        dist = float(np.sqrt(((b - a) ** 2).sum()))
        tt = dist / 10.0
        key = ((o_pos[1] if o_pos[1] is not None else o_pos[0]), d_pos[0])
        cache = getattr(self, "travel_time_infos", None)
        if cache is not None and key not in cache:
            cache[key] = (tt, tt, dist)
        return (tt, tt, dist)


class _GrowingFleet(list):
    """`sim_vehicles`: a list indexed by vid, which grows rather than raising IndexError.

    A fixture convenience only: the real `sim_vehicles` always holds every vid, and this keeps a
    test from having to enumerate vehicles it does not care about.
    """

    def __getitem__(self, vid):
        if isinstance(vid, int) and 0 <= vid >= len(self):
            while len(self) <= vid:
                self.append(StubVehicle())
        return list.__getitem__(self, vid)


class _GrowingPlans(dict):
    """`veh_plans`: an empty committed plan for any vid a test did not spell out."""

    def __missing__(self, vid):
        plan = StubPlan({})
        self[vid] = plan
        return plan


class StubOperator:
    """Stands in for `RLPoolingIRSOnly` as the observers read it.

    `sim_vehicles` is a **list indexed by vid** and `veh_plans` a dict keyed by vid, as
    `FleetControlBase` holds them, so candidate vids are integers here too. The global observer
    iterates the same list.
    """

    def __init__(self, candidates=(), statuses=None, vehicles=None, veh_plans=None,
                 rq_dict=None, routing_engine=None):
        self.routing_engine = routing_engine or StubNetwork()
        if vehicles is not None:
            self.sim_vehicles = _GrowingFleet(vehicles)
        elif statuses is not None:
            self.sim_vehicles = _GrowingFleet(StubVehicle(status=s) for s in statuses)
        else:
            self.sim_vehicles = _GrowingFleet()
        needed = max([int(vid) for vid, _p, _c in candidates], default=-1) + 1
        while len(self.sim_vehicles) < needed:
            self.sim_vehicles.append(StubVehicle())
        self.veh_plans = _GrowingPlans(veh_plans or {})
        for vid, _plan, _cfv in candidates:
            self.veh_plans.setdefault(vid, StubPlan({}))
        self.rq_dict = dict(rq_dict or {})
        self.rq_dict_store = dict(self.rq_dict)


class StubFleetCtrl(StubOperator):
    """The global observer's view: a fleet of the given statuses."""

    def __init__(self, statuses):
        super().__init__(statuses=statuses)


def candidate_observer(k_max=K, scenario=None, **kw):
    """`CandidateObserver` over the stub coordinates, so tests need no node file.

    `scenario={}` must reach the constructor and raise, so the default is a None check rather
    than an `or`.
    """
    return CandidateObserver(k_max, SCENARIO if scenario is None else scenario,
                             StubCoords(), **kw)


#: a tiny baseline grid covering the stub scenario's minutes, written to a temp file by
#: `make_baseline`. Three months, both windows, the whole [START, END) minute range would be
#: 747 rows, so the tests use a coarse grid and widen START/END only where needed.
BASELINE_MONTHS = (5, 6, 10)


def write_baseline(tmpdir, rows=None, start=25200, end=70000, step=60):
    """A real `arrival_baseline.csv` plus the manifest that hashes it.

    The tests build the **real** `ArrivalBaseline` over this rather than a stub double: the
    lookup, the sd floor, `check_scenario` and `verify_baseline` are all things a stub would
    reimplement and then fail to keep in step.
    """
    from src.rl_gym import corpus
    if rows is None:
        rows = []
        for month in BASELINE_MONTHS:
            for t in range(start, end, step):
                for window_s in (1800, 3600):
                    # mean rises with the month and the hour; sd stays small so the floor bites
                    mean = month + (t - start) / 3600.0 + (0.0 if window_s == 1800 else 10.0)
                    rows.append((month, t, window_s, round(mean, 6), 0.25 * month))
    path = os.path.join(tmpdir, "arrival_baseline.csv")
    with open(path, "w", newline="") as fh:
        fh.write(",".join(corpus.BASELINE_COLUMNS) + "\n")
        for month, t, window_s, mean, sd in rows:
            fh.write(f"{month},{t},{window_s},{corpus.FLOAT_FMT.format(mean)},"
                     f"{corpus.FLOAT_FMT.format(sd)}\n")
    manifest = {"baseline_hash": corpus.baseline_hash(rows), "baseline_rows": len(rows),
                "sd_floor": 1.0, "minute_s": 60}
    return path, manifest, rows


def write_active_fleet(tmpdir, pairs=((25200, 0.7), (28800, 0.9), (36000, 1.0),
                                      (61200, 0.9))):
    from src.misc.globals import G_ACT_VEH_SHARE as share_col
    path = os.path.join(tmpdir, "example_active_fleetsize.csv")
    with open(path, "w", newline="") as fh:
        fh.write(f"time,{share_col}\n")
        for t, share in pairs:
            fh.write(f"{t},{share}\n")
    return path


def wide_global_observer(tmpdir):
    """A global observer whose baseline covers 00:00-25:00, for the clock-time features.

    The production grid is the scenario's own `[start_time, end_time)` and a lookup outside it
    raises by design (guard 1) — a decision cannot occur there, so a `KeyError` would be a real
    bug. The `time_sin`/`time_cos`/`day_progress` tests deliberately probe times no decision can
    reach, so they get a table that covers them rather than a tolerant lookup.
    """
    from src.rl_gym.observers import ActiveFleetSchedule, ArrivalBaseline
    sub = os.path.join(tmpdir, "wide")
    os.makedirs(sub, exist_ok=True)
    rows = [(6, t, w, 1.0, 1.0) for t in range(0, 90120, 60) for w in (1800, 3600)]
    path, manifest, _rows = write_baseline(sub, rows=rows)
    obs = GlobalStateObserver(SCENARIO, ArrivalBaseline(manifest, path),
                              ActiveFleetSchedule(write_active_fleet(sub)))
    obs.set_stats_tracker(_ZeroTracker())
    obs.set_episode_month(6)
    return obs


def global_observer(tmpdir, scenario=None, **kw):
    """`GlobalStateObserver` over a real baseline and a real schedule in `tmpdir`."""
    from src.rl_gym.observers import ActiveFleetSchedule, ArrivalBaseline
    path, manifest, _rows = write_baseline(tmpdir)
    return GlobalStateObserver(SCENARIO if scenario is None else scenario,
                               ArrivalBaseline(manifest, path),
                               ActiveFleetSchedule(write_active_fleet(tmpdir)), **kw)


class _ZeroTracker:
    """A rolling-stats tracker that has seen nothing."""

    def zone_count(self, kind, window_s, node):
        return 0

    def count(self, kind, window_s, cells=None):
        return 0

    def wait_gap_mean(self, window_s):
        return None


class _RecordingTracker:
    """Returns a distinguishable number per query and records what was asked."""

    def __init__(self, counts=None, wait_gap=None):
        self.calls = []
        self.fleet_calls = []
        self.counts = counts or {}
        self.wait_gap = wait_gap

    def zone_count(self, kind, window_s, node):
        self.calls.append((kind, window_s, int(node)))
        return 11

    def count(self, kind, window_s, cells=None):
        self.fleet_calls.append((kind, window_s, cells))
        return self.counts.get((kind, window_s), 0)

    def wait_gap_mean(self, window_s):
        return None if self.wait_gap is None else self.wait_gap.get(window_s)


def make_candidates(waits, cfvs, rq_time=30000.0, rid_struct=7):
    """Candidates whose plans quote the given waits (seconds) and insertion costs."""
    return [
        (i, StubPlan({rid_struct: [rq_time + w, rq_time + w + 600.0]}), c)
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
        self.obs = candidate_observer()

    def test_output_length_is_twelve_k(self):
        self.assertEqual(self.obs.output_length, 12 * K)
        self.assertEqual(candidate_observer(3).output_length, 36)

    def test_shape_and_dtype(self):
        ctx = StubCtx(make_candidates([100.0, 200.0], [1.0, 2.0]))
        vec = self.obs.observe_vector(ctx.op, ctx)
        self.assertEqual(vec.shape, (12 * K,))
        self.assertEqual(vec.dtype, np.float32)

    def test_is_valid_comes_from_shared_function(self):
        for n in range(0, 12):
            with self.subTest(n=n):
                ctx = StubCtx(make_candidates([100.0] * n, list(range(n))))
                parts = self.obs.observe(ctx.op, ctx)
                np.testing.assert_array_equal(
                    parts["is_valid"], candidate_slot_validity(n, K).astype(np.float32)
                )

    def test_padded_slots_are_all_zero(self):
        ctx = StubCtx(make_candidates([100.0, 200.0], [5.0, 9.0]))
        parts = self.obs.observe(ctx.op, ctx)
        for name in ("is_valid", "delta_cfv", "offered_wait"):
            np.testing.assert_array_equal(parts[name][2:], np.zeros(K - 2, dtype=np.float32))

    def test_delta_cfv_transformed_not_raw(self):
        ctx = StubCtx(make_candidates([100.0], [100.0]))
        parts = self.obs.observe(ctx.op, ctx)
        self.assertAlmostEqual(float(parts["delta_cfv"][0]), math.log1p(100.0), places=5)

    def test_negative_delta_cfv(self):
        ctx = StubCtx(make_candidates([100.0], [-100.0]))
        parts = self.obs.observe(ctx.op, ctx)
        self.assertAlmostEqual(float(parts["delta_cfv"][0]), -math.log1p(100.0), places=5)

    def test_offered_wait_matches_the_operator_formula(self):
        # exactly what _create_user_offer quotes: pax_info[rid][0] - prq.rq_time
        rq_time = 30000.0
        ctx = StubCtx(make_candidates([300.0, 750.0], [1.0, 2.0], rq_time=rq_time),
                      rq_time=rq_time)
        parts = self.obs.observe(ctx.op, ctx)
        self.assertAlmostEqual(float(parts["offered_wait"][0]), 300.0 / MAX_WT, places=6)
        self.assertAlmostEqual(float(parts["offered_wait"][1]), 750.0 / MAX_WT, places=6)

    def test_normalizer_read_from_scenario_parameters_not_hardcoded(self):
        # trap 8: the two constant configs disagree on op_max_wait_time (1500 vs 4200)
        other = dict(SCENARIO, **{G_OP_MAX_WT: 4200.0})
        ctx = StubCtx(make_candidates([600.0], [1.0]))
        self.assertAlmostEqual(
            float(candidate_observer(scenario=other).observe(ctx.op, ctx)["offered_wait"][0]),
            600.0 / 4200.0, places=6)

    def test_truncates_to_k_preserving_order(self):
        cfvs = [float(i) for i in range(11)]
        ctx = StubCtx(make_candidates([100.0] * 11, cfvs))
        parts = self.obs.observe(ctx.op, ctx)
        np.testing.assert_allclose(
            parts["delta_cfv"], signed_log1p(np.array(cfvs[:K])).astype(np.float32), atol=1e-6)
        np.testing.assert_array_equal(parts["is_valid"], np.ones(K, dtype=np.float32))

    def test_slot_zero_is_the_minimum_for_a_sorted_list(self):
        cfvs = [1.0, 4.0, 9.0, 50.0]
        ctx = StubCtx(make_candidates([100.0] * 4, cfvs))
        d = self.obs.observe(ctx.op, ctx)["delta_cfv"]
        valid = self.obs.observe(ctx.op, ctx)["is_valid"] > 0
        self.assertEqual(int(np.argmin(d[valid])), 0)

    def test_missing_pax_info_yields_zero_wait_not_a_crash(self):
        # defensive: a plan that does not carry this rid must not raise
        ctx = StubCtx([(0, StubPlan({}), 1.0)])
        parts = self.obs.observe(ctx.op, ctx)
        self.assertEqual(float(parts["offered_wait"][0]), 0.0)
        self.assertEqual(float(parts["is_valid"][0]), 1.0)

    def test_empty_candidate_list(self):
        ctx = StubCtx([])
        parts = self.obs.observe(ctx.op, ctx)
        for name in CandidateObserver.FEATURE_NAMES:
            np.testing.assert_array_equal(parts[name], np.zeros(K, dtype=np.float32))

    def test_rejects_bad_config(self):
        with self.assertRaises(ValueError):
            candidate_observer(0)
        with self.assertRaises(ValueError):
            candidate_observer(scenario=dict(SCENARIO, **{G_OP_MAX_WT: 0.0}))
        with self.assertRaises(KeyError):
            candidate_observer(scenario={})

    def test_no_nan_or_inf(self):
        ctx = StubCtx(make_candidates([0.0, 1500.0, 3000.0], [-1e6, 0.0, 1e6]))
        vec = self.obs.observe_vector(ctx.op, ctx)
        self.assertTrue(np.all(np.isfinite(vec)))


class TestGlobalStateObserver(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.obs = global_observer(self.tmp.name)
        self.obs.set_stats_tracker(_ZeroTracker())
        self.obs.set_episode_month(6)

    def test_output_length_is_nineteen(self):
        self.assertEqual(self.obs.output_length, 19)

    def test_shape_and_dtype(self):
        fc = StubFleetCtrl([VRL_STATES.IDLE] * 3)
        vec = self.obs.observe_vector(fc, StubCtx([], sim_time=START))
        self.assertEqual(vec.shape, (19,))
        self.assertEqual(vec.dtype, np.float32)

    def test_time_of_day_is_absolute_clock_not_episode_fraction(self):
        fc = StubFleetCtrl([VRL_STATES.IDLE])
        # 06:00 -> t_of_day = 0.25 -> sin = 1, cos = 0
        parts = wide_global_observer(self.tmp.name).observe(fc, StubCtx([], sim_time=21600.0))
        self.assertAlmostEqual(float(parts["time_sin"]), 1.0, places=5)
        self.assertAlmostEqual(float(parts["time_cos"]), 0.0, places=5)
        # midnight -> sin = 0, cos = 1
        parts = wide_global_observer(self.tmp.name).observe(fc, StubCtx([], sim_time=0.0))
        self.assertAlmostEqual(float(parts["time_sin"]), 0.0, places=5)
        self.assertAlmostEqual(float(parts["time_cos"]), 1.0, places=5)

    def test_time_of_day_wraps_at_midnight(self):
        fc = StubFleetCtrl([VRL_STATES.IDLE])
        a = wide_global_observer(self.tmp.name).observe(fc, StubCtx([], sim_time=3600.0))
        b = wide_global_observer(self.tmp.name).observe(fc, StubCtx([], sim_time=3600.0 + 86400.0))
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
        parts = wide_global_observer(self.tmp.name).observe(fc, StubCtx([], sim_time=END + 14400.0))
        self.assertEqual(float(parts["day_progress"]), 1.0)
        parts = wide_global_observer(self.tmp.name).observe(fc, StubCtx([], sim_time=START - 100.0))
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
            global_observer(self.tmp.name,
                            scenario=dict(SCENARIO, **{G_SIM_START_TIME: 70000.0,
                                                       G_SIM_END_TIME: 25200.0}))
        with self.assertRaises(ValueError):
            global_observer(self.tmp.name,
                            scenario=dict(SCENARIO, **{G_SIM_START_TIME: 100.0,
                                                       G_SIM_END_TIME: 100.0}))
        with self.assertRaises(KeyError):
            global_observer(self.tmp.name, scenario={})


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
        self.obs = candidate_observer()
        # two valid candidates, both POSITIVE, sorted ascending as insertion returns them
        self.ctx = StubCtx(make_candidates([100.0, 200.0], [3.0, 7.0]))
        parts = self.obs.observe(self.ctx.op, self.ctx)
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
                parts = self.obs.observe(ctx.op, ctx)
                valid_idx = np.flatnonzero(parts["is_valid"] > 0)
                picked = int(valid_idx[np.argmin(parts["delta_cfv"][valid_idx])])
                self.assertEqual(picked, 0)
                if n < K:  # a padded slot exists, so the unrestricted form must disagree
                    self.assertNotEqual(int(np.argmin(parts["delta_cfv"])), picked)

    def test_restricted_argmin_also_correct_under_negative_costs(self):
        # the real scenario's sign: restriction must be harmless, not just necessary
        cfvs = [-500.0, -200.0, -10.0]
        ctx = StubCtx(make_candidates([100.0] * 3, cfvs))
        parts = self.obs.observe(ctx.op, ctx)
        valid_idx = np.flatnonzero(parts["is_valid"] > 0)
        self.assertEqual(int(valid_idx[np.argmin(parts["delta_cfv"][valid_idx])]), 0)
        self.assertEqual(int(np.argmin(parts["delta_cfv"])), 0)  # coincides here

    def test_single_valid_slot_with_positive_cost(self):
        ctx = StubCtx(make_candidates([100.0], [42.0]))
        parts = self.obs.observe(ctx.op, ctx)
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
        self.obs_builder = candidate_observer()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.glob = global_observer(self.tmp.name)
        self.glob.set_stats_tracker(_ZeroTracker())
        self.glob.set_episode_month(6)
        # the same map the env publishes, built from the same specs in the same order (P2.8)
        self.layout = build_observation_layout([self.obs_builder, self.glob])

    def _full_obs(self, ctx):
        fc = StubFleetCtrl([VRL_STATES.IDLE])
        return np.concatenate([self.obs_builder.observe_vector(ctx.op, ctx),
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
        parts = self.obs_builder.observe(ctx.op, ctx)
        obs = self._full_obs(ctx)
        np.testing.assert_allclose(obs[self.layout["delta_cfv"]], parts["delta_cfv"], atol=0)
        np.testing.assert_allclose(obs[self.layout["is_valid"]], parts["is_valid"], atol=0)
        self.assertEqual(greedy_action(obs, build_action_mask(3, K), self.layout), 1)


class TestCombinedLength(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_sum_of_observer_lengths_is_twelve_k_plus_nineteen(self):
        total = candidate_observer().output_length + \
            global_observer(self.tmp.name).output_length
        self.assertEqual(total, 12 * K + 19)
        self.assertEqual(total, 115)

    def test_scales_with_k_without_hardcoding(self):
        for k in (1, 4, 8, 16):
            with self.subTest(k=k):
                total = candidate_observer(k).output_length + \
                    global_observer(self.tmp.name).output_length
                self.assertEqual(total, 12 * k + 19)

    def test_concatenated_vector_shape_and_dtype(self):
        cand = candidate_observer()
        glob = global_observer(self.tmp.name)
        glob.set_stats_tracker(_ZeroTracker())
        glob.set_episode_month(6)
        ctx = StubCtx(make_candidates([100.0, 200.0], [1.0, 2.0]))
        fc = StubFleetCtrl([VRL_STATES.IDLE, VRL_STATES.ROUTE])
        vec = np.concatenate([cand.observe_vector(fc, ctx), glob.observe_vector(fc, ctx)])
        self.assertEqual(vec.shape, (12 * K + 19,))
        self.assertEqual(vec.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(vec)))


class TestNodeCoordinateScaling(unittest.TestCase):
    """§5.2's bbox is the min/max over `nodes.csv`, not `return_network_bounding_box()`.

    P2.1 measured why: the reprojected UTM corners fall *inside* the node latitude hull,
    because the grid is turned about 0.5-0.7° from true north. Scaling by them puts 13 of the
    4,084 nodes outside [-1, 1]. These tests are what fails if anyone switches to the helper.
    """

    @classmethod
    def setUpClass(cls):
        cls.coords = NodeCoordinates()
        import pandas as pd
        cls.nodes = pd.read_csv(os.path.join(
            os.path.dirname(__file__), "..", "data", "networks", "example_network",
            "base", "nodes.csv"))

    def test_all_4084_nodes_map_within_the_unit_box(self):
        self.assertEqual(len(self.coords), 4084)
        out = []
        for node in self.nodes["node_index"]:
            lat, lon = self.coords.scaled(node)
            if not (-1.0 <= lat <= 1.0) or not (-1.0 <= lon <= 1.0):
                out.append((int(node), lat, lon))
        self.assertEqual(out, [], f"{len(out)} node(s) outside [-1, 1]")

    def test_the_extremes_land_exactly_on_the_box(self):
        lats = [self.coords.scaled(n)[0] for n in self.nodes["node_index"]]
        lons = [self.coords.scaled(n)[1] for n in self.nodes["node_index"]]
        self.assertAlmostEqual(min(lats), -1.0, places=12)
        self.assertAlmostEqual(max(lats), 1.0, places=12)
        self.assertAlmostEqual(min(lons), -1.0, places=12)
        self.assertAlmostEqual(max(lons), 1.0, places=12)

    def test_the_hull_is_the_node_columns_not_the_utm_corners(self):
        self.assertAlmostEqual(self.coords.lat_min, float(self.nodes.lat.min()), places=12)
        self.assertAlmostEqual(self.coords.lat_max, float(self.nodes.lat.max()), places=12)
        self.assertAlmostEqual(self.coords.lon_min, float(self.nodes.lon.min()), places=12)
        self.assertAlmostEqual(self.coords.lon_max, float(self.nodes.lon.max()), places=12)

    def test_scaling_by_the_utm_corners_would_escape_the_box(self):
        # the counter-example, pinned: 13 nodes outside, reaching -1.0106 and +1.0222 (P2.1)
        from pyproj import Transformer
        t = Transformer.from_proj("EPSG:32610", "epsg:4326")
        lats, _lons = t.transform([self.nodes.pos_x.min(), self.nodes.pos_x.max()],
                                  [self.nodes.pos_y.min(), self.nodes.pos_y.max()])
        scaled = 2 * (self.nodes.lat - lats[0]) / (lats[1] - lats[0]) - 1
        self.assertEqual(int(((scaled < -1) | (scaled > 1)).sum()), 13)
        self.assertLess(scaled.min(), -1.0)
        self.assertGreater(scaled.max(), 1.0)

    def test_scaling_is_linear_and_monotone(self):
        pairs = sorted((float(r.lat), self.coords.scaled(int(r.node_index))[0])
                       for r in self.nodes.head(200).itertuples())
        self.assertEqual([s for _l, s in pairs], sorted(s for _l, s in pairs))

    def test_an_unknown_node_raises_rather_than_returning_a_sentinel(self):
        for node in (-1, 4084, 10 ** 9):
            with self.assertRaises(KeyError):
                self.coords.scaled(node)


class TestRequestObserver(unittest.TestCase):
    """The request block's 24 features (§5.2, P2.9)."""

    WEEKDAYS = ["Tue", "Wed", "Thu"]
    MONTHS = list(range(1, 11))

    def _observer(self, tracker=None, weekday="Thu", month=6):
        obs = RequestObserver(NodeCoordinates(), ZoneTable(),
                              weekday_vocab=self.WEEKDAYS, month_vocab=self.MONTHS)
        if tracker is not None:
            obs.set_stats_tracker(tracker)
        if weekday is not None:
            obs.set_episode_day(weekday, month)
        return obs

    def _ctx(self, o_node=0, d_node=100, nr_pax=1):
        class Prq:
            pass
        prq = Prq()
        prq.o_pos, prq.d_pos = (o_node, None, None), (d_node, None, None)
        prq.nr_pax = nr_pax
        prq.rq_time = 30000.0
        ctx = StubCtx(make_candidates([100.0], [1.0]))
        ctx.prq = prq
        return ctx

    def test_declares_twenty_four_features(self):
        obs = self._observer(tracker=_ZeroTracker())
        self.assertEqual(obs.output_length, 24)
        self.assertEqual(sum(n for _, n in obs.feature_spec()), 24)

    def test_feature_names_and_widths(self):
        spec = dict(self._observer(tracker=_ZeroTracker()).feature_spec())
        self.assertEqual(spec["o_lat"], 1)
        self.assertEqual(spec["dow"], 3)
        self.assertEqual(spec["month"], 10)
        for stem in ("zone_arrivals", "zone_declines", "zone_cancels"):
            for w in (30, 60):
                self.assertEqual(spec[f"{stem}_{w}"], 1)

    def test_coordinates_come_from_the_requests_own_nodes(self):
        coords = NodeCoordinates()
        obs = self._observer(tracker=_ZeroTracker())
        parts = obs.observe(None, self._ctx(o_node=7, d_node=2000))
        self.assertAlmostEqual(float(parts["o_lat"]), coords.scaled(7)[0], places=6)
        self.assertAlmostEqual(float(parts["o_lon"]), coords.scaled(7)[1], places=6)
        self.assertAlmostEqual(float(parts["d_lat"]), coords.scaled(2000)[0], places=6)
        self.assertAlmostEqual(float(parts["d_lon"]), coords.scaled(2000)[1], places=6)

    def test_n_pax_is_raw(self):
        obs = self._observer(tracker=_ZeroTracker())
        for n in (1, 2, 4):
            parts = obs.observe(None, self._ctx(nr_pax=n))
            self.assertEqual(float(parts["n_pax"]), float(n))

    def test_one_hots_are_exactly_one_hot(self):
        for weekday, month in (("Tue", 1), ("Wed", 10), ("Thu", 6)):
            obs = self._observer(tracker=_ZeroTracker(), weekday=weekday, month=month)
            parts = obs.observe(None, self._ctx())
            np.testing.assert_array_equal(
                parts["dow"], [1.0 if w == weekday else 0.0 for w in self.WEEKDAYS])
            np.testing.assert_array_equal(
                parts["month"], [1.0 if m == month else 0.0 for m in self.MONTHS])
            self.assertEqual(parts["dow"].sum(), 1.0)
            self.assertEqual(parts["month"].sum(), 1.0)

    def test_zone_counts_come_from_the_tracker_for_the_requests_origin(self):
        tracker = _RecordingTracker()
        obs = self._observer(tracker=tracker)
        parts = obs.observe(None, self._ctx(o_node=42))
        self.assertEqual(sorted(tracker.calls), sorted([
            ("arrival", 1800, 42), ("arrival", 3600, 42),
            ("decline", 1800, 42), ("decline", 3600, 42),
            ("cancel", 1800, 42), ("cancel", 3600, 42)]))
        self.assertEqual(float(parts["zone_arrivals_30"]), 11.0)   # the stub's answer

    def test_an_unknown_weekday_or_month_raises_at_set_episode_day(self):
        # at reset, not at the first decision: a misconfigured episode must fail before it runs.
        # set_episode_day is called directly rather than through the helper, which skips it for
        # a None weekday -- `None` is exactly what P2.3's parse_day_stem returns for a non-ISO
        # filename, so it is the case that most needs covering.
        for weekday, month in (("Mon", 6), ("Thu", 11), (None, 6), ("Thu", None),
                               (None, None), ("thu", 6), ("Thu", 0)):
            with self.subTest(weekday=weekday, month=month):
                obs = RequestObserver(NodeCoordinates(), ZoneTable(),
                                      weekday_vocab=self.WEEKDAYS, month_vocab=self.MONTHS)
                with self.assertRaises(ValueError):
                    obs.set_episode_day(weekday, month)

    def test_observe_without_a_tracker_raises(self):
        obs = RequestObserver(NodeCoordinates(), ZoneTable(),
                              weekday_vocab=self.WEEKDAYS, month_vocab=self.MONTHS)
        obs.set_episode_day("Thu", 6)
        with self.assertRaises(RuntimeError) as cm:
            obs.observe(None, self._ctx())
        self.assertIn("rolling-stats tracker", str(cm.exception))

    def test_observe_without_an_episode_day_raises(self):
        obs = RequestObserver(NodeCoordinates(), ZoneTable(),
                              weekday_vocab=self.WEEKDAYS, month_vocab=self.MONTHS)
        obs.set_stats_tracker(_ZeroTracker())
        with self.assertRaises(RuntimeError) as cm:
            obs.observe(None, self._ctx())
        self.assertIn("episode day", str(cm.exception))

    def test_empty_or_repeating_vocabularies_raise(self):
        for wd, mo in (([], self.MONTHS), (self.WEEKDAYS, []),
                       (["Tue", "Tue"], self.MONTHS), (self.WEEKDAYS, [1, 1])):
            with self.assertRaises(ValueError):
                RequestObserver(NodeCoordinates(), ZoneTable(),
                                weekday_vocab=wd, month_vocab=mo)

    def test_observe_matches_its_declaration(self):
        obs = self._observer(tracker=_ZeroTracker())
        vec = obs.observe_vector(None, self._ctx())          # raises on any disagreement
        self.assertEqual(vec.shape, (24,))
        self.assertEqual(vec.dtype, np.float32)


class TestObservationLayout(unittest.TestCase):
    """The name -> slice map (P2.8), built from the static specs."""

    def setUp(self):
        self.cand = candidate_observer()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.glob = global_observer(self.tmp.name)
        self.glob.set_stats_tracker(_ZeroTracker())
        self.glob.set_episode_month(6)
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
        self.assertEqual(self.layout["idle_fraction"], slice(12 * K + 3, 12 * K + 4))
        self.assertEqual(self.layout["active_60"], slice(12 * K + 18, 12 * K + 19))

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
            build_observation_layout([self.cand, candidate_observer()])
        self.assertIn("duplicate", str(cm.exception).lower())
        self.assertIn("is_valid", str(cm.exception))

    def test_a_non_positive_declared_length_raises(self):
        class Bad(CandidateObserver):
            def feature_spec(self):
                return [("broken", 0)]
        with self.assertRaises(ValueError):
            build_observation_layout([Bad(K, SCENARIO, StubCoords())])

    def test_an_observe_that_disagrees_with_its_spec_raises(self):
        class Renamed(GlobalStateObserver):
            def observe(self, fleetpy_module, ctx):
                parts = super().observe(fleetpy_module, ctx)
                parts["time_sine"] = parts.pop("time_sin")       # a typo'd rename
                return parts
        ctx = StubCtx(make_candidates([100.0], [1.0]))
        with self.assertRaises(ValueError) as cm:
            renamed = Renamed(SCENARIO, self.glob.baseline, self.glob.active_fleet)
            renamed.set_stats_tracker(_ZeroTracker())
            renamed.set_episode_month(6)
            renamed.observe_vector(StubFleetCtrl([VRL_STATES.IDLE]), ctx)
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
        self.cand = candidate_observer()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.glob = global_observer(self.tmp.name)
        self.glob.set_stats_tracker(_ZeroTracker())
        self.glob.set_episode_month(6)
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


class TestCandidateBlock(unittest.TestCase):
    """§5.2's candidate features 4-12 (P2.10).

    Stub doubles for the arithmetic, the guards and the state discipline. The hand computations
    against a real episode, the feature distributions and the P1.10 gate live in P2.10's
    verification, not here.
    """

    def setUp(self):
        self.obs = candidate_observer()

    # ------------------------------------------------------------------ #
    # the spec
    # ------------------------------------------------------------------ #

    def test_twelve_features_in_sect_5_2_order(self):
        spec = self.obs.feature_spec()
        self.assertEqual([n for n, _ in spec], [
            "is_valid", "delta_cfv", "offered_wait",
            "veh_lat", "veh_lon", "free_seats", "remaining_stops", "directionality",
            "lateness_pre", "lateness_post", "added_distance", "added_pickup_delay"])
        self.assertEqual({w for _n, w in spec}, {K})

    def test_the_phase_one_features_keep_their_positions(self):
        # §5.2 keeps the layout feature-major, with the three Phase 1 features first
        self.assertEqual([n for n, _ in self.obs.feature_spec()[:3]],
                         ["is_valid", "delta_cfv", "offered_wait"])

    def test_the_feature_names_do_not_depend_on_the_position_encoding(self):
        # the layout must not move when the encoding changes (§5.2: same length either way)
        a = candidate_observer(position_encoding="latlon").feature_spec()
        b = candidate_observer(position_encoding="relative").feature_spec()
        self.assertEqual(a, b)

    def test_rejects_a_bad_position_encoding(self):
        for bad in ("h3", "", "latlng", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as cm:
                    candidate_observer(position_encoding=bad)
                self.assertIn("position_encoding", str(cm.exception))

    def test_rejects_a_non_positive_wait_normalizer(self):
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                candidate_observer(wait_normalizer_s=bad)

    # ------------------------------------------------------------------ #
    # the wait normalizer (§5.3)
    # ------------------------------------------------------------------ #

    def test_offered_wait_is_bit_identical_with_the_normalizer_absent(self):
        # §5.3: absent -> op_max_wait_time, which must reproduce Phase 1 exactly
        ctx = StubCtx(make_candidates([137.0, 912.0], [-3.0, -1.0]))
        got = self.obs.observe(ctx.op, ctx)["offered_wait"]
        want = np.array([137.0 / MAX_WT, 912.0 / MAX_WT] + [0.0] * (K - 2),
                        dtype=np.float32)
        np.testing.assert_array_equal(got, want)
        self.assertEqual(self.obs.wait_normalizer_s, MAX_WT)

    def test_an_explicit_normalizer_divides_every_wait_feature(self):
        obs = candidate_observer(wait_normalizer_s=600.0)
        ctx = self._lateness_ctx()
        parts = obs.observe(ctx.op, ctx)
        self.assertAlmostEqual(float(parts["lateness_pre"][0]), 100.0 / 600.0, places=6)
        self.assertAlmostEqual(float(parts["lateness_post"][0]), 200.0 / 600.0, places=6)
        self.assertAlmostEqual(float(parts["added_pickup_delay"][0]), 150.0 / 600.0, places=6)

    # ------------------------------------------------------------------ #
    # features 4-5: position
    # ------------------------------------------------------------------ #

    def test_vehicle_position_at_a_node(self):
        ctx = StubCtx(make_candidates([100.0], [-1.0]))
        ctx.op.sim_vehicles[0] = StubVehicle(pos=(5, None, None))
        parts = self.obs.observe(ctx.op, ctx)
        lat, lon = StubCoords().scaled(5)
        self.assertAlmostEqual(float(parts["veh_lat"][0]), lat, places=6)
        self.assertAlmostEqual(float(parts["veh_lon"][0]), lon, places=6)

    def test_vehicle_position_interpolates_along_the_edge_by_rel_pos(self):
        # rel_pos is the fraction ALREADY TRAVELLED: return_position_coordinates forms
        # rel * c_dest + (1 - rel) * c_origin, so rel = 0 is the origin node
        coords = StubCoords()
        for rel in (0.0, 0.25, 0.5, 1.0):
            with self.subTest(rel=rel):
                ctx = StubCtx(make_candidates([100.0], [-1.0]))
                ctx.op.sim_vehicles[0] = StubVehicle(pos=(2, 8, rel))
                parts = self.obs.observe(ctx.op, ctx)
                o_lat, o_lon = coords.scaled(2)
                d_lat, d_lon = coords.scaled(8)
                self.assertAlmostEqual(float(parts["veh_lat"][0]),
                                       o_lat + rel * (d_lat - o_lat), places=6)
                self.assertAlmostEqual(float(parts["veh_lon"][0]),
                                       o_lon + rel * (d_lon - o_lon), places=6)

    def test_relative_encoding_is_the_offset_from_the_request_origin(self):
        obs = candidate_observer(position_encoding="relative")
        ctx = StubCtx(make_candidates([100.0], [-1.0]), o_pos=(3, None, None))
        ctx.op.sim_vehicles[0] = StubVehicle(pos=(9, None, None))
        parts = obs.observe(ctx.op, ctx)
        v_lat, v_lon = StubCoords().scaled(9)
        o_lat, o_lon = StubCoords().scaled(3)
        self.assertAlmostEqual(float(parts["veh_lat"][0]), v_lat - o_lat, places=6)
        self.assertAlmostEqual(float(parts["veh_lon"][0]), v_lon - o_lon, places=6)

    def test_relative_encoding_is_zero_when_the_vehicle_sits_at_the_origin(self):
        obs = candidate_observer(position_encoding="relative")
        ctx = StubCtx(make_candidates([100.0], [-1.0]), o_pos=(6, None, None))
        ctx.op.sim_vehicles[0] = StubVehicle(pos=(6, None, None))
        parts = obs.observe(ctx.op, ctx)
        self.assertAlmostEqual(float(parts["veh_lat"][0]), 0.0, places=9)
        self.assertAlmostEqual(float(parts["veh_lon"][0]), 0.0, places=9)

    # ------------------------------------------------------------------ #
    # features 6-7: capacity and stops
    # ------------------------------------------------------------------ #

    def test_free_seats_counts_passengers_not_rider_objects(self):
        cases = [((), 4, 1.0),
                 ((StubRider(1), StubRider(2)), 4, 0.5),
                 ((StubRider(1, nr_pax=3),), 4, 0.25),
                 ((StubRider(1, nr_pax=4),), 4, 0.0)]
        for pax, max_pax, want in cases:
            with self.subTest(n=len(pax)):
                ctx = StubCtx(make_candidates([100.0], [-1.0]))
                ctx.op.sim_vehicles[0] = StubVehicle(pax=pax, max_pax=max_pax)
                parts = self.obs.observe(ctx.op, ctx)
                self.assertAlmostEqual(float(parts["free_seats"][0]), want, places=6)

    def test_remaining_stops_is_the_raw_pre_insertion_count(self):
        for n in (0, 1, 3):
            with self.subTest(n=n):
                ctx = StubCtx(make_candidates([100.0], [-1.0]))
                stops = [StubStop((i + 2, None, None)) for i in range(n)]
                ctx.op.veh_plans[0] = StubPlan({}, stops=stops)
                parts = self.obs.observe(ctx.op, ctx)
                self.assertEqual(float(parts["remaining_stops"][0]), float(n))

    def test_remaining_stops_is_pre_insertion_not_the_candidate_plan(self):
        ctx = StubCtx(make_candidates([100.0], [-1.0]))
        ctx.op.veh_plans[0] = StubPlan({}, stops=[StubStop((4, None, None))])
        # the candidate plan carries the new rider's two extra stops
        ctx.candidates[0][1].list_plan_stops = [StubStop((4, None, None)),
                                                StubStop((5, None, None)),
                                                StubStop((6, None, None))]
        parts = self.obs.observe(ctx.op, ctx)
        self.assertEqual(float(parts["remaining_stops"][0]), 1.0)

    # ------------------------------------------------------------------ #
    # feature 8: directionality
    # ------------------------------------------------------------------ #

    def _directionality(self, veh_pos, route_end, o_pos, d_pos, coords=None):
        nw = StubNetwork(coords=coords)
        ctx = StubCtx(make_candidates([100.0], [-1.0]), o_pos=o_pos, d_pos=d_pos,
                      op=StubOperator(make_candidates([100.0], [-1.0]),
                                      routing_engine=nw))
        ctx.op.sim_vehicles[0] = StubVehicle(
            pos=veh_pos,
            assigned_route=[StubLeg(route_end)] if route_end is not None else [])
        return float(self.obs.observe(ctx.op, ctx)["directionality"][0])

    def test_directionality_of_a_parallel_vehicle_is_one(self):
        coords = {1: (0.0, 0.0), 2: (100.0, 0.0), 3: (0.0, 0.0), 4: (50.0, 0.0)}
        self.assertAlmostEqual(
            self._directionality((3, None, None), (4, None, None),
                                 (1, None, None), (2, None, None), coords), 1.0, places=6)

    def test_directionality_of_an_opposed_vehicle_is_minus_one(self):
        coords = {1: (0.0, 0.0), 2: (100.0, 0.0), 3: (50.0, 0.0), 4: (0.0, 0.0)}
        self.assertAlmostEqual(
            self._directionality((3, None, None), (4, None, None),
                                 (1, None, None), (2, None, None), coords), -1.0, places=6)

    def test_directionality_of_a_perpendicular_vehicle_is_zero(self):
        coords = {1: (0.0, 0.0), 2: (100.0, 0.0), 3: (0.0, 0.0), 4: (0.0, 100.0)}
        self.assertAlmostEqual(
            self._directionality((3, None, None), (4, None, None),
                                 (1, None, None), (2, None, None), coords), 0.0, places=6)

    def test_directionality_matches_a_hand_dot_product(self):
        coords = {1: (0.0, 0.0), 2: (3.0, 4.0), 3: (0.0, 0.0), 4: (1.0, 0.0)}
        # request direction (0.6, 0.8); vehicle direction (1, 0) -> cosine 0.6
        self.assertAlmostEqual(
            self._directionality((3, None, None), (4, None, None),
                                 (1, None, None), (2, None, None), coords), 0.6, places=6)

    def test_directionality_is_zero_for_an_idle_vehicle(self):
        # no assigned route at all: `filter_directionality` treats this as perpendicular, and
        # with repositioning off `remaining_stops == 0` is what separates it from a real zero
        coords = {1: (0.0, 0.0), 2: (100.0, 0.0), 3: (0.0, 0.0)}
        self.assertEqual(
            self._directionality((3, None, None), None,
                                 (1, None, None), (2, None, None), coords), 0.0)

    def test_directionality_is_zero_when_the_route_ends_where_the_vehicle_is(self):
        coords = {1: (0.0, 0.0), 2: (100.0, 0.0), 3: (7.0, 9.0)}
        self.assertEqual(
            self._directionality((3, None, None), (3, None, None),
                                 (1, None, None), (2, None, None), coords), 0.0)

    # ------------------------------------------------------------------ #
    # feature 11: added distance
    # ------------------------------------------------------------------ #

    def test_added_distance_is_the_difference_of_the_two_route_distances_in_km(self):
        nw = StubNetwork(coords={0: (0.0, 0.0), 1: (0.0, 0.0), 2: (1000.0, 0.0),
                                 3: (3000.0, 0.0)})
        cands = [(0, StubPlan({7: [30100.0, 30700.0]},
                              stops=[StubStop((2, None, None)), StubStop((3, None, None))]),
                  -1.0)]
        op = StubOperator(cands, routing_engine=nw)
        op.sim_vehicles[0] = StubVehicle(pos=(1, None, None))
        op.veh_plans[0] = StubPlan({}, stops=[StubStop((3, None, None))])
        ctx = StubCtx(cands, op=op)
        parts = self.obs.observe(op, ctx)
        # pre: 1 -> 3 = 3000 m. post: 1 -> 2 -> 3 = 1000 + 2000 = 3000 m. difference 0
        self.assertAlmostEqual(float(parts["added_distance"][0]), 0.0, places=9)

    def test_added_distance_keeps_its_sign_when_the_insertion_shortens_the_route(self):
        # routes minimise time rather than distance and stops can merge, so a negative value is
        # real and must not be clipped (§5.2)
        nw = StubNetwork(coords={0: (0.0, 0.0), 1: (0.0, 0.0), 2: (500.0, 0.0),
                                 3: (4000.0, 0.0)})
        cands = [(0, StubPlan({7: [30100.0, 30700.0]},
                              stops=[StubStop((2, None, None))]), -1.0)]
        op = StubOperator(cands, routing_engine=nw)
        op.sim_vehicles[0] = StubVehicle(pos=(1, None, None))
        op.veh_plans[0] = StubPlan({}, stops=[StubStop((3, None, None))])
        ctx = StubCtx(cands, op=op)
        parts = self.obs.observe(op, ctx)
        self.assertAlmostEqual(float(parts["added_distance"][0]), (500.0 - 4000.0) / 1000.0,
                               places=9)

    def test_added_distance_is_zero_for_two_empty_plans(self):
        ctx = StubCtx(make_candidates([100.0], [-1.0]))
        self.assertEqual(float(self.obs.observe(ctx.op, ctx)["added_distance"][0]), 0.0)

    # ------------------------------------------------------------------ #
    # features 9, 10, 12: pending riders
    # ------------------------------------------------------------------ #

    @staticmethod
    def _lateness_ctx(onboard=(), pending=(101, 102), quotes=True, plan_times=True):
        """Two pending riders, 101 quoted 1300 and 102 quoted 2100.

        Re-timed pre-insertion pickups 1400 and 2000; under plan `k` 1500 and 2050. So
        `lateness_pre = max(+100, -100) = +100`, `lateness_post = max(+200, -50) = +200`,
        `added_pickup_delay = 100 + 50 = 150`.
        """
        post = {7: [30100.0, 30700.0]}
        pre_retimed = {}
        if plan_times:
            post.update({101: [1500.0, 9000.0], 102: [2050.0, 9000.0]})
            pre_retimed = {101: [1400.0, 9000.0], 102: [2000.0, 9000.0]}
        cands = [(0, StubPlan(post), -5.0)]
        rq_dict = {}
        if quotes:
            rq_dict = {101: StubPrq(1000.0, offer=StubOffer(300.0)),
                       102: StubPrq(2000.0, offer=StubOffer(100.0))}
        op = StubOperator(cands, rq_dict=rq_dict)
        op.sim_vehicles[0] = StubVehicle(pax=[StubRider(r) for r in onboard])
        op.veh_plans[0] = StubPlan(
            {}, stops=[StubStop((4, None, None), boarding=list(pending) + list(onboard))],
            retimed_pax_info=pre_retimed)
        return StubCtx(cands, op=op)

    def test_lateness_and_added_delay_match_a_hand_computation(self):
        ctx = self._lateness_ctx()
        parts = self.obs.observe(ctx.op, ctx)
        self.assertAlmostEqual(float(parts["lateness_pre"][0]), 100.0 / MAX_WT, places=6)
        self.assertAlmostEqual(float(parts["lateness_post"][0]), 200.0 / MAX_WT, places=6)
        self.assertAlmostEqual(float(parts["added_pickup_delay"][0]), 150.0 / MAX_WT,
                               places=6)

    def test_lateness_is_a_max_and_added_delay_a_sum(self):
        # the two differ in kind: the worst promise against the total pushed onto others
        ctx = self._lateness_ctx()
        parts = self.obs.observe(ctx.op, ctx)
        self.assertGreater(float(parts["added_pickup_delay"][0]),
                           float(parts["lateness_post"][0]) - float(parts["lateness_pre"][0]))

    def test_lateness_keeps_a_negative_sign(self):
        # common while the planner assumes 300 s per stop against a realized mean near 56 s
        ctx = self._lateness_ctx(pending=(102,))
        parts = self.obs.observe(ctx.op, ctx)
        self.assertAlmostEqual(float(parts["lateness_pre"][0]), -100.0 / MAX_WT, places=6)
        self.assertAlmostEqual(float(parts["lateness_post"][0]), -50.0 / MAX_WT, places=6)

    def test_no_pending_riders_gives_zero_for_all_three(self):
        ctx = self._lateness_ctx(pending=())
        parts = self.obs.observe(ctx.op, ctx)
        for name in ("lateness_pre", "lateness_post", "added_pickup_delay"):
            self.assertEqual(float(parts[name][0]), 0.0, name)

    def test_onboard_riders_are_not_pending(self):
        # they can no longer cancel and the reward carries no ride-time term (§10, §5.2)
        ctx = self._lateness_ctx(onboard=(101,), pending=(102,))
        parts = self.obs.observe(ctx.op, ctx)
        self.assertAlmostEqual(float(parts["lateness_pre"][0]), -100.0 / MAX_WT, places=6)
        self.assertAlmostEqual(float(parts["added_pickup_delay"][0]), 50.0 / MAX_WT, places=6)

    def test_a_rider_whose_boarding_is_in_progress_counts_as_onboard(self):
        # a rider joins veh_obj.pax when the boarding leg STARTS (Vehicles.py:254), which is
        # why get_nr_pax_without_currently_boarding exists; such a rider is not pending
        ctx = self._lateness_ctx(onboard=(101, 102), pending=())
        parts = self.obs.observe(ctx.op, ctx)
        for name in ("lateness_pre", "lateness_post", "added_pickup_delay"):
            self.assertEqual(float(parts[name][0]), 0.0, name)

    def test_the_new_rider_is_never_pending(self):
        cands = [(0, StubPlan({7: [30100.0, 30700.0]}), -1.0)]
        op = StubOperator(cands, rq_dict={7: StubPrq(29000.0, offer=StubOffer(100.0))})
        op.veh_plans[0] = StubPlan({}, stops=[StubStop((4, None, None), boarding=[7])],
                                   retimed_pax_info={7: [30000.0, 30600.0]})
        ctx = StubCtx(cands, op=op)
        parts = self.obs.observe(op, ctx)
        for name in ("lateness_pre", "lateness_post", "added_pickup_delay"):
            self.assertEqual(float(parts[name][0]), 0.0, name)

    def test_a_pending_rider_without_a_quote_is_counted_not_guessed(self):
        ctx = self._lateness_ctx(quotes=False)
        parts = self.obs.observe(ctx.op, ctx)
        self.assertEqual(float(parts["lateness_pre"][0]), 0.0)
        self.assertEqual(float(parts["lateness_post"][0]), 0.0)
        # feature 12 needs no quote, so it is still computed
        self.assertAlmostEqual(float(parts["added_pickup_delay"][0]), 150.0 / MAX_WT,
                               places=6)
        self.assertEqual(self.obs.diagnostics["riders_without_quote"], 2)

    def test_a_pending_rider_without_a_plan_time_is_counted_not_guessed(self):
        # the re-timer skips a boarding rid that is in no_shows_store_for_veh
        ctx = self._lateness_ctx(plan_times=False)
        parts = self.obs.observe(ctx.op, ctx)
        for name in ("lateness_pre", "lateness_post", "added_pickup_delay"):
            self.assertEqual(float(parts[name][0]), 0.0, name)
        self.assertEqual(self.obs.diagnostics["riders_without_plan_time"], 2)

    # ------------------------------------------------------------------ #
    # padding
    # ------------------------------------------------------------------ #

    def test_every_feature_of_an_unoccupied_slot_is_zero(self):
        ctx = self._lateness_ctx()
        parts = self.obs.observe(ctx.op, ctx)
        for name in CandidateObserver.FEATURE_NAMES:
            np.testing.assert_array_equal(
                parts[name][1:], np.zeros(K - 1, dtype=np.float32), name)

    def test_no_nan_or_inf_across_the_whole_block(self):
        ctx = self._lateness_ctx()
        vec = self.obs.observe_vector(ctx.op, ctx)
        self.assertEqual(vec.shape, (12 * K,))
        self.assertTrue(np.all(np.isfinite(vec)))


class TestCandidateBlockLeavesSimulationStateAlone(unittest.TestCase):
    """Trap 14 and §5.2 exception (b), at the unit level.

    The real arbiter is the P1.10 gate re-run; these fail faster and say which rule broke.
    """

    def setUp(self):
        self.obs = candidate_observer()

    def _ctx_with_pending(self):
        cands = [(0, StubPlan({7: [30100.0, 30700.0], 101: [1500.0, 9000.0]}),
                  -5.0)]
        op = StubOperator(cands, rq_dict={101: StubPrq(1000.0, offer=StubOffer(300.0))})
        op.veh_plans[0] = StubPlan(
            {101: [1234.0, 5678.0]},
            stops=[StubStop((4, None, None), boarding=[101]), StubStop((6, None, None))],
            utility=-42.0,
            retimed_pax_info={101: [1400.0, 9000.0]})
        op.sim_vehicles[0] = StubVehicle(pos=(1, None, None))
        return StubCtx(cands, op=op)

    def test_the_stored_plan_is_never_re_timed_only_a_copy_is(self):
        ctx = self._ctx_with_pending()
        stored = ctx.op.veh_plans[0]
        self.obs.observe(ctx.op, ctx)
        self.assertEqual(stored.retimed_on_self, 0,
                         "trap 14: the stored plan was re-timed in place")
        self.assertEqual(stored.copies_retimed, 1, "exactly one re-timed copy per candidate")

    def test_the_stored_plans_stops_pax_info_and_utility_survive(self):
        ctx = self._ctx_with_pending()
        stored = ctx.op.veh_plans[0]
        before_pax = {k: list(v) for k, v in stored.pax_info.items()}
        before_stops = [(s.get_pos(), s.get_list_boarding_rids()) for s in
                        stored.list_plan_stops]
        before_utility = stored.utility
        self.obs.observe(ctx.op, ctx)
        self.assertEqual({k: list(v) for k, v in stored.pax_info.items()}, before_pax)
        self.assertEqual([(s.get_pos(), s.get_list_boarding_rids())
                          for s in stored.list_plan_stops], before_stops)
        self.assertEqual(stored.utility, before_utility)

    def test_no_copy_is_made_when_there_are_no_pending_riders(self):
        ctx = StubCtx(make_candidates([100.0], [-1.0]))
        self.obs.observe(ctx.op, ctx)
        self.assertEqual(ctx.op.veh_plans[0].copies_retimed, 0)
        self.assertEqual(self.obs.diagnostics["retimings"], 0)

    def test_one_copy_per_candidate_vehicle_shared_by_both_features(self):
        ctx = self._ctx_with_pending()
        self.obs.observe(ctx.op, ctx)
        # features 9 and 12 share one re-timed copy (§5.2), so one per candidate vehicle
        self.assertEqual(self.obs.diagnostics["retimings"], 1)

    # ------------------------------------------------------------------ #
    # the routing cache (§5.2 exception b)
    # ------------------------------------------------------------------ #

    def _routing_ctx(self, network=None):
        nw = network or StubNetwork()
        cands = [(0, StubPlan({7: [30100.0, 30700.0]},
                              stops=[StubStop((2, None, None)), StubStop((3, None, None))]),
                  -1.0)]
        op = StubOperator(cands, routing_engine=nw)
        op.sim_vehicles[0] = StubVehicle(pos=(1, None, None))
        op.veh_plans[0] = StubPlan({}, stops=[StubStop((3, None, None))])
        return StubCtx(cands, op=op), nw

    def test_the_cache_holds_the_same_keys_before_and_after(self):
        ctx, nw = self._routing_ctx()
        before = set(nw.travel_time_infos)
        self.obs.observe(ctx.op, ctx)
        self.assertEqual(set(nw.travel_time_infos), before)

    def test_entries_the_observer_adds_are_deleted_and_counted(self):
        ctx, nw = self._routing_ctx()
        self.obs.observe(ctx.op, ctx)
        self.assertEqual(nw.travel_time_infos, {})
        # the restore deletes exactly what was added, which is also the number of genuine route
        # computations the observer triggered
        self.assertGreater(self.obs.diagnostics["cache_entries_restored"], 0)
        self.assertGreater(len(nw.queries), 0)

    def test_the_simulations_own_entries_survive_untouched(self):
        ctx, nw = self._routing_ctx()
        nw.travel_time_infos[(1, 3)] = ("sentinel", 1.0, 2.0)
        nw.travel_time_infos[(99, 98)] = ("other", 3.0, 4.0)
        self.obs.observe(ctx.op, ctx)
        self.assertEqual(nw.travel_time_infos,
                         {(1, 3): ("sentinel", 1.0, 2.0), (99, 98): ("other", 3.0, 4.0)})

    def test_the_restore_runs_even_when_a_feature_raises(self):
        ctx, nw = self._routing_ctx()

        def explode(*_a, **_kw):
            raise RuntimeError("feature bug")

        # make the pending-rider path raise after the routing has already cached something
        self.obs._pending_rids = explode
        with self.assertRaises(RuntimeError):
            self.obs.observe(ctx.op, ctx)
        self.assertEqual(nw.travel_time_infos, {},
                         "the restore must sit in a finally, or a feature bug leaves the "
                         "simulation's routing cache polluted")

    def test_without_the_restore_the_cache_would_be_polluted(self):
        """The counter-example, so the restore test above cannot pass vacuously.

        If the observer stopped adding entries — a future routing engine that caches nothing,
        say — `test_entries_the_observer_adds_are_deleted_and_counted` would still pass while
        testing nothing. This pins that there is really something to restore.
        """
        class NoRestore(CandidateObserver):
            def _restore_cache(self, routing_engine, before):
                pass

        obs = NoRestore(K, SCENARIO, StubCoords())
        ctx, nw = self._routing_ctx()
        obs.observe(ctx.op, ctx)
        self.assertNotEqual(nw.travel_time_infos, {},
                            "the observer's routing must actually write to the cache, or the "
                            "restore tests prove nothing")

    def test_a_network_without_a_cache_needs_no_restore(self):
        ctx, nw = self._routing_ctx(network=StubNetwork(with_cache=False))
        self.obs.observe(ctx.op, ctx)        # must not raise
        self.assertFalse(hasattr(nw, "travel_time_infos"))
        self.assertEqual(self.obs.diagnostics["cache_entries_restored"], 0)

    def test_the_cache_is_read_through_the_attribute_not_held(self):
        """A rebind mid-observation must not resurrect the old dict's keys.

        `_reset_internal_attributes_after_travel_time_update` **rebinds**
        `travel_time_infos` to a fresh dict rather than clearing it, and `load_tt_file` does
        that at every operator travel-time file load. Code that snapshots the dict object
        instead of the attribute goes stale from the first load onwards, and nothing raises.
        """
        class RebindingNetwork(StubNetwork):
            def __init__(self):
                super().__init__()
                self.rebound = False

            def return_travel_costs_1to1(self, o, d, customized_section_cost_function=None):
                if not self.rebound:
                    # as load_tt_file does, between the snapshot and the restore
                    self.travel_time_infos = {}
                    self.rebound = True
                return super().return_travel_costs_1to1(o, d)

        ctx, nw = self._routing_ctx(network=RebindingNetwork())
        nw.travel_time_infos[("stale", "key")] = (0.0, 0.0, 0.0)
        self.obs.observe(ctx.op, ctx)
        self.assertTrue(nw.rebound)
        # everything in the new dict was computed by the observer, so all of it goes
        self.assertEqual(nw.travel_time_infos, {})


class TestArrivalBaseline(unittest.TestCase):
    """P2.6's table as the anomaly reads it, plus guards 1 and 2 (P2.11)."""

    def setUp(self):
        from src.rl_gym.observers import ArrivalBaseline
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path, self.manifest, self.rows = write_baseline(self.tmp.name)
        self.baseline = ArrivalBaseline(self.manifest, self.path)

    def test_mean_is_read_per_month_and_minute(self):
        for month, t, window_s, mean, _sd in self.rows[:40]:
            self.assertAlmostEqual(self.baseline.expected(month, t, window_s), mean, places=9)

    def test_sd_is_pooled_across_months(self):
        """P2.6 pools the SD over all training days; the committed table reflects that.

        Verified against the real artifact in P2.11: exactly one distinct `sd` per
        `(t_seconds, window_s)` across all ten months. Here the fixture deliberately writes a
        *per-month* sd, so this test pins what the lookup does — it is keyed by `(t, window)`
        and the last row written wins — rather than claiming the fixture is pooled.
        """
        from src.rl_gym.corpus import read_baseline
        committed = read_baseline()
        by_key = {}
        for month, t, window_s, _mean, sd in committed:
            by_key.setdefault((t, window_s), set()).add(sd)
        self.assertEqual(max(len(v) for v in by_key.values()), 1,
                         "the committed baseline's sd must be pooled across months (P2.6)")

    def test_z_uses_the_sd_floor(self):
        """`z = (observed − expected) / max(sd, sd_floor)` (§5.2), tested both sides of it.

        Explicit rows rather than the shared fixture: its `sd` is keyed by `(t, window)`, so
        the last month written wins and which value lands there is not something to reason
        about in a test about the floor.
        """
        from src.rl_gym.observers import ArrivalBaseline
        rows = [(6, 25200, 1800, 10.0, 0.25),      # sd below the floor
                (6, 25260, 1800, 10.0, 4.0),       # sd above it
                (6, 25200, 3600, 10.0, 0.25),
                (6, 25260, 3600, 10.0, 4.0)]
        path, manifest, _rows = write_baseline(os.path.join(self.tmp.name), rows=rows)
        baseline = ArrivalBaseline(manifest, path)
        self.assertEqual(baseline.sd_floor, 1.0)

        # sd 0.25 -> the floor of 1.0 applies, so a 3-arrival excess is 3 SD, not 12
        exp, z = baseline.z(13.0, 6, 25200, 1800)
        self.assertEqual(exp, 10.0)
        self.assertAlmostEqual(z, 3.0, places=9)

        # sd 4.0 -> the real sd applies
        exp, z = baseline.z(14.0, 6, 25260, 1800)
        self.assertEqual(exp, 10.0)
        self.assertAlmostEqual(z, 1.0, places=9)

    def test_z_is_zero_when_observed_equals_expected(self):
        month, t, window_s = 6, 25200, 3600
        exp = self.baseline.expected(month, t, window_s)
        _e, z = self.baseline.z(exp, month, t, window_s)
        self.assertEqual(z, 0.0)

    def test_z_keeps_its_sign(self):
        month, t, window_s = 6, 25200, 1800
        exp = self.baseline.expected(month, t, window_s)
        self.assertLess(self.baseline.z(exp - 5.0, month, t, window_s)[1], 0.0)
        self.assertGreater(self.baseline.z(exp + 5.0, month, t, window_s)[1], 0.0)

    def test_floor_minute(self):
        for sim_time, want in ((25200, 25200), (25259.9, 25200), (25260, 25260),
                               (33720, 33720), (33725, 33720)):
            self.assertEqual(self.baseline.floor_minute(sim_time), want)

    def test_an_unknown_month_or_minute_raises_rather_than_returning_zero(self):
        with self.assertRaises(KeyError):
            self.baseline.expected(11, 25200, 1800)
        with self.assertRaises(KeyError):
            self.baseline.expected(6, 25230, 1800)       # not on the minute grid
        with self.assertRaises(KeyError):
            self.baseline.sd(99999, 1800)

    # ---- guard 1 ---------------------------------------------------- #

    def test_check_scenario_rejects_an_uncovered_window(self):
        with self.assertRaises(ValueError) as cm:
            self.baseline.check_scenario(25200, 70000, (1800, 7200))
        self.assertIn("7200", str(cm.exception))

    def test_check_scenario_rejects_a_scenario_outside_the_minute_grid(self):
        with self.assertRaises(ValueError) as cm:
            self.baseline.check_scenario(0.0, 70000, (1800, 3600))
        self.assertIn("covers minutes", str(cm.exception))
        with self.assertRaises(ValueError):
            self.baseline.check_scenario(25200, 100000, (1800, 3600))

    def test_check_scenario_accepts_the_reference_window(self):
        self.baseline.check_scenario(25200, 70000, (1800, 3600))      # must not raise

    # ---- guard 2 ---------------------------------------------------- #

    def test_an_edited_baseline_fails_the_manifest_hash(self):
        from src.rl_gym.observers import ArrivalBaseline
        edited = list(self.rows)
        month, t, window_s, mean, sd = edited[5]
        edited[5] = (month, t, window_s, mean + 0.5, sd)
        path, _manifest, _rows = write_baseline(os.path.join(self.tmp.name), rows=edited)
        with self.assertRaises(ValueError) as cm:
            ArrivalBaseline(self.manifest, path)         # the OLD manifest
        self.assertIn("baseline_hash", str(cm.exception))
        self.assertIn("trap 16", str(cm.exception))

    def test_a_manifest_without_a_baseline_hash_raises(self):
        from src.rl_gym.observers import ArrivalBaseline
        bad = {k: v for k, v in self.manifest.items() if k != "baseline_hash"}
        with self.assertRaises(ValueError) as cm:
            ArrivalBaseline(bad, self.path)
        self.assertIn("baseline_hash", str(cm.exception))

    def test_a_missing_baseline_file_names_the_build_command(self):
        from src.rl_gym.observers import ArrivalBaseline
        with self.assertRaises(ValueError) as cm:
            ArrivalBaseline(self.manifest, os.path.join(self.tmp.name, "nope.csv"))
        self.assertIn("build_arrival_baseline", str(cm.exception))

    def test_the_committed_artifacts_agree_with_each_other(self):
        """The real pair, through the runtime path rather than the builder's `--check`."""
        from src.rl_gym.corpus import load_manifest
        from src.rl_gym.observers import ArrivalBaseline
        baseline = ArrivalBaseline(load_manifest())      # raises if they disagree
        self.assertEqual(len(baseline.rows), 14940)
        self.assertEqual(baseline.months, list(range(1, 11)))
        self.assertEqual(baseline.windows_s, [1800, 3600])
        self.assertEqual(len(baseline.minutes), 747)
        baseline.check_scenario(25200, 70000, (1800, 3600))


class TestActiveFleetSchedule(unittest.TestCase):
    """`active_*` from `op_act_fs_file`, with `TimeBasedFS`'s own formula (§5.2, P2.11)."""

    def setUp(self):
        from src.rl_gym.observers import ActiveFleetSchedule
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.schedule = ActiveFleetSchedule(write_active_fleet(self.tmp.name))

    def test_share_in_force_is_the_last_row_at_or_below_t(self):
        for t, want in ((25200, 0.7), (28799, 0.7), (28800, 0.9), (35999, 0.9),
                        (36000, 1.0), (61199, 1.0), (61200, 0.9)):
            with self.subTest(t=t):
                self.assertAlmostEqual(self.schedule.share_at(t), want, places=9)

    def test_the_last_row_applies_beyond_the_end_of_the_file(self):
        # §5.2: the schedule's last row holds for the rest of the day
        self.assertAlmostEqual(self.schedule.share_at(70000), 0.9, places=9)
        self.assertAlmostEqual(self.schedule.share_at(200000), 0.9, places=9)

    def test_before_the_first_row_the_first_share_applies(self):
        # unreachable in the reference scenario, whose first row is exactly start_time, but a
        # clamp beats an IndexError
        self.assertAlmostEqual(self.schedule.share_at(0), 0.7, places=9)

    def test_the_rounding_is_the_point_not_the_share(self):
        # TimeBasedFS: min(max(int(np.round(share * n)), 0), n). 0.7 * 12 = 8.4 -> 8 -> 0.667,
        # which is NOT 0.7 -- which is why nr_vehicles is needed at all
        self.assertAlmostEqual(self.schedule.active_fraction(25200, 12), 8 / 12, places=9)
        self.assertNotAlmostEqual(self.schedule.active_fraction(25200, 12), 0.7, places=3)

    def test_the_reference_days_steps(self):
        """The figures §8's P2.11 expects, times the fleet of 12."""
        for t, want in ((25200, 8), (28800, 11), (32400, 11), (36000, 12),
                        (57600, 12), (61200, 11), (64800, 11)):
            with self.subTest(t=t):
                self.assertEqual(round(self.schedule.active_fraction(t, 12) * 12), want)

    def test_the_formula_clamps_to_the_fleet(self):
        from src.rl_gym.observers import ActiveFleetSchedule
        sched = ActiveFleetSchedule(write_active_fleet(
            self.tmp.name, pairs=((25200, 1.4), (28800, -0.2))))
        self.assertEqual(sched.active_fraction(25200, 12), 1.0)
        self.assertEqual(sched.active_fraction(28800, 12), 0.0)

    def test_no_vehicles_gives_zero_rather_than_dividing_by_zero(self):
        self.assertEqual(self.schedule.active_fraction(25200, 0), 0.0)

    def test_a_missing_file_or_column_raises(self):
        from src.rl_gym.observers import ActiveFleetSchedule
        with self.assertRaises(ValueError):
            ActiveFleetSchedule(os.path.join(self.tmp.name, "nope.csv"))
        bad = os.path.join(self.tmp.name, "bad.csv")
        with open(bad, "w") as fh:
            fh.write("time,something_else\n25200,0.7\n")
        with self.assertRaises(ValueError) as cm:
            ActiveFleetSchedule(bad)
        self.assertIn("share_active_fleet_size", str(cm.exception))

    def test_the_committed_file_reproduces_the_reference_curve(self):
        from src.rl_gym.observers import ActiveFleetSchedule
        real = ActiveFleetSchedule(os.path.join(
            os.path.dirname(__file__), "..", "data", "fleetctrl", "elastic_fleet_size",
            "example_active_fleetsize.csv"))
        for t, want in ((25200, 8), (28800, 11), (36000, 12), (61200, 11), (69999, 11)):
            with self.subTest(t=t):
                self.assertEqual(round(real.active_fraction(t, 12) * 12), want)


class TestGlobalBlock(unittest.TestCase):
    """§5.2's 15 new global features (P2.11)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fc = StubFleetCtrl([VRL_STATES.IDLE] * 12)
        self.fc.nr_vehicles = 12

    def _observer(self, counts=None, wait_gap=None, month=6, **kw):
        obs = global_observer(self.tmp.name, **kw)
        self.tracker = _RecordingTracker(counts=counts, wait_gap=wait_gap)
        obs.set_stats_tracker(self.tracker)
        obs.set_episode_month(month)
        return obs

    def test_nineteen_features_in_sect_5_2_order(self):
        names = [n for n, _w in self._observer().feature_spec()]
        self.assertEqual(names, [
            "time_sin", "time_cos", "day_progress", "idle_fraction",
            "arrivals_30", "arrivals_60",
            "arrivals_expected_30", "arrivals_expected_60",
            "arrivals_z_30", "arrivals_z_60",
            "declines_30", "declines_60", "cancels_30", "cancels_60",
            "wait_gap_30", "wait_gap_60",
            "active_now", "active_30", "active_60"])
        self.assertEqual(len(names), 19)

    def test_the_four_phase_one_features_keep_their_names_and_positions(self):
        self.assertEqual([n for n, _w in self._observer().feature_spec()[:4]],
                         ["time_sin", "time_cos", "day_progress", "idle_fraction"])

    def test_counts_come_from_the_tracker_fleet_wide(self):
        counts = {("arrival", 1800): 14, ("arrival", 3600): 34,
                  ("decline", 1800): 3, ("decline", 3600): 7,
                  ("cancel", 1800): 1, ("cancel", 3600): 5}
        obs = self._observer(counts=counts)
        parts = obs.observe(self.fc, StubCtx([], sim_time=33720.0))
        self.assertEqual(float(parts["arrivals_30"]), 14.0)
        self.assertEqual(float(parts["arrivals_60"]), 34.0)
        self.assertEqual(float(parts["declines_30"]), 3.0)
        self.assertEqual(float(parts["declines_60"]), 7.0)
        self.assertEqual(float(parts["cancels_30"]), 1.0)
        self.assertEqual(float(parts["cancels_60"]), 5.0)
        # fleet-wide means `cells` is not passed (§5.2); the zone counts are the request block's
        self.assertTrue(all(cells is None for _k, _w, cells in self.tracker.fleet_calls))

    def test_counts_are_raw_not_scaled(self):
        obs = self._observer(counts={("arrival", 1800): 97})
        parts = obs.observe(self.fc, StubCtx([], sim_time=33720.0))
        self.assertEqual(float(parts["arrivals_30"]), 97.0)

    def test_the_observed_count_and_the_anomaly_use_the_same_number(self):
        """§5.2: "the fleet-wide arrival count and the anomaly's observed count are the same
        number and appear once"."""
        obs = self._observer(counts={("arrival", 1800): 20, ("arrival", 3600): 50})
        t = 33720.0
        parts = obs.observe(self.fc, StubCtx([], sim_time=t))
        for window_s, key in ((1800, "30"), (3600, "60")):
            expected = obs.baseline.expected(6, obs.baseline.floor_minute(t), window_s)
            sd = max(obs.baseline.sd(obs.baseline.floor_minute(t), window_s),
                     obs.baseline.sd_floor)
            self.assertAlmostEqual(float(parts[f"arrivals_expected_{key}"]), expected,
                                   places=4)
            self.assertAlmostEqual(
                float(parts[f"arrivals_z_{key}"]),
                (float(parts[f"arrivals_{key}"]) - expected) / sd, places=4)

    def test_the_baseline_is_read_at_the_floored_minute(self):
        obs = self._observer(counts={("arrival", 1800): 5})
        a = obs.observe(self.fc, StubCtx([], sim_time=33720.0))
        b = obs.observe(self.fc, StubCtx([], sim_time=33779.0))     # same minute
        c = obs.observe(self.fc, StubCtx([], sim_time=33780.0))     # the next one
        self.assertEqual(float(a["arrivals_expected_30"]),
                         float(b["arrivals_expected_30"]))
        self.assertNotEqual(float(a["arrivals_expected_30"]),
                            float(c["arrivals_expected_30"]))

    def test_the_month_selects_the_row(self):
        counts = {("arrival", 1800): 10}
        t = 33720.0
        seen = {}
        for month in BASELINE_MONTHS:
            obs = self._observer(counts=counts, month=month)
            parts = obs.observe(self.fc, StubCtx([], sim_time=t))
            seen[month] = float(parts["arrivals_expected_30"])
        # the fixture's mean rises with the month, so a wrong month cannot read the same value
        self.assertEqual(sorted(seen, key=seen.get), sorted(BASELINE_MONTHS))

    def test_wait_gap_divides_by_the_wait_normalizer(self):
        obs = self._observer(wait_gap={1800: 150.0, 3600: -30.0})
        parts = obs.observe(self.fc, StubCtx([], sim_time=33720.0))
        self.assertAlmostEqual(float(parts["wait_gap_30"]), 150.0 / MAX_WT, places=6)
        self.assertAlmostEqual(float(parts["wait_gap_60"]), -30.0 / MAX_WT, places=6)

    def test_an_explicit_wait_normalizer_is_used_for_the_wait_gap(self):
        obs = self._observer(wait_gap={1800: 150.0, 3600: 150.0}, wait_normalizer_s=600.0)
        parts = obs.observe(self.fc, StubCtx([], sim_time=33720.0))
        self.assertAlmostEqual(float(parts["wait_gap_30"]), 150.0 / 600.0, places=6)

    def test_an_empty_window_writes_zero_for_the_wait_gap(self):
        # the tracker returns None when no pickup in the window carried a quote (§5.2)
        obs = self._observer(wait_gap=None)
        parts = obs.observe(self.fc, StubCtx([], sim_time=33720.0))
        self.assertEqual(float(parts["wait_gap_30"]), 0.0)
        self.assertEqual(float(parts["wait_gap_60"]), 0.0)

    def test_active_features_lead_by_thirty_and_sixty_minutes(self):
        obs = self._observer()
        # at 07:30 the schedule reads 0.7 now, 0.9 at 08:00 (+30) and 0.9 at 08:30 (+60)
        parts = obs.observe(self.fc, StubCtx([], sim_time=27000.0))
        self.assertAlmostEqual(float(parts["active_now"]), 8 / 12, places=6)
        self.assertAlmostEqual(float(parts["active_30"]), 11 / 12, places=6)
        self.assertAlmostEqual(float(parts["active_60"]), 11 / 12, places=6)

    def test_active_features_use_the_operators_fleet_size(self):
        obs = self._observer()
        small = StubFleetCtrl([VRL_STATES.IDLE] * 6)
        small.nr_vehicles = 6
        parts = obs.observe(small, StubCtx([], sim_time=25200.0))
        # 0.7 * 6 = 4.2 -> 4 -> 0.667, a different number from the 12-vehicle case
        self.assertAlmostEqual(float(parts["active_now"]), 4 / 6, places=6)

    def test_everything_is_finite_and_the_right_shape(self):
        obs = self._observer(counts={("arrival", 1800): 14}, wait_gap={1800: 10.0})
        vec = obs.observe_vector(self.fc, StubCtx([], sim_time=33720.0))
        self.assertEqual(vec.shape, (19,))
        self.assertEqual(vec.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(vec)))

    # ---- guard 3 and the injected dependencies ---------------------- #

    def test_an_unknown_month_raises_at_set_episode_month(self):
        for month in (None, 0, 11, 99):
            with self.subTest(month=month):
                obs = global_observer(self.tmp.name)
                with self.assertRaises(ValueError) as cm:
                    obs.set_episode_month(month)
                self.assertIn("month", str(cm.exception))

    def test_observe_raises_without_a_tracker(self):
        obs = global_observer(self.tmp.name)
        obs.set_episode_month(6)
        with self.assertRaises(RuntimeError) as cm:
            obs.observe(self.fc, StubCtx([], sim_time=33720.0))
        self.assertIn("rolling-stats tracker", str(cm.exception))

    def test_observe_raises_without_an_episode_month(self):
        obs = global_observer(self.tmp.name)
        obs.set_stats_tracker(_ZeroTracker())
        with self.assertRaises(RuntimeError) as cm:
            obs.observe(self.fc, StubCtx([], sim_time=33720.0))
        self.assertIn("episode month", str(cm.exception))

    def test_the_two_artifacts_are_required_not_optional(self):
        """A missing baseline could only mean zeros for 10 of the 19 features."""
        from src.rl_gym.observers import ActiveFleetSchedule
        with self.assertRaises(ValueError) as cm:
            GlobalStateObserver(SCENARIO, None,
                                ActiveFleetSchedule(write_active_fleet(self.tmp.name)))
        self.assertIn("zeros", str(cm.exception))
        path, manifest, _rows = write_baseline(self.tmp.name)
        from src.rl_gym.observers import ArrivalBaseline
        with self.assertRaises(ValueError):
            GlobalStateObserver(SCENARIO, ArrivalBaseline(manifest, path), None)

    def test_the_windows_are_checked_against_the_baseline_at_construction(self):
        """Guard 1: a window the baseline has no rows for must fail here, not mid-episode."""
        from src.rl_gym.observers import ActiveFleetSchedule, ArrivalBaseline

        class WiderWindows(GlobalStateObserver):
            WINDOWS_S = (1800, 7200)

        path, manifest, _rows = write_baseline(self.tmp.name)
        with self.assertRaises(ValueError) as cm:
            WiderWindows(SCENARIO, ArrivalBaseline(manifest, path),
                         ActiveFleetSchedule(write_active_fleet(self.tmp.name)))
        self.assertIn("7200", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
