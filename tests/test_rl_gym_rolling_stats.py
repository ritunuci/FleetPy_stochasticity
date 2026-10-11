"""Tests for src/rl_gym/rolling_stats.py and the P2.7 wiring.

`python -m unittest discover tests`. Mostly unit tests against a stub zone table; the two
classes at the end build real simulations and are the slow part.

The reconciliation against `1_user-stats.csv` over a full greedy day lives in P2.7's
verification script, not here -- it needs a 40-second episode with output on.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rl_gym.reward import RewardTracker, classify_exit  # noqa: E402
from src.rl_gym.rolling_stats import MAX_WINDOW_S, RollingStatsTracker  # noqa: E402

SCS = os.path.join(os.path.dirname(__file__), "..", "studies", "example_study", "scenarios")


class StubZones:
    """Nodes 0-5 in cell A, 6-9 in cell B, 10+ unknown. A and B are neighbours."""

    RING = {"A": frozenset({"A", "B"}), "B": frozenset({"A", "B"}), "C": frozenset({"C"})}

    def cell_of_node(self, node):
        n = int(node)
        if 0 <= n <= 5:
            return "A"
        if 6 <= n <= 9:
            return "B"
        raise KeyError(f"node {node} is not in the stub network")

    def zone_set(self, cell):
        return self.RING[cell]

    def zone_set_of_node(self, node):
        return self.zone_set(self.cell_of_node(node))


class StubRq:
    def __init__(self, rq_time=1000.0, pu_time=None, do_time=None,
                 diffusion_cancelled=False, rider_declined=False, no_show=False,
                 chosen_operator_id=None):
        self.rq_time = rq_time
        self.pu_time = pu_time
        self.do_time = do_time
        self.diffusion_cancelled = diffusion_cancelled
        self.rider_declined = rider_declined
        self.no_show = no_show
        self.chosen_operator_id = chosen_operator_id


def tracker(**kw):
    return RollingStatsTracker(StubZones(), **kw)


class TestClockAndWindows(unittest.TestCase):

    def test_counts_are_zero_before_a_clock_is_set(self):
        t = tracker()
        t.on_arrival(1, 100.0, 0)
        self.assertEqual(t.count("arrival", 1800), 0)
        self.assertIsNone(t.wait_gap_mean(1800))

    def test_window_is_half_open_on_both_ends(self):
        t = tracker()
        t.on_arrival(1, 1000.0, 0)      # exactly at now - W  -> counted
        t.on_arrival(2, 2000.0, 0)      # exactly at now      -> NOT counted
        t.on_arrival(3, 1500.0, 0)
        t.set_clock(2000.0)
        self.assertEqual(t.count("arrival", 1000), 2)

    def test_a_request_is_not_counted_in_its_own_observation(self):
        # rq_time == sim_time at the arrival step, and the window excludes its right end
        t = tracker()
        t.set_clock(3600.0)
        t.on_arrival(1, 3600.0, 0)
        self.assertEqual(t.count("arrival", 1800), 0)
        t.set_clock(3660.0)
        self.assertEqual(t.count("arrival", 1800), 1)

    def test_window_must_be_within_the_pruning_horizon(self):
        t = tracker()
        t.set_clock(10_000.0)
        for bad in (0, -1, MAX_WINDOW_S + 1):
            with self.assertRaises(ValueError):
                t.count("arrival", bad)
        with self.assertRaises(ValueError):
            t.wait_gap_mean(MAX_WINDOW_S + 1)

    def test_unknown_kind_raises(self):
        t = tracker()
        t.set_clock(1.0)
        with self.assertRaises(ValueError):
            t.count("nonsense", 1800)


class TestPruning(unittest.TestCase):

    def test_pruning_never_drops_an_event_inside_the_longest_window(self):
        t = tracker()
        t.on_arrival(1, 1000.0, 0)
        t.set_clock(1000.0 + MAX_WINDOW_S)          # exactly at the left edge: kept
        self.assertEqual(t.count("arrival", MAX_WINDOW_S), 1)
        t.set_clock(1000.0 + MAX_WINDOW_S + 1)      # now strictly older: dropped
        self.assertEqual(t.count("arrival", MAX_WINDOW_S), 0)

    def test_pruning_bounds_memory_over_a_long_run(self):
        t = tracker()
        for i in range(5000):
            t.set_clock(float(i * 60))
            t.on_arrival(i, float(i * 60), i % 6)
        self.assertLessEqual(len(t._events["arrival"]), MAX_WINDOW_S // 60 + 2)
        self.assertEqual(t.episode_summary()["stats_arrivals"], 5000)

    def test_pruned_pickups_leave_the_distinct_total_intact(self):
        t = tracker(quoted_wait_fn=lambda rid: 100.0)
        t.set_clock(0.0)
        t.on_arrival(1, 0.0, 0)
        t.on_pickup(1, StubRq(rq_time=0.0, pu_time=200.0))
        t.set_clock(200.0 + MAX_WINDOW_S + 1)
        self.assertEqual(t.count("pickup", 1800), 0)
        self.assertEqual(t.episode_summary()["stats_pickups"], 1)


class TestZoneCounts(unittest.TestCase):

    def test_zone_set_includes_the_cell_and_its_ring(self):
        t = tracker()
        t.on_arrival(1, 100.0, 0)        # cell A
        t.on_arrival(2, 100.0, 7)        # cell B, a neighbour of A
        t.set_clock(200.0)
        self.assertEqual(t.zone_count("arrival", 1800, 0), 2)
        self.assertEqual(t.count("arrival", 1800, cells={"A"}), 1)
        self.assertEqual(t.count("arrival", 1800, cells={"B"}), 1)
        self.assertEqual(t.count("arrival", 1800), 2)

    def test_exits_and_pickups_use_the_riders_origin_cell(self):
        t = tracker(quoted_wait_fn=lambda rid: 50.0)
        t.set_clock(100.0)
        t.on_arrival(1, 100.0, 0)        # A
        t.on_arrival(2, 100.0, 7)        # B
        t.set_clock(200.0)
        t.on_exit(1, StubRq(rider_declined=True), "rider_declined")
        t.on_exit(2, StubRq(diffusion_cancelled=True), "diffusion_cancelled")
        t.on_pickup(1, StubRq(rq_time=100.0, pu_time=180.0))
        t.set_clock(300.0)
        self.assertEqual(t.count("decline", 1800, cells={"A"}), 1)
        self.assertEqual(t.count("decline", 1800, cells={"B"}), 0)
        self.assertEqual(t.count("cancel", 1800, cells={"B"}), 1)
        self.assertEqual(t.count("pickup", 1800, cells={"A"}), 1)

    def test_an_unplaceable_origin_counts_fleet_wide_but_in_no_zone(self):
        # zones raise on an unknown node (P2.5); the arrival is still real
        t = tracker()
        t.on_arrival(1, 100.0, 999)
        t.set_clock(200.0)
        self.assertEqual(t.count("arrival", 1800), 1)
        self.assertEqual(t.count("arrival", 1800, cells={"A", "B"}), 0)
        self.assertEqual(t.episode_summary()["stats_unknown_cell_rids"], 1)


class TestExitRecording(unittest.TestCase):

    def test_only_declines_and_cancellations_are_windowed(self):
        t = tracker()
        t.set_clock(500.0)
        for rid, outcome in enumerate(("served", "no_show", "no_candidates",
                                       "operator_rejected", "unserved_at_horizon",
                                       "unclassified")):
            t.on_arrival(rid, 100.0, 0)
            t.on_exit(rid, StubRq(), outcome)
        t.set_clock(600.0)
        self.assertEqual(t.count("decline", 1800), 0)
        self.assertEqual(t.count("cancel", 1800), 0)
        s = t.episode_summary()
        self.assertEqual((s["stats_declines"], s["stats_cancels"]), (0, 0))

    def test_a_none_outcome_records_nothing(self):
        # None is what RewardTracker.on_exit returns for an already-handled rid
        t = tracker()
        t.set_clock(500.0)
        t.on_arrival(1, 100.0, 0)
        t.on_exit(1, StubRq(rider_declined=True), None)
        t.set_clock(600.0)
        self.assertEqual(t.count("decline", 1800), 0)
        self.assertEqual(t.episode_summary()["stats_declines"], 0)

    def test_exits_are_timed_by_the_clock_not_by_rq_time(self):
        t = tracker()
        t.set_clock(5000.0)
        t.on_arrival(1, 100.0, 0)                    # arrived long ago
        t.on_exit(1, StubRq(rq_time=100.0, diffusion_cancelled=True), "diffusion_cancelled")
        t.set_clock(5100.0)
        self.assertEqual(t.count("cancel", 1800), 1, "cancellation is timed at 5000, not 100")
        self.assertEqual(t.count("arrival", 1800), 0, "the arrival is timed at 100")


class TestPickupsAndWaitGap(unittest.TestCase):

    def test_gap_is_realized_minus_quoted(self):
        t = tracker(quoted_wait_fn=lambda rid: 300.0)
        t.set_clock(0.0)
        t.on_arrival(1, 0.0, 0)
        t.on_pickup(1, StubRq(rq_time=0.0, pu_time=500.0))      # realized 500, quoted 300
        t.set_clock(600.0)
        self.assertAlmostEqual(t.wait_gap_mean(1800), 200.0, places=9)

    def test_gap_is_negative_when_pickup_beats_the_quote(self):
        # the common case here: the planner assumes 300 s per stop against a realized ~56 s
        t = tracker(quoted_wait_fn=lambda rid: 600.0)
        t.set_clock(0.0)
        t.on_arrival(1, 0.0, 0)
        t.on_pickup(1, StubRq(rq_time=0.0, pu_time=400.0))
        t.set_clock(500.0)
        self.assertAlmostEqual(t.wait_gap_mean(1800), -200.0, places=9)

    def test_repeat_boarding_replaces_rather_than_appends(self):
        # §6.4's last-write-wins: rid 50's first boarding is overwritten, so the window must
        # hold ONE sample, priced at the final pu_time -- appending would double-count it and
        # keep a stale wait
        t = tracker(quoted_wait_fn=lambda rid: 100.0)
        t.set_clock(0.0)
        t.on_arrival(50, 32940.0, 0)
        t.set_clock(33100.0)
        t.on_pickup(50, StubRq(rq_time=32940.0, pu_time=33032.898))
        t.on_pickup(50, StubRq(rq_time=32940.0, pu_time=33230.0))
        t.set_clock(33300.0)
        self.assertEqual(t.count("pickup", 1800), 1)
        self.assertAlmostEqual(t.wait_gap_mean(1800), (33230.0 - 32940.0) - 100.0, places=9)
        s = t.episode_summary()
        self.assertEqual(s["stats_pickups"], 1)
        self.assertEqual(s["stats_boarding_calls"], 2)

    def test_mean_over_several_riders(self):
        t = tracker(quoted_wait_fn=lambda rid: 100.0)
        t.set_clock(0.0)
        for rid in (1, 2, 3):
            t.on_arrival(rid, 0.0, 0)
        t.set_clock(1000.0)
        t.on_pickup(1, StubRq(rq_time=0.0, pu_time=200.0))      # gap +100
        t.on_pickup(2, StubRq(rq_time=0.0, pu_time=100.0))      # gap    0
        t.on_pickup(3, StubRq(rq_time=0.0, pu_time=50.0))       # gap  -50
        t.set_clock(1100.0)
        self.assertAlmostEqual(t.wait_gap_mean(1800), (100.0 + 0.0 - 50.0) / 3, places=9)

    def test_none_when_the_window_holds_no_pickup(self):
        t = tracker(quoted_wait_fn=lambda rid: 100.0)
        t.set_clock(10_000.0)
        self.assertIsNone(t.wait_gap_mean(1800))

    def test_an_unpriced_pickup_counts_but_does_not_enter_the_mean(self):
        # a Rejection carries offered_waiting_time = None
        t = tracker(quoted_wait_fn=lambda rid: None)
        t.set_clock(0.0)
        t.on_arrival(1, 0.0, 0)
        t.on_pickup(1, StubRq(rq_time=0.0, pu_time=200.0))
        t.set_clock(300.0)
        self.assertEqual(t.count("pickup", 1800), 1)
        self.assertIsNone(t.wait_gap_mean(1800))

    def test_a_missing_pu_time_is_skipped_not_crashed(self):
        t = tracker(quoted_wait_fn=lambda rid: 100.0)
        t.set_clock(0.0)
        t.on_pickup(1, StubRq(pu_time=None))
        self.assertEqual(t.episode_summary()["stats_pickups"], 0)
        self.assertEqual(t.episode_summary()["stats_boarding_calls"], 1)

    def test_a_raising_quote_lookup_cannot_kill_the_simulation(self):
        def boom(rid):
            raise RuntimeError("operator state is gone")
        t = tracker(quoted_wait_fn=boom)
        t.set_clock(0.0)
        t.on_arrival(1, 0.0, 0)
        t.on_pickup(1, StubRq(rq_time=0.0, pu_time=200.0))
        t.set_clock(300.0)
        self.assertEqual(t.count("pickup", 1800), 1)
        self.assertIsNone(t.wait_gap_mean(1800))


class TestOneClassifierForBothTrackers(unittest.TestCase):
    """Trap 13: one §6.4 ladder, one classification per exit."""

    def test_reward_tracker_delegates_to_the_shared_function(self):
        rt = RewardTracker()
        rt.note_no_candidates(7)
        for rid, rq in ((7, StubRq()), (8, StubRq(rider_declined=True)),
                        (9, StubRq(pu_time=1.0, do_time=2.0)),
                        (10, StubRq(diffusion_cancelled=True)),
                        (11, StubRq(no_show=True)), (12, StubRq())):
            self.assertEqual(rt.classify(rid, rq),
                             classify_exit(rid, rq, rt._no_candidate_rids, rt._rejected_rids))

    def test_on_exit_returns_the_outcome_it_charged(self):
        rt = RewardTracker()
        self.assertEqual(rt.on_exit(1, StubRq(rider_declined=True)), "rider_declined")
        self.assertEqual(rt.on_exit(2, StubRq(diffusion_cancelled=True)),
                         "diffusion_cancelled")
        self.assertEqual(rt.on_exit(3, StubRq(pu_time=1.0, do_time=2.0)), "served")

    def test_a_repeated_exit_returns_none(self):
        # Ritun's condition 2: the fan-out keys off this to stay idempotent
        rt = RewardTracker()
        rq = StubRq(rider_declined=True)
        self.assertEqual(rt.on_exit(1, rq), "rider_declined")
        self.assertIsNone(rt.on_exit(1, rq))
        self.assertIsNone(rt.on_exit(1, rq))
        self.assertEqual(rt.episode_summary()["rider_declined"], 1)

    def test_the_fan_out_pattern_records_each_exit_once(self):
        # the env's _on_exit, inlined: classify once, skip when None
        rt, st = RewardTracker(), tracker()
        st.set_clock(500.0)
        st.on_arrival(1, 100.0, 0)
        rq = StubRq(rider_declined=True)
        for _ in range(3):
            outcome = rt.on_exit(1, rq)
            if outcome is not None:
                st.on_exit(1, rq, outcome)
        st.set_clock(600.0)
        self.assertEqual(st.count("decline", 1800), 1)
        self.assertEqual(st.episode_summary()["stats_declines"], 1)
        self.assertEqual(rt.episode_summary()["rider_declined"], 1)


class TestArrivalHookIsANoOpWhenUnattached(unittest.TestCase):
    """Ritun's condition 1: P1.3, P1.4 and the P1.10 gate run the operator with no tracker."""

    def test_operator_sets_the_slot_in_init_so_the_class_has_none(self):
        # the operator assigns self._stats_tracker in __init__, so unlike the sim env there is
        # no class attribute at all. Asserted as an absence, not with a hasattr guard that
        # would make the check pass vacuously.
        from src.rl_gym.fleetctrl_rl import RLPoolingIRSOnly
        self.assertNotIn("_stats_tracker", vars(RLPoolingIRSOnly))
        self.assertIn("_stats_tracker", RLPoolingIRSOnly.__init__.__code__.co_names)

    def test_sim_env_defaults_to_no_stats_tracker(self):
        from src.rl_gym.sim_env_rl import RLImmediateDecisionsSimulation
        self.assertIn("_stats_tracker", vars(RLImmediateDecisionsSimulation))
        self.assertIsNone(RLImmediateDecisionsSimulation._stats_tracker)


class TestTheTrackerSlotIsPerInstanceNotPerClass(unittest.TestCase):
    """The class-level default must never be *written*, only shadowed.

    If `set_stats_tracker` ever assigned to the class instead of the instance, every later
    simulation built in the same process would inherit a dead tracker from a finished episode —
    including P1.4's driver, which constructs `RLImmediateDecisionsSimulation` directly with no
    gym and no tracker. Nothing would raise; the stale tracker would simply keep recording.
    """

    def _bare(self):
        from src.rl_gym.sim_env_rl import RLImmediateDecisionsSimulation as C
        # attribute mechanics only, so no FleetPy construction is needed
        return C, C.__new__(C)

    def test_attaching_creates_an_instance_attribute_and_leaves_the_class_alone(self):
        C, sim = self._bare()
        self.assertNotIn("_stats_tracker", vars(sim))
        self.assertIsNone(sim._stats_tracker)
        sentinel = object()
        sim.set_stats_tracker(sentinel)
        self.assertIn("_stats_tracker", vars(sim), "must shadow, not write the class")
        self.assertIs(sim._stats_tracker, sentinel)
        self.assertIsNone(C._stats_tracker, "the class default must still be None")

    def test_a_later_instance_in_the_same_process_sees_no_tracker(self):
        C, first = self._bare()
        first.set_stats_tracker(object())
        later = C.__new__(C)
        self.assertIsNone(later._stats_tracker)
        self.assertNotIn("_stats_tracker", vars(later))

    def test_detaching_also_leaves_the_class_alone(self):
        C, sim = self._bare()
        sim.set_stats_tracker(object())
        sim.set_stats_tracker(None)
        self.assertIsNone(sim._stats_tracker)
        self.assertIsNone(C._stats_tracker)

    def test_two_real_envs_in_one_process_do_not_share_a_tracker(self):
        """The end-to-end version: two full simulations, one process."""
        from src.rl_gym.gym_env import SDPDPAssignmentEnv
        from src.rl_gym.sim_env_rl import RLImmediateDecisionsSimulation as C
        cfg = {
            "constant_cfg_path": os.path.join(SCS, "constant_config_depot_cali_sc_1.csv"),
            "scenario_cfg_path": os.path.join(SCS, "example_depot_cali_sc_1.csv"),
            "K": 8, "skip_output": True,
        }
        a = SDPDPAssignmentEnv({**cfg, "env_id": 0})
        b = SDPDPAssignmentEnv({**cfg, "env_id": 1})
        try:
            a.reset()
            b.reset()
            self.assertIsNot(a._stats, b._stats, "each reset builds its own tracker (§10)")
            self.assertIs(a.sim._stats_tracker, a._stats)
            self.assertIs(b.sim._stats_tracker, b._stats)
            self.assertIsNone(C._stats_tracker,
                              "two live simulations must not have written the class slot")
            # and a third simulation built afterwards still starts clean
            self.assertIsNone(C.__new__(C)._stats_tracker)
        finally:
            a.close()
            b.close()

    def test_a_full_request_path_runs_with_the_hook_unattached(self):
        """Build a real sim, detach the tracker, and step: the hook must do nothing."""
        from src.rl_gym.gym_env import SDPDPAssignmentEnv
        from src.rl_gym.policies import greedy_action
        env = SDPDPAssignmentEnv({
            "constant_cfg_path": os.path.join(SCS, "constant_config_depot_cali_sc_1.csv"),
            "scenario_cfg_path": os.path.join(SCS, "example_depot_cali_sc_1.csv"),
            "K": 8, "skip_output": True, "env_id": 0,
        })
        try:
            obs, _info = env.reset()
            # detach from both the operator and the sim env, as a non-gym run would have it
            env.sim.operators[0].set_stats_tracker(None)
            env.sim.set_stats_tracker(None)
            before = env._stats.episode_summary()
            for _ in range(25):
                obs, _r, term, _t, _i = env.step(
                    greedy_action(obs, env.action_masks(), env.observation_layout))
                if term:
                    break
            self.assertEqual(env._stats.episode_summary()["stats_arrivals"],
                             before["stats_arrivals"],
                             "the arrival hook must record nothing once detached")
        finally:
            env.close()


class TestWiredToARealSimulation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.rl_gym.gym_env import SDPDPAssignmentEnv
        from src.rl_gym.policies import greedy_action
        cls.env = SDPDPAssignmentEnv({
            "constant_cfg_path": os.path.join(SCS, "constant_config_depot_cali_sc_1.csv"),
            "scenario_cfg_path": os.path.join(SCS, "example_depot_cali_sc_1.csv"),
            "K": 8, "skip_output": True, "env_id": 0,
        })
        obs, cls.first_info = cls.env.reset()
        cls.steps = 0
        terminated = False
        while not terminated and cls.steps < 150:
            obs, _r, terminated, _t, cls.info = cls.env.step(
                greedy_action(obs, cls.env.action_masks(), cls.env.observation_layout))
            cls.steps += 1

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_the_clock_tracks_the_simulation(self):
        self.assertIsNotNone(self.env._stats.now)
        self.assertEqual(self.env._stats.now, self.info["sim_time"])

    def test_arrivals_are_recorded_and_exceed_the_step_count(self):
        # every loaded request is an arrival, including the internally-resolved ones, so
        # arrivals >= gym steps
        s = self.env._stats.episode_summary()
        self.assertGreaterEqual(s["stats_arrivals"], self.steps)
        self.assertEqual(s["stats_unknown_cell_rids"], 0,
                         "every request origin must resolve to one of the 37 cells")

    def test_all_four_streams_see_events(self):
        s = self.env._stats.episode_summary()
        for key in ("stats_arrivals", "stats_declines", "stats_cancels", "stats_pickups"):
            self.assertGreater(s[key], 0, key)

    def test_windowed_counts_are_consistent_across_window_lengths(self):
        st = self.env._stats
        for kind in ("arrival", "decline", "cancel", "pickup"):
            self.assertLessEqual(st.count(kind, 1800), st.count(kind, 3600), kind)

    def test_zone_counts_never_exceed_fleet_wide(self):
        st = self.env._stats
        node = self.env.sim.operators[0].rq_dict[self.info["rid"]].o_pos[0]
        for kind in ("arrival", "decline", "cancel", "pickup"):
            for w in (1800, 3600):
                self.assertLessEqual(st.zone_count(kind, w, node), st.count(kind, w),
                                     f"{kind} {w}")

    def test_episode_summary_carries_the_stats_totals(self):
        s = self.env.episode_summary()
        for key in ("stats_arrivals", "stats_declines", "stats_cancels", "stats_pickups",
                    "stats_boarding_calls", "stats_unknown_cell_rids"):
            self.assertIn(key, s)

    def test_reward_and_stats_agree_on_the_exit_counts(self):
        # the point of the single classification: the two trackers cannot disagree
        s = self.env.episode_summary()
        self.assertEqual(s["stats_declines"], s["rider_declined"])
        self.assertEqual(s["stats_cancels"], s["diffusion_cancelled"])
        self.assertEqual(s["stats_pickups"], s["pickups"])
        self.assertEqual(s["stats_boarding_calls"], s["boarding_calls"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
