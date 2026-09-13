"""Unit tests for src/rl_gym/reward.py (P1.7).

Stdlib unittest: `python -m unittest discover tests`. Stub request objects; the reconciliation
against docs/baseline_user_stats.csv across a real day lives in P1.7's verification script.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rl_gym.reward import DEFAULT_REWARD_WEIGHTS, RewardTracker  # noqa: E402

W = DEFAULT_REWARD_WEIGHTS


class StubRq:
    """Stands in for a RequestBase, carrying only the attributes §6.4 reads."""

    def __init__(self, rq_time=30000.0, pu_time=None, do_time=None,
                 diffusion_cancelled=False, rider_declined=False, no_show=False,
                 chosen_operator_id=None):
        self.rq_time = rq_time
        self.pu_time = pu_time
        self.do_time = do_time
        self.diffusion_cancelled = diffusion_cancelled
        self.rider_declined = rider_declined
        self.no_show = no_show
        self.chosen_operator_id = chosen_operator_id


def pickup_value(wait_seconds, weights=W):
    return weights["w_pickup"] - weights["w_wait"] * (wait_seconds / 60.0)


class TestWeights(unittest.TestCase):

    def test_defaults_match_spec_6_1(self):
        self.assertEqual(W["w_pickup"], 1.0)
        self.assertEqual(W["w_wait"], 0.05)
        self.assertEqual(W["w_reject"], 0.5)
        self.assertEqual(W["w_decline"], 0.1)
        self.assertEqual(W["w_cancel"], 0.6)
        self.assertEqual(W["w_noshow"], 0.8)
        self.assertEqual(W["w_horizon"], 0.0)
        self.assertEqual(W["w_no_candidates"], 0.0)

    def test_overrides_applied(self):
        t = RewardTracker({"w_pickup": 5.0})
        self.assertEqual(t.weights["w_pickup"], 5.0)
        self.assertEqual(t.weights["w_wait"], W["w_wait"])  # untouched key keeps its default

    def test_unknown_weight_rejected(self):
        # a typo in the env config must not silently do nothing
        with self.assertRaises(ValueError):
            RewardTracker({"w_pickups": 1.0})

    def test_defaults_not_mutated_by_an_instance(self):
        t = RewardTracker({"w_pickup": 99.0})
        self.assertEqual(DEFAULT_REWARD_WEIGHTS["w_pickup"], 1.0)
        self.assertEqual(t.weights["w_pickup"], 99.0)


class TestPickup(unittest.TestCase):

    def test_charges_pickup_minus_wait(self):
        t = RewardTracker()
        t.on_pickup(1, StubRq(rq_time=30000.0, pu_time=30600.0))   # 600 s = 10 min
        self.assertAlmostEqual(t.flush(), 1.0 - 0.05 * 10.0, places=9)

    def test_longer_wait_earns_less(self):
        a, b = RewardTracker(), RewardTracker()
        a.on_pickup(1, StubRq(pu_time=30000.0 + 120.0))
        b.on_pickup(1, StubRq(pu_time=30000.0 + 1200.0))
        self.assertGreater(a.flush(), b.flush())

    def test_counts_distinct_rids_not_calls(self):
        t = RewardTracker()
        t.on_pickup(1, StubRq(pu_time=30100.0))
        t.on_pickup(1, StubRq(pu_time=30200.0))
        t.on_pickup(2, StubRq(pu_time=30300.0))
        s = t.episode_summary()
        self.assertEqual(s["pickups"], 2)
        self.assertEqual(s["boarding_calls"], 3)
        self.assertEqual(s["duplicate_boardings"], 1)

    def test_last_write_wins_net_is_one_pickup_at_the_final_time(self):
        # the rid-50 case: two record_boarding calls, the later pu_time is what lands in the
        # output file, so the net reward must price that one and only that one
        t = RewardTracker()
        rq = StubRq(rq_time=32940.0)
        rq.pu_time = 33032.898                       # first call, stale
        t.on_pickup(50, rq)
        rq.pu_time = 33230.0                         # replan re-fires; FleetPy overwrites
        t.on_pickup(50, rq)
        net = t.flush()
        self.assertAlmostEqual(net, pickup_value(33230.0 - 32940.0), places=9)
        self.assertEqual(t.episode_summary()["pickups"], 1)

    def test_dedupe_on_first_would_be_wrong(self):
        # documents why: the stale wait is 92.9 s against a realized 290 s, so charging the
        # first call would over-reward by a materially different amount
        stale = pickup_value(33032.898 - 32940.0)
        final = pickup_value(33230.0 - 32940.0)
        self.assertNotAlmostEqual(stale, final, places=3)
        self.assertGreater(stale, final)

    def test_repeat_delta_spans_a_flush_boundary(self):
        # the two boarding calls can straddle a gym step; the episode total must still be right
        t = RewardTracker()
        rq = StubRq(rq_time=0.0, pu_time=60.0)
        t.on_pickup(7, rq)
        first = t.flush()
        rq.pu_time = 600.0
        t.on_pickup(7, rq)
        second = t.flush()
        self.assertAlmostEqual(first + second, pickup_value(600.0), places=9)
        self.assertAlmostEqual(t.episode_summary()["episode_reward"],
                               pickup_value(600.0), places=9)

    def test_missing_pu_time_is_skipped_not_crashed(self):
        t = RewardTracker()
        t.on_pickup(1, StubRq(pu_time=None))
        self.assertEqual(t.flush(), 0.0)
        self.assertEqual(t.episode_summary()["pickups"], 0)
        self.assertEqual(t.episode_summary()["boarding_calls"], 1)


class TestClassification(unittest.TestCase):

    def setUp(self):
        self.t = RewardTracker()

    def test_no_candidates_wins_first(self):
        self.t.note_no_candidates(1)
        # even with every other flag set, branch 1 claims it
        rq = StubRq(pu_time=1.0, do_time=2.0, diffusion_cancelled=True,
                    rider_declined=True, no_show=True, chosen_operator_id=0)
        self.assertEqual(self.t.classify(1, rq), "no_candidates")

    def test_operator_rejection_wins_second(self):
        self.t.note_rejection(2)
        rq = StubRq(do_time=2.0, diffusion_cancelled=True, rider_declined=True)
        self.assertEqual(self.t.classify(2, rq), "operator_rejected")

    def test_served_wins_over_later_flags(self):
        rq = StubRq(pu_time=1.0, do_time=2.0, diffusion_cancelled=True, no_show=True)
        self.assertEqual(self.t.classify(3, rq), "served")

    def test_diffusion_before_decline(self):
        rq = StubRq(diffusion_cancelled=True, rider_declined=True)
        self.assertEqual(self.t.classify(4, rq), "diffusion_cancelled")

    def test_decline(self):
        self.assertEqual(self.t.classify(5, StubRq(rider_declined=True)), "rider_declined")

    def test_no_show_requires_pu_time_none(self):
        # a no-show-flagged rider who was actually picked up and dropped off is served
        self.assertEqual(self.t.classify(6, StubRq(no_show=True)), "no_show")
        self.assertEqual(
            self.t.classify(7, StubRq(no_show=True, pu_time=1.0, do_time=2.0)), "served")

    def test_no_show_flag_after_diffusion_cancel(self):
        # §6.4: a no-show-flagged rider who cancelled while waiting is a cancellation, not a
        # no-show -- the flag is a rider attribute read from the demand file, not an outcome
        rq = StubRq(no_show=True, diffusion_cancelled=True)
        self.assertEqual(self.t.classify(8, rq), "diffusion_cancelled")

    def test_unserved_at_horizon_requires_acceptance(self):
        accepted = StubRq(chosen_operator_id=0)
        self.assertEqual(self.t.classify(9, accepted), "unserved_at_horizon")

    def test_never_accepted_and_never_picked_up_is_unclassified(self):
        self.assertEqual(self.t.classify(10, StubRq()), "unclassified")

    def test_no_fallback_to_rider_decline(self):
        # an unmatched rider must be unclassified, never silently a decline
        self.assertNotEqual(self.t.classify(11, StubRq()), "rider_declined")

    def test_chosen_operator_zero_counts_as_accepted(self):
        # operator 0 is falsy; the check must be `is not None`, not truthiness
        self.assertEqual(self.t.classify(12, StubRq(chosen_operator_id=0)),
                         "unserved_at_horizon")


class TestExitCharging(unittest.TestCase):

    def test_each_outcome_charges_its_weight(self):
        cases = [
            ("diffusion_cancelled", StubRq(diffusion_cancelled=True), -W["w_cancel"]),
            ("rider_declined", StubRq(rider_declined=True), -W["w_decline"]),
            ("no_show", StubRq(no_show=True), -W["w_noshow"]),
            ("unserved_at_horizon", StubRq(chosen_operator_id=0), -W["w_horizon"]),
            ("served", StubRq(pu_time=1.0, do_time=2.0), 0.0),
            ("unclassified", StubRq(), 0.0),
        ]
        for name, rq, want in cases:
            with self.subTest(outcome=name):
                t = RewardTracker()
                t.on_exit(1, rq)
                self.assertAlmostEqual(t.flush(), want, places=9)
                self.assertEqual(t.episode_summary()[name], 1)

    def test_rejection_and_no_candidates_charge_at_exit_not_at_label(self):
        t = RewardTracker()
        t.note_rejection(1)
        t.note_no_candidates(2)
        self.assertEqual(t.flush(), 0.0, "labels must charge nothing")
        t.on_exit(1, StubRq())
        t.on_exit(2, StubRq())
        self.assertAlmostEqual(t.flush(), -W["w_reject"] - W["w_no_candidates"], places=9)

    def test_served_charges_no_terminal_penalty(self):
        t = RewardTracker()
        t.on_pickup(1, StubRq(rq_time=0.0, pu_time=600.0))
        before = t.flush()
        t.on_exit(1, StubRq(rq_time=0.0, pu_time=600.0, do_time=1200.0))
        self.assertEqual(t.flush(), 0.0)
        self.assertAlmostEqual(before, pickup_value(600.0), places=9)

    def test_on_exit_is_idempotent_per_rid(self):
        # record_alighting_start and record_remaining_users can both fire for a rider who is
        # mid-alighting when the day ends
        t = RewardTracker()
        rq = StubRq(rider_declined=True)
        t.on_exit(1, rq)
        t.on_exit(1, rq)
        t.on_exit(1, rq)
        self.assertAlmostEqual(t.flush(), -W["w_decline"], places=9)
        self.assertEqual(t.episode_summary()["rider_declined"], 1)
        self.assertEqual(t.episode_summary()["exits"], 1)

    def test_horizon_weight_zero_means_no_charge_but_still_counted(self):
        t = RewardTracker()
        t.on_exit(1, StubRq(chosen_operator_id=0))
        self.assertEqual(t.flush(), 0.0)
        self.assertEqual(t.episode_summary()["unserved_at_horizon"], 1)

    def test_horizon_weight_nonzero_charges(self):
        t = RewardTracker({"w_horizon": 0.4})
        t.on_exit(1, StubRq(chosen_operator_id=0))
        self.assertAlmostEqual(t.flush(), -0.4, places=9)


class TestFlush(unittest.TestCase):

    def test_flush_resets_and_sums_the_window(self):
        t = RewardTracker()
        t.on_pickup(1, StubRq(rq_time=0.0, pu_time=600.0))
        t.on_exit(2, StubRq(rider_declined=True))
        want = pickup_value(600.0) - W["w_decline"]
        self.assertAlmostEqual(t.flush(), want, places=9)
        self.assertEqual(t.flush(), 0.0, "a second flush must return nothing new")

    def test_empty_window_returns_zero(self):
        self.assertEqual(RewardTracker().flush(), 0.0)

    def test_episode_reward_survives_flushes(self):
        t = RewardTracker()
        t.on_exit(1, StubRq(rider_declined=True))
        t.flush()
        t.on_exit(2, StubRq(diffusion_cancelled=True))
        t.flush()
        self.assertAlmostEqual(t.episode_summary()["episode_reward"],
                               -W["w_decline"] - W["w_cancel"], places=9)
        self.assertEqual(t.episode_summary()["unflushed_reward"], 0.0)


class TestEpisodeSummary(unittest.TestCase):

    def test_all_outcome_keys_present_from_the_start(self):
        s = RewardTracker().episode_summary()
        for key in ("no_candidates", "operator_rejected", "served", "diffusion_cancelled",
                    "rider_declined", "no_show", "unserved_at_horizon", "unclassified",
                    "pickups", "duplicate_boardings"):
            self.assertIn(key, s)
            self.assertEqual(s[key], 0)

    def test_labels_reported_separately_from_outcomes(self):
        t = RewardTracker()
        t.note_no_candidates(1)
        t.note_rejection(2)
        s = t.episode_summary()
        self.assertEqual(s["labelled_no_candidates"], 1)
        self.assertEqual(s["labelled_rejections"], 1)
        self.assertEqual(s["no_candidates"], 0, "not an outcome until on_exit fires")
        self.assertEqual(s["unclassified"], 0)

    def test_counts_reconcile_on_a_small_mixed_episode(self):
        t = RewardTracker()
        t.note_no_candidates(100)
        t.on_pickup(1, StubRq(rq_time=0.0, pu_time=300.0))
        t.on_exit(1, StubRq(pu_time=300.0, do_time=900.0))
        t.on_exit(2, StubRq(rider_declined=True))
        t.on_exit(3, StubRq(diffusion_cancelled=True))
        t.on_exit(4, StubRq(no_show=True))
        t.on_exit(100, StubRq())
        s = t.episode_summary()
        self.assertEqual(s["served"], 1)
        self.assertEqual(s["rider_declined"], 1)
        self.assertEqual(s["diffusion_cancelled"], 1)
        self.assertEqual(s["no_show"], 1)
        self.assertEqual(s["no_candidates"], 1)
        self.assertEqual(s["unclassified"], 0)
        self.assertEqual(s["pickups"], 1)
        terminal = sum(s[k] for k in ("served", "rider_declined", "diffusion_cancelled",
                                      "no_show", "no_candidates", "operator_rejected",
                                      "unserved_at_horizon", "unclassified"))
        self.assertEqual(terminal, s["exits"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
