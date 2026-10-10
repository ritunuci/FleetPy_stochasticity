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

#: Explicit, all-distinct, all-non-zero weights for the charging tests.
#:
#: They must not come from `DEFAULT_REWARD_WEIGHTS`. Under §6.1's Phase 2 weights three of the
#: eight are 0.0 (`w_noshow`, `w_horizon`, `w_no_candidates`), so a charging test written
#: against the defaults would assert `0.0 == 0.0` for those outcomes and keep passing if the
#: branch stopped charging altogether -- the decoration problem §0 and P1.10 both rule out.
#: Distinct values also mean a swapped pair of branches fails rather than cancelling out.
#: The defaults themselves are checked by `TestWeights.test_defaults_match_spec_6_1`.
CHARGE_W = {
    "w_pickup": 2.0,
    "w_wait": 0.25,
    "w_reject": 0.3,
    "w_decline": 0.4,
    "w_cancel": 0.5,
    "w_noshow": 0.6,
    "w_horizon": 0.7,
    "w_no_candidates": 0.8,
}


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
        # §6.1's Phase 2 weights, set in P2.0b. Through P1.11 the defaults were the Phase 1
        # values, which differed in two keys: w_reject 0.5 and w_noshow 0.8 (§6.6).
        self.assertEqual(W["w_pickup"], 1.0)
        self.assertEqual(W["w_wait"], 0.05)
        self.assertEqual(W["w_reject"], 1.0)
        self.assertEqual(W["w_decline"], 0.1)
        self.assertEqual(W["w_cancel"], 0.6)
        self.assertEqual(W["w_noshow"], 0.0)
        self.assertEqual(W["w_horizon"], 0.0)
        self.assertEqual(W["w_no_candidates"], 0.0)

    def test_all_eight_weights_present_and_no_others(self):
        # P2.13 requires all eight explicitly for training; a key added here without a use
        # site, or dropped from here while still read, should fail loudly
        self.assertEqual(set(W), set(CHARGE_W))
        self.assertEqual(len(W), 8)

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
    """Charging, on explicit non-zero weights -- see CHARGE_W for why not the defaults."""

    def test_each_outcome_charges_its_weight(self):
        cases = [
            ("diffusion_cancelled", StubRq(diffusion_cancelled=True), -CHARGE_W["w_cancel"]),
            ("rider_declined", StubRq(rider_declined=True), -CHARGE_W["w_decline"]),
            ("no_show", StubRq(no_show=True), -CHARGE_W["w_noshow"]),
            ("unserved_at_horizon", StubRq(chosen_operator_id=0), -CHARGE_W["w_horizon"]),
            ("served", StubRq(pu_time=1.0, do_time=2.0), 0.0),
            ("unclassified", StubRq(), 0.0),
        ]
        for name, rq, want in cases:
            with self.subTest(outcome=name):
                t = RewardTracker(CHARGE_W)
                t.on_exit(1, rq)
                self.assertAlmostEqual(t.flush(), want, places=9)
                self.assertEqual(t.episode_summary()[name], 1)

    def test_each_outcome_charges_a_distinct_amount(self):
        # a swapped pair of branches must fail rather than cancel out
        amounts = {}
        for name, rq in (("diffusion_cancelled", StubRq(diffusion_cancelled=True)),
                         ("rider_declined", StubRq(rider_declined=True)),
                         ("no_show", StubRq(no_show=True)),
                         ("unserved_at_horizon", StubRq(chosen_operator_id=0))):
            t = RewardTracker(CHARGE_W)
            t.on_exit(1, rq)
            amounts[name] = t.flush()
        self.assertEqual(len(set(amounts.values())), 4, amounts)
        for name, amount in amounts.items():
            self.assertLess(amount, 0.0, f"{name} must charge a penalty, not nothing")

    def test_rejection_and_no_candidates_charge_at_exit_not_at_label(self):
        t = RewardTracker(CHARGE_W)
        t.note_rejection(1)
        t.note_no_candidates(2)
        self.assertEqual(t.flush(), 0.0, "labels must charge nothing")
        t.on_exit(1, StubRq())
        t.on_exit(2, StubRq())
        self.assertAlmostEqual(
            t.flush(), -CHARGE_W["w_reject"] - CHARGE_W["w_no_candidates"], places=9)

    def test_rejection_and_no_candidates_are_charged_separately(self):
        # both are 0.0-weighted or equal under some weight sets; keep them distinguishable
        a, b = RewardTracker(CHARGE_W), RewardTracker(CHARGE_W)
        a.note_rejection(1)
        a.on_exit(1, StubRq())
        b.note_no_candidates(1)
        b.on_exit(1, StubRq())
        self.assertAlmostEqual(a.flush(), -CHARGE_W["w_reject"], places=9)
        self.assertAlmostEqual(b.flush(), -CHARGE_W["w_no_candidates"], places=9)
        self.assertNotAlmostEqual(-CHARGE_W["w_reject"], -CHARGE_W["w_no_candidates"])

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
        t = RewardTracker(CHARGE_W)
        rq = StubRq(rider_declined=True)
        t.on_exit(1, rq)
        t.on_exit(1, rq)
        t.on_exit(1, rq)
        self.assertAlmostEqual(t.flush(), -CHARGE_W["w_decline"], places=9)
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


class TestEventTracking(unittest.TestCase):
    """The per-event breakdown (P2.0b). `flush()` must be identical in both modes."""

    def test_off_by_default(self):
        t = RewardTracker()
        t.on_pickup(1, StubRq(rq_time=0.0, pu_time=600.0))
        t.on_exit(2, StubRq(rider_declined=True))
        self.assertFalse(t.track_events)
        self.assertEqual(t.flush_events(), [])

    def test_flush_is_unchanged_by_tracking(self):
        # the point of emitting zero-amount events: the numbers must not move
        def run(track):
            t = RewardTracker(CHARGE_W, track_events=track)
            t.note_no_candidates(9)
            t.on_pickup(1, StubRq(rq_time=0.0, pu_time=300.0))
            t.on_exit(1, StubRq(pu_time=300.0, do_time=900.0))
            t.on_exit(2, StubRq(rider_declined=True))
            t.on_exit(9, StubRq())
            first = t.flush()
            t.on_exit(3, StubRq(no_show=True))
            return first, t.flush(), t.episode_summary()["episode_reward"]

        self.assertEqual(run(False), run(True))

    def test_zero_weight_exits_still_appear_as_events(self):
        # §6.1's weights zero out no_show, horizon and no_candidates, and `served` carries no
        # weight at all -- those are exactly the labels the trace exists to check
        t = RewardTracker(track_events=True)          # defaults: §6.1's Phase 2 weights
        t.note_no_candidates(4)
        t.on_exit(1, StubRq(pu_time=1.0, do_time=2.0))
        t.on_exit(2, StubRq(no_show=True))
        t.on_exit(3, StubRq(chosen_operator_id=0))
        t.on_exit(4, StubRq())
        kinds = [e[0] for e in t.flush_events()]
        self.assertEqual(kinds, ["served", "no_show", "unserved_at_horizon", "no_candidates"])
        self.assertEqual(t.flush(), 0.0, "all four carry a zero weight")

    def test_event_fields(self):
        t = RewardTracker(CHARGE_W, track_events=True)
        t.on_pickup(50, StubRq(rq_time=32940.0, pu_time=33230.0))
        t.on_exit(7, StubRq(diffusion_cancelled=True))
        events = t.flush_events()
        self.assertEqual(len(events), 2)
        kind, rid, amount, detail = events[0]
        self.assertEqual((kind, rid), ("pickup", 50))
        self.assertAlmostEqual(detail, 290.0, places=9)          # realized wait, seconds
        self.assertAlmostEqual(amount, pickup_value(290.0, CHARGE_W), places=9)
        kind, rid, amount, detail = events[1]
        self.assertEqual((kind, rid, detail), ("diffusion_cancelled", 7, None))
        self.assertAlmostEqual(amount, -CHARGE_W["w_cancel"], places=9)

    def test_amounts_sum_to_the_window_flush(self):
        t = RewardTracker(CHARGE_W, track_events=True)
        t.on_pickup(1, StubRq(rq_time=0.0, pu_time=120.0))
        t.on_exit(2, StubRq(rider_declined=True))
        t.on_exit(3, StubRq(no_show=True))
        events = t.flush_events()
        self.assertAlmostEqual(sum(e[2] for e in events), t.flush(), places=9)

    def test_events_clear_per_window(self):
        t = RewardTracker(CHARGE_W, track_events=True)
        t.on_exit(1, StubRq(rider_declined=True))
        self.assertEqual(len(t.flush_events()), 1)
        self.assertEqual(t.flush_events(), [], "a second read must return nothing new")
        t.on_exit(2, StubRq(rider_declined=True))
        self.assertEqual(len(t.flush_events()), 1)

    def test_repeat_pickup_emits_the_delta(self):
        # last-write-wins: two events whose amounts sum to one pickup at the final pu_time
        t = RewardTracker(CHARGE_W, track_events=True)
        rq = StubRq(rq_time=32940.0, pu_time=33032.898)
        t.on_pickup(50, rq)
        rq.pu_time = 33230.0
        t.on_pickup(50, rq)
        events = t.flush_events()
        self.assertEqual([e[0] for e in events], ["pickup", "pickup"])
        self.assertAlmostEqual(sum(e[2] for e in events),
                               pickup_value(290.0, CHARGE_W), places=9)
        self.assertAlmostEqual(events[1][3], 290.0, places=9)

    def test_skipped_pickup_emits_nothing(self):
        t = RewardTracker(CHARGE_W, track_events=True)
        t.on_pickup(1, StubRq(pu_time=None))
        self.assertEqual(t.flush_events(), [])

    def test_idempotent_exit_emits_one_event(self):
        t = RewardTracker(CHARGE_W, track_events=True)
        rq = StubRq(rider_declined=True)
        t.on_exit(1, rq)
        t.on_exit(1, rq)
        self.assertEqual(len(t.flush_events()), 1)

    def test_labels_emit_no_event(self):
        t = RewardTracker(CHARGE_W, track_events=True)
        t.note_rejection(1)
        t.note_no_candidates(2)
        self.assertEqual(t.flush_events(), [], "labels charge nothing, so they log nothing")


class TestHorizonMargin(unittest.TestCase):
    """`last_dropoff` / `horizon_margin` (P2.0b item 3)."""

    def test_none_when_nobody_served(self):
        t = RewardTracker(end_time=70000.0)
        t.on_exit(1, StubRq(rider_declined=True))
        s = t.episode_summary()
        self.assertIsNone(s["last_dropoff"])
        self.assertIsNone(s["horizon_margin"])

    def test_largest_dropoff_wins_regardless_of_arrival_order(self):
        t = RewardTracker(end_time=70000.0)
        t.on_exit(1, StubRq(pu_time=1.0, do_time=69805.3))
        t.on_exit(2, StubRq(pu_time=1.0, do_time=60000.0))        # earlier, must not overwrite
        s = t.episode_summary()
        self.assertAlmostEqual(s["last_dropoff"], 69805.3, places=6)
        self.assertAlmostEqual(s["horizon_margin"], 194.7, places=6)

    def test_margin_is_negative_when_the_tail_did_work(self):
        t = RewardTracker(end_time=70000.0)
        t.on_exit(1, StubRq(pu_time=1.0, do_time=70500.0))
        self.assertAlmostEqual(t.episode_summary()["horizon_margin"], -500.0, places=6)

    def test_only_served_riders_count(self):
        # a rider claimed by an earlier branch cannot set the last dropoff
        t = RewardTracker(end_time=70000.0)
        t.note_rejection(1)
        t.on_exit(1, StubRq(pu_time=1.0, do_time=69000.0))
        self.assertIsNone(t.episode_summary()["last_dropoff"])
        self.assertEqual(t.episode_summary()["operator_rejected"], 1)

    def test_margin_none_without_end_time(self):
        t = RewardTracker()
        t.on_exit(1, StubRq(pu_time=1.0, do_time=69805.3))
        s = t.episode_summary()
        self.assertAlmostEqual(s["last_dropoff"], 69805.3, places=6)
        self.assertIsNone(s["horizon_margin"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
