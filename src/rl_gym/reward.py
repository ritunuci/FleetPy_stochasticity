"""Reward accumulation and terminal-outcome classification (§6).

Reward is anchored on **pickup**, not trip completion (D4): no positive reward at assignment,
a positive term when the rider actually boards. Events accumulate here and are flushed once
per gym step, so `step(action)` returns the sum of everything that fired after that action
was applied and before the next decision point.

Four inputs, two of which only label:

- `on_pickup(rid, rq)`   from `Demand.record_boarding`  -- charges the pickup term
- `on_exit(rid, rq)`     from `Demand.record_user`      -- charges every terminal outcome
- `note_rejection(rid)`          from `commit_assignment_choice`
- `note_no_candidates(rid)`      from `build_assignment_context`

`note_*` record a rid in a set and charge nothing. All charging happens in `on_exit`, once per
rid -- `record_alighting_start` and `record_remaining_users` can both fire for a rider who is
mid-alighting when the day ends, so `on_exit` must be idempotent.

`on_pickup` is **last-write-wins, not dedupe-on-first** (§6.4). `record_boarding` fires more
than once for the same rid when a no-show shares a boarding stop with another rider on the same
vehicle: `sim_veh_no_show_requests_cleanup` calls `simple_remove_bulk` and the replan re-fires
the boarding stop for the co-located rider. FleetPy overwrites `pu_time` each time and the
output file records the last value, so the reward must price the last value too. Charging on the
first call would price a stale, shorter wait and bias the policy against exactly those riders
delayed by someone else's no-show. So the amount charged per rid is remembered and a repeat
charges the delta, leaving a net of one pickup at the final `pu_time`.

With `track_events` on, every charge is also appended to a per-window event list as
`(kind, rid, amount, detail)`, readable with `flush_events()`. It exists to check that §6.4's
ladder puts the right label on each event, which the episode-level counts cannot show -- 118
declines could be the correct total with several individually misclassified. Off by default:
training does not need it and 7,000 episodes should not pay for it. `flush()` returns the same
number in both modes.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

LOG = logging.getLogger(__name__)

#: §6.1's Phase 2 weights, DECIDED for the short test training (P2.15). `w_noshow = 0.0` is
#: Ritun's decision on §4 gap 2: a no-show rider forfeits the pickup reward and nothing more.
#: Set in P2.0b, replacing the Phase 1 values (`w_reject = 0.5`, `w_noshow = 0.8`) that shipped
#: through P1.11 -- see §6.6. They are defaults only: P2.13 requires all eight explicitly for
#: training, and the sweep that revises them is Ritun's step after Phase 2 (§8), not a work item.
DEFAULT_REWARD_WEIGHTS = {
    "w_pickup": 1.0,
    "w_wait": 0.05,
    "w_reject": 1.0,
    "w_decline": 0.1,
    "w_cancel": 0.6,
    "w_noshow": 0.0,
    "w_horizon": 0.0,
    "w_no_candidates": 0.0,
}

#: terminal outcomes from §6.4, in classification order
OUTCOMES = (
    "no_candidates",
    "operator_rejected",
    "served",
    "diffusion_cancelled",
    "rider_declined",
    "no_show",
    "unserved_at_horizon",
    "unclassified",
)


def classify_exit(rid, rq, no_candidate_rids: Set[Any], rejected_rids: Set[Any]) -> str:
    """§6.4's ordered classification. First match wins; no fallback to rider decline.

    **This is the single ladder, shared by `RewardTracker` and `RollingStatsTracker`**
    (trap 13). Two copies would drift with nothing raising -- the reward would charge as a
    decline what the statistics count as a cancellation. In practice the two never even call
    this separately: `RewardTracker.on_exit` returns the outcome it computed here and the env's
    exit fan-out hands that one string to the statistics tracker (P2.7), so there is one
    classification per exit, not two agreeing ones.

    :param no_candidate_rids: labelled by `build_assignment_context` on the empty-candidate
        branch -- membership, not a request attribute, which is why the sets are arguments.
    :param rejected_rids: labelled by `commit_assignment_choice` when the policy rejects.
    """
    if rid in no_candidate_rids:
        return "no_candidates"
    if rid in rejected_rids:
        return "operator_rejected"
    if getattr(rq, "do_time", None) is not None:
        return "served"
    if getattr(rq, "diffusion_cancelled", False):
        return "diffusion_cancelled"
    if getattr(rq, "rider_declined", False):
        return "rider_declined"
    pu_time = getattr(rq, "pu_time", None)
    # no_show is a rider attribute read from the demand file, not an outcome, so it must be
    # gated on pu_time and checked after diffusion_cancelled -- otherwise a no-show-flagged
    # rider who cancelled while waiting would be reported as a no-show
    if pu_time is None and getattr(rq, "no_show", False):
        return "no_show"
    if pu_time is None and getattr(rq, "chosen_operator_id", None) is not None:
        return "unserved_at_horizon"
    return "unclassified"


class RewardTracker:
    """Accumulates reward events between gym steps and classifies terminal outcomes."""

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 track_events: bool = False, end_time: Optional[float] = None):
        """
        :param weights: reward weights; missing keys fall back to DEFAULT_REWARD_WEIGHTS.
            Never hardcode a weight at a use site -- they all come from the env config (§6.1).
        :param track_events: record a per-window `(kind, rid, amount, detail)` list, readable
            with `flush_events()`. Off by default; `flush()` is unaffected either way.
        :param end_time: the scenario's `end_time`, used only for `horizon_margin` in
            `episode_summary()`. Absent leaves that field `None`.
        """
        self.weights: Dict[str, float] = dict(DEFAULT_REWARD_WEIGHTS)
        if weights:
            unknown = set(weights) - set(DEFAULT_REWARD_WEIGHTS)
            if unknown:
                raise ValueError(f"unknown reward weight(s): {sorted(unknown)}")
            self.weights.update(weights)

        self.track_events = bool(track_events)
        self.end_time = None if end_time is None else float(end_time)

        self._pending: float = 0.0                      # reward since the last flush
        self._events: List[Tuple[str, Any, float, Optional[float]]] = []
        self._no_candidate_rids: Set[Any] = set()       # labelled by build_assignment_context
        self._rejected_rids: Set[Any] = set()           # labelled by commit_assignment_choice
        self._charged_pickup: Dict[Any, float] = {}     # rid -> amount already charged
        self._exited_rids: Set[Any] = set()             # on_exit idempotency
        self._boarding_calls: Dict[Any, int] = {}       # rid -> record_boarding call count
        self._counts: Dict[str, int] = {name: 0 for name in OUTCOMES}
        self._counts["pickups"] = 0
        self._episode_reward: float = 0.0
        self._last_dropoff: Optional[float] = None      # largest do_time among served riders

    # ---------------------------------------------------------------- #
    # labels -- no reward charged
    # ---------------------------------------------------------------- #

    def note_rejection(self, rid) -> None:
        """Record that the operator rejected `rid`. Charges nothing; `on_exit` does (§6.4)."""
        self._rejected_rids.add(rid)

    def note_no_candidates(self, rid) -> None:
        """Record that `rid` had no feasible candidate. Charges nothing."""
        self._no_candidate_rids.add(rid)

    # ---------------------------------------------------------------- #
    # charging
    # ---------------------------------------------------------------- #

    def on_pickup(self, rid, rq) -> None:
        """Charge the pickup term: `+w_pickup - w_wait * (pu_time - rq_time)/60`.

        Last-write-wins: on a repeat call for the same rid, charges only the difference, so the
        net is one pickup priced at the final `pu_time`.
        """
        self._boarding_calls[rid] = self._boarding_calls.get(rid, 0) + 1

        pu_time = getattr(rq, "pu_time", None)
        rq_time = getattr(rq, "rq_time", None)
        if pu_time is None or rq_time is None:
            LOG.warning(f"on_pickup({rid}): pu_time={pu_time} rq_time={rq_time}, skipping")
            return

        wait_minutes = (pu_time - rq_time) / 60.0
        amount = self.weights["w_pickup"] - self.weights["w_wait"] * wait_minutes

        already = self._charged_pickup.get(rid)
        if already is None:
            self._counts["pickups"] += 1          # distinct rids, not call count
            delta = amount
        else:
            delta = amount - already              # re-price at the final pu_time
        self._charged_pickup[rid] = amount
        # detail is the realized wait in seconds, so the trace can be read against
        # 1_user-stats.csv without recomputing the weights
        self._add(delta, kind="pickup", rid=rid, detail=pu_time - rq_time)

    def on_exit(self, rid, rq) -> Optional[str]:
        """Classify `rid`'s terminal outcome and charge it, once per rid (§6.4).

        :return: the §6.4 outcome, or **None** on a repeated call for the same rid. The env's
            exit fan-out keys off that: `None` means "already handled", so the statistics
            tracker records each exit at most once too (P2.7). `record_alighting_start` and
            `record_remaining_users` can both fire for a rider mid-alighting at end of day.
        """
        if rid in self._exited_rids:
            return None                            # idempotent: two paths can both fire
        self._exited_rids.add(rid)

        outcome = self.classify(rid, rq)
        self._counts[outcome] += 1

        if outcome == "served":
            do_time = getattr(rq, "do_time", None)
            if do_time is not None and (self._last_dropoff is None
                                        or do_time > self._last_dropoff):
                self._last_dropoff = float(do_time)

        penalty = {
            "no_candidates": self.weights["w_no_candidates"],
            "operator_rejected": self.weights["w_reject"],
            "diffusion_cancelled": self.weights["w_cancel"],
            "rider_declined": self.weights["w_decline"],
            "no_show": self.weights["w_noshow"],
            "unserved_at_horizon": self.weights["w_horizon"],
        }.get(outcome, 0.0)
        # every classified exit is recorded, including the ones whose weight is 0.0 -- `served`,
        # `no_show`, `no_candidates` and `unserved_at_horizon` under §6.1's weights are exactly
        # the labels the trace exists to check, and a zero amount changes no total. `_add`
        # leaves the accumulators untouched when the amount is zero.
        self._add(-penalty if penalty else 0.0, kind=outcome, rid=rid)
        return outcome

    def classify(self, rid, rq) -> str:
        """§6.4's ordered classification, via the shared `classify_exit` (trap 13)."""
        return classify_exit(rid, rq, self._no_candidate_rids, self._rejected_rids)

    # ---------------------------------------------------------------- #
    # readout
    # ---------------------------------------------------------------- #

    def flush(self) -> float:
        """Return the reward accumulated since the last flush, and reset the accumulator.

        Called after `.send()`, never before: the simulation is frozen while the agent decides,
        so the window boundaries are exact (§6.1).
        """
        reward = self._pending
        self._pending = 0.0
        return reward

    def flush_events(self) -> List[Tuple[str, Any, float, Optional[float]]]:
        """Return the events charged since the last `flush_events()`, and clear them.

        Each entry is `(kind, rid, amount, detail)`: `kind` is `"pickup"` or the §6.4 outcome
        label, `detail` the realized wait in seconds for pickups and `None` otherwise. Empty
        unless `track_events` is set. The amounts of one window sum to that window's `flush()`.
        """
        events = self._events
        self._events = []
        return events

    def episode_summary(self) -> Dict[str, Any]:
        """Per-episode event counts and diagnostics.

        `unclassified` non-zero means a `record_user` path was missed. `duplicate_boardings`
        is the number of rids whose `record_boarding` fired more than once -- expected 1 on the
        reference day (§6.4).

        `last_dropoff` is the largest `do_time` among served riders and `horizon_margin` is
        `end_time - last_dropoff`: negative when the post-`end_time` tail did work, `None` if
        nobody was served (or no `end_time` was given). On the reference day under greedy the
        margin is about 194.7 s, and it is the one Phase 1 measurement a trained policy can
        move -- if it closes, §6.4's branch 7 goes live and `w_horizon = 0.0` stops being a
        placeholder (P2.13).
        """
        summary = dict(self._counts)
        summary["duplicate_boardings"] = sum(
            1 for n in self._boarding_calls.values() if n > 1
        )
        summary["boarding_calls"] = sum(self._boarding_calls.values())
        summary["labelled_no_candidates"] = len(self._no_candidate_rids)
        summary["labelled_rejections"] = len(self._rejected_rids)
        summary["exits"] = len(self._exited_rids)
        summary["episode_reward"] = self._episode_reward
        summary["unflushed_reward"] = self._pending
        summary["last_dropoff"] = self._last_dropoff
        summary["horizon_margin"] = (
            None if (self._last_dropoff is None or self.end_time is None)
            else self.end_time - self._last_dropoff
        )
        return summary

    def _add(self, amount: float, kind: Optional[str] = None, rid: Any = None,
             detail: Optional[float] = None) -> None:
        """Charge `amount` and, with `track_events` on, record it as an event.

        A zero amount leaves both accumulators untouched, so emitting zero-weight exits as
        events cannot perturb `flush()` or `episode_reward`.
        """
        if amount:
            self._pending += amount
            self._episode_reward += amount
        if self.track_events:
            self._events.append((kind, rid, amount, detail))
