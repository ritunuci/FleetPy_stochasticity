"""Trailing-window arrival, decline, cancellation and pickup statistics (P2.7).

Feeds §5.2's request block (`zone_arrivals_*`, `zone_declines_*`, `zone_cancels_*`) and global
block (`arrivals_*`, `declines_*`, `cancels_*`, `wait_gap_*`). Every window is trailing and
half-open: an event counts when its time is in `[sim_time - W, sim_time)`, `W` in
{30, 60} minutes (§5.2). A request arriving in the current step is therefore not counted in its
own observation, since `rq_time == sim_time` at its arrival step.

Four event streams, each from the hook that already exists:

- **arrival** -- from `RLPoolingIRSOnly.build_assignment_context`, the one entry point every
  loaded request passes through, internally-resolved ones included (§1, verified in P2.7).
  Timed by `rq_time`.
- **decline** and **post-match cancellation** -- from the env's exit fan-out, which classifies
  each exit **once** with `reward.classify_exit` and hands the outcome here, so the two trackers
  cannot disagree about an exit (trap 13). Timed by the clock, i.e. the step in which they fire.
- **pickup** -- from the boarding callback. Timed by `rq.pu_time`, and carries the wait gap,
  realized minus quoted.

**Why a clock.** The exit callback receives only `(rid, rq)`, with no time, so
`step_generator` sets `set_clock(sim_time)` at the start of every step. It is stale during the
end-of-day finalisation, after the step loop -- and harmlessly so: only declines and
cancellations read it, and neither fires there (the diffusion check is inside `step`), while
pickups carry their own `pu_time`. The windowed queries are read only at decision epochs, which
are all inside the loop.

**Events are keyed to the event rider's origin cell** (§5.2). The cell is resolved once, at
arrival, and cached per rid -- an exit or a pickup always follows that rider's arrival, so no
later lookup is needed and `ZoneTable` is touched once per request rather than four times.

**Scale.** The busiest (month, minute) in P2.6's baseline averages 73.7 arrivals in a 60-minute
window and reaches about 115 at +3 SD; with declines, cancellations and pickups on top a
60-minute window holds roughly 200 events fleet-wide. Scanning a few hundred deque entries a
handful of times per decision is far cheaper than maintaining per-zone indices, so the queries
scan.

The tracker is rebuilt with the simulation at every reset (§10).
"""

import logging
from collections import deque
from typing import Any, Callable, Deque, Dict, Iterable, Optional, Tuple

LOG = logging.getLogger(__name__)

#: the longest window any query may ask for; pruning keeps everything at least this old
MAX_WINDOW_S = 3600

#: event kinds held as time-ordered deques. Pickups are keyed per rid instead -- see `on_pickup`.
DEQUE_KINDS = ("arrival", "decline", "cancel")

#: the §6.4 outcomes this tracker records, mapped to its own event kinds. Everything else --
#: served, no_show, no_candidates, operator_rejected, unserved_at_horizon, unclassified -- is
#: counted but not windowed: §5.2 asks for declines and post-match cancellations only.
_OUTCOME_TO_KIND = {"rider_declined": "decline", "diffusion_cancelled": "cancel"}


class RollingStatsTracker:
    """Trailing-window counts by zone set and fleet-wide, plus the mean wait gap."""

    def __init__(self, zone_table, quoted_wait_fn: Optional[Callable[[Any], Optional[float]]]
                 = None, max_window_s: int = MAX_WINDOW_S):
        """
        :param zone_table: P2.5's `ZoneTable`, for `cell_of_node`.
        :param quoted_wait_fn: `rid -> quoted wait in seconds, or None`. The quote lives on the
            **operator's** `PlanRequest` (`prq.offer.offered_waiting_time`), not on the demand
            object the boarding callback passes, so the env injects a closure over the operator.
            Nothing from the diffusion model is ever read (§5.2, proposal §3.1.2).
        :param max_window_s: the longest queryable window; events older than this are pruned.
        """
        self.zones = zone_table
        self.quoted_wait_fn = quoted_wait_fn
        self.max_window_s = int(max_window_s)

        self._now: Optional[float] = None
        self._events: Dict[str, Deque[Tuple[float, str]]] = {k: deque() for k in DEQUE_KINDS}
        #: rid -> (pu_time, cell, wait_gap_s). Per rid, **replaced** on a repeat boarding, so a
        #: rider contributes one sample at the final `pu_time` (§6.4's last-write-wins, §5.2's
        #: "one entry per distinct rid"). Appending instead would double-count rid 50 and keep
        #: a stale wait.
        self._pickups: Dict[Any, Tuple[float, str, Optional[float]]] = {}
        self._cell_of_rid: Dict[Any, str] = {}
        self._totals: Dict[str, int] = {"arrival": 0, "decline": 0, "cancel": 0, "pickup": 0}
        self._boarding_calls = 0
        self._unknown_cell_rids = 0

    # ---------------------------------------------------------------- #
    # clock
    # ---------------------------------------------------------------- #

    def set_clock(self, sim_time: float) -> None:
        """Set the current simulation time, and prune what no window can reach any more."""
        self._now = float(sim_time)
        self._prune()

    @property
    def now(self) -> Optional[float]:
        return self._now

    def _prune(self) -> None:
        if self._now is None:
            return
        # an event at exactly now - max_window_s is still inside the longest window, which is
        # half-open on the left: [now - W, now). Drop only what is strictly older.
        cutoff = self._now - self.max_window_s
        for dq in self._events.values():
            while dq and dq[0][0] < cutoff:
                dq.popleft()
        if self._pickups:
            stale = [rid for rid, (pu, _c, _g) in self._pickups.items() if pu < cutoff]
            for rid in stale:
                del self._pickups[rid]

    # ---------------------------------------------------------------- #
    # events
    # ---------------------------------------------------------------- #

    def on_arrival(self, rid, rq_time: float, origin_node) -> None:
        """Record an arrival, and resolve this rider's origin cell once (§5.2)."""
        cell = self._resolve_cell(rid, origin_node)
        self._totals["arrival"] += 1
        self._append("arrival", float(rq_time), cell)

    def on_exit(self, rid, rq, outcome: Optional[str]) -> None:
        """Record a decline or a post-match cancellation, timed by the clock.

        :param outcome: the §6.4 label, computed **once** by `RewardTracker.on_exit` and passed
            through by the env's fan-out (trap 13). `None` means the exit was a repeat for an
            already-handled rid, so nothing is recorded -- which is what keeps this tracker's
            counts idempotent per rid without a second `_exited_rids` set of its own.
        """
        kind = _OUTCOME_TO_KIND.get(outcome) if outcome else None
        if kind is None:
            return
        if self._now is None:
            LOG.warning(f"on_exit({rid}): no clock set, {kind} not recorded")
            return
        self._totals[kind] += 1
        self._append(kind, self._now, self._cell_of_rid.get(rid))

    def on_pickup(self, rid, rq) -> None:
        """Record a pickup and its wait gap, replacing any earlier entry for this rid."""
        self._boarding_calls += 1
        pu_time = getattr(rq, "pu_time", None)
        rq_time = getattr(rq, "rq_time", None)
        if pu_time is None or rq_time is None:
            LOG.warning(f"on_pickup({rid}): pu_time={pu_time} rq_time={rq_time}, skipping")
            return
        quoted = None
        if self.quoted_wait_fn is not None:
            try:
                quoted = self.quoted_wait_fn(rid)
            except Exception:                      # a stats bug must never kill a simulation
                LOG.warning(f"on_pickup({rid}): quoted-wait lookup failed", exc_info=True)
        gap = None if quoted is None else (pu_time - rq_time) - quoted
        if rid not in self._pickups:
            self._totals["pickup"] += 1            # distinct rids, not call count
        self._pickups[rid] = (float(pu_time), self._cell_of_rid.get(rid), gap)

    def _append(self, kind: str, when: float, cell: Optional[str]) -> None:
        dq = self._events[kind]
        # times arrive non-decreasing in practice (the clock only moves forward, and arrivals
        # are retrieved in time order), but a boarding replan can re-fire slightly out of
        # order, so keep the deque sorted rather than assuming
        if dq and when < dq[-1][0]:
            items = list(dq)
            items.append((when, cell))
            items.sort(key=lambda e: e[0])
            self._events[kind] = deque(items)
        else:
            dq.append((when, cell))

    def _resolve_cell(self, rid, origin_node) -> Optional[str]:
        try:
            cell = self.zones.cell_of_node(origin_node)
        except Exception:
            # zones raise on an unknown node rather than returning empty (P2.5). A request the
            # table cannot place still counts fleet-wide; it is reported, not silently dropped.
            self._unknown_cell_rids += 1
            LOG.warning(f"on_arrival({rid}): no zone for origin node {origin_node!r}")
            return None
        self._cell_of_rid[rid] = cell
        return cell

    # ---------------------------------------------------------------- #
    # queries -- all over [now - W, now)
    # ---------------------------------------------------------------- #

    def count(self, kind: str, window_s: int, cells: Optional[Iterable[str]] = None) -> int:
        """Events of `kind` in `[now - window_s, now)`; restricted to `cells` when given."""
        self._check_window(window_s)
        if self._now is None:
            return 0
        lo, hi = self._now - window_s, self._now
        if kind == "pickup":
            items = ((pu, cell) for pu, cell, _gap in self._pickups.values())
        else:
            items = self._events[self._kind(kind)]
        if cells is None:
            return sum(1 for when, _cell in items if lo <= when < hi)
        wanted = cells if isinstance(cells, (set, frozenset)) else frozenset(cells)
        return sum(1 for when, cell in items
                   if lo <= when < hi and cell is not None and cell in wanted)

    def zone_count(self, kind: str, window_s: int, origin_node) -> int:
        """`count` over the zone set of `origin_node`'s cell -- the cell plus its ring (§5.2)."""
        return self.count(kind, window_s, cells=self.zones.zone_set_of_node(origin_node))

    def wait_gap_mean(self, window_s: int) -> Optional[float]:
        """Mean realized-minus-quoted wait over riders picked up in the window.

        `None` when the window holds no priced pickup; §5.2's observer writes 0.0 for that.
        Mean rather than median deliberately: on the reference day 17% of pickups land within
        1 s of their quote, so a 30-minute median sits within 1 s of zero in 26% of windows,
        while the mean responds to the late tail.
        """
        self._check_window(window_s)
        if self._now is None:
            return None
        lo, hi = self._now - window_s, self._now
        gaps = [gap for pu, _cell, gap in self._pickups.values()
                if lo <= pu < hi and gap is not None]
        return (sum(gaps) / len(gaps)) if gaps else None

    def _check_window(self, window_s: int) -> None:
        if not 0 < window_s <= self.max_window_s:
            raise ValueError(
                f"window {window_s} s is outside (0, {self.max_window_s}]; events older than "
                f"{self.max_window_s} s are pruned, so a longer window would silently "
                f"undercount"
            )

    @staticmethod
    def _kind(kind: str) -> str:
        if kind not in DEQUE_KINDS:
            raise ValueError(f"unknown event kind {kind!r}; expected one of "
                             f"{DEQUE_KINDS + ('pickup',)}")
        return kind

    # ---------------------------------------------------------------- #
    # readout
    # ---------------------------------------------------------------- #

    def episode_summary(self) -> Dict[str, Any]:
        """End-of-day totals, for reconciliation against `1_user-stats.csv` (P2.7)."""
        return {
            "stats_arrivals": self._totals["arrival"],
            "stats_declines": self._totals["decline"],
            "stats_cancels": self._totals["cancel"],
            "stats_pickups": self._totals["pickup"],
            "stats_boarding_calls": self._boarding_calls,
            "stats_unknown_cell_rids": self._unknown_cell_rids,
        }
