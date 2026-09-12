"""RL-drivable fleet control: PoolingIRSOnly with user_request split in two.

`PoolingInsertionHeuristicOnly.user_request` does everything for an incoming request
in one call, choosing the winner with `min(list_tuples, key=lambda x: x[2])`. That
`argmin` is the decision the policy replaces, so this subclass splits the method there:

    build_assignment_context(rq, sim_time)  -> everything up to the argmin
    commit_assignment_choice(pending, ...)  -> everything after it

Nothing else is overridden. Offers, confirmations, cancellations, no-show cleanup and
`_return_expected_pickup_time` are all inherited, so the diffusion and no-show work is
untouched and non-RL scenarios keep running (D2).

The split must not perturb the simulation: `user_request` below is a faithful greedy
fallback (build, then commit with choice 0), and since the candidate list arrives sorted
ascending by `delta_cfv`, slot 0 is exactly what the parent's `min` selects. The
byte-for-byte test against `docs/baseline_user_stats.csv` is what proves it.

This class knows nothing about K or the action space. It receives a semantic choice --
a candidate index, or None to reject -- never a raw action. K lives only in
SDPDPAssignmentEnv (P1.8), which owns the translation.
"""

import logging
import time
from typing import Any, List, Optional, Tuple

from src.fleetctrl.PoolingIRSOnly import PoolingInsertionHeuristicOnly
from src.fleetctrl.planning.PlanRequest import PlanRequest
from src.fleetctrl.pooling.immediate.insertion import insertion_with_heuristics
from src.misc.globals import *

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_RLPoolingIRSOnly = {
    "doc": "PoolingIRSOnly with the assignment decision exposed for reinforcement learning",
    "inherit": "PoolingInsertionHeuristicOnly",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}


class PendingDecision:
    """One decision epoch: a request awaiting a choice among ranked candidates.

    :param prq: the PlanRequest built for this request
    :param candidates: list of (vid, vehplan, delta_cfv) exactly as
        insertion_with_heuristics returned it -- already sorted ascending by delta_cfv
        by a stable sort, so element 0 is the greedy choice. Never re-sorted, never
        tie-broken: upstream order within ties comes from a backwards Dijkstra ordered
        by proximity to the request origin, and any re-sort would pick a different
        vehicle on tied costs and break the byte-for-byte gate (D3).
    :param sim_time: simulation time of the request
    :param rid_struct: request id
    :param cpu_elapsed: seconds of fleet-control computation spent in
        build_assignment_context. A DURATION, not a timestamp -- commit adds its own
        span to it. Timing the whole build-to-commit span with one timestamp would
        fold the agent's inference and scheduler latency into G_FCTRL_CT_RQU, which is
        meant to measure fleet-control computation and is read in evaluation runs.
    """

    __slots__ = ("prq", "candidates", "sim_time", "rid_struct", "cpu_elapsed")

    def __init__(self, prq: PlanRequest, candidates: List[Tuple[Any, Any, float]],
                 sim_time: int, rid_struct: Any, cpu_elapsed: float):
        self.prq = prq
        self.candidates = candidates
        self.sim_time = sim_time
        self.rid_struct = rid_struct
        self.cpu_elapsed = cpu_elapsed

    def __repr__(self):
        return (f"PendingDecision(rid={self.rid_struct}, sim_time={self.sim_time}, "
                f"n_candidates={len(self.candidates)})")


class RLPoolingIRSOnly(PoolingInsertionHeuristicOnly):
    """PoolingIRSOnly with the assignment decision split out for RL control."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # optional; stays None for non-RL scenarios so nothing changes for them
        self._reward_tracker = None

    def set_reward_tracker(self, tracker) -> None:
        """Attach the RewardTracker (P1.7). Both call sites are None-guarded."""
        self._reward_tracker = tracker

    # ---------------------------------------------------------------- #
    # user_request, split in two
    # ---------------------------------------------------------------- #

    def build_assignment_context(self, rq, sim_time) -> Optional[PendingDecision]:
        """Everything `PoolingInsertionHeuristicOnly.user_request` does up to the argmin.

        Returns a PendingDecision when the operator has a real choice to make, and None
        when the request was fully resolved internally and there is nothing to decide:
        origin equals destination, the reservation branch, or an empty candidate list.
        Those three reproduce the parent exactly, including which of them records CPU
        time and which does not.

        :param rq: request object
        :param sim_time: current simulation time
        :return: PendingDecision, or None if already resolved
        """
        t0 = time.perf_counter()
        LOG.debug(f"Incoming request {rq.__dict__} at time {sim_time}")
        self.sim_time = sim_time
        prq = PlanRequest(rq, self.routing_engine, min_wait_time=self.min_wait_time,
                          max_wait_time=self.max_wait_time,
                          max_detour_time_factor=self.max_dtf, max_constant_detour_time=self.max_cdt,
                          add_constant_detour_time=self.add_cdt, min_detour_time_window=self.min_dtw,
                          boarding_time=self.const_bt)

        rid_struct = rq.get_rid_struct()
        self.rq_dict[rid_struct] = prq
        self.rq_dict_store[rid_struct] = prq  # Ritun added: planrequest object is not deleted from this dict

        if prq.o_pos == prq.d_pos:
            LOG.debug(f"automatic decline for rid {rid_struct}!")
            self._create_rejection(prq, sim_time)
            # the parent returns here WITHOUT recording cpu time -- keep it that way, or
            # this branch writes a G_FCTRL_CT_RQU value stock FleetPy never writes
            return None

        o_pos, t_pu_earliest, t_pu_latest = prq.get_o_stop_info()
        if t_pu_earliest - sim_time > self.opt_horizon:
            self.reservation_module.add_reservation_request(prq, sim_time)
            offer = self.reservation_module.return_immediate_reservation_offer(prq.get_rid_struct(), sim_time)
            LOG.debug(f"reservation offer for rid {rid_struct} : {offer}")
            self._record_request_cpu_time(sim_time, time.perf_counter() - t0)
            return None

        # list arrives sorted ascending by delta_cfv from insertion.py; stored as returned
        list_tuples = insertion_with_heuristics(sim_time, prq, self, force_feasible_assignment=True)
        if len(list_tuples) == 0:
            LOG.debug(f"rejection for rid {rid_struct}")
            self._create_rejection(prq, sim_time)
            if self._reward_tracker is not None:
                # labels only -- records the rid in a set and charges nothing (6.4)
                self._reward_tracker.note_no_candidates(rid_struct)
            self._record_request_cpu_time(sim_time, time.perf_counter() - t0)
            return None

        return PendingDecision(prq=prq, candidates=list_tuples, sim_time=sim_time,
                               rid_struct=rid_struct, cpu_elapsed=time.perf_counter() - t0)

    def commit_assignment_choice(self, pending: PendingDecision, choice: Optional[int],
                                 sim_time) -> None:
        """Everything `user_request` does after the argmin, for a semantic choice.

        :param pending: the PendingDecision returned by build_assignment_context
        :param choice: index into pending.candidates, or None to reject
        :param sim_time: current simulation time
        """
        # catches an off-by-one in the env's action translation, which is otherwise silent
        assert choice is None or 0 <= choice < len(pending.candidates), \
            (f"choice {choice} out of range for {len(pending.candidates)} candidates "
             f"(rid {pending.rid_struct})")

        t0 = time.perf_counter()
        prq = pending.prq
        rid_struct = pending.rid_struct

        if choice is None:
            LOG.debug(f"rejection for rid {rid_struct}")
            self._create_rejection(prq, sim_time)
            if self._reward_tracker is not None:
                # labels only -- records the rid in a set and charges nothing (6.4)
                self._reward_tracker.note_rejection(rid_struct)
        else:
            (vid, vehplan, delta_cfv) = pending.candidates[choice]
            self.tmp_assignment[rid_struct] = vehplan
            offer = self._create_user_offer(prq, sim_time, vehplan)
            LOG.debug(f"new offer for rid {rid_struct} : {offer}")

        self._record_request_cpu_time(sim_time, pending.cpu_elapsed + (time.perf_counter() - t0))

    def user_request(self, rq, sim_time):
        """Greedy fallback, so the class works with no gym attached.

        Build, then commit slot 0. The candidate list is sorted ascending by delta_cfv,
        so slot 0 is the minimum and this is exactly the parent's
        `min(list_tuples, key=lambda x: x[2])`.

        Returns None, like the parent -- the offer is fetched via get_current_offer(rid).
        """
        pending = self.build_assignment_context(rq, sim_time)
        if pending is not None:
            self.commit_assignment_choice(pending, 0, sim_time)

    # ---------------------------------------------------------------- #
    # helpers
    # ---------------------------------------------------------------- #

    def _record_request_cpu_time(self, sim_time, dt: float) -> None:
        """Accumulate fleet-control computation time under G_FCTRL_CT_RQU.

        Mirrors the parent's closing block; `dt` is a duration measured by the caller so
        that time spent waiting for the agent never enters the total.
        """
        dt = round(dt, 5)
        old_dt = self._get_current_dynamic_fleetcontrol_value(sim_time, G_FCTRL_CT_RQU)
        if old_dt is None:
            new_dt = dt
        else:
            new_dt = old_dt + dt
        output_dict = {G_FCTRL_CT_RQU: new_dt}
        self._add_to_dynamic_fleetcontrol_output(sim_time, output_dict)
