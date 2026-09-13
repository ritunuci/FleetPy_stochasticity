"""Generator-driven simulation environment: the gym owns the time loop.

`ImmediateDecisionsSimulation` runs its own loop and calls `user_request` internally,
which gives an RL agent nowhere to stand. This subclass inverts that control with a
generator: `run_generator()` yields a `PendingDecision` at every decision epoch and
receives the chosen candidate index back through `.send()` (D1).

No threads, no queues, no abort sentinel, no poll timeouts. The environment is
steppable in a normal debugger and profileable with a normal profiler. The cost is
mirroring ~35 lines of `step()` and ~20 of the `run()` loop, which is acceptable in a
fork that already diverges heavily from upstream.

The mirrored methods must stay in step with their parents. `step_generator` is
`ImmediateDecisionsSimulation.step` with only the request loop changed; `run_generator`
is `FleetSimulationBase.run` with the time loop delegating to `step_generator`. If
either parent changes, these change with it -- the byte-for-byte test against
`docs/baseline_user_stats.csv` is what catches a drift.
"""

import datetime
import logging
import time

from src.ImmediateDecisionsSimulation import ImmediateDecisionsSimulation
from src.misc.globals import *

LOG = logging.getLogger(__name__)

INPUT_PARAMETERS_RLImmediateDecisionsSimulation = {
    "doc": "ImmediateDecisionsSimulation driven by a generator, so an RL agent makes the "
           "assignment decision at each request",
    "inherit": "ImmediateDecisionsSimulation",
    "input_parameters_mandatory": [],
    "input_parameters_optional": [],
    "mandatory_modules": [],
    "optional_modules": []
}


class RLImmediateDecisionsSimulation(ImmediateDecisionsSimulation):
    """ImmediateDecisionsSimulation with the request loop exposed as a generator."""

    def _rl_operator(self, op_id):
        """Return operator `op_id`, requiring it to be RL-capable.

        Raising here rather than falling back to `user_request` is deliberate: a silent
        fallback would run greedy while the caller believed a policy was in control, and
        the symptom -- a trained policy that reproduces the baseline exactly -- would not
        surface for weeks.
        """
        op = self.operators[op_id]
        if not hasattr(op, "build_assignment_context"):
            raise TypeError(
                f"sim_env {type(self).__name__} requires every operator to expose "
                f"build_assignment_context, but operator {op_id} is "
                f"{type(op).__name__} (op_module="
                f"{self.list_op_dicts[op_id].get(G_OP_MODULE)!r}). Set op_module to "
                f"RLPoolingIRSOnly, or use sim_env ImmediateDecisionsSimulation for a "
                f"non-RL run."
            )
        return op

    def step_generator(self, sim_time):
        """Mirror of `ImmediateDecisionsSimulation.step`, yielding at each decision epoch.

        Yields a `PendingDecision` and expects the chosen candidate index -- or None to
        reject -- back via `.send()`. Requests resolved internally by
        `build_assignment_context` yield nothing.

        :param sim_time: new simulation time
        """
        # 1)
        self.update_sim_state_fleets(sim_time - self.time_step, sim_time)
        new_travel_times = self.routing_engine.update_network(sim_time)
        if new_travel_times:
            for op_id in range(self.n_op):
                self.operators[op_id].inform_network_travel_time_update(sim_time)
        # 2)
        list_undecided_travelers = list(self.demand.get_undecided_travelers(sim_time))
        last_time = sim_time - self.time_step
        if last_time < self.start_time:
            last_time = None
        list_new_traveler_rid_obj = self.demand.get_new_travelers(sim_time, since=last_time)
        # 3) interleaved per request, so the heuristic re-runs against an updated plan for
        #    the next same-timestamp arrival
        for rid, rq_obj in list_undecided_travelers + list_new_traveler_rid_obj:
            for op_id in range(self.n_op):
                LOG.debug(f"Request {rid}: Checking AMoD option of operator {op_id} ...")
                op = self._rl_operator(op_id)
                ctx = op.build_assignment_context(rq_obj, sim_time)
                if ctx is not None:
                    # the only difference from the parent: the argmin becomes the agent's
                    choice = yield ctx
                    op.commit_assignment_choice(ctx, choice, sim_time)
                amod_offer = op.get_current_offer(rid)
                LOG.debug(f"amod offer {amod_offer}")
                if amod_offer is not None:
                    rq_obj.receive_offer(op_id, amod_offer, sim_time)
            self._rid_chooses_offer(rid, rq_obj, sim_time)

        # 4b) Ritun added: Diffusion model for passenger request cancellation
        self._check_request_cancellations_diffusion_model(sim_time)
        # 5)
        for op in self.operators:
            op.time_trigger(sim_time)
        # 6)
        for ch_op_dict in self.charging_operator_dict.values():
            for ch_op in ch_op_dict.values():
                ch_op.time_trigger(sim_time)
        # record at the end of each time step
        self.record_stats()

    def run_generator(self, tqdm_position=0):
        """Mirror of `FleetSimulationBase.run`, with the time loop delegating to
        `step_generator`.

        Drive it with `.send(choice)`; `StopIteration` marks the end of the episode.
        """
        self._start_realtime_plot()
        t_run_start = time.perf_counter()
        if not self._started:
            self._started = True
            completed = False
            try:
                for sim_time in range(self.start_time, self.end_time, self.time_step):
                    yield from self.step_generator(sim_time)
                    self._update_realtime_plots_dict(sim_time)
                completed = True
            finally:
                # only teardown belongs here. generator.close() raises GeneratorExit at the
                # yield above; this also covers an exception escaping the loop. On the normal
                # path the plot is torn down at the end instead, exactly where run() does it,
                # so ordering is unchanged.
                if not completed:
                    self._end_realtime_plot()

            # finalisation, reached only on a normally completed loop
            self.record_stats()
            # RL-GYM: P1.2 guards, mirrored exactly
            if not self.skip_output:
                self.save_final_state()
            # record_remaining_assignments advances the simulation up to end_time + 14400
            # until every vehicle finishes its route, firing record_boarding /
            # record_alighting_start / record_no_show throughout; record_remaining_users is
            # the end-of-day record_user sweep. Neither is an output operation and neither
            # may run for an aborted episode, which is why they sit outside the finally.
            self.record_remaining_assignments()
            self.demand.record_remaining_users()
        t_run_end = time.perf_counter()
        if not self.skip_output:
            self.evaluate()
        t_eval_end = time.perf_counter()
        # RL-GYM: the per-scenario timing report is noise across a thousand episodes
        if not self.skip_output:
            t_init = datetime.timedelta(seconds=int(t_run_start - self.t_init_start))
            t_sim = datetime.timedelta(seconds=int(t_run_end - t_run_start))
            t_eval = datetime.timedelta(seconds=int(t_eval_end - t_run_end))
            prt_str = f"Scenario {self.scenario_name} finished:\n" \
                      f"{'initialization':>20} : {t_init} h\n" \
                      f"{'simulation':>20} : {t_sim} h\n" \
                      f"{'evaluation':>20} : {t_eval} h\n"
            print(prt_str)
            LOG.info(prt_str)
        self._end_realtime_plot()
