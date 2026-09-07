# SDPDP Gym Environment — Implementation Spec v2

Repository: `FleetPy_stochasticity`, branch: `fleetpy_stochasticity_RL` 
(this has been created by branching out from `no_show_model_update` branch and retains everything from that branch)
Fleet control: `PoolingIRSOnly` → `src/fleetctrl/PoolingIRSOnly.py::PoolingInsertionHeuristicOnly`
Simulation env: `ImmediateDecisionsSimulation`
Simulation config files are: `constant_config_depot_cali_sc_1.csv` and `example_depot_cali_sc_1.csv`
`baseline_user_stats.csv` lives in `docs` directory at the root.
RL library: Stable-Baselines3 + `sb3-contrib` (MaskablePPO)
Companion document: `SDVRP_project_proposal.pdf` — the research design this implements

The action space is top-K ranked candidates, the architecture is generator-based,
and reward is anchored on **pickup**.

---

## 0. Working protocol — READ THIS FIRST

This project is implemented in **manual, permission-gated mode**. The protocol overrides
any default tendency to work autonomously.

**Before writing any code for a work item:**
1. State which work item you are starting (e.g. "P1.3").
2. List every file you will create or modify.
3. Summarise what you will change in each, and flag any `VERIFY` item that turned out
   different from what this document says.
4. **Ask for permission and stop.** Wait for an explicit go-ahead.

**After implementing a work item:**
1. Report exactly which files changed and what changed in each.
2. Run that work item's verification yourself, including a full simulation run where the
   check requires one.
3. Report the result of every check: what passed, what failed, and for each failure your
   diagnosis of the cause — not just the symptom.
4. Record every `VERIFY` item from that work item in the DONE entry, including ones that
   matched — a silent match is indistinguishable from a skipped check. When a `VERIFY`
   item differs from this spec, propose the correction to §2 in the same turn; the spec's
   ground truth must stay true or later work items inherit a false premise.
5. If anything failed, propose a fix and **ask whether to apply it**. Do not apply it
   unasked. Do not proceed to the next work item either way.
6. **Stop.** Do not start the next work item.

Ritun will then re-run the verification independently, confirm the feature works, commit to
the repo, and signal you to continue. Nothing proceeds without that signal. Your
verification does not replace his — it catches failures before he spends simulation time
discovering them.

**On the byte-for-byte comparisons (P1.3, P1.4, P1.10):**
A scripted greedy policy is the same policy as stock `PoolingIRSOnly` — slot 0 is the
`argmin`. There is no legitimate source of difference, so any mismatch indicates a defect.
Report how many rows differ, which columns, and your diagnosis. **Never characterise a
difference as minor or acceptable, and never propose relaxing the comparison.** Likely
causes, in order: the build/commit split consumes global RNG differently from the parent's
`user_request`; `insertion_with_heuristics` has a side effect the split does not restore;
the candidate sort tie-breaks differently; an observation feature is not monotonic in
`delta_cfv`, so the scripted driver is not actually reproducing greedy. Small output
differences are what all four of these look like. Ritun decides what happens next.

**Additional rules:**
- **Do not run git commands that change state.** No commits, branches, stashes, resets, or
  checkouts — Ritun handles version control. Read-only inspection is permitted and expected:
  `git status`, `git log`, `git diff`.
- You may run unit tests and short Python snippets freely. You may also run full
  simulations — use a truncated `end_time` while iterating, and a full run only for the
  final verification of a work item. **Never overwrite `baseline_user_stats.csv`.**
- One work item at a time. Never bundle two, even if the second is trivial.
- Sections marked **DECIDED** are settled. If you believe one is wrong, say so and stop.
  Do not substitute your own design.
- Every item marked **VERIFY** must be checked against the actual source in this repository
  before you rely on it. If a `VERIFY` item contradicts this document, **stop and report**
  rather than working around it — the design decisions depend on those facts.
- Every edit to a file outside `src/rl_gym/` and `dev/` carries a `# RL-GYM:` comment.
- Prefer adding a parameter with a safe default over changing an existing default.
- If you run into any confusion or have any questions, pause and ask Ritun, do not assume something.
  Ritun will tell you what to do and then you can continue.

**If a session is interrupted — usage limit, crash, closed window:**

Maintain `docs/RL_GYM_PROGRESS.md` as an append-only log. Never edit or delete earlier
entries. Before starting a work item, append a STARTED entry. After implementing and
reporting, append a DONE entry. Format:

    ## P1.3 — STARTED <date time>
    Files I will touch: src/rl_gym/__init__.py, src/rl_gym/fleetctrl_rl.py

    ## P1.3 — DONE <date time> — awaiting Ritun's verification
    Files changed: <list>
    Verification: <what you ran, what passed, what failed>
    VERIFY findings:
    - <item>: matched the spec  |  DIFFERED — <what the source actually says>
      (one line per VERIFY item in the work item, including the ones that matched)

    Decisions:
    - Asked: <the question>
      Ritun: <the answer>
      (one entry per time you stopped and asked; omit the section if you never did)

**On resuming, before writing or changing anything:**
1. Re-read this spec in full.
2. Read `docs/RL_GYM_PROGRESS.md`.
3. Run `git status` and `git log --oneline -10`.
4. Report: the last STARTED entry, whether a matching DONE exists, whether the working tree
   is clean, and which commits have landed.
5. State whether you believe the last item completed, partially completed, or never began,
   and what you propose doing next. **Ask before acting.**

An item interrupted mid-implementation leaves uncommitted changes with no matching commit
and no DONE entry. **Do not try to salvage partial work.** Propose reverting to the last
commit and redoing the item cleanly. Ritun decides. Half-finished edits that look plausible
are the most dangerous state this project can be in — the byte-for-byte tests are the only
thing that would catch them, and they only run at the end of an item.

---

## 1. Goal and phasing

Expose the ride-pooling assignment decision of `PoolingIRSOnly` as a Gymnasium environment,
so an SB3 policy replaces the greedy `argmin(delta_cfv)` rule that currently chooses which
vehicle serves each incoming request.

- One gym step = one request for which the operator has an actual choice to make. Requests
  resolved internally with no choice available — origin equals destination, the reservation
  branch, a duplicate rid, or an empty candidate list — do not generate a step. See P1.3.
- One episode = one full simulation day (defined by simulation start and end time).

**Phase 1 — plumbing.** A correct, valid, single-threaded Gym environment with a minimal
observation and a counting-only reward. No learning. Exit gate: a scripted greedy policy
driven through the Gym API reproduces the stock greedy simulation byte for byte.

**Phase 2 — learning.** Full observation (H3 zones, rolling statistics), the real reward
weights, MaskablePPO, baselines, training runs.

---

## 2. Ground truth about this repository

Established by reading the source. `VERIFY` each before depending on it.

### 2.1 The decision point

`PoolingInsertionHeuristicOnly.user_request(self, rq, sim_time)`, around line 88:

1. Builds a `PlanRequest` (`prq`), stores it in `self.rq_dict[rid_struct]` and
   `self.rq_dict_store[rid_struct]`.
2. Auto-rejects if `prq.o_pos == prq.d_pos`.
3. Routes to `self.reservation_module` if the request is beyond `self.opt_horizon`.
4. Calls `insertion_with_heuristics(sim_time, prq, self, force_feasible_assignment=True)`,
   returning `List[Tuple[vid, VehiclePlan, delta_cfv]]`.
5. **`min(list_tuples, key=lambda x: x[2])`** ← the line the RL replaces.
6. Stores the winner in `self.tmp_assignment[rid_struct]`, calls
   `self._create_user_offer(prq, sim_time, vehplan)`.
7. If the list is empty, calls `self._create_rejection(prq, sim_time)`.
8. Records CPU time under `G_FCTRL_CT_RQU`.

`user_request` returns `None`. The offer is fetched separately via `get_current_offer(rid)`.

### 2.2 `ImmediateDecisionsSimulation.step(sim_time)`

Confirmed structure, and it is short and self-contained:

1. `update_sim_state_fleets(sim_time - time_step, sim_time)`; `routing_engine.update_network`;
   `inform_network_travel_time_update` per operator if travel times changed.
2. `list_undecided_travelers = list(demand.get_undecided_travelers(sim_time))` and
   `list_new_traveler_rid_obj = demand.get_new_travelers(sim_time, since=last_time)`.
3. **Interleaved per request**: for each `(rid, rq_obj)` in
   `list_undecided_travelers + list_new_traveler_rid_obj` → `user_request` →
   `get_current_offer` → `receive_offer` → `_rid_chooses_offer`. Then the next request.
4. `_check_request_cancellations_diffusion_model(sim_time)`.
5. `op.time_trigger(sim_time)` per operator.
6. Charging operator triggers; `record_stats()`.

**This already satisfies the proposal's §3.1 requirement** that the insertion heuristic
re-runs after each assignment so an updated vehicle plan is visible for the next
same-timestamp request. No change needed.

Note that step 3 iterates undecided travelers *plus* new arrivals. **VERIFY** whether
`get_undecided_travelers` is ever non-empty under `ImmediateDecisionsSimulation`; if it is,
report it, because §1 defines the epoch on actionable requests, and a re-requesting undecided traveller would be a second epoch for the same rid.

### 2.3 After the offer

`_rid_chooses_offer` → `rq_obj.choose_offer(...)`. With `rq_type = BasicRequest_no_show` and
`stochastic_rider_service_decline = TRUE`, the rider declines with probability

```
1 - (rider_decline_max_wait_time - offered_wait_time) / (rider_decline_max_wait_time - waiting_time_lower_bound)
```

Accepted → `user_confirms_booking` → `assign_vehicle_plan` commits the plan from
`tmp_assignment`. Declined → `user_cancels_request`, `tmp_assignment` discarded.

**The action selects which offer to make, not which vehicle gets the rider.** Acceptance is
a stochastic transition. The reward must distinguish operator rejection from rider decline.

### 2.4 Event choke points for reward

| Event | Method | Calls `record_user`? |
|---|---|---|
| **Pickup** | `Demand.record_boarding` (demand.py ~line 229) | **No** |
| Alighting | `Demand.record_alighting_start` | Yes |
| No-show | `Demand.record_no_show` | Yes |
| Rider decline / operator reject | `FleetSimulationBase._rid_chooses_offer` calls `demand.record_user` directly (lines ~721, ~746) | Yes |
| Diffusion cancellation | `FleetSimulationBase` line ~887 | Yes |
| Undecided leaves system / chose −1 | `FleetSimulationBase` lines ~722, ~747 | Yes |
| End of day | `Demand.record_remaining_users` | Yes |

`Demand.user_cancels_request` (demand.py:350) also calls `record_user`, but is dead code —
nothing invokes it. Do not instrument it. Every reference in the codebase is to
`operator.user_cancels_request` on `FleetControlBase` / `PoolingIRSOnly`, which is a
different method and never touches `Demand`.

Because reward is pickup-anchored, **two callbacks are needed**, not one: `record_boarding`
for pickup and `record_user` for every terminal outcome. `demand.py` also contains
`SlaveDemand(Demand)` from line 297 with its own `record_boarding`. **VERIFY** which class
`ImmediateDecisionsSimulation` instantiates and instrument that one.

Callers live in `FleetSimulationBase.update_sim_state_fleets` around lines 647
(`record_boarding`), 654 (`record_alighting_start`), 669 (`record_no_show`).

### 2.5 Stochasticity present in the simulator

| Uncertainty (proposal §1) | Implemented? | Where |
|---|---|---|
| Travel time | Yes | `NetworkBasic`, `stochastic_tt = TRUE` |
| Post-match booking cancellation | Yes | `FleetSimulationBase._check_request_cancellations_diffusion_model` |
| Rider no-show | Yes | `BasicRequest_no_show`, `PoolingIRSOnly.sim_veh_no_show_requests_cleanup` |
| **Service duration at stops** | **Yes — empirically, already implemented** | Per-request boarding/alighting durations come from the demand file columns `actual_pudo_boarding_duration` and `pudo_alighting_duration`, read in `TravelerModels.py` (~line 561) into `duration_pudo_boarding` / `duration_pudo_alighting` and carried on `PlanRequest` as `real_duration_boarding` / `real_duration_alighting`. With `insertion_with_heterogenous_PUDO_duration = FALSE`, the **planner** assumes the constant `op_const_boarding_time = 300.0` while the **realized** duration comes from the data. That planner-versus-reality gap *is* the service duration uncertainty. Leave it exactly as it is. |

Plus rider decline on offer (`stochastic_rider_service_decline`) and dynamic fleet size
(`TimeBasedFS` + `op_act_fs_file`).

**Do not disable, seed-freeze, or short-circuit any of these.** They are the point of the
formulation.

### 2.6 Constructor side effects — the reset problem

`FleetSimulationBase.__init__` does all of this on **every** instantiation:

- `create_or_empty_dir(self.dir_names[G_DIR_OUTPUT])` — **wipes the output directory**
- `self.save_scenario_inputs()`
- removes all `logging.root.handlers`, then `logging.basicConfig` with a new `FileHandler` —
  **process-global**
- `random.seed(G_RANDOM_SEED)`, `np.random.seed(G_RANDOM_SEED)` — **process-global**
- Loads the routing engine (~4,084 nodes, ~9,602 edges, 52 MB network directory), demand,
  charging, fleet control, vehicles, initial state

`run()` additionally calls at the end: `record_stats()`, `save_final_state()`,
`record_remaining_assignments()`, `demand.record_remaining_users()`, `evaluate()` →
`standard_evaluation`.

There is **no** `skip_output` flag in this fork. You will add one (P1.2).

### 2.7 Module registration

`src/misc/init_modules.py` imports `dev.misc.init_modules` inside a `try/except
ModuleNotFoundError` and calls ten `add_*` functions on it:
`add_dev_simulation_environments`, `add_dev_routing_engines`, `add_request_models`,
`add_fleet_control_modules`, `add_repositioning_modules`, `add_charging_strategy_modules`,
`add_dynamic_pricing_strategy_modules`, `add_dynamic_fleetsizing_strategy_modules`,
`add_reservation_strategy_modules`, `add_ride_pooling_batch_optimizer_modules`.

Use this hook. Do not edit `get_src_fleet_control_modules()` directly.

### 2.8 Scale and the reference scenario

**The reference scenario is the one `run_examples.py` actually runs**, not the
`constant_config_depot.csv` / `example_depot.csv` pair, which is stale and materially
different. Use:

- `studies/example_study/scenarios/constant_config_depot_cali_sc_1.csv`
- `studies/example_study/scenarios/example_depot_cali_sc_1.csv`

Key parameters:

| Parameter | Value |
|---|---|
| `start_time` / `end_time` | 25200 / **70000** (~12.4 h) |
| `time_step` | 10 → 4,480 simulation steps per episode |
| `op_fleet_composition` | **`default_vehtype:12`** |
| `op_max_wait_time` | **1500.0** |
| `waiting_time_lower_bound` | 700.0 |
| `rider_decline_max_wait_time` | 1500.0 |
| `op_max_detour_time_factor` | 20.0 |
| `op_const_boarding_time` | 300.0 (planner assumption only, see §2.5) |
| `no_show_wait_time` | 180.0 |
| `rq_file` / `day_dir_name` | `2024-06-20.csv` / `thursday` |


Demand `2024-06-20.csv` has **445 requests**. Expect **fewer than 445 gym steps** — requests
resolved internally generate no step (§1), and on the baseline that was about 9 empty-candidate cases plus any reservation-branch requests. Two of the 445 carry `no_show_status = True`.

Because `rider_decline_max_wait_time = 1500` and `waiting_time_lower_bound = 700`, the rider-decline
window from §2.3 is only 800 s wide: decline probability is 0 at an offered wait of 700 s and
1 at 1500 s. Every observation normalizer must read these from `scenario_parameters`.

The proposal specifies a 5-second time step, which would double simulation steps to ~9,000
per episode and roughly double wall-clock, while leaving the gym step count unchanged.
**Confirm with Ritun before changing `time_step`** — it is pure overhead for the RL loop.

`example_depot_cali_sc_1.csv` contains 10 rows, of which only the `random_seed = 42` row is
active — the rest are disabled by a `#` prefix on `op_module`. `baseline_user_stats.csv` 
was generated from that row; the run's own output directory is
`studies/example_study/results/example_depot_time_pool_irsonly_rs_42/`.

---

## 3. DECIDED design choices

### D1. Generator-based control inversion. No threads.

The gym owns the time loop. `RLImmediateDecisionsSimulation` exposes a generator that yields
at each decision epoch and receives the action via `.send()`.

This is a change from v1, which specified a background thread and a queue pair. The proposal's
design — decisions only at request arrival, requests processed FIFO with the wall clock
frozen, never pausing mid-time-step — makes the generator approach viable, and
`ImmediateDecisionsSimulation.step()` is short and self-contained enough to mirror.

What this buys: no threads, no queues, no abort sentinel, no poll timeouts, no liveness
checks, no `error_slot`. The environment is steppable in a normal debugger and profileable
with a normal profiler.

What it costs: mirroring ~35 lines of `step()` and ~20 lines of the `run()` loop in a
subclass. Acceptable — this fork already diverges heavily from upstream.

Fall back to threads only if a future decision point cannot be reached from `step()` without
restructuring. Repositioning (deferred) fires in `op.time_trigger(sim_time)`, which is inside
`step()` at a fixed point, so a second yield site there is equally easy.

### D2. Subclass, never edit, `PoolingInsertionHeuristicOnly`.

`RLPoolingIRSOnly(PoolingInsertionHeuristicOnly)` splits `user_request` into two methods and
overrides nothing else. Offers, confirmations, cancellations, no-show cleanup, and
`_return_expected_pickup_time` are all inherited. The diffusion and no-show work stays
untouched and non-RL scenarios keep running.

### D3. Action space is `Discrete(K + 1)` over **ranked candidates**.

Per proposal §3.1.2 and §3.2. Slot `k` for `k in [0, K)` = "offer the insertion plan for the
`k`-th cheapest candidate". Slot `K` = "reject". Reject is always legal.

- `K` is a config parameter. Start at `K = 8`. `K` is owned by `SDPDPAssignmentEnv` alone. 
  `RLPoolingIRSOnly` receives a semantic choice(candidate index or reject), never a raw action.
- Candidates are used in the order `insertion_with_heuristics` returns them. That list is
  already sorted ascending by `delta_cfv` (`insertion.py` line ~569, a stable sort), so
  element 0 is exactly what stock's `min(list_tuples, key=lambda x: x[2])` selects. 
  **Do not re-sort and do not add a tie-break.** Any re-sort risks diverging from stock on
  tied costs: the upstream order within ties comes from a backwards Dijkstra ordered by
  proximity to the request origin, not by vid, so a `vid` tie-break would pick a different
  vehicle whenever tied candidates sit at different positions — and break the byte-for-byte
  exit gate.
- When fewer than `K` candidates return, pad the remaining slots and mask them.
- Slot 0 is by construction the greedy choice, so the policy is learning *when to deviate
  from greedy* — which is research question 2.

The ranked design is fleet-size-independent, so one trained policy stays valid
across fleet-sizing scenarios.

Use `sb3_contrib.MaskablePPO`. `SDPDPAssignmentEnv` implements `action_masks()` directly
(P1.8). **Do not use `ActionMasker`** — sb3-contrib documents that it cannot be used with
`SubprocVecEnv`, which D8 requires, and masks would silently fail to reach the policy.
Do not use plain PPO with an unmasked space and an invalid-action penalty.

For evaluation, use `MaskableEvalCallback` from `sb3_contrib.common.maskable.callbacks` and
`evaluate_policy` from `sb3_contrib.common.maskable.evaluation`. The base SB3 versions ignore
masks and will report misleading results.

### D4. Reward is anchored on **pickup**, not trip completion.

Per Ritun's decision. No positive reward at assignment; the positive reward fires when the
rider is successfully picked up. See §6 for the full specification and for the one
consequence that needs a deliberate choice.

### D5. Fixed-size observation, `K` candidate slots.

Phase 1 ships a deliberately minimal observation with the correct shape and mechanism.
Phase 2 replaces the contents without touching any plumbing. See §5.

### D6. Episode end is `terminated=True`, not `truncated`.

One episode is one service day, no state carries to the next day, and normalized time-of-day
is in the observation, so the critic can learn `V → 0` at the horizon. Flush all pending
reward events into the final step.

The final flush includes reward events from `record_remaining_assignments`, which completes
in-flight trips after `end_time`. `w_horizon` therefore applies only to riders never picked
up even after that tail, which is the intended meaning.

### D7. Per-episode reseeding is mandatory.

`reset(seed=...)` writes a fresh `scenario_parameters[G_RANDOM_SEED]` before the simulation
is constructed. Per proposal §3.1.2, each day is replayed under several seeds and the agent
must see demand and travel-time variation. Without this every episode replays one sample
path.

### D8. `SubprocVecEnv`, never `DummyVecEnv`.

`FleetSimulationBase.__init__` reconfigures process-global logging and global RNG state.
One env per OS process.

---

## 4. Relationship to the project proposal

### Implemented as specified

Decision epochs are per actionable request. In practice a request with no available choice — origin equals
destination, reservation branch, duplicate rid, or no feasible candidate — is resolved inside
`build_assignment_context` and generates no gym step, because there is nothing for the policy
to decide. One episode per day; days replayed across seeds and
across 11 months (Ritun may change the number of months to be used later) of demand files;
`K+1` action space with reject always available; no immediate positive reward for acceptance; 
delayed credit; explicit penalty for post-match
cancellation; FIFO handling of same-timestamp arrivals with the heuristic re-running between
them (already native to `ImmediateDecisionsSimulation`).

### Deferred to Phase 2

H3 zone indexing, all rolling-window statistics, the rate-anomaly features, the full
candidate and global feature sets, reward weight design, training.

### Gaps that need decisions from Ritun

1. **Service duration uncertainty is already implemented and must not be touched.** It is
   data-driven rather than distribution-driven: realized boarding durations come from the
   demand file (mean ~56 s, range 0–473 s on `2024-06-20.csv`) while the planner assumes
   `op_const_boarding_time = 300.0`. All four of the proposal's uncertainties are therefore
   present. Do not add stochastic draws, do not add related globals or config keys, and do
   not propose replacing the empirical mechanism. See §10.
2. **No-show penalty asymmetry.** The proposal gives cancellation both a forfeited reward and
   an explicit penalty, but gives no-show only the forfeited reward. Operationally no-show is
   the more expensive failure — the vehicle drove to the pickup and idled for
   `no_show_wait_time`. Since rider no-show is a headline uncertainty, it needs its own
   penalty term or the policy has no gradient toward avoiding no-show-prone assignments.
   §6 includes one; the weight is Ritun's call or check with Ritun if he wants to consider the no-show in the reward signal at all.

---

## 5. Observation

### 5.1 Phase 1 — minimal

Length `3 * K + 4`. For `K = 8` that is **28**. `float32`.

**Per candidate slot `k`** (3 features × K):
1. `is_valid` — 1 if a candidate occupies this slot
2. `delta_cfv_k`, transformed as `sign(x) * log1p(|x|)`. No further scaling in Phase 1 —
   the transform already compresses the range and `VecNormalize` (§5.3) handles the rest in
   Phase 2. Do not invent a scaling constant.
3. offered wait time under plan `k`: `pax_info[rid][0] - prq.rq_time`, divided by
   `op_max_wait_time`

**Global (4)**:
1. `sin(2π · t_of_day)`
2. `cos(2π · t_of_day)`
3. fraction of the day elapsed
4. fraction of active vehicles currently idle: `IDLE` count divided by the count of vehicles
   whose status is not `OUT_OF_SERVICE`. `OUT_OF_SERVICE` (status 5) is what
   `veh_search_for_immediate_request` itself skips (`searchVehicles.py` line ~21), so it is
   the operative definition of inactive. `IDLE` means `VRL_STATES.IDLE` only — not `WAITING`,
   `PLANNED_STOP` or `REPO_TARGET`. Guard the divide: if no vehicle is active, emit 0.

All wait-related features divide by `op_max_wait_time` read from `scenario_parameters`
(1500 in the reference scenario), never a literal.

Feature 2 is required for the Phase 1 exit gate — the scripted greedy driver picks
`argmin` over the `delta_cfv` slots, so it must be readable from the observation.
Do NOT add an implied-decline-probability feature, in either phase. It is derivable
from feature 3 and `scenario_parameters`, and Ritun has excluded it from the state
design. Do not propose it.

### 5.2 Phase 2 — full

Per proposal §3.1.1–§3.1.3. Same `Box`, same observer→dict→concat mechanism, wider vector.

**Request block**: H3 zone ID for pickup and dropoff; direct travel time; time of day, day of
week, month; number of passengers; cancellation rate over the last 15–20 min in the request
zone plus its 6 H3 neighbors; rejection-rate anomaly for the destination zone plus its 6
neighbors over the last 15 min.

**Candidate block** (per slot): vehicle location as H3 zone; residual capacity; directionality
score in `{−1, 0, +1}`; ETA to the current request's pickup; the vehicle's ETA health for
already-assigned riders; insertion cost; number of assigned-but-unserved requests in the vehicle plan.

**Global block**: fraction of fleet idle; fraction of fleet carrying an infeasible-plan flag
(latest pickup violated, max trip time violated, latest arrival violated); per-minute arrivals
over the last 5/10/20 min normalized by the 95th-percentile historical rate; rejection rate;
fraction cancelled over the last 5/20 min; mean and median promised-vs-realized ETA gap for
cancelled riders and separately for picked-up riders; fraction of riders declining the initial
quote.

**Position encoding is a config key, not a fixed choice.** `position_encoding` takes
`"h3"` (zone ID), `"xy"` (normalized coordinates from
`routing_engine.return_network_bounding_box()`), or `"relative"` (offset from the request
origin). It applies to the request origin and destination and to each candidate's vehicle
location. Every observer branches on it internally and reports its own contribution to the
observation length, so switching is a config change and never a code change. H3 remains the
aggregation key for all rolling statistics regardless of this setting.

Two new components are needed and neither exists yet:

- **`RollingStatsTracker`** — subscribes to the same callbacks as the reward tracker, keeps
  deques of recent events keyed by H3 cell, exposes windowed aggregates.
- **Historical baselines** — an offline preprocessing step over the 11 months of demand
  files producing per-zone arrival rates and rejection rates. "Anomaly" is undefined without
  them.

### 5.3 Normalization

FleetPy quantities are raw seconds and metres. Two layers:

1. Manual scaling using values read from `scenario_parameters` — `op_max_wait_time`, the
   network bounding box from `routing_engine.return_network_bounding_box()`. **Never hardcode
   any wait-time literal** or any other scenario-dependent constant.
2. `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0)` on top (Phase 2). Save and
   reload its statistics with the model — evaluating with mismatched `VecNormalize` stats is
   a silent and very common failure.

---

## 6. Reward

### 6.1 Structure

Reward events accumulate in a `RewardTracker` and are flushed each gym step. `step(action)` returns the sum of everything that fired after that action was applied, up to
the next decision point — the action's own immediate consequence plus whatever the simulation
produced while advancing. `flush()` is called after `.send()`, never before. The simulation is
frozen while the agent decides, so the window boundaries are exact.

| Event | Fires at | Term |
|---|---|---|
| Operator rejected | `record_user`; labelled at the decision by `commit_assignment_choice` | `-w_reject` |
| Rider declined the offer | `record_user` from `_rid_chooses_offer` | `-w_decline` |
| Post-match cancellation | Diffusion model via `record_user` | `-w_cancel` |
| No-show | `record_no_show` via `record_user` | `-w_noshow` |
| **Pickup** | **`record_boarding`** | `+w_pickup - w_wait * (pu_time - rq_time)/60` |
| Unserved at horizon | End of episode | `-w_horizon` per request accepted but never picked up (pu_time is None) at end of episode |
| No candidates | `record_user`; labelled by `build_assignment_context` when the list is empty | `-w_no_candidates` |

Starting weights, to be tuned by Ritun:
`w_pickup=1.0, w_wait=0.05, w_reject=0.5, w_decline=0.1, w_cancel=0.6, w_noshow=0.8, w_horizon=0.0, w_no_candidates=0.0`.

All weights come from the env config. Do not hardcode.

### 6.2 Why pickup-anchoring works well here

Charging `-w_wait` at each rider's *own* pickup automatically internalizes the delay that an
insertion imposes on other already-assigned riders. If accepting request *j* pushes back the
pickups of three riders already on that vehicle's plan, those three each incur a larger wait
penalty at their own pickup events. The pooling tradeoff is priced correctly without any
explicit "delay imposed on others" reward term.

It also shortens the credit path considerably relative to completion-anchoring: pickup
typically follows assignment by minutes, whereas dropoff can be 30+ minutes later.

### 6.3 What pickup-anchoring deliberately leaves unpriced

In-vehicle time after pickup, and fleet kilometres, carry no reward term. Reward is
throughput plus pre-pickup wait. This is settled — do not add a detour term, a distance
term, or a planned-ride-time term, and do not propose them.

### 6.4 Classifying terminal outcomes

`record_user` is reached from six live paths, one of which is success. Classification order
matters; the first match wins.

1. `rid` in the no-candidates set → no candidates
2. `rid` in the operator-rejection set → operator rejected
3. `rq.do_time is not None` → served; no terminal penalty
4. `rq.diffusion_cancelled` → post-match cancellation
5. `rq.rider_declined` → rider declined
6. `rq.pu_time is None` and `rq.no_show` → no-show
7. otherwise → unclassified; count it, charge nothing, and report the count in
   `episode_summary()`. A non-zero count here means a path was missed.

**There is no fallback to rider decline.** Anything unmatched is unclassified, not a decline.

`rq.no_show` is set at load time from the demand file column (`TravelerModels.py:555`), so it
is a rider attribute, not an outcome. It must be checked *after* `diffusion_cancelled` and
must be gated on `pu_time is None`, or a no-show-flagged rider who cancelled while waiting
gets reported as a no-show.

`note_rejection` and `note_no_candidates` only **label** — they record the rid in a set and
charge nothing. All reward is charged once, in `on_exit`. `on_exit` must be idempotent per
rid: `record_alighting_start` and `record_remaining_users` can both fire for a rider who is
mid-alighting when the day ends.

### 6.5 Discounting

A pickup reward can land 10–30 decisions after the assignment that caused it. Set
**`gamma = 0.999`**, not SB3's default of 0.99. With ~440 steps per episode, 0.99 gives an
effective horizon of ~100 decisions, which is marginal for pickup credit.

### 6.6 Phase 1 behaviour

The tracker records and **counts** every event. Use the weights in §6.1 as given, including
`w_horizon = 0.0` — Phase 1 delivers the wiring and the event counts, not a tuned reward.
No weight design in Phase 1; the economics are Phase 2.

---

## 7. Phase 1 work items

Each is independently implementable, verifiable, and committable. Follow the §0 protocol
around every one.

---

### P1.1 — `dev` extension package

**Files:** `dev/__init__.py`, `dev/misc/__init__.py`, `dev/misc/init_modules.py`

Expose all ten `add_*` functions from §2.7. Eight return `{}`. The two that matter:

```
add_fleet_control_modules()      -> {"RLPoolingIRSOnly": ("src.rl_gym.fleetctrl_rl", "RLPoolingIRSOnly")}
add_dev_simulation_environments() -> {"RLImmediateDecisionsSimulation": ("src.rl_gym.sim_env_rl", "RLImmediateDecisionsSimulation")}
```

Point them at modules that do not exist yet; that is fine, the dict values are lazy.

**VERIFY** the exact ten function names against `src/misc/init_modules.py`. Note the
inconsistency between `add_dev_simulation_environments` and `add_simulation_environments`.
Missing one raises `AttributeError` at import time for **every** simulation in the repo, RL
or not.

**How Ritun verifies:** `python -c "import src.misc.init_modules"` succeeds;
`get_src_fleet_control_modules()["PoolingIRSOnly"]` still resolves; the existing greedy
scenario still runs and produces output identical to `baseline_user_stats.csv`.

**Commit:** `RL-GYM: add dev extension package for module registration`

---

### P1.2 — Output, logging, and seed guards

**Files:** `src/misc/globals.py`, `src/FleetSimulationBase.py`

Add `G_SKIP_OUTPUT = "skip_output"` and `G_RL_MODE = "rl_mode"`.

Guard behind `if not self.scenario_parameters.get(G_SKIP_OUTPUT, False):`
- `create_or_empty_dir(self.dir_names[G_DIR_OUTPUT])`
- `self.save_scenario_inputs()`
- in `run()`: `save_final_state()`, `demand.record_remaining_users()`, `evaluate()`
- in `_load_fleetctr_vehicles()`, line ~432: `veh_type_df.to_csv(veh_type_f, index=False)`,
  which writes `2_vehicle_types.csv`. Guard the write only — `veh_type_list` and
  `self.sim_vehicles` must still be built, so nothing in memory changes. This is a separate
  method from `save_scenario_inputs` and is called from `__init__` at line ~295, so it is
  easy to miss.

`record_stats()` fires at the end of **every** simulation time step
(`ImmediateDecisionsSimulation.step()` line ~119) with `force=True` by signature default, so
it writes on any step where a buffer is non-empty. Guarding only the `run()` call is
insufficient. Guard the three write sites, not the `record_stats` call itself, and each site
must still clear its buffer:
`Demand.save_user_stats` (~line 182), `record_stats` (~line 620), and
`FleetControlBase.record_dynamic_fleetcontrol_output` (~line 860).

**Do NOT guard `record_remaining_assignments()`.** Despite its name it is not an output
operation — it advances the simulation past `end_time` (up to `end_time + 14400`) until every
vehicle finishes its assigned route, and `update_sim_state_fleets` fires `record_boarding`,
`record_alighting_start` and `record_no_show` throughout. Riders in flight at `end_time` are
picked up and dropped off during this tail (D6). Suppressing it would mean the agent trains
against a different reward than it is evaluated on. It writes nothing itself — its only output
is `record_stats()` at line ~604, already covered by the write-site guards above — so it runs
identically in both modes.

**`record_user` must keep running with output off** — the reward callbacks ride on it. Only
file writing is suppressed.

**VERIFY** that all three buffers still clear when writes are suppressed: `user_stat_buffer`,
`op_output[op_id]`, and `dyn_output_dict`. The third is on the operator, not the demand
object, and is the one an early-return from `record_stats` would silently skip.

**VERIFY** that every file write in `src/FleetSimulationBase.py` is covered by a guard. Grep
for `to_csv`, `.write(`, and `open(` with a write mode. There are five write sites in the
current source: lines ~432, ~448, ~569, ~620. Lines 448 and 569 sit inside
`save_scenario_inputs` and `save_final_state`, which are already guarded. Confirm line 620 is
reachable only through `record_stats`, and report any site that is not covered.

**VERIFY** that nothing under `src/evaluation/` reads `2_vehicle_types.csv`. If something
does, note it — `evaluate()` is guarded by the same flag, so the two are skipped together and
evaluation runs are unaffected, but the dependency should be on record.

Logging: guard the handler setup so it runs once per process.

```
# RL-GYM: configure process logging exactly once
if not getattr(FleetSimulationBase, "_logging_configured", False):
    ... existing handler setup ...
    FleetSimulationBase._logging_configured = True
```

In RL mode with `G_SKIP_OUTPUT`, force `log_level` to warning with a `NullHandler`. Left
unguarded, 1,000 episodes in one worker opens 1,000 file handlers.

**How Ritun verifies:** the greedy scenario with default config still matches
`baseline_user_stats.csv` exactly; the same scenario with `skip_output = 1` produces no files
and does not erase an existing results directory; constructing the sim object twice in one
process does not add a second log handler.

**Commit:** `RL-GYM: add skip_output flag and guard global logging setup`

---

### P1.3 — Split `user_request` in `RLPoolingIRSOnly`

**Files:** `src/rl_gym/__init__.py`, `src/rl_gym/fleetctrl_rl.py`

```
class PendingDecision:
    prq, candidates (list of (vid, vehplan, delta_cfv) exactly as insertion_with_heuristics
    returned it — do not re-sort), sim_time, rid_struct, cpu_t0

class RLPoolingIRSOnly(PoolingInsertionHeuristicOnly):
    def build_assignment_context(self, rq, sim_time) -> PendingDecision | None
    def commit_assignment_choice(self, pending, choice: int | None, sim_time) -> None
    def user_request(self, rq, sim_time)   # greedy fallback: build + commit with choice 0
    def set_reward_tracker(self, tracker) -> None
```

`build_assignment_context` performs everything `user_request` does up to the `argmin`. It
returns `None` when the request was fully resolved internally and no decision is needed:
`o_pos == d_pos` auto-reject, the reservation branch, a duplicate rid, or an empty candidate
list. In those cases it must reproduce the parent's behaviour exactly, including
`_create_rejection` and the `G_FCTRL_CT_RQU` bookkeeping.

Sorting has already happened inside `insertion_with_heuristics`; `build_assignment_context`
stores the list as returned. Do not re-sort it — see D3.

`commit_assignment_choice` takes a semantic choice, not a raw action. `choice` is an index
into `pending.candidates`, or `None` to reject. On an index it sets `tmp_assignment`, calls
`_create_user_offer`, and closes the CPU-time bookkeeping; on `None` it calls
`_create_rejection`. Assert `choice is None or 0 <= choice < len(pending.candidates)` — this
catches an off-by-one in the env's translation, which is otherwise silent.

`RLPoolingIRSOnly` knows nothing about `K` or the action space. `K` lives only in the env
(P1.8), which owns the single source of truth and performs the translation.

`user_request` must be a faithful greedy fallback so the class works outside RL: build, then commit with choice 0. 
Since slot 0 is the minimum by construction, this is exactly the
parent's behaviour.

`RLPoolingIRSOnly` holds an optional reward tracker, attached via
`set_reward_tracker(tracker)`, defaulting to `None` so non-RL scenarios are unaffected. Two
call sites, both guarded by a `None` check: `commit_assignment_choice` calls
`tracker.note_rejection(rid)` when `choice is None`; `build_assignment_context` calls
`tracker.note_no_candidates(rid)` on the empty-candidate branch only. Both only **label** —
they record the rid in a set and charge no reward. All charging happens in `on_exit` (§6.4).
Both are pure bookkeeping and must not perturb the simulation; the byte-for-byte test below
confirms it.

**VERIFY** that `insertion_with_heuristics` has no side effects on `fleetctrl` state beyond
what the parent already tolerates — read
`src/fleetctrl/pooling/immediate/insertion.py::immediate_insertion_with_heuristics`. If it
mutates `veh_plans` or `rid_to_assigned_vid`, the split path must restore state identically.

**VERIFY** that `op_vpi_nr_plans` (`G_VPI_KEEP`) is unset so at most one plan per vid is
returned. If more than one can appear, keep the best per vid so `vid → plan` is a function.

**How Ritun verifies:** run the existing scenario with `op_module = RLPoolingIRSOnly` and no
gym involved. Output must match `baseline_user_stats.csv` **byte for byte**. This is the most
important test in Phase 1 — it proves the split does not perturb the simulation.

**Commit:** `RL-GYM: split PoolingIRSOnly.user_request into build/commit with greedy fallback`

---

### P1.4 — `RLImmediateDecisionsSimulation` generator

**Files:** `src/rl_gym/sim_env_rl.py`, `src/misc/init_modules.py`

```
class RLImmediateDecisionsSimulation(ImmediateDecisionsSimulation):
    def step_generator(self, sim_time)   # mirrors step(), yields PendingDecision, receives action via send()
    def run_generator(self)              # mirrors the run() loop, `yield from step_generator(t)`, then finalisation
```

`step_generator` mirrors §2.2 exactly. Inside the request loop:

```
ctx = op.build_assignment_context(rq_obj, sim_time)
if ctx is None:
    pass                      # already resolved internally
else:
    action = yield ctx
    op.commit_assignment_choice(ctx, action, sim_time)
amod_offer = op.get_current_offer(rid)
...
self._rid_chooses_offer(rid, rq_obj, sim_time)
```

`run_generator` mirrors `run()`'s loop over `range(start_time, end_time, time_step)` and then
runs the same finalisation, subject to the P1.2 guards. Wrap the loop in `try/finally` so
`generator.close()` unwinds cleanly.

**Do not** add `hook_manager` or any new positional parameter to
`FleetSimulationBase.__init__`. If `load_simulation_environment` needs a change, make it a
keyword argument with a default.

**How Ritun verifies:** a small driver script that calls `run_generator()` and sends action 0
at every yield produces output identical to `baseline_user_stats.csv`. Still no gymnasium
involved.

**Commit:** `RL-GYM: add generator-driven simulation environment`

---

### P1.5 — Candidate truncation and action masking

**Files:** `src/rl_gym/spaces.py`

Top-`K` truncation, padding, and mask construction. **No sorting or tie-breaking** — the
list arrives ordered from `insertion_with_heuristics` and truncation takes a prefix (D3).
Slot `K` (reject) is always `True`. The mask must never be all-`False`.

**How Ritun verifies:** unit tests — truncation preserves the input order exactly; fewer
than `K` candidates pad and mask correctly; `K` larger than the candidate count leaves the
list unchanged and masks only the surplus slots.

**Commit:** `RL-GYM: add candidate truncation and action masking utilities`

---

### P1.6 — Minimal observation builder

**Files:** `src/rl_gym/observers.py`

`AbstractObserver.observe(fleetpy_module, ctx) -> dict`, plus `CandidateObserver` and
`GlobalStateObserver` producing §5.1. Concatenation happens in the env. Keep the two
observers separate so each can be unit-tested for shape independently.

Read all normalization constants from `scenario_parameters`.

Each observer exposes its own output length, and the env builds `observation_space` by
summing them. Do not hardcode the shape anywhere. Phase 2 changes the feature set and the
position encoding (§5.2), and the space must follow automatically.

`CandidateObserver` reads `K` from the env config — the same value the env uses for
truncation and translation. Do not parameterise it separately.

**How Ritun verifies:** shape is `(3*K + 4,)`, dtype `float32`, no NaN or inf across a full
scripted run; slot 0's `delta_cfv` is always the minimum.

**Commit:** `RL-GYM: add minimal observation builders`

---

### P1.7 — `RewardTracker` with event wiring

**Files:** `src/rl_gym/reward.py`, `src/demand/demand.py`

```
class RewardTracker:
    def __init__(self, weights: dict)
    def on_pickup(self, rid, rq)          # from record_boarding
    def on_exit(self, rid, rq)            # from record_user
    def note_rejection(self, rid)         # from commit_assignment_choice
    def note_no_candidates(self, rid)     # from build_assignment_context
    def flush(self) -> float
    def episode_summary(self) -> dict
```

Two callbacks in `demand.py`, both optional and both defensive:

```
# RL-GYM: optional observers for RL reward attribution
self._boarding_callback = None
self._exit_callback = None
```

Called at the end of `Demand.record_boarding` and inside `Demand.record_user` respectively,
each wrapped in `try/except` so a reward bug can never kill a simulation.

**VERIFY** which of `Demand` / `SlaveDemand` is instantiated by `ImmediateDecisionsSimulation`
and instrument that one — expected to be `Demand`, with `SlaveDemand` reserved for MobiTopp
coupling, but confirm before wiring.

Phase 1 uses counting-only weights (§6.6).

**How Ritun verifies:** across a scripted greedy run, the tracker's counts reconcile against
`baseline_user_stats.csv`:
- pickups = rows with non-empty `pickup_time`
- no-shows = rows with `no_show_stat = True`
- post-match cancellations = rows with `diffusion_cancelled = True`
- rider declines = rows with `rider_declined = True`
- no-candidates = rows with `rider_declined = False`, empty `pickup_time`, and both
  `diffusion_cancelled` and `no_show_stat` `False` — 9 on the reference day
- operator rejections = 0, since a scripted greedy policy never selects slot `K`
- unclassified = 0 — every request that reaches `record_user` matched one of the branches
  in §6.4

Any mismatch means a choke point was missed.

**Commit:** `RL-GYM: add reward tracker with pickup and exit event wiring`

---

### P1.8 — `SDPDPAssignmentEnv`

**Files:** `src/rl_gym/gym_env.py`

`gymnasium.Env` subclass. `reset()` builds a fresh simulation, creates the generator, advances
to the first decision. `step(action)` uses `.send(action)`, catches `StopIteration` for
episode end. `close()` calls `generator.close()`. `action_masks()` returns the mask that
arrived with the current observation. `action_masks()` must be callable at any point after `reset()` or `step()` returns, without
advancing the simulation. Store the mask alongside the observation when the generator yields.

Config keys: `constant_cfg_path`, `scenario_cfg_path`, `scenario_pool`, `env_id`, `K`,
`skip_output`, `reward_weights`, `base_seed`.

`SDPDPAssignmentEnv` owns `K` and is the only place it appears. Per decision it truncates
`pending.candidates` to the first `K` for the observation and mask — a prefix, so slot
indices map directly onto candidate indices — and translates the policy's action before
calling down: `action == K` → `commit_assignment_choice(pending, None, sim_time)`;
`action < K` → `commit_assignment_choice(pending, action, sim_time)`. Actions at or beyond
`len(candidates)` other than `K` are masked and must never be sent; assert rather than
handle them.

`__init__` loads `ConstantConfig` + `ScenarioConfig` from `src/misc/config.py`, forces
`sim_env = RLImmediateDecisionsSimulation`, `op_module = RLPoolingIRSOnly`, sets
`G_SKIP_OUTPUT`, `G_RL_MODE`, `n_cpu_per_sim = 1`, `evaluate = 0`.

After `reset()` builds the simulation, attach the tracker before advancing the generator:
`sim.operators[0].set_reward_tracker(tracker)`, plus the two `demand.py` callbacks from P1.7.

**VERIFY** the `ConstantConfig` / `ScenarioConfig` API and the config-addition idiom against
`src/misc/config.py`. This fork uses CSV, not YAML.

Keep all diagnostic state in this class. Do not put subclass-specific state in a base class.

**How Ritun verifies:** `gymnasium.utils.env_checker.check_env(env)` passes; `reset()` returns
the declared shape and dtype; 10 arbitrary valid actions step without error; 20 reset/close
cycles leak no memory and no file handles.

**Commit:** `RL-GYM: add SDPDPAssignmentEnv gymnasium environment`

---

### P1.9 — Seeding and scenario pool

**Files:** `src/rl_gym/gym_env.py`

`reset(seed=None)` seeds the env's own `np.random.Generator` on first call, then draws an
episode seed from it and writes `scenario_parameters[G_RANDOM_SEED] = int(episode_seed)`
before construction.

Rotate the scenario row across episodes from `scenario_pool`, so the policy sees multiple
demand days (proposal §3.1.2). Each `SubprocVecEnv` worker gets a different base seed derived
from `env_id`.

When `skip_output` is False, set
`scenario_name = f"{base}_env{env_id}_pid{os.getpid()}_ep{counter}"`. Both the pid and the
counter are needed — pid because independently-launched processes can share an `env_id`,
counter because `create_or_empty_dir` would otherwise erase the previous episode.

**How Ritun verifies:** two resets with the same seed give identical trajectories under a
fixed action sequence; two resets with different seeds produce different `1_user-stats.csv`
output. Both must hold.

**Commit:** `RL-GYM: add per-episode seeding and scenario pool rotation`

---

### P1.10 — Phase 1 exit gate: scripted greedy rollout

**Files:** `tests/test_rl_gym.py`

Drive the env entirely from outside: `obs, _ = env.reset()`, then at each step choose
`argmin` over the `delta_cfv` slots in the observation, step, repeat until `terminated`. Run
with `skip_output = 0` so a user-stats file is produced.

The resulting `1_user-stats.csv` must match `baseline_user_stats.csv` **byte for byte**.

If it passes, the generator, observation extraction, ranking, masking, action translation,
reward wiring, seeding, and episode termination are all correct simultaneously — with no RL
library involved.

**If it nearly matches, do not relax the comparison to a tolerance.** A near-match means the
RL path consumes global RNG differently from the greedy path, and that will quietly
contaminate every stochastic result reported later. Report it and stop.

Also report: step count, wall-clock seconds, steps/second, reward-event breakdown, and how
many requests took the reservation branch or returned an empty candidate list.

Also report the distribution of wall-clock gaps between consecutive gym steps
(min, median, mean, 95th percentile, max).

Write these figures to `docs/RL_GYM_PHASE1_RESULTS.md` as well as reporting them: step
count, reservation-branch count, empty-candidate count, wall-clock seconds, steps/second,
the full `episode_summary()` breakdown, and the candidate-list length distribution
(min, median, mean, max, and the count exceeding `K`).

**How Ritun verifies:** the byte comparison, plus a step-count reconciliation:
step count + reservation-branch count + empty-candidate count + same-origin-destination
count + duplicate-rid count = 445. Every request must be accounted for in exactly one bucket.

**Commit:** `RL-GYM: add scripted greedy rollout test (Phase 1 exit gate)`

---

### P1.11 — `SubprocVecEnv` smoke test

**Files:** `train_sdpdp.py` (skeleton only)

`make_env(cfg, env_id)` closure; envs constructed **inside** `_init()`, never before
`SubprocVecEnv`. FleetPy objects hold routing engines and file handles and are not picklable.

4 workers, random masked actions, one episode each. No learning algorithm yet.

Append to (creating if absent) `docs/RL_GYM_PHASE1_RESULTS.md`: worker count, per-episode wall clock under
four workers, peak memory, and whether any process orphaned or any log file interleaved.

**How Ritun verifies:** four processes run to completion, no crashes, no orphaned processes
after `env.close()`, no interleaved or corrupted log files.

**Commit:** `RL-GYM: add SubprocVecEnv smoke test harness`

---

## 8. Phase 2 work items

Do not begin any P2 item until Ritun confirms Phase 1 is closed and this section
has been expanded into full work items. These bullets are topics, not specifications.

Order matters.

- **P2.1** H3 zone system: `h3` dependency, node→hex mapping preprocessing, neighbour lookup,
  integration as a FleetPy zone system
- **P2.2** Offline historical baselines from the 11 months of demand files: per-zone arrival
  rates, rejection rates, 95th-percentile normalizers
- **P2.3** `RollingStatsTracker` on the P1.7 callbacks
- **P2.4** Full observation per §5.2, plus `VecNormalize`
- **P2.5** Reward weight design per §6, including the §6.3 decision
- **P2.6** `MaskablePPO` training script, `MaskableEvalCallback`, mask-aware
  `evaluate_policy`, callbacks, tensorboard, the reward-breakdown logger
- **P2.7** Greedy baseline evaluation harness and the KPI comparison table

For P2.7, the KPI set: served count, mean wait, mean in-vehicle detour, decline rate,
post-match cancellation rate, no-show rate, total fleet VKT. **The RL result is meaningless
without the greedy baseline on the same held-out days.**

---

## 9. Known traps

1. **`create_or_empty_dir` deletes results.** Two workers sharing a `scenario_name` silently
   erase each other. Covered by P1.9.
2. **Global RNG.** `random.seed` / `np.random.seed` in `__init__` are process-global. This is
   why `DummyVecEnv` is forbidden.
3. **Logging handler accumulation.** Covered by P1.2.
4. **`record_boarding` does not call `record_user`.** Pickup needs its own callback. This is
   the single most likely thing to get wrong in P1.7.
5. **`user_request` returns `None`.** The offer comes from `get_current_offer(rid)`. Do not
   change the return type.
6. **Six paths reach `record_user`, one of them success.** Operator rejection, no-candidate
   rejection and rider decline all arrive via `FleetSimulationBase._rid_chooses_offer`
   calling `demand.record_user` directly — not via `Demand.user_cancels_request`, which is
   dead code (§2.4). Successful alighting arrives there too. §6.4 gives the classification
   order; there is no fallback to rider decline.
7. **`self._started` in `FleetSimulationBase.run()`.** Each episode needs a fresh simulation
   object. Never re-run an existing one.
8. **`op_max_wait_time` is 1500 in the reference scenario, not 4200.** The two constant
   config files in `studies/example_study/scenarios/` disagree on this and on fleet size,
   end time, and detour factor. Several normalizers depend on it. Read every such value from
   `scenario_parameters`; never hardcode one.
9. **Reservation requests bypass the decision point.** Count them, or P1.10's step-count
   sanity check will fail confusingly.
10. **Do not re-sort the candidate list.** It arrives sorted by `delta_cfv` from
    `insertion.py` line ~569. Re-sorting with any tie-break changes which vehicle wins on
    tied costs and breaks P1.3, P1.4 and P1.10.
11. **Generator finalisation.** `generator.close()` raises `GeneratorExit` inside the
    simulation. Make sure the `finally` block is safe when the episode did not run to
    completion.

---

## 10. Out of scope

Do not implement, do not add config keys or spaces for. Do keep the code shaped so a second
decision point can be added without touching `SDPDPAssignmentEnv`.

- Repositioning or rebalancing as an action
- Rerouting or reassignment of already-committed plans
- Dynamic pricing or fare as an action
- Multi-operator or broker settings
- Multi-agent formulations
- Charging and depot decisions
- Any modification to the diffusion cancellation, no-show, or stochastic travel time models
- **Anything touching service / PUDO durations.** The empirical per-request mechanism in
  `TravelerModels.py`, `PlanRequest.py`, and `insertion.py` is already the intended model of
  service duration uncertainty. Do not add stochastic draws, do not add globals or config
  keys for them, do not change `insertion_with_heterogenous_PUDO_duration`, and do not raise
  it as a suggestion. This is settled.
- **Add no other reward term beyond what is mentioned in §6.1**. Specifically: no fleet-distance 
  or operating-cost term, no in-vehicle
  detour term at alighting, no planned-ride-time term charged at assignment or pickup, and
  no substitute for any of these under a different name. In-vehicle time and fleet
  kilometres are deliberately unpriced. Do not add them, do not make them
  config-switchable, and do not propose them.
