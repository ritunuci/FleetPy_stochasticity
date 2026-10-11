# SDPDP Gym Environment — Implementation Spec v2

Repository: `FleetPy_stochasticity`, branch: `fleetpy_stochasticity_RL` 
(this has been created by branching out from `no_show_model_update` branch and retains everything from that branch)
Environment: `fleetpy_rl` (conda), captured in `env_fleetpy_rl.yml` (in the branch: `fleetpy_stochasticity_RL`). The other environment
files predate the RL work, do not use them for this project.
Fleet control: `PoolingIRSOnly` → `src/fleetctrl/PoolingIRSOnly.py::PoolingInsertionHeuristicOnly`
Simulation env: `ImmediateDecisionsSimulation`
Simulation config files are: `constant_config_depot_cali_sc_1.csv` and `example_depot_cali_sc_1.csv`
`baseline_user_stats.csv` lives in `docs` directory at the root.
RL library: Stable-Baselines3 + `sb3-contrib` (MaskablePPO)
Companion reaserch proposal document: `SDVRP_project_proposal.pdf` — the research design this implements

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
- Every edit to a **source-code** file outside `src/rl_gym/` and `dev/` carries a `# RL-GYM:`
  comment. Data files under `data/` (CSV, JSON, `crs.info`) and documentation (`docs/`, any
  `*.md`) are exempt: a comment would break their readers or adds nothing. Record such changes in
  the commit message and the progress log instead.
- Prefer adding a parameter with a safe default over changing an existing default.
- **Do not install, upgrade, or remove any package.** If an import fails, stop and report
  which package is missing. Ritun manages the environment — a dependency resolution that
  moves `numpy`, `scipy` or `torch` would change floating-point results and invalidate
  `baseline_user_stats.csv`.
- If you run into any confusion or have any questions, pause and ask Ritun, do not assume something.
  Ritun will tell you what to do and then you can continue.

**Verify, don't trust — this document included.**

This spec was written from reading the code and the data, by people and by an AI, across many
sessions. It can be wrong, out of date, or inconsistent with itself; a review before Phase 2
found and corrected about twenty such problems, and more may remain. Treat it accordingly:

- Every factual claim here — what a function does, line numbers, column contents, file names,
  counts, the figures a verification expects — is a hypothesis until you have checked it.
  `VERIFY` marks the claims already known to be load-bearing; hold every other claim that your
  work item or its verification depends on to the same standard. Line numbers are approximate:
  find code by symbol, not by line.
- Before relying on such a claim, check it independently: read the source, inspect the data, or
  run a short experiment. Prefer evidence you produced in this session over anything written
  here, including in the progress log.
- If something does not hold — a claim is false; two sections disagree; a verification cannot
  pass as written, or could pass while the thing it guards is broken; a Files list omits a file
  the item clearly needs; or following an instruction would break a gate or a DECIDED choice —
  **stop and report before writing code that depends on it.** Say what the spec says (section
  and line), what you found (file and line, or command output), why it matters, and the
  correction you propose. Do not silently choose an interpretation, and do not work around it.
- Skepticism is not licence to redesign. DECIDED choices stay decided: propose a change, never
  substitute one.
- Record each such check in the DONE entry next to the `VERIFY` findings — the ones that
  confirmed the spec as well as the ones that did not. A silent match cannot be told apart from
  a skipped check.

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
  branch, or an empty candidate list — do not generate a step. There are three such cases,
  not four: `PoolingInsertionHeuristicOnly.user_request` has no duplicate-rid branch, it
  simply overwrites `rq_dict[rid_struct]`. See P1.3.
- **A fourth kind of request never reaches the operator at all, so it is not one of those
  cases.** `load_demand_file` removes any demand row whose `start` node equals its `end` node
  before the simulation begins (`demand.py` ~105, the second of its two filters). Such a row
  never enters `future_requests`, never reaches `build_assignment_context` and never calls
  `record_user`, so it generates no step **and appears in no reconciliation bucket**. Measured
  under P2.4: 7 of the 127 kept days carry exactly one such row, and the operator-resolved
  origin-equals-destination bucket is **0 on all 127 days, those 7 included**. The
  `o_pos == d_pos` branch in `user_request` can therefore only fire if a `PlanRequest` built
  from *distinct* origin and destination nodes resolves to a single position — untested, so
  keep the branch and keep the bucket, but do not expect either to fire.
- One episode = one full simulation day (defined by simulation start and end time).

**Phase 1 — plumbing.** A correct, valid, single-threaded Gym environment with a minimal
observation and a counting-only reward. No learning. Exit gate: a scripted greedy policy
driven through the Gym API reproduces the stock greedy simulation byte for byte.

**Phase 2 — learning machinery.** The full observation (§5.2), rolling statistics, the frozen
corpus with multi-day rotation, `VecNormalize`, MaskablePPO, and the stock-greedy evaluation
harness. Exit gate: a short test training run that exercises all of it end to end (P2.15).
Performance is not judged in Phase 2; the weight sweep, repositioning and full-length training
come after it (§8).

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

**This already satisfies the research proposal's §3.1 requirement** that the insertion heuristic
re-runs after each assignment so an updated vehicle plan is visible for the next
same-timestamp request. No change needed. Verified under P1.4: the loop at
`ImmediateDecisionsSimulation.py` line ~91 processes each request completely — `user_request`,
`get_current_offer`, `receive_offer`, `_rid_chooses_offer` — before starting the next.

Note that step 3 iterates undecided travelers *plus* new arrivals. **`get_undecided_travelers`
is never non-empty in this scenario**, verified under P1.4 over a full day: empty at all 4,480
steps, and all 445 requests reached `user_request` exactly once. No rid produces a second
decision epoch, so §1's "one gym step = one actionable request" holds with no re-request case
to handle.

**That guarantee is config-dependent, not structural.** `get_new_travelers` adds every arrival
to `undecided_rq` (`demand.py` ~line 216) and `_rid_chooses_offer` removes it in all three of
its branches (`FleetSimulationBase.py` ~lines 759, 784, 797). An entry survives only through
line ~764 — `choose_offer` returned `None` *and* the rider did not leave the system. What keeps
that path unreachable is `ImmediateDecisionsSimulation.check_sim_env_spec_inputs`, which raises
unless `user_max_decision_time` (`G_AR_MAX_DEC_T`) is 0. A non-zero value would let riders stay
undecided across time steps and silently produce two decision epochs for one rid. If that
parameter ever changes, this analysis must be redone.

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

`user_cancels_request` (demand.py ~line 355) also calls `record_user`, but is dead code —
nothing invokes it. Do not instrument it. Every reference in the codebase is to
`operator.user_cancels_request` on `FleetControlBase` / `PoolingIRSOnly`, which is a
different method and never touches `Demand`. Note it is defined on **`SlaveDemand`, not
`Demand`** (it sits inside `class SlaveDemand(Demand)`), so it is dead twice over: nothing calls
it, and the class it belongs to is never instantiated here.

Because reward is pickup-anchored, **two callbacks are needed**, not one: `record_boarding`
for pickup and `record_user` for every terminal outcome.

`Demand` is the class to instrument, verified under P1.7: `_load_demand_module` takes the
`SlaveDemand` branch only when `sim_env == "MobiTopp"`, and this project's `sim_env` is
`RLImmediateDecisionsSimulation`.

Two facts about where the hooks land:

- `record_user` is defined **only** on `Demand`, so one callback there covers both classes.
- `record_boarding` **is** overridden by `SlaveDemand` (demand.py ~line 363). Instrumenting
  `Demand.record_boarding` therefore does **not** cover that override. Correct by scope here,
  since `SlaveDemand` is never instantiated — but it is a silent gap for anyone who later runs
  MobiTopp, who would get exit events and no pickup events at all.

Callers live in `FleetSimulationBase.update_sim_state_fleets` around lines 647
(`record_boarding`), 654 (`record_alighting_start`), 669 (`record_no_show`).

### 2.5 Stochasticity present in the simulator

| Uncertainty | Implemented? | Where |
|---|---|---|
| Travel time | Yes | `NetworkBasic`, `stochastic_tt = TRUE` |
| Post-match booking cancellation | Yes | `FleetSimulationBase._check_request_cancellations_diffusion_model` |
| Rider no-show | Yes | `BasicRequest_no_show`, `PoolingIRSOnly.sim_veh_no_show_requests_cleanup` |
| **Service duration at stops** | **Yes — empirically, already implemented** | Per-request boarding/alighting durations come from the demand file columns `actual_pudo_boarding_duration` and `pudo_alighting_duration`, read in `TravelerModels.py` (~line 561) into `duration_pudo_boarding` / `duration_pudo_alighting` and carried on `PlanRequest` as `real_duration_boarding` / `real_duration_alighting`. With `insertion_with_heterogenous_PUDO_duration = FALSE`, the **planner** assumes the constant `op_const_boarding_time = 300.0` while the **realized** duration comes from the data. That planner-versus-reality gap *is* the service duration uncertainty. Leave it exactly as it is. |

Plus rider decline on offer (`stochastic_rider_service_decline`) and dynamic fleet size
(`TimeBasedFS` + `op_act_fs_file`).

**Do not disable, seed-freeze, or short-circuit any of these.** They are the point of the
formulation.

**Known defect, deliberately left in place through Phase 2.** `NetworkBasic._set_stoch_edge_tt`
writes each realized travel-time draw into the edge object (`edge_obj.set_tt`). `Router.py`'s
Dijkstra reads edge objects, so operator routing computed after a redraw uses the realized draw
instead of `operator_exp_tt`, and `NetworkBasicWithStore` caches that result in
`travel_time_infos` until the next operator file load (07:00, 09:00, 12:00, 16:00, 18:00).
Ritun will correct this after P2.15 and regenerate the baselines (§8, After Phase 2). Until then
**do not fix it** (§10): every gate depends on the current behaviour. It is also why observation
code must leave the routing cache exactly as it found it (§5.2, trap 14).

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

The research proposal specifies a 5-second time step, which would double simulation steps to ~9,000
per episode and roughly double wall-clock, while leaving the gym step count unchanged.
**Confirm with Ritun before changing `time_step`** — it is pure overhead for the RL loop.

`example_depot_cali_sc_1.csv` contains 10 rows, of which only the `random_seed = 42` row is
active — the rest are disabled by a `#` prefix on `op_module`. `baseline_user_stats.csv` 
was generated from that row; the run's own output directory is
`studies/example_study/results/example_depot_time_pool_irsonly_rs_42/`.

### 2.9 The demand corpus — frozen

Training and evaluation days live in three sibling directories under
`data/demand/example_demand/matched/example_network/`:

| Directory | Days | Role |
|---|---|---|
| `corpus/` | 101 | training |
| `corpus_eval/` | 26 | held-out evaluation |
| `corpus_excluded/` | 5 | kept on record, never read |

That is 132 demand days, 2 Jan – 31 Oct 2024, exactly 44 each of Tuesday, Wednesday and
Thursday. `rq_file` is the date, e.g. `2024-06-20.csv`. Weekday and month for §5.2's request
block are derived from the filename — the demand rows carry `rq_time` as seconds from midnight
and no date column, so **the filename is the only record of which day a file is**. Every file
has the reference column set, unique request ids and minute-aligned `rq_time` — the ids checked
under P2.4, which found zero duplicate `request_id` in any of the 127 kept files.

**`load_demand_file` applies two filters, and request counts anywhere in Phase 2 use the count
after both** (`demand.py` ~103-105):

1. `rq_time` in `[start_time, end_time)`. Four requests fall before `start_time` — one each on
   02-29 and 03-05, two on 03-06 — and are dropped.
2. `start` node ≠ `end` node. **One row is dropped on each of 7 days**: 2024-02-15, 2024-02-20,
   2024-03-05, 2024-05-23, 2024-05-30, 2024-06-12 and 2024-06-13, measured under P2.4. These
   rows never reach the operator at all (§1), so they appear in no reconciliation bucket.

2024-03-05 loses one row to each filter: 426 rows in the file, 425 after the window, **424
loaded**. Modelling only the first filter is what made P2.4's first pass report a one-request
shortfall on exactly those 7 days.

**Excluded — calendar-explained days only:** 2024-01-02, 2024-01-03, 2024-01-04 (the week
after New Year; 237–264 requests), 2024-06-19 (Juneteenth; 295), 2024-07-04 (Independence
Day; 216). No day is excluded on volume alone.

**Demand level shift.** From the week of 24 Sep the daily level rises: 24–26 Sep run 551–579
and all 15 October days run 507–600 (median 556), against about 450 for January to
mid-September. Within October no day stands out (robust z −2.2 to +2.0 against October's own
median and MAD), and 24–26 Sep sit inside October's range, so this is a level change, not a set
of outliers. Those days stay in. Jun 4–6 (553–634) and the single-day volume outliers (Apr 9,
Feb 22, Apr 16, Aug 8) also stay in: none has a calendar explanation, and in both high
clusters the uplift is spread over the whole day and the core H3 cells, with no new cells,
unchanged passengers per request, and no dominant origin–destination pair.

**Split rule.** Per month, `n = round(0.2 × kept days)` days are held out (Python `round`; here
`n` is always 2 or 3). One generator, `rng = numpy.random.default_rng(20260929)`, is used
throughout, months in calendar order, with a running count of held-out days per weekday that
starts at 0:

1. If `n = 3`, the weekdays are taken in the order Tue, Wed, Thu, with no draw.
2. If `n = 2`, draw `keys = rng.random(3)` — one key each for Tue, Wed, Thu, in that order —
   sort the three weekdays ascending by (running count, key), and take the first two, in that
   order.
3. For each chosen weekday, in that order, the day is `pool[rng.integers(len(pool))]`, where
   `pool` is that month's kept days of that weekday, sorted; then add 1 to its running count.

This reproduces the list below exactly (checked). Result: 26 evaluation days (Tue 8, Wed 9,
Thu 9) and 101 training days (Tue 35, Wed 33, Thu 33); every month appears in both splits, with
9–12 training days each.

**Evaluation days (26):**
2024-01-11, 2024-01-30, 2024-02-07, 2024-02-08, 2024-02-27, 2024-03-19, 2024-03-20,
2024-04-09, 2024-04-11, 2024-04-17, 2024-05-07, 2024-05-15, 2024-05-23, 2024-06-12,
2024-06-20, 2024-07-09, 2024-07-25, 2024-07-31, 2024-08-13, 2024-08-15, 2024-08-28,
2024-09-04, 2024-09-12, 2024-10-01, 2024-10-09, 2024-10-24

The training days are the remaining 101. Files are **moved, not copied** — no day can sit in
two splits.

**The reference scenario is not part of this.** `thursday/2024-06-20.csv` stays exactly where
it is, and `constant_config_depot_cali_sc_1.csv` keeps `day_dir_name = thursday`. The
byte-for-byte gates compare against a baseline generated from that path, and nothing about the
corpus may be able to affect them. 2024-06-20 happens to fall in the evaluation split, so its
corpus copy lives in `corpus_eval/`. The two files have identical content but different line
endings, so their checksums differ — compare them with pandas, not md5.

`thursday/` also holds **`2024-01-18.csv`**, which is **byte-identical** to its corpus copy
(same md5, 426 rows) rather than a line-ending twin. No scenario reads it —
`constant_config_depot_cali_sc_1.csv` names `rq_file = 2024-06-20.csv` — and 2024-01-18 is not
in the evaluation split, so it stays a **training** day in `corpus/`. Recorded so that a later
reader checking "no day sits in two places" does not mistake it for a split error: the three
corpus directories are disjoint, and `thursday/` sits outside the split system by design.

**List demand days by `*.csv` only.** Every reader of the corpus directories enumerates days by
the `.csv` extension and ignores every other entry. macOS creates and rewrites `.DS_Store` in
these directories on its own — `corpus/` holds one now — and it is untracked, so it can appear,
change or vanish between runs without any demand file changing. A reader that enumerates
directory entries instead would see a phantom day, and in P2.6's manifest a `.DS_Store` would
put a volatile hash into a frozen artifact and make every later corpus check fail for no real
reason. This binds **P2.3's day rotation, P2.4's sweep, P2.6's manifest and P2.9's corpus
check**; P2.9 additionally carries a test that a stray `.DS_Store` does not make construction
raise. Do not delete these files to work around it — they come back.

**Frozen.** Adding, removing or moving a demand file after P2.2 shifts the arrival baseline
(P2.6) and silently invalidates anything trained against it. P2.6's manifest records a hash of
every training file's name and contents, and from P2.9 on the env refuses to start when
`corpus/` no longer matches (trap 16).

### 2.10 Network coordinate reference — corrected

`data/networks/example_network/base/crs.info` said `EPSG:32632` (UTM zone 32N, central
Europe). The network is in **`EPSG:32610`** (UTM zone 10N). The evidence:

- `base/nodes.csv` carries `lat` and `lon` columns alongside `pos_x`/`pos_y`. Converting all
  4,084 `pos_x`/`pos_y` pairs from EPSG:32610 reproduces those columns to within 4.5e-9° of
  latitude and 6.2e-10° of longitude.
- Read as EPSG:32632, the same coordinates land at longitude 9.89°–10.08°, exactly 132° (22 UTM
  zones) east of the network. Latitudes survive the error (37.29°–37.41°), which is why it went
  unnoticed.

Who reads the file: `NetworkBasic.__init__` reads it into `self.crs` at every construction
(`NetworkBasic.py` ~194). Only `return_network_bounding_box` and `return_positions_lon_lat` use
`self.crs`, and only the realtime plot calls them (`FleetSimulationBase` ~1093, ~1126), which is
off in the reference scenario (`realtime_plot` unset). Outside a simulation run:
`ReplayFromResult.py` (post-hoc replay, which draws the network in Germany until the fix),
`NetworkTTMatrix` (not this scenario's network type), and a zone preprocessing script
(`create_rectangle_zones.py` ~161). The correction is therefore expected to have no runtime
effect; P2.1 makes the byte gate the arbiter. `base - Copy/crs.info` carries the same wrong
value; nothing loads that directory and it is left alone.

Two additions from the re-search P2.1 requires, neither of which changes the above:

- **That one read reaches every registered routing engine but `NetworkTTMatrix`**, which has its
  own. `NetworkBasicWithStore` (the reference scenario's), `NetworkPartialPreprocessed` and
  `NetworkBasicCpp` inherit from `NetworkBasic` directly; `NetworkImmediatePreproc`,
  `NetworkBasicWithStoreCpp` and `NetworkPartialPreprocessedCpp` transitively. So the corrected
  value reaches whichever engine a scenario selects.
- **Three further readers, all unregistered dead copies**: `NetworkBasic_backup.py` ~193,
  `Backup_files_by_Ritun/NetworkBasic.py` ~189, and
  `Backup_files_by_Ritun/NetworkBasic_with_sto_netdyn_update.py` ~190. Each reads `crs.info` into
  `self.crs` and carries the same two `Transformer.from_proj` users. None appears in
  `get_src_routing_engines()` (`src/misc/init_modules.py` ~39–54), so none is reachable from a
  config; they would inherit the corrected value if ever run, which is the right outcome. Do not
  edit them.

Phase 2's position features read `lat`/`lon` from `nodes.csv` (§5.2), not those helpers —
`return_positions_lon_lat` builds a new pyproj transformer on every call. Directionality works
in `pos_x`/`pos_y` metres through `return_position_coordinates` and is unaffected either way.

---

## 3. DECIDED design choices

### D1. Generator-based control inversion. No threads.

The gym owns the time loop. `RLImmediateDecisionsSimulation` exposes a generator that yields
at each decision epoch and receives the action via `.send()`.

The research proposal's design — decisions only at request arrival, requests processed FIFO with the wall clock
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

Slot `k` for `k in [0, K)` = "offer the insertion plan for the candidate at index `k` in the
ordering below". Slot `K` = "reject". Reject is always legal.

**The ordering, stated exactly:** ascending by `delta_cfv`, so element 0 holds the **minimum**
— the most negative value — and each later slot is larger. "Cheapest" is avoided deliberately:
`delta_cfv` is a *change* in the control objective, and on this scenario it is negative
throughout (§5.1), because the objective penalises unserved requests and serving one improves
it. **A more negative `delta_cfv` is a larger improvement to the objective**, so slot 0 is the
best insertion and slot `K-1` the worst of those kept. Stock's
`min(list_tuples, key=lambda x: x[2])` selects that same minimum, which is why slot 0 is the
greedy choice.

- `K` is a config parameter. **`K = 8`, confirmed against measured data.** The candidate-list
  length distribution over the 436 decisions on the reference day is min 1, median 5, mean
  4.60, max 11, with 13 lists longer than 8. Those 13 lose only their tail — the largest
  `delta_cfv` values, i.e. the weakest insertions — which a policy would not select. `K` is
  owned by `SDPDPAssignmentEnv` alone. 
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

### D7. Per-episode reseeding is mandatory for training, and is switched on by `base_seed`.

`reset(seed=...)` draws an episode seed and writes it to
`scenario_parameters[G_RANDOM_SEED]` before the simulation is constructed. Per research
proposal §3.1.2, each day is replayed under several seeds and the agent must see demand
variation over the months that Ritun will finally decide to work on. Without this every episode
replays one sample path.

**Reseeding happens only when `base_seed` is set in the env config. With `base_seed` absent,
the scenario row's own `random_seed` is used unchanged and nothing is overwritten.** That is
not a weakening of D7; it is what makes D7 compatible with the rest of Phase 1. The
byte-for-byte gates in P1.3, P1.4 and P1.10 compare against `baseline_user_stats.csv`, which
was generated at `random_seed = 42`. P1.10 drives *this env* and requires the output to match
byte for byte, so if `reset()` always overwrote the seed, that gate would fail on a different
sample path — travel times, declines and no-shows all differ — for a reason having nothing to
do with the plumbing it exists to test. **Do not "fix" this back to unconditional reseeding:**
doing so breaks the Phase 1 exit gate and every byte comparison behind it.

`reset(seed=...)` still seeds the env's own `np.random.Generator` in both modes, so the
Gymnasium contract holds either way. Per-worker streams come from
`np.random.SeedSequence([base_seed, env_id])` — numpy's intended idiom for independent streams,
which also sidesteps any question about adjacent seeds correlating.

**The default that is safe for reproducibility is unsafe for training.** If `base_seed` is
absent during a real training run, every episode silently replays one sample path and nothing
errors — the agent would overfit a single day's realisation while the logs looked healthy.
Two guards: `train_sdpdp.py` (P1.11) must **raise** if `base_seed` is missing rather than
defaulting, and the env records the `G_RANDOM_SEED` actually used for the episode in
`episode_summary()`, so the seed is visible in training logs rather than inferred.

### D8. `SubprocVecEnv`, never `DummyVecEnv`.

`FleetSimulationBase.__init__` reconfigures process-global logging and global RNG state.
One env per OS process.

### D9. The greedy baseline for evaluation is stock `PoolingIRSOnly`.

P2.14 runs unmodified FleetPy — no gym, no observation, no normalization — on the same
(day, seed) pairs as the RL evaluation. P1.10 proved that env-driven greedy reproduces stock
greedy byte for byte, so stock is the same policy with nothing in between that could differ.
`greedy_action` in `policies.py` stays, but only to drive the P1.10 gate and the P2.4 corpus
sweep. It is not the evaluation baseline.

### D10. Training days rotate in global shuffled passes.

Within each pass, every training day is assigned exactly once. Pass `p` uses one permutation of
the sorted training-day list, drawn from `np.random.SeedSequence([base_seed, 1010, p + 1])` —
identical in every worker. The constant `1010` tags the stream, and `p + 1` keeps the last entry
non-zero: `SeedSequence` pads with zeros, so `[base_seed, 1010, 0]` would equal
`[base_seed, 1010]`. Together they keep it from ever coinciding with a worker's episode-seed
stream `SeedSequence([base_seed, env_id])` (D7, P1.9). Worker `env_id` of
`n_workers` takes the days at positions `env_id, env_id + n_workers, …` of that permutation, then
moves on to pass `p + 1`.

Across a whole run, visit counts stay close but are not exactly ±1. With 101 days and 4 workers,
worker 0 owns 26 positions per pass and the others 25, and SB3 advances workers by steps rather
than episodes, so workers move through passes at slightly different rates; over about 70 passes,
counts differ by a few visits. The day-visit histogram (P2.13) reports it. Episode seeds stay
independent per D7. Drawing a day uniformly at random per episode is not used: it balances only
on average, and over a short run can skip days entirely.

---

## 4. Relationship to the project proposal

### Implemented as specified

Decision epochs are per actionable request. In practice a request with no available choice — origin equals
destination, reservation branch, or no feasible candidate — is resolved inside
`build_assignment_context` and generates no gym step, because there is nothing for the policy
to decide. One episode per day; days replayed across seeds and
across the frozen 132-day demand corpus (§2.9);
`K+1` action space with reject always available; no immediate positive reward for acceptance; 
delayed credit; explicit penalty for post-match
cancellation; FIFO handling of same-timestamp arrivals with the heuristic re-running between
them (already native to `ImmediateDecisionsSimulation`).

### Phase 2 and after

Phase 2 implements the full observation — H3-zoned rolling statistics, the arrival anomaly, the
full candidate and global feature sets (§5.2) — plus the multi-day training machinery and a
short test training run (§8). The reward weight sweep and full-length training follow Phase 2.

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
   **Decided:** `w_noshow = 0.0` (§6.1). A no-show rider forfeits the pickup reward and nothing
   more.

---

## 5. Observation

### 5.1 Phase 1 — minimal

Length `3 * K + 4`. For `K = 8` that is **28**. `float32`.

**The space is `Box(low=-inf, high=inf, shape=(3*K + 4,), dtype=np.float32)`.** Unbounded is the
honest declaration: `delta_cfv` has no a priori bound even after the `sign * log1p` transform,
and inventing one would either clip real values or be a lie the checker cannot catch. Bounding
happens in Phase 2 where it belongs, via `VecNormalize(clip_obs=10.0)` (§5.3).

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
4. fraction of active vehicles currently idle

Features 1–2 and feature 3 are **different quantities**. `t_of_day` is absolute clock time,
`(sim_time mod 86400) / 86400`, so the `sin`/`cos` pair is a circular encoding of where the
day sits on a 24-hour clock. Feature 3 is episode progress,
`(sim_time - start_time) / (end_time - start_time)`, which is the horizon signal D6 relies on.
On the reference day the episode covers 07:00–19:26, so the two differ throughout. The circular
encoding is convention here — this day never wraps midnight, so `sin`/`cos` only earn their
keep if a full 24-hour day is ever run.

Feature 4 is the `IDLE` count divided by the count of vehicles whose status is not
`OUT_OF_SERVICE`. `IDLE` means `VRL_STATES.IDLE` only — not `WAITING`, `PLANNED_STOP` or
`REPO_TARGET`. Guard the divide: if no vehicle is active, emit 0.

`OUT_OF_SERVICE` is the operative definition of inactive because **`simple_insert`
(`insertion.py` line ~36) returns early on it**: `if veh_obj.status == VRL_STATES.OUT_OF_SERVICE:
return`. An out-of-service vehicle therefore yields no insertion and never becomes a candidate.
That is the live filter.

**Three other comparisons that look like this filter are inert and have never fired:**
`searchVehicles.py` line ~21, `searchVehicles.py` line ~90, and `insertion.py` line ~414 all
test `veh_obj.status == 5`. `VRL_STATES` is a plain `Enum` whose members hold `(int, str)`
tuples and which defines no `__eq__` against ints, so `VRL_STATES.OUT_OF_SERVICE == 5` is
unconditionally `False`. `veh_obj.status` is always a `VRL_STATES` member
(`Vehicles.py` line ~68). **This is upstream FleetPy, not this fork — do not read it as a
defect introduced here, and do not "fix" it.** Changing those lines would start excluding
vehicles that are currently included and would break every byte-for-byte comparison.

Feature 4 as defined is a coarse fleet-utilisation signal, not a true availability count:
`simple_insert` also returns early on `veh_obj.no_show_event`, so the set of vehicles that could
actually serve the request is strictly narrower, and the two notions diverge exactly while a
no-show is being waited out. That is accepted for Phase 1; see §5.2.

**Padding.** Every feature of an unoccupied candidate slot is `0.0`. The justification is
`VecNormalize`: a large sentinel would distort its per-feature running mean and variance in
Phase 2 (§5.3), and 0.0 is neutral there. That is the whole reason.

It is *not* justified by what a naive `argmin` would pick. Measured under P1.6, real
`delta_cfv` on this scenario is **negative** — transformed range [−14.508, 0.000] over 436
decisions, where the 0.000 end is padding and the raw minimum is about −2.0e6. So padding at
0.0 is the **largest** value in the slot, not the smallest, and an unrestricted `argmin` would
still land on slot 0 here. See P1.10 for why the `is_valid` restriction is nonetheless required.

All wait-related features divide by `op_max_wait_time` read from `scenario_parameters`
(1500 in the reference scenario), never a literal. Phase 2 routes this through
`wait_normalizer_s` (§5.3), whose fallback when absent is exactly `op_max_wait_time`, so these
values do not change.

Feature 2 is required for the Phase 1 exit gate — the scripted greedy driver picks
`argmin` over the `delta_cfv` slots, so it must be readable from the observation.
Do NOT add an implied-decline-probability feature, in either phase. It is derivable
from feature 3 and `scenario_parameters`, and Ritun has excluded it from the state
design. Do not propose it.

### 5.2 Phase 2 — full (DECIDED)

Per proposal §3.1.1–§3.1.3, as settled with Ritun. Same `Box`, same observer→dict→concat
mechanism, wider vector. For `K = 8` the length is **139**:

| Block | Features | Length at `K = 8` |
|---|---|---|
| Candidate | 12 per slot | 96 |
| Request | 24 | 24 |
| Global | 19 | 19 |

Block order in the vector is candidate, request, global. Inside the candidate block the layout
stays feature-major, as in Phase 1 — each feature is a length-`K` vector — and the three
Phase 1 features keep their positions, so `delta_cfv` is still at `obs[K:2K]`. **Nothing may
depend on that.** P2.8 publishes a layout map, and every consumer reads features by name.

#### Candidate block

Padding is 0.0 for every feature of an unoccupied slot (§5.1).

| # | Feature | Definition | Scaling |
|---|---|---|---|
| 1 | `is_valid` | unchanged from §5.1, from `spaces.candidate_slot_validity` (trap 12). Stays first. | — |
| 2 | `delta_cfv` | unchanged from §5.1. Kept for now; Ritun may drop it later (note below). | `sign·log1p` |
| 3 | `offered_wait` | unchanged from §5.1 | ÷ `wait_normalizer_s` |
| 4–5 | `veh_lat`, `veh_lon` | vehicle's current position | [−1, 1] over the node bbox |
| 6 | `free_seats` | `max_pax − Σ rq.nr_pax` over `veh_obj.pax` (passengers, not riders) | ÷ `max_pax` |
| 7 | `remaining_stops` | `len(list_plan_stops)` of the **pre-insertion** plan | raw count |
| 8 | `directionality` | cosine in [−1, 1], formula of `SelectRV.filter_directionality`, pre-insertion | — |
| 9 | `lateness_pre` | max over pending riders of planned pickup in the re-timed pre-insertion plan − quoted pickup | ÷ `wait_normalizer_s` |
| 10 | `lateness_post` | max over pending riders of planned pickup in plan `k` − quoted pickup | ÷ `wait_normalizer_s` |
| 11 | `added_distance` | route distance of plan `k` − route distance of the pre-insertion plan | km |
| 12 | `added_pickup_delay` | Σ over pending riders of planned pickup in plan `k` − planned pickup in the re-timed pre-insertion plan | ÷ `wait_normalizer_s` |

What the features mean together: 4–9 describe the vehicle as it is, independent of the
decision. 10–12 describe what inserting this request into plan `k` does. `lateness_pre` is how
late the vehicle already runs against its promises; `added_pickup_delay` is how much further
this insertion pushes the riders already waiting for it; `lateness_post` is the worst promise
after the insertion. Riders already onboard can no longer cancel and the reward carries no
ride-time term (§10), so their delay is deliberately not counted.

Definitions:

- **Pre-insertion plan** = `veh_plans[vid]`, the vehicle's committed plan without the new
  request. **Plan `k`** = the candidate plan from `insertion_with_heuristics`, which already
  contains the new rider (P1.6 reads its `pax_info` for `offered_wait`).
- **Re-timed copy.** Stored plan times refresh only at leg events — the vehicle finishes a leg,
  or starts boarding or alighting — because `update_sim_state_fleets` runs with
  `force_update_plan=False`. After a stochastic travel-time update they are stale until the
  vehicle's next leg event. Candidate plans do not have this problem: `insertion.py` copies the
  plan (~line 45) and re-times the copy with `update_tt_and_check_plan` at decision time.
  Features 9 and 12 do the same: `copy()` the pre-insertion plan, call
  `update_tt_and_check_plan(veh_obj, sim_time, routing_engine)` on the copy exactly as
  `insertion.py` does, read the times, discard the copy. One re-timed copy per candidate
  vehicle per decision, shared by both features. **Never re-time, reassign or otherwise touch
  `veh_plans[vid]` itself (trap 14).** VERIFY the call signature, and that re-timing a copy
  leaves the stored plan's stops, `pax_info` and utility unchanged. Two exceptions were found in
  review:
  - **(a) Shallow copy.** `PlanStop.copy()` copies `boarding_dict` shallowly, so the no-show
    branch of `update_tt_and_check_plan` (`VehiclePlan.py`, `boarding_dict[-1].remove(rid)`)
    can change the stored plan's list through the copy. Insertion already does exactly this for
    every candidate vehicle at the same decision, so the observer's call finds the rider already
    removed and changes nothing further. Report what you observe; the P1.10 gate decides.
  - **(b) Routing cache.** Re-timing, and `added_distance`, call
    `routing_engine.return_travel_costs_1to1`, which on `NetworkBasicWithStore` stores every new
    origin–destination result in `travel_time_infos`. Until the §2.5 defect is fixed, Dijkstra
    runs on the live random draw, so a cache entry the observer creates now changes what later
    planning reads, and the gate can fail. The candidate observer therefore records the cache's
    keys before computing its features and deletes every key it added afterwards, leaving the
    cache exactly as it found it. VERIFY the attribute name; if the network type has no such
    cache, there is nothing to restore. Once the defect is fixed this is unnecessary but
    harmless.
- **Pending riders** of a vehicle: riders with a boarding entry in the pre-insertion plan's
  stops who are not in `veh_obj.pax`. The new rider is never pending. VERIFY how a rider whose
  boarding is in progress appears, and record it.
- **Quoted pickup** of pending rider `i` = `prq_i.rq_time` + the offered wait the operator
  quoted in `_create_user_offer`, as stored on the operator side (`prq.offer`, set by
  `set_service_offered`). VERIFY the attribute and that the quote is not overwritten after
  acceptance. **Never read `rq_obj.expected_pickup_time`, `ETA_update_time_stamp`,
  `diffusion_state_*`, or any other diffusion-model attribute** — the operator has no knowledge
  of the behavioural model (proposal §3.1.2).
- **No pending riders** → 0.0 for features 9, 10 and 12.
- **Signs are kept.** Lateness is negative when a rider will be picked up ahead of the quote,
  which will be common while the planner assumes 300 s per stop against a realized mean of about
  56 s (§4 gap 1). `added_distance` and `added_pickup_delay` can be slightly negative because
  routes minimise time rather than distance, and stops can merge. Do not clip any of them.
- **`added_distance`**: remaining route distance from `veh_obj.pos` through the plan's stops,
  summing `routing_engine.return_travel_costs_1to1(last_pos, pos)[2]` and skipping consecutive
  equal positions — exactly the distance term of `distance_and_user_times_with_walk` in
  `objectives.py` — computed the same way for both plans. It is an observation, never a reward
  term (§10).
- **`remaining_stops`**: pre-insertion. With repositioning off, 0 means idle, and that is what
  separates idle vehicles from perpendicular ones in feature 8, so no companion flag is added.
  VERIFY whether the stop currently being served still counts. The "0 means idle" equivalence
  must be rechecked when repositioning goes on.
- **`directionality`**: re-implement the dot product of `SelectRV.filter_directionality`
  (lines ~15–38); do not call the function — the heuristic behind it is off in this scenario,
  and it returns a filtered vehicle list, not scores. Request vector: destination minus origin
  coordinates. Vehicle vector: coordinates of the destination of the last leg of
  `veh_obj.assigned_route` minus the vehicle's current coordinates. Both through
  `routing_engine.return_position_coordinates` (`pos_x`/`pos_y` metres — never lat/lon, which
  would skew every angle by cos 37.35° ≈ 0.79). Unit-normalise; a zero-length vector gives 0.
  `assigned_route` is the committed route, so this is pre-insertion by construction.
- **Vehicle position**: `veh_obj.pos` is `(node, None, None)` at a node or
  `(o_node, d_node, rel_pos)` on an edge; interpolate linearly between the two nodes' lat/lon by
  `rel_pos`. VERIFY that `rel_pos` is the fraction already travelled.
- **Why `delta_cfv` carries almost nothing.** The objective subtracts `LARGE_INT = 1,000,000`
  per rider in the plan (`objectives.py`), so raw `delta_cfv` is about −1,000,000 plus a few
  hundred cents of real cost. After `sign·log1p` the real cost moves only the sixth significant
  digit — one float32 step is about one cent, i.e. 40 m or 2.2 s of rider time — and Phase 1's
  raw minimum near −2.0e6 shows the offset occasionally doubles, which will set the slot's scale
  under `VecNormalize`. Features 11 and 12 carry the insertion-cost signal instead.

#### Request block

| Feature | Definition | Scaling |
|---|---|---|
| `o_lat`, `o_lon`, `d_lat`, `d_lon` | request origin and destination node | [−1, 1] over the node bbox |
| `n_pax` | `prq.nr_pax` | raw |
| `dow_*` (3) | one-hot over the weekday vocabulary (Tue, Wed, Thu) | — |
| `month_*` (10) | one-hot over the month vocabulary (Jan–Oct) | — |
| `zone_arrivals_30`, `zone_arrivals_60` | arrivals in the request's zone set | raw count |
| `zone_declines_30`, `zone_declines_60` | rider declines in the zone set | raw count |
| `zone_cancels_30`, `zone_cancels_60` | post-match cancellations in the zone set | raw count |

- Vocabularies come from P2.6's manifest, never literals. An unknown weekday or month raises.
- VERIFY the `PlanRequest` attributes for the origin and destination nodes.
- **Zone set** = the request's origin H3 cell (resolution 7) plus its neighbours from P2.5's
  table. Events are keyed to **the event rider's origin cell**.

#### Global block

| Feature | Definition |
|---|---|
| `time_sin`, `time_cos`, `day_progress`, `idle_fraction` | unchanged from §5.1 |
| `arrivals_30`, `arrivals_60` | fleet-wide arrivals in the window — also the anomaly's "observed" |
| `arrivals_expected_30`, `arrivals_expected_60` | P2.6 baseline mean for this month at this minute of day |
| `arrivals_z_30`, `arrivals_z_60` | `(observed − expected) / max(sd, 1.0)`, `sd` from P2.6 |
| `declines_30`, `declines_60` | fleet-wide rider declines |
| `cancels_30`, `cancels_60` | fleet-wide post-match cancellations |
| `wait_gap_30`, `wait_gap_60` | mean of (realized wait − quoted wait) over riders picked up in the window; 0.0 if none; ÷ `wait_normalizer_s` |
| `active_now`, `active_30`, `active_60` | scheduled active vehicles now, at +30 min and at +60 min, ÷ fleet size |

- The fleet-wide arrival count and the anomaly's observed count are the same number and appear
  once.
- **`wait_gap`**: realized wait = `pu_time − rq_time`; the quoted wait is the operator's quote as
  above. Mean rather than median: on the reference day under greedy, 17% of pickups land within
  1 s of their quote, so a 30-minute window's median sits within 1 s of zero in 26% of windows,
  while the mean responds to the late tail (6% of pickups arrive more than 5 min after their
  quote). One entry per distinct rid, following `on_pickup`'s last-write-wins (§6.4).
- **`active_*`**: computed from `op_act_fs_file` (`example_active_fleetsize.csv`), read once at
  env construction from the same path `TimeBasedFS.add_init` uses
  (`dir_names[G_DIR_FCTRL]/elastic_fleet_size/<file>`), with the identical formula
  `min(max(int(np.round(share × nr_vehicles)), 0), nr_vehicles)` from
  `TimeBasedFS._read_active_vehicle_file`. `TimeBasedFS` keeps no schedule that can be queried —
  the parsed table is local to that method, and the activation lists it leaves behind are
  consumed as they fire — so reading the file is the only route. The share in force at time `t`
  is the last row with time ≤ `t`, and the last row beyond the end. Read-only: nothing on the
  fleet-sizing module is touched. VERIFY the path and the formula. The fleet is
  `default_vehtype:12` and the schedule is the same on every day, so these are a deterministic
  function of clock time, included as an explicit signal.

#### Windows

All windows are trailing and half-open: an event counts when its time is in
`[sim_time − W, sim_time)`, `W ∈ {30, 60}` minutes. Arrivals are timed by `rq_time`, declines
and cancellations by the moment they fire, pickups by pickup time. Requests with the same
timestamp that were processed earlier in the same step are therefore not counted. P2.6's
offline baseline uses the identical definition, which is what keeps observed and expected
comparable.

#### Positions

**Position encoding** is a config key, `position_encoding ∈ {"latlon", "relative"}`, default
`"latlon"`. `"relative"` replaces a candidate's `veh_lat`, `veh_lon` with its offset from the
request origin, scaled by the bbox half-spans; request coordinates stay absolute. The length is
the same either way. The earlier `"h3"` option is removed: a raw cell id is not a usable
network input, and a one-hot would add 37 dimensions per position. H3 is the aggregation key
for rolling statistics only.

**Node coordinates** come from the `lat`/`lon` columns of `base/nodes.csv` (§2.10), loaded once
at env construction as static data. The bbox is the min/max over all nodes.

#### Not in the observation

Settled; do not add, and see §10: infeasible-plan fractions, a rejection-rate anomaly, a
separate availability fraction, plan-independent proximity (distance or ETA to pickup), slack
to a latest-pickup bound, H3 centroid coordinates, an idle companion flag, the ETA gap for
cancelled riders, an implied-decline-probability feature, the request's direct travel time.

### 5.3 Normalization

Two layers.

1. **Manual scaling**, as listed per feature in §5.2. Wait-type features (`offered_wait`,
   `lateness_pre`, `lateness_post`, `added_pickup_delay`, `wait_gap_*`) divide by
   **`wait_normalizer_s`**, an env config key. **Absent → `op_max_wait_time` from
   `scenario_parameters`** (1500 in the reference scenario), which reproduces Phase 1's
   `offered_wait` bit for bit. Set it explicitly once `op_max_wait_time` is relaxed, so that
   relaxing the operator's bound does not silently rescale every wait feature. Positions scale
   to [−1, 1] over the node bbox, so 0.0 padding sits mid-range. Distances are in km, counts are raw. 
   No other wait-time or scenario-dependent literal anywhere.
2. **`VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.999)`.** Its `gamma`
   must equal PPO's (§6.5) and come from the same config value: `VecNormalize` scales rewards
   by a running estimate of the discounted return, and its default of 0.99 would normalise
   against the wrong horizon. Save its statistics with every checkpoint and reload them with the
   model; at evaluation set `training = False` and `norm_reward = False`. Evaluating with
   mismatched statistics is a silent and very common failure.

The env raises if any observation value is non-finite (P2.12).

---

## 6. Reward

### 6.1 Structure

Reward events accumulate in a `RewardTracker` and are flushed each gym step. `step(action)` returns the sum of everything that fired after that action was applied, upto
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

**Phase 2 weights (DECIDED for the short test training, P2.15):**
`w_pickup=1.0, w_wait=0.05, w_reject=1.0, w_decline=0.1, w_cancel=0.6, w_noshow=0.0, w_horizon=0.0, w_no_candidates=0.0`.

`w_noshow = 0.0` is Ritun's decision on §4 gap 2: a no-show rider forfeits the pickup reward and
nothing more. The weights are revised by a sweep over `w_pickup`, `w_wait`, `w_reject`,
`w_cancel` and `w_decline` after Phase 2 and before full-length training (§8, After Phase 2).

How these weights shape rejection: with `w_reject = w_pickup`, an offer's own outcome rarely
scores below rejection's −1 while `w_decline < 1` — a pickup at the 1500 s cap scores −0.25, a
cancellation −0.6, a declined offer `−w_decline`. Rejection then pays mainly when an insertion
delays riders already assigned. Once `w_decline > 1`, an offer near 1500 s (decline probability
approaching 1) scores below rejection. The implicit ETA threshold of research question 2 is
therefore partly set by `w_decline / w_reject`, and the sweep should span values on both sides
of 1.

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
7. `rq.pu_time is None` and `rq.chosen_operator_id is not None` → unserved at horizon,
   `-w_horizon`
8. otherwise → unclassified; count it, charge nothing, and report the count in
   `episode_summary()`. A non-zero count here means a path was missed.

**There is no fallback to rider decline.** Anything unmatched is unclassified, not a decline.

**Branch 7 is where `-w_horizon` is charged — not in a separate end-of-episode sweep.** A rider
accepted but never picked up reaches `record_user` through `record_remaining_users` at end of
day, so it is already an `on_exit` arrival; charging it anywhere else would create a second
charging site and contradict the rule below that all reward is charged once, in `on_exit`. It
also keeps branch 8 meaning strictly "a path was missed" rather than doubling as a legitimate
outcome.

`chosen_operator_id` is initialized to `None` (`TravelerModels.py` ~line 112) and set only
inside `choose_offer` on acceptance, so "accepted but never picked up" is directly expressible.
Branches 4–6 run first, so everyone who accepted and then failed another way — diffusion
cancellation, decline, no-show — is already claimed before branch 7 is reached.

**Branch 7 is unreachable on the reference day, and that is expected, not evidence it is dead.**
Classifying all 445 baseline rows by this order gives served 288, rider declined 118, diffusion
cancelled 28, no candidates 9, no-show 2, unclassified 0.

The reason nobody is stranded is narrower than it looks, and was measured under P1.10 rather
than inferred: **the day simply finishes its work before `end_time`.** The last request arrives
at `rq_time = 68400`, the latest pickup is 69064.2 and the latest dropoff 69805.3, all below
`end_time = 70000`. Nothing is in flight when the main loop ends, so
`record_remaining_assignments` completes no trips and advances the simulation **zero** extra
seconds — it is not that the tail rescues in-flight riders, it is that there are none. The
margin is only about **195 simulated seconds**.

**The margin closing is not what makes the branch live, and the reference day's margin is
atypical.** Measured under greedy across all 127 kept days (P2.4): **79 of 127 days already
have a negative margin**, the median is **−234 s**, the range is −1918 s to +1104 s, and the
reference day's +195 s is late in the distribution — **96 of 127 days finish later than it**.
A negative margin only means `record_remaining_assignments` is completing trips after
`end_time`, which is **expected on busier days**: more requests arriving late leaves more trips
running at `end_time`, and the tail finishes them.

The branch becomes live only when **that tail's own cap binds**:
`if c_time > self.end_time + 2 * 7200: break` (`FleetSimulationBase.py` ~625 — written
`2 * 7200`, which is why grepping for `14400` does not find it), which breaks the loop with
vehicles still en route and strands whoever is left. **The indicator is therefore the longest
tail against the 14,400 s cap, not the margin.** Measured across the corpus: the longest tail
anywhere is **1,950 s on 2024-05-01, 13.5% of the cap**, leaving 12,450 s of headroom — the cap
is 7.4× the worst tail observed — the cap bound on **0 of 127 days**, and branch 7 fired on
**0 of 127 days**. At most 5 riders were ever in flight when a tail began.

The tail is a near-exact function of the margin where it matters: on every one of the 79
negative-margin days, `tail = ceil(−margin / 10) × 10 + 30` seconds, the 10 s being
`time_step` and the 30 s the loop's three closing iterations. It does **not** hold on the 48
non-negative-margin days, where the tail is 0 s (46 days) or 10 s (2 days) — a vehicle can
still hold an assigned route at `end_time` without a late dropoff. So the margin bounds the
tail in the long-tail regime and says nothing in the short one, which is why P2.13 logs the
tail itself.

Do not delete the branch for want of a firing, and do not treat `w_horizon = 0.0` as settled —
but what would unsettle it is a tail approaching 14,400 s, not a margin going negative.
*(Corrected in P2.4: an earlier draft said the branch "becomes live as soon as that margin
closes" and rested `w_horizon = 0.0` on the reference day's 195 s margin. Both were wrong —
the margin had already closed on most of the corpus with no effect on branch 7.)*

`rq.no_show` is set at load time from the demand file column (`TravelerModels.py:555`), so it
is a rider attribute, not an outcome. It must be checked *after* `diffusion_cancelled` and
must be gated on `pu_time is None`, or a no-show-flagged rider who cancelled while waiting
gets reported as a no-show.

`note_rejection` and `note_no_candidates` only **label** — they record the rid in a set and
charge nothing. All reward is charged once, in `on_exit`. `on_exit` must be idempotent per
rid: `record_alighting_start` and `record_remaining_users` can both fire for a rider who is
mid-alighting when the day ends.

**`on_pickup` is last-write-wins, not dedupe-on-first.** `record_boarding` can fire more than
once for the same rid. The cause is the no-show mechanism: when a no-show rider shares a
boarding stop with another rider on the same vehicle,
`sim_veh_no_show_requests_cleanup` calls `simple_remove_bulk`, and the replan re-fires the
boarding stop for the co-located rider. It therefore recurs whenever a no-show shares a
boarding stop, not only in the one observed case. On the reference day this is rid 50 — two
calls, vid 4, same `pu_pos`, at `sim_time = 33032.898...` then `33230`, with rid 45 the
no-show at the same stop, same vehicle, same origin/destination.

FleetPy overwrites `pu_time` on each call and the output file records the **last** value
(33230 for rid 50), so the reward must match that. Charging on the first call would price a
92.9 s wait against a realized 290 s and would bias the policy against precisely those riders
delayed by someone else's no-show.

`on_pickup` therefore stores the amount charged per rid and, on a repeat, charges the
**delta**, so the net is exactly one pickup priced at the final `pu_time`. Dedupe-on-first is
wrong for the same reason: it keeps the stale wait.

P1.7 must report the duplicate-boarding count in `episode_summary()` — the number of rids
whose `record_boarding` fired more than once — so it is visible whether this stays at one per
day.

### 6.5 Discounting

A pickup reward can land 10–30 decisions after the assignment that caused it. Set
**`gamma = 0.999`**, not SB3's default of 0.99. With ~440 steps per episode, 0.99 gives an
effective horizon of ~100 decisions, which is marginal for pickup credit.

### 6.6 Phase 1 behaviour — superseded by P2.0b

The tracker records and **counts** every event. Phase 1 delivered the wiring and the event
counts, not a tuned reward, and did no weight design; the economics are Phase 2.

**Phase 1 ran its own weights, not §6.1's.** `DEFAULT_REWARD_WEIGHTS` through P1.11 held
`w_pickup=1.0, w_wait=0.05, w_reject=0.5, w_decline=0.1, w_cancel=0.6, w_noshow=0.8,
w_horizon=0.0, w_no_candidates=0.0` — differing from §6.1 in two keys, `w_reject` (0.5 against
1.0) and `w_noshow` (0.8 against 0.0). Every Phase 1 reward figure is at those values:
`episode_reward = 116.510947` in `docs/RL_GYM_PHASE1_RESULTS.md` reproduces only with
`w_noshow = 0.8`; §6.1's weights give 118.110947 on the same day. **P2.0b replaced the defaults
with §6.1's Phase 2 weights**, so that figure is not reproducible from the current code and the
two must not be compared. An earlier draft of this section said Phase 1 used §6.1's weights as
given; it never did. *(Corrected in P2.0b.)*

No Phase 1 gate depended on the weights — they scale the reward signal and touch no decision,
which is why the byte-for-byte gates were blind to the difference.

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
Missing one raises `AttributeError` the first time the corresponding `get_src_*()` getter is
called — not at import. The `dev_content.add_*()` calls sit inside the getters, so a bare
import will not surface the problem; it will instead fail partway into a simulation.

**How Ritun verifies:** every one of the ten `get_src_*()` getters is called and returns
without error, still containing its stock entries. Importing the module is not sufficient —
the `add_*` calls live inside the getters. Confirm
`get_src_fleet_control_modules()["PoolingIRSOnly"]` and
`get_src_simulation_environments()["ImmediateDecisionsSimulation"]` still resolve and that
the two RL entries now appear. Then run the existing greedy scenario and match
`baseline_user_stats.csv`.

**Commit:** `RL-GYM: add dev extension package for module registration`

---

### P1.2 — Output, logging, and seed guards

**Files:** `src/misc/globals.py`, `src/FleetSimulationBase.py`, `src/demand/demand.py`,
`src/fleetctrl/FleetControlBase.py`

Two of the three write sites named below live outside `FleetSimulationBase`:
`Demand.save_user_stats` is in `src/demand/demand.py` and
`record_dynamic_fleetcontrol_output` is in `src/fleetctrl/FleetControlBase.py`. Note that
`src/demand/demand.py` is also P1.7's file — P1.2 touches it first, for a different reason.

Add `G_SKIP_OUTPUT = "skip_output"` and `G_RL_MODE = "rl_mode"`.

Guard behind `if not self.scenario_parameters.get(G_SKIP_OUTPUT, False):`
- `create_or_empty_dir(self.dir_names[G_DIR_OUTPUT])`
- `self.save_scenario_inputs()`
- in `run()`: `save_final_state()`, `evaluate()`
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
`FleetControlBase.record_dynamic_fleetcontrol_output` (~line 860). Guard the fourth write
that hangs off the last line of `record_dynamic_fleetcontrol_output` as well:
`self.repo.record_repo_stats()`, which writes via `RepositioningBase.record_repo_stats`
(~line 66). `self.repo` is `None` in the reference scenario, but "`skip_output = 1` produces
no files" must hold unconditionally, not only while `repo` happens to be unset.

`self.op_output[op_id]` is created at `FleetSimulationBase.py` ~line 394 as a list **shared
with every `SimulationVehicle`** — each vehicle is handed the same list object at
construction. Its clear must stay the in-place `.clear()`. Rebinding it to `[]` orphans the
vehicles' reference: they keep appending to the old list, the guarded buffer never fills,
and nothing raises. The other two clears (`user_stat_buffer`, `dyn_output_dict`) are
rebinds and may stay rebinds.

**Do NOT guard `record_remaining_assignments()` or `demand.record_remaining_users()`.**
Neither is an output operation despite its name.

`record_remaining_users` (`demand.py` ~line 275) is the end-of-day `record_user` sweep —
`for rid in list(self.rq_db.keys()): self.record_user(rid)` followed by one
`save_user_stats(force=True)`. That trailing call is its only write and is already covered by
the write-site guard above. Guarding the method itself would contradict the
"`record_user` must keep running with output off" rule below and §2.4, which lists it as a
`record_user` choke point; it would delete the §6.1 unserved-at-horizon events and every §6.4
terminal classification for riders still in `rq_db` at the end of the day. RL episodes would
then be scored differently from the P1.7 reconciliation run, which runs with output on — the
same failure the `record_remaining_assignments` argument below guards against.

As for `record_remaining_assignments()`: despite its name it is not an output
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

Those greps do not find every site that touches the filesystem. Two more, both real and both
guarded by this work item:

- `create_or_empty_dir(...)` (~line 188) reaches `os.makedirs` / `os.remove` / `os.rmdir`
  (~lines 92–104). It is destructive rather than a write, and matches none of the three greps.
- `logging.FileHandler(self.log_file)` (~lines 220 and 223) **creates `00_simulation.log` on
  disk**. Handler construction is a file-creating site; the logging guard below is what covers it.

`src/evaluation/standard.py` line ~226 and `src/evaluation/standard_with_peak.py` line ~230
both `pd.read_csv(os.path.join(output_dir, "2_vehicle_types.csv"))`. `evaluate()` is guarded by
the same flag as the write, so write and read are suppressed together and evaluation runs are
unaffected.

Logging: guard the handler setup so it runs once per process — **in RL mode only**.

```
# RL-GYM: configure process logging exactly once in RL mode
if rl_mode and not getattr(FleetSimulationBase, "_logging_configured", False):
    ... existing handler setup ...
    FleetSimulationBase._logging_configured = True
```

Non-RL behaviour must stay bit-identical. `run_examples.py` line ~97 runs scenarios
sequentially **in one process** whenever `n_parallel_sim == 1`, which is how `run_examples.py`
invokes it, and `example_depot_cali_sc_1.csv` holds ten seed rows. A blanket once-per-process
guard would route every scenario's log into the first scenario's `00_simulation.log` and leave
the remaining output directories with no log file at all. It would not change
`1_user-stats.csv`, so the byte-for-byte gate would not catch it. The motivation for the guard
— 1,000 episodes in one worker opening 1,000 file handlers — is an RL-mode concern only.

In RL mode with `G_SKIP_OUTPUT`, force `log_level` to warning with a `NullHandler`.

**How Ritun verifies:** the greedy scenario with default config still matches
`baseline_user_stats.csv` exactly; the same scenario with `skip_output = 1` produces no files
and does not erase an existing results directory; constructing the sim object twice in one
process in RL mode does not add a second log handler, while two non-RL constructions still
get one handler each, as they do today.

**Commit:** `RL-GYM: add skip_output flag and guard global logging setup`

---

### P1.3 — Split `user_request` in `RLPoolingIRSOnly`

**Files:** `src/rl_gym/__init__.py`, `src/rl_gym/fleetctrl_rl.py`

```
class PendingDecision:
    prq, candidates (list of (vid, vehplan, delta_cfv) exactly as insertion_with_heuristics
    returned it — do not re-sort), sim_time, rid_struct, cpu_elapsed

class RLPoolingIRSOnly(PoolingInsertionHeuristicOnly):
    def build_assignment_context(self, rq, sim_time) -> PendingDecision | None
    def commit_assignment_choice(self, pending, choice: int | None, sim_time) -> None
    def user_request(self, rq, sim_time)   # greedy fallback: build + commit with choice 0
    def set_reward_tracker(self, tracker) -> None
```

`build_assignment_context` performs everything `user_request` does up to the `argmin`. It
returns `None` when the request was fully resolved internally and no decision is needed:
`o_pos == d_pos` auto-reject, the reservation branch, or an empty candidate list — three
cases, not four. There is no duplicate-rid branch in the parent to reproduce; lines 109–111
overwrite `rq_dict[rid_struct]` unconditionally.

In those cases it must reproduce the parent's behaviour exactly, including `_create_rejection`
and the `G_FCTRL_CT_RQU` bookkeeping — **but the bookkeeping applies to the reservation and
empty-candidate branches only.** The `o_pos == d_pos` branch is a bare `return` at
`PoolingIRSOnly.py` line ~116 that jumps over the bookkeeping at lines ~133–141, and must keep
doing so. Adding bookkeeping there would write a `G_FCTRL_CT_RQU` value the parent never writes.

**CPU-time accounting across the yield.** The parent measures `perf_counter() - t0` across the
whole of `user_request`. In the split, the agent's decision sits between build and commit, so a
single timestamp would fold policy inference and scheduler latency into `G_FCTRL_CT_RQU`.
`G_FCTRL_CT_RQU` measures fleet-control computation, and the contamination would be worst in
evaluation runs, which is exactly where that number is read. So `PendingDecision` carries
`cpu_elapsed` — a **duration, not a timestamp**: build stores its own elapsed span, and commit
adds its own span before writing the sum. The two are identical for the greedy fallback, where
build and commit are adjacent.

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

**Files:** `src/rl_gym/sim_env_rl.py`

`src/misc/init_modules.py` needs no edit: P1.1 already registers
`RLImmediateDecisionsSimulation` through the dev hook, and
`get_src_simulation_environments()` returns it.

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

**Only the loop goes in the `try`, and only teardown goes in the `finally`** —
`_end_realtime_plot()`. The finalisation (`record_stats`, `save_final_state`,
`record_remaining_assignments`, `record_remaining_users`, `evaluate`) runs *after* a normally
completed loop and must stay out of the `finally`: `record_remaining_assignments` advances the
simulation up to `end_time + 14400` until every vehicle finishes its route, and an aborted
episode — `close()` on an env being torn down — must not pay that cost or advance the
simulation past `end_time`. An abandoned episode ends cheaply.

`step_generator` requires every operator to expose `build_assignment_context`, i.e. to be
`RLPoolingIRSOnly` or a subclass. If one does not, **raise**, naming both the `sim_env` and the
offending `op_module`. Do not fall back to `user_request`: a silent fallback would train
against greedy while reporting that it is learning, and the symptom would be a trained policy
that matches the baseline exactly, discovered weeks later.

Two presentation-only deviations from `run()`, both deliberate: the tqdm progress bar and the
end-of-scenario timing report are suppressed when `skip_output` is set and kept otherwise.
Across a thousand training episodes both are noise, and neither touches an output file.

**Do not** add `hook_manager` or any new positional parameter to
`FleetSimulationBase.__init__`.

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

**`is_valid` must come from `spaces.candidate_slot_validity` (P1.5), never from a fresh
`k < len(candidates)`.** That function is the single source of truth shared by the action mask
and this feature, and the two must agree slot for slot. See trap 12.

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

`Demand` is the instantiated class — confirmed, see §2.4. `SlaveDemand` is reserved for MobiTopp
coupling and is never built here.

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
- duplicate boardings — rids whose `record_boarding` fired more than once. Expected 1 on the
  reference day (rid 50). The pickup count above must still be 288 distinct rids, not 289
  calls; `on_pickup` is last-write-wins per §6.4.

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
`src/misc/config.py`. This reference scenario uses CSV, though the API accepts YAML too.

Keep all diagnostic state in this class. Do not put subclass-specific state in a base class.

**`check_env` and the assert are in direct conflict — resolved with a test-only shim.**
`gymnasium.utils.env_checker.check_env` calls
`env_step_passive_checker(env, env.action_space.sample())`, a uniform draw over `Discrete(K+1)`
that ignores the mask entirely. Measured on a stand-in with 3 valid slots, `check_env` failed
**24 of 40 runs**, and nondeterministically, which reads as a flaky environment rather than a
contract mismatch. Both requirements were written without noticing they exclude each other.

The resolution: **keep the assert, and run `check_env` against a thin shim defined in the test
file** whose `step` maps an illegal sampled action onto the nearest legal one before delegating.
`check_env` then still exercises everything it is good for — space validity and limits,
reset/step return types, observation-in-space, reset-seed determinism, step determinism — while
the real env keeps a hard assert, which gets its own direct unit test so it is not decoration.
The shim never appears in `gym_env.py`.

Do **not** resolve it by clamping an invalid action to reject: a policy bug would then surface
as an apparent preference for rejecting riders, which is exactly the class of silent corruption
this design keeps engineering out. Do not add a config flag that disables masking for the check
either — that puts a production code path in the tree solely for a test.

`check_env` performs several resets and each builds a full FleetPy simulation, so it takes
seconds, not milliseconds.

**How Ritun verifies:** `check_env` passes through the shim; `reset()` returns the declared
shape and dtype; 10 arbitrary valid actions step without error; 20 reset/close
cycles leak no memory and no file handles.

**Commit:** `RL-GYM: add SDPDPAssignmentEnv gymnasium environment`

---

### P1.9 — Per-episode seeding

**Files:** `src/rl_gym/gym_env.py`

`reset(seed=None)` seeds the env's own `np.random.Generator` on first call, then draws an
episode seed from it and writes `scenario_parameters[G_RANDOM_SEED] = int(episode_seed)`
before construction. Each `SubprocVecEnv` worker gets a different base seed derived from
`env_id`, via `np.random.SeedSequence([base_seed, env_id])`.

**This applies only when `base_seed` is set (D7).** With `base_seed` absent, `reset()` seeds
the env's own generator but leaves `G_RANDOM_SEED` at whatever the scenario row provides, so
the byte-for-byte gates keep testing the plumbing rather than the seeding. P1.10 runs in that
mode.

Episode seeds are drawn from `[0, MAX_EPISODE_SEED]`, with
`MAX_EPISODE_SEED = (2**32 - 1) // 1712 = 2,508,742` (defined in `gym_env.py`):
`np.random.seed` raises above `2**32 - 1`, and `Demand.load_demand_file` seeds with 1712 × the
episode seed. *(Corrected before Phase 2: an earlier draft of this item said `[0, 2**31)`; the
code has always used the bound above.)* Every stochastic component in the simulator — rider decline, the diffusion random term, no-show,
initial vehicle state and SoC, stochastic travel time — draws from the global `np.random`
stream that `FleetSimulationBase.__init__` seeds from this key, so writing it before
construction is both sufficient and complete.

The env records the seed actually used in `episode_summary()` under `random_seed`, so a
training log shows which sample path each episode ran.

**Phase 1 implements seeding only.** The demand day is not per-scenario-row —
`example_depot_cali_sc_1.csv` carries only `op_module`, `scenario_name` and `random_seed`,
while `day_dir_name` and `rq_file` live in the constant config, so rotating rows would
change the seed and nothing else. Multi-day rotation is deferred to Phase 2 and will write
`G_DAY_DIR` / `G_RQ_FILE` into `scenario_parameters` directly (research proposal §3.1.2).
Accept `scenario_pool` in the config for forward compatibility but leave it unused.

When `skip_output` is False, set
`scenario_name = f"{base}_env{env_id}_pid{os.getpid()}_ep{counter}"`. Both the pid and the
counter are needed — pid because independently-launched processes can share an `env_id`,
counter because `create_or_empty_dir` would otherwise erase the previous episode.

The uniqueness key is therefore `(env_id, pid, episode counter)`. **Two env *instances* sharing
an `env_id` inside one process still collide** on their first episode, since all three
components match — verified. That requires constructing several envs in a single process with
the same `env_id`, which is the `DummyVecEnv` pattern D8 forbids; under `SubprocVecEnv` each
worker is its own process with its own `env_id`. Give every env a distinct `env_id` and the key
holds.

**How Ritun verifies:** two resets with the same seed give identical trajectories under a
fixed action sequence; two resets with different seeds produce different `1_user-stats.csv`
output. Both must hold.

**Commit:** `RL-GYM: add per-episode seeding`

---

### P1.10 — Phase 1 exit gate: scripted greedy rollout

**Files:** `src/rl_gym/policies.py`, `tests/test_rl_gym.py`

`policies.py` holds `greedy_action(obs, mask, k_max)` — one file, one function. It is the
tested helper the driver selects through. Its dependence on the observation layout
(`delta_cfv` at `obs[K:2K]`) is stated in its docstring rather than implied by where the file
sits. *Superseded in Phase 2:* the evaluation baseline is stock `PoolingIRSOnly` (D9), not this
function, and P2.8 makes it select `delta_cfv` by name through the layout map.

Drive the env entirely from outside: `obs, _ = env.reset()`, then at each step choose the
`argmin` of `delta_cfv` **restricted to slots where `is_valid = 1`**. Equivalently, since the
list arrives sorted and truncation is a prefix, the scripted greedy action is always slot 0
whenever any candidate is valid. Step, repeat until `terminated`. Run with
`skip_output = 0` so a user-stats file is produced.

**The restriction is required independent of the sign of `delta_cfv`.** On this scenario real
`delta_cfv` happens to be negative (§5.1), so 0.0 padding is the largest value in the slot and
an unrestricted `argmin` would coincidentally also return slot 0 — meaning **this byte-for-byte
gate would pass even with the restriction removed, and does not by itself protect against a
padding bug.** The sign is a property of this objective function
(`distance_and_user_times_with_walk`, which penalises unserved requests so that serving one
improves the objective) and could change with a different control function, a different
scenario, or a Phase 2 reward-shaping change. A padded slot holds padding, not a cost, and must
never compete. The counter-example — positive costs in the valid slots, where an unrestricted
`argmin` returns a padded slot — is pinned by
`tests/test_rl_gym_observers.py::TestPaddingMustNotCompeteInArgmin`.

Those tests pin the contract, not the driver. **P1.10's driver must therefore select its action
through a tested helper rather than an inline `np.argmin`**, and that helper must be covered by a
test that fails if the `is_valid` restriction is dropped. A guard with no test that fails when it
is removed is decoration.

The resulting `1_user-stats.csv` must match `baseline_user_stats.csv` **byte for byte**.

If it passes, the generator, observation extraction, ranking, masking, action translation,
reward wiring, seeding, and episode termination are all correct simultaneously — with no RL
library involved.

**If it nearly matches, do not relax the comparison to a tolerance.** A near-match means the
RL path consumes global RNG differently from the greedy path, and that will quietly
contaminate every stochastic result reported later. Report it and stop.

Also report: step count, wall-clock seconds, steps/second, reward-event breakdown, and how
many requests took the reservation branch or returned an empty candidate list.

Also report the distribution of gaps between consecutive decision epochs
(min, median, mean, 95th percentile, max), in **both** of these, separately labelled:

- **simulated seconds** — `sim_time` elapsed between one epoch and the next. This is the
  one that matters for the reward question: the simulated gap is how much world each reward
  window sweeps up, so it bounds how many pickups, declines, cancellations and no-shows can
  land in a single `flush()`. A long tail here means high per-step reward variance.
- **wall-clock seconds** — compute time between epochs, for throughput planning only.

Measured under P1.4 with a scripted greedy driver, wall-clock: 436 steps in 37.93 s,
11.5 steps/second, gaps min 0.0068 s, median 0.0759 s, mean 0.0864 s, p95 0.1664 s,
max 0.4497 s. The simulated-second distribution has not been measured yet.

Create if `docs/RL_GYM_PHASE1_RESULTS.md` is absent. Write these figures to `docs/RL_GYM_PHASE1_RESULTS.md` as well as reporting them: step
count, reservation-branch count, empty-candidate count, wall-clock seconds, steps/second,
the full `episode_summary()` breakdown, and the candidate-list length distribution
(min, median, mean, max, and the count exceeding `K`). Measured under P1.3 on the reference
day, as the figure to reproduce: 436 decisions, lengths min 1, median 5, mean 4.60, max 11,
13 exceeding `K = 8` (D3).

**How Ritun verifies:** the byte comparison, plus a step-count reconciliation:
step count + reservation-branch count + empty-candidate count + same-origin-destination
count + duplicate-rid count = 445. Every request must be accounted for in exactly one bucket.
The duplicate-rid bucket is expected to be **0** — the parent has no such branch (§1). Keep
the bucket and report it: a non-zero value means the source changed.

**445 is the count after *both* of `load_demand_file`'s filters, not the row count of the
file** (§2.9). The reference day happens to lose nothing to either, so the two coincide here
and the distinction is invisible — it is not invisible on the corpus. The
**same-origin-destination bucket is also expected to be 0, and for a different reason than the
duplicate-rid one**: same-node rows are deleted at load and never reach the operator (§1), so
the bucket stays 0 even on the 7 corpus days that contain one. P2.4 measured it as 0 on all 127
days. Keep it and report it, but do not read a 0 as evidence that no such request existed.

**Commit:** `RL-GYM: add scripted greedy rollout test (Phase 1 exit gate)`

---

### P1.11 — `SubprocVecEnv` smoke test

**Files:** `train_sdpdp.py` (skeleton only)

`make_env(cfg, env_id)` closure; envs constructed **inside** `_init()`, never before
`SubprocVecEnv`. FleetPy objects hold routing engines and file handles and are not picklable.

**`train_sdpdp.py` must raise if `base_seed` is missing from the config, never default it.**
With `base_seed` absent the env deliberately does not reseed (D7), so a training run would
replay one sample path every episode with nothing erroring and the logs looking healthy.

4 workers, random masked actions, one episode each. No learning algorithm yet.

Append to (creating if absent) `docs/RL_GYM_PHASE1_RESULTS.md`: worker count, per-episode wall clock under
four workers, peak memory, and whether any process orphaned or any log file interleaved.

**How Ritun verifies:** four processes run to completion, no crashes, no orphaned processes
after `env.close()`, no interleaved or corrupted log files.

**Commit:** `RL-GYM: add SubprocVecEnv smoke test harness`

---

## 8. Phase 2 work items

Phase 1 is closed: the exit gate passes byte for byte and the results are in
`docs/RL_GYM_PHASE1_RESULTS.md`. The items below were expanded from the Phase 2 topics in the
design sessions of 27–29 Sep 2026. §0's protocol applies unchanged. **Order matters.**

Phase 2 builds **machinery**. It ends with a short test training run (P2.15) that exercises
every component end to end, with repositioning off. Performance is not judged in Phase 2; the
weight sweep, the scenario changes and full-length training come after it (end of this
section).

**Every item that touches the env, the trackers or the observation (P2.0b, P2.3, P2.7–P2.14)
re-runs the P1.10 gate as part of its verification.** Reading a feature must never change the simulation (trap 14), and
the gate is the only check that would notice. *(P2.13 was added to this list in P2.4: it now
edits `sim_env_rl.py` and `gym_env.py` for the `tail_seconds` field, so it touches the env like
the rest.)*

---

### P2.0 — Resolve the duplicate OpenMP runtime — **DEFERRED by Ritun; do not start**

**Status:** deferred. Phase 2 starts at P2.0b. The import-order workaround (§8a) stays in place
for all of Phase 2, and §8a is not deleted. Every new entry point that imports `torch`,
`stable_baselines3` or `sb3_contrib` — scripts, test files and notebooks alike — imports `numpy`
first, per §8a. Do not reorder those imports, and do not run an import sorter such as `isort`
over them.

Why it is deferred: the workaround works and has passed every gate, including P1.11's
multi-worker run; a violation fails loudly and at once (`OMP: Error #15`), never silently; the
conflict is specific to macOS, while full-length training runs on Linux (HPC3); and the fix
itself risks moving `numpy` or `scipy`, which would invalidate `baseline_user_stats.csv`. The
text below is kept for when Ritun revisits it.

**Files:** `train_sdpdp.py`; §8a of this spec

Install `torch` from conda-forge so `fleetpy_rl` links a single `libomp` — **Ritun does the
install** (§0: Claude Code does not install packages) — then **re-run the P1.10 gate**. If
`1_user-stats.csv` still matches `baseline_user_stats.csv` byte for byte, remove the
import-order workaround in `train_sdpdp.py` and delete §8a. If it does not match, that is the
finding — report it and stop; the environment as it stands works and has passed every gate, so
nothing is lost by leaving the workaround in place.

This is its own task, not a step inside another. It moves `torch`, and a resolution that also
moved `numpy` or `scipy` would change floating-point results and invalidate the baseline with no
visible symptom — which is exactly why the re-verification is part of the task rather than an
afterthought.

**How Ritun verifies**, after the install and again after removing the workaround:
1. The P1.10 gate, byte for byte.
2. In fresh interpreters with no workaround, `python -c "import torch"`,
   `python -c "import stable_baselines3"` and `python -c "import torch, numpy"` all exit 0. The
   P1.10 gate never imports `torch`, so on its own it cannot show that the OpenMP conflict is
   gone.
3. P1.11's `SubprocVecEnv` smoke test completes.

**Commit:** `RL-GYM: single OpenMP runtime; remove import-order workaround`

---

### P2.0b — Per-epoch reward breakdown

**Files:** `src/rl_gym/reward.py`, `src/rl_gym/gym_env.py`, `tests/test_rl_gym_reward.py`,
`src/rl_gym/tools/reward_trace.py` (new); report `docs/RL_GYM_REWARD_TRACE.md`

Three related changes to the tracker and its readout. None changes what is charged under explicit
weights.

1. **Per-event breakdown.** `RewardTracker._add` takes optional `kind`, `rid` and `detail` and
   appends `(kind, rid, amount, detail)` to a per-window event list; `detail` is the wait in
   seconds for pickups and `None` otherwise. `flush()` returns the total as now; a companion
   `flush_events()` returns the list and clears it. `SDPDPAssignmentEnv` puts the list in
   `info["reward_events"]` on each step. With tracking on, **every classified exit appears as an
   event**, with amount 0 where its weight is 0 — `on_exit` currently adds nothing when the
   penalty is 0, and with §6.1's weights that would hide served, no-show, no-candidate and
   unserved-at-horizon exits, which are exactly the labels the trace exists to check.
   Zero-amount events change no total. Behind a config key, `track_events`, defaulting to
   `False` — training does not need it and 7,000 episodes should not pay for it. `flush()`'s
   return value must be unchanged in both modes.
2. **Defaults follow §6.1.** `DEFAULT_REWARD_WEIGHTS` still holds the Phase 1 values
   (`w_reject = 0.5`, `w_noshow = 0.8`), and the env passes `reward_weights = None` by default,
   so anything run without explicit weights — this item's trace, P2.4's sweep — would silently
   use them. Change the defaults to §6.1's Phase 2 weights, and update the tests that pin the old
   values in `tests/test_rl_gym_reward.py`. P2.13 additionally requires explicit weights for
   training.
3. **Horizon margin in the summary.** `episode_summary()` gains `last_dropoff` — the largest
   `do_time` among served riders, taken in `on_exit` — and
   `horizon_margin = end_time − last_dropoff`: negative when the post-`end_time` tail did work,
   `None` if nobody was served. P2.4 and P2.13 read these; nothing else records the last dropoff.

**How Ritun verifies:**
- The P1.10 gate, byte for byte: the tracker must not perturb the simulation.
- A scripted greedy run with `track_events = True`, through `reward_trace.py`, writes to
  `docs/RL_GYM_REWARD_TRACE.md` the first 40 steps and any step with three or more events: for
  each, the step's reward and the events composing it, each with its kind, rid and amount, and
  for pickups the wait. Report any step where the listed amounts do not sum to the step's
  reward, and confirm the sum over all steps equals `episode_reward`.
- On the reference day, `last_dropoff` is 69805.3 and `horizon_margin` about 194.7 s (P1.10).

The point of the breakdown is to check the §6.4 ladder assigns the right label per event, which
the episode-level counts cannot show — 118 declines could be the correct total with several
individually misclassified.

**Commit:** `RL-GYM: add per-epoch reward breakdown, Phase 2 default weights, horizon margin`

---

### P2.1 — Correct the network CRS

**Files:** `data/networks/example_network/base/crs.info`

Change `EPSG:32632` to `EPSG:32610` (§2.10). Every other byte of the file stays as it is. The
file has no trailing newline and must not gain one. **No `# RL-GYM:` comment** — the file is read raw into `self.crs` and
handed to pyproj, so a comment line would corrupt it (trap 15). Search the repository again for
readers of `crs.info` and users of `self.crs`; the list in §2.10 comes from drafting, so report
any addition. Leave `base - Copy/` alone.

**How Ritun verifies:** the P1.10 gate, byte for byte; `return_network_bounding_box()` now
returns the corners (−122.1102, 37.2921) and (−121.9153, 37.4076), each to within 1e-3°. These
are the two corners of the UTM rectangle reprojected, so they do not coincide with the
`lat`/`lon` hull in `nodes.csv` (37.2915–37.4089, −122.1095 to −121.9162): the **longitudes
fall fractionally outside** it (−122.1102 against −122.1095; −121.9153 against −121.9162) while
the **latitudes fall fractionally inside** (37.2921 against 37.2915; 37.4076 against 37.4089),
because the UTM grid here is turned about 0.5–0.7° from true north (grid convergence), so a
corner of the UTM rectangle is not at the extreme latitude or longitude of the node set. Compare
against the stated corners, not against the hull. Under the old value the longitudes read
9.89–10.08.

**Commit:** `RL-GYM: correct example_network CRS to EPSG:32610`

---

### P2.2 — Split and freeze the demand corpus

**Files:** the demand directories named in §2.9 only.

Create `corpus_eval/` and `corpus_excluded/` beside `corpus/`, and move the 26 evaluation files
and the 5 excluded files per §2.9's lists. Filesystem moves only — no git commands (§0); Ritun
commits, and git records the moves as renames. **The 26 days listed in §2.9 are authoritative:
move exactly those.** As a cross-check, re-derive the list with §2.9's algorithm before moving
anything. It reproduces the list exactly, so a mismatch means the corpus or the reading of the
algorithm has changed — report the difference and stop.

**How Ritun verifies:** 101 / 26 / 5 files, 132 in total, no filename in two directories;
weekday counts Tue 35 / Wed 33 / Thu 33 for training and 8 / 9 / 9 for evaluation; every month
present in both; each moved file's checksum unchanged by the move; `thursday/2024-06-20.csv`
untouched; the P1.10 gate still byte for byte, since the reference path never reads the corpus.

**Commit:** `RL-GYM: split demand corpus into train, eval and excluded`

---

### P2.3 — Multi-day episodes and day rotation

**Files:** `src/rl_gym/day_rotation.py`, `src/rl_gym/gym_env.py`, `train_sdpdp.py`,
`tests/test_rl_gym_env.py`

P1.9 deferred this here. The env writes the demand directory and file into
`scenario_parameters` before construction (research proposal §3.1.2). D10's selection lives in
`day_rotation.py` as pure functions of `(sorted days, base_seed, n_workers, env_id, episode
index)` — no FleetPy import, no env state — so a pass over several workers is a millisecond
unit test instead of one real episode per position. New config keys:

- `day_dir` — demand directory name: `corpus` for training, `corpus_eval` for evaluation.
  **Absent → the scenario's own `day_dir_name` / `rq_file` are used unchanged**, which is the
  reference path the gates need.
- `day` — optional fixed day (filename stem). When set, every episode runs that day. Used by
  P2.4 and P2.14.
- `n_workers` and `base_seed` — **both** required by D10 whenever `day_dir` is set and neither
  `day` nor `eval_pairs` is. `n_workers` splits each pass; `base_seed` is what
  `SeedSequence([base_seed, 1010, p + 1])` draws the permutation from, so without it there is no
  pass order at all. The env raises for either, and raises on `env_id >= n_workers` — a worker
  numbered outside the split owns no position in any pass and would silently run nothing of the
  rotation. **`n_workers` is not a `train_sdpdp.py` config key**: it is passed into `make_env`
  from the same value that sizes the `SubprocVecEnv`, because a config value of 4 run with
  `--workers 3` would drop a quarter of every pass with every episode still succeeding.
- `episode_seed` — optional. When set, every episode runs with exactly this `G_RANDOM_SEED`,
  which must lie in `[0, MAX_EPISODE_SEED]` (P1.9). It cannot be combined with `base_seed`; the
  env raises if both are set. With `day`, it pins one (day, seed) pair: used by P2.13's
  checkpoint check, P2.14 and P2.15.
- `eval_pairs` — optional ordered list of (day, seed). Episode `i` runs pair `i mod len`.
  Requires `day_dir`; cannot be combined with `day`, `episode_seed` or `base_seed`. Used by
  P2.13's evaluation callback.

With `day_dir` set and neither `day` nor `eval_pairs`, days rotate per D10. `episode_summary()`
records the day, and also `day_dir`, `weekday`, `month`, `pass` and `pass_position` — in
rotation mode the output directory does *not* name the day, so this is the only record of which
day an episode ran, and P2.13's day-visit histogram is built from it. When output is on and the
pair is fixed (`episode_seed` or `eval_pairs`), the output directory is
`f"{base}_{day}_s{seed}_env{env_id}_pid{pid}_ep{counter}"`, so it names the pair it holds.
`scenario_pool` stays accepted and unused.

Every day named in the config — `day`, and each day of `eval_pairs` — must exist as a `*.csv`
in `day_dir`, checked at construction; otherwise a typo surfaces as a path error deep inside
`load_demand_file`. Day listings are `*.csv` only, per §2.9.

The env parses weekday and month from the filename and exposes them for the request block
(P2.9). A stem that is not an ISO date yields `None` for both rather than raising, so the
reference path's `rq_file` need not be ISO-named; P2.9 is where an unknown weekday or month
raises, against P2.6's vocabularies (§5.2).

VERIFY: the `scenario_parameters` keys that carry the demand directory and file, and that no
component caches the demand path before construction. *(Checked in P2.3: `G_DAY_DIR` and
`G_RQ_FILE`; the directory is assembled at construction by `get_directory_dict`
(`globals.py` ~741) from one call in `FleetSimulationBase.__init__` (~150), the filename read at
~344 and passed straight to `load_demand_file`. Nothing else caches it — the only other
`get_directory_dict` callers are the evaluation and replay tools, outside a run.)*

**How Ritun verifies:**
1. With `day_dir` absent, the P1.10 gate, byte for byte.
2. With `day_dir = "corpus_eval"`, `day = "2024-06-20"` and `base_seed` absent, the scripted
   greedy run's `1_user-stats.csv` is byte-identical to `baseline_user_stats.csv`. The input
   differs from `thursday/` only in line endings, which the demand loader does not see.
3. A unit test with 3 workers and a 10-day list, each worker run through its own share of every
   pass (4 episodes for worker 0, 3 for workers 1 and 2): over two full passes every day appears
   exactly twice across the workers; each worker's day sequence is deterministic given
   `base_seed`; episode seeds still differ from episode to episode; and pass `p`'s permutation
   comes from `SeedSequence([base_seed, 1010, p + 1])` (D10).
4. With `day_dir = "corpus_eval"`, `day = "2024-06-20"` and `episode_seed = 42`, the output is
   byte-identical to `baseline_user_stats.csv` (42 is the row's seed); with `episode_seed = 43`
   it differs. Setting both `episode_seed` and `base_seed` raises.
5. A unit test: `eval_pairs` runs its pairs in order and wraps around; each episode's
   `G_RANDOM_SEED` and day match its pair.

**Commit:** `RL-GYM: add multi-day episodes and shuffled-pass day rotation`

---

### P2.4 — Corpus sanity sweep under greedy

**Files:** `src/rl_gym/tools/corpus_sweep.py` (new); report `docs/RL_GYM_CORPUS_SWEEP.md`

Every Phase 1 figure comes from one day. Run the P1.10 greedy driver through the env on each of
the 127 kept days (fixed-day mode from P2.3, `base_seed` absent so the row's seed 42 applies),
one env per process (D8), in parallel. No change to the env or the simulation.

Per day, report:
- requests in the file, requests loaded after the window filter, gym steps, and the
  reconciliation buckets of P1.10 (empty candidate list, reservation, origin = destination,
  duplicate rid) — every loaded request in exactly one bucket;
- `unclassified` (must be 0) and duplicate boardings;
- candidate-list length: max, p95, and the count exceeding `K`;
- `last_dropoff` and `horizon_margin` from `episode_summary()` (P2.0b), and the simulated
  seconds added by the `record_remaining_assignments` tail, measured as in P1.10;
- pickups, declines, post-match cancellations, no-shows;
- episode wall clock.

Summarise: any day with `unclassified ≠ 0` (stop and report — a new §6.4 path exists), the days
where lists most exceed `K`, the days with the thinnest horizon margin, and any day where the
tail does real work. Decisions on `K` and `w_horizon` stay with Ritun.

**How Ritun verifies:** the 2024-06-20 row reproduces P1.10's figures exactly — 445 loaded,
436 steps, 9 empty lists, max list 11, 13 lists over `K`, `horizon_margin` about 194.7 s, 288
pickups, 118 declines, 28 cancellations, 2 no-shows.

**Commit:** `RL-GYM: add corpus sanity sweep and report`

---

### P2.5 — H3 neighbour table

**Files:** `src/rl_gym/offline/build_zone_table.py` (new),
`data/rl_gym/example_network/h3_neighbours.csv` (new), `src/rl_gym/zones.py` (new),
`tests/test_rl_gym_zones.py` (new)

The cells are the 37 resolution-7 cells in the `h3_id` column of `base/nodes.csv` (checked
against `h3` 4.5.0 `latlng_to_cell` for all 4,084 nodes during drafting). Neighbours are
computed from `data/networks/example_network/h3_hexagons.geojson` with shapely, which
`fleetpy_rl` already has: two cells are neighbours when their polygons lie within 1e-9° of each
other, which in a hexagonal tiling is exactly edge-sharing. **That tolerance is not a rounding
decision — it sits in a gap seven orders of magnitude wide on either side.** Measured over all
666 cell pairs under P2.5: the 88 neighbour pairs are at shapely distance **exactly 0.0**,
because adjacent hexagons in this geojson share identical vertex coordinates, and the nearest
**non**-neighbour pair is at **1.377e-02**; every threshold from 0 to 1e-2 yields the same 88
pairs. Do not tighten or loosen it thinking it is delicate. Output rows `(cell, neighbour)`,
including `(cell, cell)`, restricted to the 37 cells — neighbours outside them contain no nodes,
so no event can land there. **Nothing imports `h3`** (§10).

`zones.py` loads the node → cell map from `nodes.csv` and the table, and exposes
`cell_of_node(node)` and `zone_set(cell)`.

**How Ritun verifies:** against these figures, computed during drafting with `h3` 4.5.0
`grid_disk(cell, 1)` restricted to the 37 cells — `h3` is not installed and must not be (§0,
§10), so compare with the numbers rather than calling it: 88 undirected neighbour pairs, every
cell with 2–6 neighbours inside the set, 17 cells with all 6. Also: the relation is symmetric,
and the geojson's 37 ids equal the `nodes.csv` set.

**Commit:** `RL-GYM: add H3 neighbour table and zone lookup`

---

### P2.6 — Arrival baseline and corpus manifest

**Files:** `src/rl_gym/offline/build_arrival_baseline.py` (new),
`data/rl_gym/example_network/arrival_baseline.csv` (new),
`data/rl_gym/example_network/corpus_manifest.json` (new), `tests/test_rl_gym_baseline.py` (new)

Inputs: **training days only** (`corpus/`), never evaluation or excluded days.

For each training day `d`, each minute `t` in `[start_time, end_time)` of the reference scenario,
and `W ∈ {30, 60}`: `c_d(t, W)` = requests in the file with `rq_time` in `[t − W, t)` and inside
`[start_time, end_time)` — the §5.2 window definition, so observed and expected are comparable.

- `mean(m, t, W)` = mean of `c_d(t, W)` over the training days of month `m`.
- `sd(t, W)` = sample standard deviation, over **all** training days, of
  `c_d(t, W) − mean(month(d), t, W)`. Pooled across months, because 9–12 days per month are too
  few to estimate spread.

Online (P2.11): `expected = mean(month, t, W)` and `z = (observed − expected) / max(sd, 1.0)`,
with `t` = `sim_time` floored to the minute.

**Why month, not weekday.** Month explains 50% of the variance in daily request totals; weekday
explains none (Tue, Wed and Thu average 453–454). Against a weekday baseline, October days read
about +0.7 SD high on average (+0.66 at 10:00, +0.78 at 15:30), so the feature would mostly
report "it is October", which the month one-hot already says. Weekday does matter around the
15:00–16:00 peak — at 15:30, Tuesdays average 22.1 arrivals per 30 min against 27.2 overall —
and the weekday one-hot stays in the observation for that. Averaged over the day, the typical
residual spread is 4.9 with the month baseline, 5.3 with weekday, 5.6 with none; a weekday shape
scaled by month level gave 5.07. September's mean blends 7 training days before the level shift
with 3 after it (§2.9); accepted.

**Manifest:** `files` — every training file, sorted by filename, with the SHA-256 of its bytes;
`corpus_hash`; the month vocabulary (Jan–Oct); the weekday vocabulary (Tue, Wed, Thu);
`start_time`, `end_time`, the windows, and the SD floor. `corpus_hash` is the SHA-256 of the
UTF-8 text made of one line per entry of `files`, in order, each line
`<filename>:<SHA-256 of the file's bytes>`, joined with `\n`. It covers contents as well as
names, so an edited file is caught too, and the per-file list lets the env say which files
differ. The env-side check that uses it is
added in P2.9, the first item that reads the manifest (trap 16).

**How Ritun verifies:** for 3 random training days and 3 random minutes, `c_d` recomputed by
hand from the CSV matches; a plot of `mean(m, t, 30)` per month shows October above the rest;
the manifest hash matches the directory.

**Commit:** `RL-GYM: add month-conditioned arrival baseline and corpus manifest`

---

### P2.7 — `RollingStatsTracker`

**Files:** `src/rl_gym/rolling_stats.py` (new), `src/rl_gym/reward.py` (shared classifier),
`src/rl_gym/sim_env_rl.py` (tracker clock; arrival hook, unless it belongs in
`src/rl_gym/fleetctrl_rl.py`), `src/rl_gym/gym_env.py` (wiring),
`tests/test_rl_gym_rolling_stats.py` (new)

Events:
- **arrival** — rid, `rq_time`, origin cell — at the single entry point every loaded request
  passes through, including origin = destination, reservation and no-candidate requests.
  VERIFY that this is `build_assignment_context`.
- **decline** and **post-match cancellation** — from the same exit callback as `RewardTracker`,
  labelled by one §6.4 classification function shared by both trackers (trap 13); keyed to the
  rider's origin cell, timed when they fire. The exit callback receives only `(rid, rq)`, so the
  tracker keeps a clock: `step_generator` sets it to `sim_time` at the start of every step. A
  decline is therefore timed at its arrival step, and a post-match cancellation at the step its
  diffusion check fires.
- **pickup** — from the boarding callback; stores pickup time and realized minus quoted wait,
  last-write-wins per rid (§6.4). Quoted wait from the operator side (§5.2); nothing from the
  diffusion model.

Queries, all over `[t − W, t)`: counts for a zone set, fleet-wide counts, and the mean wait gap
(`None` when empty; the observer writes 0.0). Per-kind deques, pruned beyond 60 minutes; a
60-minute window holds a few dozen events fleet-wide, so zone counts can scan the deque. The
tracker is rebuilt with the simulation at every reset (§10).

**How Ritun verifies**, on a greedy reference-day run:
- end-of-day totals match P1.10: 445 arrivals, 118 declines, 28 cancellations, 288 distinct
  pickups;
- at 20 random decision times, every windowed count and the wait-gap mean equal an independent
  recomputation from `1_user-stats.csv`. VERIFY which columns carry each event's time. Found in
  review: declines leave `decision_time` empty (all 118 on the reference day) and fire in their
  arrival step, so their time is `rq_time`; post-match cancellations fire at `decision_time`
  (`leave_system_time`, set to the step time), while `diffusion_decision_time` is model-internal
  and differs in 4 of the 28; pickups are at `pickup_time`; the quoted wait is `t_wait` in
  `offers`. Draw the 20 times from outside [33032.9, 33230): rid 50's first boarding falls there
  and is later overwritten (§6.4), so `1_user-stats.csv` cannot reproduce a window containing
  it;
- online fleet-wide arrival counts at every decision equal P2.6's definition recomputed from the
  demand file;
- the P1.10 gate, byte for byte.

**Commit:** `RL-GYM: add rolling statistics tracker`

---

### P2.8 — Observation layout map; greedy by name

**Files:** `src/rl_gym/observers.py`, `src/rl_gym/gym_env.py`, `src/rl_gym/policies.py`,
`src/rl_gym/spaces.py` (docstring), `src/rl_gym/tools/corpus_sweep.py` (P2.4's driver calls
`greedy_action`), `tests/test_rl_gym.py`, `tests/test_rl_gym_observers.py`

No feature changes in this item — the vector stays identical to Phase 1.

- Each observer declares its features statically: `feature_spec()` returns an ordered list of
  `(name, length)`, fixed at construction. The env publishes `observation_layout` — an ordered
  mapping from feature name to slice — built from those specs at construction. The observers'
  dicts exist only after `observe()` runs, so they cannot be the source. A test asserts that
  every `observe()` returns exactly the declared names, in the declared order and lengths.
- `greedy_action(obs, mask, layout)` finds `delta_cfv` by name and keeps the `is_valid`
  restriction (P1.10). The hardcoded `(3K + 4,)` shape check goes.
- Fix two stale docstrings: `spaces.py`'s "k-th cheapest candidate" becomes "k-th candidate in
  ascending `delta_cfv` order" (D3); `observers.py`'s padding rationale is replaced by §5.1's
  (`VecNormalize` is the whole reason).

**How Ritun verifies:** the P1.10 gate, byte for byte; the layout's slices tile
`[0, obs_len)` with no gap or overlap; `check_env` through `MaskRespectingShim` passes with
`base_seed` absent and set.

**Commit:** `RL-GYM: add observation layout map; greedy selects by feature name`

---

### P2.9 — Request block

**Files:** `src/rl_gym/observers.py` (`RequestObserver`), `src/rl_gym/gym_env.py`,
`tests/test_rl_gym_observers.py`, `tests/test_rl_gym_env.py` (it asserts the observation shape)

Per §5.2's request block. Node lat/lon and the bbox are loaded once at env construction from
`nodes.csv` (§2.10); vocabularies from P2.6's manifest; zone counts from P2.7 through P2.5's
zone sets.

This item also adds the corpus check (trap 16): at construction the env loads P2.6's manifest,
recomputes `corpus_hash` over the files in the corpus directory exactly as P2.6 defines it, and
on a mismatch raises, naming the files that are missing, extra or changed (from the manifest's
per-file list). The corpus directory comes from a new config key, `corpus_dir`, defaulting to
`data/demand/example_demand/matched/example_network/corpus`, so a test can point it at a
temporary copy. Like every other corpus reader it enumerates days by `*.csv` only (§2.9).

**How Ritun verifies:** the P1.10 gate, byte for byte, with the block present; for 10 decisions,
every feature matches a hand computation — coordinates from `nodes.csv`, one-hots from the
filename, zone counts from P2.7's recomputation; observation length +24; and with `corpus_dir`
pointed at a temporary copy of `corpus/`, removing one file makes construction raise and name
it, and editing one file does the same. **Also: dropping a stray `.DS_Store` into that copy must
leave construction working** — §2.9's `*.csv`-only rule, which is otherwise the kind of thing
that passes every test here and then fails on Ritun's machine the first time Finder opens the
directory.

**Commit:** `RL-GYM: add Phase 2 request observation block`

---

### P2.10 — Candidate block

**Files:** `src/rl_gym/observers.py` (`CandidateObserver`), `src/rl_gym/gym_env.py` (config
keys `wait_normalizer_s`, `position_encoding`), `tests/test_rl_gym_observers.py`,
`tests/test_rl_gym_env.py` (it asserts the observation shape)

Per §5.2's candidate block: features 4–12 appended after the three Phase 1 features. One
re-timed pre-insertion copy per candidate vehicle per decision, shared by features 9 and 12. The
routing cache is restored after every observation, per §5.2 (b).

**How Ritun verifies:**
- **the P1.10 gate, byte for byte — the critical check for this item**, since it re-times plan
  copies and queries the router (trap 14). Tests also assert that `veh_plans[vid]` — stops,
  `pax_info`, utility — is unchanged across an observation, apart from §5.2's exception (a), and
  that `routing_engine.travel_time_infos` holds the same keys before and after one;
- with `wait_normalizer_s` absent, `offered_wait` is bit-identical to Phase 1's;
- for 10 decisions where some candidate has two or more pending riders, features 9–12 match a
  hand computation;
- `directionality` matches `filter_directionality`'s formula on constructed cases, including an
  idle vehicle (0) and a vehicle whose route ends at its own position (0);
- a distribution table over the reference day for features 6–12 (min, median, p95, max, share
  exactly zero). Expect `lateness_pre` to be mostly ≤ 0 (§5.2);
- throughput: steps per second against Phase 1's 11.47, and `reset()` time.

**Commit:** `RL-GYM: add Phase 2 candidate observation block`

---

### P2.11 — Global block

**Files:** `src/rl_gym/observers.py` (`GlobalStateObserver`), `src/rl_gym/gym_env.py`,
`tests/test_rl_gym_observers.py`, `tests/test_rl_gym_env.py` (it asserts the observation shape)

Per §5.2's global block: rolling counts and the wait gap from P2.7, the anomaly from P2.6's
baseline, `active_*` from `op_act_fs_file` read as §5.2 describes.

**How Ritun verifies:** the P1.10 gate, byte for byte; for 10 decisions, every feature matches a
hand computation; across the reference day `active_now` × 12 steps 8 → 11 at 08:00, 11 → 12 at
10:00 and 12 → 11 at 17:00, with `active_30` and `active_60` leading by 30 and 60 min; total
observation length 139 at `K = 8`.

**Commit:** `RL-GYM: add Phase 2 global observation block`

---

### P2.12 — `VecNormalize` and the full-observation check

**Files:** `train_sdpdp.py`, `src/rl_gym/gym_env.py`, `tests/test_rl_gym_env.py`

- Wrap the `SubprocVecEnv` in `VecNormalize` per §5.3, `gamma` taken from the same config value
  as PPO's.
- Save/load helpers: `vecnormalize.pkl` beside each checkpoint; the evaluation loader sets
  `training = False` and `norm_reward = False`.
- The env raises `ValueError` on any non-finite observation value, at reset and at step.

**How Ritun verifies:** `check_env` through `MaskRespectingShim` on the full observation, with
`base_seed` absent and set; a save → load round trip reproduces identical normalised observations
for a fixed input; the P1.10 gate, byte for byte (`VecNormalize` sits outside the env, and the
gate runs the unwrapped env).

**Commit:** `RL-GYM: add VecNormalize wiring and non-finite observation guard`

---

### P2.13 — Training script

**Files:** `train_sdpdp.py`, `src/rl_gym/callbacks.py` (new), `tests/test_rl_gym_train.py` (new),
`src/rl_gym/sim_env_rl.py` and `src/rl_gym/gym_env.py` (the `tail_seconds` field below)

- `MaskablePPO` (`sb3-contrib`), `MlpPolicy`, `SubprocVecEnv` (D8), `VecNormalize` (P2.12),
  training days by D10 rotation (`day_dir = "corpus"`), `base_seed` required (the D7 guard
  stays). **Do not pass `seed=` to `MaskablePPO`.** In SB3 2.9.0 that calls `VecEnv.seed`, which
  hands worker `i` the seed `base_seed + i` at its next reset; the env would then seed its
  generator from that number and never use D7's `SeedSequence([base_seed, env_id])`, silently
  bypassing D7. Instead, before building the model, call
  `stable_baselines3.common.utils.set_random_seed(s)` in the parent process with
  `s = int(np.random.SeedSequence([base_seed, 2020]).generate_state(1)[0])`, so network
  initialisation and action sampling are reproducible too; reward weights from config (§6.1) — **all eight must be given
  explicitly, and the script raises if any is missing** rather than falling back to defaults;
  `skip_output = True`.
- Hyperparameters are config keys with these defaults: `gamma = 0.999` (§6.5),
  `gae_lambda = 0.95`, `learning_rate = 3e-4`, `n_steps = 1024` per env, `batch_size = 256`,
  `n_epochs = 10`, `clip_range = 0.2`, `ent_coef = 0.0`, `net_arch = [256, 256]`.
- `MaskableEvalCallback` on a separate evaluation `SubprocVecEnv` (D8) over `corpus_eval` in
  `eval_pairs` mode (P2.3), sharing the training `VecNormalize` statistics (synced,
  `training = False`). The pairs are a config list (P2.15 takes them from P2.14's seed file).
  Each evaluation must run every pair exactly once: VERIFY how SB3 spreads `n_eval_episodes`
  across the evaluation envs, split the pairs to match, and log the (day, seed) of every
  evaluation episode. Its size and frequency are config keys and small by default.
- TensorBoard, per episode: reward; the `episode_summary()` counts (pickups, declines,
  cancellations, no-shows, rejections, no-candidates, and `unclassified`, which must stay 0);
  day and seed; steps per second. Aggregated: the day-visit histogram (D10) and the
  candidate-list length distribution. Two more that must be logged:
  - **Slot-selection frequency** — how often the policy picks each slot `0..K`. Frequent
    selection of slot `K − 1` means `K` is binding: the truncation is cutting off candidates
    the policy wants, and `K` would need raising. Read it together with the candidate-list
    lengths from P1.10 and P2.4 (D3).
  - **The per-episode horizon margin and the tail duration**, from
    `episode_summary()["horizon_margin"]` (P2.0b) and `episode_summary()["tail_seconds"]`,
    which this item adds. **A negative margin is the normal case, not a warning**: under greedy
    it is negative on 79 of the 127 kept days, median −234 s (P2.4). What §6.4's branch 7 turns
    on is the **tail against its 14,400 s cap**, so that is the number to watch; greedy's
    longest tail anywhere in the corpus is 1,950 s, 13.5% of the cap.
    **`tail_seconds` is logged rather than derived from the margin**, because the derivation is
    only valid in one regime: `tail = ceil(−margin / 10) × 10 + 30` is exact on all 79
    negative-margin days and wrong on all 48 non-negative ones, where the tail is 0 s or 10 s
    against a predicted 30 s (P2.4). A quantity that is exact in one regime and wrong in
    another is precisely what gets misread later. The tail is measured in
    `RLImmediateDecisionsSimulation`, which owns the finalisation that runs
    `record_remaining_assignments`, and surfaced through `episode_summary()`.
    A trained policy can move this: accepting more marginal requests pushes dropoffs later and
    lengthens tails. **If a tail approaches the cap, branch 7 goes live and `w_horizon = 0.0`
    stops being a placeholder and becomes a real decision** — so this is watched from the first
    training run, not discovered afterwards.
- Checkpoints every N episodes with their `VecNormalize` statistics and run metadata, including
  every `base_seed` the run has used. Resuming from a checkpoint requires a `base_seed` that is
  not in that list, and the script raises otherwise: resuming with the same `base_seed` would
  restart D10's rotation and the episode-seed streams from the beginning, and replay sample paths
  already trained on. The rotation restarts at pass 0 of the new seed; the day-visit histogram
  shows the small imbalance that leaves.

**How Ritun verifies:** a 2-worker, ~8-episode run completes and TensorBoard shows every listed
scalar; a checkpoint reloads and, with `deterministic = True` on a fixed (day, seed) — `day` plus
`episode_seed` (P2.3) — reproduces the same action sequence as before saving; each worker's first-episode `random_seed` in
`episode_summary()` is the one D7 derives from `SeedSequence([base_seed, env_id])`, not
`base_seed + env_id`; two runs with the same `base_seed` produce identical first-update losses;
the script raises when any reward weight is missing, and when resuming with a `base_seed`
already used; no orphaned workers after `close()` (P1.11's check).

**Commit:** `RL-GYM: add MaskablePPO training script and callbacks`

---

### P2.14 — Evaluation harness: stock greedy baseline and KPI table

**Files:** `src/rl_gym/evaluate.py` (new), `src/rl_gym/kpis.py` (new),
`src/rl_gym/gym_env.py` (episode-summary file), `docs/RL_GYM_EVAL_SEEDS.csv` (new),
`tests/test_rl_gym_kpis.py` (new)

**The RL result is meaningless without the greedy baseline on the same held-out days.**

- **Seeds.** `eval_seeds_per_day` (default 10) per evaluation day, drawn once with
  `np.random.default_rng(np.random.SeedSequence(eval_seed_root))`, `eval_seed_root = 20261009`,
  as integers in `[0, MAX_EPISODE_SEED]` (`MAX_EPISODE_SEED = 2,508,742`, defined in
  `gym_env.py`; P1.9), days in sorted order. They are written to `docs/RL_GYM_EVAL_SEEDS.csv`
  once, never regenerated, and reused for every policy ever evaluated. The comparison is
  **unpaired**: rider decline, the diffusion random term and stochastic travel time all draw from
  one global stream (§2.5, P1.9), so two policies on the same seed share a sample path only until
  their first different decision. The seed list buys reproducibility, not common random numbers.
- **Greedy baseline (D9).** Stock FleetPy through `run_examples.run_single_simulation(
  scenario_parameters)` — the function `run_scenarios` calls once per row; `run_scenarios` itself
  takes only file paths and cannot accept overrides. Build `scenario_parameters` exactly as
  `run_scenarios` does for the reference scenario's active row (constant config plus row,
  `study_name`, `n_cpu_per_sim = 1`, and `evaluate`, `log_level` and `keep_old` as
  `run_scenarios` sets them), then override `day_dir_name = "corpus_eval"`,
  `rq_file = "<day>.csv"`, `random_seed = <seed>` and
  `scenario_name = f"greedy_{day}_s{seed}"`. The unique name is required: every run otherwise
  inherits the row's `example_depot_time_pool_irsonly_rs_42`, and `create_or_empty_dir` would
  wipe each run's output, and the reference run's (trap 1). Output on; one process per run.
  VERIFY that these overrides reach the simulation unchanged and that no scenario file is
  edited.
- **RL policy.** The env with `day_dir = "corpus_eval"`, `day` and `episode_seed` (P2.3) set to
  the pair, `skip_output = False`, the checkpoint and its `VecNormalize` statistics
  (`training = False`), deterministic actions. Keep the log level at `warning` (the P1.11
  logging caveat). When output is on, the env also writes `rl_episode_summary.json` — the
  `episode_summary()` dict — into its output directory; this item adds that to `gym_env.py`.
- **KPI script** reads only the output directory, identically for both policies:
  - served count;
  - mean wait (`pickup_time − rq_time`) over served;
  - mean in-vehicle detour ratio, `(dropoff_time − pickup_time) / direct_route_travel_time`, over
    served;
  - decline rate (declines / offers made);
  - post-match cancellation rate (cancellations / accepted offers);
  - no-show rate (no-shows / accepted offers);
  - operator rejections and no-candidate requests. Both leave `offers = "0:"` in
    `1_user-stats.csv`, so the output file alone cannot separate them: for env runs read the
    split from `rl_episode_summary.json`; for stock greedy, operator rejections are 0 by
    construction and every such row is a no-candidate request;
  - total fleet VKT = Σ `driven_distance` in `2-0_op-stats.csv` / 1000, as
    `evaluation/standard.py` computes `op_total_km`.

  Tabulated per day, per month and overall, as mean ± standard error over seeds.

**How Ritun verifies:** the P1.10 gate, byte for byte (this item edits `gym_env.py`); the greedy
baseline on (2024-06-20, seed 42) writes a `1_user-stats.csv` byte-identical to
`baseline_user_stats.csv`, and two baseline runs on different pairs leave both output
directories intact; the KPI script's counts on that run reconcile with
P1.10's `episode_summary()`; the env-driven greedy on the same pair gives an identical KPI row,
which is D9's premise.

**Commit:** `RL-GYM: add evaluation harness and KPI table`

---

### P2.15 — Short test training (Phase 2 exit gate)

**Files:** no code; report `docs/RL_GYM_PHASE2_RESULTS.md`

Purpose: prove that all the machinery works together, with repositioning off. **Not a
performance test, and not a learning test.** It is sized to exercise every component a few
times, nothing more: about 20–25 minutes on the M4 MacBook.

Setup: §6.1's Phase 2 weights (`w_decline = 0.1`); 4 workers; 12 episodes per worker, 48 in
total — about half of pass 0 over the 101 training days (D10); P2.13's default hyperparameters except `n_steps = 256` for this run only, so the update path runs about 20 times instead of about 5; an evaluation callback that fires at least 3 times; checkpoints at least twice. At the end, P2.14's harness on 5 evaluation days — 2024-01-11, 2024-03-19, 2024-06-20, 2024-08-13, 2024-10-01 — × the first 2 seeds of each from `docs/RL_GYM_EVAL_SEEDS.csv`, for the trained
checkpoint and for stock greedy on the same pairs. All 26 days and the full `eval_seeds_per_day` belong to real training runs.

**Pass criteria:**
1. The run completes; worker exit codes are 0; no orphaned workers.
2. `unclassified = 0` in every episode.
3. No non-finite observation (the env would have raised).
4. No masked action is ever selected.
5. No training day is visited twice, and the days visited are exactly those D10 assigns:
   worker `env_id` runs positions `env_id, env_id + 4, …` of pass 0's permutation, in that
   order. Pass boundaries are covered by P2.3's unit test, not here.
6. A reloaded checkpoint reproduces its deterministic actions on a fixed (day, seed).
7. The KPI table exists for both policies on identical (day, seed) pairs.
8. Every TensorBoard scalar from P2.13 is present.

**Reported, not judged:** slot-selection frequency (is `K` binding?); the
the horizon margin and the tail-to-cap distance (does branch 7 approach? — the cap is the
indicator, not the margin, and greedy's worst tail anywhere in the corpus is 13.5% of it);
throughput — steps per second and episode wall clock
under the full observation; peak RSS per worker; and the projected wall clock of a full-length
run. The learning curve is logged but not read: 48 episodes say nothing about learning, and it is not
used to size the weight sweep.

**Commit:** `RL-GYM: Phase 2 short test training and results`

---

### After Phase 2 — Ritun's steps, not work items

In this order:

1. **Stochastic travel-time routing fix** (§2.5): keep realized draws out of operator routing,
   regenerate the byte-gate baseline and the stock-greedy KPI baselines, re-run the gates and
   P2.4's sweep, and run a short training on the corrected simulator to size the weight sweep.
2. **Reward weight sweep** over `w_pickup`, `w_wait`, `w_reject`, `w_cancel` and `w_decline`,
   with repositioning off, to set the weights for full-length training. Its design — the rule
   for choosing a winner, validation days carved out of the training split (never the
   evaluation days), the per-point budget from a short training run on the simulator as it will be trained, 
   since P2.15 is too short to size it — is settled then. Only ratios to `w_pickup` affect the optimal policy,
   and `VecNormalize` rescales rewards anyway.
3. **Planner stop duration.** Ritun may change `op_const_boarding_time` from 300 s to 40–60 s.
   This is his decision; Claude Code does not touch it (§10).
4. **Turn on stock repositioning.** It needs a FleetPy zone system and forecast inputs.
5. **Regenerate both baselines**: the byte-gate file — keeping the current one under a new name,
   so the repositioning-off configuration stays testable — and the stock-greedy KPI runs. A
   single training run at the swept weights then checks that they carry over.
6. **Full-length training** (~7,000 episodes).

Steps 1, 3 and 4 change the simulation. They invalidate every earlier baseline comparison, and the
gates must be re-established against the new baselines.

---

## 8a. Environment note — import numpy first (temporary workaround)

**Every Phase 2 entry point — training and evaluation scripts, test files and notebooks — must
`import numpy` before importing `torch`, `stable_baselines3`, `sb3_contrib`, `ax` or anything
that pulls them in.** Otherwise the process dies immediately:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
Abort trap: 6
```

`torch` is pip-installed (`pypi_0`) while `numpy` and `scipy` come from conda-forge, so two
copies of `libomp` are linked and whichever loads second raises. Measured: `import torch` alone
aborts, `import stable_baselines3` alone aborts, `import torch, numpy` aborts;
`import numpy, torch` and `import numpy, stable_baselines3` are fine. The condition is present
in **both** `fleetpy_rl` and `fleetpy_Ax_client`, so it predates and is independent of the RL
work.

**Do not set `KMP_DUPLICATE_LIB_OK=TRUE`.** The OpenMP message itself warns it "may cause
crashes or silently produce incorrect results", and anything that perturbs the numeric stack
invalidates `baseline_user_stats.csv` and every byte-for-byte gate behind it. Import order
carries no such risk.

**This is a workaround, not the design, and it stays for all of Phase 2.** The proper fix —
installing `torch` from conda-forge so a single OpenMP runtime is linked — is P2.0, which Ritun
has deferred. If he revisits it, it is followed by **re-running the P1.10 gate**: if the gate
still matches, the import-order workaround comes out; if it does not, that is itself the
finding, and the current environment still works.

**How the ordering reaches the workers — via forkserver, not spawn.** `mp`'s default start
method on macOS is `spawn`, but `SubprocVecEnv` overrides it: it picks **`forkserver`** whenever
that is available, falling back to spawn only otherwise. Measured under P1.11:
`venv.processes[0]._start_method` is `forkserver`.

That changes the mechanism, though not the conclusion. `multiprocessing/forkserver.py` sets
`self._preload_modules = ['__main__']` by default and SB3 never overrides it, so the forkserver
process **imports the entry script once** — `forkserver.py` calls
`spawn.import_main_path(main_path)` — and every worker is then **forked from that image**.
Workers do not each re-import `__main__`. The numpy-first ordering is therefore established
exactly once, in the forkserver, and inherited by every worker.

The `if __name__ == "__main__":` guard is still required, for a reason specific to this path:
the preload *executes* the entry script top to bottom, under `run_name="__mp_main__"`
(`spawn.py`). Without the guard, `main()` would run inside the forkserver process itself.

**The "construct envs inside `_init()`" rule in P1.11 is unaffected and rests on something
else.** SB3 passes `CloudpickleWrapper(env_fn)` as a `Process` argument, so whatever crosses
the process boundary must pickle. A pre-constructed env holding a routing engine and open file
handles cannot, under forkserver or spawn alike. That requirement is about picklability, not
about how `__main__` is imported.

**An alphabetising linter or `isort` run will reintroduce a hard crash.** The import site
carries a comment saying so; keep it.

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
12. **The action mask and the `is_valid` observation feature must come from one source.**
    Both answer the same question — which of the `K` candidate slots is occupied — so both read
    `spaces.candidate_slot_validity` (P1.5). Computing them independently lets them drift, and
    **nothing would raise**: the policy would either see a valid slot it cannot select, or be
    able to select a slot the observation labels as padding, and training would quietly
    optimise against a misreported action set. The likely way this gets reintroduced is a later
    "simplification" of `is_valid` to an inline `k < len(candidates)` in `CandidateObserver`
    (P1.6), which looks equivalent and is the same expression the shared function evaluates —
    until one side changes.
13. **One classifier for reward and statistics.** `RewardTracker` and `RollingStatsTracker`
    must label every exit — decline, post-match cancellation, no-show, rejection,
    no-candidates — with the same §6.4 ladder, from one function. Two copies drift and nothing
    raises: the reward would charge as a decline what the statistics count as a cancellation.
14. **Observation code never mutates simulation state.** Read `veh_plans[vid]`; never re-time,
    reassign or edit it — re-time copies only (§5.2). Never write a `PlanRequest` field, such as
    `t_pu_latest`, to compute a feature: that changes insertion feasibility for every later
    request. Leave no new entries in the routing cache either (§5.2, exception (b)): until the
    §2.5 defect is fixed, a cached route changes what later planning reads. The P1.10 gate re-run
    in each observation item is what catches a violation.
15. **`crs.info` carries no comment.** It is read raw into `self.crs` and handed to pyproj, so
    §0's `# RL-GYM:` rule cannot apply to it. The change is recorded in §2.10 and its commit
    message instead.
16. **Artifacts are tied to the frozen corpus.** P2.6's manifest stores `corpus_hash`, computed
    over the training files' names and contents, and from P2.9 on the env raises at construction
    when `corpus/` no longer matches. A baseline computed
    over a different set of days produces plausible-looking anomalies that are simply wrong.

---

## 10. Out of scope

Do not implement, do not add config keys or spaces for. Do keep the code shaped so a second
decision point can be added without touching `SDPDPAssignmentEnv`.

- **Caching the routing engine or any simulation object across `reset()`.** Each episode
  constructs a fresh simulation; that is what keeps stochastic travel-time state, dynamic
  network state and RNG state from leaking between episodes, and `_started` makes a reused
  object unrunnable anyway (trap 7). Measured under P1.8, `reset()` takes **~0.46 s** (median
  over five; 1.07 s on the first, cold, then 0.43–0.47 s) against a ~38 s episode — **about 1.5%
  of episode wall clock.** Caching would trade a correctness risk for a saving that is
  invisible. Settled — do not implement it, do not make it config-switchable, and do not propose
  it when reporting wall-clock figures.
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
- **Phase 2 observation exclusions**, settled with Ritun: infeasible-plan features, a
  rejection-rate anomaly, an availability fraction separate from `idle_fraction`,
  plan-independent proximity (distance or ETA to pickup), slack to a latest-pickup bound, H3
  centroid coordinates, an `"h3"` position encoding, an `is_idle` companion flag, the ETA gap
  for cancelled riders, the request's direct travel time. Do not add them and do not add config keys for them.
- **`h3` as a dependency of anything under `src/`.** Neighbour relations come from P2.5's table.
- **Registering H3 as a FleetPy zone system.** The zones are an RL-internal lookup. A FleetPy
  zone system becomes relevant only when repositioning goes on, after Phase 2.
- **Repositioning, `op_const_boarding_time`, and the reward weight sweep.** All three are
  Ritun's steps after Phase 2 (§8), not work items. Stock repositioning stays off throughout
  Phase 2.
