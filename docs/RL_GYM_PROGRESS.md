# RL Gym — implementation progress log

Append-only. Earlier entries are never edited or deleted.
Protocol: `docs/SDPDP_GYM_SPEC_v2.md` §0.

---

## P1.1 — STARTED 2026-09-12 11:54
Files I will touch: dev/__init__.py, dev/misc/__init__.py, dev/misc/init_modules.py,
docs/RL_GYM_PROGRESS.md

## P1.1 — DONE 2026-09-12 11:59 — awaiting Ritun's verification

Files changed (all new; no existing file modified — `git diff --stat -- run_examples.py src/`
is empty):
- `dev/__init__.py` — empty, makes `dev` a regular package
- `dev/misc/__init__.py` — empty
- `dev/misc/init_modules.py` — the ten `add_*` functions. Import-free by design.
  `add_dev_simulation_environments` -> `{"RLImmediateDecisionsSimulation":
  ("src.rl_gym.sim_env_rl", "RLImmediateDecisionsSimulation")}`;
  `add_fleet_control_modules` -> `{"RLPoolingIRSOnly": ("src.rl_gym.fleetctrl_rl",
  "RLPoolingIRSOnly")}`; the other eight return `{}`.
- `docs/RL_GYM_PROGRESS.md` — this file, created (§0 requires it, did not exist)

Verification — all checks passed, nothing failed:

1. Registration, fresh process (18 assertions, all PASS):
   - `dev_content is not None`
   - all ten `add_*` reachable as attributes on `dev_content`
   - all ten `get_src_*()` getters called and returned without error
   - entry counts vs. the pre-P1.1 baseline, with exactly the two intended deltas:
     sim envs 5 -> 6, fleet control 9 -> 10; routing 8, request 12, repositioning 4,
     charging 1, pricing 2, fleet sizing 2, reservation 1, batch optimizers 2 all unchanged
   - `PoolingIRSOnly` still `("src.fleetctrl.PoolingIRSOnly", "PoolingInsertionHeuristicOnly")`
   - `ImmediateDecisionsSimulation` still
     `("src.ImmediateDecisionsSimulation", "ImmediateDecisionsSimulation")`
   - both RL entries present with the specified tuples
   - `load_fleet_control_module("PoolingIRSOnly")` still resolves to the real class
   - `load_fleet_control_module("RLPoolingIRSOnly")` raises `ModuleNotFoundError:
     No module named 'src.rl_gym'` — confirms the entries are lazy, as intended until P1.3/P1.4

2. Full greedy scenario (`run_examples.py`, `constant_config_depot_cali_sc_1.csv` +
   `example_depot_cali_sc_1.csv`, `random_seed = 42`), ~42 s:
   - `1_user-stats.csv` vs `docs/baseline_user_stats.csv`: **byte for byte identical**
     (`cmp` reports no differing byte; md5 `53879ea79cf211d700f670dee1173dd5` both; 117,056 bytes)
   - ran a second time to separate my change from run-to-run noise: user-stats identical
     across run 1, run 2 and the baseline
   - `docs/baseline_user_stats.csv` not written (mtime still Sep 5 15:26)

Two non-findings, checked rather than assumed:
- `3-0_op-dyn_atts.csv` differs between run 1 and run 2 with no code change in between,
  in exactly the two CPU-time columns (`user_requests` = `G_FCTRL_CT_RQU`, and
  `dyn_fleetsizing_strategy`); `sim_time` and `reservation_time_trigger` are identical.
  The file is wall-clock-nondeterministic by construction, unrelated to P1.1.
- `run_examples.py` line 214 asserts `"number users": 88`; the run reports 288. Pre-existing
  and stale — 88 is the upstream figure for `example_ir_only.csv` (commented out at line 209),
  left behind when the scenario was swapped to the cali pair. `run_examples.py` is untracked-clean
  and the user-stats output matches a baseline generated before `dev/` existed, so the
  simulation is unchanged.

VERIFY findings (§2.7, the only VERIFY items in P1.1):
- The ten `add_*` function names: **matched the spec** — exact names, all ten present, in the
  same order §2.7 lists them (src/misc/init_modules.py lines 35, 52, 73, 91, 104, 115, 126,
  137, 147, 158; `grep -c "dev_content\."` = 10)
- `dev.misc.init_modules` imported inside `try/except ModuleNotFoundError`: **matched the
  spec** — lines 4-8
- `add_*` calls sit inside the `get_src_*()` getters, not at import: **matched the spec** —
  each getter carries its own `if dev_content is not None:` block, so a bare import surfaces
  no problem
- The `add_dev_*` vs `add_*` naming inconsistency §2.7 flags: **matched the spec** — exactly
  two carry the `dev` infix (`add_dev_simulation_environments`, `add_dev_routing_engines`),
  the other eight do not
- Ten `get_src_*()` getters exist for P1.1's verification: **matched the spec** — with their
  own asymmetric naming (`_modules` for request / fleet control / batch optimizer,
  `_strategies` for repositioning / charging / pricing / fleet sizing / reservation)

No correction to §2 is needed; §2.7 is accurate as written.

Five facts found that §2.7 does not state (none contradict it):
1. `src/FleetSimulationBase.py:413` already imports `dev.fleetctrl.LinebasedFleetControl`,
   but only in the `else` branch of `if not operator_module_name == "LinebasedFleetControl"`
   (line 392). Creating `dev/` without `dev/fleetctrl/` leaves that path failing exactly as
   it did before. No scenario changes behaviour.
2. `src/misc/` has no `__init__.py` (implicit namespace package) while `src/__init__.py`
   exists. Created both `dev/__init__.py` and `dev/misc/__init__.py` as the spec specifies;
   `import_module` works either way and explicit packages are less fragile.
3. The `except ModuleNotFoundError` swallows a `ModuleNotFoundError` raised *inside*
   `dev/misc/init_modules.py` too — the symptom would arrive much later as
   `IOError: Fleet control module RLPoolingIRSOnly is invalid!`. Hence the import-free file,
   and a docstring recording the constraint.
4. `src/misc/init_modules.py:6` now prints "Loading modules from development content." on
   every import, in every process, including future `SubprocVecEnv` workers. Verified it goes
   to stdout only — it does not appear in `00_simulation.log` — and touches no output file.
5. `.gitignore` holds only `studies/example_study/results/`, so `dev/` is committable.
   Note that 83 `__pycache__` files are already tracked in this repo, so a bare `git add dev/`
   will pick up `dev/**/__pycache__/*.pyc`; consistent with existing repo convention, but
   Ritun may prefer `git add dev/**/*.py`.

Decisions:
- Asked: May I create the four files (`dev/__init__.py`, `dev/misc/__init__.py`,
  `dev/misc/init_modules.py`, `docs/RL_GYM_PROGRESS.md`)? Flagged that the P1.1 verification
  run re-runs `create_or_empty_dir` over
  `studies/example_study/results/example_depot_time_pool_irsonly_rs_42/`, which is gitignored
  and whose contents were already byte-identical to the docs baseline.
  Ritun: "Yes, go ahead."

---

## P1.2 — STARTED 2026-09-12 12:24
Files I will touch: src/misc/globals.py, src/FleetSimulationBase.py, src/demand/demand.py,
src/fleetctrl/FleetControlBase.py, docs/SDPDP_GYM_SPEC_v2.md, docs/RL_GYM_PROGRESS.md

## P1.2 — DONE 2026-09-12 12:46 — awaiting Ritun's verification

Files changed:
- `src/misc/globals.py` — added `G_SKIP_OUTPUT = "skip_output"` and `G_RL_MODE = "rl_mode"`
- `src/FleetSimulationBase.py` — cached `self.skip_output` / `self.rl_mode` in `__init__`;
  guarded `create_or_empty_dir`, `save_scenario_inputs()`, the `2_vehicle_types.csv` write,
  the op-stats write inside `record_stats` (clear kept outside the guard), and in `run()`
  `save_final_state()` and `evaluate()`; RL-mode-only once-per-process logging guard with a
  `NullHandler` and forced warning level under `skip_output`
- `src/demand/demand.py` — `self.skip_output` in `Demand.__init__`; guarded the `to_csv` in
  `save_user_stats`, keeping `self.user_stat_buffer = []` outside the guard
- `src/fleetctrl/FleetControlBase.py` — `self.skip_output` in `__init__`; guarded the `to_csv`
  in `record_dynamic_fleetcontrol_output`, keeping `self.dyn_output_dict = {}` outside the
  guard; guarded `self.repo.record_repo_stats()`
- `docs/SDPDP_GYM_SPEC_v2.md` — folded in the four corrections agreed this turn (below)
- `docs/RL_GYM_PROGRESS.md` — this entry

`record_remaining_assignments()` and `demand.record_remaining_users()` left unguarded, per
decision A.

Verification — everything passed; one probe of mine was wrong and was corrected, and one
pre-existing simulator behaviour was found (both below):

1. Default config (no `skip_output`), full run: `1_user-stats.csv` **byte for byte identical**
   to `docs/baseline_user_stats.csv` (`cmp` clean, md5 `53879ea79cf211d700f670dee1173dd5`), and
   all nine usual output files produced.
2. `skip_output = 1`, full run pointed at the populated results directory: no file deleted, no
   file added, no file modified (md5 + size snapshot before/after).
3. All four buffers empty after that run: `user_stat_buffer`, `op_output[0]`,
   `operators[0].dyn_output_dict`, and `repo` confirmed `None` so the fourth guarded write site
   is unreached in this scenario. Also asserted `operators[0].sim_vehicles[0].op_output is
   sim.op_output[0]` — the in-place `.clear()` does reach the vehicles' shared list.
4. Callback survival with output off: `record_user` fired 445 times and appended 445 rows,
   reconciling exactly with the 445 rows in `baseline_user_stats.csv`. `record_boarding` fired
   for 288 distinct rids, reconciling exactly with the 288 baseline rows carrying a
   `pickup_time`; the set difference is empty in both directions.
5. Logging. RL mode: a second construction in the same process adds no handler (1 -> 1) and the
   only handler is a `NullHandler`, so no log file is opened. Non-RL: two constructions in one
   process still rebuild the handler onto each scenario's own `00_simulation.log` (verified both
   files exist on disk), handler count stays at 1, and `FleetSimulationBase._logging_configured`
   is never set. Non-RL behaviour is unchanged.
6. All four edited modules byte-compile; grep for `to_csv` / `.write(` / `open(` across them
   shows every remaining write site sitting inside a guard.

A probe of mine that was wrong, not the code: I first asserted that `demand.rq_db` would still
be populated at the end of a run and that pickups could be counted off it. Both are false by
design — `rq_db` entries are deleted as each rider terminates (`demand.py` lines 255 and 276),
so an empty `rq_db` at the end is the expected state and `record_remaining_users` correctly
finds nothing left to sweep. Replaced with the direct invocation counting in check 4.

Pre-existing behaviour found, unrelated to P1.2 but material to P1.7: **`record_boarding` fires
twice for rid 50** — 289 calls for 288 distinct rids. Both calls are for vid 4 at the same
`pu_pos` (node 1391) and arrive through the same path
(`update_sim_state_fleets` -> `demand.record_boarding`), at `sim_time = 33032.898...` and then
at `sim_time = 33230`. The baseline CSV records `pickup_time = 33230.0`, so the later call wins:
`user_boards_vehicle` overwrites `pu_time` on each call. This is not caused by P1.2 —
`skip_output` touches no simulation logic, and the default-config run of this same code is
byte-identical to the pre-P1.2 baseline. It matters for P1.7 because §6.1 charges
`+w_pickup - w_wait * (pu_time - rq_time)/60` on `record_boarding`: without per-rid idempotency
`on_pickup` would pay rid 50 twice, and charging on the first call would use a `pu_time` that
never reaches the output file. §6.4 requires `on_exit` to be idempotent but says nothing about
`on_pickup`. Flagged for P1.7, not acted on here.

VERIFY findings:
- All three buffers still clear when writes are suppressed: **matched the spec**, and the
  spec's warning is exactly right — all three clears sit inside their write blocks, so each
  guard wraps the write statement only. Confirmed at runtime in check 3.
- Every file write in `src/FleetSimulationBase.py` covered by a guard: **matched the spec** —
  `to_csv` / `.write(` / `open(` finds lines 432, 448+449, 569, 620 (487 is commented out),
  i.e. the four locations and five statements the spec describes; 448 and 569 sit inside
  `save_scenario_inputs` / `save_final_state`, guarded at their call sites, and 620 is reachable
  only through `record_stats` (callers: `run()` and `record_remaining_assignments()`).
  **Two sites those greps do not find**, both now guarded and both recorded in the spec:
  `create_or_empty_dir` (reaches `os.makedirs` / `os.remove` / `os.rmdir`) and
  `logging.FileHandler(self.log_file)`, whose construction is what creates `00_simulation.log`.
- Nothing under `src/evaluation/` reads `2_vehicle_types.csv`: **DIFFERED** —
  `src/evaluation/standard.py:226` and `src/evaluation/standard_with_peak.py:230` both
  `pd.read_csv(os.path.join(output_dir, "2_vehicle_types.csv"))`. The spec's contingency holds:
  `evaluate()` is guarded by the same flag, so write and read are suppressed together and
  evaluation runs are unaffected. Spec updated to state the finding.

Spec changes made this turn (§7 P1.2):
- Files line now lists `src/demand/demand.py` and `src/fleetctrl/FleetControlBase.py`.
- "Do NOT guard" extended to `demand.record_remaining_users()`, with the argument.
- `self.repo.record_repo_stats()` added as a fourth write site to guard.
- `op_output[op_id]` shared-list note added: the clear must stay in-place `.clear()`.
- `create_or_empty_dir` and `logging.FileHandler` recorded as file-touching sites the
  prescribed greps miss.
- `src/evaluation/` VERIFY replaced with the finding.
- Logging guard scoped to RL mode, with the `run_examples.py` sequential-run reasoning.
- "How Ritun verifies" updated for the RL-mode-only logging guard.

Decisions:
- Asked: (A) guard `demand.record_remaining_users()` or not; (B) blanket vs RL-scoped logging
  guard; (C) P1.2's file list omits two files.
  Ritun: all three accepted as proposed — leave `record_remaining_users` unguarded, scope the
  logging guard to RL mode with non-RL behaviour bit-identical, update the Files line.
- Asked: leave `self.repo.record_repo_stats()` unguarded since `repo` is None here?
  Ritun: guard it too — "skip_output = 1 produces no files" should hold unconditionally.

---

## P1.3 — STARTED 2026-09-12 16:19
Files I will touch: src/rl_gym/__init__.py, src/rl_gym/fleetctrl_rl.py,
docs/SDPDP_GYM_SPEC_v2.md, docs/RL_GYM_PROGRESS.md

## P1.3 — DONE 2026-09-12 16:26 — awaiting Ritun's verification

Files changed — both new; **no existing source file modified** (`git diff --stat -- src/` is
empty; `RLPoolingIRSOnly` resolves through the P1.1 dev hook):
- `src/rl_gym/__init__.py` — package marker, deliberately import-free
- `src/rl_gym/fleetctrl_rl.py` — `PendingDecision` and
  `RLPoolingIRSOnly(PoolingInsertionHeuristicOnly)` with `build_assignment_context`,
  `commit_assignment_choice`, greedy-fallback `user_request`, `set_reward_tracker`, and the
  `_record_request_cpu_time` helper
- `docs/SDPDP_GYM_SPEC_v2.md` — corrections 6 and 7 and the CPU-timing decision (below)
- `docs/RL_GYM_PROGRESS.md` — this entry

Verification — everything passed:

1. **Byte-for-byte, the Phase 1 headline test.** Full day with `op_module = RLPoolingIRSOnly`,
   no gym: `1_user-stats.csv` **identical to `docs/baseline_user_stats.csv`** (`cmp` reports no
   differing byte; md5 `53879ea79cf211d700f670dee1173dd5`; 117,056 bytes).
2. **Byte-for-byte again with a reward tracker attached.** Re-ran the same day with a stub
   tracker on `set_reward_tracker`: still identical, same md5. This is what proves the two
   label hooks are pure bookkeeping that does not perturb the simulation.
3. **Request reconciliation, exact:** 436 decisions + 0 reservation-branch + 9 empty-candidate
   + 0 same-origin-destination + 0 duplicate-rid = **445**. Every request in exactly one bucket.
   The 9 empty-candidate cases match the figure in §2.8.
4. **Tracker labels:** `note_no_candidates` fired for exactly 9 rids
   (79, 85, 86, 220, 264, 266, 332, 341, 372) — the same 9 as bucket 3, and P1.7's expected
   count. `note_rejection` fired 0 times under greedy, as P1.7 expects.
5. **Per-decision invariants asserted on all 436 decisions:** slot 0 is the `argmin`; the
   candidate list is sorted ascending by `delta_cfv`; no vid appears twice, so `vid -> plan` is
   a function. Candidate list length: min 1, median 5, mean 4.60, max 11, with 13 lists longer
   than K=8 (relevant to P1.5 truncation).
6. **Paths greedy never reaches**, on a truncated run driven by a scripted policy: 20
   rejections via `choice=None` produced `note_rejection` exactly 20 times, left no rejected rid
   in `tmp_assignment`, and yielded no real offer; 41 assignments at the **last** candidate
   index (deliberately not slot 0) committed without error.
7. **Bounds assertion** in `commit_assignment_choice` rejects `choice` of 2 and 5 against a
   2-candidate list and rejects -1, each with the rid in the message.
8. `_reward_tracker` defaults to `None`, so non-RL scenarios are unaffected.

VERIFY findings:
- `insertion_with_heuristics` side effects on `fleetctrl` state: **DIFFERED from "none" — one
  exists, and it is benign.** `insert_prq_in_selected_veh_list` calls
  `veh_plan.set_utility(current_vehplan_utility)`, mutating `fleetctrl.veh_plans[vid]` by
  caching the utility when it was `None`. No restoration is needed: `build_assignment_context`
  calls `insertion_with_heuristics` exactly once per actionable request at the same point the
  parent calls it, so the mutation is identical. Nothing touches `rid_to_assigned_vid` and
  nothing reassigns `veh_plans`.
- `op_vpi_nr_plans` (`G_VPI_KEEP`) unset so at most one plan per vid: **matched the spec, and
  more strongly than it assumed.** The key is absent from the constant config and is never
  added to `rv_heuristics` (`FleetControlBase.py` lines 174–203), *and*
  `insert_prq_in_selected_veh_list` defaults `nr_plans_per_vehicle` to 1 and truncates
  `keep_plans[:1]`. So `vid -> plan` is a function unconditionally, not merely by config.
  Confirmed empirically in check 5.

Three facts checked beyond the VERIFY list, all de-risking the byte gate:
- **No RNG anywhere in the insertion path.** `random` is imported at `PoolingIRSOnly.py:3` but
  never used, and `insertion.py` / `searchVehicles.py` contain no `random` reference at all.
  The §0 failure mode "the build/commit split consumes global RNG differently" cannot arise
  through this path.
- `rv_heuristics` is `{}` in the reference scenario (the run banner prints `RV Heuristics: {}`),
  so no post-insertion truncation fires and the returned list is the complete sorted list.
- `min(list_tuples, key=lambda x: x[2])` is exactly `list_tuples[0]`: the sort in `insertion.py`
  is a stable ascending `sorted()` and `min` returns the first minimum on ties. D3 holds.

Spec changes made this turn:
- §1 and §4: "duplicate rid" removed — three internal-resolution cases, not four, because the
  parent has no duplicate-rid branch (lines 109–111 overwrite `rq_dict[rid_struct]`).
- P1.3: same correction; plus the `G_FCTRL_CT_RQU` bookkeeping now stated to apply to the
  reservation and empty-candidate branches only, the `o_pos == d_pos` branch returning before
  it; plus `cpu_t0` renamed `cpu_elapsed` with the CPU-time accounting decision recorded.
- P1.10: duplicate-rid bucket kept, expected 0, with a note that non-zero means the source
  changed.

Note: P1.3 verification left two gitignored result directories,
`studies/example_study/results/p13_rl_greedy_check/` and `p13_tracker_greedy_check/`. Both are
reproducible and safe to delete. `example_depot_time_pool_irsonly_rs_42/` was not touched and
still matches the baseline.

Decisions:
- Asked: (6) the spec lists a duplicate-rid internal-resolution case the parent does not have;
  (7) the `o_pos == d_pos` branch skips the CPU-time bookkeeping, contradicting "including the
  `G_FCTRL_CT_RQU` bookkeeping".
  Ritun: both corrections accepted. Remove "duplicate rid" from §1, §4 and P1.3; keep P1.10's
  bucket and note that a non-zero value means the source changed. Do not add bookkeeping to the
  `o_pos == d_pos` branch; amend P1.3 to scope the bookkeeping to the other two branches.
- Asked: should commit time the whole build-to-commit span, folding in agent latency, or sum
  the two computation spans and exclude the wait?
  Ritun: sum the spans, exclude the wait — `G_FCTRL_CT_RQU` should measure fleet-control
  computation, not policy inference plus scheduler latency, and it surfaces in evaluation runs
  where the contamination would be worst. Rename `PendingDecision.cpu_t0` to `cpu_elapsed`, a
  duration rather than a timestamp, with commit adding its own span before writing.

---

## P1.4 — STARTED 2026-09-12 17:00
Files I will touch: src/rl_gym/sim_env_rl.py, docs/SDPDP_GYM_SPEC_v2.md,
docs/RL_GYM_PROGRESS.md

## P1.4 — DONE 2026-09-12 17:07 — awaiting Ritun's verification

Files changed — one new source file; **no existing source file modified**
(`git diff --stat -- src/` is empty):
- `src/rl_gym/sim_env_rl.py` — `RLImmediateDecisionsSimulation` with `step_generator`,
  `run_generator` and the `_rl_operator` guard
- `docs/SDPDP_GYM_SPEC_v2.md` — §2.2 undecided-traveller finding, P1.4 Files line, the
  try/finally scoping reason, the raise-don't-fall-back rule, the two presentation deviations
- `docs/RL_GYM_PROGRESS.md` — this entry

Verification — everything passed:

1. **Byte-for-byte.** A driver calling `run_generator()` and sending `0` at every yield, with
   `sim_env = RLImmediateDecisionsSimulation` and `op_module = RLPoolingIRSOnly`, no gymnasium:
   `1_user-stats.csv` **identical to `docs/baseline_user_stats.csv`** (`cmp` reports no
   differing byte; md5 `53879ea79cf211d700f670dee1173dd5`).
2. **436 gym steps**, matching P1.3's 436 decisions exactly. Slot 0 asserted to be the `argmin`
   and the candidate list asserted non-empty at every yield.
3. **Wrong operator raises.** With `op_module = PoolingIRSOnly` the generator raises `TypeError`
   on the first `next()`, and the message names both `RLImmediateDecisionsSimulation` and
   `PoolingIRSOnly`. No silent fallback to `user_request`.
4. **`close()` mid-episode is clean and cheap.** After 6 steps, `close()` raised nothing,
   returned in 0.000 s, left the generator exhausted, and — the point of the try/finally
   scoping — ran **neither** `record_remaining_assignments` nor `record_remaining_users`.
5. **Reject through the generator:** `send(None)` committed 8 rejections without error.
6. **`_started` guard:** a second `run_generator()` on a used simulation object yields nothing
   (trap 7).
7. `src/rl_gym/sim_env_rl.py` byte-compiles.

Timing figures, for P1.10's results file: 436 steps in 37.93 s wall, 11.5 steps/second;
inter-step wall gaps min 0.0068 s, median 0.0759 s, mean 0.0864 s, p95 0.1664 s, max 0.4497 s.

VERIFY findings (both from §2.2, checked here because P1.4 is where `step()` is mirrored):
- Interleaved per-request processing as claimed: **matched the spec.** The loop at
  `ImmediateDecisionsSimulation.py` ~line 91 completes `user_request`, `get_current_offer`,
  `receive_offer` and `_rid_chooses_offer` for one request before starting the next, so the
  heuristic does re-run against an updated plan for a same-timestamp arrival. The proposal's
  §3.1 requirement is satisfied natively; no change needed.
- Whether `get_undecided_travelers` is ever non-empty: **verified empirically — it is not.**
  Over a full day it was empty at all 4,480 steps, and all 445 requests reached `user_request`
  exactly once. No rid produces a second decision epoch. Recorded in §2.2 along with *why*:
  the guarantee is config-dependent, resting on `check_sim_env_spec_inputs` enforcing
  `user_max_decision_time == 0`; a non-zero value would let riders stay undecided across steps
  and silently produce two epochs for one rid.

One implementation detail worth recording: `_end_realtime_plot()` is **not** called
unconditionally in the `finally`. The parent calls it as the last statement of `run()`, after
finalisation, so calling it right after the loop would reorder teardown relative to
`record_stats` / `evaluate` on the normal path. It is instead guarded by a `completed` flag —
the `finally` tears the plot down only on an abort (GeneratorExit or an escaping exception),
and the normal path calls it at the end exactly where `run()` does. With
`realtime_plot_flag = 0` in the reference scenario both are no-ops, so this is about not
leaving a latent ordering bug for whoever enables plotting.

Spec changes made this turn:
- §2.2: the interleaving item recorded as verified; the undecided-traveller VERIFY replaced
  with the finding, the mechanism, and the `user_max_decision_time == 0` dependency.
- P1.4: Files line reduced to `src/rl_gym/sim_env_rl.py`; the `load_simulation_environment`
  contingency dropped; try/finally scoping and its reason added; the raise-don't-fall-back rule
  added; the tqdm and timing-report suppression recorded.
- D3 and P1.10: `K = 8` confirmed against the measured candidate-length distribution
  (436 decisions, min 1, median 5, mean 4.60, max 11, 13 exceeding K), recorded as the figure
  P1.10 must reproduce.
- P2.6: slot-selection frequency must be logged — frequent selection of slot 7 means `K` is
  binding.

Decisions:
- Asked: (1) P1.4's Files line lists `src/misc/init_modules.py`, which the P1.1 dev hook makes
  unnecessary; (2) tqdm and the timing report across a thousand episodes; (3) what belongs in
  the try/finally.
  Ritun: all three accepted. Files line is `src/rl_gym/sim_env_rl.py` only and the
  `load_simulation_environment` contingency is dropped; suppress tqdm and the report under
  `skip_output` and keep them otherwise; loop in the try with `_end_realtime_plot` in the
  finally and finalisation only after normal completion, with the reason noted — an aborted
  episode must not advance the simulation past `end_time`.
- Asked: raise or fall back when the operator is not RL-capable?
  Ritun: raise, naming both the `sim_env` and the `op_module`. A silent fallback means training
  runs greedy while reporting that it is learning, and it would surface weeks later as a
  trained policy matching the baseline exactly.
- Ritun: `K` stays at 8, justified by the P1.3 distribution. Record it in P1.10's results file
  and note in P2.6 that slot-selection frequency should be logged.

---

## P1.5 — STARTED 2026-09-12 21:12
Files I will touch: src/rl_gym/spaces.py, tests/test_rl_gym_spaces.py,
docs/SDPDP_GYM_SPEC_v2.md, docs/RL_GYM_PROGRESS.md

Note: this session was interrupted by a usage limit after P1.5 was approved but before any
file was written. Resume check found P1.4 committed with a matching DONE, no P1.5 STARTED
entry, and no P1.5 code — the item never began, so there was nothing to salvage. The one
uncommitted file at that point was the P1.10 gap-distribution spec amendment, which Ritun
committed separately as 8909652.

## P1.5 — DONE 2026-09-12 21:16 — awaiting Ritun's verification

Files changed — two new; **no existing source file modified** (`git diff --stat -- src/` is
empty):
- `src/rl_gym/spaces.py` — `make_action_space`, `reject_action`, `truncate_candidates`,
  `candidate_slot_validity`, `build_action_mask`. Pure functions, no state; `K` is passed in,
  since it is owned by `SDPDPAssignmentEnv` (P1.8).
- `tests/test_rl_gym_spaces.py` — 29 stdlib `unittest` tests. `tests/` created.
- `docs/SDPDP_GYM_SPEC_v2.md` — P1.6 `is_valid` requirement and trap 12
- `docs/RL_GYM_PROGRESS.md` — this entry

Verification — **29 tests, all pass** (`python -m unittest discover tests`, 0.008 s):

The three cases P1.5 names:
1. Truncation preserves input order exactly — including a tied-`delta_cfv` list in an order no
   sort would produce, and a deliberately *descending* list that must come back descending.
   That is the direct test that no sort or tie-break is applied (D3, trap 10).
2. Fewer than `K` candidates: validity is `[True]*n + [False]*(K-n)` and the mask masks only
   the surplus slots.
3. `K` larger than the candidate count leaves the list unchanged; `K` smaller takes a prefix of
   the first `K`, so the dropped candidates are always the most expensive ones.

Beyond those: exactly `K` candidates; empty list; `n_candidates` of 0 leaving reject as the
only legal action; reject always legal across `n = 0..14`; mask never all-`False` across
`n = 0..14` and `K in {1, 2, 8, 20}`; mask shape `(K+1,)` and dtype `bool`;
`ValueError` on `k_max <= 0` and negative `n_candidates`, `TypeError` on non-int arguments
(including `bool`, which is an `int` subclass and would otherwise slip through);
`truncate_candidates` returning a new list so a caller's mutation cannot reach back into
`PendingDecision.candidates`.

Three invariants later work items depend on, asserted here so a drift fails loudly:
- **mask[:K] is exactly `candidate_slot_validity(n, K)`** across `n = 0..14` — trap 12's
  single-source requirement, checked rather than merely documented.
- **validity agrees with truncation length** across `n = 0..14`: exactly the slots truncation
  fills are the valid ones, and no others.
- **every unmasked slot `k < K` is a legal index into the truncated list** — the same condition
  `RLPoolingIRSOnly.commit_assignment_choice` asserts, so the mask can never offer an action
  that commit would reject.
- `len(build_action_mask(n, K)) == make_action_space(K).n` for `K in {1, 2, 8, 20}` — an
  off-by-one here would silently shift which slot MaskablePPO thinks it is choosing.
- The candidate lengths actually observed on the reference day (1, 4, 5, 8, 9, 11) all behave,
  covering both sides of `K = 8`.

Also confirmed `import src.rl_gym.spaces` pulls in only `src.rl_gym` and itself — no FleetPy
fleet-control stack — so the unit tests stay fast and P1.8 can import it cheaply.

VERIFY findings: **P1.5 contains no `VERIFY` items** (confirmed by grep over the work item's
spec text: zero matches). Nothing to record.

Spec changes made this turn:
- P1.6: `CandidateObserver` must derive `is_valid` from `spaces.candidate_slot_validity`, never
  from a fresh `k < len(candidates)`, with a pointer to trap 12.
- New trap 12: the action mask and the `is_valid` feature must come from one source; two
  independent computations can drift with nothing raising, and the likely reintroduction is a
  later "simplification" of `is_valid` to an inline `k < len(candidates)` in `CandidateObserver`
  — an expression that looks equivalent and is in fact what the shared function evaluates,
  until one side changes.

Decisions:
- Asked: `pytest` is absent from the `fleetpy_rl` environment and from `env_fleetpy_rl.yml`, and
  the repo has no existing test suite; P1.5's Files line names no test file.
  Ritun: use stdlib `unittest`, no new dependency. `tests/` and the separate file name
  `tests/test_rl_gym_spaces.py` are both fine; keep `tests/test_rl_gym.py` for P1.10's rollout
  gate.
- Asked: put `translate_action` in `spaces.py` next to the mask, or leave it to P1.8?
  Ritun: leave it to P1.8 — agreed, to avoid bundling two work items.
- Ritun: the mask/`is_valid` coupling goes into the spec now rather than at P1.6, so it does not
  exist only in chat if a session ends.

---

## P1.6 — STARTED 2026-09-12 22:30
Files I will touch: src/rl_gym/observers.py, tests/test_rl_gym_observers.py,
docs/SDPDP_GYM_SPEC_v2.md, docs/RL_GYM_PROGRESS.md

## P1.6 — DONE 2026-09-12 22:40 — awaiting Ritun's verification

Files changed — two new; **no existing source file modified** (`git diff --stat -- src/` is
empty):
- `src/rl_gym/observers.py` — `AbstractObserver` (with `observe`, `output_length`,
  `observe_vector`), `signed_log1p`, `CandidateObserver` (3*K), `GlobalStateObserver` (4)
- `tests/test_rl_gym_observers.py` — 37 stdlib `unittest` tests
- `docs/SDPDP_GYM_SPEC_v2.md` — §5.1 rewritten per the decisions below; §5.2 availability-count
  note
- `docs/RL_GYM_PROGRESS.md` — this entry

Verification:

**Unit tests: 66 pass across the suite** (37 new for P1.6, plus P1.5's 29), 0.011 s.
Covering `signed_log1p` oddness, strict monotonicity, and that `argmin` over the transform
agrees with `argmin` over raw `delta_cfv` on 200 random draws (P1.10 depends on that);
`is_valid` sourced from `candidate_slot_validity` for every `n` in 0..11; padded slots all-zero
across all three features; the offered-wait formula against the `_create_user_offer`
computation; the normalizer read from `scenario_parameters` rather than hardcoded (checked by
swapping `op_max_wait_time` to 4200, trap 8); truncation preserving order; `time_sin`/`time_cos`
as absolute clock time including the midnight wrap, and distinct from episode progress;
progress endpoints, midpoint, and clipping past `end_time`; `idle_fraction` excluding
`OUT_OF_SERVICE` from the denominator, counting only `VRL_STATES.IDLE` and not `WAITING` /
`PLANNED_STOP` / `REPO_TARGET`, and guarding the divide for an all-out-of-service and an empty
fleet; observer lengths summing to `3*K + 4` for `K` in 1, 4, 8, 16.

**Full scripted greedy day, 436 decision epochs, every check passed:**
- observation length 28 = `3*K + 4` at every epoch, all identical
- dtype `float32` everywhere
- **no NaN and no inf** across all 12,208 scalars
- slot 0's `delta_cfv` is the minimum over valid slots at all 436 epochs
- `is_valid` equals the action mask's candidate half at all 436 epochs (trap 12, enforced
  rather than assumed)
- episode progress within [0, 1]; offered wait non-negative on every valid slot

Observed feature ranges over the day: `is_valid` 0–1 (mean 0.568); `delta_cfv` −14.508 to 0.000
(mean −7.851); `offered_wait` 0.000–0.9999 (mean 0.398); `time_sin` −1.000 to 0.965; `time_cos`
−1.000 to 0.259; `day_progress` 0.0013–0.9643; `idle_fraction` 0.000–1.000 (mean 0.190).

**A premise in the spec turned out to be false, and it is one added this same turn.** Real
`delta_cfv` values on this scenario are **negative** — transformed range [−14.508, 0.000], the
0.000 end being padding, so the raw most-negative value is about −2.0e6. Inserting a request
*improves* the objective, because an unserved request carries a penalty, and stock's
`min(list_tuples, key=...)` therefore picks the most negative delta. Consequently **padding at
0.0 is the largest value in the slot, not the smallest**, and a naive unrestricted `argmin`
over all `K` slots would still land on slot 0.

That inverts the stated rationale in two places: P1.10's "padded slots, whose `delta_cfv` is
padding rather than a cost and would win", and the sentence added to §5.1 this turn, "a padded
`delta_cfv` of 0.0 wins a naive `argmin` over positive real costs". The **decision** is
unaffected — 0.0 remains the right padding value, on the `VecNormalize` argument alone — and the
**instruction** to restrict the scripted driver to `is_valid` should stand, because it is
correct regardless of sign and the sign is a property of this objective function rather than of
the design. But the justification is wrong and the spec should not keep it. A side effect worth
naming: P1.10's byte-for-byte gate would pass even with the `is_valid` restriction removed, so
that test does not actually protect against a padding bug on this scenario. Correction proposed
to Ritun, not yet applied.

**`OUT_OF_SERVICE` vehicles are real and frequent on this day**, which makes the dead-code
finding consequential rather than cosmetic: of 12 vehicles, between 0 and 4 are out of service
at a decision epoch (mean 0.45), and **147 of the 436 epochs have at least one**. Had the
observer used the upstream `== 5` comparison, feature 4's denominator would have been a constant
12 and the feature wrong at a third of all decisions.

VERIFY findings — P1.6 carries no `VERIFY` markers of its own, but it depends on three §5.1
source claims, all checked:
- `pax_info[rid][0] - prq.rq_time` as the offered wait: **matched the spec, and it is the
  authoritative formula.** `PoolingIRSOnly._create_user_offer` (~line 365) unpacks
  `pu_time, do_time = assigned_vehicle_plan.pax_info.get(prq.get_rid_struct())` and quotes
  `pu_time - prq.rq_time`. Feature 3 reproduces it exactly.
- `VRL_STATES` values: **matched the spec.** `IDLE = (0, "idle")`,
  `OUT_OF_SERVICE = (5, "out_of_service")`, with `WAITING (4)`, `PLANNED_STOP (6)` and
  `REPO_TARGET (7)` all distinct, as feature 4 requires.
- `veh_search_for_immediate_request` skipping `OUT_OF_SERVICE` at `searchVehicles.py` ~line 21:
  **DIFFERED — that line is dead code.** `VRL_STATES` is a plain `Enum` whose members hold
  `(int, str)` tuples and which defines no `__eq__` against ints, so
  `VRL_STATES.OUT_OF_SERVICE == 5` is unconditionally `False` (verified at runtime);
  `veh_obj.status` is always an enum member (`Vehicles.py` ~line 68). The same inert comparison
  appears at `searchVehicles.py` ~line 90 and `insertion.py` ~line 414. The **live** filter is
  `simple_insert` at `insertion.py` ~line 36, which uses the correct enum comparison and returns
  early, so an out-of-service vehicle yields no insertion and never becomes a candidate. §5.1's
  conclusion stands; only its citation was wrong. This is upstream FleetPy, not this fork.
  Ritun verified the finding independently.

Spec changes made this turn:
- §5.1: features 1–2 and 3 documented as different quantities, with `t_of_day` as absolute clock
  time and feature 3 as episode progress, plus the note that the circular encoding is convention
  here because this day never wraps midnight.
- §5.1: feature 4's justification recited to `simple_insert` (`insertion.py` ~36) as the live
  filter, with an explicit note that `searchVehicles.py` ~21, ~90 and `insertion.py` ~414 are
  inert comparisons that have never fired, that this is upstream FleetPy rather than a defect
  introduced here, and that "fixing" them would change which vehicles are considered and break
  every byte-for-byte comparison.
- §5.1: feature 4 recorded as a coarse fleet-utilisation signal rather than a true availability
  count, because `simple_insert` also skips `no_show_event` vehicles.
- §5.1: padding documented as 0.0 for all three features of an unoccupied slot. **The
  `VecNormalize` half of that rationale holds; the `argmin` half is disproven above and needs
  correcting.**
- §5.2: a proper availability count added to the global block — the fraction of the fleet that
  could actually receive the request, excluding both `OUT_OF_SERVICE` and `no_show_event`, with a
  note that the two notions diverge exactly while a no-show is being waited out.

Decisions:
- Asked: is `t_of_day` absolute clock time or episode fraction, given §5.1 lists `sin`/`cos` of
  it *and* "fraction of the day elapsed"?
  Ritun: different quantities, as read — `sin`/`cos` over `(sim_time mod 86400)/86400` for
  absolute clock time, feature 3 as episode progress for the D6 horizon signal. Note that the
  `sin`/`cos` pair is convention here since this day never wraps midnight.
- Asked: what value pads an invalid slot?
  Ritun: 0.0 for all three features; beyond the `argmin` argument, a large sentinel would distort
  `VecNormalize`'s per-feature running statistics in Phase 2.
- Asked: feature 4's denominator is broader than the set of vehicles that could actually serve
  the request, because `simple_insert` also excludes `no_show_event` vehicles.
  Ritun: implement §5.1 as written and record the narrowing; note the proper availability count
  in §5.2's global block.
- Ritun: correct §5.1 to cite `simple_insert` as the live filter and state explicitly that the
  three `== 5` comparisons are inert upstream code, so it does not later read as our bug.

## P1.6 — FOLLOW-UP 2026-09-12 23:09 — corrections after Ritun's verification

Ritun approved correcting the two passages whose reasoning P1.6 disproved, plus the D3 ordering
language, plus a test with teeth.

Files changed:
- `docs/SDPDP_GYM_SPEC_v2.md`
  - §5.1 padding: the `argmin` rationale removed. `VecNormalize` is now stated as the whole
    justification for 0.0, with the measured fact recorded — real `delta_cfv` is negative
    (transformed range [−14.508, 0.000] over 436 decisions, raw minimum about −2.0e6), so
    padding at 0.0 is the largest value in the slot, not the smallest.
  - P1.10: the rationale replaced. The `is_valid` restriction is required **independent of the
    sign of `delta_cfv`**, the sign being a property of this objective function
    (`distance_and_user_times_with_walk` penalises unserved requests, so serving one improves
    the objective) and changeable with a different control function, scenario, or Phase 2
    shaping. Records explicitly that the byte-for-byte gate would pass with the restriction
    removed and so does not by itself protect against a padding bug.
  - D3: "the `k`-th cheapest candidate" replaced with exact ordering language — ascending by
    `delta_cfv`, element 0 the minimum and most negative, each later slot larger, and **a more
    negative `delta_cfv` is a larger improvement to the objective**. "Cheapest" is now avoided
    deliberately and the reason is stated. The truncation note no longer says "most expensive
    candidates" but "the largest `delta_cfv` values, i.e. the weakest insertions".
  - P1.10 additionally now requires its driver to select through a **tested helper** rather than
    an inline `np.argmin`, with a test that fails if the restriction is dropped — "a guard with
    no test that fails when it is removed is decoration".
- `tests/test_rl_gym_observers.py` — new `TestPaddingMustNotCompeteInArgmin`, 6 tests.

Verification: **72 tests pass** (66 before, 6 new), 0.011 s.

The new tests supply the counter-example the real scenario does not, using positive `delta_cfv`
in the valid slots with 0.0 padding: that padding is then strictly the minimum; that an
unrestricted `argmin` lands on a **padded** slot and is wrong; that the restricted `argmin`
returns slot 0; that this holds for every valid-slot count 1..K, with the unrestricted form
disagreeing whenever a padded slot exists; that the restricted form is also correct under the
real scenario's negative costs, where the two coincide — so the restriction is harmless as well
as necessary; and the single-valid-slot case.

**Scope note, stated plainly:** these 6 tests pin the *contract and the counter-example*. They do
not yet make a removed guard fail, because the guard lives in P1.10's driver, which does not
exist. That is why the spec now requires P1.10's driver to go through a tested helper. The direct
"remove the restriction, a test fails" coverage lands in P1.10.

---

## P1.7 — STARTED 2026-09-12 23:54
Files I will touch: src/rl_gym/reward.py, src/demand/demand.py,
tests/test_rl_gym_reward.py, docs/SDPDP_GYM_SPEC_v2.md, docs/RL_GYM_PROGRESS.md

## P1.7 — DONE 2026-09-13 00:04 — awaiting Ritun's verification

Note on the working tree: the P1.6 FOLLOW-UP changes above (spec corrections plus
`TestPaddingMustNotCompeteInArgmin`) were not committed before P1.7 began, so they sit in the
tree alongside this item. Last commit is 6baf6c1 "RL-GYM: add minimal observation builders".

Files changed:
- `src/rl_gym/reward.py` — **new.** `RewardTracker` with `on_pickup`, `on_exit`,
  `note_rejection`, `note_no_candidates`, `flush`, `episode_summary`, plus `classify` and
  `DEFAULT_REWARD_WEIGHTS` (the §6.1 values).
- `src/demand/demand.py` — **modified**, +23 lines, three `# RL-GYM:` sites: the two callback
  attributes defaulting to `None` in `Demand.__init__`, the exit hook in `record_user`, and the
  boarding hook at the end of `record_boarding`. Both hooks are `None`-guarded and wrapped in
  `try/except` so a reward bug can never kill a simulation.
- `tests/test_rl_gym_reward.py` — **new**, 34 stdlib `unittest` tests.
- `docs/SDPDP_GYM_SPEC_v2.md` — §6.4 eighth branch and its notes; §2.4 corrections.
- `docs/RL_GYM_PROGRESS.md` — this entry.

Verification — everything passed:

**Unit tests: 106 pass across the suite** (34 new for P1.7), 0.011 s. Covering the §6.1 weight
values; unknown-weight keys rejected so a config typo cannot silently do nothing; the pickup
term; last-write-wins across a repeat, including one that straddles a `flush` boundary; the
dedupe-on-first counter-example (stale 92.9 s vs realized 290 s); every §6.4 branch in order,
with each earlier branch shown to win against all later flags set; `no_show` gated on
`pu_time is None` and checked after `diffusion_cancelled`; `chosen_operator_id = 0` counting as
accepted, since operator 0 is falsy and the check must be `is not None`; no fallback to rider
decline; `on_exit` idempotent across three calls; labels charging nothing until `on_exit`;
`w_horizon = 0` counted but not charged, and charged when non-zero; `flush` summing and
resetting its window; `episode_reward` surviving flushes.

**Full scripted greedy day with both callbacks wired, all 13 reconciliation checks exact:**

| | tracker | baseline |
|---|---|---|
| pickups (distinct rids) | 288 | 288 |
| served | 288 | 288 |
| rider declined | 118 | 118 |
| diffusion cancelled | 28 | 28 |
| no candidates | 9 | 9 |
| no-show | 2 | 2 |
| operator rejected | 0 | 0 |
| **unclassified** | **0** | 0 |
| unserved at horizon | 0 | 0 |
| duplicate boardings | 1 | 1 |
| boarding calls | 289 | 289 |
| exits | 445 | 445 |
| terminal outcomes sum | 445 | 445 |

**Byte-for-byte, both ways.** `demand.py` is an existing file every scenario uses, so both
paths were re-run: the RL greedy day **with** the tracker and both callbacks attached is
identical to `docs/baseline_user_stats.csv` (md5 `53879ea79cf211d700f670dee1173dd5`), and the
**stock non-RL** `run_examples.py` scenario with the callbacks left `None` is identical too.
The hooks perturb nothing, attached or not.

**Independent arithmetic check of the reward.** Recomputing the episode total straight from
`baseline_user_stats.csv` — 288 pickups at `w_pickup`, less `w_wait` times the summed wait, less
118 declines, 28 cancellations, 2 no-shows and 9 no-candidates at their weights — gives
**116.510947**, against the tracker's **116.510947**, difference 0.000000000. Mean wait 588.7 s
over 288 pickups, consistent with the 0.398 mean normalized offered-wait P1.6 measured against
`op_max_wait_time = 1500`.

**Flush accounting (D6).** 436 flushes for 436 gym steps, 263 of them non-zero; the flush sum
equals `episode_reward` exactly and `unflushed_reward` is 0.0, so the final flush did capture
the episode tail — the events `record_remaining_assignments` and `record_remaining_users`
produce after `end_time`.

VERIFY findings:
- Which of `Demand` / `SlaveDemand` `ImmediateDecisionsSimulation` instantiates:
  **matched the spec — `Demand`.** `_load_demand_module` takes the `SlaveDemand` branch only
  when `sim_env == "MobiTopp"`; this project's is `RLImmediateDecisionsSimulation`.

Two facts recorded in §2.4 alongside it, both found while wiring:
- `record_user` is defined **only** on `Demand`, so the single exit hook covers both classes.
- `record_boarding` **is** overridden by `SlaveDemand` (~line 363), so the boarding hook on
  `Demand.record_boarding` does not cover that override. Correct by scope here, but a silent gap
  for anyone who later runs MobiTopp: they would get exit events and no pickup events at all.

§2.4 correction: the dead `user_cancels_request` is on **`SlaveDemand`**, not `Demand` —
demand.py ~line 355, inside `class SlaveDemand(Demand)`, and the line number had also drifted
from 350 after P1.2. The conclusion is stronger than the spec stated: dead twice over, since
nothing calls it *and* its class is never instantiated here. Confirmed no caller anywhere.

Six live `record_user` paths confirmed, matching §6.4's count exactly:
`FleetSimulationBase.py` ~756 (undecided leaves system), ~781 (chose < 0), ~921 (diffusion
cancellation), `demand.py` ~253 (`record_no_show`), ~271 (`record_alighting_start`, the success
path), ~282 (`record_remaining_users`). All five §6.4 request attributes exist on `RequestBase`:
`no_show` (~66), `diffusion_cancelled` (~80), `rider_declined` (~90), `pu_time` (~116),
`do_time` (~119), plus `chosen_operator_id` (~112) for the new branch 7.

Spec changes made this turn:
- §6.4 now has **eight** branches, with `pu_time is None and chosen_operator_id is not None ->
  unserved at horizon, -w_horizon` inserted at 7 and unclassified moved to 8.
- §6.4 records why the charge lives there rather than in an env-side end-of-episode sweep: a
  second charging site would contradict the rule that all reward is charged once in `on_exit`,
  and it keeps branch 8 meaning strictly "a path was missed".
- §6.4 records that branch 7 is **unreachable on the reference day** and why that is expected
  rather than evidence it is dead — 0 unclassified, 288 pickups against 288 dropoffs, so
  `record_remaining_assignments` completes every in-flight trip — and when it becomes live: a day
  where the `end_time + 14400` cap binds, or a policy that over-commits.
- §2.4: `user_cancels_request` reattributed to `SlaveDemand` with the stronger conclusion; the
  `Demand`-is-instantiated VERIFY replaced with the confirmed finding; the `record_user`-only-on-
  `Demand` and `record_boarding`-overridden-by-`SlaveDemand` facts recorded.

Decisions:
- Asked: §6.1 charges `-w_horizon` at "end of episode" but §6.4's classification has no such
  branch, so an accepted-but-never-picked-up rider falls into "unclassified — charge nothing",
  which §6.4 says means a path was missed. Explicit branch, or a separate env-side sweep?
  Ritun: take the recommendation — explicit branch between no-show and unclassified, making
  §6.4 eight branches. He confirmed `chosen_operator_id` independently: initialized `None` at
  `TravelerModels.py:112`, set only inside `choose_offer` on acceptance; diffusion-cancelled
  baseline rows carry 0 while declined and no-candidate rows are empty, so the attribute means
  what is needed and earlier branches catch everyone who accepted and then failed another way.
  He also asked that the argument against the env-side sweep be kept in §6.4, and that branch 7's
  unreachability on this day be recorded so a future reader does not assume it is dead.
- Ritun: §2.4 correction accepted, with the stronger conclusion recorded, plus the
  `SlaveDemand.record_boarding` override noted as a silent gap for MobiTopp.

---

## P1.8 — STARTED 2026-09-13 01:02
Files I will touch: src/rl_gym/gym_env.py, tests/test_rl_gym_env.py,
docs/SDPDP_GYM_SPEC_v2.md, docs/RL_GYM_PROGRESS.md

## P1.8 — DONE 2026-09-13 01:10 — awaiting Ritun's verification

Files changed — two new; **no existing source file modified** (`git diff --stat -- src/` empty):
- `src/rl_gym/gym_env.py` — `SDPDPAssignmentEnv` and `derive_study_name`
- `tests/test_rl_gym_env.py` — 25 tests, including the `MaskRespectingShim` used only by the
  `check_env` test
- `docs/SDPDP_GYM_SPEC_v2.md` — §5.1 Box bounds; P1.8 collision and shim resolution; §10
  caching entry
- `docs/RL_GYM_PROGRESS.md` — this entry

Verification — **131 tests pass across the suite, no failures and no skips**, 37.8 s
(the env tests build real simulations).

- `check_env` **passes** through the shim, in 5.3 s.
- `reset()` returns the declared shape `(28,)` and dtype `float32`, in-space.
- 10 valid actions step without error; reject is legal at every step; mask is `(K+1,)`, bool,
  never all-False, and agrees slot-for-slot with the observation's `is_valid` block.
- `action_masks()` returns the same array across repeated calls without advancing the
  simulation.
- 20 reset/close cycles leak no file handles (`psutil.num_fds`, tolerance 2).
- Termination: driving reject to the horizon gives `terminated=True`, `truncated=False` (D6),
  a zeros observation that is in-space, `episode_summary` in `info`, and a mask with reject as
  the only legal action.
- `close()` idempotent and safe before `reset()`; `step()` before reset and after termination
  both raise `RuntimeError`; out-of-space actions raise `ValueError`.

**The masked-action assert is genuinely exercised.** My first version of that test skipped —
the first decision of the day happens to fill all 8 slots, so there was no masked slot to send.
A skipped test on the central guard is exactly the decoration problem, so the test now advances
until it finds a decision with a masked slot, then asserts on **every** masked slot rather than
just the first, with a complementary test that valid slots (including the last valid one, not
only slot 0) do not assert.

**`reset()` wall clock, measured for the §10 decision:** 1.068 s on the first, cold reset, then
0.455 / 0.460 / 0.431 / 0.469 s — **median 0.46 s, mean 0.577 s** over five. Against a ~38 s
episode that is **~1.5% of episode wall clock**, materially below the 2–4 s that had been
assumed. Recorded in §10: caching the routing engine or any simulation object across `reset()`
is out of scope, because the saving is invisible and the correctness risk — stochastic
travel-time state, dynamic network state and RNG state leaking across episodes — is not.

**`check_env`'s seed-determinism check passes vacuously at P1.8.** Until P1.9 wires per-episode
seeding, `reset()` accepts the `seed` argument for API conformance but builds the simulation
with the scenario row's own `random_seed`, so two resets with the same seed match trivially
rather than because seeding works. P1.9's verification — same seed giving identical
trajectories, different seeds giving different output — is therefore **not** redundant with this;
it is the first real test of seeding.

VERIFY findings:
- `ConstantConfig` / `ScenarioConfig` API and the config-addition idiom: **matched the spec.**
  `ConstantConfig` is a `dict` subclass loading CSV or YAML by extension, so plain
  `cfg[key] = value` is the idiom; `__add__` returns a **new** `ConstantConfig` via
  `{**self, **other}`, non-mutating, right operand winning; `ScenarioConfig` is a `list` of
  `ConstantConfig` rows whose `read_csv` passes `comment="#"`, which is what drops the nine
  disabled rows so the reference scenario file yields exactly one.

Two facts the idiom does not make obvious, both handled:
- **`study_name` must be derived, not hardcoded.** `run_scenarios` computes
  `basename(dirname(dirname(abspath(path))))` and `get_directory_dict` depends on it.
  `derive_study_name` reproduces it and was confirmed against four paths **outside**
  `studies/example_study` — an absolute path under a different study, a path with no `studies`
  segment at all, a path containing `..`, and a `.yaml` — matching `run_examples`' own
  computation on every one. Hardcoding was the failure mode caught during planning; deriving it
  wrongly in the other direction would have been invisible on the reference scenario alone.
- **`n_cpu_per_sim` is mandatory**, read as `scenario_parameters["n_cpu_per_sim"]` in
  `FleetControlBase.__init__` with no default. Set to 1, as P1.8 specifies.

Spec changes made this turn:
- §5.1: the observation space declared as `Box(low=-inf, high=inf, shape=(3K+4,),
  dtype=np.float32)`, with the reason — `delta_cfv` has no a priori bound even after
  `sign * log1p`, and bounding belongs in Phase 2's `VecNormalize(clip_obs=10.0)`.
- P1.8: the `check_env`-versus-assert collision stated outright, with the measurement
  (24 of 40 runs failed, nondeterministically) and the shim resolution, plus explicit
  instructions not to resolve it by clamping to reject or by a masking-off config flag.
- §10: caching the routing engine or any simulation object across `reset()` added as out of
  scope, with the measured 0.46 s figure.

Decisions:
- Asked: P1.8 requires both "assert on masked actions" and "`check_env` passes", which exclude
  each other — `check_env` samples uniformly from `Discrete(K+1)` ignoring the mask. Measured
  24/40 failures on a stand-in.
  Ritun: the collision is real and was his; keep the assert and run `check_env` through the
  test-only shim; the argument against clamping is decisive, since a policy bug appearing as a
  preference for rejecting riders is the silent corruption this design keeps engineering out.
  Amend P1.8 so the next reader does not try to reconcile the two requirements again.
- Asked: terminal observation, P1.8/P1.9 split, and `info` contents.
  Ritun: all three accepted, including zeros for the terminal observation — that was in v1 and
  was lost in v2.
- Ritun: declare the Box bounds in §5.1; report the `reset()` wall clock and write the §10
  caching entry with the measured figure, but report and stop instead if it came out materially
  above a few seconds; record the vacuous seed-determinism point so P1.9 does not later look
  redundant; and confirm the `study_name` derivation against a path outside
  `studies/example_study`.

---

## P1.9 — STARTED 2026-09-13 01:27
Files I will touch: src/rl_gym/gym_env.py, tests/test_rl_gym_env.py,
docs/SDPDP_GYM_SPEC_v2.md, docs/RL_GYM_PROGRESS.md

## P1.9 — DONE 2026-09-13 01:39 — awaiting Ritun's verification

Files changed — only RL-owned files (`git diff --stat -- src/` touches `src/rl_gym/` alone):
- `src/rl_gym/gym_env.py` — `MAX_EPISODE_SEED`, `_worker_base_seed`, `_apply_episode_seed`,
  `_apply_episode_scenario_name`; `reset()` seeds and names the episode; `episode_summary()`
  reports `random_seed`, `env_id` and `episode`
- `tests/test_rl_gym_env.py` — `TestSeeding` (11) and `TestScenarioNameUniqueness` (2), plus a
  second `check_env` test with `base_seed` set
- `docs/SDPDP_GYM_SPEC_v2.md` — D7 reworded; P1.9 seed range, the `episode_summary` seed field
  and the uniqueness-key caveat; P1.11 must raise on a missing `base_seed`
- `docs/RL_GYM_PROGRESS.md` — this entry

**A real bug the tests caught, and it would not have been obvious later.** My first
implementation drew episode seeds from `[0, 2**31)`. Seven tests then failed with
`ValueError: Seed must be between 0 and 2**32 - 1` raised from **inside simulation
construction**, at `demand.py` line 78. `Demand.load_demand_file` does
`np.random.seed(int(1712 * np_random_seed))` — it **multiplies the seed by 1712** before using
it, and `load_parcel_demand_file` (~line 132) does the same. The usable range is therefore
`[0, (2**32 - 1) // 1712]` = `[0, 2_508_742]`, not the full 32-bit range. Fixed by deriving
`MAX_EPISODE_SEED` from that expression rather than picking a literal, with the derivation in a
comment. Had this shipped, training would have crashed partway into an episode with a message
pointing at demand loading rather than at seeding — roughly 99.9% of draws from `[0, 2**31)`
exceed the bound, so it would have failed almost immediately, but for a reason that reads as
someone else's fault.

Verification — **145 tests pass across the suite**, 75.3 s, no failures and no skips.

Seeding (11 tests):
- `base_seed` absent leaves `G_RANDOM_SEED` at the row's **42**, even when `reset(seed=999)` is
  passed — this is what keeps P1.10's byte gate comparing like with like
- `base_seed` set gives a distinct seed every episode, none equal to 42, all within
  `MAX_EPISODE_SEED`
- same `reset(seed=...)` → same episode seed; different → different
- **same seed → identical trajectories**: rid sequence, `sim_time`, candidate counts and the
  full reward sequence compared exactly over 40 steps
- **different seeds → different trajectories**
- four `env_id`s with one `base_seed` give four distinct streams; `_worker_base_seed` is
  reproducible for the same `(base_seed, env_id)` and differs across `env_id`
- `MAX_EPISODE_SEED` bound asserted directly: `1712 * MAX_EPISODE_SEED` is accepted by
  `np.random.seed` and `1712 * (MAX_EPISODE_SEED + 1)` raises
- `episode_summary()` reports `random_seed`, `env_id`, `episode`

File-level, with `skip_output = 0` and truncated episodes:
- same reset seed → **identical `1_user-stats.csv`** (md5 `5e00f189…` twice)
- different reset seed → **different `1_user-stats.csv`** (`9a04a9ce…`), episode seeds 335672
  vs 1934854
- successive resets on one env write `…_ep1`, `…_ep2`, `…_ep3` — three distinct directories
- four distinct `env_id`s write four distinct directories at the same episode index
- all directories cleaned up afterwards; no strays left under `results/`

`check_env` (3): passes through the shim with `base_seed` absent **and** with `base_seed` set.
Recording why both were run: with `base_seed` absent the seed-determinism check passes
vacuously, since `reset()` leaves `G_RANDOM_SEED` alone; only with `base_seed` set does it
actually exercise the seeding path. That was the open point carried over from P1.8, now closed.

**A check of mine that was wrong, not the code.** My first file-level script rolled out three
*separate env instances* and asserted they would write three distinct directories. They all
wrote `…_ep1`, because `episode_counter` starts at 1 in each instance — by design. The real
claims are that successive resets on one env differ, and that distinct `env_id`s differ; both
verified after correcting the script.

That did surface a genuine edge, now recorded in P1.9's spec text: the uniqueness key is
`(env_id, pid, episode counter)`, so **two env instances sharing an `env_id` in one process do
collide** on their first episode. Verified deliberately. It requires the `DummyVecEnv` pattern
D8 forbids; under `SubprocVecEnv` each worker is its own process with its own `env_id`.

VERIFY findings: P1.9 carries no `VERIFY` markers. The source facts it depends on were checked
during planning and all held — every stochastic component (rider decline, diffusion random
term, no-show, initial vehicle state and SoC, stochastic travel time) draws from the global
`np.random` stream that `FleetSimulationBase.__init__` seeds from `G_RANDOM_SEED`, so writing
that key before construction is sufficient and complete. The 1712 multiplier above is the one
thing that planning missed and testing caught.

Spec changes made this turn:
- **D7 reworded.** It said "Per-episode reseeding is mandatory", which under this change would
  have contradicted P1.9 — a DECIDED section against a work item, which §0 says to stop on. It
  now says reseeding is mandatory *for training* and is switched on by `base_seed`; with
  `base_seed` absent the scenario row's seed is used unchanged, because the byte-for-byte gates
  in P1.3, P1.4 and P1.10 compare against a baseline generated at seed 42. The reason is stated
  so nobody restores unconditional reseeding and breaks the exit gate.
- D7 also records that the default which is safe for reproducibility is unsafe for training —
  absent `base_seed`, a real run silently replays one sample path with nothing erroring — and
  names the two guards: `train_sdpdp.py` must raise, and the env reports the seed in
  `episode_summary()`.
- P1.9: the `SeedSequence([base_seed, env_id])` idiom, the `[0, MAX_EPISODE_SEED]` range with
  the 1712 reason, the `episode_summary` seed field, and the uniqueness-key caveat.
- P1.11: must raise if `base_seed` is missing rather than defaulting it.

Decisions:
- Asked: if `reset()` always overwrites `G_RANDOM_SEED`, P1.10's byte-for-byte gate cannot pass,
  since the baseline was generated at seed 42. Make `base_seed` the switch, with absent meaning
  "do not reseed"?
  Ritun: yes — the only resolution that keeps P1.10 testing the plumbing. The D7/P1.10 collision
  was his, written without checking the two were compatible. `SeedSequence([base_seed, env_id])`
  agreed as numpy's intended idiom. Amend D7 in the same turn, since a DECIDED section
  contradicting a work item is a stop condition; note that the reproducibility-safe default is
  training-unsafe, with `train_sdpdp.py` raising and the seed recorded in `episode_summary`; and
  exercise `check_env`'s seed-determinism check in both modes.

---

## P1.10 — STARTED 2026-09-13 01:57
Files I will touch: src/rl_gym/policies.py, tests/test_rl_gym.py,
tests/test_rl_gym_observers.py, docs/RL_GYM_PHASE1_RESULTS.md,
docs/SDPDP_GYM_SPEC_v2.md, docs/RL_GYM_PROGRESS.md

## P1.10 — DONE 2026-09-13 02:08 — awaiting Ritun's verification

# # # THE PHASE 1 EXIT GATE PASSES, BYTE FOR BYTE. # # #

Files changed — no existing source file modified (`git diff --stat -- src/` empty):
- `src/rl_gym/policies.py` — **new.** `greedy_action(obs, mask, k_max)` and `greedy_choice`
- `tests/test_rl_gym.py` — **new.** The exit gate, 7 tests
- `tests/test_rl_gym_observers.py` — `TestGreedyActionHelper`, 9 tests
- `docs/RL_GYM_PHASE1_RESULTS.md` — **new**, all figures below
- `docs/SDPDP_GYM_SPEC_v2.md` — P1.10 Files line and the `policies.py` rationale; P2.7 must
  import the same helper; §6.4 branch-7 correction (below)
- `docs/RL_GYM_PROGRESS.md` — this entry

### The gate

`1_user-stats.csv` from a scripted greedy rollout driven entirely through the Gym API is
**identical to `docs/baseline_user_stats.csv`** — md5 `53879ea79cf211d700f670dee1173dd5` both,
117,056 bytes, zero differing bytes. `base_seed` absent so `G_RANDOM_SEED` stayed at 42 (D7).

Reconciliation, exact: 436 decisions + 0 reservation + 9 empty-candidate + 0 same-origin-
destination + 0 duplicate-rid = **445**.

Reward counts reconcile against the baseline again through the gym path: pickups 288, served
288, declined 118, diffusion-cancelled 28, no-candidates 9, no-show 2, operator-rejected 0,
**unclassified 0**, exits 445, duplicate boardings 1, `episode_reward` 116.510947 with
`unflushed_reward` 0.0 and the flush sum matching it exactly.

Throughput: 436 steps in 38.00 s, 11.47 steps/second.

### Gaps between decision epochs

Simulated seconds, all gaps: min 0.0, median 60.0, mean 99.2, p95 300.0, max 660.0.
**Zero-gap count reported separately as its own figure: 119 of 435, 27.4%.** Those windows
carry no simulated time at all — two requests in the same time step — so they can contain only
the action's own immediate consequence. Excluding them: min 60.0, median 120.0, mean 136.5,
p95 360.0, max 660.0. Two distinct regimes, which is what matters for P2.5.

Wall clock, for throughput only: min 0.0071, median 0.0771, mean 0.0872, p95 0.1707,
max 0.4494 s.

### The `record_remaining_assignments` tail — and a correction it forced

**The tail adds zero simulated seconds, with zero riders in flight at `end_time`.** The cap
(`end_time + 14400`) is 0% used, so it is nowhere near binding.

The instrumentation confirmed `record_remaining_assignments` was entered, so this is a real
measurement, not a missed hook. The reason is that the day finishes its work early: last
request at `rq_time` 68400, latest pickup 69064.2, latest dropoff 69805.3, all below
`end_time` 70000. The baseline has no row with a pickup or dropoff past 70000.

**This corrects a claim I made in the P1.7 entry and wrote into §6.4.** I had said 288 pickups
against 288 dropoffs meant "`record_remaining_assignments` completes every in-flight trip".
That inference was wrong. The tail completes *nothing*, because nothing is in flight. Same
conclusion — nobody stranded — but a different mechanism, and the difference matters: the
margin is only about **195 simulated seconds** between the last dropoff and `end_time`. A later
demand tail, heavier load, longer trips, or a policy that defers pickups would push work past
`end_time` and the tail would start doing real work. §6.4 now states the measured reason and
records that `w_horizon = 0.0` is a placeholder resting on that 195 s margin, not a settled
decision.

So, to Ritun's question about the tail being a surprise at scale: **on this day it costs
nothing**, and the thing to watch is the 195 s margin rather than the tail itself.

### Candidate lengths

436 decisions, min 1, median 5, mean 4.60, max 11, 13 exceeding `K = 8` — reproducing P1.3
exactly through a different code path.

### The tested helper

`greedy_action` lives in `src/rl_gym/policies.py`, one file and one function, with the
`obs[K:2K]` layout dependency stated in its docstring. Nine tests, beside
`TestPaddingMustNotCompeteInArgmin`: it picks slot 0 on sorted negative and sorted positive
lists; **an unrestricted `argmin` lands on a padded slot and disagrees** under positive costs;
it selects by value rather than assuming index 0, so it does not silently depend on
`insertion.py`'s ordering; it rejects when no candidate is valid; the returned action is legal
under the mask for every candidate count 0..11; shape mismatches raise; and the `delta_cfv`
slice is pinned so a reordering of `CandidateObserver`'s dict fails here rather than silently
selecting on `is_valid` or `offered_wait`.

Suite total: **161 tests pass**, 114.9 s, no failures and no skips. No stray result directories.

VERIFY findings: P1.10 carries no `VERIFY` markers. Its reconciliation rests on §2.8's 445
requests and P1.3's bucket counts, both re-established independently here through the gym API.

Decisions:
- Asked: where should the tested greedy helper live — `observers.py` (owns the layout),
  `spaces.py`, or a new `policies.py`?
  Ritun: `policies.py`. The deciding factor is P2.7: the greedy baseline for the RL comparison
  is this same function, and if it diverged from the greedy P1.10 tested, the headline result
  would be quietly invalid. One file, one function, layout dependency in the docstring rather
  than implied by co-location; add it to P1.10's Files line and note in P2.7 that it imports
  from there.
- Ritun: report the zero-gap count as its own figure rather than folded into the distribution,
  since same-timestamp windows are a distinct regime; and if the tail approached +14400, treat
  it as a finding and say so plainly. It did not — it is 0.

---

## P1.11 — STARTED 2026-09-13 02:28
Files I will touch: train_sdpdp.py, docs/RL_GYM_PHASE1_RESULTS.md,
docs/SDPDP_GYM_SPEC_v2.md, docs/RL_GYM_PROGRESS.md

## P1.11 — DONE 2026-09-13 02:43 — awaiting Ritun's verification

Files changed:
- `train_sdpdp.py` — **new.** `make_env`, `check_config`, `smoke_test`, CLI. Skeleton only.
- `src/rl_gym/gym_env.py` — one line: `log_level` accepted from the env config, defaulting to
  `"warning"` exactly as before. Needed for the log-separation check (below); a parameter with a
  safe default rather than a changed default.
- `docs/RL_GYM_PHASE1_RESULTS.md` — P1.11 section appended
- `docs/SDPDP_GYM_SPEC_v2.md` — new §8a on the import-order workaround
- `docs/RL_GYM_PROGRESS.md` — this entry

Verification — four workers, one episode each, random legal actions:

| | |
|---|---|
| start method | **forkserver**, not `spawn` — SB3 prefers it where available |
| construct 4 workers | 1.01 s |
| wall clock, four episodes | 61.4 s (vs 38.0 s for one alone: 1.6× for 4× throughput) |
| per-worker steps | 443 / 443 / 444 / 443 |
| peak worker RSS | 1.61 GB total, 0.40 GB per worker |
| worker exit codes | `[0, 0, 0, 0]` |
| orphaned workers after `close()` | **none** |
| log files | four separate, 292–296 KB, **no interleaving**, none truncated |

Each worker drew a distinct seed and reported `unclassified = 0`. Under random actions they
rejected 76–85 and picked up 196–205 against greedy's 288, so the env behaves sensibly off the
greedy path. Full suite still **161 tests pass**, including the P1.10 byte gate, after the
`gym_env.py` change.

**A measurement of mine that was wrong, corrected.** My first run reported two orphaned
processes after `close()`. They were not workers: they were multiprocessing's own
`resource_tracker` and `forkserver` helpers, which live as long as the parent. The real workers
had already exited with code 0. I had been diffing "all new child PIDs" instead of reading the
worker PIDs from `venv.processes`. Fixed to take the PIDs from SB3 and to report the helper
count separately, so the distinction is visible rather than silently folded in. This also
surfaced that SB3 chooses **forkserver**, which the spec had assumed was `spawn`.

**The log check was vacuous until I forced it.** At the env's default `log_level = "warning"`
each worker creates its log file but writes nothing to it, because nothing warns — and an empty
file cannot demonstrate non-interleaving. Added `log_level` as an env config key (default
unchanged) and re-ran at `info`: each log then carries 3800+ lines and, since the format is
`%(process)d-…`, the PID prefix identifies the writer. Every file contains **exactly one**
distinct PID, its own. No cross-writing, and all four end on a complete newline-terminated line.

Recorded as a caveat rather than fixed: in RL mode the P1.2 logging guard configures handlers
once per process, so a *second* episode in the same worker with output on would keep logging
into the first episode's directory. Training uses `skip_output = 1` and writes no logs, and
P1.9's unique `scenario_name` separates the user-stats regardless, so this only affects someone
debugging a multi-episode run with output enabled.

**The OpenMP blocker, and the environment answer Ritun asked for.**

Importing `stable_baselines3` or `torch` aborts the process outright —
`OMP: Error #15: … libomp.dylib already initialized`, `Abort trap: 6`. Isolated to pure import
order: `import torch` alone aborts, `import stable_baselines3` alone aborts, `import torch,
numpy` aborts; `import numpy, torch` and `import numpy, stable_baselines3` are fine. torch is
pip-installed while numpy and scipy come from conda-forge, so two `libomp` copies are linked.
Worked around by importing numpy first, with a loud comment at the import site; **not** with
`KMP_DUPLICATE_LIB_OK=TRUE`, which OpenMP itself warns may silently produce incorrect results.
Recorded in §8a as temporary, to be removed once torch comes from conda-forge and P1.10 has
been re-verified.

On whether the clone disturbed the original environment: **it did not.**
- `fleetpy_Ax_client` currently holds numpy 2.2.6, torch 2.10.0, botorch 0.16.1, gpytorch
  1.15.2 — identical to `fleetpy_rl`, and self-consistent, since botorch 0.16.1 requires
  torch ≥ 2.x and could not run on 1.13.1.
- The `torch==1.13.1` pin lives only in `env_fleetpy_linux.yml` and `fleetpy_backup.yml`, the
  legacy files the spec header says predate the RL work. `env_fleetpy_rl.yml` correctly records
  2.10.0. So the original was upgraded as a set at some earlier point, not by the clone.
- The duplicate-libomp condition is present in **both** environments, so it predates and is
  independent of the RL work.
- `AxClient` (used by the calibration scripts) imports fine. The top-level
  `from ax import optimize` API that `MOBO_for_micro.py` and `SOBO_for_micro.py` use no longer
  exists in this Ax version — those two are already broken independently of OpenMP and of
  anything done here.

**One live footgun found while checking: `conda run -n fleetpy_Ax_client` silently runs
`fleetpy_rl`'s interpreter** in this shell — `sys.prefix` comes back as
`/opt/anaconda3/envs/fleetpy_rl`. My first probes of the original environment were therefore
invalid and were redone with `/opt/anaconda3/envs/fleetpy_Ax_client/bin/python` directly. Use
the direct interpreter path for anything meant to run in the Ax environment, or work in an
activated shell.

VERIFY findings: P1.11 carries no `VERIFY` markers.

Decisions:
- Asked: handle the OpenMP conflict by import order, or fix the environment first?
  Ritun: import-order workaround now; the reasoning against `KMP_DUPLICATE_LIB_OK` is right,
  since "silently produce incorrect results" is the failure mode this whole design exists to
  prevent. The environment has passed ten byte-for-byte gates, which is empirical proof its
  numpy reproduces the baseline, and reinstalling torch or numpy risks moving numpy and
  breaking that with no visible cause. The proper fix — torch from conda-forge — happens as its
  own task before Phase 2 training, followed by re-running the P1.10 gate; if it still matches,
  the workaround comes out. Record the workaround as temporary, with a loud comment and a spec
  note.
- Ritun: record the logging caveat as a debugging annoyance rather than a correctness issue,
  and confirm the clone did not disturb `fleetpy_Ax_client`.
- Ritun (carried from P1.10): add the per-episode last-dropoff-to-`end_time` margin to P2.6's
  logging plan, since a trained policy can move it and closing it makes `w_horizon` a real
  decision. Added.

## P1.11 — FOLLOW-UP 2026-09-13 02:53 — forkserver correction, Phase 1 closed

Ritun flagged that §8a's reasoning assumed `spawn` re-importing `__main__`, while SB3 actually
uses forkserver, which forks from a template. He was right, and the mechanism I had written was
wrong even though the conclusion held. Traced it to the source rather than reasoning about it:

- `subproc_vec_env.py`: `start_method = "forkserver" if forkserver_available else "spawn"` —
  SB3 **overrides** `mp`'s default, and forkserver is available on macOS. Confirmed at runtime:
  `venv.processes[0]._start_method == "forkserver"`.
- `multiprocessing/forkserver.py` line 40: `self._preload_modules = ['__main__']` by default,
  and SB3 never calls `set_forkserver_preload`. Line 173 calls `spawn.import_main_path(...)`.
- `multiprocessing/spawn.py` lines 286–290: the entry script is executed under
  `run_name="__mp_main__"`, which is what makes the `if __name__ == "__main__":` guard False
  inside the forkserver.

So the ordering reaches the workers by a different route than I wrote: **the forkserver imports
the entry script once and every worker is forked from that image**; workers do not each
re-import `__main__`. The numpy-first ordering is established exactly once and inherited. The
guard is still required, but because the preload *executes* the module — without it, `main()`
would run inside the forkserver process itself.

**P1.11's "construct envs inside `_init()`" requirement is unaffected, and its stated reason was
already correct**: SB3 passes `CloudpickleWrapper(env_fn)` as a `Process` argument, so whatever
crosses the boundary must pickle, and a pre-built env holding a routing engine and open file
handles cannot. That is true under forkserver and spawn alike — it is about picklability, not
about how `__main__` is imported. Only the §8a paragraph needed correcting.

Files changed:
- `docs/SDPDP_GYM_SPEC_v2.md` — §8a mechanism corrected with the source citations; **new P2.0**
  added to §8 as a standalone prerequisite task: install `torch` from conda-forge, re-run the
  P1.10 gate, and only then remove the import-order workaround and §8a — with the explicit
  instruction that if the gate stops matching, that is the finding, report and stop, since the
  current environment works and has passed every gate.
- `train_sdpdp.py` — the stale "required for spawn" comment on the `__main__` guard replaced
  with the verified forkserver mechanism; the startup line no longer prints `mp`'s default
  start method as though it were the one in use.

Re-ran the smoke test after both edits: 2 workers, exit codes `[0, 0]`, no orphans, forkserver
confirmed.

Phase 1 is closed: P1.1 through P1.11 implemented, verified and committed, with the exit gate
passing byte for byte.

---

## P2.0b — STARTED 2026-10-10 11:54
Files I will touch: src/rl_gym/reward.py, src/rl_gym/gym_env.py,
tests/test_rl_gym_reward.py, src/rl_gym/tools/__init__.py,
src/rl_gym/tools/reward_trace.py, docs/RL_GYM_REWARD_TRACE.md,
docs/RL_GYM_PHASE1_RESULTS.md, docs/SDPDP_GYM_SPEC_v2.md, docs/RL_GYM_PROGRESS.md

Phase 2 starts here: P2.0 is deferred by Ritun (§8), so the import-order workaround of §8a
stays in place for all of Phase 2.

Approved before starting, from the §8 review this session:
- Finding 1 — §6.6 said Phase 1 used §6.1's weights; the code shipped w_reject = 0.5 and
  w_noshow = 0.8 (reward.py:36-45), which is what P1.10's episode_reward 116.510947 requires.
  Ritun: accepted; name the Phase 1 weights in §6.6, state that P2.0b replaced them, and record
  the weights behind 116.510947 in docs/RL_GYM_PHASE1_RESULTS.md. Also refresh the stale comment
  above DEFAULT_REWARD_WEIGHTS ("Phase 1 ships these unchanged", "weight design is P2.5").
- Finding 2 — after the default change, w_noshow, w_horizon and w_no_candidates are all 0.0, so
  tests/test_rl_gym_reward.py's charging tests would assert 0.0 == 0.0 and pass even if those
  branches stopped charging. Ritun: accepted; run the charging tests against an explicit
  non-zero weight dict and check the defaults separately in test_defaults_match_spec_6_1.
- src/rl_gym/tools/__init__.py is not in P2.0b's Files line; the import works without it
  (namespace subpackage, tested), but P1.1 settled on explicit packages. Ritun: add it.
- §2.9's present-tense corpus layout: leave it, P2.2 makes it true.
- P2.1's bbox explanation (lat corners are inside the node hull, not outside): correct it in
  P2.1, not here.

## P2.0b — DONE 2026-10-10 12:14 — awaiting Ritun's verification

Files changed:
- `src/rl_gym/reward.py` — **modified.** `DEFAULT_REWARD_WEIGHTS` moved to §6.1's Phase 2
  values (`w_reject` 0.5 → 1.0, `w_noshow` 0.8 → 0.0; the other six already matched), with the
  stale comment above it replaced — it had said Phase 1 ships these unchanged and pointed at
  "P2.5" for weight design. `RewardTracker(weights, track_events=False, end_time=None)`;
  `_add(amount, kind=None, rid=None, detail=None)` appends `(kind, rid, amount, detail)` when
  tracking is on and leaves both accumulators untouched on a zero amount; `flush_events()`;
  `on_exit` now emits an event for **every** classified outcome including the zero-weighted
  ones; `episode_summary()` gains `last_dropoff` and `horizon_margin`.
- `src/rl_gym/gym_env.py` — **modified.** `track_events` config key; the tracker receives it
  and the scenario's `end_time`; `info["reward_events"]` on both `step()` paths.
- `tests/test_rl_gym_reward.py` — **modified**, 34 → 52 tests.
- `src/rl_gym/tools/__init__.py` — **new.** Package marker, with the one-way dependency rule:
  nothing in `src/rl_gym/` may import from `tools/`.
- `src/rl_gym/tools/reward_trace.py` — **new.** The trace driver.
- `docs/RL_GYM_REWARD_TRACE.md` — **new**, generated.
- `docs/SDPDP_GYM_SPEC_v2.md` — §6.6 rewritten (Finding 1).
- `docs/RL_GYM_PHASE1_RESULTS.md` — one paragraph recording the weights behind 116.510947.
- `docs/RL_GYM_PROGRESS.md` — this entry.

No source file outside `src/rl_gym/` was touched: `git diff --stat -- src/` reports
`src/rl_gym/` only.

Verification — everything passed, nothing failed:

1. **The P1.10 gate, byte for byte.** Full suite: **179 tests pass**, 116.4 s, no failures and
   no skips (161 before; +18 reward tests). Independently, a trace run with
   `track_events = True` and output on: `cmp docs/baseline_user_stats.csv
   …_env0_pid5461_ep1/1_user-stats.csv` reports **no differing byte**, md5
   `53879ea79cf211d700f670dee1173dd5` both, 117,056 bytes both. The event list perturbs
   nothing. The committed `example_depot_time_pool_irsonly_rs_42/1_user-stats.csv` still
   carries the same md5 — untouched. The trace run's directory was removed afterwards.
2. **`flush()` is identical in both modes, at full scale.** Two greedy episodes, tracking off
   and on: the **per-step reward sequences are bit-identical** across all 436 steps,
   `episode_reward` is bit-identical (`118.11094709822194` both), and **no `episode_summary()`
   field differs**. This is what the zero-amount events had to be shown not to disturb.
3. **The trace.** 436 steps, **734 events**, written to `docs/RL_GYM_REWARD_TRACE.md`.
   Σ event amounts **118.110947** against `episode_reward` **118.110947**, difference
   **0.00e+00**; Σ step rewards the same; `unflushed_reward` 0.0. **Steps whose events do not
   sum to the step reward: 0.** Labels reconcile exactly with P1.10: 289 `pickup` events over
   288 distinct rids, 288 `served`, 118 `rider_declined`, 28 `diffusion_cancelled`, 9
   `no_candidates`, 2 `no_show`, 445 exits, `unclassified` **0**.
4. **`horizon_margin`.** `last_dropoff` **69805.25709890177**, `horizon_margin` **194.74 s** —
   matching P2.0b's expected 69805.3 and ~194.7 s.
5. Arithmetic cross-check of the new defaults against `baseline_user_stats.csv`: 288 pickups,
   2825.7811 wait-minutes, 118 declines, 28 cancellations, 2 no-shows, 9 no-candidates give
   116.510947 at the Phase 1 weights and **118.110947** at §6.1's — reproducing the measured
   episode total to the last digit, from the output file alone.

Two things the trace surfaced that the episode-level counts cannot show, which is its whole
point:

- **The rid-50 pair is now visible as two events.** Step 52 charges `+0.922585` at the stale
  92.9 s wait; step 55 charges `-0.164252`; the two sum to `+0.758333`, which is exactly one
  pickup priced at the final 290 s wait — the value `1_user-stats.csv` records (§6.4,
  last-write-wins). Step 55 also carries the `no_show` event for rid 45 at the same stop, so
  the mechanism and its consequence appear side by side.
- **Zero-amount events are load-bearing for readability, not just completeness.** On this day
  `served` (288), `no_candidates` (9) and `no_show` (2) all carry amount 0.000000 under §6.1's
  weights. Without them the trace would show 436 steps of pickups, declines and cancellations
  and nothing else — and the §6.4 ladder for 299 of the 445 exits would be unobservable.

One judgement call I made inside P2.0b, worth a look:

- `reward_events` is drained on **both** `step()` paths and on **neither** `reset()` path.
  `reset()` does not flush, so anything charged while advancing to the first decision belongs
  to step 1's reward window; draining it in `reset()` would drop those events from the trace
  while their reward stayed in the sum, and the per-step reconciliation would fail for step 1.
  Nothing is charged before the first decision on this day, so the choice is invisible here —
  it is there so the invariant holds on a day where something is.
- I added a **"Riders whose `record_boarding` fired more than once"** section to the report,
  beyond the step listing P2.0b specifies. Reason: the step selection rule ("first 40 steps,
  and any step with three or more events") happened to list rid 50's *second* event and not its
  first, so the trace showed a lone negative pickup, which reads as a defect rather than as the
  re-price. The section lists both halves of every such pair. The specified selection rule is
  unchanged.

VERIFY findings: **P2.0b carries no `VERIFY` markers** (grep over the work item's spec text:
zero matches). Per §0's "verify, don't trust", the claims P2.0b and its verification rest on
were each checked before use, and all held:

- `last_dropoff` 69805.3 and `horizon_margin` ≈194.7 s (P2.0b line 1810): **matched** —
  69805.25709890177 and 194.7429, recomputed from `baseline_user_stats.csv` before writing any
  code and reproduced by the tracker afterwards.
- `DEFAULT_REWARD_WEIGHTS` holds the Phase 1 values (P2.0b line 1793): **matched** —
  `reward.py:36-45` before the change, and `116.510947` is arithmetically reachable only with
  `w_noshow = 0.8`.
- The env passes `reward_weights = None` by default, so anything run without explicit weights
  uses the defaults (P2.0b line 1794): **matched** — `gym_env.py` `cfg.get("reward_weights")`,
  and `train_sdpdp.py`'s `DEFAULT_CFG` carries no `reward_weights` key.
- `on_exit` adds nothing when the penalty is 0 (P2.0b line 1786): **matched** — the
  `if penalty:` guard at `reward.py:143` before the change.
- `src/rl_gym/tools/` needs no `__init__.py` to import: **matched** — probed with a temporary
  module inside `src/rl_gym/`, which imported cleanly as a namespace subpackage. Added the
  marker anyway, per Ritun and P1.1's convention.
- Eight weights in §6.1, all eight consumed: **matched** — pinned by a new test.

Spec changes made this turn:
- §6.6 retitled "Phase 1 behaviour — superseded by P2.0b" and rewritten: it now names Phase 1's
  eight weights, states the two that differ from §6.1, records that `116.510947` reproduces
  only at `w_noshow = 0.8` while §6.1's weights give `118.110947` on the same day, says plainly
  that an earlier draft claiming Phase 1 used §6.1's weights was wrong, and notes that no
  Phase 1 gate depended on the weights — they scale the signal and touch no decision, which is
  why the byte gates were blind to the difference.

Note on the working tree: three tracked `__pycache__/*.pyc` files show as modified
(`src/rl_gym/gym_env`, `src/rl_gym/reward`, `tests/test_rl_gym_reward`) plus
`src/routing/NetworkBasic`, which was already modified at the start of the session. These are
bytecode byproducts of running the tests; `.gitignore` covers `__pycache__` since b1fd0a6 but
these files were tracked before that, so they still appear. Ritun's call whether to commit or
untrack them.

Phase 1 result directories from earlier items are still present and are reproducible:
`p13_rl_greedy_check`, `p13_tracker_greedy_check`, `p14_generator_check`, `p17_reward_check`.
P2.0b left no new ones.

Nothing failed, so there is no fix to propose. **Stopping here** — not starting P2.1.

---

## P2.1 — STARTED 2026-10-10 12:39
Files I will touch: data/networks/example_network/base/crs.info,
docs/SDPDP_GYM_SPEC_v2.md, docs/RL_GYM_PROGRESS.md

Resume state: P2.0b committed as 12eddb2, the bytecode untracking as cf01072, working tree
clean.

Approved before starting:
- The one-value change to `crs.info`, written with `printf` so the file stays exactly 10 bytes
  with no trailing newline (trap 15: no `# RL-GYM:` comment — the file is read raw into
  `self.crs` and handed to pyproj). `base - Copy/crs.info` left alone.
- P2.1's bbox explanation corrected: the corner longitudes fall outside the `nodes.csv` hull,
  the latitudes inside. Ritun supplied the mechanism clause — grid convergence of about
  0.5–0.7° — replacing my "lines of constant northing bow away from the central meridian".
  Verified before writing it: 0.5403°–0.6575° by closed form, 0.5382°–0.6549° measured as the
  lean of a grid-north step off true north over all 4,084 nodes.
- §2.10's reader list extended with the re-search result P2.1 asks for.

## P2.1 — DONE 2026-10-10 12:44 — awaiting Ritun's verification

Files changed:
- `data/networks/example_network/base/crs.info` — `EPSG:32632` → `EPSG:32610`. Written with
  `printf 'EPSG:32610' >` so the file could not gain a newline. Exactly **two bytes** differ
  (`cmp -l`: position 9 `3`→`1`, position 10 `2`→`0`); still **10 bytes**, still no trailing
  newline (`git diff` reports `\ No newline at end of file` on both sides). No `# RL-GYM:`
  comment (trap 15).
- `docs/SDPDP_GYM_SPEC_v2.md` — P2.1's bbox explanation corrected; §2.10's reader list extended
  with the re-search result.
- `docs/RL_GYM_PROGRESS.md` — this entry.

`base - Copy/crs.info` left alone: still `EPSG:32632`, and `git status` reports it unmodified.
No source file was touched — `git diff --stat -- src/` is empty.

Verification — everything passed, nothing failed:

1. **The P1.10 gate, byte for byte — the arbiter for this item.** Full suite: **179 tests
   pass**, 117.8 s, no failures and no skips. Independently, a scripted greedy run with output
   on: `cmp docs/baseline_user_stats.csv …_env0_pid6600_ep1/1_user-stats.csv` reports **no
   differing byte**, md5 `53879ea79cf211d700f670dee1173dd5` both, 117,056 bytes both. §2.10's
   prediction holds: the correction has no runtime effect in the reference scenario. The run's
   directory was removed afterwards; no strays left.
2. **`return_network_bounding_box()` through the real engine.** Constructed
   `NetworkBasicWithStore` on `example_network`: `self.crs` reads back as exactly
   `'EPSG:32610'` — no stray whitespace from the newline-free file — and the method returns
   `[(-122.11017985264547, 37.29207658100817), (-121.91532037433075, 37.407608253664044)]`.
   All four values are within **2.4e-5°** of P2.1's stated corners, well inside the 1e-3°
   tolerance.
3. **The reward trace is unchanged by the correction.** Re-running `reward_trace.py` after the
   change reproduces 436 steps, **734 events**, Σ events = `episode_reward` = **118.110947**,
   0 non-reconciling steps, `horizon_margin` 194.74 s, `unclassified` 0 — byte-identical to the
   committed report except for the wall-clock line (see the note below).

Checks of §2.10's evidence, all of which held (P2.1 carries no `VERIFY` markers; these are the
claims the change rests on):

- Converting all 4,084 `pos_x`/`pos_y` pairs from EPSG:32610 reproduces the `lat`/`lon`
  columns: **matched** — max |dlat| **4.504e-09**, max |dlon| **6.237e-10**, against §2.10's
  4.5e-9 and 6.2e-10.
- Read as EPSG:32632 the coordinates land exactly 132° east, latitudes surviving: **matched** —
  max |dlon| **1.320e+02**, while |dlat| is **identical (4.504e-09) under both CRSs**, since
  UTM zones differ only in central meridian. That is precisely why the error went unnoticed.
- Grid convergence here is about 0.5–0.7° (Ritun's wording for the bbox explanation):
  **matched** — **0.5403°–0.6575°** by the closed form
  `arctan(tan(λ − λ₀)·sin φ)` with λ₀ = −123°, and **0.5382°–0.6549°** measured as the lean of
  a 1 km grid-north step off true north, over all 4,084 nodes.
- `realtime_plot` unset in the reference scenario, so nothing calls the two `self.crs` users
  during a run: **matched** — the key is absent from both config files.
- Only two `crs.info` files exist in the repository: **matched**.

Spec changes made this turn:
- **P2.1's "How Ritun verifies"**: the corner figures and the 1e-3° tolerance are unchanged;
  the explanation is replaced. It had said the corners "sit fractionally outside the range of
  the `lat`/`lon` columns", which is **false for latitude** — measured, the corner longitudes
  fall outside the hull (−122.1102 against −122.1095; −121.9153 against −121.9162) while the
  corner latitudes fall **inside** it (37.2921 against 37.2915; 37.4076 against 37.4089). The
  mechanism is now named: the UTM grid here is turned about 0.5–0.7° from true north (grid
  convergence), so a corner of the UTM rectangle is not at the extreme latitude or longitude of
  the node set. A verifier comparing hulls rather than the stated corners would have read a
  pass as a failure.
- **§2.10's "Who reads the file"**: line references added for the `NetworkBasic` read and the
  zone script; plus the two re-search findings — that the single `NetworkBasic` read reaches
  every registered routing engine but `NetworkTTMatrix` (`NetworkBasicWithStore`,
  `NetworkPartialPreprocessed` and `NetworkBasicCpp` inherit directly, the other three
  transitively), and that three further readers exist as unregistered dead copies
  (`NetworkBasic_backup.py` ~193, `Backup_files_by_Ritun/NetworkBasic.py` ~189,
  `Backup_files_by_Ritun/NetworkBasic_with_sto_netdyn_update.py` ~190), absent from
  `get_src_routing_engines()` and so unreachable from a config, marked do-not-edit.

One thing checked beyond the list, since the no-newline rule exists for it: all four read sites
consume a newline-free file correctly. `NetworkBasic` ~195 and `NetworkTTMatrix` ~117 do a bare
`f.read()`, which is exactly why the rule exists; `ReplayFromResult.py` ~431 does
`.read().strip()` and then `int(crs[5:])`, giving 32610; `create_rectangle_zones.py` ~165 does
`.read().strip()`.

**Note on `docs/RL_GYM_REWARD_TRACE.md`.** It shows as modified, by one line: the wall-clock
figure, 38.75 s → 39.08 s. I used `reward_trace.py --output` as P2.1's independent byte check,
and the tool rewrites its report. Every reward figure, event count and label in it is
byte-identical to the committed version, so the diff is pure timing noise — and the fact that
it is identical is itself evidence the CRS change altered nothing. I have not reverted it: §0
forbids git commands that change state, and `git checkout -- <file>` is one. Ritun's call
whether to restore or keep it. For later items I will drive the byte check from the P1.10 gate
or a scratchpad driver instead, so a committed report is not rewritten as a side effect.

Nothing failed, so there is no fix to propose. **Stopping here** — not starting P2.2.

---

## P2.2 — STARTED 2026-10-10 14:08
Files I will touch:
data/demand/example_demand/matched/example_network/corpus_eval/ (new, 26 files moved in),
data/demand/example_demand/matched/example_network/corpus_excluded/ (new, 5 files moved in),
data/demand/example_demand/matched/example_network/corpus/ (132 -> 101 csv),
docs/SDPDP_GYM_SPEC_v2.md, docs/RL_GYM_PROGRESS.md

Resume state: P2.1 committed as 2ef25f3, trace-report change discarded by Ritun, tree clean.

Approved before starting:
- Filesystem moves only, no git commands (§0). The 26 evaluation days and 5 excluded days come
  from §2.9's literal lists, which P2.2 makes authoritative; §2.9's algorithm is re-derived
  first as a cross-check, and a mismatch is a stop condition.
- `.DS_Store` stays in `corpus/`. It is untracked and `git ls-files` counts exactly 132 files
  under `corpus/`, so only CSVs move.
- Two §2.9 additions: (a) `thursday/` also holds `2024-01-18.csv`, byte-identical to its corpus
  copy and unused by any scenario, which stays a training day; (b) a new rule that every reader
  of the corpus directories lists demand days by `*.csv` only and ignores every other entry,
  since macOS creates and rewrites `.DS_Store` on its own — binding on P2.3's rotation, P2.4's
  sweep, P2.6's manifest and P2.9's corpus check, with P2.9 additionally carrying a test that a
  stray `.DS_Store` does not make construction raise. Ritun asked for the rule; I am also
  putting the test requirement into P2.9's verification list so it is not missed when P2.9 is
  implemented.

## P2.2 — DONE 2026-10-10 14:14 — awaiting Ritun's verification

Files changed — filesystem moves only, no git commands:
- `data/demand/example_demand/matched/example_network/corpus_eval/` — **new**, the 26
  evaluation days from §2.9's list.
- `…/corpus_excluded/` — **new**, the 5 calendar-explained excluded days.
- `…/corpus/` — **132 → 101** CSVs. The untracked `.DS_Store` was left in place; only `.csv`
  files moved.
- `docs/SDPDP_GYM_SPEC_v2.md` — two §2.9 additions and one line in P2.9.
- `docs/RL_GYM_PROGRESS.md` — this entry.

No code file touched; `git diff --stat -- src/ tests/` is empty.

Verification — everything passed, nothing failed:

1. **The cross-check P2.2 requires, run before anything moved.** Re-deriving the evaluation
   list with §2.9's algorithm — `default_rng(20260929)`, months in calendar order, running
   per-weekday counts — reproduced §2.9's literal 26-day list **exactly**, so the stop
   condition at P2.2 line 1882 did not trigger. Per-month `n` came out 2,3,2,3,3,2,3,3,2,3 over
   127 kept days. The 31 files were then moved **by name from §2.9's literal lists**, which
   P2.2 makes authoritative, not from the derivation.
2. **Counts: 101 / 26 / 5, 132 in total.** No filename in two of the three directories (all
   three pairwise intersections empty).
3. **Weekday counts exact:** training **Tue 35 / Wed 33 / Thu 33**, evaluation
   **Tue 8 / Wed 9 / Thu 9**.
4. **Every month 1–10 present in both splits.** Training days per month 9, 10, 10, 10, 11, 9,
   10, 10, 10, 12 — inside §2.9's stated 9–12.
5. **Checksums unchanged by the move.** md5 was snapshotted for all 132 files before the first
   move and recompared afterwards across all three directories: **zero mismatches, all 132
   accounted for**.
6. **`thursday/2024-06-20.csv` untouched** — md5 `2ad73f37ce40dac2eade120470bf7e8e` and mtime
   `1788379479.6600573`, both identical to the pre-move snapshot. `thursday/` still holds
   exactly its two files.
7. **The P1.10 gate, byte for byte.** Full suite **179 tests pass**, 123.4 s, no failures and
   no skips; the gate alone re-run verbosely, all 7 tests `ok` including
   `test_user_stats_matches_baseline_byte_for_byte`. Driven from the gate this time rather than
   `reward_trace.py`, so no committed report was rewritten — the lesson from P2.1. No stray
   result directories.

Pre-flight checks, all of which held:
- 132 days, 44 each of Tue/Wed/Thu, Jan 2 – Oct 31 2024, no missing or extra date: **matched**.
- All 26 evaluation and 5 excluded days present in `corpus/` before the move, the two lists
  disjoint: **matched**.
- **Nothing in any `.py` reads `corpus/`**: **matched** — only `docs/` mentions it, so the move
  could not break code. This is why no test needed changing.
- `corpus/` has no subdirectories: **matched**.
- `.DS_Store` untracked, `git ls-files corpus/` counting exactly 132: **matched**.

**A mistake of mine, caught by its own failure.** My first attempt moved the files with a shell
`for d in $EVAL` loop. zsh does not word-split unquoted parameter expansions, so `$EVAL` stayed
one word and `mv` was handed a single 300-character filename. It failed with
`File name too long` **before writing anything**, and because the loop was in an `&&` chain the
`corpus_excluded` moves never ran either. Verified the state afterwards rather than assuming:
`corpus/` still held all 132, both new directories were empty, no file with a space in its name
existed anywhere under the demand tree, and `git status` on the demand tree was clean. Redone in
Python with `shutil.move` and a pre-flight assertion per file — every source present, every
destination absent — which is what the move should have been from the start, since the file list
is data rather than shell words.

**What `git status` shows, so it is not a surprise:** 31 deletions under `corpus/` plus two
untracked directories, not 31 renames. Rename detection runs against staged content, so the
pairing appears once the new directories are `git add`-ed; the contents are bit-identical
(check 5), so the similarity index will be 100%.

VERIFY findings: **P2.2 carries no `VERIFY` markers.** The §2.9 claims it rests on were each
checked and are recorded above — the split algorithm, the 132-day inventory, the weekday and
month structure, and the `thursday/` carve-out.

Spec changes made this turn (§2.9, plus one line in P2.9):
- **The second content-identical twin recorded.** `thursday/2024-01-18.csv` is byte-identical
  to its corpus copy — same md5, 426 rows — rather than a line-ending twin like 2024-06-20. No
  scenario reads it, and 2024-01-18 is not in the evaluation split, so it stays a **training**
  day in `corpus/`. §2.9 previously read as though 2024-06-20 were the only overlap, so a later
  reader checking "no day sits in two places" would have found this and not known whether it
  was expected.
- **New rule: list demand days by `*.csv` only.** Every reader of the corpus directories
  enumerates by extension and ignores every other entry, because macOS creates and rewrites
  `.DS_Store` in these directories on its own and it is untracked — it can appear, change or
  vanish between runs with no demand file changing. A reader enumerating directory entries
  would see a phantom day, and in P2.6's manifest a `.DS_Store` would put a volatile hash into
  a frozen artifact and fail every later corpus check for no real reason. Stated as binding on
  **P2.3's rotation, P2.4's sweep, P2.6's manifest and P2.9's corpus check**, with "do not
  delete these files to work around it — they come back."
- **P2.9's verification** gains the test Ritun asked for: dropping a stray `.DS_Store` into the
  temporary corpus copy must leave construction working. Put into P2.9's own text, not only
  §2.9, so it is not missed when P2.9 is implemented — otherwise it is the kind of rule that
  passes every test and then fails the first time Finder opens the directory.

Nothing failed, so there is no fix to propose. **Stopping here** — not starting P2.3.

---

## P2.3 — STARTED 2026-10-10 14:29
Files I will touch: src/rl_gym/day_rotation.py (new), src/rl_gym/gym_env.py, train_sdpdp.py,
tests/test_rl_gym_env.py, docs/SDPDP_GYM_SPEC_v2.md, docs/RL_GYM_PROGRESS.md

Resume state: P2.2 committed as 7a0cacd, tree clean.

Approved before starting:
- `src/rl_gym/day_rotation.py` as a new file, not in P2.3's Files line. D10's selection is a
  pure function of (sorted days, base_seed, n_workers, env_id, episode index), so verification
  item 3's two passes over 3 workers and 10 days are millisecond unit tests instead of 20 real
  episodes (~13 minutes), and the rotation is testable without FleetPy.
- **Rotation mode requires `base_seed` as well as `n_workers`**, and the env raises without it.
  P2.3 line 1921 named only `n_workers`, but D10's permutation comes from
  `SeedSequence([base_seed, 1010, p + 1])`; defaulting it is the failure mode D7 forbids.
  P2.3's sentence and Files line updated.
- Construction validates that every named day (`day`, and each day in `eval_pairs`) exists as a
  `*.csv` in `day_dir`; otherwise a typo fails deep inside `load_demand_file`.
- Rotation-mode output directories keep P1.9's name; only fixed pairs get the day-and-seed name
  P2.3 specifies. The counter keeps them unique and `episode_summary()` records the day.
- An unparseable filename stem leaves weekday/month `None` rather than raising at `reset()`, so
  a non-ISO `rq_file` on the reference path still runs; P2.9 raises when it needs them (§5.2).
- The smoke test may use rotation.
- **Ritun's change: `n_workers` must NOT go in `DEFAULT_CFG`.** It is passed into `make_env`
  from the same value that sizes the `SubprocVecEnv`, so the two cannot disagree — a config
  value of 4 run with `--workers 3` would silently drop a quarter of every pass. The env
  additionally raises when `env_id >= n_workers`, with a test for that raise.

## P2.3 — DONE 2026-10-10 14:44 — awaiting Ritun's verification

Files changed:
- `src/rl_gym/day_rotation.py` — **new.** `training_days`, `pass_permutation`,
  `positions_per_pass`, `day_for_episode`, `PASS_STREAM_TAG`. Pure functions, no FleetPy import.
- `src/rl_gym/gym_env.py` — **modified.** `parse_day_stem`; `_configure_days`,
  `_check_seed_range`, `_resolve_demand_dir`, `_apply_episode_day`, `_episode_pair_seed`;
  `_apply_episode_seed` honours a pinned seed; `_apply_episode_scenario_name` names the pair
  when one is fixed; `episode_summary()` gains `day`, `day_dir`, `weekday`, `month`, `pass`,
  `pass_position`.
- `train_sdpdp.py` — **modified.** `day_dir = "corpus"` in `DEFAULT_CFG`; `n_workers`
  deliberately **not** a config key — `make_env(cfg, env_id, n_workers)` takes it from the same
  value that sizes the `SubprocVecEnv`, and itself rejects `env_id >= n_workers`; the per-worker
  report prints day, pass and position, and warns if a day repeats within a pass.
- `tests/test_rl_gym_env.py` — **modified**, 25 → 57 tests.
- `docs/SDPDP_GYM_SPEC_v2.md` — P2.3's Files line, the `base_seed`/`n_workers`/`env_id` rules,
  the day-validation and stem-parsing paragraphs, the summary fields, and the VERIFY finding.
- `docs/RL_GYM_PROGRESS.md` — this entry.

Verification — all five of P2.3's criteria passed, nothing failed:

1. **`day_dir` absent → the P1.10 gate, byte for byte.** Full suite **211 tests pass**,
   119.6 s, no failures and no skips (179 before; +32). The gate's own
   `test_user_stats_matches_baseline_byte_for_byte` is among them, and a construction test
   asserts that with `day_dir` absent the scenario row's `day_dir_name = thursday` and
   `rq_file = 2024-06-20.csv` are untouched.
2. **`day_dir = "corpus_eval"`, `day = "2024-06-20"`, `base_seed` absent → byte-identical.**
   md5 `53879ea79cf211d700f670dee1173dd5`, 436 steps, seed 42, pickups 288 / declined 118 /
   cancelled 28 / no-show 2 / unclassified 0. The line-ending difference between
   `corpus_eval/2024-06-20.csv` and `thursday/2024-06-20.csv` is indeed invisible to the loader.
3. **The 3-worker / 10-day rotation**, as pure unit tests: per-pass shares **4 / 3 / 3**; over
   two full passes **every day appears exactly twice**; within one pass the three workers
   **partition** the days; each worker's sequence is deterministic given `base_seed` and changes
   when `base_seed` does; and pass `p`'s permutation equals
   `default_rng(SeedSequence([base_seed, 1010, p + 1])).permutation(days)` for p = 0, 1, 5.
4. **`episode_seed = 42` → byte-identical** (md5 as above), **`episode_seed = 43` → differs**
   (md5 `471db32e…`, 427 steps, 297 pickups, 101 declines). **Both `episode_seed` and
   `base_seed` → raises.** The fixed-pair output directory named the pair:
   `…_rs_42_2024-06-20_s42_env0_pid8909_ep1`.
5. **`eval_pairs` runs its pairs in order and wraps**, checked over 7 resets of a 3-pair list:
   each episode's `day`, `day_dir` and `G_RANDOM_SEED` matched its pair.

Beyond the five:

- **The rotation works end to end under `SubprocVecEnv`, and matches the pure function.**
  `train_sdpdp.py --workers 3` ran three workers on three different training days — worker 0
  `2024-09-26`, worker 1 `2024-03-27`, worker 2 `2024-07-02`, all pass 0 at positions 0, 1, 2 —
  with three distinct episode seeds, exit codes `[0, 0, 0]`, no orphans, `unclassified = 0` in
  every worker. Recomputing those three days from `day_for_episode` over the real 101-day corpus
  **predicted all three exactly**. A full pass across 3 workers covers all 101 days exactly once
  (shares 34 / 34 / 33).
- **Every rotation misconfiguration raises** rather than defaulting: no `base_seed`, no
  `n_workers`, `env_id >= n_workers`, `episode_seed` with `base_seed`, `episode_seed` out of
  range, `eval_pairs` with `day` / `episode_seed` / `base_seed`, `eval_pairs` without `day_dir`,
  empty `eval_pairs`, an `eval_pairs` seed out of range, an unknown `day`, an unknown day inside
  `eval_pairs`, an unknown `day_dir`, and — worth its own test — a **training** day named
  against `corpus_eval`, which exists but in the other split.
- `training_days` lists `*.csv` only: a `.DS_Store` and a `notes.txt` dropped into a temporary
  directory are both ignored (§2.9), and the real `corpus/` comes back as exactly 101 sorted
  days with no evaluation day among them.

VERIFY findings:
- **The `scenario_parameters` keys carrying the demand directory and file, and whether anything
  caches the demand path before construction: matched the spec.** `G_DAY_DIR = "day_dir_name"`
  (`globals.py:23`), `G_RQ_FILE = "rq_file"` (`globals.py:104`). The directory is assembled at
  construction by `get_directory_dict` —
  `dirs[G_DIR_DEMAND] = data/demand/<demand_name>/matched/<network_name>/<day_dir>`
  (`globals.py:741`) — from a single call in `FleetSimulationBase.__init__` (`:150`); the
  filename is read at `:344` and passed straight into `load_demand_file` with `G_RANDOM_SEED`.
  **Nothing caches it**: the only other `get_directory_dict` callers are
  `ReplayFromResult.py:406`, `evaluation/temporal.py:465,999` and the two `standard*.py`
  evaluators, all outside a run. `G_RQ_FILE` is in `input_parameters_mandatory` and read with
  `[]`, so a missing key fails loudly. Recorded in P2.3's VERIFY line.
- D10's two seed details, both checked rather than assumed: `SeedSequence` pads with zeros, so
  `[base_seed, 1010, 0]` **is** the same stream as `[base_seed, 1010]` — which is exactly why
  `p + 1` is required; and across 80 passes and 16 worker ids the pass streams are all distinct
  and **never** intersect the episode-seed streams.

A design note, since it is the one place I went beyond the spec's structure: `_resolve_demand_dir`
validates day names by calling FleetPy's own `get_directory_dict` on a copy of the scenario
parameters rather than reassembling the path. That way the validation cannot drift from what
construction will compute. It caches nothing for the simulation's benefit — each `reset()` writes
`G_DAY_DIR` / `G_RQ_FILE` and lets construction derive its own `dir_names`.

Spec changes made this turn:
- P2.3's Files line now lists `src/rl_gym/day_rotation.py`, with the reason it exists (a pass
  over several workers becomes a unit test rather than one episode per position).
- The `n_workers` bullet became `n_workers` **and** `base_seed`, with the mechanism — D10 draws
  each pass from `SeedSequence([base_seed, 1010, p + 1])`, so without `base_seed` there is no
  pass order at all — plus the `env_id >= n_workers` raise and the rule that `n_workers` is not
  a `train_sdpdp.py` config key but comes from the value sizing the `SubprocVecEnv`.
- Added: every named day must exist in `day_dir`, checked at construction; day listings are
  `*.csv` only (§2.9); `episode_summary()` records `day_dir`, `weekday`, `month`, `pass` and
  `pass_position`, and in rotation mode is the only record of which day an episode ran; a
  non-ISO stem yields `None` for weekday and month rather than raising, with P2.9 as the place
  that raises.
- P2.3's VERIFY line now carries the finding instead of only the instruction.

Also in this turn: the STARTED entry above was written with the time 15:02, which was wrong — I
composed it before `date` returned. Corrected to 14:29 in the same turn. Noting it rather than
leaving a silently edited timestamp in an append-only log.

Nothing failed, so there is no fix to propose. **Stopping here** — not starting P2.4.

---

## P2.4 — STARTED 2026-10-10 15:05
Files I will touch: src/rl_gym/tools/corpus_sweep.py (new),
docs/RL_GYM_CORPUS_SWEEP.md (new), docs/RL_GYM_PROGRESS.md

Resume state: P2.3 committed as 88bebaa, tree clean.

Approved before starting:
- `corpus_sweep.py` carries its own copy of P1.10's instrumentation — the bucket classifier
  monkeypatching `build_assignment_context`, and the tail measurement monkeypatching
  `record_remaining_assignments` / `update_sim_state_fleets` — rather than importing it from
  `tests/test_rl_gym.py`, which would point a shipped tool at a test module. The duplication is
  made self-checking: when 2024-06-20 is in the sweep set the tool asserts its row against
  P1.10's figures and fails loudly otherwise, so P2.4's manual verification becomes automatic.
- At most 4 processes (16 GB Mac), and the full sweep runs under `caffeinate -i`.
- **Ritun's change 1: the three trial days run one after another in a single process, with
  2024-06-20 last.** Training builds many simulations in one worker process and the Pool reuses
  processes the same way, so the reference day reproducing P1.10 exactly *after* two other days
  have been simulated in the same process is evidence that nothing carries over between
  simulations. If it does not match, stop and report.
- **Ritun's change 2:** the docstring and report header write the command as
  `caffeinate -i python -m src.rl_gym.tools.corpus_sweep …`, with a note that `caffeinate` is
  macOS-only, rather than an absolute `/opt/anaconda3` interpreter path — the tool will be
  re-run after the routing fix, possibly on HPC3.

Trial days, in execution order: 2024-03-06 (the only day with two requests before
`start_time`), 2024-10-09 (October, the high-demand regime), 2024-06-20 (the reference day,
last).

## P2.4 — DONE 2026-10-10 16:25 — awaiting Ritun's verification

Files changed:
- `src/rl_gym/tools/corpus_sweep.py` — **new.** The sweep driver, with its own copy of P1.10's
  instrumentation and an automatic reference-day check.
- `docs/RL_GYM_CORPUS_SWEEP.md` — **new**, generated: 127 per-day rows and the summaries.
- `docs/RL_GYM_PROGRESS.md` — this entry.

No change to the env or the simulation (P2.4 line 1991); `git diff --stat -- src/rl_gym/` shows
only the new tool. P2.4 is not in §8's list of items that re-run the P1.10 gate (line 1727);
the reference-day row is the equivalent check here, and it is asserted automatically.

Verification — the sweep completed, and everything reconciles. One defect of **mine** was found
and fixed first; details below.

- **127 days swept, 24.2 min, 4 processes, 4 distinct worker pids.**
- **Reconciliation failures: 0.** Three checks per day: `steps + empty + reservation + o=d +
  duplicate-rid = loaded`, `exits = loaded`, and the eight §6.4 terminal outcomes summing to
  `loaded`.
- **`unclassified = 0` on all 127 days.** No new §6.4 path exists anywhere in the corpus.
- **The reference day reproduces P1.10 exactly** — 445 loaded, 436 steps, 9 empty lists, max
  list 11, 13 over `K`, `horizon_margin` 194.7 s, 288 pickups, 118 declines, 28 cancellations,
  2 no-shows, 1 duplicate boarding — asserted by the tool, which reports `MATCHES`.
- Corpus totals: 59,190 requests loaded (mean 466/day, 356–634), 57,447 gym steps (mean 452,
  352–603). Reservation bucket 0, duplicate-rid bucket 0, operator-resolved o=d bucket 0 across
  the whole corpus. Duplicate boardings on 9 of 127 days, at most 2 per day.

**Trial first, per Ritun: three days in one process with 2024-06-20 last.** 2024-03-06,
2024-10-09, then the reference day, all in pid 9749. The reference day matched P1.10 exactly
*after* two other full simulations had run in the same process — evidence that nothing carries
over between simulations: global RNG, routing caches, stochastic travel-time state, logging.
That is the arrangement training uses, so it is worth more than the same check run cold.

**A defect in my own counting, which the sweep caught.** The first full pass reported 21
reconciliation failures — 7 days × 3 checks, every one exactly one request short. Diagnosis,
not symptom: `load_demand_file` applies **two** filters, not one (`demand.py` ~103-105):

    future_requests = tmp_df[(tmp_df[G_RQ_TIME] >= start_time) & (… < end_time)]
    future_requests = future_requests[future_requests[G_RQ_ORIGIN] != …[G_RQ_DESTINATION]]

I had modelled only the time filter. A row whose `start` node equals its `end` node is removed
**before the simulation begins**: it never enters `future_requests`, never reaches
`build_assignment_context`, never calls `record_user`. So it is correctly absent from every
bucket, and counting it as "loaded" was the error. The 7 failing days are **exactly** the 7
corpus days carrying one such row — 2024-02-15 (rid 120), 2024-02-20 (307), 2024-03-05 (72),
2024-05-23 (33), 2024-05-30 (73), 2024-06-12 (310), 2024-06-13 (166) — an exact set match
against an independent scan of all 127 files, with no false positives either way. Two other
hypotheses were tested and rejected first: requests past the loop's last retrieval step
(`rq_time > 69990`; none exist, and `rq_time` is minute-aligned), and duplicate `request_id`
(zero duplicates anywhere in the corpus, so §2.9's "unique request ids" holds).

Fixed by applying both filters and reporting the dropped count as its own column, `o=d drop`,
kept distinct from the `o=d` bucket — the first is dropped at load, the second is resolved by
the operator. Re-ran the 7 days: all reconcile. Then re-ran the full 127 so the committed
report is clean rather than carrying 21 explained-away failures.

### `K = 8`, the corpus-wide picture Ritun asked for

- **share of all decisions whose candidate list exceeds `K = 8`: 2,947 of 57,447 = 5.13%**
- **days whose p95 list length is above 8: 63 of 127**
- days with at least one list above 8: **127 of 127**
- per-day p95: min 7, median 8, max 10
- per-day max: min 9, median 11, **largest list anywhere 12**
- lists over `K` per day: min 4, median 21, max 55 (2024-03-05)

The reference day's 13 lists over `K` is well below the corpus median of 21, so P1.10's figure
understates how often truncation bites. Truncation still removes only the weakest insertions
(D3); whether the policy wants what it cannot see is answered by P2.13's slot-selection
frequency. **Decision on `K` stays with Ritun, before P2.8.**

### The tail against its cap — the branch-7 indicator

Reported as Ritun directed, in place of my earlier framing:

- **longest tail across the sweep: 1,950 s on 2024-05-01 — 13.5% of the 14,400 s cap**, leaving
  12,450 s of headroom; the cap is **7.4×** the longest tail observed.
- **days where the cap bound: 0.** **Days where §6.4's branch 7 fired: 0.**
- days where the tail advanced the simulation at all: **81 of 127**; days with a negative margin:
  **79 of 127**; margin min −1917.5 s, p25 −690.3, median −234.4, p75 +186.5, max +1103.8;
  48 of 127 days finish before `end_time`.
- most riders in flight when the tail starts, anywhere in the corpus: **5**.

The cap is `if c_time > self.end_time + 2 * 7200: break` (`FleetSimulationBase.py` ~625 — written
`2 * 7200`, which is why grepping for `14400` does not find it), and it breaks the loop with
vehicles still en route. That break is the only way a rider who accepted an offer can end the
day unpicked-up, so the tail-to-cap distance is the indicator and the margin is not.

Ritun's correction to my trial finding was right and is recorded in the report: a negative
margin does not bring branch 7 closer, it only means the tail is completing trips after
`end_time`, which is **expected on busier days**. My trial wording had conflated the two.

VERIFY findings: **P2.4 carries no `VERIFY` markers.** The claims it rests on were checked:
- the 127 kept days are 101 `corpus/` + 26 `corpus_eval/`, `*.csv` only, never
  `corpus_excluded/`: **matched**;
- §2.9's "unique request ids": **matched** — zero duplicate `request_id` in any of the 127 files;
- §2.9's four pre-`start_time` requests: **matched** earlier in P2.2's checks;
- the tail cap is `end_time + 14400`: **matched**, written `2 * 7200` at
  `FleetSimulationBase.py` ~625, with the `break` that makes branch 7 reachable;
- **the `o_pos == d_pos` internal-resolution case is not reached by same-node demand rows** —
  see the spec proposals below. This is the one finding that contradicts the spec as written.

No spec edits were made this turn. Three passages rest on the premise Ritun corrected, and a
fourth on the one above; proposed wording for all four is in the report to Ritun, **not applied**
— he asked to see it first.

Nothing failed at the end of the item. **Stopping here** — not starting P2.5.

## P2.4 — FOLLOW-UP 2026-10-10 16:49 — spec corrections after Ritun's review

Ritun accepted all four proposals and asked for two adjustments. Applied to
`docs/SDPDP_GYM_SPEC_v2.md` only; no code changed, and the full suite still passes
(**211 tests**, 118.7 s, no failures, no skips).

**Adjustment to proposal 2, answered with data first.** Ritun's point: P2.13 must log a value
that exists, so check whether the tail is derivable from `horizon_margin` before writing a
bullet that depends on it. Tested across all 127 days:

- On **all 79** negative-margin days, `tail = ceil(−margin / 10) × 10 + 30` seconds **exactly**
  — the 10 s is `time_step`, the 30 s is the tail loop's three closing iterations.
- On **all 48** non-negative-margin days it fails: the tail is 0 s on 46 and 10 s on 2
  (2024-02-07 margin +25.4, 2024-09-11 margin +24.6, both with nothing in flight — a vehicle
  can hold an assigned route at `end_time` without a late dropoff), against a predicted 30 s.

So the relation does **not** hold on every day, and per Ritun's rule `episode_summary()` gains a
`tail_seconds` field in P2.13. Recorded in the bullet with the reason: a quantity that is exact
in one regime and wrong in another is what gets misread later. The tail is measured in
`RLImmediateDecisionsSimulation`, which owns the finalisation that runs
`record_remaining_assignments`.

**A consequence worth flagging:** P2.13 now edits `sim_env_rl.py` and `gym_env.py`, so it
touches the env. §8's gate list (line ~1797) named P2.0b, P2.3, P2.7–P2.12 and P2.14; P2.13 is
now in it, so the range reads P2.7–P2.14. Without that, the one item that adds a field to the
simulation env would have been the only env-touching item not re-running the P1.10 gate.

Spec changes made this turn:
- **§6.4's branch-7 paragraph rewritten.** The old text said the branch "becomes live as soon as
  that margin closes" and rested `w_horizon = 0.0` on the reference day's 195 s margin. Both
  were wrong: 79 of 127 days already have a negative margin (median −234 s, range −1918 to
  +1104), and the reference day's +195 s is late in the distribution — 96 of 127 days finish
  later. The branch turns on the **tail against its 14,400 s cap** instead, with the cap's
  source cited (`if c_time > self.end_time + 2 * 7200: break`, `FleetSimulationBase.py` ~625,
  written `2 * 7200`, which is why grepping for 14400 misses it) and the measurements: longest
  tail 1,950 s = 13.5% of the cap, 12,450 s of headroom, cap bound on 0 days, branch 7 fired on
  0 days, at most 5 riders ever in flight when a tail began. The margin↔tail relation and its
  two regimes are recorded there too. Marked as corrected in P2.4.
- **P2.13's horizon-margin bullet rewritten** per the adjustment above: a negative margin is the
  normal case and not a warning; `tail_seconds` is logged rather than derived; `sim_env_rl.py`
  and `gym_env.py` added to P2.13's Files line.
- **§8's gate list** now reads P2.7–P2.14, with a note saying why P2.13 joined it.
- **P2.15's "Reported, not judged"**: "horizon margin (does branch 7 approach?)" became the
  margin *and* the tail-to-cap distance, naming the cap as the indicator.
- **§1 gained a fourth bullet**: a demand row whose `start` node equals its `end` node never
  reaches the operator — removed by `load_demand_file`'s second filter (`demand.py` ~105) — so
  it generates no step **and appears in no bucket**. 7 of 127 days carry one, and the
  operator-resolved o=d bucket is 0 on all 127 including those 7. The `o_pos == d_pos` branch
  can only fire if distinct nodes resolve to one position, which I have **not** tested, so the
  wording keeps it an open question rather than declaring the branch dead.
- **§2.9's post-filter sentence replaced** with both filters, per Ritun's second adjustment: the
  four early requests, **and** one same-node row on each of the 7 named days. 2024-03-05 loses
  one row to each filter — 426 in the file, 425 after the window, 424 loaded — and the note
  records that modelling only the first filter is what produced P2.4's first-pass shortfall.
- **P1.10's reconciliation note**: 445 is the count after *both* filters, which coincides with
  the file's row count only because the reference day loses nothing to either; and the
  same-origin-destination bucket is expected to be 0 for a different reason than the
  duplicate-rid bucket — such requests are deleted before the operator sees them, so a 0 is not
  evidence that none existed.

**Stopping here** — not starting P2.5.

---

## P2.5 — STARTED 2026-10-10 17:01
Files I will touch: src/rl_gym/offline/__init__.py (new),
src/rl_gym/offline/build_zone_table.py (new),
data/rl_gym/example_network/h3_neighbours.csv (new), src/rl_gym/zones.py (new),
tests/test_rl_gym_zones.py (new), docs/SDPDP_GYM_SPEC_v2.md, docs/RL_GYM_PROGRESS.md

Resume state: P2.4 committed as 4c9969d, tree clean. K stays at 8 for now, per Ritun; D3 and
P2.13's K-binding watch unchanged.

Approved before starting:
- `src/rl_gym/offline/__init__.py`, not in P2.5's Files line — the same explicit-package
  convention as `tools/__init__.py` in P2.0b.
- A `--check` mode on the builder, rebuilding in memory and diffing against the committed CSV,
  so the artifact is re-verifiable without regenerating it.
- **Ritun's adjustment: `--check` compares the parsed `(cell, neighbour)` pairs as sets, not
  file bytes**, so a line-ending difference on another machine cannot fail it — the same lesson
  as the corpus/thursday twins in §2.9.
- One sentence added to P2.5 recording the measured gap behind the 1e-9° rule: the 88 neighbour
  pairs are at distance exactly 0.0 and the nearest non-neighbour is 1.377e-02, so every
  threshold from 0 to 1e-2 gives the same 88 pairs. The rule is exact, not approximate, and the
  sentence exists so nobody later "tightens" it thinking it is delicate.
- Both lookups raise on an unknown key rather than returning empty: P2.7 keys events by origin
  cell and P2.9 reads the counts, so an empty set would zero out every zone count silently.

## P2.5 — DONE 2026-10-10 17:08 — awaiting Ritun's verification

Files changed — all new except the two docs:
- `src/rl_gym/offline/__init__.py` — package marker, with the one-way dependency rule: nothing
  in `src/rl_gym/` may import from `offline/`, so a builder cannot be pulled into a training run.
- `src/rl_gym/offline/build_zone_table.py` — the builder, with `--check`.
- `data/rl_gym/example_network/h3_neighbours.csv` — **213 rows + header**, sorted.
- `src/rl_gym/zones.py` — `ZoneTable` with `cell_of_node`, `zone_set`, `zone_set_of_node`,
  `neighbours`, `cells`, `n_nodes`.
- `tests/test_rl_gym_zones.py` — 38 tests.
- `docs/SDPDP_GYM_SPEC_v2.md` — the approved sentence on the measured gap.
- `docs/RL_GYM_PROGRESS.md` — this entry.

No existing source file modified: `git diff --stat -- src/ tests/` is empty, everything under
`src/rl_gym/offline/`, `src/rl_gym/zones.py` and `tests/test_rl_gym_zones.py` is new. P2.5 is
not in §8's gate list (P2.0b, P2.3, P2.7–P2.14) and `zones.py` is unused until P2.7 and P2.9,
so no P1.10 gate was required; the full suite was run anyway.

Verification — everything passed, nothing failed:

1. **Full suite: 249 tests pass**, 118.7 s, no failures and no skips (211 before; +38).
   Re-run under `-W error::ResourceWarning` as well, to confirm the suite is warning-clean.
2. **Every figure P2.5 names, read back from the committed CSV** rather than from the builder's
   in-memory data, by an independent script: header `cell,neighbour`, **213 rows**, **37 cells**,
   **88 undirected pairs**, degree histogram **2:1, 3:9, 4:5, 5:5, 6:17** — so every cell has
   2–6 neighbours inside the set and **17 have all 6** — symmetric, reflexive, sorted, no
   duplicate rows.
3. **The geojson's 37 ids equal the `nodes.csv` set**, both directions, and every id in the
   table is one of them.
4. **`--check` passes**, comparing parsed `(cell, neighbour)` pairs. A test writes a CRLF copy of
   the table, asserts the two files differ **as bytes**, and asserts `read_table` returns the
   same pair set from both — Ritun's adjustment, exercised rather than just implemented.
5. `zone_set` sizes are `1 + degree`: 3 on one cell, 4 on nine, 5 on five, 6 on five, 7 on
   seventeen. `cell_of_node` matches `nodes.csv` on sampled nodes and accepts numpy ints and
   int-valued floats, which is what FleetPy positions carry.
6. **Unknown keys raise**, with the offending key in the message: an unknown node (−1, 4084,
   1e9), a non-numeric node, an unknown cell. `zone_set` never returns empty for any of the 37.
7. **Malformed tables are rejected at construction**, not at the first decision: missing file
   (the message names the build command), wrong header, asymmetric relation, missing self row,
   empty table, and a table that is internally consistent but does not cover every cell the
   network's nodes sit in.
8. **`h3` is not imported and not installed**, asserted three ways: importing `zones` and the
   builder leaves `h3` out of `sys.modules`; `import h3` raises; and a regex scan of every
   `*.py` under `src/rl_gym/` finds no `import h3` / `from h3` line (§10).

VERIFY findings: **P2.5 carries no `VERIFY` markers.** Every figure in its "How Ritun verifies"
list was checked and **matched**: 88 undirected pairs, 2–6 neighbours per cell, 17 cells with
all 6, symmetry, and the geojson/`nodes.csv` id agreement. Also checked and matched: 37 valid
single polygons, no null `h3_id`, `node_index` unique over 0–4083, 4,084 nodes mapped,
1–379 nodes per cell (median 68).

**The finding behind the spec sentence.** P2.5 describes the neighbour rule as "polygons lie
within 1e-9° of each other", which reads like a float tolerance chosen by judgement. Measured
over all 666 cell pairs: the 88 neighbour pairs are at shapely distance **exactly 0.0** —
adjacent hexagons in this geojson share identical vertex coordinates, so edge-sharing is exact,
not approximate — and the nearest **non**-neighbour pair is at **1.377e-02**. Every threshold
from 0 to 1e-2 yields the same 88 pairs. The rule is therefore doing no rounding at all, and the
tolerance sits in a gap seven orders of magnitude wide on either side. Three tests pin this: all
neighbours at exactly 0.0, every non-neighbour more than 1e6× the tolerance away, and the pair
count invariant across 1e-15 … 1e-3.

Two design choices worth recording:
- **`ZoneTable` is a class, not module-level functions with global state.** P2.5 says `zones.py`
  "exposes `cell_of_node(node)` and `zone_set(cell)`", which methods satisfy; a class lets a test
  point the loader at a temporary table, which is what the six malformed-table tests need.
  `zone_set` returns a `frozenset` and is precomputed per cell, so it is O(1) per decision —
  P2.7 calls it for every event and P2.9 for every request.
- **The self rows are what make `zone_set` mean "cell plus ring"** without the reader adding the
  cell back, and `zones.py` refuses a table missing one rather than silently returning a zone set
  that omits its own cell.

Spec changes made this turn: the one approved sentence in P2.5, recording the measured gap with
the numbers and an explicit "do not tighten or loosen it thinking it is delicate".

Nothing failed. **Stopping here** — not starting P2.6.

---

## P2.6 — STARTED 2026-10-10 17:19
Files I will touch: src/rl_gym/offline/build_arrival_baseline.py (new),
data/rl_gym/example_network/arrival_baseline.csv (new),
data/rl_gym/example_network/corpus_manifest.json (new),
tests/test_rl_gym_baseline.py (new), docs/SDPDP_GYM_SPEC_v2.md, docs/RL_GYM_PROGRESS.md

Resume state: P2.5 committed as 743fd5d, tree clean.

Approved before starting:
- **`c_d(t, W)` counts requests as `load_demand_file` keeps them — both filters.** 5 of the 7
  same-node days are training days (2024-02-15, 02-20, 03-05, 05-30, 06-13); counting their
  same-node rows would inflate 150 (month, minute) buckets at W=30 and 300 at W=60 by up to
  0.11 arrivals, and the online side cannot see them, since P2.7's arrival hook is
  `build_assignment_context`.
- Correcting the two figures that did not reproduce, plus verifying the other three in that
  paragraph, with the instruction: **if any re-measured figure no longer supports the
  month-over-weekday conclusion, stop and report rather than just updating the number.**
  It still supports it — see the DONE entry.
- The builder writes **both artifacts or neither**.
- **Ritun's adjustment: no metadata inside `arrival_baseline.csv`** — a comment line breaks a
  plain `pd.read_csv`. The manifest records a hash of the baseline next to `corpus_hash`, so the
  manifest ties corpus and baseline together while the CSV stays a plain table. The baseline
  hash is over **parsed content**, like `--check`, not bytes.

## P2.6 — DONE 2026-10-10 17:28 — awaiting Ritun's verification

Files changed:
- `src/rl_gym/offline/build_arrival_baseline.py` — **new.** Builds both artifacts or neither,
  with `--check`.
- `data/rl_gym/example_network/arrival_baseline.csv` — **new.** 14,940 rows + header,
  `month,t_seconds,window_s,mean,sd`. A plain table: no comments, no metadata.
- `data/rl_gym/example_network/corpus_manifest.json` — **new.** 101 files with hashes,
  `corpus_hash`, `baseline_hash`, the two vocabularies, and the window and floor constants.
- `tests/test_rl_gym_baseline.py` — **new**, 31 tests.
- `docs/SDPDP_GYM_SPEC_v2.md` — the `c_d` definition, five figure corrections, the
  manifest/baseline link, and a note on the verification's day choice.
- `docs/RL_GYM_PROGRESS.md` — this entry.

No existing source file modified. P2.6 is not in §8's gate list and touches no env code — the
online side is P2.11 — so no P1.10 gate was required; the full suite was run anyway.

Verification — everything passed, nothing failed:

1. **Full suite: 280 tests pass**, 119.4 s, no failures and no skips (249 before; +31). The new
   file was also run under `-W error::ResourceWarning`.
2. **`--check` passes on both artifacts**, comparing parsed content: the baseline matches a
   fresh build, the manifest's `corpus_hash` matches the directory, and its `baseline_hash`
   matches the committed CSV. P2.5's table still checks out too.
3. **P2.6's check 1 — `c_d` by hand.** For 3 training days × 3 minutes × 2 windows, the count is
   recomputed straight from the dataframe with both filters applied inline and compared against
   `counts_for_day`. All 18 match. A separate test pins the half-open window: an arrival exactly
   at `t` is excluded, one exactly at `t − W` included.
4. **P2.6's check 2 — October above the rest.** October's day-mean level is **22.45** arrivals
   per 30 min, the highest of the ten months (next: June 19.75, September 19.53; lowest August
   16.76), and October exceeds **every** other month at **412 of 747 minutes (55.2%)**. Reported
   numerically rather than as a plot: with 9–12 days per month, minute-level noise means no
   month dominates at every single minute, and the level comparison is the claim that matters.
5. **P2.6's check 3 — manifest hashes.** `files` equals a fresh directory scan entry for entry,
   sorted by filename, 101 entries, no evaluation day among them; `corpus_hash` equals the hash
   of the `<filename>:<sha256>` lines; and it changes if a file is removed, if one file's hash
   is altered, or if the order is reversed.
6. **The same-node exclusion, asserted directly.** The five training days are re-derived from
   the corpus (not hardcoded from P2.4) and match; `kept_arrival_times` returns one fewer
   arrival than the in-window rows on each, at exactly the right minute; and rebuilding one
   affected February bucket **both ways** shows the inflation is exactly **0.1** arrivals and
   that the committed baseline carries the correct value, not the inflated one.
7. `sd` is pooled, so it is constant across months for each `(minute, window)` — asserted over
   all 1,494 keys. Sample sd, `ddof = 1`. No NaN anywhere; every mean and sd non-negative; the
   60-minute mean is never below the 30-minute mean; rows sorted.
8. **The SD floor is not decoration**: `sd` at `W = 30` runs min 0.000, median 4.678, max
   10.148, and **6 of 747 minutes fall below the 1.0 floor** — including `t = start_time`, where
   the trailing window lies entirely before the day and every day's count is 0.
9. A `.DS_Store` and a `notes.txt` dropped into a copy of the corpus change neither the day list
   nor `corpus_hash` (§2.9's `*.csv`-only rule).
10. A stale-manifest scenario is exercised: regenerate the baseline, leave the manifest, and
    `baseline_hash` no longer matches — which is the failure that field exists to catch.

**The five "Why month" figures Ritun asked me to verify — the conclusion holds.** Ritun's
instruction was to stop and report if any re-measured figure no longer supported the
month-over-weekday choice. It still supports it, and slightly more strongly:

| Figure | Spec said | Measured |
|---|---|---|
| residual spread, month baseline | 4.9 | **4.75** |
| residual spread, weekday baseline | 5.3 | **5.20** |
| residual spread, no baseline | 5.6 | **5.45** |
| residual spread, weekday shape × month level | 5.07 | **4.91** |
| October vs a weekday baseline, whole day | ~+0.7 SD | **+0.69** |
| … at 10:00 | +0.66 | **+0.58** |
| … at 15:30 | +0.78 | **+0.84** |
| 15:30 window, Tuesdays | 22.1 | **21.7** |
| 15:30 window, overall | 27.2 | **27.2** |
| month share of daily-total variance | 50% | **56.0%** |
| weekday share | "none" | **0.2%** |
| weekday daily means | 453–454 | **463.4 / 468.8 / 465.6** |

Same ordering throughout, and the month-over-weekday gap is **wider** than the original figures
showed (0.45 against 0.40). The five that were off all predate the October days landing in the
corpus (`6eeb05e`); October runs 507–600 against about 450 earlier (§2.9), which raises both the
weekday means and month's explanatory share. Recorded in the spec with that explanation.

One figure I added rather than corrected: in the 15:30 window **Wednesdays average 37.3** against
Tuesdays' 21.7. The spec cited only the Tuesday half, which understates the case for keeping the
weekday one-hot in the observation — a 15.6-arrival spread between two weekdays at the afternoon
peak is the strongest evidence in the paragraph for that feature, and §5.2 keeps it.

**A mistake of mine, caught by its own test.** My first version of the by-hand check picked
2024-05-15 as a "random training day". It is an **evaluation** day, so the test failed with a
`FileNotFoundError` rather than a wrong number. Fixed by picking from `corpus/` and asserting
membership in `training_day_files()` first, so the probe days cannot silently drift out of the
training split. Added the same caution to P2.6's "How Ritun verifies" line, since the next person
to pick three days by eye will hit the same trap.

VERIFY findings: **P2.6 carries no `VERIFY` markers.** The claims it rests on were checked and
are recorded above — the two filters, the 747-minute grid, the 101 training days and their
9–12-per-month distribution, the vocabularies, and all twelve "Why month" figures.

Spec changes made this turn:
- **`c_d`'s definition now names both of the loader's filters**, with the five affected training
  days, the measured inflation (150 buckets at W=30, 300 at W=60, up to 0.11 arrivals) and the
  reason it matters: comparability with what P2.7 observes is the whole point of sharing §5.2's
  window definition. Marked as corrected in P2.6.
- **The "Why month, not weekday" paragraph re-measured**, with all figures updated, the
  in-sample/parameter-count caveat stated, and a note that the stale figures predate October.
- **The manifest paragraph** now records `baseline_hash`, the both-or-neither rule, that the hash
  is over parsed rows rather than bytes, and **why it lives in the manifest rather than in the
  CSV** — a comment or metadata line would break a bare `pd.read_csv`.
- **P2.6's verification line** now warns to pick the three days from `corpus/`.

Nothing failed. **Stopping here** — not starting P2.7.

## P2.6 — FOLLOW-UP 2026-10-10 17:41 — month-only baseline kept, bias recorded

Ritun: keep the month-only baseline; record the Wednesday peak shift as a known bias of it, and
add a weekday-by-time adjustment to "After Phase 2" as an option to consider before full
training. Documentation only — `git diff --name-only` is the two docs files, no code touched.

**The +1 SD figure, measured against the baseline actually in use** rather than taken on trust.
Against the **month-only** baseline, in the window ending 15:30: Wednesdays **+1.07 SD**
(worst +1.09 at 15:25), Tuesdays **−0.58**, Thursdays **−0.45**, pooled sd 9.19 at that minute.
Across the 15:00–16:00 band, Wednesday **+0.62**. Over the whole day the weekday bias is
**negligible — −0.02, +0.02, +0.01** — which is the useful half of the finding: month-only is
sound in general, and the defect is confined to about an hour a day.

Spec changes made this turn:
- **§8 P2.6's "Why month" paragraph** gained a "Known bias of the month-only baseline, accepted
  deliberately" note with those figures, stating that `arrivals_z_*` is reliably positive on
  Wednesday afternoons and negative on the other two weekdays for a reason the anomaly feature
  itself cannot express, that the whole-day bias is negligible, and that the weekday one-hot is
  what lets the policy discount it.
- **§8 "After Phase 2" gained a step 4**, the weekday-by-time adjustment, written as an option
  rather than a decision, with the cost stated: 30 weekday×month cells over 101 training days is
  3–4 days per cell, too few for a per-minute mean, so any version needs smoothing or a scaled
  weekday shape — and the scaled shape measured **4.91** residual spread against month-only's
  **4.75**, i.e. **worse in-sample**, so it is explicitly not a drop-in improvement. Also records
  that adopting it rebuilds the baseline and the manifest, changes `baseline_hash`, and
  invalidates anything trained against the old baseline (trap 16), which is why it belongs
  before full-length training.
- **Renumbering, which the insertion forced**: the old steps 4–6 became 5–7, and the trailing
  "Steps 1, 3 and 4 change the simulation" became "Steps 1, 3 and 5", with a new sentence
  distinguishing step 4 — it changes no simulation output but invalidates trained policies.
  Checked that nothing else in the spec cites these steps by number; the two other references to
  "After Phase 2" (lines ~315, ~1040) name the section, not a step.

Re-ran `TestMonthOverWeekday` (4 tests, OK) to confirm the figures written into the spec still
match what the code measures. **Stopping here** — not starting P2.7.

## P2.6 — FOLLOW-UP 2 2026-10-10 17:45 — step 4 rewritten as the additive weekday term

Ritun corrected my step 4: the option is not per-(weekday, month) cells and not a weekday shape
scaled by month, it is **additive** —

    expected(m, wd, t, W) = mean(m, t, W) + δ(wd, t, W)

with `δ(wd, t, W)` the mean, over all training days of that weekday (33–35 days, pooled across
months), of `c_d(t, W) − mean(month(d), t, W)`. No weekday×month cells, no smoothing. He also
supplied leave-one-training-day-out figures to verify. Documentation only; `git diff
--name-only` is the two docs files.

**His LOO figures reproduce on training days only**, refitting month means *and* `δ` on the
other 100 days each time (rms residual, month-only → additive):

| | whole day | 14:00–17:00 |
|---|---|---|
| `W = 30` | 5.50 → **5.18** (his: 5.50 → 5.18) | 7.08 → **6.10** (his: 7.09 → 6.11) |
| `W = 60` | 8.15 → **7.72** (his: 8.15 → 7.72) | 10.69 → **9.32** (his: 10.70 → 9.33) |

Four exact, two off by 0.01 — band-edge or rounding. Better out of sample on both windows and
both ranges. Weekday counts confirmed: Tue 35, Wed 33, Thu 33, so `δ` is estimated from 33–35
days rather than the 3–4 a weekday×month cell would hold.

**One instruction I did not follow as given, with the evidence.** Ritun asked me to "keep the
scaled-shape figure as the variant that does not help". Measured on *his* leave-one-out metric,
the scaled shape **does** help: 5.26 whole day and 6.38 in the band at `W = 30`, against
month-only's 5.50 and 7.08. It only looked unhelpful on the in-sample per-minute-SD metric I had
used earlier in P2.6 (4.91 against month-only's 4.75), where it is indeed worse. The two metrics
therefore **disagree about the scaled shape** and **agree about the additive term**, which is the
stronger reason to prefer the additive form than "the other one does not work".

So step 4 now says the scaled shape is the **weaker variant**, gives both metrics' verdicts on
it, and states plainly that the additive term wins on both. It also warns that the two metrics
are not comparable — one is an rms over every day-minute residual, the other a per-minute SD
averaged over the day — so nobody reads 5.18 against 4.75 as a comparison. Writing "does not
help" would have put a claim in the spec that my own verification contradicts, which §0 forbids
working around silently.

Spec changes made this turn:
- **§8 "After Phase 2" step 4 rewritten**: the additive formula, what `δ` is and why pooling
  across months is what makes it estimable, the verified LOO table, the scaled-shape comparison
  on both metrics with the non-comparability warning, and the retained cost note — a weekday
  dimension in `arrival_baseline.csv`, both artifacts rebuilt, `baseline_hash` changed, anything
  trained against the old baseline invalidated (trap 16). Step numbering 1–7 and "Steps 1, 3 and
  5 change the simulation" re-checked and still correct.
- **§10's after-Phase-2 line** now names four items instead of three, adding the weekday term,
  with "P2.6's baseline stays month-only — do not add the weekday term, and do not add a config
  key for it."

**Stopping here** — not starting P2.7.

---

## P2.7 — STARTED 2026-10-10 18:00
Files I will touch: src/rl_gym/rolling_stats.py (new), src/rl_gym/reward.py,
src/rl_gym/fleetctrl_rl.py, src/rl_gym/sim_env_rl.py, src/rl_gym/gym_env.py,
tests/test_rl_gym_rolling_stats.py (new), docs/SDPDP_GYM_SPEC_v2.md,
docs/RL_GYM_PROGRESS.md

Resume state: P2.6 committed as e1e61be, tree clean.

VERIFY confirmed before starting: the single entry point every loaded request passes through is
`build_assignment_context` — `step_generator` calls it unconditionally for every
`(rid, rq_obj)` (`sim_env_rl.py:90`), before any branch, so it fires for the internally-resolved
cases too.

Approved before starting:
1. **The arrival hook lives in `fleetctrl_rl.py`**, right after `self.rq_dict[rid_struct] = prq`
   and *before* the `o_pos == d_pos` early return — the `PlanRequest` with the origin node is
   created there, and hooking the operator covers both the gym generator and `user_request`'s
   greedy fallback. P2.7's Files line stops hedging.
2. **Trap 13 enforced by construction**: the §6.4 ladder becomes a module-level
   `classify_exit(...)` in `reward.py`, `RewardTracker.on_exit` **returns** the outcome it
   already computes, and the exit fan-out passes that one outcome to the stats tracker. One
   classification per exit, so the two trackers cannot label the same exit differently.
3. The clock is read only by declines and post-match cancellations, both of which fire inside
   the step loop; pickups are timed by `rq.pu_time`. The clock is stale during the end-of-day
   finalisation and that is harmless — asserted in a test rather than left as reasoning.
4. Pickups are stored **per rid**, replaced not appended, so a repeat `record_boarding` leaves
   one wait-gap sample at the final `pu_time` (§6.4, §5.2).
5. The "a few dozen events" figure corrected to the computed values.

Ritun's two conditions:
1. **The arrival hook must be a no-op when no rolling-stats tracker is attached**, since the
   P1.3/P1.4 comparisons run the RL operator outside the gym. Test required.
2. **`RewardTracker.on_exit` is idempotent per rid: a repeated call must return `None`, and the
   fan-out must skip it**, so the stats tracker records each exit at most once. Test required,
   with a rider whose exit fires twice.

## P2.7 — DONE 2026-10-10 18:15 — awaiting Ritun's verification

Files changed:
- `src/rl_gym/rolling_stats.py` — **new.** `RollingStatsTracker`: `set_clock`, `on_arrival`,
  `on_exit`, `on_pickup`, `count`, `zone_count`, `wait_gap_mean`, `episode_summary`.
- `src/rl_gym/reward.py` — the §6.4 ladder extracted to a module-level `classify_exit`;
  `RewardTracker.classify` delegates to it; `on_exit` now **returns** the outcome, or `None` on
  a repeat.
- `src/rl_gym/fleetctrl_rl.py` — `set_stats_tracker`, and the arrival hook in
  `build_assignment_context`.
- `src/rl_gym/sim_env_rl.py` — a class-level `_stats_tracker = None`, `set_stats_tracker`, and
  the clock at the top of `step_generator`.
- `src/rl_gym/gym_env.py` — `_zone_table`, `_make_quoted_wait_fn`, `_on_pickup`, `_on_exit`;
  wiring in `reset()`; the stats totals in `episode_summary()`.
- `tests/test_rl_gym_rolling_stats.py` — **new**, 36 tests.
- `docs/SDPDP_GYM_SPEC_v2.md` — the VERIFY findings, the Files line, the event-rate figure.
- `docs/RL_GYM_PROGRESS.md` — this entry.

Verification — all four of P2.7's checks passed, nothing failed:

1. **The P1.10 gate, byte for byte** — md5 `53879ea79cf211d700f670dee1173dd5`. The arrival hook
   sits inside `build_assignment_context`, on the live path, so this is the check that matters
   here. Full suite: **316 tests pass**, 137.3 s, no failures and no skips (280 before; +36).
2. **End-of-day totals match P1.10 exactly**: 445 arrivals, 118 declines, 28 cancellations, 288
   distinct pickups, 289 boarding calls, and 0 rids whose origin could not be placed in a cell.
3. **20 random decision times against an independent recomputation from
   `1_user-stats.csv`: 0 mismatches** across 20 times × 16 counts × 2 wait-gap means — that is
   fleet-wide and zone-set counts for all four event kinds at both windows, plus both means.
   Drawn from the 364 of 436 decisions whose windows cannot contain rid 50's overwritten first
   boarding, per P2.7's caveat.
4. **Fleet-wide arrival counts at every decision**, against P2.6's definition recomputed from
   the demand file with both loader filters: **0 mismatches across 436 decisions × 2 windows**.

Ritun's two conditions, both tested:
1. **The arrival hook is a no-op when unattached.** Three tests: the operator and the sim env
   both default to `None`; and a real simulation is built, the tracker detached from both, 25
   steps driven, and the arrival total asserted unchanged. This is the path P1.3, P1.4 and the
   P1.10 gate run, so it is load-bearing for three byte-for-byte comparisons.
2. **A repeated exit returns `None` and the fan-out skips it.** `on_exit` returns the outcome on
   the first call and `None` on the second and third; a test drives the env's fan-out pattern
   three times over one rid and asserts the stats tracker recorded exactly one decline. The
   integration tests additionally assert the two trackers agree on every exit count —
   `stats_declines == rider_declined`, `stats_cancels == diffusion_cancelled`,
   `stats_pickups == pickups`, `stats_boarding_calls == boarding_calls`.

Beyond those: the half-open window on both ends; that a request is not counted in its own
observation (`rq_time == sim_time` at its arrival step); pruning never dropping an event still
inside the longest window, and bounding memory over 5,000 arrivals while the cumulative total
stays exact; exits timed by the clock rather than `rq_time`; the per-rid pickup store replacing
rather than appending on rid 50's double boarding, leaving one sample at the final `pu_time`;
negative wait gaps (the common case here — the planner assumes 300 s per stop against a realized
~56 s); an unpriced pickup counting but staying out of the mean; a raising quote lookup not
killing the simulation; and an unplaceable origin counting fleet-wide but in no zone.

VERIFY findings:
- **The single entry point every loaded request passes through is `build_assignment_context`:
  matched the spec.** `step_generator` calls it unconditionally for every `(rid, rq_obj)` at
  `sim_env_rl.py` ~90, before any branch, so internally-resolved requests are arrivals too;
  §2.2's always-empty `get_undecided_travelers` makes it exactly once per request. Recorded in
  §8 P2.7, and the Files line no longer hedges about where the hook goes.
- **Which column carries each event's time: matched** — declines at `rq_time` (all 118 have an
  empty `decision_time`), post-match cancellations at `decision_time`, pickups at
  `pickup_time`, the quoted wait as `t_wait` in `offers`, and `diffusion_decision_time`
  differing from `decision_time` in 4 of the 28 cancellations.
- **The quoted wait is `prq.offer.offered_waiting_time` and is not overwritten after
  acceptance: matched.** `_create_user_offer` builds it as `pu_time - prq.rq_time`, already
  relative to `rq_time`; the only callers in this scenario are `commit_assignment_choice` and
  the parent's unused `user_request`, and nothing re-offers. A rejected request carries a
  `Rejection` with `offered_waiting_time = None`.
- **The "a few dozen events in a 60-minute window" figure: DIFFERED** — understated by about
  5×. P2.6's busiest (month, minute) averages **73.7** arrivals in a 60-minute window and
  reaches about **115** at +3 SD; with the other three kinds a window holds roughly **200**
  events. The scan design is unaffected and the spec now carries the computed figures, since
  that number is the stated justification for it.

Two things worth recording:
- **`_stats_tracker` on the sim env is a class-level default, not an `__init__` assignment.**
  `RLImmediateDecisionsSimulation` defines no `__init__`, and P1.4 forbids adding a parameter to
  `FleetSimulationBase.__init__`, so a class attribute is the way to give it a safe default
  without touching construction.
- **`ZoneTable` is built once per env and shared across episodes**, while the tracker is rebuilt
  every reset. The table is static data — `nodes.csv` plus a committed CSV — so §10's
  no-caching rule, which is about simulation state leaking between episodes, does not reach it.

**A mess of mine, cleaned up.** My scratchpad verification script crashed twice before reaching
its `shutil.rmtree` — first because it read `env._stats` before `reset()` created it, then
because it parsed the user-stats `start` column as a bare node when it is a position string like
`1169;-1;-1`. Each crash left a result directory behind (`…_env0_pid13776_ep1`,
`…_env0_pid13804_ep1`); both removed, and `example_depot_time_pool_irsonly_rs_42/` confirmed
still at md5 `53879ea7…`. Neither was a defect in the item, but the cleanup belonged in a
`finally` and did not start there. The position-string detail is now recorded in P2.7's
verification line, since the next person writing that recomputation will hit it.

Nothing failed. **Stopping here** — not starting P2.8.

## P2.7 — FOLLOW-UP 2026-10-10 18:25 — the tracker slot is per instance, verified

Ritun asked, before committing, whether the class-level default is only ever *shadowed* by an
instance attribute and never written on the class — because a class-level write would hand a
dead tracker to every later simulation built in the same process, including P1.4's driver, which
constructs `RLImmediateDecisionsSimulation` directly with no gym and no tracker. Nothing would
raise; the stale tracker would simply keep recording.

**Verified, not reasoned about. It is per instance.**

- `set_stats_tracker` does `self._stats_tracker = tracker` (`sim_env_rl.py:51`), which creates
  an instance attribute shadowing the class default.
- Measured on the class: before attaching, `_stats_tracker` is absent from the instance `__dict__`
  and reads `None` from the class; after attaching it is present in the instance `__dict__`, the
  instance sees its own tracker, and **the class attribute is still `None`**. A fresh instance in
  the same process reads `None` with a clean `__dict__`. Detaching leaves the class alone too.
- A grep over `src/` and `tests/` finds **no assignment to the class attribute anywhere**: the
  only writes are the two `self._stats_tracker = tracker` setters, and the only call sites are
  `gym_env.py:438-439`, both on instances (`op`, `self.sim`).

Four new tests pin it, in a class named for the concern —
`TestTheTrackerSlotIsPerInstanceNotPerClass`: attaching shadows rather than writes; a later
instance in the same process sees no tracker; detaching also leaves the class alone; and an
end-to-end version building **two real envs in one process**, asserting each sim points at its
own tracker, that the two trackers are distinct objects (§10 rebuilds per reset), that the class
slot is still `None` with two live simulations, and that a third simulation built afterwards
still starts clean.

**A weak test of mine, found while answering and fixed.**
`test_operator_defaults_to_no_stats_tracker` read

    self.assertIsNone(RLPoolingIRSOnly._stats_tracker
                      if hasattr(RLPoolingIRSOnly, "_stats_tracker") else None)

The operator sets the slot in `__init__`, so the class has no such attribute, the `hasattr`
guard was False, and the assertion was `assertIsNone(None)` — it would have passed no matter
what the code did. That is precisely the decoration §0 rules out. Replaced with an assertion of
the actual structure: the name is **absent** from `vars(RLPoolingIRSOnly)` and **present** in
`__init__`'s code names, so the operator's slot is per instance by construction rather than by
a vacuous check. The sim env's test now also asserts the class attribute genuinely exists before
asserting it is `None`.

Full suite after the additions: **320 tests pass**, 137.0 s, no failures and no skips (316
before; +4). No stray result directories. Tests only — no source file changed, so the P1.10
gate result from the DONE entry above still stands.

**Stopping here** — not starting P2.8.

---

## P2.8 — STARTED 2026-10-10 18:31
Files I will touch: src/rl_gym/observers.py, src/rl_gym/gym_env.py, src/rl_gym/policies.py,
src/rl_gym/spaces.py, src/rl_gym/tools/corpus_sweep.py, src/rl_gym/tools/reward_trace.py,
tests/test_rl_gym.py, tests/test_rl_gym_observers.py, tests/test_rl_gym_rolling_stats.py,
docs/SDPDP_GYM_SPEC_v2.md, docs/RL_GYM_PROGRESS.md

Resume state: P2.7 committed as f1189dc, tree clean.

Approved before starting:
- **Two files P2.8's Files line omits**, both callers of the function whose signature changes,
  both created after §8's P2.8 text was written: `src/rl_gym/tools/reward_trace.py` (P2.0b) and
  `tests/test_rl_gym_rolling_stats.py` (P2.7). Without them the item does not compile.
- The layout **raises on a duplicate feature name** — P2.9 and P2.10 both add blocks, and a
  collision would leave one feature unreachable by name while the vector still had the right
  length.
- `src/rl_gym/README.md` is left for its post-Phase-2 revision (commit 74ee9c0), so its
  `greedy_action(obs, mask, k_max)` and `obs[K:2K]` lines go stale on purpose.

**Ritun's addition: `greedy_action` must also read `is_valid` by name and raise unless it
exactly equals the mask.** Two failures at once:
  1. It enforces trap 12 on **every gate run**, not just in the unit tests — mask and
     `is_valid` must come from one source, and now a drift stops the gate.
  2. It **refuses a `VecNormalize`'d observation** (P2.12). After normalization `is_valid` is no
     longer 0/1, and more importantly each `delta_cfv` slot is scaled by its **own** running
     mean and variance, so an argmin across slots would compare z-scores from different
     distributions rather than costs — plausible actions, silently wrong. The `is_valid` check
     is what catches that, since a normalized `is_valid` cannot be exactly 0/1.
  Test required: a normalized-looking `is_valid` raises.

## P2.8 — DONE 2026-10-10 18:41 — awaiting Ritun's verification

Files changed:
- `src/rl_gym/observers.py` — `AbstractObserver.feature_spec()`; `output_length` derived from
  it; `observe_vector` raises when `observe()` disagrees with the spec; new
  `build_observation_layout`; both observers declare their specs; the padding docstring
  replaced with §5.1's.
- `src/rl_gym/gym_env.py` — publishes `observation_layout`, built from the observers' specs at
  construction; `obs_len` summed from the same observers tuple.
- `src/rl_gym/policies.py` — `greedy_action(obs, mask, layout)` and
  `greedy_choice(obs, mask, layout)`; `delta_cfv` and `is_valid` read by name; the hardcoded
  `(3K + 4,)` check gone; the `is_valid`-equals-mask enforcement added.
- `src/rl_gym/spaces.py` — the "k-th cheapest candidate" docstring replaced with D3's exact
  ordering language.
- `src/rl_gym/tools/corpus_sweep.py`, `src/rl_gym/tools/reward_trace.py`,
  `tests/test_rl_gym.py`, `tests/test_rl_gym_rolling_stats.py` — callers updated.
- `tests/test_rl_gym_observers.py` — callers updated; two new test classes, 18 tests.
- `docs/SDPDP_GYM_SPEC_v2.md` — the Files line, the derived `output_length`, the duplicate-name
  raise, and the `is_valid`/`VecNormalize` rule.
- `docs/RL_GYM_PROGRESS.md` — this entry.

Verification — everything passed, nothing failed:

1. **The P1.10 gate, byte for byte** — all 7 gate tests `ok`, including
   `test_user_stats_matches_baseline_byte_for_byte`. Full suite **340 tests pass**, 142.4 s, no
   failures and no skips (320 before; +20).
2. **The layout's slices tile `[0, obs_len)` with no gap or overlap**, asserted two ways:
   consecutive slices meet exactly and the last ends at `obs_len`; and every index is covered
   exactly once. Also that the layout's names and order equal the concatenated declarations,
   that the Phase 1 positions are `is_valid` 0:8, `delta_cfv` 8:16, `offered_wait` 16:24,
   `idle_fraction` 27:28, and that each slice addresses the values `observe()` actually
   produced in a real vector.
3. **`check_env` through `MaskRespectingShim` passes with `base_seed` absent and set** — both
   tests `ok`.
4. Both tools re-run after the signature change: `corpus_sweep.py` on the reference day reports
   0 reconciliation failures, 0 unclassified, and `2024-06-20 vs P1.10: MATCHES`.

Ritun's addition, tested eight ways: an agreeing mask and `is_valid` are accepted for every
candidate count 0..K; a mask that disagrees raises and names trap 12; **a normalized
observation raises**, with the message naming `VecNormalize`; a wholly normalized vector raises
too; `0.999` is rejected, because the check is exact on purpose; a layout missing `delta_cfv`
or missing `is_valid` raises rather than guessing; and an observation too short to cover the
slices raises.

Also tested: a duplicate feature name raises and names the colliding feature; a non-positive
declared length raises; and an `observe()` that renames a feature raises from `observe_vector`,
naming `feature_spec`, rather than silently shifting every later slice.

VERIFY findings: **P2.8 carries no `VERIFY` markers.** Two findings of its own:
- **The Files line omitted two callers**, both created after §8's P2.8 text was written:
  `src/rl_gym/tools/reward_trace.py` (P2.0b) and `tests/test_rl_gym_rolling_stats.py` (P2.7).
  The item does not compile without them. Added, with the reason noted in the spec.
- **`tests/test_rl_gym_env.py:61` defines its own local `greedy_action(env, obs)`** — a
  different function with a different signature, unaffected by this change. Checked rather than
  assumed, since a global rename would have been wrong there.

One consequence worth stating plainly, because it is a deliberate loss of a check:
**removing the `(3K + 4,)` shape assertion means a trailing-truncated observation is no longer
rejected.** That is the point of the item — the layout's slices are the contract, so a vector
that covers them reads correctly and P2.9–P2.11 can widen it without touching any consumer. A
test pins the new behaviour explicitly (`greedy_action` gives the same answer on a vector with
the global block dropped), so the change is recorded as intended rather than discovered later.
What replaces it is narrower and more useful: the slices must be in range, the mask must match
the derived `K`, and `is_valid` must equal the mask.

**A test of mine that asserted the old contract.** `test_shape_mismatches_raise` required
`greedy_action(obs[:-1], ...)` to raise — true only while the total length was checked, which
this item removes. It failed, correctly. Split into three: a wrong-length mask raises, an
observation that does not cover the layout raises, and the total length is deliberately no
longer the contract.

`src/rl_gym/README.md` is now stale on two lines — it documents `greedy_action(obs, mask, k_max)`
and the `obs[K:2K]` slice. Left for the post-Phase-2 revision commit 74ee9c0 promises, as agreed.

Nothing failed. **Stopping here** — not starting P2.9.

---

## P2.9 — STARTED 2026-10-10 19:34
Files I will touch: src/rl_gym/corpus.py (new), src/rl_gym/observers.py,
src/rl_gym/day_rotation.py, src/rl_gym/offline/build_arrival_baseline.py,
src/rl_gym/gym_env.py, tests/test_rl_gym_observers.py, tests/test_rl_gym_env.py,
docs/SDPDP_GYM_SPEC_v2.md, docs/RL_GYM_PROGRESS.md

Resume state: P2.8 committed as d0dc3a7, tree clean.

VERIFY confirmed before starting: the `PlanRequest` origin and destination attributes are
`prq.o_pos` / `prq.d_pos` (`PlanRequest.py:55,59`), node at element `[0]`, and `prq.nr_pax`
(line 47). Both default to `rq.o_pos` / `rq.d_pos` because `build_assignment_context` passes no
`pickup_pos`/`dropoff_pos`, so they are the raw request nodes. No boarding-point or walking keys
are configured, despite the control function being named `distance_and_user_times_with_walk`.

**Ritun's coordinate-scaling instruction, measured:** scaling by `return_network_bounding_box()`
would put **13 of 4,084 nodes outside [−1, 1]** in latitude, reaching −1.0106 and +1.0222,
because P2.1 showed the reprojected UTM corners fall *inside* the node latitude hull. Scaling by
`nodes.csv`'s own min/max puts the extremes exactly at ±1. §5.2 already specifies the node hull,
so this reinforces the spec; the test Ritun asked for pins it.

Approved before starting:
- **`src/rl_gym/corpus.py`**, a shared runtime module, because `offline/__init__.py` forbids
  `src/rl_gym/` importing from `offline/` and the alternative is two definitions of
  `corpus_hash` that agree today and drift later — the exact failure the manifest exists to
  prevent. The offline builder imports it, so the dependency still runs one way.
- The observer gets P2.7's tracker through an explicit `set_stats_tracker`, called from
  `reset()`, rather than reaching into `fleetpy_module._stats_tracker`.
- **The P1.10 gate now depends on corpus integrity** (trap 16): the manifest check runs at
  construction unconditionally, so a corpus edit breaks the byte gate even though the gate reads
  `thursday/`. The raise names the offending files so the message explains itself.

Ritun's two conditions on `corpus.py`:
1. After the builder imports from it, `build_arrival_baseline --check` must pass and the
   committed hashes must be unchanged. Recorded here for the comparison:
   `corpus_hash   6b232cf105d05ad3ccc6c9121e845dace88e406471e3bbd04286edc5dabb85a3`
   `baseline_hash 5638cb266b54615492c85cb52de1be49faefde16cfe55e532f623fe5913622e1`
2. **The same single-definition argument applies to listing training days.** There are three
   copies of the `*.csv` listing today — `day_rotation.py:45`, `build_arrival_baseline.py:81`
   and `corpus_sweep.py:94` — so D10's rotation, the manifest and the corpus check can disagree
   about what a training day is. `day_rotation.training_days()` will delegate to `corpus.py`,
   and `day_rotation.py` joins the Files line.

## P2.9 — DONE 2026-10-10 19:49 — awaiting Ritun's verification

Files changed:
- `src/rl_gym/corpus.py` — **new.** The single definitions: `training_day_files`,
  `file_sha256`, `corpus_file_hashes`, `corpus_hash`, `load_manifest`, `verify_corpus`.
- `src/rl_gym/day_rotation.py` — `training_days` now delegates to `corpus.py`.
- `src/rl_gym/offline/build_arrival_baseline.py` — imports the four shared functions instead of
  defining its own.
- `src/rl_gym/observers.py` — **new** `NodeCoordinates` and `RequestObserver` (24 features).
- `src/rl_gym/gym_env.py` — `corpus_dir` config key; the manifest load and corpus check at
  construction; the request observer built and wired; the observation concatenated from the
  observers tuple so the new block is picked up; `_zones` initialised before the observers need
  it.
- `tests/test_rl_gym_observers.py` — `TestNodeCoordinateScaling` (6) and `TestRequestObserver`
  (11).
- `tests/test_rl_gym_env.py` — `TestCorpusCheckAtConstruction` (7) and
  `TestRequestBlockInTheEnv` (9); the two existing shape assertions updated to 52.
- `docs/SDPDP_GYM_SPEC_v2.md` — the Files line and `corpus.py`'s rationale, the bbox rule with
  the measurement, the injected-dependency paragraph, and the gate-coupling note.
- `docs/RL_GYM_PROGRESS.md` — this entry.

Verification — everything passed, nothing failed:

1. **The P1.10 gate, byte for byte, with the block present** — md5
   `53879ea79cf211d700f670dee1173dd5`. Full suite **373 tests pass**, 145.9 s, no failures and
   no skips (340 before; +33).
2. **Observation length 28 → 52**, the request block contributing exactly 24, the layout still
   tiling `[0, 52)` with no gap or overlap, and the block order candidate → request → global
   (§5.2).
3. **Every request feature at 10 random decisions, against an independent hand computation:
   0 mismatches** across 10 decisions × 17 checks — the four coordinates recomputed from
   `nodes.csv` with the hull applied by hand, `n_pax` from the `PlanRequest`, both one-hots, and
   all six zone counts recomputed from `1_user-stats.csv` through P2.5's zone sets.
4. **Ritun's scaling test: all 4,084 nodes land within [−1, 1]**, and the extremes sit exactly
   at ±1. The counter-example is pinned too: scaling by `return_network_bounding_box()` puts
   **13** nodes outside, at −1.0106 and +1.0222, so a switch to the helper fails a test rather
   than drifting.
5. **The corpus check**, each against a temporary copy so the committed corpus is never touched:
   an untouched copy constructs; **removing** a file raises and names it; **editing** one byte of
   a file raises and names it; **adding** a day raises and names it; a missing directory raises;
   the message contains both the consequence ("invalid") and the fix
   ("build_arrival_baseline"); and **a stray `.DS_Store` plus a `notes.txt` leave construction
   working** — §2.9's `*.csv`-only rule, the test P2.2 promised.
6. An unknown weekday or month raises at `set_episode_day`, i.e. at reset, over seven cases
   including `None` (what a non-ISO filename yields), a lowercase `"thu"`, and month 0.
   `observe` raises if either per-episode dependency is missing, and the request observer is
   re-pointed at the current episode's tracker on every reset.
7. `check_env` through `MaskRespectingShim` passes with `base_seed` absent and set (both in the
   suite).

**Ritun's condition 1 — the refactor moves neither hash.** After the builder started importing
from `corpus.py`:
- `build_arrival_baseline --check` passes: "the committed baseline and manifest match a fresh
  build, and agree with each other".
- `corpus_hash` **6b232cf105d05ad3ccc6c9121e845dace88e406471e3bbd04286edc5dabb85a3** — identical
  to the value recorded in the STARTED entry, to the committed manifest, and to a fresh
  recomputation through `corpus.py`.
- `baseline_hash` **5638cb266b54615492c85cb52de1be49faefde16cfe55e532f623fe5913622e1** — the
  same three ways.
- `git status -- data/rl_gym/` is empty: `--check` wrote nothing.

**Ritun's condition 2 — one definition of a training day.** There were three `*.csv` listings
(`day_rotation.py:45`, `build_arrival_baseline.py:81`, `corpus_sweep.py:94`), so D10's rotation,
the manifest and the corpus check could have disagreed about what a training day is.
`day_rotation.training_days` and the builder now both call `corpus.training_day_files`, and a
test asserts the two return the same 101 days. `corpus_sweep.py:94` is deliberately left alone:
it enumerates **two** directories as `(day_dir, day)` pairs including `corpus_eval/`, which is a
different question from "what are the training days", and folding it in would have meant giving
the shared function a second shape for one caller.

VERIFY findings:
- **The `PlanRequest` origin and destination attributes: matched the spec** — `prq.o_pos` and
  `prq.d_pos` (`PlanRequest.py:55,59`), node at element `[0]`, and `prq.nr_pax` at line 47. Both
  default to `rq.o_pos`/`rq.d_pos` because `build_assignment_context` passes no
  `pickup_pos`/`dropoff_pos`, so they are the raw request nodes. **No boarding-point or walking
  keys are configured**, despite the control function being named
  `distance_and_user_times_with_walk` — checked, because the name invites the opposite
  conclusion.
- **The bbox source: DIFFERED from what a reader might assume**, and Ritun's instruction is the
  correct one. §5.2 already said "the min/max over all nodes", but P2.1's verification figures
  come from `return_network_bounding_box()`, and using that helper here would put 13 nodes
  outside [−1, 1]. Both the rule and the measurement are now in §8 P2.9 so the two cannot be
  conflated again.

Two mistakes of mine, both caught by their own tests:
- **An initialisation-order bug.** `_zones` was set in the "live episode state" block, *after*
  the observers that ask for it at construction, so the first `SDPDPAssignmentEnv(...)` raised
  `AttributeError: no attribute '_zones'`. Moved above the observers, with a comment saying why
  the position matters.
- **A test helper that skipped the case it was testing.** My
  `test_an_unknown_weekday_or_month_raises_at_set_episode_day` went through a helper that only
  calls `set_episode_day` when the weekday is not `None` — so the `(None, 6)` case never called
  it and the assertion failed. Rewritten to call `set_episode_day` directly over seven cases.
  `None` is precisely the case that most needs covering, since it is what P2.3's
  `parse_day_stem` returns for a non-ISO filename.

Also hit and fixed: my new env test called `greedy_action` and got
`tests/test_rl_gym_env.py`'s **local** `greedy_action(env, obs)` helper rather than the policy
function — the name collision flagged in P2.8. Now imported as `policy_greedy_action`, with a
comment at the call site.

Nothing failed. **Stopping here** — not starting P2.10.

---

## P2.10 — STARTED 2026-10-10 20:21
Files I will touch: src/rl_gym/observers.py, src/rl_gym/gym_env.py,
tests/test_rl_gym_observers.py, tests/test_rl_gym_env.py, docs/SDPDP_GYM_SPEC_v2.md,
docs/RL_GYM_PROGRESS.md

Resume state: P2.9 committed as 29ec24b, tree clean. `K` stays 8.

Measured before starting, so the "after" figure has a same-tree baseline: the gate driver on
this tree runs **436 steps in 38.2 / 38.5 s → 11.48 and 11.32 steps/s** (median 11.40), against
P1.10's 11.47. P2.7's tracker and P2.9's request block cost nothing measurable.

Confirmed before starting (the two load-bearing facts of §5.2 exception b):
- **`travel_time_infos` is the only cache the observer's calls can write.**
  `NetworkBasicWithStore` declares exactly one store (`:72`) with two writers:
  `_add_to_database` (`:151`), which writes **only if the key is absent** so a query can never
  overwrite an entry, and `add_travel_infos_to_database` (`:125`) for externally computed
  multiprocessing results, which nothing on the observer's path calls. On the base class both
  hooks are `pass` (`NetworkBasic.py:976,979`) and every routing entry point funnels writes
  through `_add_to_database` (`:514,559,619`), so the one override covers `1to1`, `1toX`,
  `Xto1` and the `best_route` family. The observer's only routing call is
  `return_travel_costs_1to1` — directly for `added_distance`, and as the **sole** routing call
  inside the re-timing (`VehiclePlan.py:860`); `return_position_coordinates`,
  `get_section_overhead` and `test_and_get_trivial_route_tt_and_dis` are pure reads.
- **A correction that changes how the restore must be written:**
  `_reset_internal_attributes_after_travel_time_update` **rebinds** `travel_time_infos` to a
  fresh dict (`:123`) rather than clearing it, and `load_tt_file` calls it at every operator
  travel-time file load — observed at sim_times 32400, 43200, 57600, 64800 in a full episode.
  So the restore reads `routing_engine.travel_time_infos` through the attribute each time and
  never holds the dict object. My own first measurement held a reference, went stale and
  reported nonsense; it fails invisibly.
- **Two mutations the restore cannot undo, both inert.** Every Dijkstra writes scratch fields on
  `Node` objects (`prev`, `next`, `cost`, `settled`, `cost_index`) and advances
  `nw.current_dijkstra_number`. Each `Router` takes `current_dijkstra_number + 1` as its
  generation (`Router.py:62`) and the counter is initialised to 1 at construction
  (`NetworkBasic.py:191`) and never reset, so no run can read an earlier run's leftovers as its
  own, and the only number that could collide with the node defaults (`settled = 1`,
  `cost_index = -1`) is 1, which no run ever gets.
- **No random draw on the observation path.** The only RNG in the routing stack is the
  stochastic travel-time redraw, which fires exclusively from `update_network`
  (`NetworkBasic.py:291` → `_draw_stochastic_tt_new`, `np.random.lognormal` at `:367,372`). The
  observer never calls `update_network`, `load_tt_file`, `reset_network`,
  `assign_route_to_network` or `move_along_route`. `Router` has no RNG, `VehiclePlan.py` has
  none, and `src/rl_gym/` draws only in `gym_env` (episode seed) and `day_rotation` (D10's
  permutation), both outside the observation path.

Projected before starting, by enumerating the legs the observer would route and checking each
against the live cache read-only (routing nothing): 7,956 queries per episode, 18.2 per
decision, of which 191 trivial, **7,765 already cached by the simulation and 0 genuine new
Dijkstras**. The reason is structural: `insertion_with_heuristics` re-times every candidate plan
it returns, so every leg of plan `k` is cached by construction, and at 0.72 pre-insertion stops
per candidate the pre-insertion legs are a near-subset of what insertion already explored. For
scale, the episode already runs 22,465 genuine Dijkstras itself, 27.6 s of its 38 s. So the
restore is expected to have nothing to delete on the reference day, and it stays in regardless.

Approved before starting:
- The remedy (a private observer-side memo and a read-through proxy engine) is **out of P2.10**.
  If the measurement contradicts the projection, report and stop; do not implement it.
- `position_encoding` defaults to `"latlon"`; `"relative"` is implemented and tested but unused.
- **Ritun's addition to the verification:** after implementing, also run the greedy driver on
  **2024-10-09** (`corpus_eval`, fixed-day mode) and report steps per second, the number of new
  route computations the observer triggers, and the number of cache entries the restore deletes,
  alongside the reference-day figures. If either count is non-zero there, report how often and
  **stop before proposing any remedy**.

## P2.10 — DONE 2026-10-10 21:05 — awaiting Ritun's verification

Files changed:
- `src/rl_gym/observers.py` — `CandidateObserver` extended from 3 to **12 features per slot**
  (§5.2's 4–12); new module-level `operator_plan_request`, `quoted_waiting_time` (moved here
  from `gym_env`), `quoted_pickup_time`, `route_distance` and `unit_vector`; new
  `_cache_keys` / `_restore_cache` / `_retimed_copy` / `_scaled_position` / `_pending_rids`;
  a `diagnostics` counter dict; `FEATURE_NAMES` as the vector order; module docstring now
  carries the two rules (trap 14 and no-RNG).
- `src/rl_gym/gym_env.py` — config keys `wait_normalizer_s` and `position_encoding`;
  `NodeCoordinates` hoisted to a local and shared by the candidate and request blocks;
  `_make_quoted_wait_fn` now delegates to `observers.quoted_waiting_time` so the quote has one
  definition.
- `tests/test_rl_gym_observers.py` — the stub harness rebuilt (`StubStop`, `StubPlan` with a
  copy/re-time record, `StubOffer`, `StubRider`, `StubLeg`, `StubCoords`, `StubNetwork` with
  `NetworkBasicWithStore`'s write-if-absent cache semantics, one `StubOperator` that both
  observers can read, integer vids as the real operator uses); `TestCandidateBlock` (29) and
  `TestCandidateBlockLeavesSimulationStateAlone` (11).
- `tests/test_rl_gym_env.py` — `OBS_LEN` / `CANDIDATE_FEATURES` constants replacing the
  hardcoded 52; `TestCandidateBlockInTheEnv` (8), which checks the cache and plan-integrity
  rules against the **real** `VehiclePlan`, `PlanStop` and `NetworkBasicWithStore` over 15
  consecutive decisions.
- `docs/SDPDP_GYM_SPEC_v2.md` — §5.2 exception (b) gains the four cache findings; the pending
  rider, `remaining_stops` and `rel_pos` VERIFY answers; the corrected `lateness_pre` sign
  expectation; §8's P2.10 gains the length, the single quote definition, the skip-when-no-pending
  note, and a rewritten verification list (see the two corrections below).
- `docs/RL_GYM_PROGRESS.md` — this entry.

Verification — everything passed, nothing failed:

1. **The P1.10 gate, byte for byte** — all 7 gate tests `ok`, including
   `test_user_stats_matches_baseline_byte_for_byte`. Full suite **426 tests pass**, 171.8 s, no
   failures and no skips (373 before; +53).
2. **The routing cache, both days — nothing to restore, because nothing is added.**

   | | reference day | 2024-10-09 |
   |---|---|---|
   | observer `1to1` queries | 6,946 | 7,845 |
   | **new route computations** | **0** | **0** |
   | **cache entries deleted by the restore** | **0** | **0** |
   | steps | 436 | 512 |

   So **Ritun's stop condition did not trigger**: both counts are zero on 2024-10-09 as well as
   on the reference day. The reason is structural, confirmed rather than assumed:
   `insertion_with_heuristics` re-times every candidate plan it returns, so every leg of plan `k`
   is already cached at that decision, and the pre-insertion legs are a near-subset of what
   insertion explored. The restore stays in regardless — it costs 38 µs per decision, and "no
   entries today" is not "no entries ever".
3. **The two rules, on the real objects.** Over 15 consecutive decisions per test:
   `routing_engine.travel_time_infos` holds exactly the same keys before and after every
   observation, and every `veh_plans[vid]` keeps its stops, boarding/alighting lists, `pax_info`
   and `utility` — §5.2's exception (a) never fired, as predicted, because insertion has already
   made that same call for every candidate vehicle at the same decision.
4. **`offered_wait` is bit-identical to Phase 1's formula** with `wait_normalizer_s` absent:
   exact `float32` equality on **1,982/1,982** candidate slots on the reference day and
   **2,151/2,151** on 2024-10-09, 0 mismatches — and the gate agrees.
5. **Hand computation of features 9, 10 and 12: 0 mismatches.** 10 decisions × 3 features on the
   reference day (one pending rider each) and 10 decisions × 3 on 2024-10-09 **with two pending
   riders**, recomputed independently from the stored plan's boarding rids, a separately re-timed
   copy, `prq.offer` and plan `k`'s `pax_info`.
6. **`directionality`** matches `filter_directionality`'s formula on constructed cases: parallel
   +1, opposed −1, perpendicular 0, a hand dot product 0.6, an idle vehicle 0, and a vehicle
   whose route ends at its own position 0.
7. **Distribution over the reference day, valid slots only** (1,982 slots, scaled values):

   | feature | min | median | p95 | max | exactly 0 |
   |---|---|---|---|---|---|
   | `free_seats` | +0.333 | +0.833 | +1.000 | +1.000 | 0.0% |
   | `remaining_stops` | 0 | +1 | +2 | +3 | 39.6% |
   | `directionality` | −1.000 | 0.000 | +0.959 | +1.000 | 41.4% |
   | `lateness_pre` | −0.360 | 0.000 | 0.000 | +0.322 | 91.9% |
   | `lateness_post` | −0.360 | 0.000 | 0.000 | +0.801 | 91.8% |
   | `added_distance` (km) | −2.956 | +10.482 | +22.816 | +33.039 | 0.2% |
   | `added_pickup_delay` | 0.000 | 0.000 | 0.000 | +0.791 | 99.4% |

   `free_seats` never reaches 0, so no candidate vehicle is ever full. `added_distance` is large
   because `pre` is 0 for an idle vehicle, making "added" the whole trip; 3 of its 1,979 non-zero
   slots are negative, so §5.2's merge case is real and rare.
8. **Throughput — and the measurement had to be redesigned, see the corrections below.**
   Interleaved A/B in one process, Phase-1-sized block against P2.10's: **9.66 → 9.57 steps/s,
   a 0.8% cost.** The profiler agrees: the whole observation is **0.16 s of a 40 s episode
   (0.4%)**, of which `CandidateObserver` is 0.13 s (0.30 ms per decision), the cache snapshot
   38 µs per decision, and its 6,946 routing calls 1.2 µs each. `reset()` is unchanged — median
   **0.50 s** warm, 1.21 s cold, against §10's ~0.46 s. 2024-10-09 ran 512 steps at 10.81
   steps/s under instrumentation.
9. **Both new guards fail when removed**, checked by breaking them deliberately and reverting:
   moving the restore out of its `finally` fails
   `test_the_restore_runs_even_when_a_feature_raises`; holding the cache dict instead of reading
   the attribute fails `test_the_cache_is_read_through_the_attribute_not_held`. A third test
   (`test_without_the_restore_the_cache_would_be_polluted`) pins that the observer really does
   write to the cache, so the restore tests cannot pass vacuously.

VERIFY findings:
- **`travel_time_infos` is the only cache the observer's calls can write: matched** what Ritun
  and I both read. Two writers only, `_add_to_database` (write-if-absent, so `after − before` is
  exactly the added set) and `add_travel_infos_to_database` (multiprocessing results,
  unreachable here); `NetworkBasic`'s hooks are both `pass`; every routing entry point funnels
  through `_add_to_database`.
- **`rel_pos` is the fraction already travelled: matched the spec**
  (`return_position_coordinates` forms `rel * c_dest + (1 − rel) * c_origin`,
  `NetworkBasic.py:410`; `get_section_overhead` documents it as the already-travelled share).
- **A rider whose boarding is in progress: resolved, and it is onboard rather than pending.**
  `veh_obj.pax` gains the rider when the boarding leg **starts** (`Vehicles.py:254`), which is
  why `get_nr_pax_without_currently_boarding` exists. My run-time probe for the divergence found
  0 cases, so it is not observable within a decision epoch here; the source is the evidence.
- **The stop currently being served still counts in `remaining_stops`: resolved, it counts.**
  84 of 1,982 candidate slots had a first stop already started (85 locked) and
  `len(list_plan_stops)` includes it, so a vehicle mid-boarding reads 1 rather than 0.
- **The quoted pickup is not overwritten after acceptance: matched** P2.7's finding —
  `prq.offer.offered_waiting_time`, set once by `_create_user_offer`; 0 pending riders lacked a
  quote across both days.
- **`update_tt_and_check_plan`'s signature and copy behaviour: matched.** Its only routing call
  is `return_travel_costs_1to1` (`VehiclePlan.py:860`); `keep_feasible` and `init_plan_state`
  stay at their defaults so neither `set_infeasible_locked` nor the `pax_info` carry-over branch
  runs; `VehiclePlan.copy()` copies every `PlanStop` (`:578`); `pax_info` is rebound to `{}` and
  repopulated with fresh lists before anything appends, so the shallow `pax_info.copy()` leaks
  nothing. §5.2 exception (a) is confirmed at `PlanStop.copy()` (`:324`, shallow
  `boarding_dict.copy()`) and `:897` (`boarding_dict[-1].remove(rid)`), and never fired.
- **`free_seats` over `veh_obj.pax`: matched** — `rq.nr_pax` is on the traveler
  (`TravelerModels.py:98`), and `max_parcels` is a separate capacity with no parcels in this
  scenario.

Two corrections to the spec, both found by the verification failing to be runnable as written:

- **§8's P2.10 asked for "10 decisions where some candidate has two or more pending riders". No
  such candidate exists on the reference day** — the maximum there is **1**, over 1,982 candidate
  slots in 436 decisions. The requirement is not satisfiable on the day the item names, which is
  why Ritun's 2024-10-09 addition mattered beyond the cache question: that day has 23 candidate
  slots with 2 pending riders across 22 decisions, and the hand computation ran there. Both §5.2
  and §8 now record the limit and where to go for a multi-rider candidate.
- **An absolute steps/second comparison against P1.10's 11.47 measures the machine, not the
  code.** The pre-P2.10 tree measured **11.40** steps/s early in this session and **9.66** two
  hours later, unchanged, at a load average of 2.84. Had I reported the naive before/after I
  would have claimed a 16% regression that the profiler puts at 0.4%. §8's P2.10 now specifies
  the interleaved A/B instead, with the figures.

One deliberate deviation from §5.2, stated because it is a change in when work happens: **the
re-timed copy is skipped when a candidate vehicle has no pending rider.** §5.2 says "one
re-timed copy per candidate vehicle per decision"; features 9, 10 and 12 are 0.0 in that case
anyway, so the copy would be computed and discarded. 168 of 436 decisions re-time anything at
all. No feature value changes, and a test pins both the skip and the one-copy-when-needed rule.

Nothing failed. **Stopping here** — not starting P2.11.

---

## P2.11 — STARTED 2026-10-10 21:40
Files I will touch: src/rl_gym/observers.py, src/rl_gym/gym_env.py, src/rl_gym/corpus.py,
src/rl_gym/offline/build_arrival_baseline.py, tests/test_rl_gym_observers.py,
tests/test_rl_gym_env.py, docs/SDPDP_GYM_SPEC_v2.md, docs/RL_GYM_PROGRESS.md

Resume state: P2.10 committed, tree clean. `K` stays 8. Observation length 124 → **139**.

VERIFY confirmed before starting, all four against the source:
- **The `op_act_fs_file` path and formula: matched §5.2 exactly.**
  `dir_names[G_DIR_FCTRL]/elastic_fleet_size/<file>` (`TimeBasedFS.py:51`) and
  `min(max(int(np.round(share × nr_vehicles)), 0), nr_vehicles)` (`:98`) over the column
  `share_active_fleet_size`. It reproduces §8's expected steps: 0.7 × 12 = 8.4 → **8** at 07:00,
  0.9 × 12 = 10.8 → **11** at 08:00, 1.0 → **12** at 10:00, 0.9 → **11** at 17:00. The rounding
  is why the feature is not simply the share (8/12 = 0.667), which is why `nr_vehicles` is needed
  at all — taken from `fleetpy_module.nr_vehicles`, the identical attribute `TimeBasedFS`
  multiplies by, rather than parsed out of `op_fleet_composition`.
  `_read_active_vehicle_file` also drops rows before `start_time`, which changes nothing here
  because the file's first row is exactly `start_time`.
- **The committed baseline's `sd` is pooled across months**, as P2.6 defines it: exactly one
  distinct value per `(t_seconds, window_s)` over all ten months. The lookup may therefore take
  `sd` from the episode's own month row.
- **The grid is 747 minutes × 10 months × 2 windows = 14,940 rows**, `t_seconds` 25200–69960,
  and October's mean is the highest at `W = 30` (22.45 against 16.8–19.8) — consistent with
  P2.6's verification.
- **P2.7's API gives what the block needs**: `count(kind, window_s)` fleet-wide with `cells`
  omitted, and `wait_gap_mean(window_s)` returning seconds or `None`.

Approved before starting — all three guards, plus:
- **Guard 2 brings two files into the item.** `baseline_hash`, `canonical_baseline_text`,
  `FLOAT_FMT`, `BASELINE_COLUMNS` and `read_baseline` move from
  `offline/build_arrival_baseline.py` to `src/rl_gym/corpus.py`, because `src/rl_gym/` must not
  import from `offline/`; the builder imports them back. Ritun's condition, as in P2.9: the
  committed `corpus_hash` and `baseline_hash` must be **unchanged** and
  `build_arrival_baseline --check` must still pass. Recorded here for the comparison:
  `corpus_hash   6b232cf105d05ad3ccc6c9121e845dace88e406471e3bbd04286edc5dabb85a3`
  `baseline_hash 5638cb266b54615492c85cb52de1be49faefde16cfe55e532f623fe5913622e1`
- **Guard 1:** the observer validates its windows and the scenario's `start_time`/`end_time`
  against the manifest at construction, so a scenario off the baseline's grid fails there rather
  than as a `KeyError` 40 s into an episode.
- **Guard 3:** an unknown month raises at `reset()`, as P2.9's `set_episode_day` does.
- **Record the minute-alignment count in §5.2.** §5.2 calls the online and offline windows the
  "identical definition", which is exactly true only when `sim_time` is minute-aligned; otherwise
  observed `[sim_time − W, sim_time)` and expected `[t − W, t)` are offset by up to 59 s. Every
  `rq_time` in both days checked is minute-aligned, so the flooring should be a no-op. Measure
  the count of decisions with `sim_time % 60 ≠ 0`, record it, change no behaviour either way.

## P2.11 — DONE 2026-10-10 22:30 — awaiting Ritun's verification

Files changed:
- `src/rl_gym/observers.py` — `GlobalStateObserver` extended from 4 to **19 features**; new
  `ArrivalBaseline` (P2.6's table, the `z` formula, `floor_minute`, `check_scenario`) and
  `ActiveFleetSchedule` (`share_at`, `active_fraction`) loaders beside `NodeCoordinates`; both
  are **required** constructor arguments of the observer, with `set_stats_tracker` and
  `set_episode_month` as the per-episode injection.
- `src/rl_gym/corpus.py` — `BASELINE_COLUMNS`, `FLOAT_FMT`, `DEFAULT_BASELINE`,
  `canonical_baseline_text`, `baseline_hash`, `read_baseline` moved in from the offline builder,
  plus new `verify_baseline`.
- `src/rl_gym/offline/build_arrival_baseline.py` — imports those six back instead of defining
  them; its now-unused `hashlib` import removed. No behaviour change.
- `src/rl_gym/gym_env.py` — builds the baseline and the schedule at construction (the baseline
  verifying its hash and checking the scenario's grid as it loads); new `_active_fleet_path`
  resolving `op_act_fs_file` through `get_directory_dict` the way `TimeBasedFS.add_init` does;
  injects the tracker and the month into the global observer in `reset()`.
- `tests/test_rl_gym_observers.py` — `write_baseline` / `write_active_fleet` /
  `global_observer` / `wide_global_observer` factories that build the **real** loaders over temp
  artifacts rather than stub doubles; the tracker stubs gained `count` and `wait_gap_mean`;
  `TestArrivalBaseline` (14), `TestActiveFleetSchedule` (9), `TestGlobalBlock` (18).
- `tests/test_rl_gym_env.py` — `GLOBAL_BLOCK` constant and `OBS_LEN` now 139;
  `TestGlobalBlockInTheEnv` (10).
- `docs/SDPDP_GYM_SPEC_v2.md` — the `active_*` and pooled-`sd` VERIFY answers, the two new
  guards and the injection rule in §5.2; the minute-alignment measurement under §5.2 Windows;
  §8's P2.11 Files line, the `corpus.py` move and its proof; trap 16 extended to the baseline.
- `docs/RL_GYM_PROGRESS.md` — this entry.

Verification — everything passed, nothing failed:

1. **The P1.10 gate, byte for byte** — all 7 gate tests `ok`. Full suite **477 tests pass**,
   169.8 s, no failures and no skips (426 before; +51).
2. **Observation length 124 → 139**, the global block contributing exactly 19, the layout still
   tiling `[0, 139)` with no gap or overlap, block order candidate → request → global, and
   `active_60` last.
3. **Every one of the 15 new features hand-computed at 10 decisions: 0 mismatches.** The
   recomputation is independent of P2.7's tracker: arrival counts come from the demand file's
   kept rows (both `load_demand_file` filters), declines and cancellations from a separate spy on
   the exit callback, the wait gap from the pickup callback and `prq.offer`, `expected` and `sd`
   straight out of `arrival_baseline.csv` with pandas, and `active_*` from
   `example_active_fleetsize.csv` with the formula applied by hand.
4. **`active_now` × 12 over the reference day: 8 → 11 → 12 → 11**, at the first decision after
   each schedule change (07:01, 08:01, 10:00, 17:00) — §8's expected figures. `active_30` and
   `active_60` lead by 30 and 60 min, checked both in a unit test and against the committed file
   at 25200/28800/36000/61200/69999.
5. **Minute alignment: 0 of 436 decisions** have `sim_time % 60 ≠ 0`, so the baseline's flooring
   is a no-op and §5.2's "identical definition" of the online and offline windows is exactly
   true here. Recorded in §5.2 with the structural reason; a test keeps the count at 0.
6. **Distribution of the 15 new features over the reference day** (436 decisions):

   | feature | min | median | p95 | max | exactly 0 |
   |---|---|---|---|---|---|
   | `arrivals_30` | 0 | 19 | 26 | 30 | 0.5% |
   | `arrivals_60` | 0 | 39 | 48 | 51 | 0.5% |
   | `arrivals_expected_30` | 0.44 | 21.50 | 26.78 | 28.11 | 0% |
   | `arrivals_expected_60` | 0.44 | 42.72 | 51.22 | 54.44 | 0% |
   | `arrivals_z_30` | −3.63 | −0.53 | +0.97 | +1.72 | 0% |
   | `arrivals_z_60` | −2.68 | −0.64 | +0.58 | +0.89 | 0% |
   | `declines_30` / `_60` | 0 / 0 | 4 / 9 | 10 / 17 | 15 / 20 | 2.8% |
   | `cancels_30` / `_60` | 0 / 0 | 1 / 2 | 4 / 5 | 5 / 5 | 38.5% / 17.4% |
   | `wait_gap_30` | −0.099 | −0.022 | +0.058 | +0.136 | 0.7% |
   | `wait_gap_60` | −0.087 | −0.025 | +0.047 | +0.097 | 0.7% |
   | `active_now` | 0.667 | 1.000 | 1.000 | 1.000 | 0% |
   | `active_60` | 0.917 | 1.000 | 1.000 | 1.000 | 0% |

   Worth noting, not a defect: **`arrivals_z` runs negative on this day** (median −0.53 at
   W = 30). 2024-06-20 is an evaluation day with 445 requests, and the baseline is built from
   **training** days only (§8 P2.6), June's averaging 19.75 arrivals per 30 min against this
   day's median 19. A single day sitting half an SD below its month's training mean is ordinary;
   the negative median is the anomaly feature working, not misreading.
7. **Guard 1**, four ways: a window the baseline has no rows for raises and names it; a scenario
   starting before or ending after the grid raises and names the covered minutes; the reference
   window is accepted; and an observer subclass declaring a 7200 s window fails at construction.
8. **Guard 2**, five ways: an edited baseline row makes construction raise, naming
   `baseline_hash` and trap 16; a manifest without `baseline_hash` raises; a missing baseline
   names the build command; the committed pair loads cleanly through the runtime path; and the
   end-to-end env test corrupts a copy of the real baseline and gets the raise with the rebuild
   command in it.
9. **Guard 3**: an unknown month raises at `set_episode_month` over four cases including `None`;
   `observe` raises without a tracker and without a month; and the env re-injects both on every
   reset (asserted by identity, so a stale tracker from the previous episode fails).
10. **Throughput: no measurable cost.** Interleaved A/B in one process, 4-feature global block
    against 19: **10.98 → 10.99 steps/s, −0.0%**. Measured this way for the reason P2.10
    established — an absolute figure compared across sessions measures the machine.

**Ritun's condition on the `corpus.py` move — neither hash moved.**
- `build_arrival_baseline --check`: "the committed baseline and manifest match a fresh build,
  and agree with each other".
- `corpus_hash` **6b232cf105d05ad3ccc6c9121e845dace88e406471e3bbd04286edc5dabb85a3** and
  `baseline_hash` **5638cb266b54615492c85cb52de1be49faefde16cfe55e532f623fe5913622e1** — each
  identical three ways: the value recorded in the STARTED entry, the committed manifest, and a
  fresh recomputation through `corpus.py`.
- `git status -- data/` is empty: nothing under `data/` changed.

VERIFY findings:
- **The `op_act_fs_file` path and formula: matched the spec exactly**, including the figures §8
  expects. Recorded in §5.2 with the note that the rounding is why the feature is not the share,
  and that `nr_vehicles` comes from `fleetpy_module.nr_vehicles`.
- **The committed baseline's `sd` is pooled across months: matched P2.6's definition** — one
  distinct value per `(t_seconds, window_s)` over all ten months. A test asserts it against the
  real artifact, because the lookup is keyed by `(t, window)` and would silently take one
  month's value if that ever stopped being true.
- **The baseline's grid: matched** — 14,940 rows, 747 minutes, 25200–69960, ten months, both
  windows, `sd_floor` 1.0 and `minute_s` 60 read from the manifest rather than literals.
- **P2.7's query API: matched** — `count(kind, window_s)` fleet-wide with `cells` omitted (a test
  asserts the global block never passes `cells`, which is the request block's job) and
  `wait_gap_mean` returning seconds or `None`.

One mistake of mine, caught immediately by a guard P2.8 added: **my first `observe` emitted
`arrivals_expected_30, arrivals_z_30, arrivals_expected_60, arrivals_z_60` while `feature_spec`
declares both `expected`s then both `z`s.** `observe_vector` raised on the very first run and
named both lists. Without P2.8's check the vector would have been built in the emission order
while every consumer read it in the declared order — two features silently swapped, which no
byte gate would catch. Fixed by computing the pair per window and emitting in declared order,
with a comment saying why.

A second mistake, in a test of mine rather than the code: `test_z_uses_the_sd_floor` asserted the
fixture's `sd` was below the floor, but the fixture writes `sd = 0.25 × month` and the lookup is
keyed by `(t, window)`, so the last month written (10) wins at 2.5 — above the floor. The test
failed, correctly. Rewritten with explicit rows that exercise **both** sides of
`max(sd, sd_floor)`: sd 0.25 with the floor applying, and sd 4.0 with the real sd applying.

**§5.2 is now complete**: 96 candidate + 24 request + 19 global = 139 at `K = 8`.

Nothing failed. **Stopping here** — not starting P2.12.

---

## P2.12 — STARTED 2026-10-10 22:55
Files I will touch: train_sdpdp.py, src/rl_gym/gym_env.py, tests/test_rl_gym_env.py,
tests/test_rl_gym_train.py (new), docs/SDPDP_GYM_SPEC_v2.md, docs/RL_GYM_PROGRESS.md

Resume state: P2.11 committed, tree clean. Observation length 139.

VERIFY confirmed before starting, against the installed sb3 2.9.0 / sb3_contrib 2.9.0 /
torch 2.10.0 / gymnasium 1.3.0 / numpy 2.2.6:
- **`VecNormalize(venv, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0,
  clip_reward=10.0, gamma=0.99, epsilon=1e-8, norm_obs_keys=None)`.** `gamma` **defaults to
  0.99**, exactly the trap §5.3 warns about: left alone it would normalise rewards against a
  ~100-step horizon while PPO discounts over ~1000. It must come from the same config value PPO
  reads.
- **`VecNormalize.save(path)` and `VecNormalize.load(path, venv)`**, the latter a static method
  taking the venv to attach to.
- **Masks pass through the wrapper untouched, structurally.** `get_action_masks` on a `VecEnv`
  does `np.stack(env.env_method("action_masks"))`; `VecNormalize` **is** a `VecEnv`, does **not**
  override `env_method`, and `VecEnvWrapper.env_method` is a bare forward to `self.venv`.
- **`is_masking_supported`** — what `MaskablePPO` itself calls, and what would silently turn
  masking off — goes through `has_attr`, which `VecEnvWrapper` also forwards.
- **`VecNormalize` transforms observations and rewards only**, so there is no path by which
  normalisation could reach a mask: masks never travel through `step()`'s return.

Approved before starting:
- **Ritun's placement decision: the wrapped-venv tests go in a new
  `tests/test_rl_gym_train.py`, not `tests/test_rl_gym_env.py`** — that keeps `torch`, and with
  it §8a's deferred OpenMP import-order constraint, out of the env test module. P2.13 extends
  the same file. `tests/test_rl_gym_train.py` joins P2.12's Files line (§8 listed it under
  P2.13).
- **Ritun's two added checks**, both on a real wrapped `SubprocVecEnv`:
  1. `get_action_masks` on the wrapped env returns each worker's mask **unchanged**, equal to
     what the unwrapped envs report.
  2. The **normalised** observation is never fed to `greedy_action`, and its `is_valid` guard
     refuses it. Two cases, which fail differently: at the first reset the running variance is
     zero so every normalised feature is 0.0, and after a few steps `is_valid` is a non-0/1
     float. The raw observation must still pass, so the test cannot pass by `greedy_action`
     refusing everything.
- Stated plainly because it bounds what the gate proves: **the P1.10 gate cannot prove
  `VecNormalize` is harmless**, since the gate drives the unwrapped env and the wrapper sits
  outside it (§8). What the gate proves here is that the non-finite guard — new code in the
  env's hot path — changes nothing. The wrapper rests on the mask and round-trip tests.

## P2.12 — DONE 2026-10-10 23:40 — awaiting Ritun's verification

Files changed:
- `train_sdpdp.py` — `build_venv(cfg, n_workers, normalize=True, training=True)`,
  `save_vecnormalize(venv, checkpoint_path)`, `load_vecnormalize(checkpoint_path, venv,
  for_evaluation=True)`; `gamma` config key (0.999) and `VECNORMALIZE_FILE` / `CLIP_OBS`
  constants; `smoke_test(..., normalize=True)` so the wrapper is exercised by
  `python train_sdpdp.py` rather than first inside a training run; `--no-vecnormalize` and
  `--gamma` flags; the worker-PID bookkeeping now reads through `venv.unwrapped`.
- `src/rl_gym/gym_env.py` — `_check_finite`, called from `_observe`, which is the single site
  both `reset()` and `step()` pass through.
- `tests/test_rl_gym_env.py` — `TestNonFiniteObservationGuard` (7). No sb3 import.
- `tests/test_rl_gym_train.py` — **new.** `TestVecNormalizeWiring` (10),
  `TestMasksPassThroughVecNormalize` (4), `TestNormalisedObservationsNeverReachGreedy` (5).
- `docs/SDPDP_GYM_SPEC_v2.md` — §5.3 gains the wiring note, the two findings and the guard's
  message rule; §8's P2.12 gains the Files line, the test-module decision and what the gate
  does and does not prove; **§8a gains the forkserver-preload measurement below**.
- `docs/RL_GYM_PROGRESS.md` — this entry.

Verification — everything passed, nothing failed:

1. **The P1.10 gate, byte for byte** — all 7 gate tests `ok`. Full suite **503 tests pass**,
   184.3 s, no failures and no skips (477 before; +26).
2. **`check_env` through `MaskRespectingShim` on the full 139-length observation**, with
   `base_seed` absent and set — both `ok`.
3. **Ritun's check 1 — the masks pass through untouched.** On a real 2-worker
   `VecNormalize(SubprocVecEnv)`: `get_action_masks(wrapped)` equals `get_action_masks(inner)`
   and `np.stack(inner.env_method("action_masks"))` element for element, at the first decision
   and across 5 further steps; shape `(2, 9)`, dtype `bool_`, values only `{True, False}`;
   reject legal in every mask; a second read agrees with the first, so reading never advances
   anything. `is_masking_supported` — what `MaskablePPO` itself calls, through `has_attr` — is
   True on both the wrapped and the unwrapped venv.
4. **Ritun's check 2 — the normalised observation is refused.** `greedy_action` raises
   `ValueError` naming `VecNormalize` on the **very first** normalised observation and again
   after 5 steps, when `is_valid` is a non-0/1 float. The control passes: the raw observation
   from the **same** wrapper (`get_original_obs`) is accepted for both workers, its `is_valid`
   equals the mask exactly, and the chosen action is legal — so the guard refuses normalised
   input rather than everything. A further test asserts the wrapper really is transforming the
   observation, so that comparison is not vacuous.
5. **Save → load round trip**: `normalize_obs` on a fixed 139-vector is identical before and
   after, and `obs_rms` mean, var and count all match. `save_vecnormalize` writes
   `vecnormalize.pkl` beside the checkpoint and raises `TypeError` on an unwrapped venv;
   `load_vecnormalize` raises `FileNotFoundError` naming the file when it is absent, sets
   `training = False` **and** `norm_reward = False` for evaluation while leaving `norm_obs` on,
   and leaves both flags on when `for_evaluation=False`.
6. **`gamma` comes from the config, not SB3's default**: 0.999 on the wrapper, asserted
   `!= 0.99`, and a config of 0.95 reaches it. §5.3's settings asserted directly:
   `norm_obs`, `norm_reward`, `clip_obs = 10.0`, `training`.
7. **The non-finite guard**, seven ways: a NaN at reset raises naming the feature and
   `VecNormalize`; a NaN at step raises; an inf raises and reports the value; a poisoned
   candidate slot names the slot (`delta_cfv[3]`); a whole poisoned feature reports
   "8 of 139 values"; 40 real decisions never trip it; and the terminal all-zero observation is
   finite and passes.
8. **The wrapped smoke test runs**: `python train_sdpdp.py --workers 2` → 547 and 410 steps,
   exit codes 0, no orphans, `VecNormalize: True`, forkserver, D10 rotation giving distinct
   days (2024-09-26, 2024-03-27), `unclassified 0` on both.

VERIFY findings:
- **`VecNormalize`'s signature and defaults: matched**, and the `gamma = 0.99` default is real —
  hence the explicit pass and the test that pins it.
- **`save(path)` / `load(path, venv)`: matched**, `load` a static method taking the venv.
- **Masks pass through `env_method`: matched** the structural reading, and now measured.
- **`is_masking_supported` forwards through `has_attr`: matched.**

**Two things I got wrong, both caught by the tests rather than by reasoning:**

- **§8a's `__main__` preload does not carry the numpy-first order into a test module's
  workers, and my proposal said it would.** Under `python -m unittest …`, `__main__` is
  unittest's own `__main__.py`: the default `preload = ['__main__']` establishes no ordering in
  the forkserver, and it would execute `unittest.main()` inside the forkserver, re-entering the
  suite. All three wrapped-venv classes died in `setUpClass` with `OMP: Error #15 … Abort trap:
  6`, while the same file run as a script passed — which is exactly the asymmetry §8a describes
  for scripts. Fixed with `mp.set_forkserver_preload(["numpy"])` at the top of the module, which
  loads numpy into the forkserver image before any worker forks from it and re-enters nothing:
  **19/19 pass both under `-m unittest` and as a script.** Recorded in §8a with the measurement,
  since the next test module that spawns workers will need the same line.
- **"The first reset normalises everything to zero" was wrong.** I predicted a zero variance
  from a single sample; SB3's `RunningMeanStd` starts at `count = 1e-4, mean = 0, var = 1`, so
  the first update pulls the mean *almost* to the sample and leaves a tiny non-zero variance. A
  raw `is_valid` of 1.0 comes out at about **0.005** with two workers and **0.007** with one —
  neither 0.0 nor 1.0. My test asserted the all-zero vector and failed, correctly. Rewritten to
  assert the measured mechanism, which is the stronger statement anyway: the guard refuses the
  normalised observation from the very first step, not only once the statistics have moved.

One smaller defect of my own, also caught by its test: the guard's message rendered offsets as
`delta_cfv[np.int64(3)]` under numpy 2, turning a message meant to point at a slot into
something you have to decode. Fixed in the code — plain `int`/`float` — not in the test.

Nothing failed. **Stopping here** — not starting P2.13.

---

## P2.13 — STARTED 2026-10-11 00:10
Files I will touch: train_sdpdp.py, src/rl_gym/callbacks.py (new), src/rl_gym/sim_env_rl.py,
src/rl_gym/gym_env.py, tests/test_rl_gym_train.py, docs/SDPDP_GYM_SPEC_v2.md,
docs/RL_GYM_PROGRESS.md

Resume state: P2.12 committed, tree clean.

Checked before starting:
- **TensorBoard is importable** — `tensorboard 2.21.0` and
  `torch.utils.tensorboard.SummaryWriter`. Verified end to end rather than by import alone:
  SB3's logger configured with `["tensorboard"]`, one scalar recorded and dumped, produced an
  `events.out.tfevents.*` file whose `rollout/probe` read back as `(1, 1.5)` through
  `EventAccumulator`. **No install needed** (§0).
- **SB3's TensorBoard writer takes strings via `add_text` and numpy arrays / torch tensors via
  `add_histogram`**, so the day logs as text and the day-visit and candidate-length
  distributions as arrays, with no custom writer.
- **`MaskableEvalCallback` syncs the statistics but does not set the flags.** `_on_step` calls
  `sync_envs_normalization(self.training_env, self.eval_env)`, gated on
  `self.model.get_vec_normalize_env() is not None`, and that function deep-copies **`obs_rms`
  and `ret_rms` only** — it never touches `training` or `norm_reward`. Those are the caller's
  job at construction, which is what §8's "synced, `training = False`" means. It also
  **asserts** the eval env is wrapped the same way, so an unwrapped eval venv fails loudly.
- **VERIFY, how SB3 spreads `n_eval_episodes`:**
  `episode_count_targets = [(n_eval_episodes + i) // n_envs for i in range(n_envs)]`, summing
  to `n_eval_episodes`, with the **later** envs taking the remainder. Giving every eval env the
  whole pair list would therefore replay the first few pairs and never reach the rest; the
  pairs must be **partitioned** by those counts.
- **VERIFY, `MaskablePPO(seed=...)` would bypass D7, as §8 says.**
  `BaseAlgorithm.set_random_seed` ends with `self.env.seed(seed)` and `VecEnv.seed` sets
  `self._seeds = [seed + idx for idx in range(num_envs)]`, handing worker `i`
  `base_seed + i` at its next reset. So no `seed=` to the model; `set_random_seed(s)` in the
  parent with `s` from `SeedSequence([base_seed, 2020])`.
- **The eight reward weights** in `DEFAULT_REWARD_WEIGHTS` match §6.1's names exactly.

Approved before starting:
- **Ritun's point 1:** `mp.set_forkserver_preload(["numpy"])` moves into
  `prepare_multiprocessing()`, called from `build_venv`, so the training venv, the evaluation
  venv and P2.14's harness all get it without relying on the caller's import order. The
  module-level call comes **out** of `tests/test_rl_gym_train.py`, which turns its existing 19
  tests into the proof, plus one test pinning the preload list itself.
- **Ritun's point 2:** `--verify-run`, a committed re-runnable command printing a pass/fail
  line per check, not a scratchpad script, because Ritun re-runs every verification himself
  (§0).
- **Ritun's point 3:** the eval-pair partitioning is tested across **two consecutive**
  evaluations, not one. His reasoning is right and it shapes the design: an extra discarded
  episode at the end of evaluation 1 advances that worker's `eval_pairs` rotation, so the
  second evaluation starts at a different offset. **Per-worker slices make that safe**, and the
  argument is worth writing down: if worker `i` holds exactly its own `L_i = (E + i) // N`
  pairs and runs `L_i` counted episodes per evaluation, then `L_i` consecutive positions of a
  cycle of length `L_i` cover every pair exactly once **whatever the starting offset** — the
  order rotates, the set does not. Both evaluations will log and assert every (day, seed).

## P2.13 — DONE 2026-10-11 01:55 — awaiting Ritun's verification

Files changed:
- `src/rl_gym/callbacks.py` — **new.** `partition_eval_pairs`; `EpisodeLogCallback` (the
  per-episode scalars, the slot frequencies, the two histograms, one dump per episode);
  `EpisodeCheckpointCallback` (episode-counted, with statistics and metadata);
  `load_run_metadata` / `check_resume_base_seed`; `PairLoggingMaskableEvalCallback`.
- `train_sdpdp.py` — `prepare_multiprocessing` (called by both venv builders),
  `DEFAULT_HYPERPARAMS`, `REQUIRED_REWARD_WEIGHTS` / `PHASE2_REWARD_WEIGHTS`,
  `make_env(..., overrides=)`, `build_eval_venv`, `parent_seed`, the extended `check_config`,
  `build_model`, `train`, `verify_run` with `_check_reload_determinism` and
  `_check_loss_determinism`; `--train`, `--verify-run`, `--timesteps`, `--out-dir`,
  `--resume-from`, `--keep`; the module docstring rewritten around the three items.
- `src/rl_gym/sim_env_rl.py` — `_max_sim_time` and the `tail_seconds` property, via a
  three-line `update_sim_state_fleets` override.
- `src/rl_gym/gym_env.py` — `tail_seconds` in `episode_summary()`; **`info["episode"]` renamed
  to `info["episode_index"]`** (see below).
- `tests/test_rl_gym_env.py` — `TestInfoKeysDoNotCollideWithSB3` (3).
- `tests/test_rl_gym_train.py` — the module-level preload call removed;
  `TestForkserverPreload` (3), `TestEvalPairPartitioning` (8), `TestEpisodeLogCallback` (6),
  `TestEpisodeCheckpointCallback` (5), `TestResumeGuard` (3), `TestTrainingConfigGuards` (6),
  `TestParentSeeding` (3).
- `docs/SDPDP_GYM_SPEC_v2.md`, `docs/RL_GYM_PROGRESS.md`.

Verification — everything passed, nothing failed:

1. **The P1.10 gate, byte for byte** — all 7 gate tests `ok`. Full suite **541 tests pass**,
   187.4 s, no failures and no skips (503 before; +38).
2. **`python train_sdpdp.py --verify-run --workers 2`: 24/24 checks passed in 8.3 min**,
   exit 0. The committed command Ritun asked for, printing a line per check:
   the three config/resume raises; the run completing with worker exit codes `[0, 0]` and 8
   episodes; `unclassified == 0` in every episode; all 19 required scalars and both histograms
   present in the event files; slot frequencies for all 9 slots; no training day visited twice
   and each worker running exactly the days D10 assigns it, in order; the episode seeds being
   D7's rather than `base_seed + env_id`; three evaluations each covering every pair exactly
   once; the evaluation env still `training=False, norm_reward=False`; two checkpoints with
   their statistics and metadata; a reloaded checkpoint reproducing its 15 deterministic
   actions; and two runs with the same `base_seed` giving identical first-update losses
   (1.027919888496399 both times).
3. **Ritun's point 2 confirmed empirically, and the shift is real.** Evaluation 1 ran the pairs
   in order; **evaluation 2 ran them as `[03-19, 08-13, 01-11, 06-20]`** — rotated, because a
   discarded episode advanced a worker's rotation — and still covered all four exactly once, as
   did evaluation 3. A unit test simulates the same thing over three evaluations with 1, 2 and
   5 discarded episodes across 1, 2 and 3 workers, asserts the order really does change (so the
   test is not vacuous), and pins the counter-example: handing every worker the whole list
   covers only 4 of 6 pairs.
4. **`tail_seconds` cross-checks against P2.4's formula exactly.** Reference day, margin
   +194.7 s → **0.0 s**; 2024-10-09, margin −1038.3 s → **1070.0 s**, which is
   `ceil(1038.3/10) × 10 + 30` to the second. Two independent measurements agreeing is what
   makes the field trustworthy rather than plausible.
5. **TensorBoard checked before being relied on** (Ritun's point 2 of the proposal):
   `tensorboard 2.21.0` importable, and SB3's logger verified writing a scalar that reads back
   through `EventAccumulator`. No install.
6. **The forkserver preload moved** into `prepare_multiprocessing`, called by `build_venv` and
   `build_eval_venv`. `tests/test_rl_gym_train.py` no longer sets it and all 54 of its tests
   pass both under `-m unittest` and as a script — the proof Ritun asked for. Three tests pin
   the mechanism: no module-level call in the test file (checked by **parsing**, not grepping,
   because one test legitimately calls it inside a function to set the broken default),
   `prepare_multiprocessing` setting `["numpy"]`, and `build_venv` restoring it after the
   broken default is deliberately installed.

VERIFY findings:
- **How SB3 spreads `n_eval_episodes`: measured**,
  `[(n_eval_episodes + i) // n_envs for i in range(n_envs)]`, later workers taking the
  remainder. The partitioning matches it, and a test compares against that formula directly
  rather than against hardcoded slice sizes.
- **`MaskablePPO(seed=...)` would bypass D7: confirmed.** `BaseAlgorithm.set_random_seed` ends
  with `self.env.seed(seed)`; `VecEnv.seed` sets `[seed + idx for idx in range(num_envs)]`. No
  `seed=` is passed; `parent_seed` tags its own stream and a test asserts it can never equal
  `base_seed + env_id` or collide with D7's or D10's streams.
- **`MaskableEvalCallback` syncs statistics but not the flags: confirmed**, and the callback
  asserts the flags afterwards.
- **The episode-seed derivation: matched.** Predicted 523045 for `base_seed=20261011, env_id=1`
  from `SeedSequence([base_seed, env_id])` → `default_rng(worker_seed).integers(...)`; the env
  produced 523045. `base_seed + env_id` would have been 20261012.

**One real defect found, and only a real training run could have found it.** The env put the
episode counter in `info["episode"]`, which **SB3 reserves** for the `Monitor` wrapper's
statistics dict: `_update_info_buffer` extends `ep_info_buffer` with it and `dump_logs` then
calls `len()` on the entry, so the first `MaskablePPO.learn` died with
`TypeError: object of type 'int' has no len()`. Nothing earlier could have caught it — the
gates, `check_env` and every observation item drive the env directly, and only PPO reads that
buffer. Renamed to `info["episode_index"]`; it stays `episode` inside `episode_summary()`,
which SB3 never inspects. Three regression tests in `tests/test_rl_gym_env.py`.

**Two mistakes of my own, both caught by the thing they were meant to check:**
- `--verify-run`'s loss-determinism check reported "no train/loss recorded". The cause is a
  real SB3 ordering detail: `OnPolicyAlgorithm.learn` calls `dump_logs` **before**
  `self.train()`, so an update's `train/*` keys are written only at the following dump, and my
  single-rollout runs never reached one. Three rollouts now, and the check passes with
  identical losses to the last digit. Recorded in §8 so the next short run does not repeat it.
- Two test-fixture errors: `BaseCallback.logger` and `.training_env` are **read-only
  properties** reading `self.model.logger` and `self.model.get_env()`, so the stubs had to
  provide them through the model. And my first "this module sets no preload" test grepped the
  source, which flagged its own fixture; it parses the AST for module-level calls instead.

Nothing failed. **Stopping here** — not starting P2.14.
