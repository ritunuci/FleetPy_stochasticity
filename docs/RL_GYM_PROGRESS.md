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
