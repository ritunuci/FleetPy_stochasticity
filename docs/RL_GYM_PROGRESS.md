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
