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
