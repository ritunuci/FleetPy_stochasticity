"""Evaluation harness: the stock-greedy baseline, the RL policy, and the KPI table (P2.14).

**The RL result is meaningless without the greedy baseline on the same held-out days.** This
module runs both on the same `(day, seed)` pairs and tabulates them through `kpis.py`, which
reads only output directories and so treats the two identically.

**Two halves, and a run must say which it means** (P2.17). The 26 held-out days are split 12
validation (`corpus_val/`) and 14 test (`corpus_eval/`): every choice -- which checkpoint, which
reward weights, when to stop -- is made on the validation half, and the test half is read once,
by the final evaluation. So `--greedy` and `--rl` require exactly one of `--validation` or
`--final-test`; there is no default, because a default is what gets used by accident. Each day's
directory is then resolved from the day itself through `corpus.day_dir_for`, which raises if the
day belongs to the other half -- in both directions, on one code path.

    # once, and never again -- the seed file is an unregenerable artifact
    python -m src.rl_gym.evaluate --derive-seeds
    python -m src.rl_gym.evaluate --check-seeds

    # the committed verification of this work item
    python -m src.rl_gym.evaluate --verify

    # choosing between checkpoints: the validation half, as often as you like
    python -m src.rl_gym.evaluate --validation --greedy --days 2024-06-20,2024-08-13 --seeds 2
    python -m src.rl_gym.evaluate --validation --rl --checkpoint runs/sdpdp/ep8.zip \
        --days 2024-06-20,2024-08-13 --seeds 2

    # the final figures: the 14 test days, once, after everything is settled
    python -m src.rl_gym.evaluate --final-test --greedy --rl \
        --checkpoint runs/sdpdp/model_ep07000.zip --out docs/final_kpis.txt

The baseline is **stock FleetPy** (D9): `run_examples.run_single_simulation` on
`scenario_parameters` built exactly as `run_scenarios` builds them, with four overrides. P1.10
proved env-driven greedy reproduces stock greedy byte for byte, so stock is the same policy with
nothing in between that could differ -- `policies.greedy_action` exists to drive the gates and
P2.4's sweep, not to be the baseline.

**The comparison is unpaired.** Rider decline, the diffusion random term and stochastic travel
time all draw from one global stream (§2.5, P1.9), so two policies on the same seed share a
sample path only until their first different decision. The seed file buys reproducibility, not
common random numbers.

**Trap 1 is guarded in code, not only watched.** The active scenario row's own `scenario_name`
*is* the reference run's, and `FleetSimulationBase.__init__` hands that name to
`create_or_empty_dir`, which wipes the directory it is given. A baseline run that failed to
override the name would therefore erase `example_depot_time_pool_irsonly_rs_42/` --- the run
`docs/baseline_user_stats.csv` and `docs/baseline_op_stats.csv` came from. `guard_output_dir`
raises before the simulation is constructed if a run's resolved output directory is the row's,
so the failure cannot happen rather than being noticed afterwards.

Every run writes under `studies/<study>/results/`, which `.gitignore` already covers in full, so
no result directory can reach git.

**Every run's output is refused inside a cloud-synced folder and checked once it ends** (P2.20,
`output_integrity`). A stock-greedy run is therefore constructed and run here rather than through
`run_examples.run_single_simulation` -- its two lines, without the `bugfix` branch that swallows
exceptions -- so the row counter can be attached to the simulation object. An exception in a
greedy run now propagates. `--verify`'s byte comparisons against both committed baselines are
what prove the baseline is still stock FleetPy.
"""

# numpy before anything that might pull in torch -- §8a. The torch and sb3 imports in this
# module are all function-local for the same reason: the greedy and KPI paths must not pay for
# them, and `tests/test_rl_gym_kpis.py` must stay importable without them.
import numpy as np  # noqa: F401  (import order is load-bearing; see §8a)

import argparse
import csv
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.misc import config  # noqa: E402
from src.misc.globals import (  # noqa: E402
    G_DAY_DIR,
    G_DIR_OUTPUT,
    G_OP_MODULE,
    G_RANDOM_SEED,
    G_RQ_FILE,
    G_SCENARIO_NAME,
    G_SIM_ENV,
    G_STUDY_NAME,
    get_directory_dict,
)
from src.rl_gym import kpis  # noqa: E402
from src.rl_gym.corpus import (  # noqa: E402
    DEFAULT_SPLIT_FILE,
    HELD_OUT_DIRS,
    TEST_DAY_DIR,
    VAL_DAY_DIR,
    check_split_file,
    day_dir_for,
    held_out_days,
    read_split_file,
    split_of_day,
    test_days,
    validation_days,
)
from src.rl_gym.gym_env import MAX_EPISODE_SEED, derive_study_name  # noqa: E402
from src.rl_gym.output_integrity import (  # noqa: E402
    RowCounter,
    check_episode_output,
    count_data_rows,
    op_stats_file,
    read_rows_record,
    refuse_synced_path,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCS = os.path.join(ROOT, "studies", "example_study", "scenarios")
DEFAULT_CONSTANT_CFG = os.path.join(SCS, "constant_config_depot_cali_sc_1.csv")
DEFAULT_SCENARIO_CFG = os.path.join(SCS, "example_depot_cali_sc_1.csv")

#: the two halves of the held-out split (§2.9, P2.17). Never `corpus/`, never
#: `corpus_excluded/`. A day's own directory decides its half -- `corpus.split_of_day` -- so
#: these names are here for messages and for the venv config, never to pick a day's directory.
VALIDATION_MODE = "val"
TEST_MODE = "test"

#: the unregenerable seed file and its layout. **Unchanged by P2.17**: the split labels went
#: into `docs/RL_GYM_HELDOUT_SPLIT.csv` instead, so this file stays byte-identical at md5
#: `559b836e7c61fb7dadd2e7bd9bc8891d` and every citation of it in P2.14 to P2.16 still holds.
SEEDS_FILE = os.path.join(ROOT, "docs", "RL_GYM_EVAL_SEEDS.csv")
SEEDS_COLUMNS = ("day", "seed_index", "seed")

#: §8 P2.14's constants. **Neither may ever change**: the seeds are reused for every policy ever
#: evaluated, so a different root or count would make new results incomparable with old ones.
EVAL_SEED_ROOT = 20261009
DEFAULT_SEEDS_PER_DAY = 10

#: `scenario_name` prefix for a stock-greedy run, which is also what makes the name unique
GREEDY_PREFIX = "greedy"

#: the byte baselines, both from the stock reference run and neither ever overwritten
BASELINE_USER_STATS = os.path.join(ROOT, "docs", "baseline_user_stats.csv")
BASELINE_OP_STATS = os.path.join(ROOT, "docs", "baseline_op_stats.csv")

#: the pair `docs/baseline_*.csv` were generated at (§2.8)
REFERENCE_DAY = "2024-06-20"
REFERENCE_SEED = 42

#: what recreates the reference run's output directory, from the repository root: `run_examples`'
#: own call for this scenario, which reproduces both committed baselines byte for byte (P2.14,
#: and again in P2.20's step 0). `--verify` names it when the directory is missing.
REFERENCE_RERUN_COMMAND = (
    "python -c \"import os, run_examples as r; s = os.path.join('studies', 'example_study', "
    "'scenarios'); r.run_scenarios(os.path.join(s, 'constant_config_depot_cali_sc_1.csv'), "
    "os.path.join(s, 'example_depot_cali_sc_1.csv'), log_level='info', n_cpu_per_sim=1, "
    "n_parallel_sim=1)\""
)


# -------------------------------------------------------------------------- #
# the seed file -- written once, never regenerated
# -------------------------------------------------------------------------- #
#
# `held_out_days`, `validation_days`, `test_days`, `split_of_day` and `day_dir_for` are imported
# from `corpus` rather than wrapped here. P2.17 renamed this module's `evaluation_days` to
# `held_out_days` and moved it there: "evaluation day" now means a **test** day specifically,
# while the seed file covers both halves, and a wrapper would have been a second definition of
# the one thing §2.9 says must have only one (trap 13).

def derive_eval_seeds(days: Sequence[str], seeds_per_day: int = DEFAULT_SEEDS_PER_DAY,
                      seed_root: int = EVAL_SEED_ROOT) -> List[Tuple[str, int, int]]:
    """`(day, seed_index, seed)` for every day, drawn once and forever.

    **The draw order is part of the artifact and must not change.** One generator,
    `default_rng(SeedSequence(seed_root))`, makes a single `integers` call for
    `len(days) * seeds_per_day` values and consumes them **row-major** -- day 0's ten, then day
    1's ten, with `days` in sorted order. Per-day `rng.integers` calls would produce a different
    sequence from the same root, and since the file is never regenerated there would be no way
    to tell afterwards which had been used.

    Seeds are in `[0, MAX_EPISODE_SEED]` because `Demand.load_demand_file` multiplies by 1712
    before `np.random.seed` (P1.9).
    """
    if seeds_per_day < 1:
        raise ValueError(f"seeds_per_day must be >= 1, got {seeds_per_day}")
    days = list(days)
    rng = np.random.default_rng(np.random.SeedSequence(int(seed_root)))
    draws = rng.integers(0, MAX_EPISODE_SEED + 1, size=len(days) * seeds_per_day)
    rows: List[Tuple[str, int, int]] = []
    for d, day in enumerate(days):
        block = draws[d * seeds_per_day:(d + 1) * seeds_per_day]
        duplicates = len(block) - len(set(int(s) for s in block))
        if duplicates:
            # two identical seeds for one day would silently halve that day's sample
            raise RuntimeError(
                f"{duplicates} duplicate seed(s) drawn for {day}: the day would be evaluated "
                f"twice on the same sample path. Raise seeds_per_day or change the seed root "
                f"deliberately -- do not paper over it."
            )
        rows.extend((day, i, int(seed)) for i, seed in enumerate(block))
    return rows


def write_seed_file(rows: Sequence[Tuple[str, int, int]],
                    path: str = SEEDS_FILE) -> str:
    """Write the seed file, **refusing to overwrite an existing one**.

    There is no `--force`. The seeds are reused for every policy ever evaluated, so
    regenerating them does not produce a new experiment, it silently invalidates the comparison
    between every result already recorded and every result to come. `--check-seeds` is the
    supported way to confirm the file still matches its derivation, and it writes nothing.

    The refusal is belt and braces: an explicit `exists` check for a message that explains
    itself, and `open(..., "x")` so there is no window between the two and no way for a later
    refactor to leave only the check.
    """
    if os.path.exists(path):
        raise FileExistsError(
            f"{path} already exists and is never regenerated (§8 P2.14). These seeds are "
            f"reused for every policy ever evaluated; rewriting them would make new KPI "
            f"tables incomparable with every one already recorded, with nothing to show that "
            f"it had happened. Use --check-seeds to verify the file against its derivation. "
            f"If it genuinely must be replaced, that is a decision to take deliberately: move "
            f"the existing file aside by hand first."
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "x", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(SEEDS_COLUMNS)
        writer.writerows(rows)
    return path


def read_seed_file(path: str = SEEDS_FILE) -> List[Tuple[str, int, int]]:
    """`(day, seed_index, seed)` rows from the committed seed file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no seed file at {path}. Create it once with "
            f"`python -m src.rl_gym.evaluate --derive-seeds`."
        )
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        # a changed header means a changed artifact, which is what must not happen silently
        if tuple(header) != SEEDS_COLUMNS:
            raise ValueError(f"{path} header is {header}, expected {list(SEEDS_COLUMNS)}")
        return [(row[0], int(row[1]), int(row[2])) for row in reader if row]


def check_seed_file(path: str = SEEDS_FILE,
                    seeds_per_day: int = DEFAULT_SEEDS_PER_DAY,
                    days: Optional[Sequence[str]] = None,
                    check_split: bool = True) -> List[str]:
    """Re-derive the seeds and compare against the file. Returns a list of differences.

    Writes nothing, which is the whole point: it is how the file's reproducibility can be
    confirmed without the write path ever being taken a second time.

    **Re-derives over both halves** (P2.17). The days come from `held_out_days()`, the sorted
    union of `corpus_val/` and `corpus_eval/`, which is exactly the 26-day list the seeds were
    drawn against -- so the split left all 260 rows identical. Reading one directory would have
    derived 14 days against 260 stored rows and failed, which is how P2.17 found this.

    Also reports **G4**: every label in `docs/RL_GYM_HELDOUT_SPLIT.csv` must equal the half the
    day's file actually sits in. The guards that refuse a cross-half run resolve the half from
    the filesystem, so a stale label could not misroute a run -- but it could misreport which
    half a committed KPI table came from, and a record nobody checks is not a record.
    """
    problems: List[str] = []
    stored = read_seed_file(path)
    derived = derive_eval_seeds(held_out_days() if days is None else days, seeds_per_day)
    if len(stored) != len(derived):
        problems.append(f"row count {len(stored)} != derived {len(derived)}")
    for i, (a, b) in enumerate(zip(stored, derived)):
        if a != b:
            problems.append(f"row {i}: stored {a} != derived {b}")
    if check_split:
        problems += [f"{os.path.basename(DEFAULT_SPLIT_FILE)}: {p}"
                     for p in check_split_file()]
    return problems


def pairs_from_seed_file(days: Optional[Sequence[str]] = None, n_seeds: Optional[int] = None,
                         path: str = SEEDS_FILE,
                         split: Optional[str] = None) -> List[Tuple[str, int]]:
    """`(day, seed)` pairs from the seed file, optionally narrowed.

    :param days: days to keep, in the order given; `None` keeps every day in file order.
    :param n_seeds: keep the **first** `n_seeds` of each day, by `seed_index`. P2.15 takes the
        first 2 of five named days this way.
    :param split: `"val"` or `"test"` (P2.17). With `days` given it **checks** them -- a day of
        the other half raises through `day_dir_for`, which is G3 and the reason a named day
        list cannot smuggle a test day into a validation run. With `days` omitted it
        **selects**: every day of that half, which is how `--final-test` means "the 14".
        `None` keeps the function a pure reader of the file, which is what the seed-file tests
        need.
    """
    rows = read_seed_file(path)
    by_day: Dict[str, List[Tuple[int, int]]] = {}
    for day, index, seed in rows:
        by_day.setdefault(day, []).append((index, seed))
    if split is not None and split not in HELD_OUT_DIRS:
        # an unrecognised split must not fall through to "the test days" by `else`
        raise ValueError(f"split must be one of {sorted(HELD_OUT_DIRS)}, got {split!r}")
    if days is None and split is not None:
        days = validation_days() if split == VALIDATION_MODE else test_days()
    if days is not None and split is not None:
        for day in days:
            day_dir_for(day, split)          # raises on a day from the other half (G3)
    order = list(days) if days is not None else list(dict.fromkeys(d for d, _, _ in rows))
    pairs: List[Tuple[str, int]] = []
    for day in order:
        if day not in by_day:
            raise ValueError(f"day {day!r} is not in {path}; it is not an evaluation day")
        entries = sorted(by_day[day])
        if n_seeds is not None:
            if n_seeds > len(entries):
                raise ValueError(
                    f"asked for {n_seeds} seeds of {day} but the file holds {len(entries)}")
            entries = entries[:n_seeds]
        pairs.extend((day, seed) for _index, seed in entries)
    return pairs


# -------------------------------------------------------------------------- #
# stock greedy baseline (D9)
# -------------------------------------------------------------------------- #

def _base_scenario_parameters(constant_cfg_path: str = DEFAULT_CONSTANT_CFG,
                              scenario_cfg_path: str = DEFAULT_SCENARIO_CFG,
                              scenario_row: int = 0):
    """`scenario_parameters` exactly as `run_scenarios` builds them, before any override.

    `run_scenarios` takes only file paths and cannot accept overrides, so the parameters are
    reassembled here the way it does -- the study name derived from the config path, then
    `n_cpu_per_sim`, `evaluate`, `log_level` and `keep_old` as it sets them, then
    `constant + row`. The four values match `run_examples`' own call for this scenario
    (`log_level="info"`, `n_cpu_per_sim=1`, and `evaluate=1` / `keep_old=False` by default), so
    a run of the reference pair is the same run that produced `docs/baseline_user_stats.csv`.

    No scenario file is edited: the overrides land on this in-memory object only.
    """
    constant_cfg = config.ConstantConfig(constant_cfg_path)
    scenario_cfgs = config.ScenarioConfig(scenario_cfg_path)
    if not scenario_cfgs:
        raise ValueError(f"no active scenario rows in {scenario_cfg_path}")
    if not 0 <= scenario_row < len(scenario_cfgs):
        raise IndexError(f"scenario_row {scenario_row} out of range for "
                         f"{len(scenario_cfgs)} row(s)")
    constant_cfg[G_STUDY_NAME] = derive_study_name(constant_cfg_path)
    constant_cfg["n_cpu_per_sim"] = 1
    constant_cfg["evaluate"] = 1
    constant_cfg["log_level"] = "info"
    constant_cfg["keep_old"] = False
    return constant_cfg + scenario_cfgs[scenario_row]


def output_dir_for(scenario_name: str, constant_cfg_path: str = DEFAULT_CONSTANT_CFG,
                   scenario_cfg_path: str = DEFAULT_SCENARIO_CFG,
                   scenario_row: int = 0) -> str:
    """Where FleetPy would write a run named `scenario_name`.

    Through `get_directory_dict`, never a reassembled path, so this cannot drift from what
    `FleetSimulationBase` computes at construction.
    """
    params = _base_scenario_parameters(constant_cfg_path, scenario_cfg_path, scenario_row)
    params[G_SCENARIO_NAME] = scenario_name
    return get_directory_dict(dict(params))[G_DIR_OUTPUT]


def results_dir(constant_cfg_path: str = DEFAULT_CONSTANT_CFG) -> str:
    """`studies/<study>/results/`, the parent of every run's output directory, both policies'.

    What the P2.20 pre-flight checks refuse inside a cloud-synced folder before a run starts.
    """
    return os.path.dirname(output_dir_for(GREEDY_PREFIX, constant_cfg_path))


def guard_output_dir(scenario_parameters, row_scenario_name: str) -> str:
    """Raise unless this run writes somewhere other than the reference run (trap 1).

    `FleetSimulationBase.__init__` calls `create_or_empty_dir` on the output directory, so a run
    that kept the row's `scenario_name` would delete the reference results --- the run both
    `docs/baseline_*.csv` files came from, and so the arbiter of every byte gate. Checked twice,
    by name and by resolved path, because the name is the mechanism but the path is the damage.

    :return: this run's output directory
    """
    name = scenario_parameters[G_SCENARIO_NAME]
    if name == row_scenario_name:
        raise RuntimeError(
            f"refusing to run with scenario_name {name!r}: that is the scenario row's own "
            f"name, so create_or_empty_dir would wipe the reference run's output directory "
            f"-- the run docs/baseline_user_stats.csv and docs/baseline_op_stats.csv came "
            f"from (trap 1). Every evaluation run must carry a unique scenario_name."
        )
    out_dir = get_directory_dict(dict(scenario_parameters))[G_DIR_OUTPUT]
    reference = get_directory_dict(
        {**dict(scenario_parameters), G_SCENARIO_NAME: row_scenario_name})[G_DIR_OUTPUT]
    if os.path.abspath(out_dir) == os.path.abspath(reference):
        raise RuntimeError(
            f"refusing to run: scenario_name {name!r} resolves to the reference run's output "
            f"directory {reference!r} (trap 1)."
        )
    return out_dir


def build_greedy_scenario_parameters(day: str, seed: int,
                                     constant_cfg_path: str = DEFAULT_CONSTANT_CFG,
                                     scenario_cfg_path: str = DEFAULT_SCENARIO_CFG,
                                     scenario_row: int = 0,
                                     split: Optional[str] = None):
    """Stock-greedy `scenario_parameters` for one `(day, seed)`.

    Four overrides on top of `_base_scenario_parameters`, per §8 P2.14: the demand directory,
    the demand file, the seed, and a unique `scenario_name`. The operator and simulation
    environment are left as the row and the constant config have them --- and asserted to be the
    **stock** ones, because a baseline that quietly ran `RLPoolingIRSOnly` would make D9's
    comparison compare the RL path with itself.

    **The demand directory is derived from the day, never passed in** (G2, P2.17): `day_dir_for`
    resolves it from where the day's file actually sits, so `corpus/` and `corpus_excluded/` are
    unreachable from here and a day cannot be run against the wrong half's directory. `split`,
    when given, additionally requires the day to belong to that half -- that is G3, and it is
    what makes `--final-test` unable to touch a validation day and `--validation` unable to
    touch a test day.
    """
    params = _base_scenario_parameters(constant_cfg_path, scenario_cfg_path, scenario_row)
    row_scenario_name = params[G_SCENARIO_NAME]
    params[G_DAY_DIR] = day_dir_for(day, split)
    params[G_RQ_FILE] = f"{day}.csv"
    params[G_RANDOM_SEED] = int(seed)
    params[G_SCENARIO_NAME] = f"{GREEDY_PREFIX}_{day}_s{int(seed)}"

    op_module = params[G_OP_MODULE]
    sim_env = params[G_SIM_ENV]
    if op_module != "PoolingIRSOnly" or sim_env != "ImmediateDecisionsSimulation":
        raise RuntimeError(
            f"the greedy baseline must be stock FleetPy (D9), but this scenario resolves to "
            f"op_module={op_module!r}, sim_env={sim_env!r}. Expected 'PoolingIRSOnly' and "
            f"'ImmediateDecisionsSimulation'."
        )
    out_dir = guard_output_dir(params, row_scenario_name)
    # P2.20, per run: the pre-flight in `run_greedy_pairs` checks the results directory once,
    # and this is the backstop for any caller that builds parameters directly
    refuse_synced_path(out_dir, "stock-greedy simulation output")
    return params


def run_greedy_pair(task: Sequence[Any]) -> Dict[str, Any]:
    """One stock-greedy run. Picklable, so it can be the body of a `Pool` task.

    The parameters are built **inside** the worker, so nothing that holds a routing engine or a
    file handle crosses the process boundary.

    :param task: `(day, seed)` or `(day, seed, split)`. The split travels with the task rather
        than being bound once in the parent, so each worker re-checks the half it is running
        (P2.17) -- the check costs two `isfile` calls and means a mixed pair list cannot have
        one element slip through.

    **Constructed and run here, not through `run_examples.run_single_simulation`** (P2.20). That
    function is `load_simulation_environment(params).run()` plus a `bugfix` branch that swallows
    exceptions; the two lines are repeated so `RowCounter` can be attached to the simulation
    object, and the branch is left out, so **an exception in a greedy run propagates** rather
    than being printed and swallowed, which would have handed the half-written directory to the
    KPI table as a finished run. The run is then checked against its count, and
    `rl_output_rows.json` written, exactly as the env does for an RL episode. That this is still
    stock FleetPy is proved by `--verify`: the reference pair reproduces both committed
    baselines byte for byte through this function.
    """
    from src.misc.init_modules import load_simulation_environment

    day, seed = str(task[0]), int(task[1])
    split = task[2] if len(task) > 2 else None
    params = build_greedy_scenario_parameters(day, seed, split=split)
    out_dir = get_directory_dict(dict(params))[G_DIR_OUTPUT]
    t0 = time.perf_counter()
    sim = load_simulation_environment(params)
    counter = RowCounter.attach(sim)
    sim.run()
    check_episode_output(out_dir, counter)
    return {"policy": "greedy", "day": day, "seed": seed, "output_dir": out_dir,
            "split": split_of_day(day), "wall_seconds": time.perf_counter() - t0}


def run_greedy_pairs(pairs: Sequence[Tuple[str, int]], processes: int = 1,
                     split: Optional[str] = None) -> List[Dict[str, Any]]:
    """Stock greedy on every pair, one simulation per process.

    `maxtasksperchild=1` so each run gets a **fresh** process. That is what §8 P2.14's "one
    process per run" asks for, and it matters: `FleetSimulationBase.__init__` seeds the global
    RNG and reconfigures process-global logging, so a reused interpreter is a shared-state risk
    in the one place a byte comparison is the arbiter.

    `split` is checked for **every** pair before the first simulation starts (P2.17), so a
    cross-half day costs no simulation time -- the same principle as P2.16's pre-flight pairing
    check.
    """
    pairs = [(str(d), int(s)) for d, s in pairs]
    # P2.20: before any process starts, like the split check below
    refuse_synced_path(results_dir(), "stock-greedy simulation output")
    if split is not None:
        for day, _seed in pairs:
            day_dir_for(day, split)
    tasks = [(d, s, split) for d, s in pairs]
    if processes <= 1:
        return [run_greedy_pair(t) for t in tasks]
    from multiprocessing import Pool
    with Pool(processes=processes, maxtasksperchild=1) as pool:
        return pool.map(run_greedy_pair, tasks)


# -------------------------------------------------------------------------- #
# env-driven greedy -- D9's premise, not the baseline
# -------------------------------------------------------------------------- #

def run_env_greedy_pair(day: str, seed: int, cfg: Optional[Dict[str, Any]] = None,
                        split: Optional[str] = None) -> Dict[str, Any]:
    """The scripted greedy policy through the Gym API on one pair, output on.

    This exists to test **D9's premise** -- that env-driven greedy and stock greedy give the
    same KPI row -- and to be the thing P1.10's gate drives on the reference path. It is not the
    evaluation baseline; stock FleetPy is (D9).

    Driven with the **raw** env observation and never a normalised one: after `VecNormalize`
    each `delta_cfv` slot is scaled by its own running statistics, so an argmin across slots
    would compare z-scores from different distributions. `greedy_action`'s `is_valid` check
    refuses such a vector outright (P2.8), which is why no wrapper appears here.

    The day's directory is resolved from the day (G2) and checked against `split` when given
    (G3), exactly as on the stock-greedy path -- the two policies must never differ in which
    half they read.
    """
    from src.rl_gym.gym_env import SDPDPAssignmentEnv
    from src.rl_gym.policies import greedy_action

    env_cfg = {
        "constant_cfg_path": DEFAULT_CONSTANT_CFG,
        "scenario_cfg_path": DEFAULT_SCENARIO_CFG,
        "K": 8,
        "skip_output": False,
        "env_id": 0,
        "day_dir": day_dir_for(day, split),
        "day": str(day),
        "episode_seed": int(seed),
        **(cfg or {}),
    }
    env = SDPDPAssignmentEnv(env_cfg)
    t0 = time.perf_counter()
    obs, _info = env.reset()
    steps = 0
    terminated = False
    while not terminated:
        action = greedy_action(obs, env.action_masks(), env.observation_layout)
        obs, _reward, terminated, truncated, _info = env.step(action)
        assert not truncated, "episode must end terminated, not truncated (D6)"
        steps += 1
    summary = env.episode_summary()
    scenario_name = env.sim.scenario_name
    env.close()
    return {"policy": "env_greedy", "day": str(day), "seed": int(seed),
            "output_dir": output_dir_for(scenario_name), "steps": steps,
            "split": split_of_day(day),
            "summary": summary, "wall_seconds": time.perf_counter() - t0}


# -------------------------------------------------------------------------- #
# the trained policy
# -------------------------------------------------------------------------- #

def resolve_pairs_split(pairs: Sequence[Tuple[str, int]],
                        split: Optional[str] = None) -> str:
    """The one half every pair belongs to, raising if they disagree (G2/G3, P2.17).

    The RL path runs all its pairs through one `SubprocVecEnv` with a single `day_dir`, so a
    mixed pair list has no valid answer -- and silently reading one directory would make half
    the pairs fail deep inside `load_demand_file`. With `split` given, every pair is checked
    against it; without, the half is derived from the days and must be unanimous.
    """
    if not pairs:
        raise ValueError("no (day, seed) pairs")
    found = {day_dir_for(day, split) for day, _seed in pairs}
    if len(found) > 1:
        halves = {d: sorted({day for day, _s in pairs
                             if day_dir_for(day) == d}) for d in sorted(found)}
        raise ValueError(
            f"the pair list spans both halves of the held-out split, which cannot run in one "
            f"evaluation: " + "; ".join(f"{d} -> {', '.join(days)}"
                                        for d, days in halves.items())
            + ". Choose validation or test and pass only that half's days (P2.17)."
        )
    return found.pop()


def run_rl_pairs(pairs: Sequence[Tuple[str, int]], checkpoint: str,
                 cfg: Optional[Dict[str, Any]] = None, n_workers: int = 1,
                 split: Optional[str] = None,
                 stats_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """The trained policy on every pair: deterministic actions, output on.

    `eval_pairs` mode (P2.3) pins both the day and the seed of each episode, and
    `callbacks.partition_eval_pairs` gives each worker its own slice -- so with the default
    single worker the episodes run the pairs in order, one each.

    **The venv is built raw and then wrapped by `load_vecnormalize`**, rather than by
    `train_sdpdp.build_eval_venv`, because the statistics must be the ones saved beside this
    checkpoint. A freshly constructed `VecNormalize` starts at `count = 1e-4, mean = 0, var = 1`
    and would feed the policy observations on a scale it never trained on -- §5.3's silent and
    very common failure. `load_vecnormalize` also sets `training = False` and
    `norm_reward = False`.

    `stats_path` names a statistics file explicitly, for an old run folder whose checkpoints
    share one `vecnormalize.pkl` (P2.16). It is pairing-checked like any other, so it is
    accepted for the checkpoint it belongs to and refused for the rest. **The pair is verified
    before the workers are built**, so a refusal costs no simulation and orphans nothing.

    `split` names the half (P2.17). The single `day_dir` the venv gets is resolved from the
    pairs themselves, so it cannot be aimed at a directory the days are not in, and a pair list
    spanning both halves raises rather than half-failing inside `load_demand_file`.

    Each episode's output directory is found through `episode_summary()["scenario_name"]`, which
    the env records for exactly this purpose (P2.14).
    """
    # P2.20, first of the pre-flight checks below and ahead even of the torch imports: each
    # worker's env would refuse a cloud-synced output folder too, but only from inside the
    # worker, after the venv is built
    refuse_synced_path(results_dir(), "RL simulation output")

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.vec_env import SubprocVecEnv

    import train_sdpdp
    from src.rl_gym.callbacks import partition_eval_pairs

    pairs = [(str(d), int(s)) for d, s in pairs]
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if n_workers > len(pairs):
        raise ValueError(f"n_workers {n_workers} exceeds the {len(pairs)} pair(s): a worker "
                         f"with no pair has nothing to run")
    env_cfg = {
        "constant_cfg_path": DEFAULT_CONSTANT_CFG,
        "scenario_cfg_path": DEFAULT_SCENARIO_CFG,
        "K": 8,
        "gamma": train_sdpdp.DEFAULT_HYPERPARAMS["gamma"],
        **(cfg or {}),
        # output is what the KPI script reads; not negotiable for this harness
        "skip_output": False,
    }
    # before anything is constructed, in this order: the half (P2.17), the statistics pairing
    # (P2.16) and the observation contract (P2.18) -- after the synced-folder check (P2.20) at
    # the top. None of a cross-half day, a mispaired statistics file or a checkpoint whose
    # features have been re-defined may cost a SubprocVecEnv whose workers the raise would then
    # orphan. The contract check builds one env in this process, which loads no simulation and
    # takes about 70 ms.
    day_dir = resolve_pairs_split(pairs, split)
    train_sdpdp.verify_statistics_pairing(checkpoint, stats_path)
    train_sdpdp.verify_artifact_contract(checkpoint, env_cfg)
    # eval_pairs fixes day and seed, and P2.3 raises if it meets any training seeding key
    base = {k: v for k, v in env_cfg.items()
            if k not in ("base_seed", "day", "episode_seed", "eval_pairs", "day_dir")}
    base["day_dir"] = day_dir
    slices = partition_eval_pairs(pairs, n_workers)

    train_sdpdp.prepare_multiprocessing()
    raw = SubprocVecEnv([
        train_sdpdp.make_env(base, i, n_workers, overrides={"eval_pairs": slices[i]})
        for i in range(n_workers)
    ])
    venv = train_sdpdp.load_vecnormalize(checkpoint, raw, for_evaluation=True,
                                         stats_path=stats_path)
    model = MaskablePPO.load(checkpoint, device="cpu")

    rows: List[Dict[str, Any]] = []
    try:
        obs = venv.reset()
        while len(rows) < len(pairs):
            masks = get_action_masks(venv)
            actions, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, _rewards, dones, infos = venv.step(actions)
            for i, done in enumerate(dones):
                if not done or len(rows) >= len(pairs):
                    continue
                summary = infos[i].get("episode_summary")
                if summary is None:
                    raise RuntimeError(
                        f"worker {i} finished an episode with no episode_summary in info; "
                        f"the KPI row cannot be located without its scenario_name")
                rows.append({
                    "policy": "rl", "day": summary.get("day"),
                    "seed": summary.get("random_seed"),
                    "output_dir": output_dir_for(summary["scenario_name"]),
                    "split": split_of_day(summary["day"]),
                    "summary": summary,
                })
    finally:
        # `SubprocVecEnv`'s worker resets unconditionally on `done`, so the step that reports
        # the last episode has already started one more. Closing abandons it, which leaves
        # **one** stray output directory per worker per invocation -- P2.13 documents the same
        # effect inside `evaluate_policy`. It is not removed here: its name would have to be
        # reconstructed by string surgery on P1.9's naming rule, and speculative `rmtree` of a
        # results directory is the one thing trap 1 says not to write. Nothing globs for runs --
        # the KPI table is built from the explicit directories returned below.
        #
        # **It can hold a partial `1_user-stats.csv`** (corrected in P2.20; P2.14 said it held
        # none). A worker that has finished its slice is still stepped -- `eval_pairs` wraps --
        # until the slowest worker finishes, so its abandoned episode runs and flushes rows at
        # every time step until this `close()`. P2.14 measured a single-pair run, where it was
        # abandoned at once. What keeps it from being read as a finished run is the record:
        # the abandoned episode never reaches its end-of-episode check, so it has no
        # `rl_output_rows.json`, and `kpis.kpi_row` refuses a directory without one.
        venv.close()
    return rows


# -------------------------------------------------------------------------- #
# the table
# -------------------------------------------------------------------------- #

def kpi_rows(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """`kpis.kpi_row` for every run, carrying its policy label, day and seed."""
    return [kpis.kpi_row(r["output_dir"], policy=r.get("policy"), day=r.get("day"),
                         seed=r.get("seed")) for r in runs]


def report_header(split: Optional[str], days: Sequence[str],
                  checkpoint: Optional[str] = None,
                  stats_file: Optional[str] = None) -> str:
    """One provenance block naming the half a KPI table came from (P2.17).

    A table of numbers does not say which days produced it, and the whole point of the split is
    that validation and test figures must never be mistaken for each other afterwards. So the
    half, the day list and the checkpoint go **in the report**, not only in the console
    scrollback that produced it.
    """
    lines = ["## Provenance", ""]
    if split is None:
        lines.append("- half: **not stated** (no split mode given)")
    else:
        lines.append(f"- half: **{split}** "
                     f"({HELD_OUT_DIRS[split]}/, "
                     f"{'every choice is made here' if split == VALIDATION_MODE else 'FINAL RESULTS -- read once'})")
    lines.append(f"- days ({len(days)}): {', '.join(days)}")
    if checkpoint:
        lines.append(f"- checkpoint: {os.path.basename(checkpoint)}")
    if stats_file:
        lines.append(f"- statistics: {os.path.basename(stats_file)}")
    lines.append(f"- seed file: {os.path.basename(SEEDS_FILE)}; "
                 f"split record: {os.path.basename(DEFAULT_SPLIT_FILE)}")
    lines.append("")
    return "\n".join(lines)


def format_report(rows: Sequence[Dict[str, Any]], header: str = "") -> str:
    """The per-run table plus the three aggregations, per policy."""
    out = ([header] if header else []) + ["## Per run", "", kpis.format_per_run_table(rows), ""]
    policies = list(dict.fromkeys(r.get("policy") for r in rows))
    for level in ("day", "month", "overall"):
        out.append(f"## By {level}")
        out.append("")
        for policy in policies:
            subset = [r for r in rows if r.get("policy") == policy]
            out.append(kpis.format_table(subset, by=level, title=f"policy = {policy}"))
            out.append("")
    return "\n".join(out)


# -------------------------------------------------------------------------- #
# the committed verification of this work item
# -------------------------------------------------------------------------- #

def _md5(path: str) -> str:
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _same_bytes(a: str, b: str) -> bool:
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()


def verify(keep: bool = False, verbose: bool = True) -> int:
    """§8 P2.14's verification, as a committed re-runnable command.

    Prints a `PASS`/`FAIL` line per check and returns the number of failures, so the CLI can
    exit non-zero. Needs no torch: every check here is greedy, stock or env-driven.

    **15 checks since P2.17** (13 before): the split's guards refusing in both directions, and
    the committed labels agreeing with the directories. The reference pair 2024-06-20 and the
    second pair 2024-08-13 are both **validation** days now, so six of the original checks read
    their demand out of `corpus_val/` -- with identical expected results, since a move changes a
    file's directory and not its bytes, and no KPI reads the directory name.

    **17 since P2.20.** The reference output directory must exist with both CSVs: through P2.19
    the byte-unchanged check compared only the files it found, so with the directory missing --
    as in a fresh clone, `results/` being git-ignored -- it compared nothing and passed. Now that
    is a FAIL naming `REFERENCE_RERUN_COMMAND`, and so is the byte-unchanged check, which has
    nothing to compare. And both reference-pair runs must carry an integrity record matching
    their files, at the row counts of the two committed baselines.
    """
    checks: List[Tuple[bool, str]] = []

    def record(ok: bool, label: str) -> None:
        checks.append((bool(ok), label))
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {label}", flush=True)

    # P2.20: before anything is written
    refuse_synced_path(results_dir(), "the verification's simulation output")

    reference_dir = output_dir_for(
        _base_scenario_parameters()[G_SCENARIO_NAME])
    reference_files = (kpis.USER_STATS_FILE, kpis.OP_STATS_FILE)
    before = {name: _md5(os.path.join(reference_dir, name))
              for name in reference_files
              if os.path.isfile(os.path.join(reference_dir, name))}
    if verbose:
        print(f"reference directory: {reference_dir}")
        for name, digest in sorted(before.items()):
            print(f"  before  {name}  md5 {digest}")
    missing = [name for name in reference_files if name not in before]
    record(not missing,
           f"the reference output directory holds both CSVs"
           f"{'' if not missing else f' -- missing {missing} in {reference_dir}; recreate it, from the repository root, with: {REFERENCE_RERUN_COMMAND}'}")

    created: List[str] = []
    try:
        # 1. the seed file exists, matches its derivation, and refuses to be overwritten
        record(not check_seed_file(), "seed file matches its derivation")
        try:
            write_seed_file([("x", 0, 1)])
            record(False, "write_seed_file refuses to overwrite the committed file")
        except FileExistsError:
            record(True, "write_seed_file refuses to overwrite the committed file")

        # 1b. P2.17's G4: the committed labels agree with the directories the files are in
        split_problems = check_split_file()
        record(not split_problems,
               f"the {len(read_split_file())} committed split labels agree with the two "
               f"directories"
               f"{'' if not split_problems else f' -- differs: {split_problems[:3]}'}")

        # 1c. P2.17's G3, in **both** directions. A final-evaluation command must not be able
        # to reach a validation day, and a validation command must not be able to reach a test
        # day; one failing direction with the other passing is the shape this would rot into.
        val_day, test_day = validation_days()[0], test_days()[0]
        both_ways = []
        for day, wrong, right in ((val_day, TEST_MODE, VALIDATION_MODE),
                                  (test_day, VALIDATION_MODE, TEST_MODE)):
            try:
                build_greedy_scenario_parameters(day, 1, split=wrong)
                both_ways.append(f"{day} was accepted in {wrong!r} mode")
            except ValueError:
                pass
            try:
                build_greedy_scenario_parameters(day, 1, split=right)
            except ValueError as exc:
                both_ways.append(f"{day} was refused in its own {right!r} mode: {exc}")
        record(not both_ways,
               f"G3: {val_day} refused in test mode and {test_day} in validation mode, each "
               f"accepted in its own"
               f"{'' if not both_ways else f' -- {both_ways}'}")

        # 2. the trap-1 guard fires on the row's own scenario_name
        params = _base_scenario_parameters()
        row_name = params[G_SCENARIO_NAME]
        try:
            guard_output_dir(params, row_name)
            record(False, "guard_output_dir refuses the reference scenario_name")
        except RuntimeError:
            record(True, "guard_output_dir refuses the reference scenario_name")

        # 3. stock greedy on the reference pair, byte for byte against both baselines.
        # The reference day is a **validation** day since P2.17, so this reads corpus_val/ --
        # the split is passed explicitly so the run states which half it means.
        stock = run_greedy_pair((REFERENCE_DAY, REFERENCE_SEED, VALIDATION_MODE))
        created.append(stock["output_dir"])
        record(_same_bytes(os.path.join(stock["output_dir"], kpis.USER_STATS_FILE),
                           BASELINE_USER_STATS),
               f"stock greedy ({REFERENCE_DAY}, {REFERENCE_SEED}) 1_user-stats.csv "
               f"== docs/baseline_user_stats.csv")
        record(_same_bytes(os.path.join(stock["output_dir"], kpis.OP_STATS_FILE),
                           BASELINE_OP_STATS),
               f"stock greedy ({REFERENCE_DAY}, {REFERENCE_SEED}) 2-0_op-stats.csv "
               f"== docs/baseline_op_stats.csv")

        # 4. a second baseline run on a different pair leaves both directories intact
        other_day, other_seed = pairs_from_seed_file(days=["2024-08-13"], n_seeds=1,
                                                     split=VALIDATION_MODE)[0]
        second = run_greedy_pair((other_day, other_seed, VALIDATION_MODE))
        created.append(second["output_dir"])
        record(os.path.isfile(os.path.join(stock["output_dir"], kpis.USER_STATS_FILE))
               and os.path.isfile(os.path.join(second["output_dir"], kpis.USER_STATS_FILE)),
               "two baseline runs on different pairs leave both output directories intact")

        # 5. the KPI row reconciles with P1.10's figures
        stock_row = kpis.kpi_row(stock["output_dir"], policy="greedy", day=REFERENCE_DAY,
                                 seed=REFERENCE_SEED)
        expected = {"requests": 445, "served": 288, "declines": 118, "cancellations": 28,
                    "no_shows": 2, "operator_rejections": 0, "no_candidate_requests": 9,
                    "offers_made": 436, "accepted_offers": 318, "unclassified": 0,
                    "unserved_at_horizon": 0}
        mismatch = {k: (stock_row[k], v) for k, v in expected.items() if stock_row[k] != v}
        record(not mismatch,
               f"stock greedy KPI counts reconcile with P1.10"
               f"{'' if not mismatch else f' -- differs: {mismatch}'}")

        # 6. D9's premise: env-driven greedy gives an identical KPI row
        env_run = run_env_greedy_pair(REFERENCE_DAY, REFERENCE_SEED, split=VALIDATION_MODE)
        created.append(env_run["output_dir"])
        env_row = kpis.kpi_row(env_run["output_dir"], policy="env_greedy", day=REFERENCE_DAY,
                               seed=REFERENCE_SEED)
        compared = list(kpis.METRICS) + list(kpis.COUNTS)
        differing = {k: (stock_row[k], env_row[k]) for k in compared
                     if stock_row[k] != env_row[k]}
        record(not differing,
               f"D9: env-driven greedy KPI row identical to stock on "
               f"({REFERENCE_DAY}, {REFERENCE_SEED})"
               f"{'' if not differing else f' -- differs: {differing}'}")

        # 7. and the two output files are byte-identical, which is the stronger statement
        record(_same_bytes(os.path.join(env_run["output_dir"], kpis.USER_STATS_FILE),
                           BASELINE_USER_STATS),
               "env-driven greedy 1_user-stats.csv == docs/baseline_user_stats.csv")
        record(_same_bytes(os.path.join(env_run["output_dir"], kpis.OP_STATS_FILE),
                           BASELINE_OP_STATS),
               "env-driven greedy 2-0_op-stats.csv == docs/baseline_op_stats.csv")

        # 8. the summary JSON exists for the env run and not for the stock run
        record(os.path.isfile(os.path.join(env_run["output_dir"],
                                           kpis.EPISODE_SUMMARY_FILE))
               and not os.path.isfile(os.path.join(stock["output_dir"],
                                                   kpis.EPISODE_SUMMARY_FILE)),
               "rl_episode_summary.json written by the env run only")

        # 8b. P2.20: both runs were checked at their end and recorded what they wrote, and the
        # record is the committed baselines' own row counts -- so the counter counts every row,
        # on the stock path and the env path alike, and no more
        expected_rows = {kpis.USER_STATS_FILE: count_data_rows(BASELINE_USER_STATS),
                         op_stats_file(0): count_data_rows(BASELINE_OP_STATS)}
        records = {}
        for label, run in (("stock", stock), ("env", env_run)):
            try:
                records[label] = read_rows_record(run["output_dir"])
            except Exception as exc:
                records[label] = f"{type(exc).__name__}: {exc}"
        record(all(r == expected_rows for r in records.values()),
               f"integrity record of both runs == the baselines' row counts "
               f"{expected_rows}"
               f"{'' if all(r == expected_rows for r in records.values()) else f' -- got {records}'}")

        # 9. the table renders at all three levels, and carries the half it came from (P2.17)
        rows = [stock_row, env_row]
        report = format_report(rows, header=report_header(
            VALIDATION_MODE, [REFERENCE_DAY, other_day]))
        record(all(level in report for level in ("## By day", "## By month", "## By overall"))
               and "## Provenance" in report and "half: **val**" in report,
               "KPI report renders per run, per day, per month and overall, with its half")
        if verbose:
            print()
            print(kpis.format_per_run_table(rows))
            print()
    finally:
        after = {name: _md5(os.path.join(reference_dir, name))
                 for name in before}
        if verbose:
            for name, digest in sorted(after.items()):
                print(f"  after   {name}  md5 {digest}")
        record(not missing and after == before,
               "the reference output directory is byte-unchanged by every run above"
               f"{'' if not missing else ' -- nothing to compare: the directory or a file is missing'}")
        if not keep:
            for directory in created:
                if os.path.isdir(directory):
                    shutil.rmtree(directory, ignore_errors=True)

    failures = sum(1 for ok, _ in checks if not ok)
    if verbose:
        print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    return failures


# -------------------------------------------------------------------------- #
# CLI
# -------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--derive-seeds", action="store_true",
                        help="write docs/RL_GYM_EVAL_SEEDS.csv; refuses if it exists")
    parser.add_argument("--check-seeds", action="store_true",
                        help="re-derive the seeds and compare against the file, and check the "
                             "split labels against the two directories; writes nothing")
    parser.add_argument("--verify", action="store_true",
                        help="run this work item's verification; exits non-zero on any failure")
    parser.add_argument("--greedy", action="store_true", help="run the stock greedy baseline")
    parser.add_argument("--rl", action="store_true", help="run a trained checkpoint")
    half = parser.add_mutually_exclusive_group()
    half.add_argument("--validation", action="store_true",
                      help="run on the 12 validation days (corpus_val/). Every choice -- "
                           "checkpoint, reward weights, when to stop -- is made here")
    half.add_argument("--final-test", action="store_true", dest="final_test",
                      help="run on the 14 TEST days (corpus_eval/): the final results, read "
                           "once, after everything is settled. Named in full on purpose -- "
                           "there is no --test, so this cannot be typed while meaning "
                           "--validation")
    parser.add_argument("--checkpoint", help="the .zip to load for --rl")
    parser.add_argument("--stats",
                        help="--rl only: an explicit VecNormalize statistics file, for a run "
                             "folder written before P2.16 whose checkpoints share one "
                             "vecnormalize.pkl. Pairing-checked like any other, so it is "
                             "refused for every checkpoint but the one it belongs to")
    parser.add_argument("--days", help="comma-separated evaluation days; default every day")
    parser.add_argument("--seeds", type=int,
                        help="first N seeds of each day; default every seed in the file")
    parser.add_argument("--processes", type=int, default=1,
                        help="parallel stock-greedy simulations (one process per run)")
    parser.add_argument("--workers", type=int, default=1, help="--rl worker count")
    parser.add_argument("--seeds-per-day", type=int, default=DEFAULT_SEEDS_PER_DAY,
                        help="only for --derive-seeds / --check-seeds")
    parser.add_argument("--out", help="write the KPI report here instead of printing only")
    parser.add_argument("--keep", action="store_true",
                        help="keep the output directories --verify creates")
    args = parser.parse_args()

    if args.derive_seeds:
        days = held_out_days()
        rows = derive_eval_seeds(days, args.seeds_per_day)
        try:
            path = write_seed_file(rows)
        except FileExistsError as exc:
            # the refusal is the designed outcome, not a crash: report it and exit non-zero
            print(f"refused: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {path}: {len(days)} day(s) x {args.seeds_per_day} seed(s) "
              f"= {len(rows)} rows, seed root {EVAL_SEED_ROOT}")
        return 0

    if args.check_seeds:
        problems = check_seed_file(seeds_per_day=args.seeds_per_day)
        if problems:
            print(f"{len(problems)} difference(s):")
            for problem in problems[:20]:
                print(f"  {problem}")
            return 1
        print(f"{SEEDS_FILE} matches its derivation "
              f"({len(read_seed_file())} rows, seed root {EVAL_SEED_ROOT}, "
              f"{len(held_out_days())} held-out days)")
        print(f"{DEFAULT_SPLIT_FILE} agrees with the directories "
              f"({len(validation_days())} validation + {len(test_days())} test)")
        return 0

    if args.verify:
        return 1 if verify(keep=args.keep) else 0

    if not (args.greedy or args.rl):
        parser.error("nothing to do: pass --derive-seeds, --check-seeds, --verify, "
                     "--greedy or --rl")

    # P2.20: the report is written last, so its folder is checked first -- a refusal must not
    # come after an hour of simulation. The simulation output is checked by the two runners.
    if args.out:
        refuse_synced_path(os.path.dirname(os.path.abspath(args.out)), "the KPI report")

    # G1: exactly one half, and no default. A default is what gets used by accident, and the
    # accident this guards against -- reading the 14 test days while choosing a checkpoint --
    # cannot be undone afterwards, because the days have then been looked at (P2.17).
    if not (args.validation or args.final_test):
        parser.error(
            "a run must say which half of the held-out split it is on: --validation for the "
            "12 validation days (corpus_val/), where every choice is made, or --final-test "
            "for the 14 test days (corpus_eval/), the final results. There is no default: "
            "choosing a checkpoint or a weight setting on the test days spends them, and "
            "nothing can put that back.")
    split = TEST_MODE if args.final_test else VALIDATION_MODE

    days = [d.strip() for d in args.days.split(",")] if args.days else None
    # with --days given the split **checks** them; without, it **selects** that half's days
    pairs = pairs_from_seed_file(days=days, n_seeds=args.seeds, split=split)
    run_days = list(dict.fromkeys(d for d, _s in pairs))
    print(f"half: {split} ({HELD_OUT_DIRS[split]}/); "
          f"{len(run_days)} day(s), {len(pairs)} (day, seed) pair(s)")
    if split == TEST_MODE:
        print("*** FINAL TEST RUN: these 14 days are the held-back half (P2.17) ***")

    runs: List[Dict[str, Any]] = []
    if args.greedy:
        runs += run_greedy_pairs(pairs, processes=args.processes, split=split)
    if args.rl:
        if not args.checkpoint:
            parser.error("--rl needs --checkpoint")
        runs += run_rl_pairs(pairs, args.checkpoint, n_workers=args.workers,
                             split=split, stats_path=args.stats)

    rows = kpi_rows(runs)
    report = format_report(rows, header=report_header(
        split, run_days, checkpoint=args.checkpoint if args.rl else None,
        stats_file=args.stats))
    print(report)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(report + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
