"""Evaluation harness: the stock-greedy baseline, the RL policy, and the KPI table (P2.14).

**The RL result is meaningless without the greedy baseline on the same held-out days.** This
module runs both on the same `(day, seed)` pairs out of `corpus_eval/` and tabulates them
through `kpis.py`, which reads only output directories and so treats the two identically.

    # once, and never again -- the seed file is an unregenerable artifact
    python -m src.rl_gym.evaluate --derive-seeds
    python -m src.rl_gym.evaluate --check-seeds

    # the committed verification of this work item
    python -m src.rl_gym.evaluate --verify

    # a real comparison: stock greedy and a trained checkpoint on the same pairs
    python -m src.rl_gym.evaluate --greedy --days 2024-06-20,2024-08-13 --seeds 2
    python -m src.rl_gym.evaluate --rl --checkpoint runs/sdpdp/ep8.zip \
        --days 2024-06-20,2024-08-13 --seeds 2

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
from src.rl_gym.corpus import training_day_files  # noqa: E402
from src.rl_gym.gym_env import MAX_EPISODE_SEED, derive_study_name  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCS = os.path.join(ROOT, "studies", "example_study", "scenarios")
DEFAULT_CONSTANT_CFG = os.path.join(SCS, "constant_config_depot_cali_sc_1.csv")
DEFAULT_SCENARIO_CFG = os.path.join(SCS, "example_depot_cali_sc_1.csv")

#: the held-out split (§2.9). Never `corpus/`, never `corpus_excluded/`.
EVAL_DAY_DIR = "corpus_eval"
EVAL_DEMAND_DIR = os.path.join(ROOT, "data", "demand", "example_demand", "matched",
                               "example_network", EVAL_DAY_DIR)

#: the unregenerable seed file and its layout
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


# -------------------------------------------------------------------------- #
# the seed file -- written once, never regenerated
# -------------------------------------------------------------------------- #

def evaluation_days(demand_dir: str = EVAL_DEMAND_DIR) -> List[str]:
    """Sorted filename stems of the evaluation days, `*.csv` only (§2.9).

    Delegates to `corpus.training_day_files`, which despite its name is the single `*.csv`
    directory lister in the project (P2.9). Reusing it is what keeps a `.DS_Store` from ever
    reading as a demand day here, in the artifact that can never be regenerated.
    """
    return training_day_files(demand_dir)


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
                    demand_dir: str = EVAL_DEMAND_DIR) -> List[str]:
    """Re-derive the seeds and compare against the file. Returns a list of differences.

    Writes nothing, which is the whole point: it is how the file's reproducibility can be
    confirmed without the write path ever being taken a second time.
    """
    problems: List[str] = []
    stored = read_seed_file(path)
    derived = derive_eval_seeds(evaluation_days(demand_dir), seeds_per_day)
    if len(stored) != len(derived):
        problems.append(f"row count {len(stored)} != derived {len(derived)}")
    for i, (a, b) in enumerate(zip(stored, derived)):
        if a != b:
            problems.append(f"row {i}: stored {a} != derived {b}")
    return problems


def pairs_from_seed_file(days: Optional[Sequence[str]] = None, n_seeds: Optional[int] = None,
                         path: str = SEEDS_FILE) -> List[Tuple[str, int]]:
    """`(day, seed)` pairs from the seed file, optionally narrowed.

    :param days: days to keep, in the order given; `None` keeps every day in file order.
    :param n_seeds: keep the **first** `n_seeds` of each day, by `seed_index`. P2.15 takes the
        first 2 of five named days this way.
    """
    rows = read_seed_file(path)
    by_day: Dict[str, List[Tuple[int, int]]] = {}
    for day, index, seed in rows:
        by_day.setdefault(day, []).append((index, seed))
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
                                     day_dir: str = EVAL_DAY_DIR):
    """Stock-greedy `scenario_parameters` for one `(day, seed)`.

    Four overrides on top of `_base_scenario_parameters`, per §8 P2.14: the demand directory,
    the demand file, the seed, and a unique `scenario_name`. The operator and simulation
    environment are left as the row and the constant config have them --- and asserted to be the
    **stock** ones, because a baseline that quietly ran `RLPoolingIRSOnly` would make D9's
    comparison compare the RL path with itself.
    """
    params = _base_scenario_parameters(constant_cfg_path, scenario_cfg_path, scenario_row)
    row_scenario_name = params[G_SCENARIO_NAME]
    params[G_DAY_DIR] = day_dir
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
    guard_output_dir(params, row_scenario_name)
    return params


def run_greedy_pair(task: Tuple[str, int]) -> Dict[str, Any]:
    """One stock-greedy run. Picklable, so it can be the body of a `Pool` task.

    The parameters are built **inside** the worker, so nothing that holds a routing engine or a
    file handle crosses the process boundary.
    """
    day, seed = task[0], int(task[1])
    params = build_greedy_scenario_parameters(day, seed)
    out_dir = get_directory_dict(dict(params))[G_DIR_OUTPUT]
    t0 = time.perf_counter()
    import run_examples
    run_examples.run_single_simulation(params)
    return {"policy": "greedy", "day": day, "seed": seed, "output_dir": out_dir,
            "wall_seconds": time.perf_counter() - t0}


def run_greedy_pairs(pairs: Sequence[Tuple[str, int]],
                     processes: int = 1) -> List[Dict[str, Any]]:
    """Stock greedy on every pair, one simulation per process.

    `maxtasksperchild=1` so each run gets a **fresh** process. That is what §8 P2.14's "one
    process per run" asks for, and it matters: `FleetSimulationBase.__init__` seeds the global
    RNG and reconfigures process-global logging, so a reused interpreter is a shared-state risk
    in the one place a byte comparison is the arbiter.
    """
    pairs = [(str(d), int(s)) for d, s in pairs]
    if processes <= 1:
        return [run_greedy_pair(p) for p in pairs]
    from multiprocessing import Pool
    with Pool(processes=processes, maxtasksperchild=1) as pool:
        return pool.map(run_greedy_pair, pairs)


# -------------------------------------------------------------------------- #
# env-driven greedy -- D9's premise, not the baseline
# -------------------------------------------------------------------------- #

def run_env_greedy_pair(day: str, seed: int, cfg: Optional[Dict[str, Any]] = None,
                        day_dir: str = EVAL_DAY_DIR) -> Dict[str, Any]:
    """The scripted greedy policy through the Gym API on one pair, output on.

    This exists to test **D9's premise** -- that env-driven greedy and stock greedy give the
    same KPI row -- and to be the thing P1.10's gate drives on the reference path. It is not the
    evaluation baseline; stock FleetPy is (D9).

    Driven with the **raw** env observation and never a normalised one: after `VecNormalize`
    each `delta_cfv` slot is scaled by its own running statistics, so an argmin across slots
    would compare z-scores from different distributions. `greedy_action`'s `is_valid` check
    refuses such a vector outright (P2.8), which is why no wrapper appears here.
    """
    from src.rl_gym.gym_env import SDPDPAssignmentEnv
    from src.rl_gym.policies import greedy_action

    env_cfg = {
        "constant_cfg_path": DEFAULT_CONSTANT_CFG,
        "scenario_cfg_path": DEFAULT_SCENARIO_CFG,
        "K": 8,
        "skip_output": False,
        "env_id": 0,
        "day_dir": day_dir,
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
            "summary": summary, "wall_seconds": time.perf_counter() - t0}


# -------------------------------------------------------------------------- #
# the trained policy
# -------------------------------------------------------------------------- #

def run_rl_pairs(pairs: Sequence[Tuple[str, int]], checkpoint: str,
                 cfg: Optional[Dict[str, Any]] = None, n_workers: int = 1,
                 day_dir: str = EVAL_DAY_DIR,
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

    Each episode's output directory is found through `episode_summary()["scenario_name"]`, which
    the env records for exactly this purpose (P2.14).
    """
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
    # before anything is constructed: a mispaired or missing statistics file must not cost a
    # SubprocVecEnv whose workers the raise would then orphan (P2.16)
    train_sdpdp.verify_statistics_pairing(checkpoint, stats_path)

    env_cfg = {
        "constant_cfg_path": DEFAULT_CONSTANT_CFG,
        "scenario_cfg_path": DEFAULT_SCENARIO_CFG,
        "K": 8,
        "gamma": train_sdpdp.DEFAULT_HYPERPARAMS["gamma"],
        **(cfg or {}),
        # output is what the KPI script reads; not negotiable for this harness
        "skip_output": False,
    }
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
                    "summary": summary,
                })
    finally:
        # `SubprocVecEnv`'s worker resets unconditionally on `done`, so the step that reports
        # the last episode has already started one more. Closing abandons it, which leaves
        # **one** stray output directory per worker per invocation -- P2.13 documents the same
        # effect inside `evaluate_policy`. Measured in P2.14: such a directory holds only the
        # construction-time files (`00_config.json`, `2_vehicle_types.csv`,
        # `3-0_op-dyn_atts.csv`) and **no `1_user-stats.csv` at all**, so it cannot be mistaken
        # for a finished run. It is not removed here: its name would have to be reconstructed
        # by string surgery on P1.9's naming rule, and speculative `rmtree` of a results
        # directory is the one thing trap 1 says not to write. Nothing globs for runs -- the
        # KPI table is built from the explicit directories returned below.
        venv.close()
    return rows


# -------------------------------------------------------------------------- #
# the table
# -------------------------------------------------------------------------- #

def kpi_rows(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """`kpis.kpi_row` for every run, carrying its policy label, day and seed."""
    return [kpis.kpi_row(r["output_dir"], policy=r.get("policy"), day=r.get("day"),
                         seed=r.get("seed")) for r in runs]


def format_report(rows: Sequence[Dict[str, Any]]) -> str:
    """The per-run table plus the three aggregations, per policy."""
    out = ["## Per run", "", kpis.format_per_run_table(rows), ""]
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
    """
    checks: List[Tuple[bool, str]] = []

    def record(ok: bool, label: str) -> None:
        checks.append((bool(ok), label))
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {label}", flush=True)

    reference_dir = output_dir_for(
        _base_scenario_parameters()[G_SCENARIO_NAME])
    before = {name: _md5(os.path.join(reference_dir, name))
              for name in (kpis.USER_STATS_FILE, kpis.OP_STATS_FILE)
              if os.path.isfile(os.path.join(reference_dir, name))}
    if verbose:
        print(f"reference directory: {reference_dir}")
        for name, digest in sorted(before.items()):
            print(f"  before  {name}  md5 {digest}")

    created: List[str] = []
    try:
        # 1. the seed file exists, matches its derivation, and refuses to be overwritten
        record(not check_seed_file(), "seed file matches its derivation")
        try:
            write_seed_file([("x", 0, 1)])
            record(False, "write_seed_file refuses to overwrite the committed file")
        except FileExistsError:
            record(True, "write_seed_file refuses to overwrite the committed file")

        # 2. the trap-1 guard fires on the row's own scenario_name
        params = _base_scenario_parameters()
        row_name = params[G_SCENARIO_NAME]
        try:
            guard_output_dir(params, row_name)
            record(False, "guard_output_dir refuses the reference scenario_name")
        except RuntimeError:
            record(True, "guard_output_dir refuses the reference scenario_name")

        # 3. stock greedy on the reference pair, byte for byte against both baselines
        stock = run_greedy_pair((REFERENCE_DAY, REFERENCE_SEED))
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
        other_day, other_seed = pairs_from_seed_file(days=["2024-08-13"], n_seeds=1)[0]
        second = run_greedy_pair((other_day, other_seed))
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
        env_run = run_env_greedy_pair(REFERENCE_DAY, REFERENCE_SEED)
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

        # 9. the table renders at all three levels
        rows = [stock_row, env_row]
        report = format_report(rows)
        record(all(level in report for level in ("## By day", "## By month", "## By overall")),
               "KPI report renders per run, per day, per month and overall")
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
        record(after == before,
               "the reference output directory is byte-unchanged by every run above")
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
                        help="re-derive the seeds and compare against the file; writes nothing")
    parser.add_argument("--verify", action="store_true",
                        help="run this work item's verification; exits non-zero on any failure")
    parser.add_argument("--greedy", action="store_true", help="run the stock greedy baseline")
    parser.add_argument("--rl", action="store_true", help="run a trained checkpoint")
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
        days = evaluation_days()
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
              f"({len(read_seed_file())} rows, seed root {EVAL_SEED_ROOT})")
        return 0

    if args.verify:
        return 1 if verify(keep=args.keep) else 0

    if not (args.greedy or args.rl):
        parser.error("nothing to do: pass --derive-seeds, --check-seeds, --verify, "
                     "--greedy or --rl")

    days = [d.strip() for d in args.days.split(",")] if args.days else None
    pairs = pairs_from_seed_file(days=days, n_seeds=args.seeds)
    print(f"{len(pairs)} (day, seed) pair(s)")

    runs: List[Dict[str, Any]] = []
    if args.greedy:
        runs += run_greedy_pairs(pairs, processes=args.processes)
    if args.rl:
        if not args.checkpoint:
            parser.error("--rl needs --checkpoint")
        runs += run_rl_pairs(pairs, args.checkpoint, n_workers=args.workers,
                             stats_path=args.stats)

    rows = kpi_rows(runs)
    report = format_report(rows)
    print(report)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(report + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
