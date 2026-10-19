"""MaskablePPO training for SDPDPAssignmentEnv, and the harnesses around it.

    python train_sdpdp.py                      # P1.11's smoke test: workers, no learning
    python train_sdpdp.py --train              # P2.13's training run
    python train_sdpdp.py --verify-run         # P2.13's verification, pass/fail per check
    python train_sdpdp.py --workers 2          # fewer workers
    python train_sdpdp.py --no-vecnormalize    # the bare venv, as P1.11 measured it

Every env is constructed **inside** its worker (`make_env`), never built in the parent and
shipped across: FleetPy objects hold routing engines and open file handles and are not
picklable. `SubprocVecEnv` and never `DummyVecEnv` (D8), because
`FleetSimulationBase.__init__` reconfigures process-global logging and seeds the global RNG.

What each item added here:

- **P1.11** — `make_env`, the worker plumbing, the smoke test and its measurements.
- **P2.12** — the `VecNormalize` wiring (`build_venv`, `save_vecnormalize`,
  `load_vecnormalize`); the smoke test runs through it by default.
- **P2.13** — `MaskablePPO` (`train`, `build_model`), the evaluation venv with its pair
  partitioning, the callbacks in `src/rl_gym/callbacks.py`, checkpoints with their statistics
  and run metadata, resume with the used-`base_seed` guard, and `--verify-run`.
- **P2.16** — **one statistics file per checkpoint** (`vecnormalize_path_for`) and the pairing
  check that enforces it (`check_statistics_pairing`). Through P2.15 every checkpoint in a
  folder shared one `vecnormalize.pkl` and overwrote it, so only the newest kept the statistics
  it was trained under; every earlier one silently loaded a later point's, which is §5.3's named
  failure. Old folders are never written to and never fall back -- the loader raises and names
  which checkpoint the shared file belongs to.

Two seeding rules that are easy to undo by accident:

- **`base_seed` is required** and never defaulted (D7): without it the env does not reseed per
  episode, so every episode replays one sample path with every log looking healthy.
- **`seed=` is never passed to `MaskablePPO`** -- see `parent_seed`. It would hand worker `i`
  the seed `base_seed + i` and bypass D7's `SeedSequence([base_seed, env_id])` silently.
"""

# ---------------------------------------------------------------------------------------- #
# IMPORT ORDER IS LOAD-BEARING. DO NOT ALPHABETISE. DO NOT LET isort/ruff REORDER THIS BLOCK.
#
# numpy MUST be imported before torch / stable_baselines3 / sb3_contrib. torch is pip-installed
# while numpy and scipy come from conda-forge, so two copies of libomp are linked and whichever
# loads second kills the process outright:
#
#     OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
#     Abort trap: 6
#
# This is a temporary workaround for the fleetpy_rl environment, not a design choice. The real
# fix is installing torch from conda-forge so one OpenMP runtime is linked; once that is done
# and the P1.10 gate has been re-verified, this ordering constraint can go. Do NOT "fix" it with
# KMP_DUPLICATE_LIB_OK=TRUE -- OpenMP warns that flag may silently produce incorrect results,
# which would invalidate baseline_user_stats.csv and every byte-for-byte gate behind it.
# See docs/SDPDP_GYM_SPEC_v2.md §8a.
import numpy as np  # noqa: I001  -- must stay first
# ---------------------------------------------------------------------------------------- #

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# the held-out split (P2.17), from the one module that defines it. `corpus` imports nothing
# heavier than csv/hashlib/json, so this costs no torch and respects §8a's ordering.
from src.rl_gym.corpus import (  # noqa: E402
    TEST_DAY_DIR,
    VAL_DAY_DIR,
    day_dir_for,
)

SCS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "studies", "example_study", "scenarios")

#: `n_workers` is deliberately **absent** here. It is passed into `make_env` from the same value
#: that sizes the SubprocVecEnv, so the two cannot disagree: D10 gives worker `env_id` the days
#: at positions `env_id, env_id + n_workers, ...` of each pass, so a config value of 4 run with
#: `--workers 3` would silently drop a quarter of every pass -- every episode still succeeding,
#: every log healthy, a quarter of the training days never visited.
DEFAULT_CFG = {
    "constant_cfg_path": os.path.join(SCS, "constant_config_depot_cali_sc_1.csv"),
    "scenario_cfg_path": os.path.join(SCS, "example_depot_cali_sc_1.csv"),
    "K": 8,
    "skip_output": True,
    "base_seed": 20260913,
    "day_dir": "corpus",          # training days; D10 rotation (P2.3)
    #: §6.5 and §5.3: PPO's discount, and **the same value** VecNormalize must use. SB3's
    #: VecNormalize defaults to 0.99, which would normalise rewards against an effective
    #: horizon of ~100 decisions while PPO discounts over ~1000. One config key feeds both
    #: (P2.13 hands this to MaskablePPO).
    "gamma": 0.999,
}

#: **Legacy** name: through P2.15 every checkpoint in a folder shared this one file and
#: overwrote it (P2.16). Kept only so the loader can recognise an old run folder and say which
#: checkpoint the file belongs to. Nothing writes it any more.
VECNORMALIZE_FILE = "vecnormalize.pkl"

#: P2.16: the statistics live beside their checkpoint, named after it --
#: `model_ep00047.zip` -> `model_ep00047_vecnormalize.pkl`.
VECNORMALIZE_SUFFIX = "_vecnormalize.pkl"

#: `RunningMeanStd.__init__` sets `count = epsilon`, and SB3's default epsilon is 1e-4. This is
#: **not** `VecNormalize(epsilon=1e-8)`, which is the normalisation floor -- a different number.
STATS_COUNT_EPSILON = 1e-4

#: Tolerance for the pairing check, and it is float safety only, not semantic slack.
#: `ret_rms.count - STATS_COUNT_EPSILON == num_timesteps` holds exactly **in arithmetic**:
#: `_update_reward` is gated on `training` alone and adds one row per env per step iteration,
#: `reset()` never touches `ret_rms`, and `num_timesteps` advances by `n_envs` on the same
#: iteration. **In floating point it does not, and the tolerance is not decorative**: a
#: *simulated* accumulation showed zero drift at 22,528, 3.2M and 8M timesteps, but P2.16's gate
#: run's own `model_ep00024` came out **1.8e-12** off (`ret_rms.count 11900.000100000001`
#: against 11,900 timesteps), which a strict equality would have rejected on a correctly paired
#: checkpoint. The smallest *real* mismatch is one rollout, orders of magnitude above this.
STATS_PAIRING_TOL = 1e-6

#: §5.3's settings. `clip_obs = 10.0` is what bounds the unbounded Box the env declares;
#: `clip_reward` stays at SB3's default.
CLIP_OBS = 10.0


#: §8's P2.13 defaults. Every one a config key; `gamma` is shared with `VecNormalize` (§5.3).
DEFAULT_HYPERPARAMS = {
    "gamma": 0.999,                # §6.5: a pickup lands 10-30 decisions after its assignment
    "gae_lambda": 0.95,
    "learning_rate": 3e-4,
    "n_steps": 1024,               # per env
    "batch_size": 256,
    "n_epochs": 10,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "net_arch": [256, 256],
}

#: §6.1's eight. **All must be given explicitly for training** -- `check_config` raises rather
#: than letting `DEFAULT_REWARD_WEIGHTS` apply, because a weight silently defaulting is a
#: different experiment than the one being recorded.
REQUIRED_REWARD_WEIGHTS = ("w_pickup", "w_wait", "w_reject", "w_decline", "w_cancel",
                           "w_noshow", "w_horizon", "w_no_candidates")

#: §6.1's Phase 2 values, for the verification run and as the documented starting point. Not a
#: default: the config must state them.
PHASE2_REWARD_WEIGHTS = {
    "w_pickup": 1.0, "w_wait": 0.05, "w_reject": 1.0, "w_decline": 0.1,
    "w_cancel": 0.6, "w_noshow": 0.0, "w_horizon": 0.0, "w_no_candidates": 0.0,
}

#: tags the parent-process seed stream, so it cannot coincide with D7's `[base_seed, env_id]`
#: or D10's `[base_seed, 1010, p + 1]`
PARENT_SEED_TAG = 2020


def make_env(cfg, env_id, n_workers, overrides=None):
    """Return a zero-argument callable that builds one env **inside the worker process**.

    The import and the construction both happen in `_init`, never in the parent: FleetPy
    objects hold a routing engine and open file handles, so an env built here and pickled
    across the process boundary would fail or, worse, share state.

    `n_workers` is a parameter rather than a config key on purpose -- see `DEFAULT_CFG`. Pass
    the same number used to size the `SubprocVecEnv`, which is what the only caller does.

    :param overrides: per-worker config, which is how each evaluation worker gets **its own**
        slice of the (day, seed) pairs (P2.13).
    """
    if not 0 <= env_id < n_workers:
        raise ValueError(f"env_id {env_id} out of range for n_workers {n_workers}")

    def _init():
        from src.rl_gym.gym_env import SDPDPAssignmentEnv
        return SDPDPAssignmentEnv({**cfg, "env_id": env_id, "n_workers": n_workers,
                                   **(overrides or {})})
    return _init


def prepare_multiprocessing():
    """Make the forkserver preload numpy, so every worker inherits §8a's import order.

    `SubprocVecEnv` prefers the forkserver start method, and `multiprocessing` preloads
    `['__main__']` into it by default. For a script that is what carries the numpy-first order
    into the workers; for a **test module** it carries nothing useful, because under
    `python -m unittest …` `__main__` is unittest's own -- and the workers then die with
    `OMP: Error #15 … Abort trap: 6` (measured in P2.12).

    Called from `build_venv`, so the training venv, the evaluation venv and P2.14's harness all
    get it without any caller remembering to. Idempotent, and it only affects forkservers not
    yet started -- the second and later venvs reuse the one already running, which by then has
    numpy loaded.
    """
    mp.set_forkserver_preload(["numpy"])


def build_venv(cfg, n_workers, normalize=True, training=True):
    """The `SubprocVecEnv`, optionally wrapped in `VecNormalize` per §5.3 (P2.12).

    `SubprocVecEnv` and never `DummyVecEnv` (D8): `FleetSimulationBase.__init__` reconfigures
    process-global logging and seeds the global RNG, so one env per OS process.

    **`gamma` is taken from the config, not left at SB3's default.** `VecNormalize` scales
    rewards by a running estimate of the discounted return, and its default of 0.99 would
    normalise against an effective horizon of ~100 decisions while PPO discounts over ~1000
    (§5.3, §6.5). The same `cfg["gamma"]` goes to PPO in P2.13, so the two cannot drift.

    `clip_obs = 10.0` is what bounds the deliberately unbounded `Box` the env declares (§5.1):
    `delta_cfv` has no a priori bound even after `sign·log1p`, so the clipping belongs here
    rather than in a space that would either lie or clip real values.

    :param normalize: wrap in `VecNormalize`. False gives the bare venv, which is what P1.11's
        smoke test measured and what a mask-pass-through comparison needs as its reference.
    :param training: `VecNormalize`'s own flag. **False for evaluation**, together with
        `norm_reward = False` -- see `load_vecnormalize`.
    :return: the venv, wrapped or not
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    prepare_multiprocessing()
    # n_workers reaches the envs from this one value, the same one sizing the venv (D10)
    venv = SubprocVecEnv([make_env(cfg, i, n_workers) for i in range(n_workers)])
    if not normalize:
        return venv
    return VecNormalize(
        venv,
        training=training,
        norm_obs=True,
        norm_reward=training,
        clip_obs=CLIP_OBS,
        gamma=float(cfg["gamma"]),
    )


def vecnormalize_path_for(checkpoint_path):
    """The statistics file belonging to `checkpoint_path`. **The only definition of this map.**

    `model_ep00047.zip` -> `model_ep00047_vecnormalize.pkl`, beside it. Every writer and reader
    goes through this function; two agreeing derivations of a path are what drift (trap 13),
    and a shared name is the P2.16 defect itself.

    **With or without the `.zip` gives the same file**, because `MaskablePPO.load` accepts both
    forms and a caller that drops the extension must not get different statistics. Only a `.zip`
    suffix is stripped, so a checkpoint named `model.v2` keeps its dot.
    """
    directory = os.path.dirname(os.path.abspath(checkpoint_path))
    stem = os.path.basename(str(checkpoint_path))
    if stem.endswith(".zip"):
        stem = stem[: -len(".zip")]
    return os.path.join(directory, f"{stem}{VECNORMALIZE_SUFFIX}")


def checkpoint_timesteps(checkpoint_path):
    """`num_timesteps` out of a checkpoint zip's `data` entry, or `None` if there is no zip.

    Read with `zipfile` + `json`, so it costs no policy construction and no torch import.

    `None` means the pairing check cannot run: a statistics file saved beside a path with no
    checkpoint (as several unit tests do) has nothing to be paired against. Every real caller
    loads the model from the same path moments later, so an absent zip fails at once rather
    than silently evaluating an unpaired policy.
    """
    import zipfile
    path = str(checkpoint_path)
    if not path.endswith(".zip"):
        path += ".zip"
    if not os.path.isfile(path):
        return None
    with zipfile.ZipFile(path) as archive:
        if "data" not in archive.namelist():
            raise ValueError(
                f"{path} has no `data` entry, so it is not a Stable-Baselines3 checkpoint and "
                f"its statistics cannot be paired with it (§5.3)."
            )
        data = json.loads(archive.read("data").decode("utf-8"))
    if "num_timesteps" not in data:
        raise ValueError(f"{path}'s `data` entry carries no num_timesteps.")
    return int(data["num_timesteps"])


def check_statistics_pairing(stats, checkpoint_path, stats_file=None):
    """Raise unless `stats` are the statistics `checkpoint_path` was trained under (P2.16).

    The test is `ret_rms.count - 1e-4 == num_timesteps` within `STATS_PAIRING_TOL`. The relation
    is exact in arithmetic; real artifacts drift by about 1e-12 (P2.16 measured 1.8e-12 at
    11,900 timesteps), which is why it is a tolerance and not an equality.
    `obs_rms.count` is **not** tested -- it carries an extra `n_envs` per `learn()` call, from
    `reset()`, and that count is not recoverable from the checkpoint -- but it is reported,
    because it is useful when diagnosing a mismatch.

    Holds across `--resume-from` too: the resumed model keeps counting from `T1`
    (`reset_num_timesteps=False`) and the loaded statistics keep accumulating from `1e-4 + T1`,
    so both sides advance together.

    :param stats: anything carrying `ret_rms` / `obs_rms` -- a `VecNormalize`, attached or not
    """
    timesteps = checkpoint_timesteps(checkpoint_path)
    if timesteps is None:
        return None
    ret_count = float(stats.ret_rms.count)
    obs_count = float(stats.obs_rms.count)
    implied = ret_count - STATS_COUNT_EPSILON
    if abs(implied - timesteps) <= STATS_PAIRING_TOL:
        return timesteps
    where = stats_file or "the statistics"
    if ret_count <= STATS_COUNT_EPSILON + STATS_PAIRING_TOL:
        raise ValueError(
            f"{where} were never updated: ret_rms.count is {ret_count!r}, the value a freshly "
            f"constructed VecNormalize starts at (count 1e-4, mean 0, var 1), while "
            f"{os.path.basename(str(checkpoint_path))} was trained for {timesteps} timesteps. "
            f"A policy fed observations normalised by unused statistics sees every feature on a "
            f"scale it never trained under -- P2.12 measured a raw is_valid of 1.0 arriving as "
            f"about 0.005. Load the statistics saved with this checkpoint (§5.3)."
        )
    raise ValueError(
        f"{where} do not belong to {os.path.basename(str(checkpoint_path))}: they imply "
        f"{implied:.0f} timesteps (ret_rms.count {ret_count!r} - 1e-4) but that checkpoint was "
        f"trained for {timesteps}. Evaluating or resuming a policy against another point's "
        f"statistics silently shifts every observation it sees (§5.3). "
        f"[diagnostic: obs_rms.count {obs_count!r}, which also carries n_envs per learn() call]"
    )


def verify_statistics_pairing(checkpoint_path, stats_path=None):
    """`check_statistics_pairing` without building anything -- the pre-flight form (P2.16).

    Reads the pickle directly, so a caller can fail **before** constructing a `SubprocVecEnv`
    whose workers would otherwise be orphaned by the raise. `load_vecnormalize` checks again
    after attaching, which is what covers the callers that do not come through here.

    :return: the statistics file that was checked
    """
    import pickle
    path = stats_path or vecnormalize_path_for(checkpoint_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(_missing_statistics_message(checkpoint_path, path))
    with open(path, "rb") as handle:
        stats = pickle.load(handle)
    check_statistics_pairing(stats, checkpoint_path, stats_file=path)
    return path


def _missing_statistics_message(checkpoint_path, path):
    """Why the statistics are missing, and -- for an old run folder -- what is actually there.

    P2.16 does not migrate and does not fall back: a silent fallback to the shared file is the
    mismatch it exists to remove. The message therefore names the shared file, says which
    checkpoint it belongs to, and says plainly that the other checkpoints' statistics are gone.
    """
    directory = os.path.dirname(os.path.abspath(checkpoint_path))
    legacy = os.path.join(directory, VECNORMALIZE_FILE)
    message = (
        f"no statistics file {os.path.basename(path)} beside "
        f"{os.path.basename(str(checkpoint_path))}. A checkpoint without the observation "
        f"statistics it was trained under cannot be evaluated faithfully (§5.3); "
        f"save_vecnormalize writes one per checkpoint at checkpoint time."
    )
    if not os.path.isfile(legacy):
        return message
    owner = "unknown"
    try:
        from src.rl_gym.callbacks import load_run_metadata
        checkpoints = load_run_metadata(directory).get("checkpoints") or []
        if checkpoints:
            owner = checkpoints[-1]
    except Exception:                                      # pragma: no cover - best effort
        pass
    return (
        f"{message}\n"
        f"This folder predates P2.16: it holds a single shared {VECNORMALIZE_FILE}, which every "
        f"checkpoint overwrote, so it holds the statistics of the LAST checkpoint only "
        f"({owner}). The earlier checkpoints' statistics no longer exist and those checkpoints "
        f"cannot be evaluated faithfully -- re-run to get them. Nothing here falls back to the "
        f"shared file, because pairing it with the wrong checkpoint is the failure this check "
        f"exists to stop. To use it deliberately for the checkpoint it does belong to, pass it "
        f"explicitly: --stats {legacy} (the pairing check still runs, and will refuse it for "
        f"any other checkpoint)."
    )


#: The fields of `observation_contract`, in the order a refusal reports them, with a short
#: phrase saying what a difference in each would silently do to a loaded policy. Driving both
#: the comparison and the message off one table is what keeps a newly added field from being
#: recorded and then never checked (P2.18).
OBSERVATION_CONTRACT_FIELDS = (
    ("baseline_hash", "the arrival baseline was rebuilt, so arrivals_expected_30/60 and "
                      "arrivals_z_30/60 mean something different"),
    ("obs_layout_fingerprint", "the observation's feature names, widths or order changed"),
    ("observation_length", "the observation is a different length"),
    ("position_encoding", "veh_lat/veh_lon switched between absolute and request-relative"),
    ("wait_normalizer_s", "every wait-type feature is divided by a different number"),
)


def contract_env_cfg(cfg):
    """The subset of a training/eval config that determines the observation contract.

    Only the keys that reach `feature_spec()` or a feature's scaling, so the pre-flight env can
    be built without `day`, `day_dir`, `episode_seed` or `eval_pairs` — P2.3 raises when the
    seeding keys meet each other, and none of them changes what a feature means.
    """
    keys = ("constant_cfg_path", "scenario_cfg_path", "scenario_row", "K", "corpus_dir",
            "wait_normalizer_s", "position_encoding")
    return {k: cfg[k] for k in keys if k in cfg}


def current_observation_contract(cfg):
    """This code and these artifacts, as an `observation_contract` dict.

    **Costs no simulation and no worker.** `SDPDPAssignmentEnv.__init__` builds the observers,
    reads the manifest and the baseline and stops — `load_simulation_environment` is in
    `reset()` — so this is about 70 ms with no routing engine loaded. That is what lets the
    guard run as a pre-flight, before any `SubprocVecEnv` exists, which is P2.16's ordering
    lesson: a refusal must orphan no workers.

    Built by constructing the real env rather than by re-deriving the fields, so the values
    compared are the ones the workers will actually observe under (trap 13).
    """
    from src.rl_gym.gym_env import SDPDPAssignmentEnv
    env = SDPDPAssignmentEnv(contract_env_cfg(cfg))
    try:
        return dict(env.observation_contract)
    finally:
        env.close()


def verify_artifact_contract(checkpoint_path, cfg, recorded=None):
    """Raise unless `checkpoint_path` was trained under this observation contract (P2.18).

    The failure this exists for: P2.18 kept 139 features and changed what 4 of them mean, so an
    older checkpoint loads, runs, and reads `arrivals_expected_30/60` and `arrivals_z_30/60` on
    a scale it never trained under, with every log looking healthy — the same class of silent
    mismatch as P2.16's shared statistics file, on the same path that produces the KPI table.
    The env's own `verify_corpus` / `verify_baseline` cannot catch it: they check that the two
    artifacts agree with **each other**, never that they are the pair a policy was trained
    against.

    **There is no escape hatch and no `--force`**, the same rule as P2.14's `write_seed_file`
    and P2.16's refusal to fall back to a shared statistics file. Everything written before
    P2.18 therefore stops being evaluable and resumable — `runs/p215/`, `runs/p215b/` and
    `runs/p216_gate/` included, by decision.

    :param recorded: the recorded contract, for a caller that already has the metadata; read
        from `run_metadata.json` beside the checkpoint otherwise
    :return: the contract that was verified
    """
    from src.rl_gym.callbacks import load_run_metadata
    path = str(checkpoint_path)
    if recorded is None:
        metadata = load_run_metadata(path)
        if not metadata:
            raise ValueError(
                f"no run_metadata.json beside {os.path.basename(path)}, so there is no record "
                f"of the observation contract it was trained under (P2.18). Nothing can say "
                f"whether its features still mean what they meant; evaluating it would be a "
                f"guess. Checkpoint: {path}"
            )
        recorded = metadata.get("observation_contract")
    if not recorded:
        raise ValueError(
            f"{os.path.basename(path)} carries no 'observation_contract' in its "
            f"run_metadata.json, so it predates P2.18 — which **re-defined four features "
            f"without changing any name or width**: the arrival baseline was conditioned on "
            f"month and is now conditioned on weekday, so arrivals_expected_30/60 and "
            f"arrivals_z_30/60 are different numbers at the same positions in a 139-long "
            f"vector. A policy trained before it would read them as if nothing had changed, "
            f"and no log, KPI or TensorBoard scalar would show it. There is no migration and no "
            f"override: re-train against the current baseline. Checkpoint: {path}"
        )

    current = current_observation_contract(cfg)
    differences = []
    for field, consequence in OBSERVATION_CONTRACT_FIELDS:
        was, now = recorded.get(field), current.get(field)
        if was != now:
            differences.append(f"  {field}: recorded {was!r}, current {now!r} — {consequence}")
    if not differences:
        return current
    raise ValueError(
        f"{os.path.basename(path)} was trained under a different observation contract "
        f"(P2.18):\n" + "\n".join(differences) + f"\nA policy reads the observation by "
        f"position, so a feature that changed meaning is read as if it had not, and nothing "
        f"downstream shows it (§5.3's silent failure, one level up). Re-train against the "
        f"current artifacts, or restore the ones this checkpoint was trained with and rebuild "
        f"the pair with `python -m src.rl_gym.offline.build_arrival_baseline`. "
        f"Checkpoint: {path}"
    )


def save_vecnormalize(venv, checkpoint_path):
    """Write the running statistics beside `checkpoint_path`, returning the file written.

    Saved with **every** checkpoint, because a policy and the statistics it was trained under
    are one artifact: evaluating a checkpoint against statistics from a different point in
    training silently shifts every observation the policy sees. §5.3 calls this out as a common
    and silent failure, which is why the two are written together and loaded together.

    **One file per checkpoint** (P2.16), named by `vecnormalize_path_for`. Until P2.16 this
    wrote a fixed `vecnormalize.pkl` per folder, so each checkpoint destroyed the previous
    statistics and only the newest was faithfully evaluable.
    """
    from stable_baselines3.common.vec_env import VecNormalize
    if not isinstance(venv, VecNormalize):
        raise TypeError(
            f"save_vecnormalize needs a VecNormalize, got {type(venv).__name__}. Build the "
            f"venv with build_venv(..., normalize=True)."
        )
    path = vecnormalize_path_for(checkpoint_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    venv.save(path)
    return path


def load_vecnormalize(checkpoint_path, venv, for_evaluation=True, stats_path=None):
    """Attach `checkpoint_path`'s own statistics to `venv`, refusing any others (P2.16).

    With `for_evaluation` (the default) this sets **`training = False`** so the statistics stop
    updating, and **`norm_reward = False`** so the reported return is in the reward's own units
    rather than normalised ones. §5.3 requires both: evaluating with mismatched or still-moving
    statistics is silent and very common.

    The file is `vecnormalize_path_for(checkpoint_path)` unless `stats_path` names one. Either
    way the pairing check runs, so an explicitly named file -- an old run folder's shared
    `vecnormalize.pkl`, say -- is accepted for the checkpoint it belongs to and **refused for
    every other**. There is no fallback to the shared file: reintroducing that silently is the
    defect P2.16 removes.

    :param venv: an unwrapped venv to attach to -- `VecNormalize.load` wraps it
    :param stats_path: an explicit statistics file, still pairing-checked
    :return: the wrapped venv
    """
    from stable_baselines3.common.vec_env import VecNormalize
    path = stats_path or vecnormalize_path_for(checkpoint_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(_missing_statistics_message(checkpoint_path, path))
    wrapped = VecNormalize.load(path, venv)
    check_statistics_pairing(wrapped, checkpoint_path, stats_file=path)
    if for_evaluation:
        wrapped.training = False
        wrapped.norm_reward = False
    return wrapped


def check_checkpoint_statistics(checkpoint_dir):
    """Every `.zip` in the folder has its own statistics, and each passes the pairing check.

    The check `--verify-run` and `phase2_run.check_training` both run (P2.16), replacing a
    folder-level `os.path.isfile(ckpt_dir/vecnormalize.pkl)` that passed with N checkpoints and
    one shared file -- and would have passed with one checkpoint's statistics and N-1 missing.

    :return: `(ok, detail)` for a pass/fail line
    """
    models = sorted(f for f in os.listdir(checkpoint_dir) if f.endswith(".zip"))
    if not models:
        return False, f"no checkpoints in {checkpoint_dir}"
    problems = []
    for name in models:
        checkpoint = os.path.join(checkpoint_dir, name)
        stats = vecnormalize_path_for(checkpoint)
        if not os.path.isfile(stats):
            problems.append(f"{name}: no {os.path.basename(stats)}")
            continue
        try:
            verify_statistics_pairing(checkpoint)
        except (ValueError, FileNotFoundError) as exc:
            problems.append(f"{name}: {str(exc).splitlines()[0]}")
    if problems:
        return False, "; ".join(problems)
    return True, f"{len(models)} checkpoint(s), each with its own statistics, each paired"


def check_eval_pairs_split(eval_pairs, eval_day_dir=VAL_DAY_DIR):
    """Raise unless every training-time evaluation day is a **validation** day (P2.17).

    The callback evaluates during training, and whatever it reports is what a run is judged and
    checkpointed on. Pointing it at the 14 test days would turn them into a selection criterion
    -- the one use holding them back is meant to prevent -- and nothing downstream would show
    it: the KPI table, the TensorBoard scalars and the checkpoints would all look normal.

    **Called before `build_venv`**, so a refusal costs no simulation and orphans no workers.
    Inside `build_eval_venv` it would already be too late: the training `SubprocVecEnv` is up by
    then, and P2.16 made exactly this ordering mistake unrepeatable for the statistics check.

    `eval_day_dir` is accepted so the caller's directory and the pairs are checked against each
    other rather than independently; a non-default directory is only ever the test half, which
    this refuses by name.
    """
    if not eval_pairs:
        return
    if eval_day_dir != VAL_DAY_DIR:
        raise ValueError(
            f"a training-time evaluation callback must read {VAL_DAY_DIR}/, not "
            f"{eval_day_dir!r}. The {TEST_DAY_DIR}/ days are the final-results half (P2.17): "
            f"evaluating on them during training makes them a selection criterion, which is "
            f"what holding them back exists to prevent."
        )
    wrong = []
    for day, _seed in eval_pairs:
        try:
            day_dir_for(day, "val")
        except ValueError as exc:
            wrong.append(f"{day}: {str(exc).splitlines()[0]}")
    if wrong:
        raise ValueError(
            f"{len(wrong)} training-time evaluation day(s) are not validation days (P2.17): "
            + "; ".join(wrong)
            + f". The callback reads {VAL_DAY_DIR}/, and every choice a run makes -- which "
              f"checkpoint, which weights -- has to be made on that half alone."
        )


def build_eval_venv(cfg, pairs, n_workers, eval_day_dir=VAL_DAY_DIR):
    """The evaluation venv: one worker per slice of the (day, seed) pairs (§8 P2.13).

    **`corpus_val`, not `corpus_eval`** (P2.17). This default is the training side's guard: it
    is `train()`'s only control over the directory the callback reads, and a training-time
    callback on the test half would make those 14 days a selection criterion -- which is the one
    thing holding them back is for. `train()` additionally refuses any `eval_pairs` day that is
    not a validation day, before either venv is built, so the default and the pairs cannot
    disagree.

    Each worker gets **its own** slice, because `evaluate_policy` gives worker `i` a target of
    `(n_eval_episodes + i) // n_envs` episodes and each env cycles through the pairs it was
    handed (P2.3) -- see `callbacks.partition_eval_pairs`, which also explains why a shifted
    start between evaluations is harmless.

    Wrapped with **`training=False`**, which is what makes `norm_reward=False` too (P2.12).
    `MaskableEvalCallback` syncs `obs_rms` and `ret_rms` from the training env at every
    evaluation but does **not** set those flags, so they have to be right from here (§5.3).

    The training seeding keys are stripped: `eval_pairs` fixes both the day and the seed of
    every episode, and P2.3 raises if it is combined with `base_seed`, `day` or `episode_seed`.
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from src.rl_gym.callbacks import partition_eval_pairs

    slices = partition_eval_pairs(pairs, n_workers)
    base = {k: v for k, v in cfg.items()
            if k not in ("base_seed", "day", "episode_seed", "eval_pairs", "day_dir")}
    base["day_dir"] = eval_day_dir
    prepare_multiprocessing()
    venv = SubprocVecEnv([
        make_env(base, i, n_workers, overrides={"eval_pairs": slices[i]})
        for i in range(n_workers)
    ])
    # observations normalised with the training statistics, reward reported in its own units
    return VecNormalize(venv, training=False, norm_obs=True, norm_reward=False,
                        clip_obs=CLIP_OBS, gamma=float(cfg["gamma"])), slices


def parent_seed(base_seed):
    """The seed for the parent process's own generators (§8 P2.13).

    **Not** passed to `MaskablePPO` as `seed=`: `BaseAlgorithm.set_random_seed` ends with
    `self.env.seed(seed)`, and `VecEnv.seed` hands worker `i` the seed `base_seed + i` at its
    next reset (verified in P2.13 against SB3 2.9.0). The env would then seed its generator
    from that number and never use D7's `SeedSequence([base_seed, env_id])` -- silently
    bypassing the per-episode reseeding, with every log looking healthy.

    So the parent seeds *itself* from a tagged stream, which makes network initialisation and
    action sampling reproducible without touching the workers' seeds.
    """
    ss = np.random.SeedSequence([int(base_seed), PARENT_SEED_TAG])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def check_config(cfg, training=True):
    """`base_seed` and all eight reward weights must be explicit for any real run.

    With `base_seed` absent the env deliberately does not reseed (D7), so every episode would
    replay one sample path with nothing erroring and the logs looking healthy. With a reward
    weight absent, `DEFAULT_REWARD_WEIGHTS` would quietly apply and the run would be a
    different experiment from the one recorded. Raise rather than default, in both cases.
    """
    if cfg.get("base_seed") is None:
        raise ValueError(
            "base_seed is required for training: without it SDPDPAssignmentEnv does not "
            "reseed per episode (D7), so every episode replays one sample path silently. "
            "Set it explicitly."
        )
    if not training:
        return
    weights = cfg.get("reward_weights")
    if not weights:
        raise ValueError(
            f"reward_weights is required for training: all eight of {REQUIRED_REWARD_WEIGHTS} "
            f"must be given explicitly (§6.1). Falling back to DEFAULT_REWARD_WEIGHTS would "
            f"make the run a different experiment from the one recorded."
        )
    missing = [w for w in REQUIRED_REWARD_WEIGHTS if w not in weights]
    if missing:
        raise ValueError(
            f"reward_weights is missing {missing}; all eight of {REQUIRED_REWARD_WEIGHTS} "
            f"must be given explicitly (§6.1, §8 P2.13)."
        )
    extra = [w for w in weights if w not in REQUIRED_REWARD_WEIGHTS]
    if extra:
        raise ValueError(
            f"reward_weights has unknown key(s) {extra}; §10 forbids reward terms beyond "
            f"§6.1's eight, and a typo'd weight would otherwise be silently ignored."
        )


def build_model(venv, cfg, tensorboard_log=None, resume_from=None):
    """`MaskablePPO` with `MlpPolicy`, hyperparameters from the config (§8 P2.13).

    **No `seed=`** -- see `parent_seed` for why that would bypass D7. The parent's own
    generators are seeded by the caller before this is called, so network initialisation is
    reproducible.
    """
    from sb3_contrib import MaskablePPO

    hp = {k: cfg.get(k, v) for k, v in DEFAULT_HYPERPARAMS.items()}
    if resume_from is not None:
        model = MaskablePPO.load(resume_from, env=venv,
                                 tensorboard_log=tensorboard_log,
                                 custom_objects={"learning_rate": hp["learning_rate"],
                                                 "clip_range": hp["clip_range"]})
        return model, hp
    model = MaskablePPO(
        "MlpPolicy",
        venv,
        gamma=hp["gamma"],
        gae_lambda=hp["gae_lambda"],
        learning_rate=hp["learning_rate"],
        n_steps=hp["n_steps"],
        batch_size=hp["batch_size"],
        n_epochs=hp["n_epochs"],
        clip_range=hp["clip_range"],
        ent_coef=hp["ent_coef"],
        policy_kwargs={"net_arch": list(hp["net_arch"])},
        tensorboard_log=tensorboard_log,
        verbose=1,
    )
    return model, hp


def train(cfg, n_workers=4, total_timesteps=10_000, out_dir="runs/sdpdp",
          eval_pairs=None, n_eval_workers=2, eval_freq=2048,
          checkpoint_every_n_episodes=4, resume_from=None, verbose=True,
          extra_callbacks=None):
    """One training run: venvs, model, callbacks, `learn`, final checkpoint (§8 P2.13).

    Returns the artifacts the verification reads -- the callbacks, the paths and the
    hyperparameters actually used -- rather than only the model, so a check can look at what
    happened instead of re-deriving it.

    :param extra_callbacks: further `BaseCallback`s to run alongside the three this builds
        (added in P2.15). A caller that needs to *observe* the rollout -- P2.15's per-step
        audit that no masked action is ever selected reads `action_masks` and `actions` out of
        `collect_rollouts`' locals -- would otherwise have to edit the training script it is
        checking. Nothing here depends on them, and the default changes no behaviour.
    """
    from stable_baselines3.common.utils import set_random_seed
    from src.rl_gym import callbacks as cb

    check_config(cfg)
    check_eval_pairs_split(eval_pairs)
    os.makedirs(out_dir, exist_ok=True)
    tb_dir = os.path.join(out_dir, "tb")

    metadata = {"base_seeds": [int(cfg["base_seed"])]}
    if resume_from is not None:
        previous = cb.load_run_metadata(resume_from)
        # raises if this base_seed was already used: it would restart D10's rotation and the
        # episode-seed streams from pass 0 and replay sample paths already trained on
        cb.check_resume_base_seed(previous, cfg["base_seed"])
        # and raises unless the checkpoint's features still mean what they meant (P2.18).
        # **Before build_venv**, like the two checks around it: a refusal here costs no
        # simulation and orphans no workers.
        verify_artifact_contract(resume_from, cfg, recorded=previous.get("observation_contract"))
        metadata["base_seeds"] = [int(s) for s in previous.get("base_seeds", [])] + \
                                 [int(cfg["base_seed"])]
        metadata["resumed_from"] = os.path.basename(resume_from)

    # the parent's own streams, tagged so they cannot coincide with D7's or D10's
    set_random_seed(parent_seed(cfg["base_seed"]))

    venv = build_venv(cfg, n_workers, normalize=True)
    eval_venv, eval_slices = (None, None)
    callback_list = []
    try:
        model, hp = build_model(venv, cfg, tensorboard_log=tb_dir, resume_from=resume_from)
        if resume_from is not None:
            # The statistics the checkpoint was trained under, not fresh ones (§5.3).
            # **Raises when they are missing** (P2.16): this was `if os.path.isfile(...)`, so a
            # resume without statistics silently restarted them at count 1e-4, mean 0, var 1 --
            # the resumed policy then saw every feature on a scale it never trained under, with
            # nothing in the log to show it. `load_vecnormalize` also pairing-checks the file,
            # and `for_evaluation=False` leaves training and norm_reward on for the resumed run.
            venv = load_vecnormalize(resume_from, venv.venv, for_evaluation=False)
            model.set_env(venv)

        training_days = []
        if cfg.get("day_dir") == "corpus":
            from src.rl_gym.corpus import training_day_files
            training_days = training_day_files()
        episode_log = cb.EpisodeLogCallback(k_max=int(cfg["K"]), training_days=training_days)
        # P2.18's contract, read off **worker 0's own env** rather than re-derived in the
        # parent: these are the values the policy will actually be trained under, so there is
        # one derivation and not two that could drift (trap 13). All workers share one cfg, so
        # a disagreement would mean an env built differently per worker -- checked, not assumed.
        contracts = venv.get_attr("observation_contract")
        if any(c != contracts[0] for c in contracts[1:]):
            raise RuntimeError(
                f"the {len(contracts)} training workers report different observation contracts, "
                f"so no single record would describe the policy being trained: {contracts}"
            )
        checkpointer = cb.EpisodeCheckpointCallback(
            every_n_episodes=checkpoint_every_n_episodes,
            save_path=os.path.join(out_dir, "checkpoints"),
            save_stats_fn=save_vecnormalize,
            metadata={**metadata, "hyperparams": hp,
                      "reward_weights": dict(cfg["reward_weights"]),
                      "day_dir": cfg.get("day_dir"), "n_workers": n_workers,
                      "observation_contract": contracts[0]},
        )
        callback_list = [episode_log, checkpointer] + list(extra_callbacks or [])

        evaluator = None
        if eval_pairs:
            eval_venv, eval_slices = build_eval_venv(cfg, eval_pairs, n_eval_workers)
            evaluator = cb.PairLoggingMaskableEvalCallback(
                eval_venv,
                n_eval_episodes=len(eval_pairs),
                eval_freq=eval_freq,
                deterministic=True,
                best_model_save_path=None,
                log_path=os.path.join(out_dir, "eval"),
                verbose=1 if verbose else 0,
            )
            callback_list.append(evaluator)

        t0 = time.perf_counter()
        model.learn(total_timesteps=total_timesteps, callback=callback_list,
                    reset_num_timesteps=resume_from is None)
        wall = time.perf_counter() - t0
        final = checkpointer.save()
    finally:
        processes = venv.unwrapped.processes
        venv.close()
        if eval_venv is not None:
            eval_venv.close()

    return {
        "model_path": final,
        "out_dir": out_dir,
        "tb_dir": tb_dir,
        "hyperparams": hp,
        "episode_log": episode_log,
        "checkpointer": checkpointer,
        "evaluator": evaluator,
        "eval_slices": eval_slices,
        "wall_seconds": wall,
        "worker_exitcodes": [p.exitcode for p in processes],
        "metadata": metadata,
    }


def random_masked_actions(masks, rng):
    """One uniformly random legal action per worker."""
    return np.array([rng.choice(np.flatnonzero(m)) for m in masks], dtype=np.int64)


def smoke_test(n_workers=4, cfg=None, verbose=True, normalize=True):
    """Four workers, one episode each, random legal actions. Returns the measurements.

    `normalize` wraps the venv in `VecNormalize` (P2.12), which is how training will run, so
    this exercises the wrapper end to end rather than leaving it first used in P2.13. The masks
    are read through `get_action_masks` on the **wrapped** venv for the same reason.
    """
    # imported here, after numpy, and after the module docstring's ordering note
    import psutil
    from sb3_contrib.common.maskable.utils import get_action_masks

    cfg = dict(cfg or DEFAULT_CFG)
    check_config(cfg)

    parent = psutil.Process()
    children_before = {c.pid for c in parent.children(recursive=True)}

    t0 = time.perf_counter()
    venv = build_venv(cfg, n_workers, normalize=normalize)
    t_construct = time.perf_counter() - t0

    # Take the worker PIDs from SB3 itself, not from "every new child". multiprocessing also
    # starts a resource_tracker and, under the forkserver start method SB3 prefers, a forkserver
    # process. Both are normal infrastructure that lives as long as the parent, and counting
    # them as workers reports phantom orphans.
    processes = venv.unwrapped.processes if normalize else venv.processes
    worker_pids = {p.pid for p in processes}
    helper_pids = ({c.pid for c in parent.children(recursive=True)}
                   - children_before - worker_pids)
    rng = np.random.default_rng(cfg["base_seed"])

    peak_rss = 0.0
    steps = np.zeros(n_workers, dtype=int)
    episode_done = np.zeros(n_workers, dtype=bool)
    finished_at = [None] * n_workers
    summaries = [None] * n_workers

    t_start = time.perf_counter()
    venv.reset()
    sample = 0
    while not episode_done.all():
        masks = get_action_masks(venv)
        obs, rewards, dones, infos = venv.step(random_masked_actions(masks, rng))
        steps += ~episode_done
        for i, done in enumerate(dones):
            if done and not episode_done[i]:
                episode_done[i] = True
                finished_at[i] = time.perf_counter() - t_start
                summaries[i] = infos[i].get("episode_summary")
        sample += 1
        if sample % 20 == 0:                       # sample memory rather than every step
            rss = sum(_rss(pid) for pid in worker_pids)
            peak_rss = max(peak_rss, rss)
            if verbose:
                print(f"  ... {steps.sum()} steps, {int(episode_done.sum())}/{n_workers} done, "
                      f"workers {rss / 1e9:.2f} GB", flush=True)
    wall = time.perf_counter() - t_start
    peak_rss = max(peak_rss, sum(_rss(pid) for pid in worker_pids))

    start_method = processes[0]._start_method if processes else None
    venv.close()
    time.sleep(1.0)                                # give the OS a moment to reap
    leaked = [pid for pid in worker_pids if _alive(pid)]
    # read after close(), so these are the real exit codes rather than None
    exitcodes = [p.exitcode for p in processes]

    return {
        "n_workers": n_workers,
        "normalized": normalize,
        "start_method": start_method,
        "construct_seconds": t_construct,
        "wall_seconds": wall,
        "per_worker_steps": steps.tolist(),
        "per_worker_episode_seconds": finished_at,
        "peak_worker_rss_bytes": peak_rss,
        "worker_pids": sorted(worker_pids),
        "helper_pids": sorted(helper_pids),
        "worker_exitcodes": exitcodes,
        "orphaned_pids": leaked,
        "summaries": summaries,
    }


#: the (day, seed) pairs the verification run evaluates on. Four pairs over two evaluation
#: workers, from `docs/RL_GYM_EVAL_SEEDS.csv`'s eventual role -- P2.14 generates that file, so
#: these are fixed here and are not the real evaluation seeds.
VERIFY_EVAL_PAIRS = [("2024-01-11", 11), ("2024-03-19", 12),
                     ("2024-06-20", 13), ("2024-08-13", 14)]

#: every TensorBoard key §8's P2.13 asks to be present, checked by name
REQUIRED_SCALARS = (
    "episode/reward", "episode/steps", "episode/steps_per_second",
    "episode/pickups", "episode/rider_declined", "episode/diffusion_cancelled",
    "episode/no_show", "episode/operator_rejected", "episode/no_candidates",
    "episode/unclassified", "episode/day_index", "episode/random_seed",
    "episode/horizon_margin", "episode/tail_seconds",
    "episode/candidate_list_mean", "episode/candidate_list_max",
    "train/episodes", "time/fps", "train/loss",
)
REQUIRED_HISTOGRAMS = ("episode/candidate_list_lengths", "train/day_visits")


def _tb_tags(tb_dir):
    """Every tag in the run's event files, by kind."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tags = {"scalars": set(), "histograms": set(), "tensors": set()}
    for root, _dirs, files in os.walk(tb_dir):
        for name in files:
            if "events.out.tfevents" not in name:
                continue
            acc = EventAccumulator(os.path.join(root, name),
                                   size_guidance={"scalars": 0, "histograms": 0,
                                                  "tensors": 0})
            acc.Reload()
            found = acc.Tags()
            for kind in tags:
                tags[kind].update(found.get(kind, []))
    return tags


def verify_run(n_workers=2, episodes=8, out_dir=None, keep=False, verbose=True):
    """§8 P2.13's verification, as a committed command: `python train_sdpdp.py --verify-run`.

    Prints one `PASS`/`FAIL` line per check and exits non-zero if any failed, so it can be
    re-run independently (§0) rather than living in a scratchpad. It trains for real -- about
    8 episodes over 2 workers, two evaluations, two checkpoints -- so expect roughly 10-15
    minutes.

    Deliberately **not** a unit test: a test that trains would put minutes into every
    `python -m unittest discover tests`. The pieces it cannot check cheaply (the callbacks'
    arithmetic, the pair partitioning, the seeding) are unit-tested in
    `tests/test_rl_gym_train.py`; what needs a real run is checked here.
    """
    import shutil
    import tempfile

    from src.rl_gym import callbacks as cb
    from src.rl_gym.day_rotation import day_for_episode

    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail
                                                              else ""), flush=True)
        return bool(ok)

    base_seed = 20261011
    tmp = out_dir or tempfile.mkdtemp(prefix="sdpdp_verify_")
    cfg = dict(DEFAULT_CFG, base_seed=base_seed, skip_output=True,
               reward_weights=dict(PHASE2_REWARD_WEIGHTS), n_steps=256,
               log_level="warning")
    # about `episodes` episodes over the workers, and two evaluations inside that
    steps_per_episode = 450
    total_timesteps = episodes * steps_per_episode
    eval_freq = max(1, (total_timesteps // n_workers) // 3)

    t0 = time.perf_counter()
    if verbose:
        print(f"verification run: {n_workers} workers, ~{episodes} episodes, "
              f"base_seed {base_seed}, out {tmp}", flush=True)
        print("-- the raises, before spending any wall clock on training --", flush=True)

    # 1) the config guards
    try:
        check_config(dict(cfg, reward_weights={k: v for k, v in
                                               PHASE2_REWARD_WEIGHTS.items()
                                               if k != "w_cancel"}))
        check("a missing reward weight raises", False, "it did not")
    except ValueError as e:
        check("a missing reward weight raises", "w_cancel" in str(e))
    try:
        check_config({k: v for k, v in cfg.items() if k != "base_seed"})
        check("a missing base_seed raises", False, "it did not")
    except ValueError as e:
        check("a missing base_seed raises", "base_seed" in str(e))
    try:
        cb.check_resume_base_seed({"base_seeds": [base_seed]}, base_seed)
        check("resuming with a used base_seed raises", False, "it did not")
    except ValueError as e:
        check("resuming with a used base_seed raises", str(base_seed) in str(e))

    if verbose:
        print("-- training --", flush=True)
    r = train(cfg, n_workers=n_workers, total_timesteps=total_timesteps, out_dir=tmp,
              eval_pairs=VERIFY_EVAL_PAIRS, n_eval_workers=2, eval_freq=eval_freq,
              checkpoint_every_n_episodes=max(1, episodes // 2), verbose=verbose)
    log, evaluator = r["episode_log"], r["evaluator"]

    if verbose:
        print("-- checks --", flush=True)
    # 2) the run itself
    check("the run completed and every worker exited 0",
          all(code == 0 for code in r["worker_exitcodes"]),
          f"exit codes {r['worker_exitcodes']}")
    check("at least 4 episodes finished", log.episodes >= 4, f"{log.episodes} episodes")

    # 3) unclassified is 0 in every episode (§6.4: non-zero means a path was missed)
    bad = [s.get("unclassified") for s in log.episode_summaries
           if s.get("unclassified")]
    check("unclassified == 0 in every episode", not bad, f"non-zero in {len(bad)}")

    # 4) every TensorBoard scalar and histogram §8 asks for
    tags = _tb_tags(r["tb_dir"])
    missing = [k for k in REQUIRED_SCALARS if k not in tags["scalars"]]
    check("every required TensorBoard scalar is present", not missing,
          f"missing {missing}" if missing else f"{len(REQUIRED_SCALARS)} keys")
    missing_h = [k for k in REQUIRED_HISTOGRAMS
                 if k not in tags["histograms"] and k not in tags["tensors"]]
    check("the day-visit and candidate-length histograms are present", not missing_h,
          f"missing {missing_h}" if missing_h else "both")
    slot_keys = [f"slots/slot_{i}" for i in range(int(cfg["K"]))] + ["slots/reject"]
    missing_s = [k for k in slot_keys if k not in tags["scalars"]]
    check("slot-selection frequency is logged for every slot", not missing_s,
          f"missing {missing_s}" if missing_s else f"{len(slot_keys)} slots")

    # 5) no training day visited twice, and the days are the ones D10 assigns
    counts = log.day_visit_counts()
    repeats = {d: c for d, c in counts.items() if c > 1}
    check("no training day visited twice", not repeats, f"repeats {repeats}")
    from src.rl_gym.corpus import training_day_files
    days = training_day_files()
    expected = {}
    for env_id in range(n_workers):
        for index in range(8):
            day, _p, _pos = day_for_episode(days, base_seed, n_workers, env_id, index)
            expected.setdefault(env_id, []).append(day)
    per_env = {}
    for s in log.episode_summaries:
        per_env.setdefault(int(s.get("env_id", -1)), []).append(s.get("day"))
    mismatch = {e: (got, expected.get(e, [])[:len(got)])
                for e, got in per_env.items()
                if got != expected.get(e, [])[:len(got)]}
    check("each worker ran the days D10 assigns it, in order", not mismatch,
          f"{mismatch}" if mismatch else f"{sum(len(v) for v in per_env.values())} episodes")

    # 6) the episode seeds are D7's, not VecEnv.seed's base_seed + env_id
    seed_ok, seed_detail = True, []
    for env_id in sorted(per_env):
        first = next((s for s in log.episode_summaries
                      if int(s.get("env_id", -1)) == env_id), None)
        if first is None:
            continue
        ss = np.random.SeedSequence([base_seed, env_id])
        worker_seed = int(ss.generate_state(1, dtype=np.uint32)[0])
        # the env seeds its own Generator from the worker seed, then draws the episode seed
        want = int(np.random.default_rng(worker_seed).integers(0, 2_508_743))
        got = int(first.get("random_seed", -1))
        if got == base_seed + env_id:
            seed_ok = False
            seed_detail.append(f"env {env_id}: got base_seed + env_id ({got})")
        elif got != want:
            seed_detail.append(f"env {env_id}: {got} (derived {want})")
    check("episode seeds come from SeedSequence([base_seed, env_id]), not base_seed + env_id",
          seed_ok, "; ".join(seed_detail) if seed_detail else "D7 holds")

    # 7) the two evaluations each ran every pair exactly once
    evals = evaluator.evaluation_pairs if evaluator is not None else []
    check("at least two evaluations fired", len(evals) >= 2, f"{len(evals)} evaluation(s)")
    for i, pairs in enumerate(evals):
        want = sorted(VERIFY_EVAL_PAIRS)
        check(f"evaluation {i + 1} ran every (day, seed) exactly once",
              sorted(pairs) == want,
              f"ran {sorted(pairs)}")
    if len(evals) >= 2:
        check("the second evaluation covered the pairs despite any rotation shift",
              sorted(evals[1]) == sorted(VERIFY_EVAL_PAIRS),
              f"order {evals[1]}")

    # 8) the evaluation statistics are synced but frozen
    check("the evaluation env kept training=False and norm_reward=False",
          evaluator is None or (not evaluator.eval_env.training
                                and not evaluator.eval_env.norm_reward))

    # 9) at least two checkpoints, each with its statistics and the metadata
    ckpt_dir = os.path.join(tmp, "checkpoints")
    models = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".zip"))
    check("at least two checkpoints were written", len(models) >= 2, f"{models}")
    # P2.16: per checkpoint, not per folder. The old check was
    # `isfile(ckpt_dir/vecnormalize.pkl)` -- one boolean that passed with N checkpoints sharing
    # one file, which is exactly the defect it was meant to exclude.
    stats_ok, stats_detail = check_checkpoint_statistics(ckpt_dir)
    check("every checkpoint has its own statistics file, each paired with it",
          stats_ok, stats_detail)
    meta = cb.load_run_metadata(ckpt_dir)
    check("the metadata records every base_seed used",
          meta.get("base_seeds") == [base_seed], f"{meta.get('base_seeds')}")
    # P2.18: the observation contract, and that the guard accepts what this run recorded. The
    # unit tests drive the guard against hand-written metadata; this is the only check that the
    # **write** path and the **read** path agree on a real run's artifacts.
    recorded = meta.get("observation_contract")
    check("the metadata records the observation contract",
          bool(recorded), f"{recorded}")
    if recorded:
        current = current_observation_contract(cfg)
        check("the recorded contract is the one this code and these artifacts produce",
              recorded == current,
              f"recorded {recorded} vs current {current}")
        try:
            verify_artifact_contract(os.path.join(ckpt_dir, models[-1]), cfg)
            check("the guard accepts the checkpoints this run wrote", True,
                  f"{models[-1]}")
        except (ValueError, FileNotFoundError) as exc:
            check("the guard accepts the checkpoints this run wrote", False,
                  str(exc).splitlines()[0])
        # and refuses the same checkpoint once a recorded field moves -- the P2.18 failure,
        # exercised on a real checkpoint rather than a placeholder
        try:
            verify_artifact_contract(os.path.join(ckpt_dir, models[-1]), cfg,
                                     recorded={**recorded, "baseline_hash": "0" * 64})
            check("the guard refuses a stale baseline_hash on a real checkpoint", False,
                  "no raise")
        except ValueError as exc:
            message = str(exc)
            check("the guard refuses a stale baseline_hash, naming both values",
                  "baseline_hash" in message and "0000" in message
                  and recorded["baseline_hash"][:12] in message,
                  message.splitlines()[1] if len(message.splitlines()) > 1 else message)
    check("the metadata records the hyperparameters and the eight weights",
          set(meta.get("reward_weights", {})) == set(REQUIRED_REWARD_WEIGHTS)
          and "gamma" in meta.get("hyperparams", {}))

    # 10) a reloaded checkpoint reproduces its deterministic actions on a fixed (day, seed)
    actions_match, detail = _check_reload_determinism(r["model_path"], cfg)
    check("a reloaded checkpoint reproduces its deterministic action sequence",
          actions_match, detail)

    # 11) two runs with the same base_seed produce identical first-update losses
    same, detail = _check_loss_determinism(cfg, n_workers)
    check("two runs with the same base_seed give identical first-update losses", same, detail)

    wall = time.perf_counter() - t0
    failed = [name for name, ok, _d in results if not ok]
    if verbose:
        print(f"\n{len(results) - len(failed)}/{len(results)} checks passed in "
              f"{wall / 60:.1f} min")
        if failed:
            print("FAILED: " + "; ".join(failed))
        print(f"artifacts: {tmp}" if keep else f"removing {tmp}")
    if not keep and out_dir is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if not failed else 1


def _check_reload_determinism(model_path, cfg):
    """Reload the checkpoint and compare 15 deterministic actions on a fixed (day, seed)."""
    from sb3_contrib import MaskablePPO
    from src.rl_gym.gym_env import SDPDPAssignmentEnv

    pinned = {k: v for k, v in cfg.items() if k not in ("base_seed", "eval_pairs")}
    # 2024-06-20 is a **validation** day since P2.17, so its demand file lives in corpus_val/.
    # Resolved rather than named, so this cannot be the thing left behind by a later re-split.
    pinned.update({"day_dir": day_dir_for("2024-06-20", "val"), "day": "2024-06-20",
                   "episode_seed": 42, "env_id": 0})

    def rollout():
        from stable_baselines3.common.vec_env import DummyVecEnv
        # DummyVecEnv is forbidden for training (D8) but this is one env in one process for a
        # handful of steps, which is what VecNormalize.load needs to attach to
        venv = DummyVecEnv([lambda: SDPDPAssignmentEnv(pinned)])
        # this checkpoint's own statistics, pairing-checked (P2.16); for_evaluation sets
        # training = False and norm_reward = False
        wrapped = load_vecnormalize(model_path, venv, for_evaluation=True)
        model = MaskablePPO.load(model_path, env=wrapped)
        obs = wrapped.reset()
        taken = []
        for _ in range(15):
            masks = np.stack(wrapped.env_method("action_masks"))
            action, _state = model.predict(obs, deterministic=True, action_masks=masks)
            taken.append(int(np.asarray(action).reshape(-1)[0]))
            obs, _r, dones, _i = wrapped.step(action)
            if dones[0]:
                break
        wrapped.close()
        return taken

    first, second = rollout(), rollout()
    return first == second, f"{first} vs {second}"


def _check_loss_determinism(cfg, n_workers, n_steps=64, rollouts=3):
    """Two short runs with the same base_seed must give the same first-update loss.

    **Three rollouts, not one.** `OnPolicyAlgorithm.learn` calls `dump_logs` *before*
    `self.train()`, so the `train/*` keys recorded by an update are only written at the
    following dump -- a single-rollout run records `train/loss` and never writes it, which is
    what made the first version of this check report "no train/loss recorded".
    """
    import shutil
    import tempfile

    losses = []
    for _ in range(2):
        tmp = tempfile.mkdtemp(prefix="sdpdp_loss_")
        try:
            r = train(dict(cfg, n_steps=n_steps), n_workers=n_workers,
                      total_timesteps=rollouts * n_steps * n_workers, out_dir=tmp,
                      eval_pairs=None, checkpoint_every_n_episodes=10_000, verbose=False)
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )
            found = None
            for root, _dirs, files in os.walk(r["tb_dir"]):
                for name in files:
                    if "events.out.tfevents" not in name:
                        continue
                    acc = EventAccumulator(os.path.join(root, name))
                    acc.Reload()
                    if "train/loss" in acc.Tags().get("scalars", []):
                        found = acc.Scalars("train/loss")[0].value
            losses.append(found)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if any(v is None for v in losses):
        return False, f"no train/loss recorded ({losses})"
    return losses[0] == losses[1], f"{losses[0]!r} vs {losses[1]!r}"


def _rss(pid):
    import psutil
    try:
        return psutil.Process(pid).memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def _alive(pid):
    import psutil
    try:
        return psutil.Process(pid).is_running() and \
            psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", action="store_true",
                    help="run MaskablePPO training (P2.13) instead of the smoke test")
    ap.add_argument("--verify-run", action="store_true",
                    help="P2.13's verification: a short real training run with a pass/fail "
                         "line per check (about 10-15 minutes)")
    ap.add_argument("--timesteps", type=int, default=100_000,
                    help="--train only: total timesteps")
    ap.add_argument("--out-dir", default=os.path.join("runs", "sdpdp"))
    ap.add_argument("--resume-from", default=None,
                    help="--train only: a checkpoint to resume. Requires a base_seed the run "
                         "has not used (§8 P2.13)")
    ap.add_argument("--keep", action="store_true",
                    help="--verify-run only: keep the run directory")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=DEFAULT_CFG["base_seed"])
    ap.add_argument("--log-level", default="warning")
    ap.add_argument("--output", action="store_true",
                    help="write simulation output (skip_output=0); off by default")
    ap.add_argument("--no-vecnormalize", dest="vecnormalize", action="store_false",
                    help="run the bare venv, as P1.11 measured it; VecNormalize is on by "
                         "default because that is how training runs (P2.12)")
    ap.add_argument("--gamma", type=float, default=DEFAULT_CFG["gamma"],
                    help="discount; VecNormalize and PPO must share it (§5.3, §6.5)")
    args = ap.parse_args()

    if args.verify_run:
        return verify_run(n_workers=max(2, args.workers if args.workers <= 4 else 2),
                          out_dir=args.out_dir if args.keep else None, keep=args.keep)

    cfg = dict(DEFAULT_CFG, base_seed=args.base_seed, skip_output=not args.output,
               log_level=args.log_level, gamma=args.gamma)

    if args.train:
        cfg["reward_weights"] = dict(PHASE2_REWARD_WEIGHTS)
        print(f"training: workers {args.workers} | base_seed {cfg['base_seed']} | "
              f"timesteps {args.timesteps} | out {args.out_dir}", flush=True)
        r = train(cfg, n_workers=args.workers, total_timesteps=args.timesteps,
                  out_dir=args.out_dir, eval_pairs=VERIFY_EVAL_PAIRS,
                  resume_from=args.resume_from)
        print(f"\nfinished in {r['wall_seconds'] / 60:.1f} min | "
              f"{r['episode_log'].episodes} episodes | checkpoint {r['model_path']}")
        print(f"day visits: {r['episode_log'].day_visit_counts()}")
        print(f"tensorboard --logdir {r['tb_dir']}")
        return 0 if all(c == 0 for c in r["worker_exitcodes"]) else 1
    # mp's process default is not what SubprocVecEnv uses; it prefers forkserver where
    # available. The one SB3 actually chose is reported after the run.
    print(f"mp default start method: {mp.get_start_method()} (SB3 picks its own) | "
          f"workers: {args.workers} | base_seed: {cfg['base_seed']} | "
          f"skip_output: {cfg['skip_output']} | vecnormalize: {args.vecnormalize} "
          f"(gamma {cfg['gamma']})", flush=True)

    r = smoke_test(n_workers=args.workers, cfg=cfg, normalize=args.vecnormalize)

    print()
    print(f"construct {args.workers} workers : {r['construct_seconds']:.2f} s")
    print(f"wall clock (all episodes) : {r['wall_seconds']:.2f} s")
    print(f"per-worker steps          : {r['per_worker_steps']}")
    print(f"per-worker episode wall   : "
          f"{[f'{t:.1f}' for t in r['per_worker_episode_seconds']]}")
    print(f"peak worker RSS (total)   : {r['peak_worker_rss_bytes'] / 1e9:.2f} GB "
          f"({r['peak_worker_rss_bytes'] / 1e9 / args.workers:.2f} GB/worker)")
    print(f"VecNormalize              : {r['normalized']}")
    print(f"start method (SB3 chose)  : {r['start_method']}")
    print(f"worker exit codes         : {r['worker_exitcodes']}")
    print(f"orphaned workers after close(): {r['orphaned_pids'] or 'none'}")
    print(f"mp helper processes       : {len(r['helper_pids'])} "
          f"(resource_tracker / forkserver; live with the parent, not orphans)")
    for i, s in enumerate(r["summaries"]):
        if s:
            print(f"  worker {i}: day {s.get('day')} (pass {s.get('pass')} pos "
                  f"{s.get('pass_position')}) seed {s.get('random_seed')} "
                  f"pickups {s.get('pickups')} rejected {s.get('operator_rejected')} "
                  f"unclassified {s.get('unclassified')}")
    days = [s.get("day") for s in r["summaries"] if s]
    if len(days) != len(set(days)):
        print(f"  WARNING: a day was visited twice within one pass: {days}")
    return 0 if not r["orphaned_pids"] else 1


if __name__ == "__main__":
    # The guard is required. SubprocVecEnv uses the forkserver start method where available
    # (macOS included), and multiprocessing's forkserver preloads ['__main__'] by default --
    # it executes this file top to bottom under run_name="__mp_main__" before forking any
    # worker. Without the guard, main() would run inside the forkserver process itself.
    # That preload is also what carries the numpy-first import order above into every worker:
    # the ordering is established once in the forkserver, and the workers inherit its image.
    mp.freeze_support()
    sys.exit(main())
