"""SubprocVecEnv smoke test for SDPDPAssignmentEnv (P1.11). Skeleton only — no learning yet.

Runs four worker processes, one episode each, taking uniformly random actions from the legal
set. Proves the environment survives the multiprocessing arrangement Phase 2 training needs:
FleetPy objects hold routing engines and open file handles and are not picklable, so every env
must be constructed *inside* the worker, never built in the parent and shipped across.

    python train_sdpdp.py                   # 4 workers, one episode each, VecNormalize on
    python train_sdpdp.py --workers 2       # fewer, for a quick check
    python train_sdpdp.py --no-vecnormalize # the bare venv, as P1.11 measured it

P2.12 added the `VecNormalize` wiring — `build_venv`, `save_vecnormalize`,
`load_vecnormalize` — and the smoke test now runs through it by default, so the wrapper is
exercised here rather than first inside a training run.

**MaskablePPO goes here in P2.13**, with the callbacks, the checkpoints and the TensorBoard
scalars. What exists now is `make_env`, the venv builders, the worker plumbing and the
measurements P1.11 reports. *(An earlier version of this line said P2.6, which was never the
item that adds the algorithm; corrected in P2.12.)*
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
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

#: filename of the saved observation/reward statistics, written beside each checkpoint (§5.3).
VECNORMALIZE_FILE = "vecnormalize.pkl"

#: §5.3's settings. `clip_obs = 10.0` is what bounds the unbounded Box the env declares;
#: `clip_reward` stays at SB3's default.
CLIP_OBS = 10.0


def make_env(cfg, env_id, n_workers):
    """Return a zero-argument callable that builds one env **inside the worker process**.

    The import and the construction both happen in `_init`, never in the parent: FleetPy
    objects hold a routing engine and open file handles, so an env built here and pickled
    across the process boundary would fail or, worse, share state.

    `n_workers` is a parameter rather than a config key on purpose -- see `DEFAULT_CFG`. Pass
    the same number used to size the `SubprocVecEnv`, which is what the only caller does.
    """
    if not 0 <= env_id < n_workers:
        raise ValueError(f"env_id {env_id} out of range for n_workers {n_workers}")

    def _init():
        from src.rl_gym.gym_env import SDPDPAssignmentEnv
        return SDPDPAssignmentEnv({**cfg, "env_id": env_id, "n_workers": n_workers})
    return _init


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


def save_vecnormalize(venv, checkpoint_path):
    """Write the running statistics beside `checkpoint_path`, returning the file written.

    Saved with **every** checkpoint, because a policy and the statistics it was trained under
    are one artifact: evaluating a checkpoint against statistics from a different point in
    training silently shifts every observation the policy sees. §5.3 calls this out as a common
    and silent failure, which is why the two are written together and loaded together.
    """
    from stable_baselines3.common.vec_env import VecNormalize
    if not isinstance(venv, VecNormalize):
        raise TypeError(
            f"save_vecnormalize needs a VecNormalize, got {type(venv).__name__}. Build the "
            f"venv with build_venv(..., normalize=True)."
        )
    directory = os.path.dirname(os.path.abspath(checkpoint_path))
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, VECNORMALIZE_FILE)
    venv.save(path)
    return path


def load_vecnormalize(checkpoint_path, venv, for_evaluation=True):
    """Attach the statistics saved beside `checkpoint_path` to `venv`.

    With `for_evaluation` (the default) this sets **`training = False`** so the statistics stop
    updating, and **`norm_reward = False`** so the reported return is in the reward's own units
    rather than normalised ones. §5.3 requires both: evaluating with mismatched or still-moving
    statistics is silent and very common.

    :param venv: an unwrapped venv to attach to -- `VecNormalize.load` wraps it
    :return: the wrapped venv
    """
    from stable_baselines3.common.vec_env import VecNormalize
    path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), VECNORMALIZE_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no {VECNORMALIZE_FILE} beside {checkpoint_path}. A checkpoint without its "
            f"observation statistics cannot be evaluated faithfully (§5.3); it is written by "
            f"save_vecnormalize at checkpoint time."
        )
    wrapped = VecNormalize.load(path, venv)
    if for_evaluation:
        wrapped.training = False
        wrapped.norm_reward = False
    return wrapped


def check_config(cfg):
    """`base_seed` must be explicit for any real run.

    With `base_seed` absent the env deliberately does not reseed (D7), so every episode would
    replay one sample path with nothing erroring and the logs looking healthy. Raise rather
    than default.
    """
    if cfg.get("base_seed") is None:
        raise ValueError(
            "base_seed is required for training: without it SDPDPAssignmentEnv does not "
            "reseed per episode (D7), so every episode replays one sample path silently. "
            "Set it explicitly."
        )


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

    cfg = dict(DEFAULT_CFG, base_seed=args.base_seed, skip_output=not args.output,
               log_level=args.log_level, gamma=args.gamma)
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
