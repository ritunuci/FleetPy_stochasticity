"""Phase 2 exit gate: the short test training run and the KPI comparison (P2.15).

Proves that all of Phase 2's machinery works together, with repositioning off. **Not a
performance test and not a learning test** -- it is sized to exercise every component a few
times. It runs MaskablePPO over the training corpus (D10 rotation), an evaluation callback
three times, checkpoints, and then P2.14's harness on 5 evaluation days x the first 2 seeds of
each, for the trained checkpoint and for stock greedy on the same pairs.

    # the whole thing, ~22-26 min on an M4 MacBook; caffeinate keeps the Mac awake
    caffeinate -i python -m src.rl_gym.tools.phase2_run --out-dir runs/p215 2>&1 | tee runs/p215/console.log
    # either half on its own
    caffeinate -i python -m src.rl_gym.tools.phase2_run --out-dir runs/p215 --skip-harness
    caffeinate -i python -m src.rl_gym.tools.phase2_run --out-dir runs/p215 --harness-only \
        --checkpoint runs/p215/checkpoints/model_ep00047.zip
    # a run folder written before P2.16, whose checkpoints share one vecnormalize.pkl: name it
    # explicitly. It is pairing-checked, so it is refused for every checkpoint but its own.
    caffeinate -i python -m src.rl_gym.tools.phase2_run --out-dir runs/p216_check \
        --harness-only --checkpoint runs/p215/checkpoints/model_ep00047.zip \
        --stats runs/p215/checkpoints/vecnormalize.pkl

`caffeinate` is macOS-only; drop it elsewhere. Prints a `PASS`/`FAIL` line per pass criterion
and exits non-zero if any failed, so Ritun can re-run it independently (§0) -- the same
arrangement as P2.13's `--verify-run` and P2.14's `--verify`.

**Why a driver exists at all.** §8's P2.15 says "no code", and the run needs four settings
`train_sdpdp.py`'s CLI cannot reach: `n_steps = 256`, the evaluation pairs taken from
`docs/RL_GYM_EVAL_SEEDS.csv` (§8 P2.13: "P2.15 takes them from P2.14's seed file"),
`eval_freq`, and `n_eval_workers`. `--train` hardcodes `VERIFY_EVAL_PAIRS` and
`eval_freq = 2048`, which on this budget would fire the callback twice and miss the sizing.
Ritun's decision was a committed driver rather than a scratchpad script, because §0 has him
re-running every verification himself; it lives beside `corpus_sweep.py` (P2.4) and
`reward_trace.py` (P2.0b), the other committed drivers for report-producing items.

The figures §8 asks to be **reported, not judged** -- slot-selection frequency, the horizon
margin and the tail against its 14,400 s cap, throughput, peak RSS per worker, the projected
wall clock of a full-length run -- are collected here and written to `p215_figures.json` beside
the artifacts, so `docs/RL_GYM_PHASE2_RESULTS.md` quotes measurements rather than retyped
numbers. Load average is recorded at start and at end, and sampled throughout: P2.10 showed
throughput figures track machine load rather than the code, so a bare steps/second is not a
comparable number.
"""

# ---------------------------------------------------------------------------------------- #
# IMPORT ORDER IS LOAD-BEARING. DO NOT ALPHABETISE. DO NOT LET isort/ruff REORDER THIS BLOCK.
# numpy MUST come before torch / stable_baselines3 / sb3_contrib, or the process dies with
# "OMP: Error #15 ... Abort trap: 6". See docs/SDPDP_GYM_SPEC_v2.md §8a.
import numpy as np  # noqa: I001  -- must stay first
# ---------------------------------------------------------------------------------------- #

import argparse
import json
import os
import statistics
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# Deliberately **not** at module level: `torch` / `sb3` are imported inside the functions that
# need them, and `run_greedy_pairs` spawns a `Pool` whose children re-import this module.

#: §8 P2.15's sizing, as settled with Ritun before the run. All of it is recorded in the
#: report and in the checkpoint metadata, so the run is reproducible from the committed values.
BASE_SEED = 20261015            #: fresh; resuming from this run's checkpoints needs another
N_WORKERS = 4
N_STEPS = 256                   #: per env, for this run only, so the update path runs ~22 times
TOTAL_TIMESTEPS = 22_528        #: 22 rollouts x 1024 = 5,632 steps per worker, ~12 episodes
EVAL_FREQ = 1_700               #: in `_on_step` calls, i.e. every EVAL_FREQ x N_WORKERS steps
N_EVAL_WORKERS = 2
CHECKPOINT_EVERY_N_EPISODES = 12
SAMPLE_INTERVAL_S = 2.0

#: the callback's pairs: the **first** seed of four of the harness days, 4 over 2 workers so
#: each holds 2 -- enough for `partition_eval_pairs`' rotation argument to be exercised, which
#: a 1-pair slice cannot do. 2024-10-01, the longest day of the five (545 greedy steps), is
#: left to the harness so three evaluations cost ~4 min rather than ~6.
CALLBACK_DAYS = ("2024-01-11", "2024-03-19", "2024-06-20", "2024-08-13")

#: §8 P2.15's five evaluation days, x the first 2 seeds of each from P2.14's seed file. All 26
#: days and the full `eval_seeds_per_day` belong to real training runs, not to this gate.
#: **All five are validation days** under P2.17's split, so the gate never reads the test half;
#: their demand files moved to `corpus_val/` and are byte-unchanged, which is why this run still
#: reproduces P2.15's `kpi_report.txt` md5.
HARNESS_DAYS = ("2024-01-11", "2024-03-19", "2024-06-20", "2024-08-13", "2024-10-01")
HARNESS_SEEDS_PER_DAY = 2

#: the half this gate runs on, end to end (P2.17). A gate that could be pointed at the test
#: days would spend them on a machinery check.
HARNESS_SPLIT = "val"

#: §6.4's branch-7 indicator: the cap on `record_remaining_assignments`, written `2 * 7200` at
#: `FleetSimulationBase.py` ~625. Greedy's longest tail anywhere in the corpus is 1,950 s (P2.4)
TAIL_CAP_S = 14_400

#: for the full-length projection (§8, After Phase 2 step 7)
FULL_LENGTH_EPISODES = 7_000

FIGURES_FILE = "p215_figures.json"
KPI_REPORT_FILE = "kpi_report.txt"


# -------------------------------------------------------------------------- #
# machine conditions
# -------------------------------------------------------------------------- #

def load_average() -> Tuple[float, float, float]:
    """The 1/5/15-minute load averages, as `uptime` reports them."""
    one, five, fifteen = os.getloadavg()
    return round(one, 2), round(five, 2), round(fifteen, 2)


class ResourceSampler:
    """Samples load average and per-child RSS on a thread while the run holds the main one.

    Peak RSS per worker is one of §8's reported figures (P1.11 measured 0.40 GB/worker), and
    the load trace is what makes the throughput figures readable later: P2.10 saw the *same*
    code run at 11.40 and 9.66 steps/s two hours apart on a loaded machine.
    """

    def __init__(self, interval_s: float = SAMPLE_INTERVAL_S):
        self.interval_s = interval_s
        self.peak_rss: Dict[int, int] = {}
        self.load_samples: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self) -> None:
        import psutil
        me = psutil.Process()
        while not self._stop.is_set():
            self.load_samples.append(os.getloadavg()[0])
            for child in me.children(recursive=True):
                try:
                    rss = child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                if rss > self.peak_rss.get(child.pid, 0):
                    self.peak_rss[child.pid] = rss
            self._stop.wait(self.interval_s)

    def start(self) -> "ResourceSampler":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> "ResourceSampler":
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)
        return self

    def summary(self) -> Dict[str, Any]:
        """Peak RSS over the biggest workers, and what the load did while they ran.

        Every child is sampled, so the list holds the FleetPy workers *and* the small
        `multiprocessing` helpers (`forkserver`, `resource_tracker`); the top-N figure is what
        corresponds to P1.11's per-worker number.
        """
        peaks = sorted(self.peak_rss.values(), reverse=True)
        return {
            "children_sampled": len(self.peak_rss),
            "peak_rss_bytes_per_worker_top": peaks[:8],
            "peak_rss_gb_max": round(max(peaks) / 1e9, 3) if peaks else None,
            "peak_rss_gb_sum": round(sum(peaks) / 1e9, 3) if peaks else None,
            "load_samples": len(self.load_samples),
            "load_min": round(min(self.load_samples), 2) if self.load_samples else None,
            "load_median": (round(statistics.median(self.load_samples), 2)
                            if self.load_samples else None),
            "load_max": round(max(self.load_samples), 2) if self.load_samples else None,
        }


def helper_pids() -> set:
    """The pids of `multiprocessing`'s own two helpers, asked of `multiprocessing` itself.

    **They cannot be told apart by name.** A forkserver child is `fork`ed without `exec`, so
    every FleetPy worker inherits the forkserver's own cmdline --
    `python -c from multiprocessing.forkserver import main; main(7, 9, ['numpy'])` -- and a
    filter on that string excludes the workers along with the helper. Measured in P2.15: with
    two workers alive, a name-based filter reported **no** live children, i.e. it could not
    have detected a real orphan in the first place.

    Resolving them by pid is exact. If an internal ever moves, this returns the empty set and
    the helpers are reported as orphans -- the check then fails loudly rather than passing
    vacuously, which is the safe direction for a guard.
    """
    pids = set()
    try:
        from multiprocessing import forkserver
        pid = getattr(forkserver._forkserver, "_forkserver_pid", None)
        if pid:
            pids.add(int(pid))
    except Exception:                                      # pragma: no cover - internals move
        pass
    try:
        from multiprocessing import resource_tracker
        pid = getattr(resource_tracker._resource_tracker, "_pid", None)
        if pid:
            pids.add(int(pid))
    except Exception:                                      # pragma: no cover
        pass
    return pids


def live_children() -> List[str]:
    """Children still alive that could be orphaned workers (P1.11's distinction).

    Two kinds of child are not orphans and are excluded:

    - **`multiprocessing`'s own helpers**, by pid (see `helper_pids`). They live with the
      parent by design; a FleetPy worker still running after `close()` does not.
    - **Anything that is not this interpreter.** Every worker of ours is a fork of
      `sys.executable`, so a child with a different executable cannot be one. `caffeinate -i
      <utility>` is why this matters, and it is how the run is meant to be launched (§8 P2.15):
      caffeinate **execs the utility in its own process and forks a child to hold the power
      assertion**, and that child keeps the original argv, so it appears as a child of this
      process with the cmdline `caffeinate -i python -m ...`. Measured in P2.15, where a
      name-only filter reported it as an orphan twice -- a false positive on top of the false
      negative above.
    """
    import psutil
    helpers = helper_pids()
    out = []
    for child in psutil.Process().children(recursive=True):
        try:
            if child.pid in helpers:
                continue
            if not child.is_running() or child.status() == psutil.STATUS_ZOMBIE:
                continue
            name = " ".join(child.cmdline()) or child.name()
            executable = child.exe()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        if executable and os.path.realpath(executable) != os.path.realpath(sys.executable):
            continue                      # not our interpreter, so not one of our workers
        out.append(f"{child.pid}: {name[:120]}")
    return out


# -------------------------------------------------------------------------- #
# pass criterion 4, per step
# -------------------------------------------------------------------------- #

def make_mask_audit_callback():
    """A callback that checks **every** action against the mask it was drawn under.

    `MaskablePPO.collect_rollouts` calls `callback.update_locals(locals())` after materialising
    the action array, so `self.locals` holds both `actions` and the `action_masks` the policy
    was given (verified against sb3-contrib 2.9.0). That makes §8 P2.15's criterion 4 -- "no
    masked action is ever selected" -- a positive per-step measurement instead of the absence
    of a crash.

    The env's own `assert action < n_valid` (`gym_env._translate`) is the enforcement and would
    kill the worker; this is the evidence. It counts what it audited, so a callback that never
    received anything fails the criterion rather than passing it vacuously.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class MaskAuditCallback(BaseCallback):
        """Also times the rollout, which is what makes the throughput figures readable.

        `EpisodeLogCallback` measures an episode's wall clock as the gap between consecutive
        episode completions on that worker, so an episode that straddles an evaluation or a
        PPO update absorbs it -- measured in the smoke run, 3.6 steps/s against a real ~10.
        This callback is stepped once per rollout step, so the gaps between its own calls
        separate the two: a gap above `PAUSE_THRESHOLD_S` is an evaluation or an update, not
        simulation.
        """

        PAUSE_THRESHOLD_S = 5.0

        def __init__(self) -> None:
            super().__init__()
            self.actions_audited = 0
            self.steps_seen = 0
            self.steps_without_masks = 0
            self.violations: List[Dict[str, Any]] = []
            self.pauses: List[float] = []
            self._last_call: Optional[float] = None

        def _on_step(self) -> bool:
            now = time.perf_counter()
            if self._last_call is not None and now - self._last_call > self.PAUSE_THRESHOLD_S:
                self.pauses.append(now - self._last_call)
            self._last_call = now
            actions = self.locals.get("actions")
            if actions is None:
                return True
            self.steps_seen += 1
            masks = self.locals.get("action_masks")
            if masks is None:
                # masking silently off is itself a defect worth seeing (D3)
                self.steps_without_masks += 1
                return True
            acts = np.asarray(actions).reshape(-1)
            mask = np.asarray(masks).reshape(len(acts), -1)
            for i, action in enumerate(acts):
                self.actions_audited += 1
                if not bool(mask[i, int(action)]):
                    self.violations.append({
                        "timestep": int(self.num_timesteps), "env_index": int(i),
                        "action": int(action), "mask": [bool(v) for v in mask[i]],
                    })
            return True

    return MaskAuditCallback()


# -------------------------------------------------------------------------- #
# the training half
# -------------------------------------------------------------------------- #

def build_cfg(base_seed: int = BASE_SEED) -> Dict[str, Any]:
    """The env/training config for this run: §6.1's eight weights, explicitly (P2.13)."""
    import train_sdpdp
    return dict(train_sdpdp.DEFAULT_CFG,
                base_seed=int(base_seed),
                day_dir="corpus",                       # D10 rotation over the 101 days
                skip_output=True,
                log_level="warning",
                n_steps=N_STEPS,
                reward_weights=dict(train_sdpdp.PHASE2_REWARD_WEIGHTS))


def run_training(out_dir: str, base_seed: int = BASE_SEED,
                 total_timesteps: int = TOTAL_TIMESTEPS,
                 n_workers: int = N_WORKERS, verbose: bool = True) -> Dict[str, Any]:
    """The `learn` call, with the audit callback attached and the sampler running."""
    import train_sdpdp
    from src.rl_gym.evaluate import pairs_from_seed_file

    cfg = build_cfg(base_seed)
    eval_pairs = pairs_from_seed_file(days=list(CALLBACK_DAYS), n_seeds=1)
    audit = make_mask_audit_callback()
    sampler = ResourceSampler().start()
    load_start = load_average()
    if verbose:
        print(f"-- training: {n_workers} workers, {total_timesteps} timesteps, "
              f"n_steps {cfg['n_steps']}, base_seed {base_seed}", flush=True)
        print(f"   callback pairs: {eval_pairs} (eval_freq {EVAL_FREQ}, "
              f"{N_EVAL_WORKERS} workers)", flush=True)
        print(f"   load average at start: {load_start}", flush=True)
    try:
        result = train_sdpdp.train(
            cfg, n_workers=n_workers, total_timesteps=total_timesteps, out_dir=out_dir,
            eval_pairs=eval_pairs, n_eval_workers=N_EVAL_WORKERS, eval_freq=EVAL_FREQ,
            checkpoint_every_n_episodes=CHECKPOINT_EVERY_N_EPISODES, verbose=verbose,
            extra_callbacks=[audit])
    finally:
        sampler.stop()
    result["audit"] = audit
    result["sampler"] = sampler.summary()
    result["cfg"] = cfg
    result["eval_pairs"] = eval_pairs
    result["load_start"] = load_start
    result["load_end_training"] = load_average()
    result["orphans_after_training"] = live_children()
    return result


# -------------------------------------------------------------------------- #
# the harness half (P2.14), criterion 7
# -------------------------------------------------------------------------- #

def run_harness(checkpoint: str, out_dir: str, greedy_processes: int = 4,
                rl_workers: int = 5, verbose: bool = True,
                stats_path: Optional[str] = None) -> Dict[str, Any]:
    """P2.14's harness on the 10 pairs, both policies, then its KPI table.

    The functions are called in process rather than through the `evaluate` CLI only so the rows
    come back structured for criterion 7's assertion; the equivalent committed command is
    printed, and is what Ritun can re-run on its own:

        python -m src.rl_gym.evaluate --greedy --rl --checkpoint <ckpt> \
            --days <the five> --seeds 2 --processes 4 --workers 5 --out <report>
    """
    import train_sdpdp
    from src.rl_gym import evaluate

    # First, before the ten stock-greedy simulations: the checkpoint must have the statistics
    # it was trained under, or be given them explicitly (P2.16), and its features must still
    # mean what they meant (P2.18). Either failure found after the greedy half would cost three
    # minutes to say what this says in a second.
    stats_file = train_sdpdp.verify_statistics_pairing(checkpoint, stats_path)
    train_sdpdp.verify_artifact_contract(checkpoint, build_cfg())

    pairs = evaluate.pairs_from_seed_file(days=list(HARNESS_DAYS),
                                          n_seeds=HARNESS_SEEDS_PER_DAY,
                                          split=HARNESS_SPLIT)
    if verbose:
        print(f"-- statistics: {os.path.basename(stats_file)}, paired with "
              f"{os.path.basename(checkpoint)}", flush=True)
    if verbose:
        print(f"-- harness: {len(pairs)} pair(s) on the {HARNESS_SPLIT} half, stock greedy on "
              f"{greedy_processes} process(es) then the checkpoint on {rl_workers} worker(s)",
              flush=True)
    t0 = time.perf_counter()
    greedy_runs = evaluate.run_greedy_pairs(pairs, processes=greedy_processes,
                                            split=HARNESS_SPLIT)
    t_greedy = time.perf_counter() - t0
    t1 = time.perf_counter()
    rl_runs = evaluate.run_rl_pairs(pairs, checkpoint, n_workers=rl_workers,
                                    split=HARNESS_SPLIT, stats_path=stats_path)
    t_rl = time.perf_counter() - t1

    rows = evaluate.kpi_rows(greedy_runs + rl_runs)
    # **Deliberately without P2.17's provenance header.** `kpi_report.txt` md5
    # `db34f5aea5f8cd59437fb2628c9534e0` is the pinned artifact P2.15 published and P2.16 and
    # P2.17 re-verify against; adding a header here would change those bytes and silently retire
    # a check three work items rely on. The half is printed to the console above and recorded in
    # `p215_figures.json`. `evaluate`'s own CLI reports do carry the header -- those are the ones
    # that get filed and read later, and none of them is a pinned hash.
    report = evaluate.format_report(rows)
    report_path = os.path.join(out_dir, KPI_REPORT_FILE)
    with open(report_path, "w") as fh:
        fh.write(report + "\n")
    if verbose:
        print(report, flush=True)
        print(f"wrote {report_path}", flush=True)
    return {
        "pairs": pairs, "rows": rows, "report": report, "report_path": report_path,
        "split": HARNESS_SPLIT,
        "greedy_seconds": t_greedy, "rl_seconds": t_rl,
        "greedy_runs": greedy_runs, "rl_runs": rl_runs,
        "orphans_after_harness": live_children(),
    }


# -------------------------------------------------------------------------- #
# the eight criteria
# -------------------------------------------------------------------------- #

class Checker:
    """Collects `PASS`/`FAIL` lines, in the shape P2.13 and P2.14 established."""

    def __init__(self, verbose: bool = True):
        self.results: List[Tuple[str, bool, str]] = []
        self.verbose = verbose

    def check(self, name: str, ok: Any, detail: str = "") -> bool:
        ok = bool(ok)
        self.results.append((name, ok, detail))
        if self.verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail
                                                              else ""), flush=True)
        return ok

    @property
    def failed(self) -> List[str]:
        return [name for name, ok, _d in self.results if not ok]


def check_training(result: Dict[str, Any], checker: Checker) -> Dict[str, Any]:
    """Criteria 1-6 and 8, from the run's own artifacts."""
    import train_sdpdp
    from src.rl_gym import callbacks as cb
    from src.rl_gym.corpus import training_day_files
    from src.rl_gym.day_rotation import day_for_episode

    log = result["episode_log"]
    evaluator = result["evaluator"]
    audit = result["audit"]
    cfg = result["cfg"]
    summaries = log.episode_summaries
    eval_summaries = list(getattr(evaluator, "episode_summaries", []))

    # --- 1) the run completed, workers exited 0, nothing orphaned ----------- #
    checker.check("1. the run completed and every training worker exited 0",
                  result["worker_exitcodes"]
                  and all(code == 0 for code in result["worker_exitcodes"]),
                  f"exit codes {result['worker_exitcodes']}, "
                  f"{result['wall_seconds'] / 60:.1f} min, {log.episodes} episodes")
    checker.check("1. no orphaned workers after close()",
                  not result["orphans_after_training"],
                  "; ".join(result["orphans_after_training"]) or "none, helpers excluded")

    # --- 2) unclassified == 0 in every episode ------------------------------ #
    # the key's *absence* must fail too: `s.get(...)` on a missing key is falsy, which would
    # make this pass without having read anything
    missing_key = [s.get("day") for s in summaries + eval_summaries
                   if "unclassified" not in s]
    bad_train = [(s.get("day"), s.get("unclassified")) for s in summaries
                 if s.get("unclassified")]
    bad_eval = [(s.get("day"), s.get("unclassified")) for s in eval_summaries
                if s.get("unclassified")]
    checker.check("2. unclassified == 0 in every training episode",
                  summaries and not bad_train and not missing_key,
                  f"{len(summaries)} episodes" if not bad_train
                  else f"non-zero in {bad_train}")
    checker.check("2. unclassified == 0 in every evaluation episode",
                  eval_summaries and not bad_eval and not missing_key,
                  f"{len(eval_summaries)} episodes" if not bad_eval
                  else f"non-zero in {bad_eval}")

    # --- 3) no non-finite observation --------------------------------------- #
    # P2.12's guard raises from `_observe`, the one site reset() and step() share, and a raise
    # in a worker aborts `learn`. So this is negative evidence, and the count is what makes it
    # worth anything: every observation of every episode passed through the guard.
    observations = int(result["total_timesteps"]) + len(summaries) + len(eval_summaries)
    checker.check("3. no non-finite observation (the env would have raised)",
                  all(code == 0 for code in result["worker_exitcodes"]),
                  f"~{observations} observations passed the guard "
                  f"(one per step plus one per reset)")

    # --- 4) no masked action ever selected ---------------------------------- #
    checker.check("4. every action audited against its own mask: none masked",
                  audit.actions_audited > 0 and not audit.violations,
                  f"{audit.actions_audited} actions audited, "
                  f"{len(audit.violations)} violation(s)"
                  + (f", {audit.violations[:3]}" if audit.violations else ""))
    checker.check("4. masks reached the policy at every rollout step",
                  audit.steps_seen > 0 and audit.steps_without_masks == 0,
                  f"{audit.steps_seen} steps, {audit.steps_without_masks} without masks")

    # --- 5) the days are D10's, each visited once --------------------------- #
    counts = log.day_visit_counts()
    repeats = {d: c for d, c in counts.items() if c > 1}
    checker.check("5. no training day visited twice", not repeats,
                  f"{len(counts)} distinct days" if not repeats else f"repeats {repeats}")
    days = training_day_files()
    per_env: Dict[int, List[str]] = {}
    for s in summaries:
        per_env.setdefault(int(s.get("env_id", -1)), []).append(s.get("day"))
    mismatch = {}
    for env_id, got in per_env.items():
        want = [day_for_episode(days, cfg["base_seed"], result["n_workers"], env_id, i)[0]
                for i in range(len(got))]
        if got != want:
            mismatch[env_id] = (got, want)
    checker.check("5. each worker ran exactly the days D10 assigns it, in order", not mismatch,
                  f"{sum(len(v) for v in per_env.values())} episodes over "
                  f"{len(per_env)} workers" if not mismatch else f"{mismatch}")
    passes = sorted({int(s.get("pass", -1)) for s in summaries})
    checker.check("5. the whole run stayed inside pass 0", passes == [0], f"passes {passes}")

    # --- 6) a reloaded checkpoint is deterministic -------------------------- #
    actions_match, detail = train_sdpdp._check_reload_determinism(result["model_path"], cfg)
    checker.check("6. a reloaded checkpoint reproduces its deterministic actions",
                  actions_match, detail)

    # --- 8) every TensorBoard scalar P2.13 lists ---------------------------- #
    tags = train_sdpdp._tb_tags(result["tb_dir"])
    missing = [k for k in train_sdpdp.REQUIRED_SCALARS if k not in tags["scalars"]]
    checker.check("8. every required TensorBoard scalar is present", not missing,
                  f"{len(train_sdpdp.REQUIRED_SCALARS)} keys" if not missing
                  else f"missing {missing}")
    missing_h = [k for k in train_sdpdp.REQUIRED_HISTOGRAMS
                 if k not in tags["histograms"] and k not in tags["tensors"]]
    checker.check("8. the day-visit and candidate-length histograms are present", not missing_h,
                  "both" if not missing_h else f"missing {missing_h}")
    slot_keys = [f"slots/slot_{i}" for i in range(int(cfg["K"]))] + ["slots/reject"]
    missing_s = [k for k in slot_keys if k not in tags["scalars"]]
    checker.check("8. slot-selection frequency is logged for every slot", not missing_s,
                  f"{len(slot_keys)} slots" if not missing_s else f"missing {missing_s}")

    # the sizing statements §8 makes, reported rather than asserted
    evals = list(getattr(evaluator, "evaluation_pairs", []))
    checker.check("sizing: the evaluation callback fired at least 3 times", len(evals) >= 3,
                  f"{len(evals)} evaluation(s)")
    want_pairs = sorted(result["eval_pairs"])
    covered = [sorted(p) == want_pairs for p in evals]
    checker.check("sizing: every evaluation ran each pair exactly once", all(covered) and evals,
                  f"{sum(covered)}/{len(evals)} evaluations covered all "
                  f"{len(want_pairs)} pairs")
    checker.check("sizing: the evaluation env kept training=False and norm_reward=False",
                  evaluator is None or (not evaluator.eval_env.training
                                        and not evaluator.eval_env.norm_reward))
    ckpt_dir = os.path.join(result["out_dir"], "checkpoints")
    models = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".zip"))
    checker.check("sizing: at least two checkpoints were written", len(models) >= 2,
                  f"{len(models)}: {models}")
    # P2.16: per checkpoint, not per folder, and each paired with its own model. Before P2.16
    # the four checkpoints here shared one vecnormalize.pkl that each save overwrote.
    stats_ok, stats_detail = train_sdpdp.check_checkpoint_statistics(ckpt_dir)
    checker.check("sizing: every checkpoint has its own statistics, paired with it",
                  stats_ok, stats_detail)
    meta = cb.load_run_metadata(ckpt_dir)
    checker.check("sizing: the metadata records the base_seed, weights and hyperparameters",
                  meta.get("base_seeds") == [cfg["base_seed"]]
                  and set(meta.get("reward_weights", {})) == set(
                      train_sdpdp.REQUIRED_REWARD_WEIGHTS)
                  and "gamma" in meta.get("hyperparams", {}),
                  f"base_seeds {meta.get('base_seeds')}")
    return {"evaluations": evals, "checkpoints": models, "metadata": meta,
            "eval_summaries": eval_summaries}


def check_harness(harness: Dict[str, Any], checker: Checker) -> None:
    """Criterion 7: a KPI table for both policies on identical (day, seed) pairs."""
    rows = harness["rows"]
    want = sorted((str(d), int(s)) for d, s in harness["pairs"])
    by_policy: Dict[str, List[Tuple[str, int]]] = {}
    for row in rows:
        by_policy.setdefault(str(row.get("policy")), []).append(
            (str(row.get("day")), int(row.get("seed"))))
    greedy = sorted(by_policy.get("greedy", []))
    rl = sorted(by_policy.get("rl", []))
    checker.check("7. the KPI table holds both policies on identical (day, seed) pairs",
                  greedy == want and rl == want,
                  f"greedy {len(greedy)}, rl {len(rl)}, of {len(want)} pairs"
                  + ("" if greedy == want and rl == want
                     else f"; greedy-only {sorted(set(greedy) - set(rl))}, "
                          f"rl-only {sorted(set(rl) - set(greedy))}"))
    missing_cols = [r for r in rows
                    if r.get("served") is None or r.get("fleet_vkt_km") is None]
    checker.check("7. every KPI row carries the metrics the table needs", not missing_cols,
                  f"{len(rows)} rows" if not missing_cols
                  else f"{len(missing_cols)} row(s) incomplete")
    checker.check("7. no orphaned processes after the harness",
                  not harness["orphans_after_harness"],
                  "; ".join(harness["orphans_after_harness"]) or "none, helpers excluded")


# -------------------------------------------------------------------------- #
# reported, not judged
# -------------------------------------------------------------------------- #

def tb_scalars(tb_dir: str, tags: Sequence[str]) -> Dict[str, List[float]]:
    """The recorded values of the named scalars, in step order.

    Per-episode `steps`, `wall_seconds`, `steps_per_second` and the candidate-list figures live
    only in TensorBoard -- `EpisodeLogCallback` computes them from `info` and clears its
    buffers, and `episode_summary()` never holds them -- so the throughput figures are read
    back from the event files rather than recomputed. Read with `EventAccumulator`, which is
    also how §8's "every scalar is present" check reads them (P2.13).
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out: Dict[str, List[Tuple[int, float]]] = {tag: [] for tag in tags}
    for root, _dirs, files in os.walk(tb_dir):
        for name in files:
            if "events.out.tfevents" not in name:
                continue
            acc = EventAccumulator(os.path.join(root, name), size_guidance={"scalars": 0})
            acc.Reload()
            available = set(acc.Tags().get("scalars", []))
            for tag in tags:
                if tag in available:
                    out[tag].extend((e.step, e.value) for e in acc.Scalars(tag))
    return {tag: [v for _s, v in sorted(points)] for tag, points in out.items()}


def _stats(values: Sequence[float]) -> Optional[Dict[str, float]]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return {"n": len(vals), "min": round(min(vals), 3),
            "median": round(statistics.median(vals), 3),
            "mean": round(statistics.fmean(vals), 3), "max": round(max(vals), 3)}


def collect_figures(result: Dict[str, Any], extra: Dict[str, Any],
                    harness: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Everything §8 P2.15 asks to be reported rather than judged."""
    log = result["episode_log"]
    audit = result["audit"]
    summaries = log.episode_summaries
    cfg = result["cfg"]

    # per-episode steps, wall clock and throughput live in TensorBoard only
    scalars = tb_scalars(result["tb_dir"],
                         ["episode/steps", "episode/wall_seconds",
                          "episode/steps_per_second", "episode/candidate_list_mean",
                          "episode/candidate_list_max", "episode/candidate_lists_over_k",
                          "time/fps"])
    steps = scalars["episode/steps"]
    walls = scalars["episode/wall_seconds"]
    rates = scalars["episode/steps_per_second"]

    total = int(audit.actions_audited) or 1
    slots = {
        ("reject" if i == int(cfg["K"]) else f"slot_{i}"): {
            "count": int(c), "share": round(float(c) / total, 4)}
        for i, c in enumerate(log.slot_counts)
    }
    tails = [float(s.get("tail_seconds") or 0.0) for s in summaries]
    margins = [s.get("horizon_margin") for s in summaries
               if s.get("horizon_margin") is not None]
    per_worker_episodes: Dict[str, int] = {}
    for s in summaries:
        key = str(s.get("env_id"))
        per_worker_episodes[key] = per_worker_episodes.get(key, 0) + 1

    episodes = max(len(summaries), 1)
    per_episode_wall_per_worker = (result["wall_seconds"] * result["n_workers"] / episodes)
    # simulation time only: the whole run minus the evaluation and update pauses the audit
    # callback timed, which is the rate a worker sustains while stepping the environment
    rollout_only_s = max(result["wall_seconds"] - sum(audit.pauses), 1e-9)
    per_worker_rate = (int(audit.actions_audited) / result["n_workers"] / rollout_only_s
                       if audit.actions_audited else 0.0)
    figures: Dict[str, Any] = {
        "sizing": {
            "base_seed": cfg["base_seed"], "n_workers": result["n_workers"],
            "n_steps": cfg["n_steps"], "total_timesteps": result["total_timesteps"],
            "eval_freq": EVAL_FREQ, "n_eval_workers": N_EVAL_WORKERS,
            "callback_pairs": result["eval_pairs"],
            "checkpoint_every_n_episodes": CHECKPOINT_EVERY_N_EPISODES,
            "reward_weights": cfg["reward_weights"], "hyperparams": result["hyperparams"],
        },
        "machine": {
            "load_start": result["load_start"],
            "load_end_training": result["load_end_training"],
            "load_end": load_average(),
            **result["sampler"],
        },
        "training": {
            "episodes": len(summaries), "episodes_per_worker": per_worker_episodes,
            "wall_minutes": round(result["wall_seconds"] / 60, 2),
            "updates": int(result["total_timesteps"] //
                           (int(cfg["n_steps"]) * result["n_workers"])),
            "timesteps": int(result["total_timesteps"]),
            "evaluations": len(extra["evaluations"]),
            "evaluation_pairs": extra["evaluations"],
            "evaluation_episodes": len(extra["eval_summaries"]),
            "checkpoints": extra["checkpoints"],
        },
        "throughput": {
            # the per-episode figures come from `EpisodeLogCallback`, so the few episodes that
            # straddle an evaluation or an update carry its pause: read the **median**
            "episode_steps": _stats(steps),
            "episode_wall_seconds": _stats(walls),
            "episode_steps_per_second": _stats(rates),
            "sb3_time_fps": _stats(scalars["time/fps"]),
            "wall_per_episode_per_worker_s": round(per_episode_wall_per_worker, 1),
            # `actions_audited` is the number of env-steps actually collected, which is the
            # honest denominator: `learn` finishes the rollout it is in, so a `total_timesteps`
            # that is not a multiple of `n_steps x n_workers` is overshot
            "actions_audited": int(audit.actions_audited),
            # the audit callback's own call gaps, which separate simulation from the pauses
            "pause_count": len(audit.pauses),
            "pause_seconds_total": round(sum(audit.pauses), 1),
            "pause_seconds_max": round(max(audit.pauses), 1) if audit.pauses else None,
            "rollout_only_minutes": round(rollout_only_s / 60, 2),
            "per_worker_steps_per_second_rollout_only": (
                round(per_worker_rate, 2) if per_worker_rate else None),
            "clean_episode_wall_estimate_s": (
                round(statistics.fmean(steps) / per_worker_rate, 1)
                if steps and per_worker_rate else None),
            # projected from the pause-free rate, which is what a worker sustains while
            # simulating; a full-length run adds its own evaluation and update time on top
            "projected_full_length_hours": (
                round(FULL_LENGTH_EPISODES * (statistics.fmean(steps) / per_worker_rate)
                      / result["n_workers"] / 3600, 1)
                if steps and per_worker_rate else None),
            "projected_full_length_hours_from_median_episode": (
                round(FULL_LENGTH_EPISODES * statistics.median(walls)
                      / result["n_workers"] / 3600, 1) if walls else None),
            "projected_full_length_episodes": FULL_LENGTH_EPISODES,
        },
        "slot_selection": slots,
        "candidate_lists": {
            "per_episode_mean": _stats(scalars["episode/candidate_list_mean"]),
            "per_episode_max": _stats(scalars["episode/candidate_list_max"]),
            "per_episode_over_k": _stats(scalars["episode/candidate_lists_over_k"]),
        },
        "horizon": {
            "tail_seconds": _stats(tails),
            "tail_cap_s": TAIL_CAP_S,
            "worst_tail_share_of_cap": (round(max(tails) / TAIL_CAP_S, 4) if tails else None),
            "horizon_margin_s": _stats(margins),
            "negative_margin_episodes": sum(1 for m in margins if m < 0),
            "greedy_corpus_worst_tail_s": 1950.0,
        },
        "episode_outcomes": {
            key: _stats([s.get(key) for s in summaries])
            for key in ("pickups", "rider_declined", "diffusion_cancelled", "no_show",
                        "operator_rejected", "no_candidates", "unclassified",
                        "episode_reward")
        },
        "days_visited": log.day_visit_counts(),
    }
    if harness is not None:
        figures["harness"] = {
            "pairs": harness["pairs"],
            # which half the figures came from (P2.17) -- `kpi_report.txt` deliberately carries
            # no header, so this is where it is recorded
            "split": harness.get("split"),
            "greedy_minutes": round(harness["greedy_seconds"] / 60, 2),
            "rl_minutes": round(harness["rl_seconds"] / 60, 2),
            "rows": harness["rows"],
            "report_path": harness["report_path"],
        }
    return figures


# -------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=os.path.join("runs", "p215"),
                        help="artifacts: checkpoints, TensorBoard, the figures JSON")
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--skip-harness", action="store_true",
                        help="training and criteria 1-6, 8 only")
    parser.add_argument("--harness-only", action="store_true",
                        help="criterion 7 only; needs --checkpoint")
    parser.add_argument("--checkpoint", default=None,
                        help="--harness-only: the .zip to evaluate")
    parser.add_argument("--stats", default=None,
                        help="an explicit VecNormalize statistics file, for a run folder "
                             "written before P2.16 whose checkpoints share one "
                             "vecnormalize.pkl. Pairing-checked, so it is refused for every "
                             "checkpoint but the one it belongs to")
    parser.add_argument("--processes", type=int, default=4,
                        help="parallel stock-greedy simulations in the harness")
    parser.add_argument("--rl-workers", type=int, default=5,
                        help="workers for the checkpoint's harness episodes")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    checker = Checker()
    t0 = time.perf_counter()
    print(f"P2.15 — Phase 2 exit gate | out {args.out_dir} | "
          f"load average at start {load_average()}", flush=True)

    result: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = {}
    harness: Optional[Dict[str, Any]] = None
    checkpoint = args.checkpoint

    if not args.harness_only:
        result = run_training(args.out_dir, base_seed=args.base_seed,
                              total_timesteps=args.timesteps, n_workers=args.workers)
        result["n_workers"] = args.workers
        result["total_timesteps"] = args.timesteps
        print("-- criteria 1-6 and 8, from the training run --", flush=True)
        extra = check_training(result, checker)
        checkpoint = result["model_path"]

    if not args.skip_harness:
        if not checkpoint:
            parser.error("--harness-only needs --checkpoint")
        harness = run_harness(checkpoint, args.out_dir, greedy_processes=args.processes,
                              rl_workers=args.rl_workers, stats_path=args.stats)
        print("-- criterion 7, from the harness --", flush=True)
        check_harness(harness, checker)

    if result is not None:
        figures = collect_figures(result, extra, harness)
        figures["wall_minutes_total"] = round((time.perf_counter() - t0) / 60, 2)
        path = os.path.join(args.out_dir, FIGURES_FILE)
        with open(path, "w") as fh:
            json.dump(figures, fh, indent=2, sort_keys=True, default=str)
        print(f"\nwrote {path}", flush=True)

    wall = time.perf_counter() - t0
    failed = checker.failed
    print(f"\n{len(checker.results) - len(failed)}/{len(checker.results)} checks passed in "
          f"{wall / 60:.1f} min | load average at end {load_average()}")
    if failed:
        print("FAILED: " + "; ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    # The guard is required: `SubprocVecEnv` uses the forkserver start method where available
    # and `multiprocessing` preloads `['__main__']` into it, executing this file top to bottom
    # before forking any worker (§8a). Without the guard, main() would run in the forkserver.
    # `train_sdpdp.prepare_multiprocessing()` additionally preloads numpy by name, which is
    # what carries §8a's import order into workers started from a `-m` invocation.
    sys.exit(main())
