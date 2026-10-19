"""Training callbacks: per-episode logging, episode-counted checkpoints, evaluation (P2.13).

Three callbacks and one pure helper, all driven by what §8's P2.13 asks to be visible:

- `EpisodeLogCallback` — the per-episode TensorBoard scalars, plus the two aggregated
  distributions. One dump per finished episode, so an episode's day, seed and counts land as
  their own point rather than being averaged into the rollout.
- `EpisodeCheckpointCallback` — a checkpoint every N **episodes** (SB3's own
  `CheckpointCallback` counts steps), together with its `VecNormalize` statistics and the run
  metadata that makes a resume safe.
- `PairLoggingMaskableEvalCallback` — `MaskableEvalCallback` plus a record of the (day, seed)
  every evaluation episode actually ran, which is the only way to see that an evaluation
  covered the pairs it was supposed to.
- `partition_eval_pairs` — splits the evaluation pairs across the evaluation workers the way
  `evaluate_policy` spreads episodes across them.

**This module imports `torch` through SB3, so nothing under `src/rl_gym/` may import it** --
only `train_sdpdp.py` and `tests/test_rl_gym_train.py` do. §8a's import-order constraint
applies here: numpy first.
"""

# ---------------------------------------------------------------------------------------- #
# numpy MUST come before torch / stable_baselines3 / sb3_contrib (§8a). DO NOT ALPHABETISE.
import numpy as np  # noqa: I001  -- must stay first
# ---------------------------------------------------------------------------------------- #

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback


def partition_eval_pairs(pairs: Sequence[Tuple[str, int]],
                         n_envs: int) -> List[List[Tuple[str, int]]]:
    """Split `pairs` across `n_envs` evaluation workers, matching `evaluate_policy`.

    `evaluate_policy` gives worker `i` a target of `(n_eval_episodes + i) // n_envs` episodes
    (verified in P2.13 against sb3-contrib 2.9.0), which sums to `n_eval_episodes` with the
    **later** workers taking the remainder. With `n_eval_episodes = len(pairs)`, giving each
    worker the slice of exactly that length makes every pair run exactly once per evaluation.

    **Why each worker gets its own slice rather than the whole list.** Each env rotates through
    the pairs it was given, `pair[(episode_index) % len]` (P2.3). Handed the whole list, worker
    `i` would run only the first `(E + i) // n_envs` of them and never reach the rest. Handed
    its own slice, it cycles within that slice.

    **And why a shifted start is harmless.** `evaluate_policy` stops only when *every* worker
    has hit its target, so a faster worker can complete an extra episode that is discarded --
    and that episode still advanced its rotation, so the next evaluation starts at a different
    offset. It does not matter: `L` consecutive positions of a cycle of length `L` cover every
    element exactly once whatever the offset. The order rotates between evaluations; the set
    does not. P2.13 tests that across two consecutive evaluations rather than one, which is the
    only way the shift would show.

    :param pairs: the (day, seed) list, in the order the config gives it
    :param n_envs: number of evaluation workers
    :return: one list per worker, concatenating back to `pairs` in order
    """
    if n_envs <= 0:
        raise ValueError(f"n_envs must be positive, got {n_envs}")
    pairs = list(pairs)
    if not pairs:
        raise ValueError("no evaluation pairs to partition")
    if len(pairs) < n_envs:
        raise ValueError(
            f"{len(pairs)} evaluation pair(s) across {n_envs} workers: a worker with no pair "
            f"would have nothing to run, and `evaluate_policy` would still wait for it. Give "
            f"at least one pair per worker."
        )
    counts = [(len(pairs) + i) // n_envs for i in range(n_envs)]
    assert sum(counts) == len(pairs), (counts, len(pairs))
    out, start = [], 0
    for count in counts:
        out.append(pairs[start:start + count])
        start += count
    return out


class EpisodeLogCallback(BaseCallback):
    """Per-episode TensorBoard scalars, and the two aggregated distributions (§8 P2.13).

    Everything here comes from `info` on the terminal step -- `episode_summary` (P2.0b, P2.3,
    P2.7, P2.11, P2.13's `tail_seconds`) and `n_candidates` on every step -- so the callback
    adds no simulation work.

    Two scalars that exist to be read rather than admired:

    - **slot-selection frequency.** Frequent selection of slot `K - 1` means `K` is binding:
      the truncation is cutting off candidates the policy wants, and `K` would need raising
      (D3). Read it with the candidate-list lengths below.
    - **`tail_seconds`.** §6.4's branch 7 goes live when the post-`end_time` tail's own
      14,400 s cap binds, and a policy that accepts more marginal requests lengthens the tail.
      Greedy's longest anywhere in the corpus is 1,950 s. **A negative `horizon_margin` is the
      normal case, not a warning** -- under greedy it is negative on 79 of 127 days, median
      -234 s (P2.4) -- so the tail against its cap is the number to watch, not the margin.
    """

    def __init__(self, k_max: int, training_days: Optional[Sequence[str]] = None,
                 verbose: int = 0):
        """
        :param k_max: `K`, so the slot histogram has the right width
        :param training_days: the sorted day list, used to log a numeric day index alongside
            the day's name. The histogram over those indices is D10's day-visit histogram.
        """
        super().__init__(verbose)
        self.k_max = int(k_max)
        self.training_days = list(training_days) if training_days else []
        self._day_index = {day: i for i, day in enumerate(self.training_days)}
        self.episodes = 0
        #: cumulative, for the aggregated views
        self.slot_counts = np.zeros(self.k_max + 1, dtype=np.int64)
        self.visited_day_indices: List[int] = []
        self.visited_days: List[str] = []
        self.episode_summaries: List[Dict[str, Any]] = []
        self._candidate_lengths: Dict[int, List[int]] = {}
        self._episode_start: Dict[int, float] = {}

    def _on_training_start(self) -> None:
        now = time.perf_counter()
        for i in range(self.training_env.num_envs):
            self._episode_start.setdefault(i, now)
            self._candidate_lengths.setdefault(i, [])

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None:
            for action in np.asarray(actions).reshape(-1):
                self.slot_counts[int(action)] += 1

        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        dones = np.asarray(dones).reshape(-1) if dones is not None else np.zeros(len(infos))

        for i, info in enumerate(infos):
            n_candidates = info.get("n_candidates")
            if n_candidates is not None:
                self._candidate_lengths.setdefault(i, []).append(int(n_candidates))
            if not dones[i]:
                continue
            self._log_episode(i, info)
        return True

    def _log_episode(self, env_index: int, info: Dict[str, Any]) -> None:
        summary = info.get("episode_summary") or {}
        self.episodes += 1
        self.episode_summaries.append(dict(summary))

        wall = time.perf_counter() - self._episode_start.get(env_index, time.perf_counter())
        self._episode_start[env_index] = time.perf_counter()
        steps = info.get("episode_steps") or summary.get("episode_steps") or 0

        rec = self.logger.record
        rec("episode/reward", float(summary.get("episode_reward", 0.0)))
        rec("episode/steps", int(steps))
        rec("episode/wall_seconds", float(wall))
        if wall > 0:
            rec("episode/steps_per_second", float(steps) / wall)
        for key in ("pickups", "rider_declined", "diffusion_cancelled", "no_show",
                    "operator_rejected", "no_candidates", "unclassified", "served",
                    "unserved_at_horizon", "duplicate_boardings"):
            if key in summary:
                rec(f"episode/{key}", int(summary[key]))
        # the two §8 singles out, logged every episode rather than at the end of the run
        if summary.get("horizon_margin") is not None:
            rec("episode/horizon_margin", float(summary["horizon_margin"]))
        rec("episode/tail_seconds", float(summary.get("tail_seconds", 0.0)))

        day = summary.get("day")
        if day is not None:
            self.visited_days.append(str(day))
            rec("episode/day", str(day))                       # add_text; the day is a name
            index = self._day_index.get(str(day))
            if index is not None:
                self.visited_day_indices.append(index)
                rec("episode/day_index", int(index))
        if summary.get("random_seed") is not None:
            rec("episode/random_seed", int(summary["random_seed"]))
        if summary.get("pass") is not None:
            rec("episode/pass", int(summary["pass"]))
        rec("episode/env_id", int(summary.get("env_id", env_index)))

        lengths = self._candidate_lengths.get(env_index) or []
        if lengths:
            # an array, so SB3's TensorBoard writer sends it to add_histogram
            rec("episode/candidate_list_lengths", np.asarray(lengths, dtype=np.float32))
            rec("episode/candidate_list_mean", float(np.mean(lengths)))
            rec("episode/candidate_list_max", int(np.max(lengths)))
            rec("episode/candidate_lists_over_k",
                int(np.sum(np.asarray(lengths) > self.k_max)))
            self._candidate_lengths[env_index] = []

        total = max(int(self.slot_counts.sum()), 1)
        for slot in range(self.k_max + 1):
            name = "reject" if slot == self.k_max else f"slot_{slot}"
            rec(f"slots/{name}", float(self.slot_counts[slot]) / total)
        if self.visited_day_indices:
            rec("train/day_visits",
                np.asarray(self.visited_day_indices, dtype=np.float32))
        rec("train/episodes", self.episodes)
        # one dump per episode: SB3 dumps per rollout, which would average several episodes'
        # days and seeds into one point
        self.logger.dump(self.num_timesteps)

    def day_visit_counts(self) -> Dict[str, int]:
        """How many times each day has been visited -- D10's histogram, as numbers."""
        counts: Dict[str, int] = {}
        for day in self.visited_days:
            counts[day] = counts.get(day, 0) + 1
        return counts


class EpisodeCheckpointCallback(BaseCallback):
    """Checkpoint every N **episodes**, with the statistics and the metadata (§8 P2.13).

    SB3's `CheckpointCallback` counts steps; episodes are the unit that matters here, because
    an episode is a day and a seed.

    Three artifacts per checkpoint, written together:

    - the model, `model_ep{episodes:05d}.zip`;
    - its `VecNormalize` statistics, through the injected `save_stats_fn`, in a file named
      after that checkpoint -- `model_ep00012_vecnormalize.pkl` beside `model_ep00012.zip`
      (P2.16). A policy and the statistics it was trained under are one artifact, and
      evaluating a checkpoint against another point's statistics silently shifts every
      observation (§5.3). Through P2.15 this was one shared `vecnormalize.pkl` per folder that
      each save overwrote, so only the newest checkpoint kept its own statistics and the rest
      lost theirs;
    - `run_metadata.json`, carrying **every `base_seed` the run has used** and the run's
      **`observation_contract`**. Resuming with a `base_seed` already in that list would restart
      D10's rotation and the episode-seed streams from the beginning and replay sample paths
      already trained on, so `check_resume_base_seed` refuses it.

    `observation_contract` records what the checkpoint's observations *mean* (P2.18):
    `baseline_hash`, the layout fingerprint, the observation length, `position_encoding` and
    the resolved `wait_normalizer_s`. `train_sdpdp.verify_artifact_contract` refuses to evaluate
    or resume a checkpoint whose record differs from the current values, and refuses one with no
    record at all. It is **one record per folder rather than per checkpoint**, which is not a
    repeat of the statistics defect above: unlike `VecNormalize` statistics these values cannot
    change within a run, and `--resume-from` is itself guarded.
    """

    METADATA_FILE = "run_metadata.json"

    def __init__(self, every_n_episodes: int, save_path: str,
                 save_stats_fn: Callable[[Any, str], str],
                 metadata: Optional[Dict[str, Any]] = None, verbose: int = 0):
        super().__init__(verbose)
        if every_n_episodes <= 0:
            raise ValueError(f"every_n_episodes must be positive, got {every_n_episodes}")
        self.every_n_episodes = int(every_n_episodes)
        self.save_path = save_path
        self.save_stats_fn = save_stats_fn
        self.metadata = dict(metadata or {})
        self.episodes = 0
        self.saved: List[str] = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is None:
            return True
        for done in np.asarray(dones).reshape(-1):
            if not done:
                continue
            self.episodes += 1
            if self.episodes % self.every_n_episodes == 0:
                self.save()
        return True

    def save(self) -> str:
        """Write the model, its statistics and the metadata. Returns the model path."""
        os.makedirs(self.save_path, exist_ok=True)
        path = os.path.join(self.save_path, f"model_ep{self.episodes:05d}.zip")
        self.model.save(path)
        venv = self.model.get_vec_normalize_env()
        if venv is None:
            raise RuntimeError(
                "the training env is not VecNormalize-wrapped, so a checkpoint would be saved "
                "without the statistics it was trained under and could not be evaluated "
                "faithfully (§5.3). Build the venv with build_venv(..., normalize=True)."
            )
        self.save_stats_fn(venv, path)
        meta = dict(self.metadata)
        meta.update({"episodes": self.episodes, "timesteps": int(self.num_timesteps),
                     "checkpoints": self.saved + [os.path.basename(path)]})
        with open(os.path.join(self.save_path, self.METADATA_FILE), "w") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
        self.saved.append(os.path.basename(path))
        return path


def load_run_metadata(path: str) -> Dict[str, Any]:
    """The metadata beside a checkpoint, or `{}` when there is none."""
    directory = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    meta_path = os.path.join(directory, EpisodeCheckpointCallback.METADATA_FILE)
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path) as fh:
        return json.load(fh)


def check_resume_base_seed(metadata: Dict[str, Any], base_seed: int) -> None:
    """Raise if `base_seed` has already been used by the run being resumed (§8 P2.13).

    Resuming with the same `base_seed` restarts D10's pass permutation and every worker's
    episode-seed stream from the beginning, so the run would replay (day, seed) pairs it has
    already trained on while every log looked healthy.
    """
    used = [int(s) for s in metadata.get("base_seeds", [])]
    if int(base_seed) in used:
        raise ValueError(
            f"base_seed {base_seed} has already been used by this run ({used}). Resuming with "
            f"it would restart D10's day rotation and the episode-seed streams from pass 0 and "
            f"replay sample paths already trained on. Choose a base_seed that is not in that "
            f"list; the rotation restarts at pass 0 of the new seed, and the day-visit "
            f"histogram shows the small imbalance that leaves."
        )


class PairLoggingMaskableEvalCallback(MaskableEvalCallback):
    """`MaskableEvalCallback` that records the (day, seed) of every evaluation episode.

    §8's P2.13 requires each evaluation to run every pair exactly once and the pair of every
    evaluation episode to be logged. Without this there is nothing to check that against: the
    callback reports a mean reward, not which days produced it.

    It hooks `_log_success_callback`, which `evaluate_policy` calls per env per step with
    `info` and `done` in its locals -- so the terminal `episode_summary` is available there
    without reimplementing `evaluate_policy`. **It keeps the whole summary too**
    (`episode_summaries`, added in P2.15), because the evaluation episodes are otherwise the
    one part of a training run with no `unclassified` reading at all.

    **On the statistics**: the parent already calls `sync_envs_normalization(training_env,
    eval_env)` at each evaluation, which deep-copies `obs_rms` and `ret_rms` only. It does
    **not** set `training` or `norm_reward` -- those must be False on the eval env from
    construction (P2.12's `build_venv(..., training=False)`), and this class asserts they
    still are after each evaluation, since a sync that quietly re-enabled them would make the
    evaluation update the statistics it is meant to hold fixed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #: one list of (day, seed) per evaluation, in order
        self.evaluation_pairs: List[List[Tuple[str, int]]] = []
        #: every counted evaluation episode's `episode_summary()`, flat and in order. P2.15's
        #: pass criterion 2 (`unclassified = 0` in **every** episode) would otherwise cover
        #: only the training episodes: `EpisodeLogCallback` rides on the training env, and the
        #: evaluation episodes run through `evaluate_policy` on a venv it never sees. The
        #: summary is already in `info` here, so recording it costs nothing. Only counted
        #: episodes appear -- `evaluate_policy` calls this callback solely for workers that
        #: have not yet reached their target, so a faster worker's discarded episode is absent,
        #: exactly as for `evaluation_pairs`.
        self.episode_summaries: List[Dict[str, Any]] = []
        self._current: List[Tuple[str, int]] = []

    def _log_success_callback(self, locals_: Dict[str, Any],
                             globals_: Dict[str, Any]) -> None:
        super()._log_success_callback(locals_, globals_)
        if not locals_.get("done"):
            return
        info = locals_.get("info") or {}
        summary = info.get("episode_summary") or {}
        if summary:
            self.episode_summaries.append(dict(summary))
        day, seed = summary.get("day"), summary.get("random_seed")
        if day is not None and seed is not None:
            self._current.append((str(day), int(seed)))

    def _on_step(self) -> bool:
        # the same condition the parent tests. `BaseCallback.on_step` increments `n_calls`
        # before calling `_on_step`, so `n_calls` here is already the current count -- using
        # `n_calls + 1` would record the pairs of the wrong evaluation.
        fires = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if not fires:
            return super()._on_step()
        self._current = []
        result = super()._on_step()
        self.evaluation_pairs.append(list(self._current))
        self._assert_statistics_stay_frozen()
        return result

    def _assert_statistics_stay_frozen(self) -> None:
        from stable_baselines3.common.vec_env import VecNormalize
        if not isinstance(self.eval_env, VecNormalize):
            return
        if self.eval_env.training or self.eval_env.norm_reward:
            raise RuntimeError(
                f"the evaluation env came back from sync_envs_normalization with "
                f"training={self.eval_env.training} and "
                f"norm_reward={self.eval_env.norm_reward}; both must stay False (§5.3), or "
                f"the evaluation updates the statistics it is meant to hold fixed and reports "
                f"a normalised return as if it were the real one."
            )
