"""D10's training-day rotation: global shuffled passes, one day per episode.

Within a pass every training day is assigned exactly once. Pass `p` uses one permutation of the
sorted training-day list, drawn from `SeedSequence([base_seed, 1010, p + 1])` -- identical in
every worker, so the workers partition a pass rather than duplicating it. Worker `env_id` of
`n_workers` takes the days at positions `env_id, env_id + n_workers, ...` of that permutation,
then moves on to pass `p + 1`.

Two details of that seed, both load-bearing and both checked:

- The constant `1010` tags the stream, and `p + 1` keeps the last entry non-zero. `SeedSequence`
  pads its entropy with zeros, so `[base_seed, 1010, 0]` and `[base_seed, 1010]` are the *same*
  stream -- verified. Together they keep the pass stream from ever coinciding with a worker's
  episode-seed stream `SeedSequence([base_seed, env_id])` (D7, P1.9).
- Drawing a day uniformly at random per episode is deliberately *not* used: it balances only on
  average and over a short run can skip days entirely.

Visit counts stay close but are not exactly +/-1 across a whole run: with 101 days and 4 workers
worker 0 owns 26 positions per pass and the others 25, and SB3 advances workers by steps rather
than episodes, so workers move through passes at slightly different rates. P2.13's day-visit
histogram reports it.

Pure functions only -- no FleetPy import, no env state -- so the rotation is testable in
milliseconds rather than through real episodes.
"""

import os
from typing import List, Sequence, Tuple

import numpy as np

#: tags the pass-permutation stream so it cannot collide with the episode-seed stream (D10)
PASS_STREAM_TAG = 1010


def training_days(day_dir: str) -> List[str]:
    """Sorted filename stems of the demand days in `day_dir`.

    **`*.csv` only** (§2.9): macOS creates and rewrites `.DS_Store` in these directories on its
    own and it is untracked, so it can appear, change or vanish between runs with no demand file
    changing. Enumerating directory entries instead would introduce a phantom day.
    """
    if not os.path.isdir(day_dir):
        raise ValueError(f"day_dir is not a directory: {day_dir}")
    days = sorted(f[:-4] for f in os.listdir(day_dir) if f.endswith(".csv"))
    if not days:
        raise ValueError(f"no *.csv demand days in {day_dir}")
    return days


def pass_permutation(days: Sequence[str], base_seed: int, p: int) -> List[str]:
    """Pass `p`'s permutation of `days` -- identical in every worker (D10).

    :param days: the sorted training-day list; order in matters, so pass `training_days()`
        output rather than an arbitrary ordering.
    :param p: pass index, zero-based. `p + 1` is what reaches `SeedSequence`; see the module
        docstring for why it must not be 0.
    """
    if p < 0:
        raise ValueError(f"pass index must be >= 0, got {p}")
    ss = np.random.SeedSequence([int(base_seed), PASS_STREAM_TAG, int(p) + 1])
    return list(np.random.default_rng(ss).permutation(list(days)))


def positions_per_pass(n_days: int, n_workers: int, env_id: int) -> int:
    """How many days worker `env_id` owns in one pass: positions env_id, +n_workers, ..."""
    return len(range(env_id, n_days, n_workers))


def day_for_episode(days: Sequence[str], base_seed: int, n_workers: int, env_id: int,
                    episode_index: int) -> Tuple[str, int, int]:
    """The day worker `env_id` runs for its `episode_index`-th episode (zero-based).

    :return: `(day, pass_index, position)` -- the position being the index into that pass's
        permutation, kept for the day-visit diagnostics P2.13 reports.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if not 0 <= env_id < n_workers:
        raise ValueError(
            f"env_id must be in [0, n_workers), got env_id={env_id} with "
            f"n_workers={n_workers}. Each worker owns the positions env_id, env_id+n_workers, "
            f"... of every pass, so an env_id at or beyond n_workers owns none of them."
        )
    if episode_index < 0:
        raise ValueError(f"episode_index must be >= 0, got {episode_index}")

    n_days = len(days)
    per_pass = positions_per_pass(n_days, n_workers, env_id)
    if per_pass == 0:
        raise ValueError(
            f"worker {env_id} of {n_workers} owns no position in a pass over {n_days} day(s): "
            f"n_workers exceeds the number of training days."
        )

    p = episode_index // per_pass
    position = env_id + (episode_index % per_pass) * n_workers
    return pass_permutation(days, base_seed, p)[position], p, position
