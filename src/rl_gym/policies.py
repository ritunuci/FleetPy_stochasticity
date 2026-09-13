"""Scripted policies over the SDPDP observation.

One function, deliberately. `greedy_action` is what P1.10's exit gate drives the environment
with, and it is also what P2.7's greedy baseline must use for the RL comparison. If the
baseline reimplemented the selection and drifted, the headline RL-versus-greedy result would be
invalid with nothing failing, so there is exactly one implementation and both import it.
"""

from typing import Optional

import numpy as np


def greedy_action(obs: np.ndarray, mask: np.ndarray, k_max: int) -> int:
    """The greedy choice: the valid candidate slot with the smallest `delta_cfv`.

    **Observation layout dependency.** `CandidateObserver` (`src/rl_gym/observers.py`) emits
    `is_valid`, then `delta_cfv`, then `offered_wait`, each of length `k_max`, before the
    global block. So `delta_cfv` occupies `obs[k_max:2 * k_max]`. If that ordering ever changes,
    this function changes with it -- the coupling is stated here rather than left to be
    inferred from where the file lives.

    **The `is_valid` restriction is load-bearing regardless of the sign of `delta_cfv`.** Padded
    slots hold 0.0, which is padding rather than a cost, and must never compete. On the
    reference scenario real `delta_cfv` is negative, so an unrestricted `argmin` would happen to
    agree -- which is exactly why the byte-for-byte gate cannot catch a dropped restriction, and
    why `tests/test_rl_gym_observers.py` pins the positive-cost counter-example instead.

    Equivalently, since `insertion_with_heuristics` returns candidates sorted ascending by
    `delta_cfv` and truncation takes a prefix, this is slot 0 whenever any candidate is valid.
    Selecting by value rather than assuming index 0 keeps the function honest if either
    property ever changes.

    :param obs: the observation vector, length `3 * k_max + 4`
    :param mask: the action mask, length `k_max + 1`; entry `k_max` is reject
    :param k_max: `K`, the number of candidate slots
    :return: a candidate slot index, or `k_max` to reject when no candidate is valid
    """
    obs = np.asarray(obs)
    mask = np.asarray(mask, dtype=bool)
    if obs.shape != (3 * k_max + 4,):
        raise ValueError(f"observation has shape {obs.shape}, expected {(3 * k_max + 4,)}")
    if mask.shape != (k_max + 1,):
        raise ValueError(f"mask has shape {mask.shape}, expected {(k_max + 1,)}")

    valid_slots = np.flatnonzero(mask[:k_max])
    if valid_slots.size == 0:
        return int(k_max)                       # nothing to offer; reject is always legal

    delta_cfv = obs[k_max:2 * k_max]
    return int(valid_slots[np.argmin(delta_cfv[valid_slots])])


def greedy_choice(obs: np.ndarray, mask: np.ndarray, k_max: int) -> Optional[int]:
    """`greedy_action` as a semantic choice: a candidate index, or `None` for reject."""
    action = greedy_action(obs, mask, k_max)
    return None if action == k_max else action
