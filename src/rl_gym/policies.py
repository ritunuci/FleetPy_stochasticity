"""Scripted policies over the SDPDP observation.

One function, deliberately. `greedy_action` is what P1.10's exit gate drives the environment
with, and what P2.4's corpus sweep selects through. If a second implementation existed and
drifted, a byte-for-byte gate could pass against a policy that is no longer greedy, so there is
exactly one and every driver imports it.

**Features are addressed by name, through the layout map** the env publishes (P2.8). The old
form sliced `delta_cfv` out of `obs[K:2K]`, which was true of the Phase 1 vector and stops being
true the moment P2.9 and P2.10 widen it.
"""

from typing import Dict, Optional

import numpy as np


def _feature(obs: np.ndarray, layout: Dict[str, slice], name: str) -> np.ndarray:
    try:
        sl = layout[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"the observation layout has no {name!r} feature; got "
            f"{sorted(layout) if isinstance(layout, dict) else type(layout).__name__}. "
            f"Pass the env's `observation_layout` (P2.8)."
        ) from None
    if sl.stop > obs.shape[-1]:
        raise ValueError(
            f"layout puts {name!r} at {sl} but the observation has only {obs.shape[-1]} values"
        )
    return obs[sl]


def greedy_action(obs: np.ndarray, mask: np.ndarray, layout: Dict[str, slice]) -> int:
    """The greedy choice: the valid candidate slot with the smallest `delta_cfv`.

    `K` is taken from the width of the `delta_cfv` slice, so nothing here assumes a total
    observation length or a block ordering.

    **The `is_valid` restriction is load-bearing regardless of the sign of `delta_cfv`.** Padded
    slots hold 0.0, which is padding rather than a cost, and must never compete. On the
    reference scenario real `delta_cfv` is negative, so an unrestricted `argmin` would happen to
    agree -- which is exactly why the byte-for-byte gate cannot catch a dropped restriction, and
    why `tests/test_rl_gym_observers.py` pins the positive-cost counter-example instead.

    **`is_valid` is read from the observation and required to equal the mask exactly.** That
    enforces trap 12 on every gate run rather than only in unit tests: the mask and the
    `is_valid` feature answer the same question and both come from
    `spaces.candidate_slot_validity`, so a drift between them is a defect, and nothing else
    would raise on it.

    It also **refuses a `VecNormalize`'d observation** (P2.12), which is the more dangerous
    case. After normalization `is_valid` is no longer 0/1 -- but the real problem is that each
    `delta_cfv` slot is scaled by its *own* running mean and variance, so an argmin across slots
    compares z-scores drawn from different distributions instead of costs. It would return
    plausible actions that are quietly not greedy, and no gate would notice. The exact-0/1 check
    is what catches it. Drive this with the raw env observation, never a normalized one.

    Equivalently, since `insertion_with_heuristics` returns candidates sorted ascending by
    `delta_cfv` and truncation takes a prefix, this is slot 0 whenever any candidate is valid.
    Selecting by value rather than assuming index 0 keeps the function honest if either
    property ever changes.

    :param obs: the raw observation vector, as the env returns it
    :param mask: the action mask, length `K + 1`; entry `K` is reject
    :param layout: the env's `observation_layout`, name -> slice
    :return: a candidate slot index, or `K` to reject when no candidate is valid
    """
    obs = np.asarray(obs)
    mask = np.asarray(mask, dtype=bool)

    delta_cfv = _feature(obs, layout, "delta_cfv")
    is_valid = _feature(obs, layout, "is_valid")
    k_max = int(delta_cfv.shape[-1])
    if is_valid.shape[-1] != k_max:
        raise ValueError(
            f"layout disagrees with itself: delta_cfv has {k_max} slots, is_valid has "
            f"{is_valid.shape[-1]}"
        )
    if mask.shape != (k_max + 1,):
        raise ValueError(f"mask has shape {mask.shape}, expected {(k_max + 1,)}")

    # trap 12, and the normalized-observation guard -- see the docstring
    if not np.all((is_valid == 0.0) | (is_valid == 1.0)):
        raise ValueError(
            f"is_valid must be exactly 0.0 or 1.0, got {np.asarray(is_valid).tolist()}. "
            f"This is what a VecNormalize'd observation looks like; greedy selection needs the "
            f"raw vector, because normalization scales each delta_cfv slot by its own running "
            f"mean and variance and an argmin across slots would then compare z-scores rather "
            f"than costs (P2.8, P2.12)."
        )
    if not np.array_equal(is_valid != 0.0, mask[:k_max]):
        raise ValueError(
            f"is_valid {np.asarray(is_valid).tolist()} disagrees with the mask "
            f"{mask[:k_max].tolist()}. Both answer which slots are occupied and both must come "
            f"from spaces.candidate_slot_validity (trap 12)."
        )

    valid_slots = np.flatnonzero(mask[:k_max])
    if valid_slots.size == 0:
        return int(k_max)                       # nothing to offer; reject is always legal

    return int(valid_slots[np.argmin(delta_cfv[valid_slots])])


def greedy_choice(obs: np.ndarray, mask: np.ndarray,
                  layout: Dict[str, slice]) -> Optional[int]:
    """`greedy_action` as a semantic choice: a candidate index, or `None` for reject."""
    k_max = int(_feature(np.asarray(obs), layout, "delta_cfv").shape[-1])
    action = greedy_action(obs, mask, layout)
    return None if action == k_max else action
