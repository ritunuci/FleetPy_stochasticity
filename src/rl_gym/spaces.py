"""Action space, candidate truncation, and action masking.

The action space is `Discrete(K + 1)` over ranked candidates (D3): slot `k` for
`k in [0, K)` offers the insertion plan for the k-th cheapest candidate, and slot `K`
rejects. Reject is always legal, so the mask is never all-False.

Two rules this module exists to enforce:

1. **Truncation is a prefix, never a sort.** `insertion_with_heuristics` returns the
   candidates already sorted ascending by `delta_cfv` via a stable sort, so element 0 is
   exactly what stock FleetPy's `min(list_tuples, key=lambda x: x[2])` picks. Re-sorting
   with any tie-break would change which vehicle wins on tied costs -- upstream order
   within ties comes from a backwards Dijkstra ordered by proximity to the request
   origin, not by vid -- and would break the byte-for-byte exit gate (D3, trap 10).

2. **Slot validity has one source.** `candidate_slot_validity` answers "which of the K
   slots is occupied", and both the action mask here and the `is_valid` observation
   feature in P1.6's `CandidateObserver` must read it. Computing them separately lets
   them drift with nothing raising: the policy would either see a slot it cannot select
   or be able to select a slot the observation calls padding (trap 12).

`K` itself is owned by `SDPDPAssignmentEnv` (P1.8) -- these are pure functions that take
it as an argument and hold no state. Action translation also lives in the env, not here.
"""

from typing import Any, List, Sequence, Tuple

import numpy as np
from gymnasium import spaces

# (vid, VehiclePlan, delta_cfv), exactly as insertion_with_heuristics returns it
Candidate = Tuple[Any, Any, float]


def _check_k_max(k_max: int) -> None:
    if not isinstance(k_max, (int, np.integer)) or isinstance(k_max, bool):
        raise TypeError(f"k_max must be an int, got {type(k_max).__name__}")
    if k_max <= 0:
        raise ValueError(f"k_max must be positive, got {k_max}")


def make_action_space(k_max: int) -> spaces.Discrete:
    """The `Discrete(k_max + 1)` action space: k_max candidate slots plus reject.

    :param k_max: number of candidate slots, `K`
    :return: gymnasium Discrete space of size k_max + 1
    """
    _check_k_max(k_max)
    return spaces.Discrete(int(k_max) + 1)


def reject_action(k_max: int) -> int:
    """The action index meaning "reject" -- always the last slot.

    :param k_max: number of candidate slots, `K`
    :return: k_max
    """
    _check_k_max(k_max)
    return int(k_max)


def truncate_candidates(candidates: Sequence[Candidate], k_max: int) -> List[Candidate]:
    """Return the first `k_max` candidates, in their original order.

    A prefix and nothing else: no sorting, no tie-breaking, no reordering. Slot indices
    therefore map directly onto candidate indices, which is what lets the env pass an
    action straight through as a candidate index.

    Returns a new list, so a later mutation by the caller cannot reach back into
    `PendingDecision.candidates`.

    :param candidates: candidate list as insertion_with_heuristics returned it
    :param k_max: number of candidate slots, `K`
    :return: new list holding at most k_max candidates, order preserved
    """
    _check_k_max(k_max)
    return list(candidates[:int(k_max)])


def candidate_slot_validity(n_candidates: int, k_max: int) -> np.ndarray:
    """Which of the `k_max` candidate slots are occupied.

    The single source of truth for slot validity, shared by `build_action_mask` here and
    by the `is_valid` observation feature in P1.6. Do not reimplement this comparison
    anywhere else (trap 12).

    :param n_candidates: number of candidates available *before* truncation
    :param k_max: number of candidate slots, `K`
    :return: bool array of length k_max; entry k is True if slot k holds a candidate
    """
    _check_k_max(k_max)
    if not isinstance(n_candidates, (int, np.integer)) or isinstance(n_candidates, bool):
        raise TypeError(f"n_candidates must be an int, got {type(n_candidates).__name__}")
    if n_candidates < 0:
        raise ValueError(f"n_candidates must be non-negative, got {n_candidates}")
    k_max = int(k_max)
    n_valid = min(int(n_candidates), k_max)
    validity = np.zeros(k_max, dtype=bool)
    validity[:n_valid] = True
    return validity


def build_action_mask(n_candidates: int, k_max: int) -> np.ndarray:
    """The action mask for `Discrete(k_max + 1)`: slot validity plus reject.

    Slot `k_max` (reject) is always True, so the mask can never be all-False and
    MaskablePPO always has at least one legal action -- including when the operator has
    no feasible candidate at all.

    :param n_candidates: number of candidates available *before* truncation
    :param k_max: number of candidate slots, `K`
    :return: bool array of length k_max + 1
    """
    validity = candidate_slot_validity(n_candidates, k_max)
    mask = np.zeros(len(validity) + 1, dtype=bool)
    mask[:len(validity)] = validity
    mask[-1] = True  # reject is always legal
    assert mask.any(), "action mask must never be all-False"
    return mask
