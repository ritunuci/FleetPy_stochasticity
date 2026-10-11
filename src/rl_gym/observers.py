"""Observation builders for the Phase 1 minimal observation (§5.1).

Two observers, each producing a dict of named feature arrays and each reporting its own
output length. The env concatenates them and builds `observation_space` by summing those
lengths, so nothing anywhere hardcodes 28 (= 3*8 + 4). Phase 2 widens the feature set and
adds a position encoding; the space follows automatically.

Design rules this module is bound by:

- **Every normalization constant comes from `scenario_parameters`.** FleetPy quantities are
  raw seconds; `op_max_wait_time` is 1500 in the reference scenario and 4200 in the stale
  config pair, so a literal would silently mis-scale (trap 8).
- **`is_valid` comes from `spaces.candidate_slot_validity`**, never a fresh
  `k < len(candidates)`. That function is the one source of truth shared with the action
  mask, and two independent computations can drift with nothing raising (trap 12).
- **Padded slots are all-zero, and `VecNormalize` is the whole reason** (§5.1). A large
  sentinel would distort its per-feature running mean and variance in Phase 2 (§5.3); 0.0 is
  neutral there. It is *not* justified by what a naive `argmin` would pick: measured under
  P1.6, real `delta_cfv` on this scenario is **negative** — transformed range
  [−14.508, 0.000] over 436 decisions — so 0.0 padding is the **largest** value in the slot,
  not the smallest, and an unrestricted `argmin` would land on slot 0 anyway. P1.10's
  `is_valid` restriction is required regardless of that sign, because a padded slot holds
  padding rather than a cost; the sign is a property of this objective function and can change.
- **No implied-decline-probability feature**, in either phase. It is derivable from the
  offered wait and `scenario_parameters`, and is excluded from the state design by decision.
- **Every observer declares its features statically** via `feature_spec()`, and
  `output_length` is derived from that declaration rather than written out separately, so the
  two cannot disagree. `build_observation_layout` turns the specs into the name → slice map the
  env publishes and `policies.greedy_action` reads by name (P2.8). The specs are fixed at
  construction; the `observe()` dicts exist only after a decision, so they cannot be the source.
"""

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.misc.globals import (
    G_OP_MAX_WT,
    G_SIM_END_TIME,
    G_SIM_START_TIME,
    VRL_STATES,
)
from src.rl_gym.spaces import candidate_slot_validity, truncate_candidates

SECONDS_PER_DAY = 86400.0


class AbstractObserver:
    """Produces part of the observation vector.

    Subclasses implement `observe` and `output_length`. Keeping observers separate lets each
    be unit-tested for shape independently, and lets the env build the observation space by
    summing lengths rather than asserting a total.
    """

    def feature_spec(self) -> List[Tuple[str, int]]:
        """This observer's features as an ordered `[(name, length)]`, fixed at construction.

        The declaration, not the `observe()` dict, is the source of the layout map (P2.8): the
        dict only exists once a decision has happened, and the env needs the layout to build
        `observation_space`.
        """
        raise NotImplementedError

    def observe(self, fleetpy_module, ctx) -> Dict[str, np.ndarray]:
        """Return this observer's named feature arrays.

        Must return exactly the names `feature_spec()` declares, in that order, with those
        lengths -- asserted by `observe_vector` and pinned by a test.

        :param fleetpy_module: the fleet control instance (`RLPoolingIRSOnly`)
        :param ctx: the `PendingDecision` for the current decision epoch
        :return: dict of feature name -> float32 array
        """
        raise NotImplementedError

    @property
    def output_length(self) -> int:
        """Scalars this observer contributes, summed from `feature_spec()`.

        Derived rather than written out separately, so a feature added to the spec without a
        matching length change cannot slip through (P2.8).
        """
        return sum(length for _name, length in self.feature_spec())

    def observe_vector(self, fleetpy_module, ctx) -> np.ndarray:
        """`observe` flattened into one float32 vector, in the declared order."""
        parts = self.observe(fleetpy_module, ctx)
        spec = self.feature_spec()
        declared = [name for name, _ in spec]
        if list(parts.keys()) != declared:
            raise ValueError(
                f"{type(self).__name__}.observe returned {list(parts.keys())}, "
                f"feature_spec declares {declared} -- the layout map is built from the spec, "
                f"so the two must agree exactly, in order"
            )
        for name, length in spec:
            got = np.size(parts[name])
            if got != length:
                raise ValueError(
                    f"{type(self).__name__}.observe gave {name!r} length {got}, "
                    f"feature_spec declares {length}"
                )
        vec = np.concatenate([np.ravel(v) for v in parts.values()]).astype(np.float32)
        assert vec.shape == (self.output_length,), \
            f"{type(self).__name__} produced {vec.shape}, declared {self.output_length}"
        return vec


def build_observation_layout(observers: Sequence[AbstractObserver]) -> Dict[str, slice]:
    """Name -> slice over the concatenated observation, in the order the env concatenates.

    Built from `feature_spec()` at construction, so it is available before any decision has
    happened and `observation_space` can be derived from it. `policies.greedy_action` reads
    `delta_cfv` and `is_valid` through this rather than assuming `obs[K:2K]` (P2.8).

    Raises on a **duplicate feature name**: P2.9's request block and P2.10's candidate block
    both add features, and a collision would leave one of them unreachable by name while the
    vector still had exactly the right length -- a silent hole in the one mechanism this map
    exists to provide.
    """
    layout: Dict[str, slice] = {}
    offset = 0
    for observer in observers:
        for name, length in observer.feature_spec():
            if name in layout:
                raise ValueError(
                    f"duplicate observation feature {name!r}: already at "
                    f"{layout[name]}, declared again by {type(observer).__name__}. "
                    f"Feature names are the only way consumers address the vector (P2.8), "
                    f"so they must be unique across observers."
                )
            if length <= 0:
                raise ValueError(f"feature {name!r} declares length {length}; must be positive")
            layout[name] = slice(offset, offset + length)
            offset += length
    return layout


def signed_log1p(x) -> np.ndarray:
    """`sign(x) * log1p(|x|)` — compresses the `delta_cfv` range without a scaling constant.

    Monotonic in x, which P1.10 depends on: the scripted greedy driver takes the argmin over
    this transform and must land on the same candidate as an argmin over raw `delta_cfv`.
    """
    arr = np.asarray(x, dtype=np.float64)
    return np.sign(arr) * np.log1p(np.abs(arr))


class CandidateObserver(AbstractObserver):
    """Per-candidate-slot features: 3 per slot, `K` slots (§5.1).

    1. `is_valid` — 1.0 if a candidate occupies the slot
    2. `delta_cfv`, transformed as `sign(x) * log1p(|x|)`
    3. offered wait under that plan, `pax_info[rid][0] - prq.rq_time`, over `op_max_wait_time`

    Feature 3 is exactly what `PoolingIRSOnly._create_user_offer` quotes to the rider: it
    reads the same `assigned_vehicle_plan.pax_info[rid_struct]` boarding time and subtracts
    the same `prq.rq_time`.
    """

    def __init__(self, k_max: int, scenario_parameters: dict):
        """
        :param k_max: `K`, owned by SDPDPAssignmentEnv and passed in — not parameterised here
        :param scenario_parameters: source of every normalization constant
        """
        self.k_max = int(k_max)
        if self.k_max <= 0:
            raise ValueError(f"k_max must be positive, got {k_max}")
        self.max_wait_time = float(scenario_parameters[G_OP_MAX_WT])
        if self.max_wait_time <= 0:
            raise ValueError(f"{G_OP_MAX_WT} must be positive, got {self.max_wait_time}")

    def feature_spec(self) -> List[Tuple[str, int]]:
        # feature-major, as §5.2 keeps it: each feature is a length-K vector. Phase 2 appends
        # features 4-12 here (P2.10) without disturbing these three.
        return [("is_valid", self.k_max),
                ("delta_cfv", self.k_max),
                ("offered_wait", self.k_max)]

    def observe(self, fleetpy_module, ctx) -> Dict[str, np.ndarray]:
        candidates = truncate_candidates(ctx.candidates, self.k_max)
        # one source of truth, shared with the action mask (trap 12)
        is_valid = candidate_slot_validity(len(ctx.candidates), self.k_max)

        delta_cfv = np.zeros(self.k_max, dtype=np.float64)
        offered_wait = np.zeros(self.k_max, dtype=np.float64)
        rid_struct = ctx.rid_struct
        rq_time = ctx.prq.rq_time

        for k, (_vid, vehplan, cfv) in enumerate(candidates):
            delta_cfv[k] = cfv
            pax_info = vehplan.pax_info.get(rid_struct)
            if pax_info:
                # pax_info[rid] is [boarding_time, deboarding_time]; the offer uses [0]
                offered_wait[k] = (pax_info[0] - rq_time) / self.max_wait_time

        return {
            "is_valid": is_valid.astype(np.float32),
            "delta_cfv": signed_log1p(delta_cfv).astype(np.float32),
            "offered_wait": offered_wait.astype(np.float32),
        }


class GlobalStateObserver(AbstractObserver):
    """Fleet- and time-level features: 4 scalars (§5.1).

    1. `sin(2π · t_of_day)`   — absolute clock time, `(sim_time mod 86400) / 86400`
    2. `cos(2π · t_of_day)`
    3. episode progress, `(sim_time - start_time) / (end_time - start_time)`
    4. fraction of active vehicles idle, `IDLE / (not OUT_OF_SERVICE)`

    Features 1-2 and 3 are different quantities: the first pair is where the day sits on a
    24-hour clock, the third is the horizon signal D6 needs. The circular encoding is
    convention here — the reference day never wraps midnight.

    Feature 4 is deliberately the coarse notion. `simple_insert` also skips vehicles with
    `no_show_event`, so the genuinely-available fleet is narrower; §5.2 carries the proper
    availability count.
    """

    def __init__(self, scenario_parameters: dict):
        self.start_time = float(scenario_parameters[G_SIM_START_TIME])
        self.end_time = float(scenario_parameters[G_SIM_END_TIME])
        span = self.end_time - self.start_time
        if span <= 0:
            raise ValueError(
                f"{G_SIM_END_TIME} must exceed {G_SIM_START_TIME}, got "
                f"{self.end_time} and {self.start_time}"
            )
        self._span = span

    def feature_spec(self) -> List[Tuple[str, int]]:
        return [("time_sin", 1), ("time_cos", 1), ("day_progress", 1), ("idle_fraction", 1)]

    def observe(self, fleetpy_module, ctx) -> Dict[str, np.ndarray]:
        sim_time = float(ctx.sim_time)

        t_of_day = (sim_time % SECONDS_PER_DAY) / SECONDS_PER_DAY
        angle = 2.0 * math.pi * t_of_day
        # clipped because record_remaining_assignments runs past end_time; no decision epoch
        # occurs there today, but an out-of-range feature would be silent if one ever did
        progress = min(max((sim_time - self.start_time) / self._span, 0.0), 1.0)

        n_active = 0
        n_idle = 0
        for veh_obj in fleetpy_module.sim_vehicles:
            # enum comparison, never `== 5`: VRL_STATES holds (int, str) tuples and defines no
            # __eq__ against ints, which is why the upstream `== 5` tests never fire (§5.1)
            if veh_obj.status == VRL_STATES.OUT_OF_SERVICE:
                continue
            n_active += 1
            if veh_obj.status == VRL_STATES.IDLE:
                n_idle += 1
        idle_fraction = (n_idle / n_active) if n_active > 0 else 0.0

        return {
            "time_sin": np.float32(math.sin(angle)),
            "time_cos": np.float32(math.cos(angle)),
            "day_progress": np.float32(progress),
            "idle_fraction": np.float32(idle_fraction),
        }
