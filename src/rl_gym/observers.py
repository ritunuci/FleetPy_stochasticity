"""Observation builders (§5.1 through §5.2).

Three observers, each producing a dict of named feature arrays and each reporting its own
output length. The env concatenates them in the order candidate, request, global (§5.2) and
builds `observation_space` by summing those lengths, so nothing anywhere hardcodes a total.
At `K = 8` the length is **139** after P2.11 — 96 candidate + 24 request + 19 global — which
completes §5.2's table.

Design rules this module is bound by:

- **Observation code never mutates simulation state** (trap 14). Stored vehicle plans are read;
  re-timing happens on copies; no `PlanRequest` field is ever written; and the routing cache is
  left exactly as it was found, because until the §2.5 travel-time defect is fixed an entry
  created here changes what later planning reads. `CandidateObserver._restore_cache` and
  `_retimed_copy` carry the detail, including §5.2's one structural exception.
- **Nothing on this path draws from a random stream.** Every stochastic component of the
  simulator draws from the one global `np.random` stream, so consuming a single draw here would
  shift every later outcome and fail the P1.10 gate for a reason unrelated to the feature. The
  only RNG in the routing stack is the stochastic travel-time redraw, and it fires solely from
  `update_network`, which observation code never calls.

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

import bisect
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.misc.globals import (
    G_ACT_VEH_SHARE,
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


def layout_fingerprint(layout: Dict[str, slice]) -> str:
    """SHA-256 over `<name>:<length>` lines of the layout, in order. **The only definition.**

    What it covers: every feature's name and width, and their order. What it deliberately does
    **not** cover: what a feature *means*. P2.18 redefined four features without touching a name
    or a width, so this value is unchanged by it and `baseline_hash` is what catches that; this
    catches a widening, a reordering or a rename — P2.19's removal of the 10-wide `month`
    one-hot, for instance.

    It is a hash rather than a recorded list because 44 names cannot be reported field by field
    in a refusal message; every *scalar* part of the observation contract gets its own recorded
    field instead, so a refusal can quote the recorded and current values (P2.18).
    """
    import hashlib
    text = "\n".join(f"{name}:{sl.stop - sl.start}" for name, sl in layout.items())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def signed_log1p(x) -> np.ndarray:
    """`sign(x) * log1p(|x|)` — compresses the `delta_cfv` range without a scaling constant.

    Monotonic in x, which P1.10 depends on: the scripted greedy driver takes the argmin over
    this transform and must land on the same candidate as an argmin over raw `delta_cfv`.
    """
    arr = np.asarray(x, dtype=np.float64)
    return np.sign(arr) * np.log1p(np.abs(arr))


def operator_plan_request(fleetpy_module, rid):
    """The operator's `PlanRequest` for `rid`, or `None`.

    `rq_dict_store` first: unlike `rq_dict` it keeps its entries for the whole episode, so a
    rider who has already left the system is still readable.
    """
    prq = fleetpy_module.rq_dict_store.get(rid)
    if prq is None:
        prq = fleetpy_module.rq_dict.get(rid)
    return prq


def quoted_waiting_time(fleetpy_module, rid) -> Optional[float]:
    """The wait the operator quoted rider `rid`, in seconds, or `None`.

    The quote lives on the operator's `PlanRequest` as `prq.offer.offered_waiting_time`, set
    once by `_create_user_offer` as `pu_time - prq.rq_time` — already relative to `rq_time`.
    Read from `rq_dict_store`, which unlike `rq_dict` keeps its entries for the whole episode.
    **Nothing from the diffusion model is read** (§5.2): not `rq_obj.expected_pickup_time`, not
    `ETA_update_time_stamp`, not any `diffusion_state_*` — the operator has no knowledge of the
    behavioural model. A rejected request carries a `Rejection`, whose `offered_waiting_time` is
    `None`; such a rider is never a pending rider, but the `None` is handled rather than assumed
    away.

    One definition, used by both P2.10's lateness features here and the wait gap P2.7's
    statistics tracker records through `gym_env._make_quoted_wait_fn`. Two copies of this lookup
    would be the same drift trap 13 rules out for the exit classifier.
    """
    prq = operator_plan_request(fleetpy_module, rid)
    offer = getattr(prq, "offer", None) if prq is not None else None
    return getattr(offer, "offered_waiting_time", None)


def quoted_pickup_time(fleetpy_module, rid) -> Optional[float]:
    """The pickup time the operator promised rider `rid`, in simulation seconds, or `None`.

    §5.2's "quoted pickup": `prq.rq_time` plus the quoted wait. Both come off the same
    `PlanRequest`, so the sum cannot mix one rider's request time with another's quote.
    """
    prq = operator_plan_request(fleetpy_module, rid)
    offer = getattr(prq, "offer", None) if prq is not None else None
    quoted = getattr(offer, "offered_waiting_time", None)
    if quoted is None:
        return None
    return prq.rq_time + quoted


def route_distance(routing_engine, start_pos, plan) -> float:
    """Remaining route distance in metres from `start_pos` through `plan`'s stops.

    Exactly the distance term of `distance_and_user_times_with_walk`
    (`objectives.py`): `return_travel_costs_1to1(last_pos, pos)[2]` summed over the stops,
    skipping consecutive equal positions. Computed the same way for both plans so their
    difference is `added_distance` and nothing else.

    An observation, never a reward term (§10). It routes, which is why every caller sits inside
    `CandidateObserver`'s cache restore (§5.2 exception b).
    """
    total = 0.0
    last_pos = start_pos
    for ps in plan.list_plan_stops:
        pos = ps.get_pos()
        if pos != last_pos:
            total += routing_engine.return_travel_costs_1to1(last_pos, pos)[2]
            last_pos = pos
    return total


def unit_vector(from_coord, to_coord) -> np.ndarray:
    """`(to - from)` normalised to unit length, or `[0, 0]` when the two coincide.

    The normalisation of `SelectRV.filter_directionality` (lines ~15-38), re-implemented
    rather than called: the heuristic behind that function is off in this scenario and it
    returns a filtered vehicle list, not scores (§5.2).
    """
    diff = np.asarray(to_coord, dtype=np.float64) - np.asarray(from_coord, dtype=np.float64)
    norm = math.sqrt(float(np.dot(diff, diff)))
    if norm == 0.0:
        return np.zeros(2, dtype=np.float64)
    return diff / norm


class CandidateObserver(AbstractObserver):
    """Per-candidate-slot features: 12 per slot, `K` slots (§5.2).

    | # | feature | what it says | scaling |
    |---|---|---|---|
    | 1 | `is_valid` | a candidate occupies this slot | — |
    | 2 | `delta_cfv` | change in the control objective | `sign·log1p` |
    | 3 | `offered_wait` | the wait this plan would quote the new rider | ÷ `wait_normalizer_s` |
    | 4-5 | `veh_lat`, `veh_lon` | where the vehicle is now | [-1, 1] over the node hull |
    | 6 | `free_seats` | spare capacity | ÷ `max_pax` |
    | 7 | `remaining_stops` | stops in the committed plan | raw count |
    | 8 | `directionality` | cosine between request and vehicle direction | [-1, 1] |
    | 9 | `lateness_pre` | how late the vehicle already runs against its promises | ÷ `wait_normalizer_s` |
    | 10 | `lateness_post` | the worst promise after this insertion | ÷ `wait_normalizer_s` |
    | 11 | `added_distance` | route distance this insertion adds | km |
    | 12 | `added_pickup_delay` | delay it pushes onto riders already waiting | ÷ `wait_normalizer_s` |

    Features 4-9 describe the vehicle as it is, independently of the decision; 10-12 describe
    what inserting this request into plan `k` does. Riders already onboard can no longer cancel
    and the reward carries no ride-time term (§10), so only **pending** riders count.

    **Signs are kept, nothing is clipped** (§5.2). `lateness_*` is negative when a rider will be
    picked up ahead of the quote, which is common while the planner assumes 300 s per stop
    against a realized mean near 56 s; `added_distance` and `added_pickup_delay` can be slightly
    negative because routes minimise time rather than distance and stops can merge.

    Feature 3 is exactly what `PoolingIRSOnly._create_user_offer` quotes to the rider: it
    reads the same `assigned_vehicle_plan.pax_info[rid_struct]` boarding time and subtracts
    the same `prq.rq_time`.

    Two rules this class exists under:

    - **It never mutates simulation state** (trap 14). `veh_plans[vid]` is read; the re-timing
      runs on a `copy()`, with `keep_feasible` and `init_plan_state` left at their defaults so
      neither `set_infeasible_locked` nor the `pax_info` carry-over branch is reached. No
      `PlanRequest` field is written. The one write that does reach the stored plan is §5.2's
      exception (a), described on `_retimed_copy`.
    - **It leaves the routing cache exactly as it found it** (§5.2 exception b), because until
      the §2.5 travel-time defect is fixed a cache entry created now changes what later planning
      reads. See `_restore_cache`.
    """

    #: the two position encodings §5.2 allows. The length is the same either way, and so are
    #: the feature names, so the layout map does not move.
    POSITION_ENCODINGS = ("latlon", "relative")

    def __init__(self, k_max: int, scenario_parameters: dict,
                 node_coords: "NodeCoordinates",
                 wait_normalizer_s: Optional[float] = None,
                 position_encoding: str = "latlon"):
        """
        :param k_max: `K`, owned by SDPDPAssignmentEnv and passed in — not parameterised here
        :param scenario_parameters: source of every normalization constant
        :param node_coords: node lat/lon and the hull they scale against (§2.10)
        :param wait_normalizer_s: divisor for every wait-type feature. **Absent ->
            `op_max_wait_time`**, which reproduces Phase 1's `offered_wait` bit for bit (§5.3).
            Set it explicitly once `op_max_wait_time` is relaxed, so that relaxing the
            operator's bound does not silently rescale every wait feature.
        :param position_encoding: `"latlon"` (default) or `"relative"`, which replaces the
            vehicle's absolute position with its offset from the request origin
        """
        self.k_max = int(k_max)
        if self.k_max <= 0:
            raise ValueError(f"k_max must be positive, got {k_max}")
        self.max_wait_time = float(scenario_parameters[G_OP_MAX_WT])
        if self.max_wait_time <= 0:
            raise ValueError(f"{G_OP_MAX_WT} must be positive, got {self.max_wait_time}")
        # the fallback is op_max_wait_time, so Phase 1's values do not change (§5.3)
        self.wait_normalizer_s = (self.max_wait_time if wait_normalizer_s is None
                                  else float(wait_normalizer_s))
        if self.wait_normalizer_s <= 0:
            raise ValueError(f"wait_normalizer_s must be positive, got {wait_normalizer_s}")
        if position_encoding not in self.POSITION_ENCODINGS:
            raise ValueError(
                f"position_encoding must be one of {self.POSITION_ENCODINGS}, got "
                f"{position_encoding!r}. The earlier \"h3\" option is removed (§5.2): a raw cell "
                f"id is not a usable network input and a one-hot would add 37 dimensions per "
                f"position."
            )
        self.position_encoding = position_encoding
        self.coords = node_coords
        #: cumulative diagnostics, for the throughput and cache reporting P2.10 owes. Counting
        #: only; nothing reads them to make a decision.
        self.diagnostics = {
            "observations": 0,          # calls to observe()
            "retimings": 0,             # pre-insertion copies re-timed
            "cache_entries_restored": 0,  # entries this observer added and then deleted, which
                                        # is also the number of genuine route computations it
                                        # triggered: _add_to_database writes exactly on one
            "riders_without_plan_time": 0,  # pending riders with no pax_info entry after
                                            # re-timing (the no-show skip in the re-timer)
            "riders_without_quote": 0,  # pending riders with no operator quote to compare to
        }

    #: (name, how many slots) in vector order. Feature-major, as §5.2 keeps it: each feature is
    #: a length-K vector, and the three Phase 1 features keep their positions, so `delta_cfv` is
    #: still at `obs[K:2K]` -- which nothing may depend on, since P2.8's layout map is the
    #: contract and every consumer reads by name.
    FEATURE_NAMES = ("is_valid", "delta_cfv", "offered_wait",
                     "veh_lat", "veh_lon", "free_seats", "remaining_stops", "directionality",
                     "lateness_pre", "lateness_post", "added_distance", "added_pickup_delay")

    def feature_spec(self) -> List[Tuple[str, int]]:
        return [(name, self.k_max) for name in self.FEATURE_NAMES]

    # ---------------------------------------------------------------- #
    # the two rules
    # ---------------------------------------------------------------- #

    @staticmethod
    def _cache_keys(routing_engine) -> Optional[frozenset]:
        """The routing cache's keys right now, or `None` if this engine has no cache.

        Read through the attribute every time and **never held across calls**:
        `NetworkBasicWithStore._reset_internal_attributes_after_travel_time_update` rebinds
        `travel_time_infos` to a fresh dict rather than clearing it, and `load_tt_file` does
        that at every operator travel-time file load (07:00, 09:00, 12:00, 16:00, 18:00 on the
        reference day). A held reference silently goes stale from the first load onwards.
        """
        cache = getattr(routing_engine, "travel_time_infos", None)
        return None if cache is None else frozenset(cache)

    def _restore_cache(self, routing_engine, before: Optional[frozenset]) -> None:
        """Delete every key this observation added, leaving the cache exactly as it was.

        Exact without duplicating the engine's key derivation, because `_add_to_database` writes
        only when the key is absent (`NetworkBasicWithStore.py:151`) and so can never overwrite
        an entry the simulation already owns: the added set is precisely `after - before`.

        Why it matters: until the §2.5 defect is fixed, `Router` runs on the live realized
        travel-time draw, so an entry the observer leaves behind changes what later planning
        reads, and the P1.10 gate fails. Once that defect is fixed this is unnecessary but
        harmless.

        `travel_time_infos` is the **only** cache in play: it is the one store
        `NetworkBasicWithStore` declares, its two writers are `_add_to_database` and
        `add_travel_infos_to_database` (which takes externally computed multiprocessing results
        and is unreachable from here), and on `NetworkBasic` both the store hook and the reset
        hook are `pass`. A network type with no such cache has nothing to restore.
        """
        if before is None:
            return
        cache = getattr(routing_engine, "travel_time_infos", None)
        if cache is None:
            return
        added = [key for key in cache if key not in before]
        for key in added:
            del cache[key]
        self.diagnostics["cache_entries_restored"] += len(added)

    def _retimed_copy(self, veh_obj, pre_plan, sim_time, routing_engine):
        """A re-timed copy of the vehicle's committed plan, for features 9 and 12.

        Stored plan times refresh only at leg events, because `update_sim_state_fleets` runs
        with `force_update_plan=False`, so after a stochastic travel-time update they are stale
        until the vehicle's next leg event. Candidate plans do not have this problem —
        `insertion.py` copies the plan and re-times the copy at decision time — and this does
        the same, with the identical call `simple_insert` makes
        (`update_tt_and_check_plan(veh_obj, sim_time, routing_engine)`, `insertion.py:98`).
        `keep_feasible` and `init_plan_state` stay at their defaults, so neither
        `set_infeasible_locked` nor the `pax_info` carry-over branch runs.

        **The stored plan is never re-timed, reassigned or edited** (trap 14). One copy per
        candidate vehicle per decision, shared by both features.

        §5.2's exception (a) is the single write that does reach the stored plan, and it is
        structural rather than avoidable: `PlanStop.copy()` copies `boarding_dict` with a
        shallow `dict.copy()`, so the +1/-1 lists are shared, and the no-show branch of the
        re-timer (`VehiclePlan.py:897`, `boarding_dict[-1].remove(rid)`) reaches the stored
        plan's alighting list through the copy. It can do nothing new here: `simple_insert`
        copies this same plan and makes this same call for every candidate vehicle at this same
        decision, and a candidate vehicle is by definition one that got through `simple_insert`,
        so the rid is already gone and `.remove` lands in the re-timer's `except` (which only
        logs). The P1.10 gate decides.
        """
        retimed = pre_plan.copy()
        retimed.update_tt_and_check_plan(veh_obj, sim_time, routing_engine)
        self.diagnostics["retimings"] += 1
        return retimed

    # ---------------------------------------------------------------- #
    # features
    # ---------------------------------------------------------------- #

    def _scaled_position(self, pos) -> Tuple[float, float]:
        """Vehicle lat/lon on [-1, 1] over the node hull, interpolated along its edge.

        `veh_obj.pos` is `(node, None, None)` at a node or `(o_node, d_node, rel_pos)` on an
        edge, where `rel_pos` is the fraction **already travelled** — verified against
        `return_position_coordinates`, which forms `rel * c_dest + (1 - rel) * c_origin`
        (`NetworkBasic.py:410`), and against `get_section_overhead`, which documents
        `position[2]` as the already-travelled share. Interpolating the two nodes' *scaled*
        coordinates is identical to scaling the interpolated ones, since the hull scaling is
        linear.
        """
        o_lat, o_lon = self.coords.scaled(pos[0])
        if pos[1] is None:
            return o_lat, o_lon
        d_lat, d_lon = self.coords.scaled(pos[1])
        rel = float(pos[2])
        return o_lat + rel * (d_lat - o_lat), o_lon + rel * (d_lon - o_lon)

    @staticmethod
    def _pending_rids(pre_plan, veh_obj, new_rid) -> List[Any]:
        """Riders with a boarding entry in the committed plan who are not yet onboard.

        §5.2's definition, and the new rider is never pending — it is not in the pre-insertion
        plan at all, but the exclusion is explicit so a later reader does not have to derive it.

        A rider whose boarding is **in progress** counts as onboard, not pending: a rider is
        appended to `veh_obj.pax` when the boarding leg starts (`Vehicles.py:254`), which is
        exactly why `get_nr_pax_without_currently_boarding` exists. So such a rider already
        occupies a seat in feature 6 and is not counted again in features 9, 10 and 12.
        """
        onboard = {rq.get_rid_struct() for rq in veh_obj.pax}
        pending = []
        for ps in pre_plan.list_plan_stops:
            for rid in ps.get_list_boarding_rids():
                if rid not in onboard and rid != new_rid:
                    pending.append(rid)
        # a rider boards at one stop, but dict.fromkeys keeps this idempotent and ordered
        return list(dict.fromkeys(pending))

    def observe(self, fleetpy_module, ctx) -> Dict[str, np.ndarray]:
        candidates = truncate_candidates(ctx.candidates, self.k_max)
        # one source of truth, shared with the action mask (trap 12)
        is_valid = candidate_slot_validity(len(ctx.candidates), self.k_max)

        out = {name: np.zeros(self.k_max, dtype=np.float64)
               for name in self.FEATURE_NAMES}
        out["is_valid"] = is_valid.astype(np.float64)

        prq = ctx.prq
        rid_struct = ctx.rid_struct
        rq_time = prq.rq_time
        sim_time = ctx.sim_time
        routing_engine = fleetpy_module.routing_engine
        wait_norm = self.wait_normalizer_s

        # the request's own direction, once per decision rather than once per candidate. In
        # pos_x/pos_y metres, never lat/lon, which would skew every angle by cos 37.35 ~ 0.79
        prq_vec = unit_vector(routing_engine.return_position_coordinates(prq.o_pos),
                              routing_engine.return_position_coordinates(prq.d_pos))
        origin_scaled = (self.coords.scaled(prq.o_pos[0])
                         if self.position_encoding == "relative" else None)

        cache_before = self._cache_keys(routing_engine)
        try:
            for k, (vid, vehplan, cfv) in enumerate(candidates):
                veh_obj = fleetpy_module.sim_vehicles[vid]
                pre_plan = fleetpy_module.veh_plans[vid]

                out["delta_cfv"][k] = cfv
                pax_info = vehplan.pax_info.get(rid_struct)
                if pax_info:
                    # pax_info[rid] is [boarding_time, deboarding_time]; the offer uses [0]
                    out["offered_wait"][k] = (pax_info[0] - rq_time) / wait_norm

                veh_lat, veh_lon = self._scaled_position(veh_obj.pos)
                if origin_scaled is not None:
                    # the difference of two hull-scaled values is already the offset divided by
                    # the bbox half-span, which is what §5.2 asks for
                    veh_lat -= origin_scaled[0]
                    veh_lon -= origin_scaled[1]
                out["veh_lat"][k] = veh_lat
                out["veh_lon"][k] = veh_lon

                # passengers, not rider objects (§5.2). max_parcels is a separate capacity and
                # no parcels occur in this scenario.
                occupied = sum(rq.nr_pax for rq in veh_obj.pax)
                out["free_seats"][k] = (veh_obj.max_pax - occupied) / veh_obj.max_pax
                # pre-insertion, raw count. With repositioning off, 0 means idle, which is what
                # separates idle vehicles from perpendicular ones in `directionality`.
                out["remaining_stops"][k] = len(pre_plan.list_plan_stops)

                if veh_obj.assigned_route:
                    # the committed route, so this is pre-insertion by construction
                    veh_vec = unit_vector(
                        routing_engine.return_position_coordinates(veh_obj.pos),
                        routing_engine.return_position_coordinates(
                            veh_obj.assigned_route[-1].destination_pos))
                else:
                    veh_vec = np.zeros(2, dtype=np.float64)
                out["directionality"][k] = float(np.dot(prq_vec, veh_vec))

                dist_pre = route_distance(routing_engine, veh_obj.pos, pre_plan)
                dist_post = route_distance(routing_engine, veh_obj.pos, vehplan)
                out["added_distance"][k] = (dist_post - dist_pre) / 1000.0

                pending = self._pending_rids(pre_plan, veh_obj, rid_struct)
                if not pending:
                    # no pending riders -> 0.0 for features 9, 10 and 12 (§5.2), and no re-timed
                    # copy is needed, which is also the cheap path
                    continue
                retimed = self._retimed_copy(veh_obj, pre_plan, sim_time, routing_engine)
                late_pre, late_post, added_delay = [], [], 0.0
                for rid in pending:
                    planned_pre = retimed.pax_info.get(rid)
                    planned_post = vehplan.pax_info.get(rid)
                    if not planned_pre or not planned_post:
                        # the re-timer skips a boarding rid that is in no_shows_store_for_veh,
                        # so an entry can legitimately be missing; counted, never guessed at
                        self.diagnostics["riders_without_plan_time"] += 1
                        continue
                    added_delay += planned_post[0] - planned_pre[0]
                    quoted_pickup = quoted_pickup_time(fleetpy_module, rid)
                    if quoted_pickup is None:
                        self.diagnostics["riders_without_quote"] += 1
                        continue
                    late_pre.append(planned_pre[0] - quoted_pickup)
                    late_post.append(planned_post[0] - quoted_pickup)
                if late_pre:
                    out["lateness_pre"][k] = max(late_pre) / wait_norm
                    out["lateness_post"][k] = max(late_post) / wait_norm
                out["added_pickup_delay"][k] = added_delay / wait_norm
        finally:
            # a finally, so a feature bug cannot leave the simulation's routing cache polluted
            self._restore_cache(routing_engine, cache_before)

        self.diagnostics["observations"] += 1
        out["delta_cfv"] = signed_log1p(out["delta_cfv"])
        return {name: out[name].astype(np.float32) for name in self.FEATURE_NAMES}


class NodeCoordinates:
    """Node lat/lon and the hull they are scaled against (§5.2, §2.10).

    Read from the `lat`/`lon` columns of `base/nodes.csv`, **not** from
    `routing_engine.return_network_bounding_box()**. P2.1 measured why: that helper reprojects
    the corners of the UTM *rectangle*, and because the grid is turned about 0.5-0.7° from true
    north those corners fall **inside** the node latitude hull. Scaling by them puts **13 of the
    4,084 nodes outside [-1, 1]**, reaching -1.0106 and +1.0222. The node hull puts the extremes
    exactly at +/-1 and nothing outside. (It is also cheaper: `return_positions_lon_lat` builds a
    new pyproj transformer on every call.)

    Static data, loaded once at env construction and shared across episodes.
    """

    def __init__(self, nodes_path: Optional[str] = None,
                 network_name: str = "example_network"):
        import pandas as pd
        path = nodes_path or os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
            "data", "networks", network_name, "base", "nodes.csv")
        if not os.path.isfile(path):
            raise ValueError(f"no node file at {path}")
        df = pd.read_csv(path, usecols=["node_index", "lat", "lon"])
        if df[["lat", "lon"]].isna().any().any():
            raise ValueError(f"{path} has null lat/lon values")
        self.path = path
        self._lat = {int(n): float(v) for n, v in zip(df["node_index"], df["lat"])}
        self._lon = {int(n): float(v) for n, v in zip(df["node_index"], df["lon"])}
        self.lat_min, self.lat_max = float(df["lat"].min()), float(df["lat"].max())
        self.lon_min, self.lon_max = float(df["lon"].min()), float(df["lon"].max())
        for name, lo, hi in (("lat", self.lat_min, self.lat_max),
                             ("lon", self.lon_min, self.lon_max)):
            if not hi > lo:
                raise ValueError(f"{path}: degenerate {name} hull, {lo} to {hi}")

    def __len__(self) -> int:
        return len(self._lat)

    def scaled(self, node) -> Tuple[float, float]:
        """`(lat, lon)` of `node`, each mapped linearly onto [-1, 1] over the node hull.

        Raises on an unknown node rather than returning a sentinel: a request origin the network
        does not contain means the demand and the network disagree, and 0.0 would read as the
        middle of the map.
        """
        key = int(node)
        try:
            lat, lon = self._lat[key], self._lon[key]
        except KeyError:
            raise KeyError(
                f"node {node!r} is not in {os.path.basename(self.path)} ({len(self)} nodes)"
            ) from None
        return (2.0 * (lat - self.lat_min) / (self.lat_max - self.lat_min) - 1.0,
                2.0 * (lon - self.lon_min) / (self.lon_max - self.lon_min) - 1.0)


class RequestObserver(AbstractObserver):
    """The request block: 24 features about the rider being decided on (§5.2).

    | feature | source | scaling |
    |---|---|---|
    | `o_lat`, `o_lon`, `d_lat`, `d_lon` | `prq.o_pos[0]`, `prq.d_pos[0]` | [-1, 1], node hull |
    | `n_pax` | `prq.nr_pax` | raw |
    | `dow` (3) | the demand filename | one-hot over Tue/Wed/Thu |
    | `month` (10) | the demand filename | one-hot over Jan-Oct |
    | `zone_arrivals_30/60`, `zone_declines_30/60`, `zone_cancels_30/60` | P2.7 | raw counts |

    **The vocabularies come from P2.6's manifest, never literals** (§5.2), and an unknown
    weekday or month raises -- at `set_episode_day`, i.e. at `reset()`, rather than at the first
    decision, so a misconfigured episode fails before it runs rather than 40 seconds in.

    **Zone set** = the request's origin H3 cell plus its neighbours, from P2.5's table; events
    are keyed to the event rider's origin cell (§5.2, P2.7).

    Two per-episode dependencies are injected rather than discovered: the rolling-stats tracker
    (rebuilt every reset, §10) and the episode's weekday and month. Both have explicit setters
    the env calls in `reset()`, and `observe` raises if either is missing -- reaching into
    `fleetpy_module._stats_tracker` would work but couples this to a private attribute of the
    operator and makes a stub tracker awkward to inject.
    """

    #: the two trailing windows §5.2 asks for, in seconds
    WINDOWS_S = (1800, 3600)
    #: (feature stem, the tracker's event kind)
    _ZONE_KINDS = (("zone_arrivals", "arrival"),
                   ("zone_declines", "decline"),
                   ("zone_cancels", "cancel"))

    def __init__(self, node_coords: NodeCoordinates, zone_table,
                 weekday_vocab: Sequence[str], month_vocab: Sequence[int]):
        """
        :param weekday_vocab: from P2.6's manifest, e.g. `["Tue", "Wed", "Thu"]`
        :param month_vocab: from P2.6's manifest, e.g. `[1, ..., 10]`
        """
        self.coords = node_coords
        self.zones = zone_table
        self.weekday_vocab = list(weekday_vocab)
        self.month_vocab = [int(m) for m in month_vocab]
        if not self.weekday_vocab or not self.month_vocab:
            raise ValueError("the weekday and month vocabularies must both be non-empty; "
                             "they come from P2.6's manifest (§5.2)")
        if len(set(self.weekday_vocab)) != len(self.weekday_vocab) or \
                len(set(self.month_vocab)) != len(self.month_vocab):
            raise ValueError("the vocabularies must not repeat a value")
        self._stats = None
        self._dow_index: Optional[int] = None
        self._month_index: Optional[int] = None

    # ---------------------------------------------------------------- #
    # per-episode injection
    # ---------------------------------------------------------------- #

    def set_stats_tracker(self, tracker) -> None:
        """Attach P2.7's tracker for this episode. Called from `reset()`."""
        self._stats = tracker

    def set_episode_day(self, weekday: Optional[str], month: Optional[int]) -> None:
        """Fix this episode's one-hots, validating against the manifest vocabularies.

        Raises here rather than at the first decision: `None` is what P2.3's `parse_day_stem`
        returns for a demand filename that is not an ISO date, and §5.2 requires an unknown
        weekday or month to raise -- P2.9 is the item that owes that.
        """
        if weekday not in self.weekday_vocab:
            raise ValueError(
                f"weekday {weekday!r} is not in the manifest vocabulary {self.weekday_vocab}. "
                f"It is parsed from the demand filename, which is the only record of which day "
                f"a file is (§2.9); `None` means the filename is not an ISO date."
            )
        if month not in self.month_vocab:
            raise ValueError(
                f"month {month!r} is not in the manifest vocabulary {self.month_vocab}. "
                f"Parsed from the demand filename (§2.9); `None` means it is not an ISO date."
            )
        self._dow_index = self.weekday_vocab.index(weekday)
        self._month_index = self.month_vocab.index(int(month))

    def feature_spec(self) -> List[Tuple[str, int]]:
        spec: List[Tuple[str, int]] = [
            ("o_lat", 1), ("o_lon", 1), ("d_lat", 1), ("d_lon", 1), ("n_pax", 1),
            ("dow", len(self.weekday_vocab)), ("month", len(self.month_vocab)),
        ]
        # the one-hots are single named blocks rather than one name per level: the layout map
        # addresses blocks, and `dow` of width 3 is what §5.2's "dow_* (3)" means
        for stem, _kind in self._ZONE_KINDS:
            for window_s in self.WINDOWS_S:
                spec.append((f"{stem}_{window_s // 60}", 1))
        return spec

    def observe(self, fleetpy_module, ctx) -> Dict[str, np.ndarray]:
        if self._stats is None:
            raise RuntimeError(
                "RequestObserver has no rolling-stats tracker; the env attaches one in reset() "
                "(P2.7). Without it the six zone counts would silently be zero."
            )
        if self._dow_index is None or self._month_index is None:
            raise RuntimeError(
                "RequestObserver has no episode day; the env calls set_episode_day() in "
                "reset() (P2.9)."
            )

        prq = ctx.prq
        o_node, d_node = prq.o_pos[0], prq.d_pos[0]
        o_lat, o_lon = self.coords.scaled(o_node)
        d_lat, d_lon = self.coords.scaled(d_node)

        dow = np.zeros(len(self.weekday_vocab), dtype=np.float32)
        dow[self._dow_index] = 1.0
        month = np.zeros(len(self.month_vocab), dtype=np.float32)
        month[self._month_index] = 1.0

        out: Dict[str, np.ndarray] = {
            "o_lat": np.float32(o_lat),
            "o_lon": np.float32(o_lon),
            "d_lat": np.float32(d_lat),
            "d_lon": np.float32(d_lon),
            "n_pax": np.float32(prq.nr_pax),
            "dow": dow,
            "month": month,
        }
        for stem, kind in self._ZONE_KINDS:
            for window_s in self.WINDOWS_S:
                out[f"{stem}_{window_s // 60}"] = np.float32(
                    self._stats.zone_count(kind, window_s, o_node))
        return out


class ArrivalBaseline:
    """P2.6's arrival baseline, **weekday-conditioned since P2.18**, as the anomaly reads it.

    `expected(weekday, t, W)` is the mean arrival count of that weekday's training days — Tue 35,
    Wed 33, Thu 33 — in the window `[t − W, t)`; `sd(t, W)` is the spread of the
    **weekday**-centred residual, **pooled across weekdays** so it depends only on the minute and
    the window. The committed table therefore holds exactly one distinct `sd` per
    `(t_seconds, window_s)` across all three weekdays, and reading it from the episode's own
    weekday row is the same number (checked in P2.11 for the month table, re-checked in P2.18).

    **The sd is centred on the weekday curve, not a month curve** (§8 P2.6, "Why weekday, not
    month"). That is what stops the Tue/Wed/Thu spread being divided into `z` as if it were
    noise: at t = 55440 the sd is 7.620 rather than the month table's 10.148 at `W = 30`.

    The table's grid is the reference scenario's minutes, `[start_time, end_time)` — 747 of them
    — so a scenario with a different window would miss every lookup. `check_scenario` raises at
    construction rather than letting that surface as a `KeyError` mid-episode (guard 1).

    Static data, loaded once at env construction and shared across episodes. The manifest's
    `baseline_hash` is verified against these rows by the env, so a baseline rebuilt or edited
    without its manifest stops the run (trap 16).
    """

    def __init__(self, manifest: Dict[str, Any], baseline_path: Optional[str] = None):
        from src.rl_gym.corpus import DEFAULT_BASELINE, read_baseline, verify_baseline
        self.path = baseline_path or DEFAULT_BASELINE
        rows = read_baseline(self.path)
        # guard 2: the manifest records the baseline it was built beside (trap 16)
        verify_baseline(manifest, rows)
        self.rows = rows
        self._mean: Dict[Tuple[str, int, int], float] = {}
        self._sd: Dict[Tuple[int, int], float] = {}
        for weekday, t, window_s, mean, sd in rows:
            self._mean[(weekday, t, window_s)] = mean
            self._sd[(t, window_s)] = sd
        self.weekdays = sorted({w for w, _t, _win, _mu, _sd in rows})
        self.windows_s = sorted({win for _w, _t, win, _mu, _sd in rows})
        self.minutes = sorted({t for _w, t, _win, _mu, _sd in rows})
        self.sd_floor = float(manifest["sd_floor"])
        self.minute_s = int(manifest["minute_s"])
        if self.sd_floor <= 0:
            raise ValueError(f"sd_floor must be positive, got {self.sd_floor}")

    def check_scenario(self, start_time: float, end_time: float,
                       windows_s: Sequence[int]) -> None:
        """Raise unless this table covers the scenario's window and the observer's windows."""
        missing = [w for w in windows_s if w not in self.windows_s]
        if missing:
            raise ValueError(
                f"{os.path.basename(self.path)} holds no rows for window(s) {missing} s; it "
                f"covers {self.windows_s}. The baseline and the observation must use the same "
                f"windows or `expected` is not comparable with `observed` (§5.2); rebuild the "
                f"baseline with `python -m src.rl_gym.offline.build_arrival_baseline`."
            )
        grid_lo, grid_hi = self.minutes[0], self.minutes[-1]
        if float(start_time) < grid_lo or float(end_time) - self.minute_s > grid_hi:
            raise ValueError(
                f"the scenario runs [{start_time:.0f}, {end_time:.0f}) but "
                f"{os.path.basename(self.path)} covers minutes [{grid_lo}, {grid_hi}]. Every "
                f"decision would need a row that is not there, so the anomaly features would "
                f"fail mid-episode. Rebuild the baseline for this scenario's window."
            )

    def floor_minute(self, sim_time: float) -> int:
        """`sim_time` floored to the baseline's minute grid (§5.2)."""
        return int(sim_time // self.minute_s) * self.minute_s

    def expected(self, weekday: str, t: int, window_s: int) -> float:
        try:
            return self._mean[(str(weekday), int(t), int(window_s))]
        except KeyError:
            raise KeyError(
                f"no baseline row for weekday {weekday!r}, minute {t}, window {window_s} s in "
                f"{os.path.basename(self.path)}"
            ) from None

    def sd(self, t: int, window_s: int) -> float:
        try:
            return self._sd[(int(t), int(window_s))]
        except KeyError:
            raise KeyError(
                f"no baseline sd for minute {t}, window {window_s} s in "
                f"{os.path.basename(self.path)}"
            ) from None

    def z(self, observed: float, weekday: str, t: int, window_s: int) -> Tuple[float, float]:
        """`(expected, z)` with `z = (observed − expected) / max(sd, sd_floor)` (§5.2)."""
        exp = self.expected(weekday, t, window_s)
        return exp, (observed - exp) / max(self.sd(t, window_s), self.sd_floor)


class ActiveFleetSchedule:
    """The scheduled active fleet size over the day, from `op_act_fs_file` (§5.2).

    Read once at env construction from the same path `TimeBasedFS.add_init` uses
    (`dir_names[G_DIR_FCTRL]/elastic_fleet_size/<file>`), with the identical formula
    `min(max(int(np.round(share × nr_vehicles)), 0), nr_vehicles)` from
    `TimeBasedFS._read_active_vehicle_file`. **Read-only: nothing on the fleet-sizing module is
    touched** — and reading the file is the only route, because `TimeBasedFS` keeps no queryable
    schedule: its parsed table is local to that method and the activation lists it leaves behind
    are consumed as they fire.

    The share in force at `t` is the last row with `time <= t`, and the last row applies beyond
    the end of the file. `TimeBasedFS` additionally drops rows before `start_time` (keeping the
    last one before it), which changes nothing here: for any `t >= start_time` the last row at or
    below `t` is the same row either way, and the reference file's first row is exactly
    `start_time`.

    The rounding is why this is not simply the share: 0.7 × 12 = 8.4 rounds to 8, i.e. 0.667.
    `nr_vehicles` therefore has to be the fleet the operator actually holds, and it is read from
    `fleetpy_module.nr_vehicles` at observation time — the identical attribute `TimeBasedFS`
    multiplies by — rather than parsed out of `op_fleet_composition`.
    """

    def __init__(self, path: str):
        import pandas as pd
        if not os.path.isfile(path):
            raise ValueError(f"no active fleet size file at {path}")
        df = pd.read_csv(path, index_col=0)
        if G_ACT_VEH_SHARE not in df.columns:
            raise ValueError(f"{path} has no {G_ACT_VEH_SHARE!r} column, so the share in force "
                             f"cannot be read; columns are {list(df.columns)}")
        self.path = path
        pairs = sorted((int(t), float(s)) for t, s in df[G_ACT_VEH_SHARE].items())
        if not pairs:
            raise ValueError(f"{path} holds no rows")
        self.times = [t for t, _s in pairs]
        self.shares = [s for _t, s in pairs]

    def share_at(self, t: float) -> float:
        """The share in force at `t`: the last row with `time <= t`."""
        i = bisect.bisect_right(self.times, int(t)) - 1
        # before the first row the first share applies; the reference scenario cannot reach this
        # (its first row is exactly start_time), but a clamp beats an IndexError
        return self.shares[max(i, 0)]

    def active_fraction(self, t: float, nr_vehicles: int) -> float:
        """Scheduled active vehicles at `t`, over the fleet size.

        `TimeBasedFS`'s own formula, then divided by `nr_vehicles` as §5.2 asks. The division
        does not cancel the rounding, which is the point.
        """
        if nr_vehicles <= 0:
            return 0.0
        count = min(max(int(np.round(self.share_at(t) * nr_vehicles)), 0), nr_vehicles)
        return count / nr_vehicles


class GlobalStateObserver(AbstractObserver):
    """Fleet- and time-level features: 19 scalars (§5.2).

    | feature | what it says |
    |---|---|
    | `time_sin`, `time_cos` | where the day sits on a 24-hour clock |
    | `day_progress` | episode progress, the horizon signal D6 needs |
    | `idle_fraction` | fraction of active vehicles idle |
    | `arrivals_30/60` | fleet-wide arrivals in the window — also the anomaly's "observed" |
    | `arrivals_expected_30/60` | P2.6's baseline mean for this **weekday** at this minute |
    | `arrivals_z_30/60` | `(observed − expected) / max(sd, sd_floor)` |
    | `declines_30/60` | fleet-wide rider declines |
    | `cancels_30/60` | fleet-wide post-match cancellations |
    | `wait_gap_30/60` | mean realized-minus-quoted wait over riders picked up in the window |
    | `active_now/30/60` | scheduled active vehicles now, at +30 min and at +60 min |

    `time_sin`/`time_cos` and `day_progress` are **different quantities**: the first pair is
    absolute clock time, `(sim_time mod 86400) / 86400`, the third is
    `(sim_time − start_time) / (end_time − start_time)`. The circular encoding is convention
    here — the reference day never wraps midnight.

    `idle_fraction` is deliberately the coarse notion: `simple_insert` also skips vehicles with
    `no_show_event`, so the genuinely-available fleet is narrower. `active_*` is the scheduled
    count, a deterministic function of clock time, included as an explicit signal.

    The fleet-wide arrival count and the anomaly's observed count are **the same number and
    appear once** (§5.2). Counts are raw; the wait gap divides by `wait_normalizer_s`;
    `active_*` divides by the fleet size.

    Two per-episode dependencies are injected rather than discovered, as P2.9 does for the
    request block: the rolling-stats tracker (rebuilt every reset, §10) and the episode's
    **weekday**, which the baseline is conditioned on since P2.18. `set_episode_weekday`
    validates against the baseline's own weekdays and **raises at reset**, not at the first
    decision (guard 3).
    """

    #: §5.2's two trailing windows, in seconds. Checked against the baseline at construction,
    #: so these and P2.6's artifact cannot silently disagree (guard 1).
    WINDOWS_S = (1800, 3600)

    def __init__(self, scenario_parameters: dict,
                 arrival_baseline: ArrivalBaseline,
                 active_fleet: ActiveFleetSchedule,
                 wait_normalizer_s: Optional[float] = None):
        """
        :param arrival_baseline: P2.6's table. **Required**, not optional: an absent baseline
            could only mean zeros for `arrivals_expected_*` and `arrivals_z_*`, and a feature
            that silently reads 0.0 instead of raising is the failure mode this project keeps
            engineering out.
        :param active_fleet: the `op_act_fs_file` curve, required for the same reason
        :param wait_normalizer_s: divisor for `wait_gap_*`; absent -> `op_max_wait_time` (§5.3)
        """
        if arrival_baseline is None or active_fleet is None:
            raise ValueError(
                "GlobalStateObserver needs both an ArrivalBaseline and an ActiveFleetSchedule: "
                "without them 10 of its 19 features could only be zeros, which reads as real "
                "data (§5.2)."
            )
        self.start_time = float(scenario_parameters[G_SIM_START_TIME])
        self.end_time = float(scenario_parameters[G_SIM_END_TIME])
        span = self.end_time - self.start_time
        if span <= 0:
            raise ValueError(
                f"{G_SIM_END_TIME} must exceed {G_SIM_START_TIME}, got "
                f"{self.end_time} and {self.start_time}"
            )
        self._span = span
        self.baseline = arrival_baseline
        self.active_fleet = active_fleet
        max_wait = float(scenario_parameters[G_OP_MAX_WT])
        self.wait_normalizer_s = (max_wait if wait_normalizer_s is None
                                  else float(wait_normalizer_s))
        if self.wait_normalizer_s <= 0:
            raise ValueError(f"wait_normalizer_s must be positive, got {wait_normalizer_s}")
        # guard 1, at construction: the windows and the minute grid must cover this scenario
        self.baseline.check_scenario(self.start_time, self.end_time, self.WINDOWS_S)
        self._stats = None
        self._weekday: Optional[str] = None

    # ---------------------------------------------------------------- #
    # per-episode injection
    # ---------------------------------------------------------------- #

    def set_stats_tracker(self, tracker) -> None:
        """Attach P2.7's tracker for this episode. Called from `reset()`."""
        self._stats = tracker

    def set_episode_weekday(self, weekday: Optional[str]) -> None:
        """Fix the weekday the baseline is read at, validating against the table (guard 3).

        Raises at reset rather than at the first decision. Two cases it catches: `None`, which
        P2.3's `parse_day_stem` returns for a demand filename that is not an ISO date, and a
        name like `"Sat"`, which it returns for an ISO date outside the corpus's three weekdays
        (§2.9). Was `set_episode_month` through P2.17.
        """
        if weekday is None or str(weekday) not in self.baseline.weekdays:
            raise ValueError(
                f"weekday {weekday!r} has no rows in "
                f"{os.path.basename(self.baseline.path)}, which covers "
                f"{self.baseline.weekdays}. The weekday is parsed from the demand filename, the "
                f"only record of which day a file is (§2.9); `None` means it is not an ISO "
                f"date, and a name outside the three means the day is not a corpus weekday."
            )
        self._weekday = str(weekday)

    def feature_spec(self) -> List[Tuple[str, int]]:
        spec: List[Tuple[str, int]] = [
            ("time_sin", 1), ("time_cos", 1), ("day_progress", 1), ("idle_fraction", 1),
        ]
        # the arrival count is also the anomaly's observed value, so it appears once (§5.2)
        for stem in ("arrivals", "arrivals_expected", "arrivals_z", "declines", "cancels",
                     "wait_gap"):
            for window_s in self.WINDOWS_S:
                spec.append((f"{stem}_{window_s // 60}", 1))
        spec.extend([("active_now", 1), ("active_30", 1), ("active_60", 1)])
        return spec

    def observe(self, fleetpy_module, ctx) -> Dict[str, np.ndarray]:
        if self._stats is None:
            raise RuntimeError(
                "GlobalStateObserver has no rolling-stats tracker; the env attaches one in "
                "reset() (P2.7). Without it every rolling count would silently be zero."
            )
        if self._weekday is None:
            raise RuntimeError(
                "GlobalStateObserver has no episode weekday; the env calls "
                "set_episode_weekday() in reset() (P2.11, P2.18)."
            )
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

        out: Dict[str, np.ndarray] = {
            "time_sin": np.float32(math.sin(angle)),
            "time_cos": np.float32(math.cos(angle)),
            "day_progress": np.float32(progress),
            "idle_fraction": np.float32(idle_fraction),
        }

        minute = self.baseline.floor_minute(sim_time)
        arrivals = {}
        for window_s in self.WINDOWS_S:
            arrivals[window_s] = self._stats.count("arrival", window_s)
            out[f"arrivals_{window_s // 60}"] = np.float32(arrivals[window_s])
        # computed together per window, but emitted in the order feature_spec declares --
        # expected for both windows, then z for both. `observe_vector` raises on a mismatch
        # (P2.8), which is what caught this being interleaved on the first run.
        anomaly = {}
        for window_s in self.WINDOWS_S:
            anomaly[window_s] = self.baseline.z(arrivals[window_s], self._weekday, minute,
                                                window_s)
        for window_s in self.WINDOWS_S:
            out[f"arrivals_expected_{window_s // 60}"] = np.float32(anomaly[window_s][0])
        for window_s in self.WINDOWS_S:
            out[f"arrivals_z_{window_s // 60}"] = np.float32(anomaly[window_s][1])
        for stem, kind in (("declines", "decline"), ("cancels", "cancel")):
            for window_s in self.WINDOWS_S:
                out[f"{stem}_{window_s // 60}"] = np.float32(
                    self._stats.count(kind, window_s))
        for window_s in self.WINDOWS_S:
            gap = self._stats.wait_gap_mean(window_s)
            # None means the window held no priced pickup; §5.2 writes 0.0 for that
            out[f"wait_gap_{window_s // 60}"] = np.float32(
                0.0 if gap is None else gap / self.wait_normalizer_s)

        # nr_vehicles from the operator, the identical attribute TimeBasedFS multiplies by
        nr_vehicles = getattr(fleetpy_module, "nr_vehicles", 0) or 0
        for name, offset in (("active_now", 0), ("active_30", 1800), ("active_60", 3600)):
            out[name] = np.float32(
                self.active_fleet.active_fraction(sim_time + offset, nr_vehicles))
        return out
