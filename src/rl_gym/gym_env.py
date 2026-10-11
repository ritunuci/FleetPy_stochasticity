"""`SDPDPAssignmentEnv` — the Gymnasium environment (P1.8).

One gym step is one request for which the operator has a real choice. `reset()` builds a
fresh simulation, attaches the reward tracker and the two `demand.py` callbacks, creates the
generator and advances to the first decision; `step(action)` translates the action, sends it,
flushes the reward window, and builds the observation for the next decision.

This class is the **only** place `K` appears. `RLPoolingIRSOnly` receives a semantic choice --
a candidate index or `None` to reject -- never a raw action, and the observers take `K` as an
argument. Per decision the env truncates `pending.candidates` to the first `K`; because
truncation is a prefix, slot indices map straight onto candidate indices and the translation is
`action == K -> reject`, `action < K -> candidate index`.

A masked action is a programming error and **asserts**. It is deliberately not clamped to
reject: a policy bug would then look like a policy that prefers rejecting riders, which is
indistinguishable from a real learned preference. `gymnasium.utils.env_checker.check_env`
samples actions without regard to the mask and so cannot be run against this env directly --
see the shim in `tests/test_rl_gym_env.py`.

**Seeding is switched on by `base_seed` (D7).** With `base_seed` set, each `reset()` draws an
episode seed and writes it to `scenario_parameters[G_RANDOM_SEED]` before construction, which
is what gives the agent demand variation across episodes. With `base_seed` absent, the scenario
row's own `random_seed` is used unchanged -- that is what lets P1.10's byte-for-byte gate
compare against a baseline generated at seed 42 and so keep testing the plumbing rather than the
seeding. `reset(seed=...)` seeds the env's own generator in both modes, so the Gymnasium
contract holds either way.
"""

import datetime as dt
import logging
import os
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from src.misc import config
from src.misc.globals import (
    G_DAY_DIR,
    G_DIR_DEMAND,
    G_DIR_FCTRL,
    G_NETWORK_NAME,
    G_OP_ACT_FLEET_SIZE,
    G_OP_MODULE,
    G_RANDOM_SEED,
    G_RL_MODE,
    G_RQ_FILE,
    G_SCENARIO_NAME,
    G_SIM_END_TIME,
    G_SIM_ENV,
    G_SKIP_OUTPUT,
    G_STUDY_NAME,
    get_directory_dict,
)
from src.misc.init_modules import load_simulation_environment
from src.rl_gym.day_rotation import day_for_episode, training_days
from src.rl_gym.corpus import DEFAULT_CORPUS_DIR, load_manifest, verify_corpus
from src.rl_gym.observers import (
    ActiveFleetSchedule,
    ArrivalBaseline,
    CandidateObserver,
    GlobalStateObserver,
    NodeCoordinates,
    RequestObserver,
    build_observation_layout,
    quoted_waiting_time,
)
from src.rl_gym.reward import RewardTracker
from src.rl_gym.rolling_stats import RollingStatsTracker
from src.rl_gym.zones import ZoneTable
from src.rl_gym.spaces import build_action_mask, make_action_space, truncate_candidates

LOG = logging.getLogger(__name__)

RL_SIM_ENV = "RLImmediateDecisionsSimulation"
RL_OP_MODULE = "RLPoolingIRSOnly"

#: Upper bound for an episode seed, and it is **not** 2**32.
#: `Demand.load_demand_file` (demand.py ~line 78) and `load_parcel_demand_file` (~line 132)
#: both do `np.random.seed(int(1712 * np_random_seed))`, multiplying the seed before using it.
#: `np.random.seed` rejects anything above 2**32 - 1, so the seed itself must stay below
#: (2**32 - 1) // 1712. Drawing from the full 32-bit range raises ValueError partway through
#: simulation construction, not at reset, which makes it look like a demand-loading fault.
MAX_EPISODE_SEED = (2 ** 32 - 1) // 1712  # 2_508_742


#: weekday vocabulary of the corpus (§2.9); the index is `date.weekday()`
_WEEKDAY_NAMES = {1: "Tue", 2: "Wed", 3: "Thu"}


def parse_day_stem(stem: str) -> Tuple[Optional[str], Optional[int]]:
    """`("Tue"|"Wed"|"Thu"|<other>, month)` from an ISO filename stem, or `(None, None)`.

    The demand rows carry `rq_time` as seconds from midnight and no date column, so the
    filename is the only record of which day a file is (§2.9). A stem that is not an ISO date
    yields `(None, None)` rather than raising: the reference path's `rq_file` is not required to
    be ISO-named, and P2.9 -- the first consumer that actually needs these -- raises on an
    unknown weekday or month against P2.6's vocabularies (§5.2).
    """
    try:
        date = dt.date.fromisoformat(stem)
    except (ValueError, TypeError):
        return None, None
    return _WEEKDAY_NAMES.get(date.weekday(), date.strftime("%a")), date.month


def derive_study_name(constant_cfg_path: str) -> str:
    """Study name from the config path, the way `run_examples.run_scenarios` derives it.

    `studies/<study>/scenarios/constant_config_x.csv` -> `<study>`. `get_directory_dict`
    depends on this, so it must not be hardcoded.
    """
    const_abs = os.path.abspath(constant_cfg_path)
    return os.path.basename(os.path.dirname(os.path.dirname(const_abs)))


class SDPDPAssignmentEnv(gym.Env):
    """Gymnasium environment over the ride-pooling assignment decision of `PoolingIRSOnly`."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: Dict[str, Any]):
        """
        :param cfg: config dict. Recognised keys:
            `constant_cfg_path`, `scenario_cfg_path` (required); `scenario_row` (default 0);
            `K` (default 8); `skip_output` (default True); `reward_weights` (default None);
            `track_events` (default False); `env_id` (default 0); `base_seed`;
            `day_dir`, `day`, `n_workers`, `episode_seed`, `eval_pairs` (P2.3);
            `corpus_dir` (P2.9); `wait_normalizer_s`, `position_encoding` (P2.10);
            `scenario_pool` (accepted and unused).

            `wait_normalizer_s` divides every wait-type feature and **defaults to
            `op_max_wait_time`**, which reproduces Phase 1's `offered_wait` bit for bit (§5.3).
            `position_encoding` is `"latlon"` (default) or `"relative"`; the observation length
            and the feature names are the same either way.
        """
        super().__init__()
        self.cfg = dict(cfg)
        self.constant_cfg_path = cfg["constant_cfg_path"]
        self.scenario_cfg_path = cfg["scenario_cfg_path"]
        self.scenario_row = int(cfg.get("scenario_row", 0))
        self.k_max = int(cfg.get("K", 8))
        self.skip_output = bool(cfg.get("skip_output", True))
        self.reward_weights = cfg.get("reward_weights")
        # per-event reward breakdown in info["reward_events"]; off by default, since training
        # does not need it and 7,000 episodes should not pay for it (P2.0b)
        self.track_events = bool(cfg.get("track_events", False))
        self.env_id = int(cfg.get("env_id", 0))
        # base_seed absent => do not reseed; the scenario row's own random_seed is used
        # unchanged, which is what the byte-for-byte gates require (D7)
        self.base_seed = cfg.get("base_seed")
        # accepted and unused; the day source is `day_dir` (P2.3), not this
        self.scenario_pool = cfg.get("scenario_pool")
        self._generator_seeded = False
        self._episode_seed = None

        #: the frozen training split the manifest was built over (P2.9). A config key so a
        #: test can point the check at a temporary copy.
        self.corpus_dir = cfg.get("corpus_dir", DEFAULT_CORPUS_DIR)
        #: P2.5's zone lookup: static data, built once and shared across episodes. Set before
        #: the observers, which ask for it at construction.
        self._zones: Optional[ZoneTable] = None

        self.scenario_parameters = self._build_scenario_parameters()
        self._configure_days(cfg)

        # trap 16, at construction and unconditionally: P2.6's baseline, its month means and
        # its SD were computed over one particular set of days, and the request block's
        # vocabularies come from the same manifest. A baseline read against a different corpus
        # produces plausible-looking anomalies that are simply wrong. This runs even on the
        # reference path, which reads `thursday/` and never touches a corpus day -- so a corpus
        # edit now breaks the P1.10 gate, deliberately, with a message naming the files.
        self.manifest = load_manifest()
        verify_corpus(self.manifest, self.corpus_dir)

        # static data, built once and shared by the candidate and request blocks: node lat/lon
        # and the hull they scale against (§2.10, §5.2)
        node_coords = NodeCoordinates(network_name=self.scenario_parameters[G_NETWORK_NAME])
        self._candidate_observer = CandidateObserver(
            self.k_max, self.scenario_parameters, node_coords,
            wait_normalizer_s=cfg.get("wait_normalizer_s"),
            position_encoding=cfg.get("position_encoding", "latlon"),
        )
        self._request_observer = RequestObserver(
            node_coords,
            self._zone_table(),
            weekday_vocab=self.manifest["weekday_vocab"],
            month_vocab=self.manifest["month_vocab"],
        )
        # P2.6's baseline, with its manifest-recorded hash verified against the parsed rows
        # (trap 16, guard 2), and the scheduled active fleet curve. Both static, both built
        # once; the baseline additionally checks that its minute grid and windows cover this
        # scenario, so a scenario off the grid fails here rather than mid-episode (guard 1).
        self._arrival_baseline = ArrivalBaseline(self.manifest)
        self._active_fleet = ActiveFleetSchedule(self._active_fleet_path())
        self._global_observer = GlobalStateObserver(
            self.scenario_parameters,
            arrival_baseline=self._arrival_baseline,
            active_fleet=self._active_fleet,
            wait_normalizer_s=cfg.get("wait_normalizer_s"),
        )
        #: the order the observers are concatenated in, and so the order of the layout map.
        #: Block order is candidate, request, global (§5.2).
        self._observers = (self._candidate_observer, self._request_observer,
                           self._global_observer)
        #: name -> slice over the observation vector, built from the observers' static
        #: `feature_spec()` declarations at construction (P2.8). Published so consumers address
        #: features by name instead of assuming a block ordering: `policies.greedy_action`
        #: reads `delta_cfv` and `is_valid` through it, and P2.9-P2.11 widen the vector without
        #: any consumer changing. Raises on a duplicate name.
        self.observation_layout = build_observation_layout(self._observers)
        obs_len = sum(o.output_length for o in self._observers)

        self.action_space = make_action_space(self.k_max)
        # unbounded: delta_cfv has no a priori bound even after sign*log1p, and VecNormalize
        # does the clipping in Phase 2 (§5.1, §5.3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32
        )

        # live episode state
        self.sim = None
        self._generator = None
        self._pending = None
        self._tracker: Optional[RewardTracker] = None
        self._stats: Optional[RollingStatsTracker] = None
        self._obs = np.zeros(obs_len, dtype=np.float32)
        self._mask = build_action_mask(0, self.k_max)
        self._terminated = True
        self.episode_counter = 0
        self.step_counter = 0

    # ---------------------------------------------------------------- #
    # config
    # ---------------------------------------------------------------- #

    def _build_scenario_parameters(self) -> dict:
        """Merge constant + scenario configs and force the RL settings."""
        constant_cfg = config.ConstantConfig(self.constant_cfg_path)
        scenario_cfgs = config.ScenarioConfig(self.scenario_cfg_path)
        if not scenario_cfgs:
            raise ValueError(f"no active scenario rows in {self.scenario_cfg_path}")
        if not 0 <= self.scenario_row < len(scenario_cfgs):
            raise IndexError(
                f"scenario_row {self.scenario_row} out of range for "
                f"{len(scenario_cfgs)} row(s) in {self.scenario_cfg_path}"
            )

        constant_cfg[G_STUDY_NAME] = derive_study_name(self.constant_cfg_path)
        # required: FleetControlBase reads this with [] and has no default
        constant_cfg["n_cpu_per_sim"] = 1
        constant_cfg["evaluate"] = 0
        constant_cfg["keep_old"] = False
        # quiet by default; raise it only to inspect a run's log (P1.11 uses this to verify
        # that four workers write four separate, non-interleaved log files)
        constant_cfg["log_level"] = self.cfg.get("log_level", "warning")

        params = constant_cfg + scenario_cfgs[self.scenario_row]
        params[G_SIM_ENV] = RL_SIM_ENV
        params[G_OP_MODULE] = RL_OP_MODULE
        params[G_SKIP_OUTPUT] = self.skip_output
        params[G_RL_MODE] = True
        return params

    # ---------------------------------------------------------------- #
    # day source and seed source (P2.3)
    # ---------------------------------------------------------------- #

    def _configure_days(self, cfg: Dict[str, Any]) -> None:
        """Resolve the day source and the seed source, and reject contradictory combinations.

        Four day sources, in precedence order: `eval_pairs` (pair `i mod len`), `day` (fixed),
        D10 rotation over `day_dir`, and -- with `day_dir` absent -- the scenario row's own
        `day_dir_name` / `rq_file` untouched, which is the path the byte-for-byte gates need.

        Everything here raises rather than defaulting. A silently wrong day source produces a
        healthy-looking run against the wrong data, which is the class of failure this whole
        design keeps engineering out.
        """
        self.day_dir = cfg.get("day_dir")
        self.fixed_day = cfg.get("day")
        self.n_workers = cfg.get("n_workers")
        self.fixed_episode_seed = cfg.get("episode_seed")
        self.eval_pairs = cfg.get("eval_pairs")

        self._episode_day = None            # filename stem of the day this episode runs
        self._episode_pass = None           # D10 pass index, rotation mode only
        self._episode_position = None       # position within that pass
        self._episode_weekday = None        # "Tue" / "Wed" / "Thu", or None if unparseable
        self._episode_month = None          # 1..12, or None if unparseable
        self._training_days = None          # sorted day list, rotation mode only

        if self.fixed_episode_seed is not None and self.base_seed is not None:
            raise ValueError(
                "episode_seed and base_seed cannot both be set: the first pins one seed for "
                "every episode, the second draws a new one per episode (D7). Choose one."
            )

        if self.eval_pairs is not None:
            for value, name in ((self.fixed_day, "day"),
                                (self.fixed_episode_seed, "episode_seed"),
                                (self.base_seed, "base_seed")):
                if value is not None:
                    raise ValueError(
                        f"eval_pairs cannot be combined with {name}: eval_pairs already fixes "
                        f"both the day and the seed of every episode."
                    )
            if self.day_dir is None:
                raise ValueError("eval_pairs requires day_dir, which names the directory its "
                                 "days live in.")
            pairs = [(str(d), int(s)) for d, s in self.eval_pairs]
            if not pairs:
                raise ValueError("eval_pairs is empty; omit it instead.")
            for _day, seed in pairs:
                self._check_seed_range(seed, "a seed in eval_pairs")
            self.eval_pairs = pairs

        if self.fixed_episode_seed is not None:
            self.fixed_episode_seed = int(self.fixed_episode_seed)
            self._check_seed_range(self.fixed_episode_seed, "episode_seed")

        # the directory days are resolved against: day_dir when given, else the scenario's own
        self._day_dir_name = self.day_dir or self.scenario_parameters[G_DAY_DIR]
        self._demand_dir = self._resolve_demand_dir(self._day_dir_name)

        rotating = (self.day_dir is not None
                    and self.fixed_day is None
                    and self.eval_pairs is None)
        if rotating:
            # D10's permutation stream is SeedSequence([base_seed, 1010, p + 1]); without
            # base_seed there is nothing to draw it from, and defaulting one is what D7 forbids
            if self.base_seed is None:
                raise ValueError(
                    "day rotation requires base_seed: D10 draws each pass's permutation from "
                    "SeedSequence([base_seed, 1010, p + 1]), and every worker must derive the "
                    "same permutation. Set base_seed, or pin a day with `day` / `eval_pairs`."
                )
            if self.n_workers is None:
                raise ValueError(
                    "day rotation requires n_workers: worker env_id takes the days at "
                    "positions env_id, env_id + n_workers, ... of each pass (D10), so the "
                    "split cannot be computed without it."
                )
            self.n_workers = int(self.n_workers)
            self._training_days = training_days(self._demand_dir)
            # raises on env_id >= n_workers, and on n_workers > len(days)
            day_for_episode(self._training_days, self.base_seed, self.n_workers,
                            self.env_id, 0)

        # a named day that does not exist must fail here, not deep inside load_demand_file
        named = []
        if self.fixed_day is not None:
            self.fixed_day = str(self.fixed_day)
            named.append(self.fixed_day)
        if self.eval_pairs is not None:
            named.extend(day for day, _seed in self.eval_pairs)
        for day in named:
            path = os.path.join(self._demand_dir, f"{day}.csv")
            if not os.path.isfile(path):
                raise ValueError(f"demand day {day!r} not found at {path}")

    @staticmethod
    def _check_seed_range(seed: int, label: str) -> None:
        if not 0 <= seed <= MAX_EPISODE_SEED:
            raise ValueError(
                f"{label} must be in [0, {MAX_EPISODE_SEED}], got {seed}. "
                f"Demand.load_demand_file multiplies the seed by 1712 before passing it to "
                f"np.random.seed, which rejects anything above 2**32 - 1 (P1.9)."
            )

    def _active_fleet_path(self) -> str:
        """`op_act_fs_file`'s absolute path, the way `TimeBasedFS.add_init` builds it.

        `dir_names[G_DIR_FCTRL]/elastic_fleet_size/<file>` (`TimeBasedFS.py:51`), through
        `get_directory_dict` rather than a reassembled path, so this cannot drift from what the
        simulation will resolve. Read-only; the fleet-sizing module is never touched (§5.2).
        """
        file_name = self.scenario_parameters.get(G_OP_ACT_FLEET_SIZE)
        if file_name is None:
            raise ValueError(
                f"the global block's `active_*` features need {G_OP_ACT_FLEET_SIZE}, which this "
                f"scenario does not set. `TimeBasedFS` raises on the same thing, so a scenario "
                f"without it could not have run anyway (§5.2)."
            )
        fctrl_dir = get_directory_dict(dict(self.scenario_parameters))[G_DIR_FCTRL]
        return os.path.join(fctrl_dir, "elastic_fleet_size", str(file_name))

    def _resolve_demand_dir(self, day_dir_name: str) -> str:
        """Absolute demand directory for `day_dir_name`, via FleetPy's own path builder.

        Uses `get_directory_dict` rather than reassembling the path, so this cannot drift from
        what `FleetSimulationBase` will compute at construction. Nothing is cached for the
        simulation's benefit -- each `reset()` writes the keys and lets construction derive its
        own `dir_names`; this copy exists only to validate day names up front.
        """
        probe = dict(self.scenario_parameters)
        probe[G_DAY_DIR] = day_dir_name
        return get_directory_dict(probe)[G_DIR_DEMAND]

    def _apply_episode_day(self, scenario_parameters: dict) -> None:
        """Choose this episode's day and write `G_DAY_DIR` / `G_RQ_FILE` before construction.

        With `day_dir` absent nothing is written: the scenario row's own keys stand, which is
        what keeps the byte-for-byte gates comparing like with like.
        """
        self._episode_pass = None
        self._episode_position = None
        index = self.episode_counter - 1              # zero-based episode index for this worker

        if self.eval_pairs is not None:
            day, _seed = self.eval_pairs[index % len(self.eval_pairs)]
        elif self.fixed_day is not None:
            day = self.fixed_day
        elif self._training_days is not None:
            day, self._episode_pass, self._episode_position = day_for_episode(
                self._training_days, self.base_seed, self.n_workers, self.env_id, index)
        else:
            day = os.path.splitext(str(scenario_parameters[G_RQ_FILE]))[0]

        self._episode_day = day
        self._episode_weekday, self._episode_month = parse_day_stem(day)
        if self.day_dir is not None or self.fixed_day is not None:
            scenario_parameters[G_DAY_DIR] = self._day_dir_name
            scenario_parameters[G_RQ_FILE] = f"{day}.csv"

    # ---------------------------------------------------------------- #
    # tracker fan-out (P2.7)
    # ---------------------------------------------------------------- #

    def _zone_table(self) -> ZoneTable:
        """P2.5's zone lookup, built once per env and reused across episodes.

        The table is static data -- `nodes.csv` and a committed CSV -- so it is not simulation
        state and §10's no-caching rule does not apply to it. The *tracker* is rebuilt every
        reset; only this lookup is shared.
        """
        if self._zones is None:
            self._zones = ZoneTable(network_name=self.scenario_parameters[G_NETWORK_NAME])
        return self._zones

    @staticmethod
    def _make_quoted_wait_fn(op):
        """`rid -> the operator's quoted wait in seconds, or None`.

        The quote lives on the operator's `PlanRequest` as `prq.offer.offered_waiting_time`,
        set once by `_create_user_offer` as `pu_time - prq.rq_time` — already relative to
        `rq_time`, so it subtracts directly from the realized wait. Read from
        `rq_dict_store`, which unlike `rq_dict` keeps its entries for the whole episode.
        **Nothing from the diffusion model is read** (§5.2): not
        `rq_obj.expected_pickup_time`, not `ETA_update_time_stamp`, not any
        `diffusion_state_*` — the operator has no knowledge of the behavioural model.
        A rejected request carries a `Rejection`, whose `offered_waiting_time` is None; such a
        rider is never picked up, but the None is handled rather than assumed away.

        Delegates to `observers.quoted_waiting_time`, which P2.10's lateness features read
        through as well: two copies of this lookup would be the drift trap 13 rules out for the
        exit classifier, in a quantity that feeds both the reward-adjacent statistics and the
        observation.
        """
        return lambda rid: quoted_waiting_time(op, rid)

    def _on_pickup(self, rid, rq) -> None:
        """`Demand.record_boarding` fan-out: reward first, then statistics."""
        self._tracker.on_pickup(rid, rq)
        self._stats.on_pickup(rid, rq)

    def _on_exit(self, rid, rq) -> None:
        """`Demand.record_user` fan-out, with **one** §6.4 classification (trap 13).

        `RewardTracker.on_exit` returns the outcome it computed, and that single string is what
        the statistics tracker records — the two can therefore never label the same exit
        differently. It returns `None` for a repeated exit on an already-handled rid, which is
        also what keeps the statistics idempotent: `record_alighting_start` and
        `record_remaining_users` can both fire for a rider mid-alighting at end of day.
        """
        outcome = self._tracker.on_exit(rid, rq)
        if outcome is None:
            return
        self._stats.on_exit(rid, rq, outcome)

    def _episode_pair_seed(self) -> Optional[int]:
        """The seed this episode is pinned to, or None when it is not pinned."""
        if self.eval_pairs is not None:
            return self.eval_pairs[(self.episode_counter - 1) % len(self.eval_pairs)][1]
        return self.fixed_episode_seed

    # ---------------------------------------------------------------- #
    # gymnasium API
    # ---------------------------------------------------------------- #

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Build a fresh simulation and advance to the first decision.

        A fresh simulation per episode is required, not an optimisation choice: `_started`
        makes a used object unrunnable (trap 7), and reusing one would leak stochastic
        travel-time state, dynamic network state and RNG state across episodes.
        """
        if seed is None and not self._generator_seeded and self.base_seed is not None:
            seed = self._worker_base_seed()
        super().reset(seed=seed)
        if seed is not None:
            self._generator_seeded = True
        self.close()

        scenario_parameters = dict(self.scenario_parameters)
        self.episode_counter += 1
        # day before seed: the scenario name needs both, and eval_pairs reads them together
        self._apply_episode_day(scenario_parameters)
        self._apply_episode_seed(scenario_parameters)
        self._apply_episode_scenario_name(scenario_parameters)

        self.sim = load_simulation_environment(scenario_parameters)
        self._tracker = RewardTracker(
            self.reward_weights,
            track_events=self.track_events,
            end_time=scenario_parameters[G_SIM_END_TIME],
        )
        op = self.sim.operators[0]
        op.set_reward_tracker(self._tracker)

        self._stats = RollingStatsTracker(
            self._zone_table(), quoted_wait_fn=self._make_quoted_wait_fn(op))
        op.set_stats_tracker(self._stats)
        self.sim.set_stats_tracker(self._stats)
        # the request block's two per-episode dependencies, injected rather than discovered
        # (P2.9). set_episode_day validates against the manifest vocabularies and raises here,
        # before the episode runs, rather than at the first decision.
        self._request_observer.set_stats_tracker(self._stats)
        self._request_observer.set_episode_day(self._episode_weekday, self._episode_month)
        # the global block's two, the same way (P2.11). set_episode_month raises here, at reset,
        # against the baseline's own months.
        self._global_observer.set_stats_tracker(self._stats)
        self._global_observer.set_episode_month(self._episode_month)

        self.sim.demand._boarding_callback = self._on_pickup
        self.sim.demand._exit_callback = self._on_exit

        self._generator = self.sim.run_generator()
        self._terminated = False
        self.step_counter = 0

        try:
            self._pending = next(self._generator)
        except StopIteration:
            # a day with no actionable request; degenerate but not an error
            self._pending = None
            self._terminated = True
            self._set_terminal_observation()
            return self._obs, self._episode_info()

        self._observe(self._pending)
        return self._obs, {"rid": self._pending.rid_struct,
                           "sim_time": self._pending.sim_time,
                           "n_candidates": len(self._pending.candidates)}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Apply `action` at the current decision, advance to the next, return the window."""
        if self._terminated:
            raise RuntimeError("step() called on a terminated episode; call reset() first")
        if self._pending is None:
            raise RuntimeError("step() called before reset()")

        choice = self._translate(int(action))
        self.step_counter += 1

        try:
            # flush AFTER send, never before: the simulation is frozen while the agent
            # decides, so the window is exactly the action's consequences plus whatever the
            # simulation produced while advancing (§6.1)
            self._pending = self._generator.send(choice)
        except StopIteration:
            self._pending = None
            self._terminated = True
            reward = self._tracker.flush()      # captures the post-end_time tail (D6)
            info = self._episode_info()
            info["reward_events"] = self._tracker.flush_events()
            self._set_terminal_observation()
            return self._obs, reward, True, False, info

        reward = self._tracker.flush()
        self._observe(self._pending)
        return self._obs, reward, False, False, {
            "rid": self._pending.rid_struct,
            "sim_time": self._pending.sim_time,
            "n_candidates": len(self._pending.candidates),
            "n_truncated": max(0, len(self._pending.candidates) - self.k_max),
            # the events composing `reward`; empty unless `track_events` is set. Drained on
            # both step paths and on neither reset path: `reset()` does not flush, so anything
            # charged while advancing to the first decision belongs to step 1's window, and
            # draining it here would drop it from the trace while its reward stayed in the sum
            "reward_events": self._tracker.flush_events(),
        }

    def action_masks(self) -> np.ndarray:
        """The mask for the current observation. Never advances the simulation.

        Callable at any point after `reset()` or `step()` returns -- the mask is stored
        alongside the observation when the generator yields, not recomputed on demand.
        """
        return self._mask

    def close(self) -> None:
        """Close the generator and drop the simulation. Idempotent, and safe before reset."""
        if self._generator is not None:
            try:
                self._generator.close()
            except Exception:
                LOG.warning("closing the simulation generator raised", exc_info=True)
            self._generator = None
        self.sim = None
        self._pending = None
        self._terminated = True

    # ---------------------------------------------------------------- #
    # internals
    # ---------------------------------------------------------------- #

    def _worker_base_seed(self) -> int:
        """This worker's stream seed, from `base_seed` and `env_id`.

        `SeedSequence` rather than `base_seed + env_id`: it is numpy's intended idiom for
        independent streams and sidesteps any question about adjacent seeds correlating.
        """
        ss = np.random.SeedSequence([int(self.base_seed), int(self.env_id)])
        return int(ss.generate_state(1, dtype=np.uint32)[0])

    def _apply_episode_seed(self, scenario_parameters: dict) -> None:
        """Draw and write this episode's `G_RANDOM_SEED` -- only when `base_seed` is set.

        With `base_seed` absent the scenario row's own seed is left untouched, which is what
        keeps P1.10's byte-for-byte gate comparing like with like (D7). Every stochastic
        component draws from the global `np.random` stream that `FleetSimulationBase.__init__`
        seeds from this key, so writing it here is sufficient and complete.
        """
        pinned = self._episode_pair_seed()
        if pinned is not None:
            # episode_seed or eval_pairs: the same G_RANDOM_SEED every episode, or the pair's
            self._episode_seed = int(pinned)
            scenario_parameters[G_RANDOM_SEED] = self._episode_seed
            return
        if self.base_seed is None:
            self._episode_seed = scenario_parameters.get(G_RANDOM_SEED)
            return
        # bounded by the 1712 multiplier inside load_demand_file, not by 2**32 -- see
        # MAX_EPISODE_SEED
        self._episode_seed = int(self.np_random.integers(0, MAX_EPISODE_SEED + 1))
        scenario_parameters[G_RANDOM_SEED] = self._episode_seed

    def _apply_episode_scenario_name(self, scenario_parameters: dict) -> None:
        """Make the output directory unique per worker, process and episode.

        Only matters when output is on: `create_or_empty_dir` wipes the directory it is given,
        so two workers sharing a `scenario_name` would erase each other (trap 1). The pid is
        needed because independently launched processes can share an `env_id`, and the counter
        because otherwise each episode would erase the previous one.
        """
        if self.skip_output:
            return
        base = self.scenario_parameters[G_SCENARIO_NAME]
        # With the pair fixed (episode_seed or eval_pairs) the directory names the pair it
        # holds (P2.3). In rotation mode the day varies per episode, so the name stays P1.9's
        # and the day is read from `episode_summary()` instead; the counter keeps it unique.
        if self._episode_pair_seed() is not None:
            base = f"{base}_{self._episode_day}_s{self._episode_seed}"
        scenario_parameters[G_SCENARIO_NAME] = (
            f"{base}_env{self.env_id}_pid{os.getpid()}_ep{self.episode_counter}"
        )

    def _translate(self, action: int) -> Optional[int]:
        """Raw action -> semantic choice. `K` is reject; `k < K` is a candidate index."""
        if not self.action_space.contains(action):
            raise ValueError(f"action {action} outside {self.action_space}")
        if action == self.k_max:
            return None
        n_valid = min(len(self._pending.candidates), self.k_max)
        # masked actions must never be sent; assert rather than handle, so a policy bug can
        # never be mistaken for a preference for rejecting riders
        assert action < n_valid, (
            f"masked action {action} sent at rid {self._pending.rid_struct}: only "
            f"{n_valid} candidate slot(s) valid, plus reject at {self.k_max}"
        )
        return action

    def _observe(self, pending) -> None:
        """Store the observation and the mask that belongs with it."""
        self._obs = np.concatenate([
            o.observe_vector(self.sim.operators[0], pending) for o in self._observers
        ]).astype(np.float32)
        self._mask = build_action_mask(len(pending.candidates), self.k_max)

    def _set_terminal_observation(self) -> None:
        """Zeros, with reject the only legal action.

        Zeros is in-space and reads correctly: every candidate slot is invalid, which is true
        -- there is no decision left. The mask keeps reject legal so it is never all-False.
        """
        self._obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        self._mask = build_action_mask(0, self.k_max)

    def _episode_info(self) -> Dict[str, Any]:
        info = {"episode_steps": self.step_counter, "episode": self.episode_counter}
        if self._tracker is not None:
            info["episode_summary"] = self.episode_summary()
        return info

    def episode_summary(self) -> Dict[str, Any]:
        """The tracker's event breakdown for the current or just-finished episode.

        Carries `random_seed`, the seed the episode actually ran under, so a training log shows
        which sample path each episode saw rather than leaving it to be inferred (D7).
        """
        summary = self._tracker.episode_summary() if self._tracker is not None else {}
        summary["random_seed"] = self._episode_seed
        summary["env_id"] = self.env_id
        summary["episode"] = self.episode_counter
        # the day this episode ran, plus where D10's rotation put it. In rotation mode the
        # output directory does not name the day, so this is the record of it (P2.3), and
        # P2.13's day-visit histogram is built from `day` and `pass`.
        summary["day"] = self._episode_day
        summary["day_dir"] = self._day_dir_name
        summary["weekday"] = self._episode_weekday
        summary["month"] = self._episode_month
        summary["pass"] = self._episode_pass
        summary["pass_position"] = self._episode_position
        if self._stats is not None:
            # P2.7's end-of-day totals, for reconciliation against 1_user-stats.csv
            summary.update(self._stats.episode_summary())
        return summary
