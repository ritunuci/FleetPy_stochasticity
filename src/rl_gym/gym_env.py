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

Per-episode seeding and the unique `scenario_name` are P1.9. Until then `reset()` accepts the
seed argument for API conformance but builds with the scenario row's own `random_seed`, so two
resets are identical trivially rather than because seeding works.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from src.misc import config
from src.misc.globals import (
    G_OP_MODULE,
    G_RL_MODE,
    G_SCENARIO_NAME,
    G_SIM_ENV,
    G_SKIP_OUTPUT,
    G_STUDY_NAME,
)
from src.misc.init_modules import load_simulation_environment
from src.rl_gym.observers import CandidateObserver, GlobalStateObserver
from src.rl_gym.reward import RewardTracker
from src.rl_gym.spaces import build_action_mask, make_action_space, truncate_candidates

LOG = logging.getLogger(__name__)

RL_SIM_ENV = "RLImmediateDecisionsSimulation"
RL_OP_MODULE = "RLPoolingIRSOnly"


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
            `env_id` (default 0); `base_seed`, `scenario_pool` (accepted, unused until P1.9).
        """
        super().__init__()
        self.cfg = dict(cfg)
        self.constant_cfg_path = cfg["constant_cfg_path"]
        self.scenario_cfg_path = cfg["scenario_cfg_path"]
        self.scenario_row = int(cfg.get("scenario_row", 0))
        self.k_max = int(cfg.get("K", 8))
        self.skip_output = bool(cfg.get("skip_output", True))
        self.reward_weights = cfg.get("reward_weights")
        self.env_id = int(cfg.get("env_id", 0))
        # accepted for forward compatibility; P1.9 owns seeding, Phase 2 owns day rotation
        self.base_seed = cfg.get("base_seed")
        self.scenario_pool = cfg.get("scenario_pool")

        self.scenario_parameters = self._build_scenario_parameters()

        self._candidate_observer = CandidateObserver(self.k_max, self.scenario_parameters)
        self._global_observer = GlobalStateObserver(self.scenario_parameters)
        obs_len = self._candidate_observer.output_length + self._global_observer.output_length

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
        constant_cfg.setdefault("log_level", "warning")

        params = constant_cfg + scenario_cfgs[self.scenario_row]
        params[G_SIM_ENV] = RL_SIM_ENV
        params[G_OP_MODULE] = RL_OP_MODULE
        params[G_SKIP_OUTPUT] = self.skip_output
        params[G_RL_MODE] = True
        return params

    # ---------------------------------------------------------------- #
    # gymnasium API
    # ---------------------------------------------------------------- #

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Build a fresh simulation and advance to the first decision.

        A fresh simulation per episode is required, not an optimisation choice: `_started`
        makes a used object unrunnable (trap 7), and reusing one would leak stochastic
        travel-time state, dynamic network state and RNG state across episodes.
        """
        super().reset(seed=seed)
        self.close()

        scenario_parameters = dict(self.scenario_parameters)
        # P1.9 will derive a per-episode seed and a unique scenario_name here

        self.sim = load_simulation_environment(scenario_parameters)
        self._tracker = RewardTracker(self.reward_weights)
        self.sim.operators[0].set_reward_tracker(self._tracker)
        self.sim.demand._boarding_callback = self._tracker.on_pickup
        self.sim.demand._exit_callback = self._tracker.on_exit

        self._generator = self.sim.run_generator()
        self._terminated = False
        self.step_counter = 0
        self.episode_counter += 1

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
            self._set_terminal_observation()
            return self._obs, reward, True, False, self._episode_info()

        reward = self._tracker.flush()
        self._observe(self._pending)
        return self._obs, reward, False, False, {
            "rid": self._pending.rid_struct,
            "sim_time": self._pending.sim_time,
            "n_candidates": len(self._pending.candidates),
            "n_truncated": max(0, len(self._pending.candidates) - self.k_max),
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
            self._candidate_observer.observe_vector(self.sim.operators[0], pending),
            self._global_observer.observe_vector(self.sim.operators[0], pending),
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
            info["episode_summary"] = self._tracker.episode_summary()
        return info

    def episode_summary(self) -> Dict[str, Any]:
        """The tracker's event breakdown for the current or just-finished episode."""
        return self._tracker.episode_summary() if self._tracker is not None else {}
