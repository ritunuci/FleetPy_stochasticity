"""Tests for src/rl_gym/gym_env.py (P1.8).

`python -m unittest discover tests`. These build real FleetPy simulations, so they are slower
than the other suites -- reset() is roughly half a second. Episode-length runs stay out; the
full rollout is P1.10.
"""

import gc
import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym  # noqa: E402

from src.misc.globals import (  # noqa: E402
    G_OP_MODULE,
    G_RL_MODE,
    G_SIM_ENV,
    G_SKIP_OUTPUT,
    G_STUDY_NAME,
)
from src.rl_gym.gym_env import (  # noqa: E402
    MAX_EPISODE_SEED,
    SDPDPAssignmentEnv,
    derive_study_name,
)

SCS = os.path.join(os.path.dirname(__file__), "..", "studies", "example_study", "scenarios")
K = 8


def make_cfg(**overrides):
    cfg = {
        "constant_cfg_path": os.path.join(SCS, "constant_config_depot_cali_sc_1.csv"),
        "scenario_cfg_path": os.path.join(SCS, "example_depot_cali_sc_1.csv"),
        "K": K,
        "skip_output": True,
    }
    cfg.update(overrides)
    return cfg


def greedy_action(env, obs):
    """Slot 0 when any candidate is valid, else reject -- restricted to is_valid (P1.10)."""
    mask = env.action_masks()
    valid = np.flatnonzero(mask[:env.k_max])
    return int(valid[0]) if len(valid) else env.k_max


class MaskRespectingShim(gym.Wrapper):
    """Test-only shim so `check_env` can run against a mask-enforcing env.

    `check_env` calls `env_step_passive_checker(env, env.action_space.sample())`, a uniform
    draw over `Discrete(K+1)` that ignores the mask. Measured on a stand-in, that made
    `check_env` fail 24 of 40 runs, nondeterministically. The env asserts on masked actions
    deliberately -- clamping to reject inside the env would make a policy bug look like a
    policy that prefers rejecting riders -- so the repair belongs here, in the test, not in
    `gym_env.py`.

    This maps an illegal sampled action onto the nearest legal one. `check_env` still exercises
    the space definitions, reset/step return types, observation-in-space, reset-seed
    determinism and step determinism; the assert itself is covered directly by
    `TestActionTranslation`.
    """

    def step(self, action):
        mask = self.env.action_masks()
        if not mask[int(action)]:
            action = int(np.flatnonzero(mask)[0])
        return self.env.step(action)


class TestStudyNameDerivation(unittest.TestCase):

    def test_matches_run_examples_derivation(self):
        def run_examples_way(p):
            return os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(p))))
        for p in ("studies/example_study/scenarios/cc.csv",
                  "/abs/elsewhere/studies/my_other_study/scenarios/cc.csv",
                  "/tmp/projects/thesis_runs/scenarios/constant_config.csv",
                  "./studies/a_b-c.2/scenarios/cc.yaml"):
            with self.subTest(path=p):
                self.assertEqual(derive_study_name(p), run_examples_way(p))

    def test_not_hardcoded_to_the_reference_study(self):
        self.assertEqual(derive_study_name("/x/studies/other/scenarios/c.csv"), "other")


class TestConstruction(unittest.TestCase):

    def test_spaces(self):
        env = SDPDPAssignmentEnv(make_cfg())
        self.assertEqual(env.action_space.n, K + 1)
        self.assertEqual(env.observation_space.shape, (3 * K + 4,))
        self.assertEqual(env.observation_space.dtype, np.float32)
        self.assertTrue(np.all(env.observation_space.low == -np.inf))
        self.assertTrue(np.all(env.observation_space.high == np.inf))

    def test_observation_space_follows_k_without_hardcoding(self):
        for k in (2, 4, 8):
            with self.subTest(K=k):
                env = SDPDPAssignmentEnv(make_cfg(K=k))
                self.assertEqual(env.observation_space.shape, (3 * k + 4,))
                self.assertEqual(env.action_space.n, k + 1)

    def test_rl_settings_forced(self):
        env = SDPDPAssignmentEnv(make_cfg())
        p = env.scenario_parameters
        self.assertEqual(p[G_SIM_ENV], "RLImmediateDecisionsSimulation")
        self.assertEqual(p[G_OP_MODULE], "RLPoolingIRSOnly")
        self.assertTrue(p[G_SKIP_OUTPUT])
        self.assertTrue(p[G_RL_MODE])
        self.assertEqual(p["n_cpu_per_sim"], 1)
        self.assertEqual(p["evaluate"], 0)
        self.assertEqual(p[G_STUDY_NAME], "example_study")

    def test_scenario_row_out_of_range_rejected(self):
        with self.assertRaises(IndexError):
            SDPDPAssignmentEnv(make_cfg(scenario_row=5))

    def test_forward_compatible_keys_accepted_and_unused(self):
        env = SDPDPAssignmentEnv(make_cfg(base_seed=123, scenario_pool=["a", "b"]))
        self.assertEqual(env.base_seed, 123)
        self.assertEqual(env.scenario_pool, ["a", "b"])


class TestResetAndStep(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = SDPDPAssignmentEnv(make_cfg())

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_reset_returns_declared_shape_and_dtype(self):
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, self.env.observation_space.shape)
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertIn("rid", info)

    def test_ten_valid_actions_step_without_error(self):
        obs, _ = self.env.reset()
        for i in range(10):
            with self.subTest(step=i):
                obs, reward, terminated, truncated, info = self.env.step(
                    greedy_action(self.env, obs))
                self.assertTrue(self.env.observation_space.contains(obs))
                self.assertIsInstance(reward, float)
                self.assertFalse(truncated)
                if terminated:
                    break

    def test_reject_action_is_always_legal(self):
        obs, _ = self.env.reset()
        for _ in range(3):
            self.assertTrue(self.env.action_masks()[K])
            obs, _, terminated, _, _ = self.env.step(K)
            if terminated:
                break

    def test_mask_shape_and_never_all_false(self):
        obs, _ = self.env.reset()
        for _ in range(5):
            mask = self.env.action_masks()
            self.assertEqual(mask.shape, (K + 1,))
            self.assertEqual(mask.dtype, np.bool_)
            self.assertTrue(mask.any())
            obs, _, terminated, _, _ = self.env.step(greedy_action(self.env, obs))
            if terminated:
                break

    def test_action_masks_does_not_advance_the_simulation(self):
        self.env.reset()
        before = self.env.step_counter
        first = self.env.action_masks()
        for _ in range(5):
            np.testing.assert_array_equal(self.env.action_masks(), first)
        self.assertEqual(self.env.step_counter, before)

    def test_mask_agrees_with_the_observation_is_valid_block(self):
        obs, _ = self.env.reset()
        for _ in range(5):
            mask = self.env.action_masks()
            np.testing.assert_array_equal(obs[:K] > 0, mask[:K])
            obs, _, terminated, _, _ = self.env.step(greedy_action(self.env, obs))
            if terminated:
                break

    def test_step_before_reset_raises(self):
        env = SDPDPAssignmentEnv(make_cfg())
        with self.assertRaises(RuntimeError):
            env.step(0)


class TestActionTranslation(unittest.TestCase):
    """The assert that `check_env` cannot be allowed to trip."""

    @classmethod
    def setUpClass(cls):
        cls.env = SDPDPAssignmentEnv(make_cfg())

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_masked_action_asserts(self):
        # the first decision may happen to fill all K slots, so advance until one does not.
        # Candidate lists are min 1, median 5 on this day, so this is found within a few steps.
        obs, _ = self.env.reset()
        masked = []
        for _ in range(50):
            mask = self.env.action_masks()
            masked = [a for a in range(K) if not mask[a]]
            if masked:
                break
            obs, _, terminated, _, _ = self.env.step(greedy_action(self.env, obs))
            self.assertFalse(terminated, "episode ended before a masked slot appeared")
        self.assertTrue(masked, "no decision with a masked slot in the first 50 steps")
        with self.assertRaises(AssertionError) as cm:
            self.env.step(masked[0])
        self.assertIn("masked action", str(cm.exception))

    def test_every_masked_slot_asserts_not_just_the_first(self):
        obs, _ = self.env.reset()
        masked = []
        for _ in range(50):
            masked = [a for a in range(K) if not self.env.action_masks()[a]]
            if masked:
                break
            obs, _, terminated, _, _ = self.env.step(greedy_action(self.env, obs))
            self.assertFalse(terminated)
        self.assertTrue(masked)
        for a in masked:
            with self.subTest(action=a):
                with self.assertRaises(AssertionError):
                    self.env.step(a)

    def test_valid_slots_do_not_assert(self):
        # the complement: every unmasked candidate slot must be accepted
        obs, _ = self.env.reset()
        for _ in range(50):
            valid = [a for a in range(K) if self.env.action_masks()[a]]
            if len(valid) >= 2:
                break
            obs, _, terminated, _, _ = self.env.step(greedy_action(self.env, obs))
            self.assertFalse(terminated)
        self.assertGreaterEqual(len(valid), 2)
        obs, _, terminated, _, _ = self.env.step(valid[-1])   # last valid slot, not slot 0
        self.assertTrue(self.env.observation_space.contains(obs))

    def test_out_of_space_action_raises_value_error(self):
        self.env.reset()
        for bad in (K + 1, 99, -1):
            with self.subTest(action=bad):
                with self.assertRaises(ValueError):
                    self.env.step(bad)

    def test_reject_translates_to_none_not_an_index(self):
        # slot K must reach commit_assignment_choice as None; if it were translated to an
        # index the operator's own bounds assert would fire instead
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(K)
        self.assertFalse(truncated)


class TestTermination(unittest.TestCase):

    def test_terminal_observation_is_zeros_and_in_space(self):
        env = SDPDPAssignmentEnv(make_cfg())
        obs, _ = env.reset()
        # reject everything: the episode still runs to the horizon, so drive to termination
        terminated = False
        steps = 0
        while not terminated and steps < 2000:
            obs, reward, terminated, truncated, info = env.step(K)
            steps += 1
        self.assertTrue(terminated, "episode did not terminate within 2000 steps")
        np.testing.assert_array_equal(obs, np.zeros(env.observation_space.shape, np.float32))
        self.assertTrue(env.observation_space.contains(obs))
        self.assertFalse(truncated, "episode end is terminated, not truncated (D6)")
        self.assertIn("episode_summary", info)
        mask = env.action_masks()
        self.assertTrue(mask[K], "reject must stay legal at termination")
        self.assertEqual(int(mask.sum()), 1)
        env.close()

    def test_step_after_termination_raises(self):
        env = SDPDPAssignmentEnv(make_cfg())
        env.reset()
        terminated = False
        steps = 0
        while not terminated and steps < 2000:
            _, _, terminated, _, _ = env.step(K)
            steps += 1
        with self.assertRaises(RuntimeError):
            env.step(K)
        env.close()


class TestLifecycle(unittest.TestCase):

    def test_close_is_idempotent_and_safe_before_reset(self):
        env = SDPDPAssignmentEnv(make_cfg())
        env.close()
        env.close()
        env.reset()
        env.close()
        env.close()

    def test_twenty_reset_close_cycles_leak_no_file_handles(self):
        try:
            import psutil
        except ImportError:
            psutil = None
        env = SDPDPAssignmentEnv(make_cfg())
        proc = psutil.Process() if psutil else None
        env.reset(); env.close(); gc.collect()
        fds_before = proc.num_fds() if proc else None
        for _ in range(20):
            env.reset()
            env.close()
        gc.collect()
        if proc:
            self.assertLessEqual(proc.num_fds() - fds_before, 2,
                                 "file handles grew across 20 reset/close cycles")

    def test_reset_after_termination_starts_a_fresh_episode(self):
        env = SDPDPAssignmentEnv(make_cfg())
        env.reset()
        first = env.episode_counter
        obs, _ = env.reset()
        self.assertEqual(env.episode_counter, first + 1)
        self.assertEqual(env.step_counter, 0)
        self.assertTrue(env.observation_space.contains(obs))
        env.close()


class TestCheckEnv(unittest.TestCase):

    def test_check_env_passes_through_the_mask_respecting_shim(self):
        warnings.filterwarnings("ignore")
        from gymnasium.utils.env_checker import check_env
        env = SDPDPAssignmentEnv(make_cfg())
        try:
            check_env(MaskRespectingShim(env), skip_render_check=True)
        finally:
            env.close()

    def test_check_env_passes_with_base_seed_set(self):
        # with base_seed absent the seed-determinism check passes vacuously, because reset()
        # leaves G_RANDOM_SEED alone; with base_seed set it is a real check of the seeding
        warnings.filterwarnings("ignore")
        from gymnasium.utils.env_checker import check_env
        env = SDPDPAssignmentEnv(make_cfg(base_seed=20260913))
        try:
            check_env(MaskRespectingShim(env), skip_render_check=True)
        finally:
            env.close()


class TestSeeding(unittest.TestCase):
    """P1.9. `base_seed` absent means do not reseed; set means per-episode reseeding (D7)."""

    def test_base_seed_absent_leaves_the_scenario_seed_untouched(self):
        env = SDPDPAssignmentEnv(make_cfg())
        self.assertIsNone(env.base_seed)
        env.reset()
        self.assertEqual(env.episode_summary()["random_seed"], 42,
                         "the byte-for-byte gates depend on this staying at the row's seed")
        env.reset(seed=999)
        self.assertEqual(env.episode_summary()["random_seed"], 42,
                         "an explicit reset seed must not override the scenario seed here")
        env.close()

    def test_base_seed_set_reseeds_every_episode(self):
        env = SDPDPAssignmentEnv(make_cfg(base_seed=12345))
        seeds = []
        for _ in range(3):
            env.reset()
            seeds.append(env.episode_summary()["random_seed"])
        env.close()
        self.assertEqual(len(set(seeds)), 3, f"episodes reused a seed: {seeds}")
        self.assertNotIn(42, seeds, "the scenario row's seed should have been replaced")
        for s in seeds:
            self.assertTrue(0 <= s <= MAX_EPISODE_SEED,
                            f"seed {s} above MAX_EPISODE_SEED; load_demand_file multiplies "
                            f"by 1712 and np.random.seed would raise")

    def test_episode_seed_bound_respects_the_1712_multiplier(self):
        # demand.py seeds with int(1712 * seed), so the usable range is far below 2**32.
        # Drawing from the full range raises inside simulation construction, which reads as a
        # demand-loading fault rather than a seeding bug.
        self.assertEqual(MAX_EPISODE_SEED, (2 ** 32 - 1) // 1712)
        self.assertLess(1712 * MAX_EPISODE_SEED, 2 ** 32)
        np.random.seed(int(1712 * MAX_EPISODE_SEED))          # must not raise
        with self.assertRaises(ValueError):
            np.random.seed(int(1712 * (MAX_EPISODE_SEED + 1)))

    def test_many_draws_all_stay_in_range(self):
        env = SDPDPAssignmentEnv(make_cfg(base_seed=2024))
        gen = np.random.default_rng(env._worker_base_seed())
        draws = gen.integers(0, MAX_EPISODE_SEED + 1, size=5000)
        self.assertTrue(draws.min() >= 0)
        self.assertTrue(draws.max() <= MAX_EPISODE_SEED)

    def test_same_reset_seed_gives_the_same_episode_seed(self):
        env = SDPDPAssignmentEnv(make_cfg(base_seed=12345))
        env.reset(seed=7)
        first = env.episode_summary()["random_seed"]
        env.reset(seed=7)
        second = env.episode_summary()["random_seed"]
        env.close()
        self.assertEqual(first, second)

    def test_different_reset_seeds_give_different_episode_seeds(self):
        env = SDPDPAssignmentEnv(make_cfg(base_seed=12345))
        env.reset(seed=1)
        a = env.episode_summary()["random_seed"]
        env.reset(seed=2)
        b = env.episode_summary()["random_seed"]
        env.close()
        self.assertNotEqual(a, b)

    def test_same_seed_gives_identical_trajectories(self):
        def rollout(seed):
            env = SDPDPAssignmentEnv(make_cfg(base_seed=555))
            obs, info = env.reset(seed=seed)
            trace = [(info["rid"], info["sim_time"], info["n_candidates"])]
            rewards = []
            for _ in range(40):
                obs, r, term, _, info = env.step(greedy_action(env, obs))
                rewards.append(r)
                if term:
                    break
                trace.append((info["rid"], info["sim_time"], info["n_candidates"]))
            seed_used = env.episode_summary()["random_seed"]
            env.close()
            return trace, rewards, seed_used
        a_trace, a_rew, a_seed = rollout(31337)
        b_trace, b_rew, b_seed = rollout(31337)
        self.assertEqual(a_seed, b_seed)
        self.assertEqual(a_trace, b_trace)
        np.testing.assert_allclose(a_rew, b_rew, rtol=0, atol=0)

    def test_different_seeds_give_different_trajectories(self):
        def rollout(seed):
            env = SDPDPAssignmentEnv(make_cfg(base_seed=555))
            obs, info = env.reset(seed=seed)
            rewards = []
            for _ in range(80):
                obs, r, term, _, _ = env.step(greedy_action(env, obs))
                rewards.append(r)
                if term:
                    break
            env.close()
            return rewards
        self.assertNotEqual(rollout(1), rollout(2),
                            "different seeds produced an identical reward sequence")

    def test_env_id_gives_each_worker_a_different_stream(self):
        seeds = []
        for env_id in range(4):
            env = SDPDPAssignmentEnv(make_cfg(base_seed=99, env_id=env_id))
            env.reset()
            seeds.append(env.episode_summary()["random_seed"])
            env.close()
        self.assertEqual(len(set(seeds)), 4, f"workers shared a seed: {seeds}")

    def test_worker_base_seed_is_reproducible(self):
        a = SDPDPAssignmentEnv(make_cfg(base_seed=99, env_id=2))._worker_base_seed()
        b = SDPDPAssignmentEnv(make_cfg(base_seed=99, env_id=2))._worker_base_seed()
        c = SDPDPAssignmentEnv(make_cfg(base_seed=99, env_id=3))._worker_base_seed()
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_episode_summary_reports_the_seed_and_worker(self):
        env = SDPDPAssignmentEnv(make_cfg(base_seed=7, env_id=3))
        env.reset()
        s = env.episode_summary()
        env.close()
        self.assertIn("random_seed", s)
        self.assertEqual(s["env_id"], 3)
        self.assertEqual(s["episode"], 1)


class TestScenarioNameUniqueness(unittest.TestCase):

    def test_scenario_name_untouched_when_output_is_skipped(self):
        env = SDPDPAssignmentEnv(make_cfg(skip_output=True))
        env.reset()
        self.assertEqual(env.sim.scenario_name,
                         env.scenario_parameters["scenario_name"])
        env.close()

    def test_scenario_name_carries_env_pid_and_counter_when_output_is_on(self):
        env = SDPDPAssignmentEnv(make_cfg(skip_output=False, env_id=2))
        made = []
        try:
            for _ in range(2):
                env.reset()
                made.append(env.sim.scenario_name)
            env.close()
            base = env.scenario_parameters["scenario_name"]
            for i, name in enumerate(made, start=1):
                self.assertTrue(name.startswith(base))
                self.assertIn("_env2_", name)
                self.assertIn(f"_pid{os.getpid()}_", name)
                self.assertTrue(name.endswith(f"_ep{i}"))
            self.assertEqual(len(set(made)), 2,
                             "episodes reused a directory; create_or_empty_dir would erase it")
        finally:
            import shutil
            results = os.path.join(os.path.dirname(__file__), "..", "studies",
                                   "example_study", "results")
            for name in made:
                d = os.path.join(results, name)
                if os.path.isdir(d):
                    shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
