"""Tests for src/rl_gym/gym_env.py (P1.8).

`python -m unittest discover tests`. These build real FleetPy simulations, so they are slower
than the other suites -- reset() is roughly half a second. Episode-length runs stay out; the
full rollout is P1.10.
"""

import gc
import os
import shutil
import sys
import tempfile
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym  # noqa: E402

from src.misc.globals import (  # noqa: E402
    G_DAY_DIR,
    G_OP_MODULE,
    G_RL_MODE,
    G_RQ_FILE,
    G_SIM_ENV,
    G_SKIP_OUTPUT,
    G_STUDY_NAME,
)
from src.rl_gym.day_rotation import (  # noqa: E402
    PASS_STREAM_TAG,
    day_for_episode,
    pass_permutation,
    positions_per_pass,
    training_days,
)
from src.rl_gym.policies import greedy_action as policy_greedy_action  # noqa: E402,F401
from src.rl_gym.gym_env import (  # noqa: E402
    MAX_EPISODE_SEED,
    SDPDPAssignmentEnv,
    derive_study_name,
    parse_day_stem,
)

SCS = os.path.join(os.path.dirname(__file__), "..", "studies", "example_study", "scenarios")
CORPUS_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "demand",
                           "example_demand", "matched", "example_network")
CORPUS_DIR = os.path.join(CORPUS_ROOT, "corpus")
K = 8
REQUEST_BLOCK = 24          # §5.2: 4 coords + n_pax + 3 dow + 10 month + 6 zone


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
        # 3K candidate + 24 request (P2.9) + 4 global; 52 at K = 8
        self.assertEqual(env.observation_space.shape, (3 * K + REQUEST_BLOCK + 4,))
        self.assertEqual(env.observation_space.dtype, np.float32)
        self.assertTrue(np.all(env.observation_space.low == -np.inf))
        self.assertTrue(np.all(env.observation_space.high == np.inf))

    def test_observation_space_follows_k_without_hardcoding(self):
        # the request block is K-independent, so only the candidate block scales
        for k in (2, 4, 8):
            with self.subTest(K=k):
                env = SDPDPAssignmentEnv(make_cfg(K=k))
                self.assertEqual(env.observation_space.shape, (3 * k + REQUEST_BLOCK + 4,))
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


class TestDayRotation(unittest.TestCase):
    """D10's rotation, as pure functions -- no simulation built (P2.3).

    These are the 3-worker / 10-day checks P2.3's verification item 3 names. Driving them
    through real episodes would be 20 full simulations, about 13 minutes; here they are
    milliseconds, and the rotation is exercised independently of FleetPy.
    """

    DAYS = [f"2024-01-{d:02d}" for d in (2, 3, 4, 9, 10, 11, 16, 17, 18, 23)]
    SEED = 20260913
    NW = 3

    def test_ten_days_three_workers_shares_are_four_three_three(self):
        shares = [positions_per_pass(len(self.DAYS), self.NW, e) for e in range(self.NW)]
        self.assertEqual(shares, [4, 3, 3])
        self.assertEqual(sum(shares), len(self.DAYS))

    def _sequence(self, env_id, n_episodes):
        return [day_for_episode(self.DAYS, self.SEED, self.NW, env_id, i)
                for i in range(n_episodes)]

    def test_two_full_passes_visit_every_day_exactly_twice(self):
        seen = []
        for env_id in range(self.NW):
            per_pass = positions_per_pass(len(self.DAYS), self.NW, env_id)
            seen += [day for day, _p, _pos in self._sequence(env_id, 2 * per_pass)]
        self.assertEqual(len(seen), 2 * len(self.DAYS))
        self.assertEqual(sorted(set(seen)), sorted(self.DAYS))
        for day in self.DAYS:
            self.assertEqual(seen.count(day), 2, f"{day} visited {seen.count(day)} times")

    def test_one_pass_partitions_the_days_across_workers(self):
        # within a pass no day may go to two workers, which is what "assigned exactly once"
        # means and what P2.15's pass criterion 5 checks on the real run
        pass0 = []
        for env_id in range(self.NW):
            per_pass = positions_per_pass(len(self.DAYS), self.NW, env_id)
            for day, p, _pos in self._sequence(env_id, per_pass):
                self.assertEqual(p, 0)
                pass0.append(day)
        self.assertEqual(sorted(pass0), sorted(self.DAYS))

    def test_sequence_is_deterministic_given_base_seed(self):
        self.assertEqual(self._sequence(0, 8), self._sequence(0, 8))

    def test_different_base_seed_gives_a_different_order(self):
        other = [day_for_episode(self.DAYS, self.SEED + 1, self.NW, 0, i)[0] for i in range(4)]
        mine = [day for day, _p, _pos in self._sequence(0, 4)]
        self.assertNotEqual(mine, other)

    def test_pass_permutation_matches_d10s_seed_sequence(self):
        for p in (0, 1, 5):
            ss = np.random.SeedSequence([self.SEED, 1010, p + 1])
            want = list(np.random.default_rng(ss).permutation(list(self.DAYS)))
            self.assertEqual(pass_permutation(self.DAYS, self.SEED, p), want)

    def test_pass_stream_tag_and_offset(self):
        # SeedSequence pads with zeros, so [seed, 1010, 0] would be the same stream as
        # [seed, 1010]; p + 1 is what keeps them apart (D10)
        self.assertEqual(PASS_STREAM_TAG, 1010)
        zero = np.random.SeedSequence([self.SEED, 1010, 0]).generate_state(4).tolist()
        bare = np.random.SeedSequence([self.SEED, 1010]).generate_state(4).tolist()
        self.assertEqual(zero, bare)
        first = np.random.SeedSequence([self.SEED, 1010, 1]).generate_state(4).tolist()
        self.assertNotEqual(first, bare)

    def test_pass_stream_never_collides_with_a_worker_episode_stream(self):
        passes = {tuple(np.random.SeedSequence([self.SEED, 1010, p + 1]).generate_state(4))
                  for p in range(80)}
        workers = {tuple(np.random.SeedSequence([self.SEED, e]).generate_state(4))
                   for e in range(16)}
        self.assertEqual(len(passes), 80)
        self.assertEqual(passes & workers, set())

    def test_env_id_at_or_beyond_n_workers_raises(self):
        # a worker numbered outside the split owns no position in any pass, so it would
        # silently run nothing of the rotation
        for env_id in (self.NW, self.NW + 1, 99):
            with self.assertRaises(ValueError) as cm:
                day_for_episode(self.DAYS, self.SEED, self.NW, env_id, 0)
            self.assertIn("env_id", str(cm.exception))

    def test_negative_env_id_and_bad_arguments_raise(self):
        with self.assertRaises(ValueError):
            day_for_episode(self.DAYS, self.SEED, self.NW, -1, 0)
        with self.assertRaises(ValueError):
            day_for_episode(self.DAYS, self.SEED, 0, 0, 0)
        with self.assertRaises(ValueError):
            day_for_episode(self.DAYS, self.SEED, self.NW, 0, -1)
        with self.assertRaises(ValueError):
            pass_permutation(self.DAYS, self.SEED, -1)

    def test_more_workers_than_days_raises(self):
        with self.assertRaises(ValueError):
            day_for_episode(self.DAYS[:2], self.SEED, 5, 4, 0)

    def test_training_days_lists_csv_only(self):
        # §2.9: macOS rewrites .DS_Store in these directories on its own, and an entry-based
        # listing would turn it into a phantom demand day
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("2024-01-02.csv", "2024-01-03.csv", ".DS_Store", "notes.txt"):
                open(os.path.join(tmp, name), "w").close()
            self.assertEqual(training_days(tmp), ["2024-01-02", "2024-01-03"])

    def test_training_days_rejects_an_empty_or_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                training_days(tmp)
            with self.assertRaises(ValueError):
                training_days(os.path.join(tmp, "nope"))

    def test_real_corpus_is_101_training_days(self):
        days = training_days(os.path.join(CORPUS_ROOT, "corpus"))
        self.assertEqual(len(days), 101)
        self.assertEqual(days, sorted(days))
        self.assertNotIn("2024-06-20", days, "an evaluation day must not be in corpus/")


class TestDaySourceConfig(unittest.TestCase):
    """Config validation for the day and seed sources (P2.3). No simulation is built."""

    def test_day_dir_absent_leaves_the_scenario_row_alone(self):
        env = SDPDPAssignmentEnv(make_cfg())
        self.assertIsNone(env.day_dir)
        self.assertEqual(env.scenario_parameters[G_DAY_DIR], "thursday")
        self.assertEqual(env.scenario_parameters[G_RQ_FILE], "2024-06-20.csv")

    def test_episode_seed_with_base_seed_raises(self):
        with self.assertRaises(ValueError) as cm:
            SDPDPAssignmentEnv(make_cfg(episode_seed=42, base_seed=7))
        self.assertIn("episode_seed", str(cm.exception))

    def test_episode_seed_out_of_range_raises(self):
        for bad in (-1, MAX_EPISODE_SEED + 1):
            with self.assertRaises(ValueError):
                SDPDPAssignmentEnv(make_cfg(episode_seed=bad))

    def test_rotation_requires_base_seed(self):
        with self.assertRaises(ValueError) as cm:
            SDPDPAssignmentEnv(make_cfg(day_dir="corpus", n_workers=2))
        self.assertIn("base_seed", str(cm.exception))

    def test_rotation_requires_n_workers(self):
        with self.assertRaises(ValueError) as cm:
            SDPDPAssignmentEnv(make_cfg(day_dir="corpus", base_seed=7))
        self.assertIn("n_workers", str(cm.exception))

    def test_env_id_at_or_beyond_n_workers_raises_at_construction(self):
        with self.assertRaises(ValueError) as cm:
            SDPDPAssignmentEnv(make_cfg(day_dir="corpus", base_seed=7, n_workers=3, env_id=3))
        self.assertIn("env_id", str(cm.exception))

    def test_eval_pairs_rejects_the_keys_it_subsumes(self):
        for extra in ({"day": "2024-06-20"}, {"episode_seed": 42}, {"base_seed": 7}):
            with self.assertRaises(ValueError):
                SDPDPAssignmentEnv(make_cfg(day_dir="corpus_eval",
                                            eval_pairs=[("2024-06-20", 42)], **extra))

    def test_eval_pairs_requires_day_dir(self):
        with self.assertRaises(ValueError) as cm:
            SDPDPAssignmentEnv(make_cfg(eval_pairs=[("2024-06-20", 42)]))
        self.assertIn("day_dir", str(cm.exception))

    def test_eval_pairs_empty_raises(self):
        with self.assertRaises(ValueError):
            SDPDPAssignmentEnv(make_cfg(day_dir="corpus_eval", eval_pairs=[]))

    def test_eval_pairs_seed_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            SDPDPAssignmentEnv(make_cfg(day_dir="corpus_eval",
                                        eval_pairs=[("2024-06-20", MAX_EPISODE_SEED + 1)]))

    def test_unknown_day_raises_at_construction(self):
        # otherwise a typo surfaces as a path error deep inside load_demand_file
        with self.assertRaises(ValueError) as cm:
            SDPDPAssignmentEnv(make_cfg(day_dir="corpus_eval", day="2024-06-21"))
        self.assertIn("2024-06-21", str(cm.exception))

    def test_unknown_day_in_eval_pairs_raises(self):
        with self.assertRaises(ValueError):
            SDPDPAssignmentEnv(make_cfg(day_dir="corpus_eval",
                                        eval_pairs=[("2024-06-20", 42), ("1999-01-01", 7)]))

    def test_unknown_day_dir_raises(self):
        with self.assertRaises(ValueError):
            SDPDPAssignmentEnv(make_cfg(day_dir="corpus_nope", day="2024-06-20"))

    def test_training_day_is_rejected_from_corpus_eval(self):
        # a day that exists but in the other split: the directory is what decides
        with self.assertRaises(ValueError):
            SDPDPAssignmentEnv(make_cfg(day_dir="corpus_eval", day="2024-01-18"))


class TestDayStemParsing(unittest.TestCase):

    def test_weekday_and_month_from_an_iso_stem(self):
        self.assertEqual(parse_day_stem("2024-06-20"), ("Thu", 6))
        self.assertEqual(parse_day_stem("2024-01-30"), ("Tue", 1))
        self.assertEqual(parse_day_stem("2024-10-09"), ("Wed", 10))

    def test_non_iso_stem_yields_none_rather_than_raising(self):
        # the reference path's rq_file need not be ISO-named; P2.9 raises when it needs these
        for stem in ("thursday", "", "2024-13-01", None):
            self.assertEqual(parse_day_stem(stem), (None, None))

    def test_weekday_outside_the_corpus_vocabulary_is_named_not_mapped(self):
        self.assertEqual(parse_day_stem("2024-06-22"), ("Sat", 6))


class TestEvalPairsRotation(unittest.TestCase):
    """`eval_pairs` cycling, exercised through reset() without stepping (P2.3 item 5)."""

    PAIRS = [("2024-06-20", 42), ("2024-01-11", 7), ("2024-10-01", 1234)]

    def test_pairs_run_in_order_and_wrap_around(self):
        env = SDPDPAssignmentEnv(make_cfg(day_dir="corpus_eval", eval_pairs=self.PAIRS))
        try:
            for i in range(len(self.PAIRS) * 2 + 1):
                env.reset()
                want_day, want_seed = self.PAIRS[i % len(self.PAIRS)]
                s = env.episode_summary()
                self.assertEqual(s["day"], want_day, f"episode {i + 1}")
                self.assertEqual(s["random_seed"], want_seed, f"episode {i + 1}")
                self.assertEqual(s["day_dir"], "corpus_eval")
        finally:
            env.close()


class TestCorpusCheckAtConstruction(unittest.TestCase):
    """Trap 16: the env refuses to start when `corpus/` no longer matches the manifest (P2.9).

    P2.6's baseline, its month means and its SD were all computed over one particular set of
    days, and the request block's vocabularies come from the same manifest. A baseline read
    against a different corpus produces plausible-looking anomalies that are simply wrong, and
    nothing else would notice.

    Each test copies the real corpus into a temporary directory and points `corpus_dir` there,
    so the committed corpus is never touched.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.pristine = os.path.join(cls._tmp.name, "corpus")
        shutil.copytree(CORPUS_DIR, cls.pristine)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _copy(self):
        """A fresh copy of the corpus for one test to mutate."""
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        target = os.path.join(d, "corpus")
        shutil.copytree(self.pristine, target)
        return target

    def test_an_untouched_copy_constructs(self):
        env = SDPDPAssignmentEnv(make_cfg(corpus_dir=self._copy()))
        self.addCleanup(env.close)
        self.assertEqual(len(env.manifest["files"]), 101)

    def test_removing_one_file_raises_and_names_it(self):
        corpus = self._copy()
        os.remove(os.path.join(corpus, "2024-03-05.csv"))
        with self.assertRaises(ValueError) as cm:
            SDPDPAssignmentEnv(make_cfg(corpus_dir=corpus))
        msg = str(cm.exception)
        self.assertIn("2024-03-05.csv", msg)
        self.assertIn("missing", msg)

    def test_editing_one_file_raises_and_names_it(self):
        corpus = self._copy()
        path = os.path.join(corpus, "2024-07-23.csv")
        with open(path, "a") as fh:
            fh.write("\n")                       # one byte, nothing semantic
        with self.assertRaises(ValueError) as cm:
            SDPDPAssignmentEnv(make_cfg(corpus_dir=corpus))
        msg = str(cm.exception)
        self.assertIn("2024-07-23.csv", msg)
        self.assertIn("changed", msg)

    def test_adding_a_day_raises_and_names_it(self):
        corpus = self._copy()
        shutil.copy(os.path.join(corpus, "2024-07-23.csv"),
                    os.path.join(corpus, "2024-12-25.csv"))
        with self.assertRaises(ValueError) as cm:
            SDPDPAssignmentEnv(make_cfg(corpus_dir=corpus))
        msg = str(cm.exception)
        self.assertIn("2024-12-25.csv", msg)
        self.assertIn("extra", msg)

    def test_a_stray_ds_store_leaves_construction_working(self):
        # §2.9's *.csv-only rule, and the test P2.2 promised: macOS writes .DS_Store into these
        # directories on its own, and it must not look like a corpus change
        corpus = self._copy()
        with open(os.path.join(corpus, ".DS_Store"), "wb") as fh:
            fh.write(b"\x00\x01\x02")
        with open(os.path.join(corpus, "notes.txt"), "w") as fh:
            fh.write("ignore me")
        env = SDPDPAssignmentEnv(make_cfg(corpus_dir=corpus))
        self.addCleanup(env.close)
        self.assertEqual(len(env.manifest["files"]), 101)

    def test_a_missing_corpus_directory_raises(self):
        with self.assertRaises(ValueError):
            SDPDPAssignmentEnv(make_cfg(corpus_dir=os.path.join(self._tmp.name, "nope")))

    def test_the_error_explains_the_consequence_and_the_fix(self):
        corpus = self._copy()
        os.remove(os.path.join(corpus, "2024-01-09.csv"))
        with self.assertRaises(ValueError) as cm:
            SDPDPAssignmentEnv(make_cfg(corpus_dir=corpus))
        msg = str(cm.exception)
        self.assertIn("invalid", msg)                       # what it means
        self.assertIn("build_arrival_baseline", msg)        # what to do


class TestRequestBlockInTheEnv(unittest.TestCase):
    """The block's wiring: shape, layout, and the per-episode injection (P2.9)."""

    @classmethod
    def setUpClass(cls):
        cls.env = SDPDPAssignmentEnv(make_cfg())
        cls.obs, cls.info = cls.env.reset()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_observation_length_is_fifty_two(self):
        self.assertEqual(self.env.observation_space.shape, (52,))
        self.assertEqual(self.obs.shape, (52,))

    def test_the_request_block_adds_twenty_four(self):
        self.assertEqual(self.env._request_observer.output_length, 24)

    def test_block_order_is_candidate_request_global(self):
        names = list(self.env.observation_layout)
        self.assertEqual(names[:3], ["is_valid", "delta_cfv", "offered_wait"])
        self.assertEqual(names[3:8], ["o_lat", "o_lon", "d_lat", "d_lon", "n_pax"])
        self.assertEqual(names[-4:],
                         ["time_sin", "time_cos", "day_progress", "idle_fraction"])

    def test_the_layout_still_tiles(self):
        slices = list(self.env.observation_layout.values())
        self.assertEqual(slices[0].start, 0)
        self.assertEqual(slices[-1].stop, 52)
        for a, b in zip(slices, slices[1:]):
            self.assertEqual(a.stop, b.start)

    def test_the_one_hots_match_the_demand_filename(self):
        L = self.env.observation_layout
        # the reference scenario runs thursday/2024-06-20.csv
        np.testing.assert_array_equal(self.obs[L["dow"]], [0.0, 0.0, 1.0])       # Thu
        np.testing.assert_array_equal(
            self.obs[L["month"]], [0.0] * 5 + [1.0] + [0.0] * 4)                # June
        self.assertEqual(self.env.episode_summary()["weekday"], "Thu")
        self.assertEqual(self.env.episode_summary()["month"], 6)

    def test_coordinates_are_inside_the_unit_box(self):
        L = self.env.observation_layout
        for name in ("o_lat", "o_lon", "d_lat", "d_lon"):
            v = float(self.obs[L[name]][0])
            self.assertGreaterEqual(v, -1.0, name)
            self.assertLessEqual(v, 1.0, name)

    def test_the_observation_is_finite_and_in_space(self):
        self.assertTrue(np.all(np.isfinite(self.obs)))
        self.assertTrue(self.env.observation_space.contains(self.obs))

    def test_zone_counts_accumulate_over_the_episode(self):
        L = self.env.observation_layout
        obs = self.obs
        first = float(obs[L["zone_arrivals_60"]][0])
        for _ in range(80):
            # policy_greedy_action, not this module's local greedy_action(env, obs) helper --
            # the two share a name and take different arguments
            obs, _r, term, _t, _i = self.env.step(policy_greedy_action(
                obs, self.env.action_masks(), self.env.observation_layout))
            if term:
                break
        later = float(obs[L["zone_arrivals_60"]][0])
        self.assertEqual(first, 0.0, "the first decision's window is empty")
        self.assertGreater(later, 0.0, "zone arrivals must accumulate")

    def test_the_observer_is_reattached_on_every_reset(self):
        env = SDPDPAssignmentEnv(make_cfg())
        self.addCleanup(env.close)
        for _ in range(2):
            env.reset()
            # a stale tracker from the previous episode would still answer, so compare identity
            self.assertIs(env._request_observer._stats, env._stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
