"""Tests for train_sdpdp.py's venv wiring (P2.12).

**This is the only test module that imports `torch` / `stable_baselines3` / `sb3_contrib` and
the only one that spawns `SubprocVecEnv` workers.** Both are deliberate: §8a's OpenMP
import-order constraint is a `torch` problem, and keeping it in one file keeps it out of
`tests/test_rl_gym_env.py` and the rest of the suite. P2.13 extends this file.

Slow by nature -- each worker builds a real FleetPy simulation -- so the wrapped venv is built
once per class and the tests take a handful of steps rather than whole episodes.

    python -m unittest tests.test_rl_gym_train
"""

# ---------------------------------------------------------------------------------------- #
# IMPORT ORDER IS LOAD-BEARING. DO NOT ALPHABETISE. DO NOT LET isort/ruff REORDER THIS BLOCK.
# numpy MUST come before torch / stable_baselines3 / sb3_contrib, or the process dies with
# "OMP: Error #15 ... Abort trap: 6". See docs/SDPDP_GYM_SPEC_v2.md §8a and train_sdpdp.py.
import numpy as np  # noqa: I001  -- must stay first
# ---------------------------------------------------------------------------------------- #

import multiprocessing as mp
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------------------- #
# THE FORKSERVER MUST PRELOAD numpy, NOT `__main__` (§8a).
#
# `SubprocVecEnv` prefers the forkserver start method, and `multiprocessing` preloads
# `['__main__']` into the forkserver by default. For a *script* like train_sdpdp.py that is
# exactly what carries the numpy-first import order into every worker. For a *test module* it
# does two wrong things at once: under `python -m unittest ...` `__main__` is unittest's own
# `__main__.py`, so the preload (a) establishes no numpy-first ordering, and the workers died
# with "OMP: Error #15 ... Abort trap: 6", and (b) would execute `unittest.main()` inside the
# forkserver process, re-entering the suite.
#
# Preloading numpy by name fixes both: the forkserver image has numpy loaded before any worker
# forks from it, and nothing re-enters the test runner. Measured: without this the three
# classes below fail in `setUpClass` under `-m unittest` while passing when the file is run as
# a script; with it, both ways pass.
mp.set_forkserver_preload(["numpy"])
# ---------------------------------------------------------------------------------------- #

from sb3_contrib.common.maskable.utils import (  # noqa: E402
    get_action_masks,
    is_masking_supported,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize  # noqa: E402

from src.rl_gym.policies import greedy_action  # noqa: E402
import train_sdpdp  # noqa: E402

SCS = os.path.join(os.path.dirname(__file__), "..", "studies", "example_study", "scenarios")
K = 8
N_WORKERS = 2


def cfg(**overrides):
    c = {
        "constant_cfg_path": os.path.join(SCS, "constant_config_depot_cali_sc_1.csv"),
        "scenario_cfg_path": os.path.join(SCS, "example_depot_cali_sc_1.csv"),
        "K": K,
        "skip_output": True,
        "base_seed": 4242,
        "gamma": 0.999,
        # day_dir absent on purpose: the reference day, so no corpus rotation is involved and
        # the workers differ only by their episode seed (D7)
    }
    c.update(overrides)
    return c


class TestVecNormalizeWiring(unittest.TestCase):
    """§5.3's settings, and the save/load pair that keeps a checkpoint with its statistics."""

    @classmethod
    def setUpClass(cls):
        cls.venv = train_sdpdp.build_venv(cfg(), N_WORKERS, normalize=True)
        cls.addClassCleanup(cls.venv.close)
        cls.obs = cls.venv.reset()

    def test_it_is_a_vecnormalize_over_a_subprocvecenv(self):
        self.assertIsInstance(self.venv, VecNormalize)
        self.assertIsInstance(self.venv.venv, SubprocVecEnv)
        self.assertEqual(self.venv.num_envs, N_WORKERS)

    def test_the_settings_are_sect_5_3s(self):
        self.assertTrue(self.venv.norm_obs)
        self.assertTrue(self.venv.norm_reward)
        self.assertEqual(self.venv.clip_obs, 10.0)
        self.assertTrue(self.venv.training)

    def test_gamma_comes_from_the_config_not_sb3s_default(self):
        """§5.3: VecNormalize's gamma must equal PPO's, from the same config value.

        SB3's default is 0.99, which would normalise rewards against an effective horizon of
        ~100 decisions while PPO discounts over ~1000 (§6.5). This is the check that the
        default is not silently in force.
        """
        self.assertEqual(self.venv.gamma, 0.999)
        self.assertNotEqual(self.venv.gamma, 0.99)
        self.assertEqual(self.venv.gamma, cfg()["gamma"])

    def test_a_different_gamma_reaches_the_wrapper(self):
        venv = train_sdpdp.build_venv(cfg(gamma=0.95), 1, normalize=True)
        try:
            self.assertEqual(venv.gamma, 0.95)
        finally:
            venv.close()

    def test_normalize_false_gives_the_bare_venv(self):
        venv = train_sdpdp.build_venv(cfg(), 1, normalize=False)
        try:
            self.assertIsInstance(venv, SubprocVecEnv)
            self.assertNotIsInstance(venv, VecNormalize)
        finally:
            venv.close()

    def test_the_observation_space_is_the_full_one_hundred_and_thirty_nine(self):
        self.assertEqual(self.venv.observation_space.shape, (139,))
        self.assertEqual(self.obs.shape, (N_WORKERS, 139))

    # ---- save / load ------------------------------------------------ #

    def test_save_load_round_trip_reproduces_identical_normalised_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = os.path.join(tmp, "model_000042.zip")
            path = train_sdpdp.save_vecnormalize(self.venv, checkpoint)
            self.assertEqual(os.path.basename(path), train_sdpdp.VECNORMALIZE_FILE)
            self.assertEqual(os.path.dirname(path), tmp)

            fixed = np.linspace(-5.0, 5.0, 139, dtype=np.float32).reshape(1, 139)
            before = self.venv.normalize_obs(fixed)

            reloaded = train_sdpdp.load_vecnormalize(checkpoint, self.venv.venv)
            try:
                np.testing.assert_array_equal(reloaded.normalize_obs(fixed), before)
                np.testing.assert_array_equal(reloaded.obs_rms.mean, self.venv.obs_rms.mean)
                np.testing.assert_array_equal(reloaded.obs_rms.var, self.venv.obs_rms.var)
                self.assertEqual(reloaded.obs_rms.count, self.venv.obs_rms.count)
            finally:
                # detach without closing the shared workers: the class owns them
                reloaded.venv = None

    def test_the_evaluation_loader_freezes_the_statistics_and_the_reward(self):
        """§5.3: at evaluation, `training = False` and `norm_reward = False`.

        Evaluating with still-moving statistics, or with a normalised reward reported as if it
        were the real one, is the silent failure §5.3 names.
        """
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = os.path.join(tmp, "model.zip")
            train_sdpdp.save_vecnormalize(self.venv, checkpoint)
            reloaded = train_sdpdp.load_vecnormalize(checkpoint, self.venv.venv)
            try:
                self.assertFalse(reloaded.training)
                self.assertFalse(reloaded.norm_reward)
                self.assertTrue(reloaded.norm_obs)       # observations are still normalised
            finally:
                reloaded.venv = None

    def test_loading_for_training_leaves_both_flags_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = os.path.join(tmp, "model.zip")
            train_sdpdp.save_vecnormalize(self.venv, checkpoint)
            reloaded = train_sdpdp.load_vecnormalize(checkpoint, self.venv.venv,
                                                     for_evaluation=False)
            try:
                self.assertTrue(reloaded.training)
                self.assertTrue(reloaded.norm_reward)
            finally:
                reloaded.venv = None

    def test_saving_an_unwrapped_venv_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError) as cm:
                train_sdpdp.save_vecnormalize(self.venv.venv, os.path.join(tmp, "m.zip"))
            self.assertIn("VecNormalize", str(cm.exception))

    def test_loading_without_the_statistics_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as cm:
                train_sdpdp.load_vecnormalize(os.path.join(tmp, "model.zip"), self.venv.venv)
            self.assertIn(train_sdpdp.VECNORMALIZE_FILE, str(cm.exception))


class TestMasksPassThroughVecNormalize(unittest.TestCase):
    """Ritun's check: the wrapper must not touch the masks.

    `get_action_masks` on a `VecEnv` calls `env_method("action_masks")`; `VecNormalize` is a
    `VecEnv` that does not override `env_method`, and `VecEnvWrapper.env_method` forwards to
    `self.venv`. So the masks should arrive exactly as each worker reports them. This is the
    test that says so rather than inferring it from the class hierarchy -- if it ever stopped
    being true, MaskablePPO would be masking against a transformed array and nothing would
    raise.
    """

    @classmethod
    def setUpClass(cls):
        cls.venv = train_sdpdp.build_venv(cfg(), N_WORKERS, normalize=True)
        cls.addClassCleanup(cls.venv.close)
        cls.obs = cls.venv.reset()

    def test_masking_is_supported_through_the_wrapper(self):
        # what MaskablePPO itself calls; it goes through has_attr, also forwarded
        self.assertTrue(is_masking_supported(self.venv))
        self.assertTrue(is_masking_supported(self.venv.venv))

    def test_the_wrapped_and_unwrapped_masks_are_identical(self):
        wrapped = get_action_masks(self.venv)
        unwrapped = get_action_masks(self.venv.venv)
        direct = np.stack(self.venv.venv.env_method("action_masks"))
        self.assertEqual(wrapped.shape, (N_WORKERS, K + 1))
        self.assertEqual(wrapped.dtype, unwrapped.dtype)
        np.testing.assert_array_equal(wrapped, unwrapped)
        np.testing.assert_array_equal(wrapped, direct)

    def test_they_stay_identical_as_the_episode_advances(self):
        obs = self.obs
        rng = np.random.default_rng(0)
        for step in range(5):
            with self.subTest(step=step):
                wrapped = get_action_masks(self.venv)
                unwrapped = get_action_masks(self.venv.venv)
                np.testing.assert_array_equal(wrapped, unwrapped)
                # reading the mask must not have advanced anything: a second read agrees
                np.testing.assert_array_equal(wrapped, get_action_masks(self.venv))
                self.assertTrue(wrapped[:, K].all(), "reject must always be legal")
                actions = np.array([rng.choice(np.flatnonzero(m)) for m in wrapped],
                                   dtype=np.int64)
                obs, _rewards, _dones, _infos = self.venv.step(actions)

    def test_the_masks_are_booleans_not_normalised_floats(self):
        masks = get_action_masks(self.venv)
        self.assertEqual(masks.dtype, np.bool_)
        self.assertTrue(set(np.unique(masks)) <= {True, False})


class TestNormalisedObservationsNeverReachGreedy(unittest.TestCase):
    """Ritun's second check: `greedy_action`'s `is_valid` guard refuses a normalised vector.

    P2.8 added that guard precisely for this, but until now it was only tested against a
    synthetic normalised vector. The danger it guards is quiet: after normalisation each
    `delta_cfv` slot is scaled by its **own** running mean and variance, so an argmin across
    slots compares z-scores drawn from different distributions instead of costs -- plausible
    actions that are not greedy, which no byte-for-byte gate would catch.
    """

    @classmethod
    def setUpClass(cls):
        cls.venv = train_sdpdp.build_venv(cfg(), N_WORKERS, normalize=True)
        cls.addClassCleanup(cls.venv.close)
        cls.norm_obs = cls.venv.reset()
        cls.layout = cls.venv.env_method("__getattribute__", "observation_layout")[0]

    def _greedy(self, obs_row, worker=0):
        masks = get_action_masks(self.venv)
        return greedy_action(np.asarray(obs_row, dtype=np.float32), masks[worker], self.layout)

    def test_the_very_first_normalised_observation_is_already_refused(self):
        """The earliest point a mistaken `greedy_action(normalised, ...)` could be reached.

        Measured rather than predicted, because the arithmetic is not what one would guess:
        SB3's `RunningMeanStd` starts at `count = 1e-4`, `mean = 0`, `var = 1`, so after the
        first update the mean is pulled *almost* to the sample and the variance is tiny but
        non-zero. A raw `is_valid` of 1.0 therefore comes out at about **0.005** with two
        workers (0.007 with one) — neither 0.0 nor 1.0, so the guard fires. An earlier draft of
        this test asserted the whole vector was zeros, on the theory that one sample gives zero
        variance; it does not, and `count` starting at epsilon is why.
        """
        row = self.norm_obs[0]
        is_valid = row[self.layout["is_valid"]]
        self.assertFalse(np.all(np.isin(is_valid, (0.0, 1.0))),
                         f"is_valid came out exactly 0/1: {is_valid}")
        self.assertTrue(np.all(np.abs(is_valid) < 0.5),
                        f"expected is_valid crushed towards 0 by normalisation: {is_valid}")
        with self.assertRaises(ValueError) as cm:
            self._greedy(row)
        self.assertIn("VecNormalize", str(cm.exception))

    def test_a_normalised_observation_is_refused_once_the_statistics_have_moved(self):
        obs = self.norm_obs
        rng = np.random.default_rng(1)
        for _ in range(5):
            masks = get_action_masks(self.venv)
            actions = np.array([rng.choice(np.flatnonzero(m)) for m in masks], dtype=np.int64)
            obs, _r, _d, _i = self.venv.step(actions)
        is_valid = obs[0][self.layout["is_valid"]]
        # the statistics are no longer degenerate, so is_valid is some non-0/1 float
        self.assertFalse(np.all(np.isin(is_valid, (0.0, 1.0))),
                         f"is_valid still looks unnormalised: {is_valid}")
        with self.assertRaises(ValueError) as cm:
            self._greedy(obs[0])
        message = str(cm.exception)
        self.assertIn("VecNormalize", message)
        self.assertIn("is_valid", message)

    def test_the_raw_observation_from_the_same_wrapper_still_passes(self):
        """The control: the guard refuses normalised input, not all input.

        Without this the two tests above would pass if `greedy_action` simply raised on
        everything.
        """
        raw = self.venv.get_original_obs()
        masks = get_action_masks(self.venv)
        for worker in range(N_WORKERS):
            with self.subTest(worker=worker):
                is_valid = raw[worker][self.layout["is_valid"]]
                np.testing.assert_array_equal(is_valid.astype(bool), masks[worker][:K])
                action = greedy_action(np.asarray(raw[worker], dtype=np.float32),
                                       masks[worker], self.layout)
                self.assertTrue(masks[worker][action])

    def test_the_wrapper_really_is_transforming_the_observation(self):
        # so the comparison above is not vacuous
        raw = self.venv.get_original_obs()
        self.assertFalse(np.allclose(raw[0], self.venv.normalize_obs(raw)[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
