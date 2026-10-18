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

import json
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------------------- #
# THE FORKSERVER PRELOAD IS NOT SET HERE -- `train_sdpdp.build_venv` does it (P2.13).
#
# It has to happen somewhere: `SubprocVecEnv` prefers the forkserver start method, and
# `multiprocessing` preloads `['__main__']` into it by default. For a *script* that carries
# §8a's numpy-first import order into every worker; for a *test module* it carries nothing,
# because under `python -m unittest ...` `__main__` is unittest's own `__main__.py` -- and the
# workers then die with "OMP: Error #15 ... Abort trap: 6" (measured in P2.12).
#
# P2.12 fixed that with a `mp.set_forkserver_preload(["numpy"])` call right here. P2.13 moved
# it into `prepare_multiprocessing`, called by every venv builder, so the training env, the
# evaluation env and P2.14's harness all get it without a caller remembering. Its absence from
# this module is therefore the proof: every class below builds a venv from a unittest module
# with no preload call of its own, and `TestForkserverPreload` pins the mechanism rather than
# resting on the absence of a crash.
# ---------------------------------------------------------------------------------------- #

from sb3_contrib.common.maskable.utils import (  # noqa: E402
    get_action_masks,
    is_masking_supported,
)
from stable_baselines3.common.vec_env import (  # noqa: E402
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
)

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
            # P2.16: named after the checkpoint, not a shared name per folder
            self.assertEqual(os.path.basename(path), "model_000042_vecnormalize.pkl")
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
            self.assertIn("model_vecnormalize.pkl", str(cm.exception))

    def test_an_explicit_statistics_path_is_used_when_given(self):
        """The `--stats` contract (P2.16): a named file, still pairing-checked."""
        with tempfile.TemporaryDirectory() as tmp:
            elsewhere = os.path.join(tmp, "somewhere_else.pkl")
            self.venv.save(elsewhere)
            checkpoint = os.path.join(tmp, "model.zip")          # no file beside it
            reloaded = train_sdpdp.load_vecnormalize(checkpoint, self.venv.venv,
                                                     stats_path=elsewhere)
            try:
                self.assertEqual(reloaded.obs_rms.count, self.venv.obs_rms.count)
                self.assertFalse(reloaded.training)
            finally:
                reloaded.venv = None

    def test_the_resume_path_raises_when_the_statistics_are_missing(self):
        """P2.16's second fix: `train()`'s resume branch used to continue on fresh statistics.

        It now goes through `load_vecnormalize(..., for_evaluation=False)`, so a folder with no
        statistics raises instead of silently restarting them at count 1e-4, mean 0, var 1 --
        which P2.12 measured turning a raw `is_valid` of 1.0 into about 0.005.
        """
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                train_sdpdp.load_vecnormalize(os.path.join(tmp, "model_ep00012.zip"),
                                              self.venv.venv, for_evaluation=False)


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


class TestForkserverPreload(unittest.TestCase):
    """P2.13: the preload belongs to the venv builders, not to the caller (Ritun's point 1).

    The classes above are the end-to-end proof -- they build venvs from a unittest module with
    no preload call in this file and their workers start cleanly, which is exactly what failed
    before P2.12. These two tests pin the mechanism, so a future refactor that drops the call
    fails here rather than as an `OMP: Error #15` in whatever module spawns next.
    """

    def test_this_module_does_not_set_the_preload_at_import_time(self):
        """No **module-level** preload call: that is `build_venv`'s job now (P2.13).

        Parsed rather than grepped, because the test below legitimately calls
        `set_forkserver_preload` inside a function to set the broken default and prove
        `build_venv` overrides it -- a grep would flag its own fixture.
        """
        import ast
        path = os.path.join(os.path.dirname(__file__), "test_rl_gym_train.py")
        tree = ast.parse(open(path).read())
        top_level = [
            node for node in tree.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and "set_forkserver_preload" in ast.dump(node.value.func)
        ]
        self.assertEqual(top_level, [],
                         "the preload is train_sdpdp.prepare_multiprocessing's job (P2.13)")

    def test_prepare_multiprocessing_sets_numpy_as_the_preload(self):
        import multiprocessing.forkserver as fs
        train_sdpdp.prepare_multiprocessing()
        self.assertEqual(list(fs._forkserver._preload_modules), ["numpy"])

    def test_build_venv_prepares_multiprocessing(self):
        import multiprocessing.forkserver as fs
        mp.set_forkserver_preload(["__main__"])          # the broken default
        venv = train_sdpdp.build_venv(cfg(), 1, normalize=False)
        try:
            self.assertEqual(list(fs._forkserver._preload_modules), ["numpy"])
        finally:
            venv.close()


class TestTrainingTimeEvaluationStaysOnValidation(unittest.TestCase):
    """The training side of P2.17's split. No training happens here.

    A callback that evaluated on the 14 test days would make them a selection criterion, and
    nothing downstream would show it -- the KPI table, the scalars and the checkpoints would
    all look normal. So the guard is a refusal, and it has to fire **before** anything is
    built.
    """

    def test_the_eval_venv_default_directory_is_the_validation_half(self):
        """This default is the guard: it is `train()`'s only control over the directory."""
        import inspect
        default = inspect.signature(train_sdpdp.build_eval_venv).parameters[
            "eval_day_dir"].default
        self.assertEqual(default, "corpus_val")
        self.assertEqual(train_sdpdp.VAL_DAY_DIR, "corpus_val")
        self.assertEqual(train_sdpdp.TEST_DAY_DIR, "corpus_eval")

    def test_the_verification_runs_pairs_are_all_validation_days(self):
        from src.rl_gym.corpus import split_of_day
        for day, _seed in train_sdpdp.VERIFY_EVAL_PAIRS:
            with self.subTest(day):
                self.assertEqual(split_of_day(day), "val")

    def test_validation_pairs_pass_the_preflight(self):
        self.assertIsNone(train_sdpdp.check_eval_pairs_split(
            train_sdpdp.VERIFY_EVAL_PAIRS))
        self.assertIsNone(train_sdpdp.check_eval_pairs_split(None))
        self.assertIsNone(train_sdpdp.check_eval_pairs_split([]))

    def test_a_test_day_in_the_callback_pairs_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            train_sdpdp.check_eval_pairs_split([("2024-01-11", 11), ("2024-01-30", 12)])
        message = str(ctx.exception)
        self.assertIn("2024-01-30", message)
        self.assertIn("not validation days", message)
        # only the offending day is named -- the message says what to fix, not what it saw
        self.assertNotIn("2024-01-11", message)

    def test_a_training_day_in_the_callback_pairs_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            train_sdpdp.check_eval_pairs_split([("2024-01-18", 11)])
        self.assertIn("2024-01-18", str(ctx.exception))

    def test_pointing_the_callback_at_the_test_directory_is_refused_by_name(self):
        with self.assertRaises(ValueError) as ctx:
            train_sdpdp.check_eval_pairs_split([("2024-01-11", 11)],
                                               eval_day_dir="corpus_eval")
        self.assertIn("selection criterion", str(ctx.exception))

    def test_the_preflight_runs_before_any_venv_is_built(self):
        """`train()` must refuse without constructing a SubprocVecEnv (P2.16's ordering).

        Patching `build_venv` to fail loudly is what makes this a measurement: if the check
        ever moves after it, the test fails with that sentinel instead of the split error.
        """
        def exploding_build_venv(*_a, **_kw):
            raise AssertionError("build_venv was reached before the split check")

        cfg = {**train_sdpdp.DEFAULT_CFG, "base_seed": 7,
               "reward_weights": dict(train_sdpdp.PHASE2_REWARD_WEIGHTS)}
        original = train_sdpdp.build_venv
        train_sdpdp.build_venv = exploding_build_venv
        try:
            with self.assertRaises(ValueError) as ctx:
                train_sdpdp.train(cfg, n_workers=1, total_timesteps=1,
                                  eval_pairs=[("2024-01-30", 1)], verbose=False)
            self.assertIn("2024-01-30", str(ctx.exception))
        finally:
            train_sdpdp.build_venv = original


class TestEvalPairPartitioning(unittest.TestCase):
    """The pairs must be split the way `evaluate_policy` spreads episodes (P2.13).

    Pure arithmetic, so it runs in milliseconds; the end-to-end version is `--verify-run`.
    """

    def setUp(self):
        from src.rl_gym.callbacks import partition_eval_pairs
        self.partition = partition_eval_pairs
        self.pairs = [(f"2024-0{i}-01", 100 + i) for i in range(1, 7)]

    @staticmethod
    def sb3_targets(n_eval_episodes, n_envs):
        """`evaluate_policy`'s own formula, verified against sb3-contrib 2.9.0 in P2.13."""
        return [(n_eval_episodes + i) // n_envs for i in range(n_envs)]

    def test_slice_sizes_match_sb3s_episode_targets(self):
        for n_envs in (1, 2, 3, 4, 5, 6):
            with self.subTest(n_envs=n_envs):
                slices = self.partition(self.pairs, n_envs)
                self.assertEqual([len(s) for s in slices],
                                 self.sb3_targets(len(self.pairs), n_envs))

    def test_the_slices_concatenate_back_to_the_input_in_order(self):
        for n_envs in (1, 2, 3, 6):
            with self.subTest(n_envs=n_envs):
                slices = self.partition(self.pairs, n_envs)
                flat = [p for s in slices for p in s]
                self.assertEqual(flat, self.pairs)

    def test_an_uneven_split_gives_the_remainder_to_the_later_workers(self):
        # 5 pairs over 2 workers: SB3 targets are [2, 3], not [3, 2]
        slices = self.partition(self.pairs[:5], 2)
        self.assertEqual([len(s) for s in slices], [2, 3])

    def test_fewer_pairs_than_workers_raises(self):
        with self.assertRaises(ValueError) as cm:
            self.partition(self.pairs[:1], 2)
        self.assertIn("at least one pair per worker", str(cm.exception))

    def test_an_empty_list_or_no_workers_raises(self):
        with self.assertRaises(ValueError):
            self.partition([], 2)
        with self.assertRaises(ValueError):
            self.partition(self.pairs, 0)

    # ---- two consecutive evaluations (Ritun's point 2) --------------- #

    def _simulate(self, pairs, n_envs, n_evaluations, extra_episodes=0):
        """Which pairs each worker *completes* in each evaluation.

        Models the two mechanisms that interact: each env cycles through the slice it was
        handed, `slice[(episode_index) % len(slice)]` (P2.3), and `evaluate_policy` counts
        `(E + i) // n_envs` completions per worker but keeps the loop alive until every worker
        has hit its target -- so a faster worker can complete `extra_episodes` more, which are
        discarded and still advance its rotation.
        """
        slices = self.partition(pairs, n_envs)
        counters = [0] * n_envs
        per_evaluation = []
        for _ in range(n_evaluations):
            ran = []
            for i, slice_ in enumerate(slices):
                target = self.sb3_targets(len(pairs), n_envs)[i]
                for _ in range(target):
                    ran.append(slice_[counters[i] % len(slice_)])
                    counters[i] += 1
                for _ in range(extra_episodes):          # discarded, but the rotation moves
                    counters[i] += 1
            per_evaluation.append(ran)
        return per_evaluation

    def test_each_evaluation_runs_every_pair_exactly_once(self):
        for n_envs in (1, 2, 3):
            with self.subTest(n_envs=n_envs):
                for ran in self._simulate(self.pairs, n_envs, n_evaluations=3):
                    self.assertEqual(sorted(ran), sorted(self.pairs))

    def test_coverage_survives_a_discarded_episode_between_evaluations(self):
        """Ritun's point 2: an extra discarded episode shifts the rotation; coverage holds.

        `L` consecutive positions of a cycle of length `L` cover every element exactly once
        whatever the offset -- the order rotates between evaluations, the set does not. This is
        why each worker gets **its own** slice rather than the whole list.
        """
        for extra in (1, 2, 5):
            for n_envs in (1, 2, 3):
                with self.subTest(extra=extra, n_envs=n_envs):
                    runs = self._simulate(self.pairs, n_envs, n_evaluations=3,
                                          extra_episodes=extra)
                    for i, ran in enumerate(runs):
                        self.assertEqual(sorted(ran), sorted(self.pairs),
                                         f"evaluation {i + 1} with {extra} discarded")

    def test_the_order_really_does_rotate_so_the_test_is_not_vacuous(self):
        runs = self._simulate(self.pairs, 2, n_evaluations=2, extra_episodes=1)
        self.assertNotEqual(runs[0], runs[1], "the offset did not shift; nothing was tested")
        self.assertEqual(sorted(runs[0]), sorted(runs[1]))

    def test_handing_every_worker_the_whole_list_would_miss_pairs(self):
        """The counter-example the partitioning exists to avoid."""
        n_envs = 2
        targets = self.sb3_targets(len(self.pairs), n_envs)
        ran = []
        for i in range(n_envs):
            for j in range(targets[i]):
                ran.append(self.pairs[j % len(self.pairs)])      # the whole list per worker
        self.assertNotEqual(sorted(ran), sorted(self.pairs))
        self.assertLess(len(set(ran)), len(self.pairs))


class _Counts:
    """Stands in for a `RunningMeanStd` -- the pairing check reads `.count` and nothing else."""

    def __init__(self, count):
        self.count = float(count)


class _Stats:
    """Stands in for a saved `VecNormalize`. Module-level so it survives a pickle round trip."""

    def __init__(self, ret_count, obs_count=None):
        self.ret_rms = _Counts(ret_count)
        self.obs_rms = _Counts(ret_count if obs_count is None else obs_count)


def write_checkpoint(path, num_timesteps, n_envs=4, data=True):
    """A checkpoint zip carrying only what the pairing check reads.

    `checkpoint_timesteps` opens the `data` entry and reads `num_timesteps`, so a synthetic zip
    is a faithful fixture and keeps these tests in milliseconds -- a real `model.save()` would
    need a policy, a venv and torch.
    """
    with zipfile.ZipFile(path, "w") as archive:
        if data:
            archive.writestr("data", json.dumps({"num_timesteps": int(num_timesteps),
                                                 "n_envs": int(n_envs)}))
        else:
            archive.writestr("policy.pth", b"not a data entry")
    return path


def write_stats(path, ret_count, obs_count=None):
    with open(path, "wb") as handle:
        pickle.dump(_Stats(ret_count, obs_count), handle)
    return path


EPS = 1e-4          # RunningMeanStd's initial count, i.e. train_sdpdp.STATS_COUNT_EPSILON


class TestPerCheckpointStatistics(unittest.TestCase):
    """One statistics file per checkpoint, and a pairing check that proves it (P2.16).

    Pure functions over synthetic zips and pickles -- no venv, no torch, milliseconds. The
    end-to-end versions are `--verify-run` and `phase2_run`.
    """

    # ---- the path map ------------------------------------------------ #

    def test_the_zip_extension_does_not_change_the_statistics_file(self):
        """`MaskablePPO.load` accepts both forms, so both must resolve to the same file."""
        with_zip = train_sdpdp.vecnormalize_path_for("/tmp/run/model_ep00047.zip")
        without = train_sdpdp.vecnormalize_path_for("/tmp/run/model_ep00047")
        self.assertEqual(with_zip, without)
        self.assertEqual(os.path.basename(with_zip), "model_ep00047_vecnormalize.pkl")

    def test_only_a_zip_suffix_is_stripped(self):
        """A dot that is not an extension stays: `model.v2` is not `model`."""
        self.assertEqual(
            os.path.basename(train_sdpdp.vecnormalize_path_for("/tmp/run/model.v2")),
            "model.v2_vecnormalize.pkl")
        self.assertEqual(train_sdpdp.vecnormalize_path_for("/tmp/run/model.v2.zip"),
                         train_sdpdp.vecnormalize_path_for("/tmp/run/model.v2"))

    def test_the_file_sits_beside_its_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = train_sdpdp.vecnormalize_path_for(os.path.join(tmp, "model_ep00001.zip"))
            self.assertEqual(os.path.dirname(path), tmp)
            self.assertTrue(os.path.isabs(path))

    def test_two_checkpoints_never_map_to_one_file(self):
        self.assertNotEqual(train_sdpdp.vecnormalize_path_for("/tmp/r/model_ep00012.zip"),
                            train_sdpdp.vecnormalize_path_for("/tmp/r/model_ep00047.zip"))

    # ---- the pairing check ------------------------------------------- #

    def test_a_checkpoints_own_statistics_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = write_checkpoint(os.path.join(tmp, "model_ep00012.zip"), 6236)
            self.assertEqual(
                train_sdpdp.check_statistics_pairing(_Stats(EPS + 6236), ckpt), 6236)

    def test_another_checkpoints_statistics_are_refused(self):
        """The defect itself: ep47's statistics against ep12's policy."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = write_checkpoint(os.path.join(tmp, "model_ep00012.zip"), 6236)
            with self.assertRaises(ValueError) as cm:
                train_sdpdp.check_statistics_pairing(_Stats(EPS + 22528), ckpt)
            message = str(cm.exception)
            self.assertIn("22528", message)          # what the statistics imply
            self.assertIn("6236", message)           # what the checkpoint says
            self.assertIn("model_ep00012.zip", message)

    def test_never_updated_statistics_get_their_own_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = write_checkpoint(os.path.join(tmp, "model_ep00012.zip"), 6236)
            with self.assertRaises(ValueError) as cm:
                train_sdpdp.check_statistics_pairing(_Stats(EPS), ckpt)
            self.assertIn("never updated", str(cm.exception))

    def test_fresh_statistics_pass_against_an_untrained_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = write_checkpoint(os.path.join(tmp, "model_ep00000.zip"), 0)
            self.assertEqual(train_sdpdp.check_statistics_pairing(_Stats(EPS), ckpt), 0)

    def test_the_pairing_holds_across_a_resume(self):
        """`reset_num_timesteps=False`, so both sides keep counting from T1."""
        t1, t2 = 22528, 10240
        with tempfile.TemporaryDirectory() as tmp:
            resumed = write_checkpoint(os.path.join(tmp, "model_ep00090.zip"), t1 + t2)
            original = write_checkpoint(os.path.join(tmp, "model_ep00047.zip"), t1)
            self.assertEqual(
                train_sdpdp.check_statistics_pairing(_Stats(EPS + t1 + t2), resumed), t1 + t2)
            with self.assertRaises(ValueError):
                train_sdpdp.check_statistics_pairing(_Stats(EPS + t1 + t2), original)

    def test_the_tolerance_is_numerical_not_semantic(self):
        """1e-6 is float safety. One timestep of slack is a real mismatch and must fail."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = write_checkpoint(os.path.join(tmp, "model_ep00012.zip"), 6236)
            train_sdpdp.check_statistics_pairing(_Stats(EPS + 6236 + 1e-7), ckpt)   # passes
            with self.assertRaises(ValueError):
                train_sdpdp.check_statistics_pairing(_Stats(EPS + 6237), ckpt)
            with self.assertRaises(ValueError):
                train_sdpdp.check_statistics_pairing(_Stats(EPS + 6236 + 1e-3), ckpt)

    def test_obs_rms_is_reported_but_never_tested(self):
        """It carries an extra n_envs per learn() call, which the checkpoint cannot tell us."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = write_checkpoint(os.path.join(tmp, "model_ep00047.zip"), 22528)
            # obs_rms 4 higher than ret_rms, exactly as a real run's one reset leaves it
            self.assertEqual(
                train_sdpdp.check_statistics_pairing(_Stats(EPS + 22528, EPS + 22532), ckpt),
                22528)
            with self.assertRaises(ValueError) as cm:
                train_sdpdp.check_statistics_pairing(_Stats(EPS + 1024, EPS + 22532), ckpt)
            self.assertIn("22532", str(cm.exception))      # reported as a diagnostic

    def test_without_a_zip_there_is_nothing_to_pair_against(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(train_sdpdp.checkpoint_timesteps(
                os.path.join(tmp, "model.zip")))
            self.assertIsNone(train_sdpdp.check_statistics_pairing(
                _Stats(EPS + 99), os.path.join(tmp, "model.zip")))

    def test_a_zip_without_a_data_entry_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = write_checkpoint(os.path.join(tmp, "model.zip"), 0, data=False)
            with self.assertRaises(ValueError) as cm:
                train_sdpdp.checkpoint_timesteps(ckpt)
            self.assertIn("Stable-Baselines3", str(cm.exception))

    def test_checkpoint_timesteps_accepts_the_path_without_the_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_checkpoint(os.path.join(tmp, "model_ep00012.zip"), 6236)
            self.assertEqual(
                train_sdpdp.checkpoint_timesteps(os.path.join(tmp, "model_ep00012")), 6236)

    # ---- the file-level form, and old run folders --------------------- #

    def test_verify_statistics_pairing_reads_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = write_checkpoint(os.path.join(tmp, "model_ep00012.zip"), 6236)
            write_stats(train_sdpdp.vecnormalize_path_for(ckpt), EPS + 6236)
            self.assertEqual(train_sdpdp.verify_statistics_pairing(ckpt),
                             train_sdpdp.vecnormalize_path_for(ckpt))

    def test_a_legacy_folder_raises_and_names_the_owning_checkpoint(self):
        """No fallback to the shared file -- that is the mismatch P2.16 removes."""
        with tempfile.TemporaryDirectory() as tmp:
            write_checkpoint(os.path.join(tmp, "model_ep00012.zip"), 6236)
            write_checkpoint(os.path.join(tmp, "model_ep00047.zip"), 22528)
            write_stats(os.path.join(tmp, train_sdpdp.VECNORMALIZE_FILE), EPS + 22528)
            with open(os.path.join(tmp, "run_metadata.json"), "w") as fh:
                json.dump({"checkpoints": ["model_ep00012.zip", "model_ep00047.zip"]}, fh)

            with self.assertRaises(FileNotFoundError) as cm:
                train_sdpdp.verify_statistics_pairing(os.path.join(tmp, "model_ep00012.zip"))
            message = str(cm.exception)
            self.assertIn(train_sdpdp.VECNORMALIZE_FILE, message)
            self.assertIn("model_ep00047.zip", message)     # which checkpoint it belongs to
            self.assertIn("--stats", message)
            self.assertIn("no longer exist", message)

    def test_the_shared_file_is_accepted_only_for_the_checkpoint_it_belongs_to(self):
        """The `--stats` escape hatch cannot re-enter the mispairing (P2.16, check (c))."""
        with tempfile.TemporaryDirectory() as tmp:
            old = write_checkpoint(os.path.join(tmp, "model_ep00012.zip"), 6236)
            newest = write_checkpoint(os.path.join(tmp, "model_ep00047.zip"), 22528)
            shared = write_stats(os.path.join(tmp, train_sdpdp.VECNORMALIZE_FILE), EPS + 22528)

            self.assertEqual(train_sdpdp.verify_statistics_pairing(newest, shared), shared)
            with self.assertRaises(ValueError) as cm:
                train_sdpdp.verify_statistics_pairing(old, shared)
            self.assertIn("6236", str(cm.exception))

    # ---- the check the verifications run ------------------------------ #

    def test_check_checkpoint_statistics_passes_when_every_checkpoint_is_paired(self):
        with tempfile.TemporaryDirectory() as tmp:
            for episodes, steps in ((12, 6236), (47, 22528)):
                ckpt = write_checkpoint(os.path.join(tmp, f"model_ep{episodes:05d}.zip"), steps)
                write_stats(train_sdpdp.vecnormalize_path_for(ckpt), EPS + steps)
            ok, detail = train_sdpdp.check_checkpoint_statistics(tmp)
            self.assertTrue(ok, detail)
            self.assertIn("2 checkpoint", detail)

    def test_check_checkpoint_statistics_fails_on_the_pre_p216_layout(self):
        """Two checkpoints sharing one file: what P2.15's folders actually hold."""
        with tempfile.TemporaryDirectory() as tmp:
            write_checkpoint(os.path.join(tmp, "model_ep00012.zip"), 6236)
            write_checkpoint(os.path.join(tmp, "model_ep00047.zip"), 22528)
            write_stats(os.path.join(tmp, train_sdpdp.VECNORMALIZE_FILE), EPS + 22528)
            ok, detail = train_sdpdp.check_checkpoint_statistics(tmp)
            self.assertFalse(ok)
            self.assertIn("model_ep00012.zip", detail)
            self.assertIn("model_ep00047.zip", detail)

    def test_check_checkpoint_statistics_fails_on_a_mispaired_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = write_checkpoint(os.path.join(tmp, "model_ep00012.zip"), 6236)
            second = write_checkpoint(os.path.join(tmp, "model_ep00047.zip"), 22528)
            write_stats(train_sdpdp.vecnormalize_path_for(first), EPS + 22528)   # wrong
            write_stats(train_sdpdp.vecnormalize_path_for(second), EPS + 22528)
            ok, detail = train_sdpdp.check_checkpoint_statistics(tmp)
            self.assertFalse(ok)
            self.assertIn("model_ep00012.zip", detail)
            self.assertNotIn("model_ep00047.zip", detail)

    def test_check_checkpoint_statistics_fails_on_an_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, detail = train_sdpdp.check_checkpoint_statistics(tmp)
            self.assertFalse(ok)
            self.assertIn("no checkpoints", detail)


class _MinimalVecEnv(VecEnv):
    """Enough `VecEnv` for `EvalCallback.__init__` to accept it without wrapping.

    Nothing below steps it: `_log_success_callback` is fed `evaluate_policy`'s locals directly,
    which is the whole point -- the end-to-end version of this is P2.15's run.
    """

    def __init__(self, num_envs=2):
        import gymnasium as gym
        super().__init__(num_envs,
                         gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,),
                                        dtype=np.float32),
                         gym.spaces.Discrete(K + 1))

    def reset(self):                                   # pragma: no cover - never called
        raise NotImplementedError

    def step_async(self, actions):                     # pragma: no cover
        raise NotImplementedError

    def step_wait(self):                               # pragma: no cover
        raise NotImplementedError

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        # `VecEnv.__init__` itself asks for "render_mode" before anything else can
        return [None] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):    # pragma: no cover
        raise NotImplementedError

    def env_method(self, method_name, *args, indices=None, **kwargs):   # pragma: no cover
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):             # pragma: no cover
        raise NotImplementedError


class TestEvalCallbackRecordsEpisodeSummaries(unittest.TestCase):
    """`PairLoggingMaskableEvalCallback` keeps each counted evaluation episode's summary.

    P2.15's pass criterion 2 is "`unclassified = 0` in **every** episode", and
    `EpisodeLogCallback` rides on the *training* env, so without this the evaluation episodes
    are the one part of a training run with no `unclassified` reading at all. The summary is
    already in `info` at this hook, so recording it costs nothing.
    """

    def make(self):
        from src.rl_gym.callbacks import PairLoggingMaskableEvalCallback
        return PairLoggingMaskableEvalCallback(_MinimalVecEnv(), n_eval_episodes=4,
                                               eval_freq=100, verbose=0)

    @staticmethod
    def locals_for(done=True, **overrides):
        return {"done": done, "info": {"episode_summary": summary(**overrides)}}

    def test_the_list_starts_empty(self):
        callback = self.make()
        self.assertEqual(callback.episode_summaries, [])

    def test_a_finished_episode_is_recorded_with_its_counts(self):
        callback = self.make()
        callback._log_success_callback(self.locals_for(unclassified=0, day="2024-03-19"), {})
        self.assertEqual(len(callback.episode_summaries), 1)
        self.assertEqual(callback.episode_summaries[0]["day"], "2024-03-19")
        self.assertEqual(callback.episode_summaries[0]["unclassified"], 0)

    def test_a_non_terminal_step_records_nothing(self):
        callback = self.make()
        callback._log_success_callback(self.locals_for(done=False), {})
        self.assertEqual(callback.episode_summaries, [])

    def test_a_non_zero_unclassified_is_visible_rather_than_swallowed(self):
        """What criterion 2 actually needs: the bad value must survive to the driver."""
        callback = self.make()
        callback._log_success_callback(self.locals_for(unclassified=3), {})
        self.assertEqual([s["unclassified"] for s in callback.episode_summaries], [3])

    def test_summaries_accumulate_flat_across_evaluations_and_match_the_pairs(self):
        callback = self.make()
        for day, seed in (("2024-01-11", 11), ("2024-03-19", 12)):
            callback._log_success_callback(self.locals_for(day=day, random_seed=seed), {})
        callback.evaluation_pairs.append(list(callback._current))
        callback._current = []
        callback._log_success_callback(self.locals_for(day="2024-06-20", random_seed=13), {})
        self.assertEqual([(s["day"], s["random_seed"]) for s in callback.episode_summaries],
                         [("2024-01-11", 11), ("2024-03-19", 12), ("2024-06-20", 13)])
        # one summary per recorded pair, so a missing summary cannot hide behind the pair list
        self.assertEqual(len(callback.episode_summaries),
                         sum(len(e) for e in callback.evaluation_pairs) + len(callback._current))

    def test_a_summary_is_copied_not_aliased(self):
        """The driver reads these after the run; an alias would let the env mutate them."""
        callback = self.make()
        locals_ = self.locals_for(unclassified=0)
        callback._log_success_callback(locals_, {})
        locals_["info"]["episode_summary"]["unclassified"] = 99
        self.assertEqual(callback.episode_summaries[0]["unclassified"], 0)


class _StubLogger:
    """Records what a callback asked to log, without a TensorBoard writer."""

    def __init__(self):
        self.values = {}
        self.dumps = []

    def record(self, key, value, exclude=None):
        self.values[key] = value

    def record_mean(self, key, value, exclude=None):
        self.values[key] = value

    def dump(self, step=0):
        self.dumps.append((step, dict(self.values)))


class _StubVenv:
    def __init__(self, num_envs=2):
        self.num_envs = num_envs


class _StubModel:
    """Just the surface a callback touches. `logger` lives here because
    `BaseCallback.logger` is a read-only property that reads `self.model.logger`."""

    def __init__(self, venv=None, logger=None, env=None):
        self._venv = venv
        self._env = env
        self.num_timesteps = 0
        self.saved = []
        self.logger = logger or _StubLogger()

    def get_env(self):
        # `BaseCallback.training_env` is a read-only property reading this
        return self._env

    def get_vec_normalize_env(self):
        return self._venv

    def save(self, path):
        self.saved.append(path)
        with open(path, "w") as fh:
            fh.write("stub")


def summary(**overrides):
    s = {"episode_reward": 1.0, "pickups": 3, "rider_declined": 1, "diffusion_cancelled": 0,
         "no_show": 0, "operator_rejected": 0, "no_candidates": 0, "unclassified": 0,
         "day": "2024-01-11", "random_seed": 7, "env_id": 0, "pass": 0,
         "horizon_margin": 12.5, "tail_seconds": 0.0}
    s.update(overrides)
    return s


class TestEpisodeLogCallback(unittest.TestCase):
    """What lands in TensorBoard per episode (§8 P2.13), against a stub logger."""

    def make(self, days=("2024-01-11", "2024-01-30"), num_envs=2):
        from src.rl_gym.callbacks import EpisodeLogCallback
        callback = EpisodeLogCallback(k_max=K, training_days=list(days))
        callback.model = _StubModel(env=_StubVenv(num_envs))
        callback._on_training_start()
        return callback

    def step(self, callback, actions, infos, dones):
        callback.locals = {"actions": np.array(actions), "infos": infos,
                           "dones": np.array(dones)}
        callback._on_step()

    def test_an_episode_logs_the_counts_the_day_and_the_seed(self):
        callback = self.make()
        self.step(callback, [0, 1], [{"n_candidates": 4}, {"n_candidates": 5}], [False, False])
        self.step(callback, [0, 2],
                  [{"episode_summary": summary(), "episode_steps": 436}, {"n_candidates": 3}],
                  [True, False])
        values = callback.model.logger.dumps[-1][1]
        self.assertEqual(values["episode/reward"], 1.0)
        self.assertEqual(values["episode/pickups"], 3)
        self.assertEqual(values["episode/unclassified"], 0)
        self.assertEqual(values["episode/day"], "2024-01-11")
        self.assertEqual(values["episode/day_index"], 0)
        self.assertEqual(values["episode/random_seed"], 7)
        self.assertEqual(values["episode/steps"], 436)
        self.assertEqual(values["episode/tail_seconds"], 0.0)
        self.assertEqual(values["episode/horizon_margin"], 12.5)
        self.assertEqual(callback.episodes, 1)

    def test_one_dump_per_episode(self):
        callback = self.make()
        for _ in range(3):
            self.step(callback, [0, 0],
                      [{"episode_summary": summary(), "episode_steps": 10},
                       {"n_candidates": 2}], [True, False])
        self.assertEqual(len(callback.model.logger.dumps), 3)

    def test_slot_selection_frequencies_cover_every_slot_and_sum_to_one(self):
        callback = self.make()
        self.step(callback, [0, K], [{}, {}], [False, False])
        self.step(callback, [1, K],
                  [{"episode_summary": summary(), "episode_steps": 2}, {}], [True, False])
        values = callback.model.logger.dumps[-1][1]
        keys = [f"slots/slot_{i}" for i in range(K)] + ["slots/reject"]
        for key in keys:
            self.assertIn(key, values)
        self.assertAlmostEqual(sum(values[k] for k in keys), 1.0, places=6)
        # two rejects out of four actions
        self.assertAlmostEqual(values["slots/reject"], 0.5, places=6)

    def test_candidate_lengths_are_logged_as_an_array_and_reset_per_episode(self):
        callback = self.make()
        self.step(callback, [0, 0], [{"n_candidates": 4}, {}], [False, False])
        self.step(callback, [0, 0], [{"n_candidates": 11}, {}], [False, False])
        self.step(callback, [0, 0],
                  [{"episode_summary": summary(), "episode_steps": 3, "n_candidates": 6}, {}],
                  [True, False])
        values = callback.model.logger.dumps[-1][1]
        lengths = values["episode/candidate_list_lengths"]
        self.assertIsInstance(lengths, np.ndarray)
        self.assertEqual(sorted(lengths.tolist()), [4.0, 6.0, 11.0])
        self.assertEqual(values["episode/candidate_list_max"], 11)
        self.assertEqual(values["episode/candidate_lists_over_k"], 1)   # 11 > K = 8
        # the next episode starts from an empty list
        self.step(callback, [0, 0],
                  [{"episode_summary": summary(), "episode_steps": 1, "n_candidates": 2}, {}],
                  [True, False])
        self.assertEqual(
            callback.model.logger.dumps[-1][1]["episode/candidate_list_lengths"].tolist(),
            [2.0])

    def test_the_day_visit_histogram_accumulates(self):
        callback = self.make()
        for day in ("2024-01-11", "2024-01-30", "2024-01-11"):
            self.step(callback, [0, 0],
                      [{"episode_summary": summary(day=day), "episode_steps": 1}, {}],
                      [True, False])
        self.assertEqual(callback.day_visit_counts(),
                         {"2024-01-11": 2, "2024-01-30": 1})
        visits = callback.model.logger.dumps[-1][1]["train/day_visits"]
        self.assertIsInstance(visits, np.ndarray)
        self.assertEqual(visits.tolist(), [0.0, 1.0, 0.0])

    def test_steps_per_second_is_positive(self):
        callback = self.make()
        self.step(callback, [0, 0],
                  [{"episode_summary": summary(), "episode_steps": 100}, {}], [True, False])
        self.assertGreater(
            callback.model.logger.dumps[-1][1]["episode/steps_per_second"], 0.0)


class TestEpisodeCheckpointCallback(unittest.TestCase):
    """Checkpoints counted in episodes, with the statistics and the metadata (§8 P2.13)."""

    def make(self, tmp, every=2, venv=None, metadata=None):
        from src.rl_gym.callbacks import EpisodeCheckpointCallback
        saved = []

        def save_stats(venv_, path):
            # P2.16: named after the checkpoint, through the one definition of that map. This
            # stub used to hardcode the shared `vecnormalize.pkl`, so the fixture reproduced
            # the overwrite defect in miniature and no test here could see it.
            target = train_sdpdp.vecnormalize_path_for(path)
            with open(target, "w") as fh:
                fh.write(f"stats for {os.path.basename(path)}")
            saved.append((venv_, target))
            return target

        callback = EpisodeCheckpointCallback(
            every_n_episodes=every, save_path=tmp, save_stats_fn=save_stats,
            metadata=metadata or {"base_seeds": [11]})
        callback.model = _StubModel(venv=venv if venv is not None else _StubVenv())
        callback.stats_saved = saved
        return callback

    def step(self, callback, dones):
        callback.locals = {"dones": np.array(dones)}
        callback._on_step()

    def test_it_saves_every_n_episodes_not_every_n_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback = self.make(tmp, every=2)
            for _ in range(3):
                self.step(callback, [False, False])
            self.assertEqual(callback.model.saved, [])
            self.step(callback, [True, False])            # 1 episode
            self.assertEqual(callback.model.saved, [])
            self.step(callback, [False, True])            # 2 episodes -> save
            self.assertEqual(len(callback.model.saved), 1)
            self.assertTrue(callback.model.saved[0].endswith("model_ep00002.zip"))

    def test_the_statistics_are_saved_beside_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback = self.make(tmp, every=1)
            self.step(callback, [True])
            self.assertTrue(os.path.isfile(
                os.path.join(tmp, "model_ep00001_vecnormalize.pkl")))
            self.assertEqual(len(callback.stats_saved), 1)

    def test_each_checkpoint_gets_its_own_statistics_file(self):
        """P2.16, and the test that fails on the pre-P2.16 shared name.

        Two checkpoints must leave two files, and the first must be **byte-unchanged** by the
        second save -- an overwrite destroys the earlier statistics rather than shadowing them,
        so only the newest checkpoint stays faithfully evaluable.
        """
        with tempfile.TemporaryDirectory() as tmp:
            callback = self.make(tmp, every=1)
            self.step(callback, [True])
            first = os.path.join(tmp, "model_ep00001_vecnormalize.pkl")
            before = open(first, "rb").read()
            self.step(callback, [True])
            second = os.path.join(tmp, "model_ep00002_vecnormalize.pkl")

            self.assertTrue(os.path.isfile(first), "the first checkpoint lost its statistics")
            self.assertTrue(os.path.isfile(second))
            self.assertNotEqual(first, second)
            self.assertEqual(open(first, "rb").read(), before,
                             "the second save overwrote the first checkpoint's statistics")
            self.assertFalse(os.path.isfile(os.path.join(tmp, train_sdpdp.VECNORMALIZE_FILE)),
                             "nothing may still write the shared pre-P2.16 name")

    def test_the_metadata_records_the_seeds_and_the_checkpoints(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            callback = self.make(tmp, every=1,
                                 metadata={"base_seeds": [11, 22], "day_dir": "corpus"})
            self.step(callback, [True])
            self.step(callback, [True])
            with open(os.path.join(tmp, "run_metadata.json")) as fh:
                meta = json.load(fh)
            self.assertEqual(meta["base_seeds"], [11, 22])
            self.assertEqual(meta["day_dir"], "corpus")
            self.assertEqual(meta["episodes"], 2)
            self.assertEqual(len(meta["checkpoints"]), 2)

    def test_an_unwrapped_training_env_raises_rather_than_saving_without_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback = self.make(tmp, every=1, venv=None)
            callback.model = _StubModel(venv=None)
            with self.assertRaises(RuntimeError) as cm:
                self.step(callback, [True])
            self.assertIn("VecNormalize", str(cm.exception))

    def test_a_non_positive_interval_raises(self):
        from src.rl_gym.callbacks import EpisodeCheckpointCallback
        with self.assertRaises(ValueError):
            EpisodeCheckpointCallback(0, "x", save_stats_fn=lambda v, p: p)


class TestResumeGuard(unittest.TestCase):
    """Resuming with a used `base_seed` would replay trained-on sample paths (§8 P2.13)."""

    def test_a_used_base_seed_raises_and_names_it(self):
        from src.rl_gym.callbacks import check_resume_base_seed
        with self.assertRaises(ValueError) as cm:
            check_resume_base_seed({"base_seeds": [11, 22]}, 22)
        self.assertIn("22", str(cm.exception))
        self.assertIn("pass 0", str(cm.exception))

    def test_a_fresh_base_seed_is_accepted(self):
        from src.rl_gym.callbacks import check_resume_base_seed
        check_resume_base_seed({"base_seeds": [11, 22]}, 33)          # must not raise
        check_resume_base_seed({}, 33)                                 # no metadata yet

    def test_metadata_is_read_from_beside_a_checkpoint(self):
        import json
        from src.rl_gym.callbacks import load_run_metadata
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "run_metadata.json"), "w") as fh:
                json.dump({"base_seeds": [5]}, fh)
            self.assertEqual(load_run_metadata(os.path.join(tmp, "model.zip")),
                             {"base_seeds": [5]})
            self.assertEqual(load_run_metadata(tmp), {"base_seeds": [5]})
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(load_run_metadata(empty), {})


class TestTrainingConfigGuards(unittest.TestCase):
    """`base_seed` and all eight weights explicit, or raise (D7, §6.1)."""

    def test_every_missing_weight_raises_and_names_itself(self):
        for name in train_sdpdp.REQUIRED_REWARD_WEIGHTS:
            with self.subTest(missing=name):
                weights = {k: v for k, v in train_sdpdp.PHASE2_REWARD_WEIGHTS.items()
                           if k != name}
                with self.assertRaises(ValueError) as cm:
                    train_sdpdp.check_config(cfg(reward_weights=weights))
                self.assertIn(name, str(cm.exception))

    def test_absent_weights_raise(self):
        with self.assertRaises(ValueError) as cm:
            train_sdpdp.check_config(cfg())
        self.assertIn("reward_weights is required", str(cm.exception))

    def test_an_unknown_weight_raises(self):
        with self.assertRaises(ValueError) as cm:
            train_sdpdp.check_config(cfg(reward_weights={
                **train_sdpdp.PHASE2_REWARD_WEIGHTS, "w_detour": 1.0}))
        self.assertIn("w_detour", str(cm.exception))

    def test_a_missing_base_seed_raises(self):
        c = cfg(reward_weights=dict(train_sdpdp.PHASE2_REWARD_WEIGHTS))
        del c["base_seed"]
        with self.assertRaises(ValueError) as cm:
            train_sdpdp.check_config(c)
        self.assertIn("base_seed", str(cm.exception))

    def test_the_full_phase_two_config_passes(self):
        train_sdpdp.check_config(
            cfg(reward_weights=dict(train_sdpdp.PHASE2_REWARD_WEIGHTS)))

    def test_the_eight_names_are_sect_6_1s(self):
        from src.rl_gym.reward import DEFAULT_REWARD_WEIGHTS
        self.assertEqual(set(train_sdpdp.REQUIRED_REWARD_WEIGHTS),
                         set(DEFAULT_REWARD_WEIGHTS))


class TestParentSeeding(unittest.TestCase):
    """The parent seeds itself; `seed=` is never handed to MaskablePPO (§8 P2.13)."""

    def test_it_is_deterministic_and_base_seed_dependent(self):
        self.assertEqual(train_sdpdp.parent_seed(7), train_sdpdp.parent_seed(7))
        self.assertNotEqual(train_sdpdp.parent_seed(7), train_sdpdp.parent_seed(8))

    def test_it_is_not_the_seed_vecenv_would_have_handed_a_worker(self):
        # VecEnv.seed(s) gives worker i the seed s + i, which is what bypassing D7 looks like
        for base in (7, 4242, 20261011):
            for env_id in range(4):
                self.assertNotEqual(train_sdpdp.parent_seed(base), base + env_id)

    def test_its_stream_cannot_coincide_with_d7s_or_d10s(self):
        base = 4242
        parent = np.random.SeedSequence([base, train_sdpdp.PARENT_SEED_TAG]) \
            .generate_state(4).tolist()
        for env_id in range(4):
            self.assertNotEqual(
                parent, np.random.SeedSequence([base, env_id]).generate_state(4).tolist())
        for p in range(4):
            self.assertNotEqual(
                parent,
                np.random.SeedSequence([base, 1010, p + 1]).generate_state(4).tolist())


if __name__ == "__main__":
    unittest.main(verbosity=2)
