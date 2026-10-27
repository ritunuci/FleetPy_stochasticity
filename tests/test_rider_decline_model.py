"""The rider decline model in `BasicRequest_no_show` (`rider_decline_model`: none, ramp or logit).

The two helpers are static, so no request, network or simulation is built. The end-to-end checks --
the ramp reproducing the pre-refactor output byte for byte, the logit reproducing itself -- were
simulation runs, not tests.

Run just these with:
    python -m unittest tests.test_rider_decline_model
"""

import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.demand.TravelerModels import BasicRequest_no_show, RETIRED_RIDER_DECLINE_KEYS  # noqa: E402
from src.misc.globals import (  # noqa: E402
    G_RIDER_ACCEPT_SCALE,
    G_RIDER_ACCEPT_W50,
    G_RIDER_DECLINE_MODEL,
    G_RIDER_MAX_WT,
    G_WAIT_LOWER_BOUND,
)

probability = BasicRequest_no_show._rider_decline_probability
check = BasicRequest_no_show._check_rider_decline_config

RAMP = {G_RIDER_DECLINE_MODEL: "ramp", G_WAIT_LOWER_BOUND: 700, G_RIDER_MAX_WT: 1500}
LOGIT = {G_RIDER_DECLINE_MODEL: "logit", G_RIDER_ACCEPT_W50: 600, G_RIDER_ACCEPT_SCALE: 120}


class TestProbability(unittest.TestCase):

    def test_none_accepts_without_a_draw(self):
        for sc in ({}, {G_RIDER_DECLINE_MODEL: None}, {G_RIDER_DECLINE_MODEL: "none"}):
            with self.subTest(sc=sc):
                self.assertIsNone(probability(5000.0, sc))

    def test_ramp_takes_no_draw_up_to_the_lower_bound(self):
        self.assertIsNone(probability(700, RAMP))
        self.assertIsNone(probability(10.0, RAMP))

    def test_ramp_rises_linearly_to_one_at_the_maximum(self):
        self.assertAlmostEqual(probability(1100, RAMP), 0.5)
        self.assertAlmostEqual(probability(1500, RAMP), 1.0)
        self.assertGreater(probability(1800, RAMP), 1.0)        # beyond the maximum: a certain decline

    def test_ramp_is_the_pre_refactor_expression_exactly(self):
        for wait in (700.5, 911.37, 1499.9):
            old = 1 - ((1500 - wait) / (1500 - 700))
            self.assertEqual(probability(wait, RAMP), old)

    def test_logit_is_one_half_at_w50_and_rises_with_the_wait(self):
        self.assertAlmostEqual(probability(600, LOGIT), 0.5)
        waits = np.linspace(0, 3000, 61)
        values = [probability(w, LOGIT) for w in waits]
        self.assertTrue(all(a < b for a, b in zip(values, values[1:])))

    def test_logit_matches_one_minus_the_acceptance_logistic(self):
        for wait in (0.0, 250.65, 600.0, 911.0, 1500.0):
            old = 1 - 1 / (1 + np.exp((wait - 600) / 120))
            self.assertAlmostEqual(probability(wait, LOGIT), old, places=15)

    def test_logit_is_stable_at_extreme_waits(self):
        tight = dict(LOGIT, **{G_RIDER_ACCEPT_SCALE: 1})
        with warnings.catch_warnings():
            warnings.simplefilter("error")                     # an overflow warning would fail here
            self.assertEqual(probability(5000.0, tight), 1.0)
            self.assertGreater(probability(0.0, tight), 0.0)   # tiny, but not rounded away

    def test_the_model_name_is_case_insensitive(self):
        self.assertAlmostEqual(probability(600, dict(LOGIT, **{G_RIDER_DECLINE_MODEL: " Logit "})), 0.5)


class TestConfigCheck(unittest.TestCase):

    def test_valid_configurations_pass(self):
        for sc in ({}, {G_RIDER_DECLINE_MODEL: "none"}, RAMP, LOGIT):
            with self.subTest(sc=sc):
                check(sc)

    def test_a_retired_switch_is_refused(self):
        for key in RETIRED_RIDER_DECLINE_KEYS:
            with self.subTest(key=key):
                with self.assertRaises(IOError) as cm:
                    check(dict(LOGIT, **{key: True}))
                self.assertIn(G_RIDER_DECLINE_MODEL, str(cm.exception))

    def test_a_retired_switch_set_false_is_harmless(self):
        check(dict(LOGIT, **{RETIRED_RIDER_DECLINE_KEYS[1]: False}))

    def test_an_unknown_model_is_refused(self):
        for bad in ("probit", True, 1):
            with self.subTest(model=bad):
                with self.assertRaises(ValueError):
                    check({G_RIDER_DECLINE_MODEL: bad})

    def test_logit_needs_numbers_and_a_positive_scale(self):
        for w50, scale in ((None, 120), (600, None), (600, 0), (600, -120), ("600", 120), (600, True)):
            with self.subTest(w50=w50, scale=scale):
                with self.assertRaises(ValueError) as cm:
                    check(dict(LOGIT, **{G_RIDER_ACCEPT_W50: w50, G_RIDER_ACCEPT_SCALE: scale}))
                self.assertIn(G_RIDER_ACCEPT_SCALE, str(cm.exception))

    def test_ramp_needs_the_maximum_above_the_lower_bound(self):
        for lower, upper in ((1500, 700), (700, 700), (None, 1500)):
            with self.subTest(lower=lower, upper=upper):
                with self.assertRaises(ValueError):
                    check(dict(RAMP, **{G_WAIT_LOWER_BOUND: lower, G_RIDER_MAX_WT: upper}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
