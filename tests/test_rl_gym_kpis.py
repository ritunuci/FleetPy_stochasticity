"""KPI arithmetic and the evaluation seed file (P2.14).

Two things are tested here, both without running a simulation and **without importing torch,
`stable_baselines3` or `sb3_contrib`** — `tests/test_rl_gym_train.py` stays the only module that
does (§8a, P2.12). The KPI functions read CSVs, so a fixture directory is enough; the seed
generator is pure numpy.

The committed `docs/baseline_user_stats.csv` and `docs/baseline_op_stats.csv` double as a real
fixture: the KPI counts on them are P1.10's figures, which is what pins the column semantics
(which column carries which outcome) rather than only the arithmetic.

Run just these with:
    python -m unittest tests.test_rl_gym_kpis
"""

import csv
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rl_gym import evaluate, kpis  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASELINE_USER_STATS = os.path.join(ROOT, "docs", "baseline_user_stats.csv")
BASELINE_OP_STATS = os.path.join(ROOT, "docs", "baseline_op_stats.csv")
SEEDS_FILE = os.path.join(ROOT, "docs", "RL_GYM_EVAL_SEEDS.csv")

#: P1.10's measurements on the reference day under greedy, which the KPI counts must reproduce
REFERENCE_COUNTS = {
    "requests": 445,
    "offers_made": 436,
    "accepted_offers": 318,
    "served": 288,
    "declines": 118,
    "cancellations": 28,
    "no_shows": 2,
    "operator_rejections": 0,
    "no_candidate_requests": 9,
    "unserved_at_horizon": 0,
    "unclassified": 0,
}


def baseline_output_dir(tmp, summary=None):
    """A fixture output directory holding the two committed baselines, plus a summary JSON."""
    os.makedirs(tmp, exist_ok=True)
    shutil.copy(BASELINE_USER_STATS, os.path.join(tmp, kpis.USER_STATS_FILE))
    shutil.copy(BASELINE_OP_STATS, os.path.join(tmp, kpis.OP_STATS_FILE))
    if summary is not None:
        with open(os.path.join(tmp, kpis.EPISODE_SUMMARY_FILE), "w") as fh:
            json.dump(summary, fh)
    return tmp


class TestKpiRowOnTheCommittedBaseline(unittest.TestCase):
    """The counts and rates of a stock-greedy run, against P1.10's figures."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="kpi_baseline_")
        cls.row = kpis.kpi_row(baseline_output_dir(cls.tmp), policy="greedy",
                               day="2024-06-20", seed=42)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_counts_reproduce_p1_10(self):
        for name, expected in REFERENCE_COUNTS.items():
            with self.subTest(name):
                self.assertEqual(self.row[name], expected)

    def test_rates_are_the_counts_over_their_denominators(self):
        self.assertAlmostEqual(self.row["decline_rate"], 118 / 436, places=12)
        self.assertAlmostEqual(self.row["cancellation_rate"], 28 / 318, places=12)
        self.assertAlmostEqual(self.row["no_show_rate"], 2 / 318, places=12)

    def test_mean_wait_and_detour_ratio(self):
        """Pinned against a computation done with plain `csv`, not with this module.

        Both figures were recomputed independently over the 288 served rows of
        `docs/baseline_user_stats.csv` -- no pandas, no `kpis.py` -- so the test checks the
        column semantics rather than echoing whatever the code under test happens to produce.
        """
        self.assertAlmostEqual(self.row["mean_wait_s"], 588.7043870907, places=6)
        self.assertAlmostEqual(self.row["mean_detour_ratio"], 1.3701307813, places=9)

    def test_vkt_matches_standard_evaluation(self):
        """`Σ driven_distance / 1000`, exactly `evaluation/standard.py`'s `op_total_km`."""
        import pandas as pd
        expected = pd.read_csv(BASELINE_OP_STATS)[kpis.VKT_COLUMN].sum() / 1000.0
        self.assertAlmostEqual(self.row["fleet_vkt_km"], float(expected), places=9)
        self.assertAlmostEqual(self.row["fleet_vkt_km"], 2568.959882, places=5)

    def test_buckets_partition_the_requests(self):
        """Every request in exactly one outcome, as §6.4 requires."""
        total = (self.row["served"] + self.row["declines"] + self.row["cancellations"]
                 + self.row["no_shows"] + self.row["operator_rejections"]
                 + self.row["no_candidate_requests"] + self.row["unserved_at_horizon"]
                 + self.row["unclassified"])
        self.assertEqual(total, self.row["requests"])

    def test_no_summary_means_zero_rejections(self):
        """A stock run has no summary JSON, and stock greedy cannot reject (D9)."""
        self.assertIsNone(kpis.read_episode_summary(self.tmp))
        self.assertEqual(self.row["operator_rejections"], 0)
        self.assertEqual(self.row["no_candidate_requests"], 9)


class TestNoOfferSplit(unittest.TestCase):
    """The one figure the user-stats file cannot supply on its own."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="kpi_split_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_summary_supplies_the_split(self):
        baseline_output_dir(self.tmp,
                            summary={"operator_rejected": 4, "no_candidates": 5})
        row = kpis.kpi_row(self.tmp)
        self.assertEqual(row["operator_rejections"], 4)
        self.assertEqual(row["no_candidate_requests"], 5)
        # and no rate moves: the group's size is unchanged, only its division
        self.assertAlmostEqual(row["decline_rate"], 118 / 436, places=12)

    def test_a_summary_that_disagrees_with_the_file_raises(self):
        """The cross-check that catches a summary and CSVs from different runs."""
        baseline_output_dir(self.tmp,
                            summary={"operator_rejected": 4, "no_candidates": 99})
        with self.assertRaises(ValueError) as ctx:
            kpis.kpi_row(self.tmp)
        self.assertIn("disagree", str(ctx.exception))
        self.assertIn("9 row(s) carry offers", str(ctx.exception))


class TestClassifyUsesTheSharedLadder(unittest.TestCase):
    """§6.4's order, through `reward.classify_exit` rather than a second copy (trap 13)."""

    def write(self, rows):
        tmp = tempfile.mkdtemp(prefix="kpi_ladder_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        header = ["request_id", "rq_time", "offers", "rider_declined", "chosen_operator_id",
                  "pickup_time", "dropoff_time", "direct_route_travel_time",
                  "diffusion_cancelled", "no_show_stat"]
        with open(os.path.join(tmp, kpis.USER_STATS_FILE), "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)
        with open(os.path.join(tmp, kpis.OP_STATS_FILE), "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([kpis.VKT_COLUMN])
            writer.writerow([1000.0])
        return tmp

    def test_no_show_is_gated_on_pickup_and_ordered_after_cancellation(self):
        """A no-show-flagged rider who cancelled is a cancellation, not a no-show (§6.4)."""
        tmp = self.write([
            # served
            [1, 100, "0:t_wait:10", False, 0, 200, 900, 400, False, False],
            # flagged no-show, never picked up -> no_show
            [2, 100, "0:t_wait:10", False, 0, "", "", 400, False, True],
            # flagged no-show that was cancelled first -> diffusion_cancelled
            [3, 100, "0:t_wait:10", False, "", "", "", 400, True, True],
            # declined
            [4, 100, "0:t_wait:10", True, "", "", "", 400, False, False],
            # no offer at all
            [5, 100, "0:", False, "", "", "", 400, False, False],
        ])
        row = kpis.kpi_row(tmp)
        self.assertEqual(row["served"], 1)
        self.assertEqual(row["no_shows"], 1)
        self.assertEqual(row["cancellations"], 1)
        self.assertEqual(row["declines"], 1)
        self.assertEqual(row["no_candidate_requests"], 1)
        self.assertEqual(row["unclassified"], 0)

    def test_accepted_but_never_picked_up_is_unserved_at_horizon(self):
        tmp = self.write([
            [1, 100, "0:t_wait:10", False, 0, "", "", 400, False, False],
        ])
        row = kpis.kpi_row(tmp)
        self.assertEqual(row["unserved_at_horizon"], 1)
        self.assertEqual(row["unclassified"], 0)

    def test_string_booleans_are_read_as_booleans(self):
        """A fixture written by hand gives `"True"`, and `== True` would be all-False."""
        tmp = self.write([
            [1, 100, "0:t_wait:10", "True", "", "", "", 400, "False", "False"],
        ])
        self.assertEqual(kpis.kpi_row(tmp)["declines"], 1)


class TestAggregation(unittest.TestCase):

    def rows(self):
        return [
            {"policy": "greedy", "day": "2024-06-20", "seed": 1, "served": 100,
             "fleet_vkt_km": 10.0},
            {"policy": "greedy", "day": "2024-06-20", "seed": 2, "served": 102,
             "fleet_vkt_km": 12.0},
            {"policy": "greedy", "day": "2024-08-13", "seed": 1, "served": 200,
             "fleet_vkt_km": 20.0},
        ]

    def test_mean_se_needs_two_observations(self):
        mean, se, n = kpis.mean_se([5.0])
        self.assertEqual((mean, se, n), (5.0, None, 1))
        mean, se, n = kpis.mean_se([4.0, 6.0])
        self.assertAlmostEqual(mean, 5.0)
        self.assertAlmostEqual(se, 1.0)       # sd 1.414.. / sqrt(2)
        self.assertEqual(n, 2)
        self.assertEqual(kpis.mean_se([None, None]), (None, None, 0))

    def test_single_seed_prints_no_error_bar(self):
        self.assertEqual(kpis.format_mean_se((5.0, None, 1), "served"), "5.0")
        self.assertEqual(kpis.format_mean_se((None, None, 0), "served"), "n/a")

    def test_grouping_by_day_month_and_overall(self):
        by_day = kpis.aggregate(self.rows(), "day")
        self.assertEqual([e["group"] for e in by_day], ["2024-06-20", "2024-08-13"])
        self.assertAlmostEqual(by_day[0]["served"][0], 101.0)
        self.assertEqual(by_day[0]["n_seeds"], 2)

        by_month = kpis.aggregate(self.rows(), "month")
        self.assertEqual([e["group"] for e in by_month], ["2024-06", "2024-08"])

        overall = kpis.aggregate(self.rows(), "overall")
        self.assertEqual(len(overall), 1)
        self.assertEqual(overall[0]["n_runs"], 3)
        self.assertAlmostEqual(overall[0]["fleet_vkt_km"][0], 14.0)

    def test_month_of_is_lenient(self):
        self.assertEqual(kpis.month_of("2024-06-20"), "2024-06")
        self.assertIsNone(kpis.month_of(None))
        self.assertIsNone(kpis.month_of("x"))

    def test_unknown_grouping_raises(self):
        with self.assertRaises(ValueError):
            kpis.aggregate(self.rows(), "weekday")


class TestEvalSeedFile(unittest.TestCase):
    """The seed file is written once and never regenerated (§8 P2.14)."""

    def test_committed_file_matches_its_derivation(self):
        self.assertEqual(evaluate.check_seed_file(), [])

    def test_shape_and_bounds(self):
        rows = evaluate.read_seed_file()
        days = sorted({d for d, _, _ in rows})
        self.assertEqual(len(days), 26, "26 evaluation days (§2.9)")
        self.assertEqual(len(rows), 26 * evaluate.DEFAULT_SEEDS_PER_DAY)
        for _day, _index, seed in rows:
            self.assertTrue(0 <= seed <= evaluate.MAX_EPISODE_SEED)

    def test_seeds_within_a_day_are_distinct(self):
        """A duplicate would silently halve that day's sample."""
        by_day = {}
        for day, _index, seed in evaluate.read_seed_file():
            by_day.setdefault(day, []).append(seed)
        for day, seeds in by_day.items():
            with self.subTest(day):
                self.assertEqual(len(seeds), len(set(seeds)))

    def test_derivation_is_deterministic(self):
        days = evaluate.evaluation_days()
        self.assertEqual(evaluate.derive_eval_seeds(days, 10),
                         evaluate.derive_eval_seeds(days, 10))

    def test_draw_order_is_row_major_over_one_stream(self):
        """Pins the documented draw order: one bulk `integers` call, consumed day by day.

        Per-day `rng.integers` calls from the same root give a different sequence, and the file
        is never regenerated — so if this ever changed there would be no way to tell afterwards
        which order had produced the committed seeds.
        """
        import numpy as np
        days = evaluate.evaluation_days()
        rng = np.random.default_rng(np.random.SeedSequence(evaluate.EVAL_SEED_ROOT))
        draws = [int(v) for v in
                 rng.integers(0, evaluate.MAX_EPISODE_SEED + 1, size=len(days) * 10)]
        self.assertEqual([s for _d, _i, s in evaluate.derive_eval_seeds(days, 10)], draws)

    def test_write_refuses_to_overwrite_and_leaves_the_file_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "seeds.csv")
            sentinel = b"do not touch me\n"
            with open(path, "wb") as fh:
                fh.write(sentinel)
            with self.assertRaises(FileExistsError) as ctx:
                evaluate.write_seed_file([("2024-06-20", 0, 1)], path)
            self.assertIn("never regenerated", str(ctx.exception))
            with open(path, "rb") as fh:
                self.assertEqual(fh.read(), sentinel,
                                 "the refusal must happen before the file is opened for writing")

    def test_write_then_read_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "seeds.csv")
            rows = [("2024-06-20", 0, 7), ("2024-06-20", 1, 8)]
            evaluate.write_seed_file(rows, path)
            self.assertEqual(evaluate.read_seed_file(path), rows)

    def test_a_changed_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "seeds.csv")
            with open(path, "w", newline="") as fh:
                csv.writer(fh).writerows([["day", "seed"], ["2024-06-20", 7]])
            with self.assertRaises(ValueError):
                evaluate.read_seed_file(path)

    def test_a_missing_file_says_how_to_make_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as ctx:
                evaluate.read_seed_file(os.path.join(tmp, "absent.csv"))
            self.assertIn("--derive-seeds", str(ctx.exception))


class TestPairSelection(unittest.TestCase):

    def test_first_n_seeds_of_named_days(self):
        """P2.15 takes the first 2 seeds of five named days this way."""
        days = ["2024-01-11", "2024-03-19", "2024-06-20", "2024-08-13", "2024-10-01"]
        pairs = evaluate.pairs_from_seed_file(days=days, n_seeds=2)
        self.assertEqual(len(pairs), 10)
        self.assertEqual([d for d, _s in pairs[:2]], ["2024-01-11", "2024-01-11"])
        rows = {(d, i): s for d, i, s in evaluate.read_seed_file()}
        self.assertEqual(pairs[0][1], rows[("2024-01-11", 0)])
        self.assertEqual(pairs[1][1], rows[("2024-01-11", 1)])

    def test_days_are_kept_in_the_order_given(self):
        pairs = evaluate.pairs_from_seed_file(days=["2024-10-01", "2024-01-11"], n_seeds=1)
        self.assertEqual([d for d, _s in pairs], ["2024-10-01", "2024-01-11"])

    def test_every_pair_by_default(self):
        self.assertEqual(len(evaluate.pairs_from_seed_file()),
                         26 * evaluate.DEFAULT_SEEDS_PER_DAY)

    def test_a_training_day_is_rejected(self):
        """2024-01-18 is a training day (§2.9), so it must not be evaluable."""
        with self.assertRaises(ValueError) as ctx:
            evaluate.pairs_from_seed_file(days=["2024-01-18"])
        self.assertIn("not an evaluation day", str(ctx.exception))

    def test_asking_for_more_seeds_than_exist_raises(self):
        with self.assertRaises(ValueError):
            evaluate.pairs_from_seed_file(days=["2024-06-20"], n_seeds=99)


class TestGreedyScenarioParameters(unittest.TestCase):
    """The stock-greedy baseline's parameters, and the trap-1 guard (D9)."""

    def test_four_overrides_and_nothing_else_that_matters(self):
        from src.misc.globals import (G_DAY_DIR, G_OP_MODULE, G_RANDOM_SEED, G_RQ_FILE,
                                      G_SCENARIO_NAME, G_SIM_ENV)
        params = evaluate.build_greedy_scenario_parameters("2024-08-13", 123)
        self.assertEqual(params[G_DAY_DIR], "corpus_eval")
        self.assertEqual(params[G_RQ_FILE], "2024-08-13.csv")
        self.assertEqual(params[G_RANDOM_SEED], 123)
        self.assertEqual(params[G_SCENARIO_NAME], "greedy_2024-08-13_s123")
        # stock, not the RL subclasses -- otherwise D9 would compare the RL path with itself
        self.assertEqual(params[G_OP_MODULE], "PoolingIRSOnly")
        self.assertEqual(params[G_SIM_ENV], "ImmediateDecisionsSimulation")

    def test_the_row_seed_and_day_are_reproduced_for_the_reference_pair(self):
        from src.misc.globals import G_RANDOM_SEED
        params = evaluate.build_greedy_scenario_parameters(evaluate.REFERENCE_DAY,
                                                           evaluate.REFERENCE_SEED)
        self.assertEqual(params[G_RANDOM_SEED], 42)

    def test_output_directory_is_unique_per_pair(self):
        a = evaluate.output_dir_for("greedy_2024-06-20_s42")
        b = evaluate.output_dir_for("greedy_2024-06-20_s43")
        self.assertNotEqual(a, b)
        self.assertTrue(a.endswith(os.path.join("results", "greedy_2024-06-20_s42")))

    def test_guard_refuses_the_reference_scenario_name(self):
        """Trap 1: the row's own name is the reference run's output directory."""
        from src.misc.globals import G_SCENARIO_NAME
        params = evaluate._base_scenario_parameters()
        row_name = params[G_SCENARIO_NAME]
        with self.assertRaises(RuntimeError) as ctx:
            evaluate.guard_output_dir(params, row_name)
        self.assertIn("create_or_empty_dir", str(ctx.exception))

    def test_guard_accepts_a_renamed_run(self):
        from src.misc.globals import G_SCENARIO_NAME
        params = evaluate._base_scenario_parameters()
        row_name = params[G_SCENARIO_NAME]
        params[G_SCENARIO_NAME] = "greedy_2024-06-20_s42"
        out = evaluate.guard_output_dir(params, row_name)
        self.assertTrue(out.endswith("greedy_2024-06-20_s42"))

    def test_the_reference_directory_is_what_the_guard_protects(self):
        """The guard's target really is the directory the committed baselines came from."""
        from src.misc.globals import G_SCENARIO_NAME
        reference = evaluate.output_dir_for(
            evaluate._base_scenario_parameters()[G_SCENARIO_NAME])
        self.assertTrue(reference.endswith("example_depot_time_pool_irsonly_rs_42"))


class TestReportRendering(unittest.TestCase):

    def test_report_has_all_four_sections(self):
        rows = [{"policy": "greedy", "day": "2024-06-20", "seed": 1, "served": 1,
                 "unclassified": 0}]
        report = evaluate.format_report(rows)
        for section in ("## Per run", "## By day", "## By month", "## By overall"):
            self.assertIn(section, report)

    def test_per_run_table_renders_missing_values_as_na(self):
        rows = [{"policy": "rl", "day": "2024-06-20", "seed": 1, "mean_wait_s": None}]
        self.assertIn("n/a", kpis.format_per_run_table(rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
