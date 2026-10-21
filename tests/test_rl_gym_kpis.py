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

from src.rl_gym import corpus, evaluate, kpis, output_integrity  # noqa: E402

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


def record_rows_as_written(tmp):
    """Write the P2.20 integrity record a checked run would carry: each CSV's own row count.

    A fixture models a run that passed its end-of-run check, and `kpi_row` refuses a directory
    without a record. The refusals themselves are tested in `test_rl_gym_output_integrity.py`.
    """
    output_integrity.write_rows_record(tmp, {
        name: output_integrity.count_data_rows(os.path.join(tmp, name))
        for name in (kpis.USER_STATS_FILE, kpis.OP_STATS_FILE)})


def baseline_output_dir(tmp, summary=None):
    """A fixture output directory holding the two committed baselines, plus a summary JSON."""
    os.makedirs(tmp, exist_ok=True)
    shutil.copy(BASELINE_USER_STATS, os.path.join(tmp, kpis.USER_STATS_FILE))
    shutil.copy(BASELINE_OP_STATS, os.path.join(tmp, kpis.OP_STATS_FILE))
    record_rows_as_written(tmp)
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
        record_rows_as_written(tmp)
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
        self.assertEqual(len(days), 26, "26 held-out days: 12 validation + 14 test (§2.9)")
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
        days = evaluate.held_out_days()
        self.assertEqual(evaluate.derive_eval_seeds(days, 10),
                         evaluate.derive_eval_seeds(days, 10))

    def test_draw_order_is_row_major_over_one_stream(self):
        """Pins the documented draw order: one bulk `integers` call, consumed day by day.

        Per-day `rng.integers` calls from the same root give a different sequence, and the file
        is never regenerated — so if this ever changed there would be no way to tell afterwards
        which order had produced the committed seeds.
        """
        import numpy as np
        days = evaluate.held_out_days()
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


class TestHeldOutSplit(unittest.TestCase):
    """P2.17's validation / test split of the 26 held-out days (§2.9).

    The split is frozen like §2.9's own 26-day list: these literals are the decision, and the
    assertions bind them to the directories, to the committed `day,split` record and to the
    seed file that must not have moved.
    """

    #: the 12 validation days, drawn at seed 20261018 by §2.9's rule (P2.17)
    VALIDATION = [
        "2024-01-11", "2024-02-07", "2024-02-27", "2024-03-19", "2024-04-17", "2024-05-23",
        "2024-06-20", "2024-07-31", "2024-08-13", "2024-08-15", "2024-09-04", "2024-10-01",
    ]
    #: the 14 test days -- the final-results half, read once
    TEST = [
        "2024-01-30", "2024-02-08", "2024-03-20", "2024-04-09", "2024-04-11", "2024-05-07",
        "2024-05-15", "2024-06-12", "2024-07-09", "2024-07-25", "2024-08-28", "2024-09-12",
        "2024-10-09", "2024-10-24",
    ]
    #: md5 of the seed file's three columns, header included, LF-terminated. **This is the
    #: guarantee that P2.17 redrew nothing**: the whole-file md5
    #: `559b836e7c61fb7dadd2e7bd9bc8891d` is checked too, and holds because the split labels
    #: went into a separate file -- but this one survives any later annotation and still fails
    #: on a single changed seed.
    SEED_COLUMNS_MD5 = "36599d9aeb886dc4b1055d777565069d"
    SEEDS_FILE_MD5 = "559b836e7c61fb7dadd2e7bd9bc8891d"

    def test_the_two_halves_are_the_committed_lists(self):
        self.assertEqual(corpus.validation_days(), self.VALIDATION)
        self.assertEqual(corpus.test_days(), self.TEST)

    def test_the_halves_are_disjoint_and_cover_the_seed_file(self):
        """A day in both halves is the copy-instead-of-move failure; a day in neither is lost."""
        self.assertEqual(set(self.VALIDATION) & set(self.TEST), set())
        self.assertEqual(corpus.held_out_days(), sorted(self.VALIDATION + self.TEST))
        self.assertEqual(corpus.held_out_days(),
                         sorted({d for d, _i, _s in evaluate.read_seed_file()}))

    def test_sizes_and_weekday_balance(self):
        """4/4/4 and 4/5/5 -- the best the 8/9/9 pool allows, and the rule's stated property."""
        import datetime as dt
        names = ("Tue", "Wed", "Thu")

        def counts(days):
            out = {n: 0 for n in names}
            for day in days:
                out[names[dt.date.fromisoformat(day).weekday() - 1]] += 1
            return out

        self.assertEqual(len(self.VALIDATION), 12)
        self.assertEqual(len(self.TEST), 14)
        self.assertEqual(counts(self.VALIDATION), {"Tue": 4, "Wed": 4, "Thu": 4})
        self.assertEqual(counts(self.TEST), {"Tue": 4, "Wed": 5, "Thu": 5})

    def test_every_month_is_in_both_halves(self):
        val_months = {int(d[5:7]) for d in self.VALIDATION}
        test_months = {int(d[5:7]) for d in self.TEST}
        self.assertEqual(val_months, set(range(1, 11)))
        self.assertEqual(test_months, set(range(1, 11)))

    def test_the_five_already_run_days_are_validation_days(self):
        """P2.15 and P2.16 evaluated a trained policy on these, so they cannot be test days."""
        for day in ("2024-01-11", "2024-03-19", "2024-06-20", "2024-08-13", "2024-10-01"):
            with self.subTest(day):
                self.assertEqual(corpus.split_of_day(day), "val")

    def test_the_seed_file_was_not_regenerated(self):
        """The three seed columns, pinned. P2.17 moved no seed -- this is what says so."""
        import hashlib
        rows = evaluate.read_seed_file()
        projection = "day,seed_index,seed\n" + "".join(f"{d},{i},{s}\n" for d, i, s in rows)
        self.assertEqual(hashlib.md5(projection.encode()).hexdigest(),
                         self.SEED_COLUMNS_MD5)
        with open(SEEDS_FILE, "rb") as fh:
            self.assertEqual(hashlib.md5(fh.read()).hexdigest(), self.SEEDS_FILE_MD5,
                             "the seed file must stay byte-frozen: the split labels live in "
                             "docs/RL_GYM_HELDOUT_SPLIT.csv")

    def test_the_seed_file_still_matches_its_derivation_over_both_halves(self):
        self.assertEqual(evaluate.check_seed_file(), [])

    def test_committed_labels_agree_with_the_directories(self):
        """G4. A stale label cannot misroute a run, but it would misreport a filed KPI table."""
        self.assertEqual(corpus.check_split_file(), [])
        labelled = dict(corpus.read_split_file())
        self.assertEqual(len(labelled), 26)
        for day in self.VALIDATION:
            self.assertEqual(labelled[day], "val")
        for day in self.TEST:
            self.assertEqual(labelled[day], "test")

    def test_split_of_day_refuses_a_training_day_and_an_unknown_day(self):
        for day in ("2024-01-18", "2024-06-21", "1999-01-01"):
            with self.subTest(day), self.assertRaises(ValueError) as ctx:
                corpus.split_of_day(day)
            self.assertIn("not a held-out day", str(ctx.exception))

    def test_a_day_in_both_halves_raises_rather_than_picking_one(self):
        """The copy-instead-of-move failure: the day would be selected on and reported on."""
        with tempfile.TemporaryDirectory() as tmp:
            val, test = os.path.join(tmp, "val"), os.path.join(tmp, "test")
            os.makedirs(val)
            os.makedirs(test)
            for d in (val, test):
                with open(os.path.join(d, "2024-06-20.csv"), "w") as fh:
                    fh.write("x\n")
            with self.assertRaises(ValueError) as ctx:
                corpus.split_of_day("2024-06-20", val, test)
            self.assertIn("both", str(ctx.exception))

    def test_day_dir_for_refuses_the_wrong_half_in_both_directions(self):
        """G3, symmetric. One failing direction with the other passing is how this rots."""
        val_day, test_day = self.VALIDATION[0], self.TEST[0]
        self.assertEqual(corpus.day_dir_for(val_day, "val"), "corpus_val")
        self.assertEqual(corpus.day_dir_for(test_day, "test"), "corpus_eval")
        with self.assertRaises(ValueError) as ctx:
            corpus.day_dir_for(val_day, "test")
        self.assertIn("**val** day", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            corpus.day_dir_for(test_day, "val")
        self.assertIn("**test** day", str(ctx.exception))

    def test_an_unrecognised_split_raises(self):
        for bad in ("validation", "final", "", "VAL"):
            with self.subTest(bad), self.assertRaises(ValueError):
                corpus.day_dir_for(self.VALIDATION[0], bad)


class TestSplitGuardsOnTheRunPaths(unittest.TestCase):
    """The guards where a command actually starts a run (G1-G3, P2.17). No simulation."""

    VAL_DAY = "2024-06-20"
    TEST_DAY = "2024-01-30"

    def test_greedy_parameters_resolve_each_half_and_refuse_the_other(self):
        from src.misc.globals import G_DAY_DIR
        for day, split, expected in ((self.VAL_DAY, "val", "corpus_val"),
                                     (self.TEST_DAY, "test", "corpus_eval")):
            with self.subTest(day):
                params = evaluate.build_greedy_scenario_parameters(day, 1, split=split)
                self.assertEqual(params[G_DAY_DIR], expected)
        with self.assertRaises(ValueError):
            evaluate.build_greedy_scenario_parameters(self.VAL_DAY, 1, split="test")
        with self.assertRaises(ValueError):
            evaluate.build_greedy_scenario_parameters(self.TEST_DAY, 1, split="val")

    def test_a_training_day_is_unreachable_from_the_greedy_path(self):
        """`corpus/` and `corpus_excluded/` cannot be named: the day decides (G2)."""
        for day in ("2024-01-18", "2024-07-04"):
            with self.subTest(day), self.assertRaises(ValueError):
                evaluate.build_greedy_scenario_parameters(day, 1)

    def test_pairs_from_seed_file_selects_a_half_when_no_days_are_given(self):
        val = evaluate.pairs_from_seed_file(split="val")
        test = evaluate.pairs_from_seed_file(split="test")
        self.assertEqual(len(val), 12 * evaluate.DEFAULT_SEEDS_PER_DAY)
        self.assertEqual(len(test), 14 * evaluate.DEFAULT_SEEDS_PER_DAY)
        self.assertEqual(sorted({d for d, _s in val}), corpus.validation_days())
        self.assertEqual(sorted({d for d, _s in test}), corpus.test_days())
        # and together they are still every pair the file holds -- nothing was dropped
        self.assertEqual(len(val) + len(test), len(evaluate.pairs_from_seed_file()))

    def test_pairs_from_seed_file_checks_named_days_against_the_half(self):
        """A named day list must not be able to smuggle a test day into a validation run."""
        with self.assertRaises(ValueError):
            evaluate.pairs_from_seed_file(days=[self.TEST_DAY], split="val")
        with self.assertRaises(ValueError):
            evaluate.pairs_from_seed_file(days=[self.VAL_DAY], split="test")
        # mixed, which is the realistic typo
        with self.assertRaises(ValueError):
            evaluate.pairs_from_seed_file(days=[self.VAL_DAY, self.TEST_DAY], split="val")

    def test_resolve_pairs_split_refuses_a_pair_list_spanning_both_halves(self):
        self.assertEqual(
            evaluate.resolve_pairs_split([(self.VAL_DAY, 1), ("2024-08-13", 2)]), "corpus_val")
        self.assertEqual(
            evaluate.resolve_pairs_split([(self.TEST_DAY, 1)]), "corpus_eval")
        with self.assertRaises(ValueError) as ctx:
            evaluate.resolve_pairs_split([(self.VAL_DAY, 1), (self.TEST_DAY, 2)])
        self.assertIn("spans both halves", str(ctx.exception))

    def test_the_report_header_names_the_half(self):
        val = evaluate.report_header("val", [self.VAL_DAY])
        test = evaluate.report_header("test", [self.TEST_DAY], checkpoint="runs/x/ep1.zip")
        self.assertIn("half: **val**", val)
        self.assertIn("half: **test**", test)
        self.assertIn("FINAL RESULTS", test)
        self.assertIn("ep1.zip", test)
        self.assertIn("## Provenance", val)

    def test_the_gate_harness_runs_on_the_validation_half(self):
        """P2.15's five days are all validation days, and its driver says so (P2.17)."""
        from src.rl_gym.tools import phase2_run
        self.assertEqual(phase2_run.HARNESS_SPLIT, "val")
        for day in phase2_run.HARNESS_DAYS + phase2_run.CALLBACK_DAYS:
            with self.subTest(day):
                self.assertEqual(corpus.split_of_day(day), "val")


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
        # 2024-08-13 is a **validation** day (P2.17), and the directory is resolved from the
        # day rather than passed in -- that is G2, and it is why this assertion is on
        # corpus_val and not on whatever default the function happens to carry
        self.assertEqual(params[G_DAY_DIR], "corpus_val")
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
