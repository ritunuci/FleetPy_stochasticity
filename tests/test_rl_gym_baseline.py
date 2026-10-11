"""Tests for the arrival baseline and corpus manifest (P2.6).

`python -m unittest discover tests`. No FleetPy simulation is built, but the baseline is
recomputed from the 101 training files, so this file is the slowest of the non-simulation
suites (a few seconds).

Most assertions read the **committed artifacts**, not the builder's in-memory output, so what
gets checked is what the env will load in P2.11.
"""

import csv
import datetime as dt
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rl_gym.offline.build_arrival_baseline import (  # noqa: E402
    BASELINE,
    BASELINE_COLUMNS,
    CORPUS,
    END_TIME,
    MANIFEST,
    MONTH_VOCAB,
    SD_FLOOR,
    START_TIME,
    WEEKDAY_VOCAB,
    WINDOWS_S,
    baseline_hash,
    canonical_baseline_text,
    compute_baseline,
    corpus_file_hashes,
    corpus_hash,
    counts_for_day,
    kept_arrival_times,
    minute_grid,
    read_baseline,
    training_day_files,
    write_both,
)

N_TRAINING_DAYS = 101
N_MINUTES = 747                       # multiples of 60 in [25200, 70000)
N_ROWS = N_MINUTES * len(MONTH_VOCAB) * len(WINDOWS_S)       # 14_940

#: the five training days carrying one `start == end` row each (P2.4), with that row's rq_time
SAME_NODE_TRAINING_DAYS = {
    "2024-02-15": 37080, "2024-02-20": 57180, "2024-03-05": 33060,
    "2024-05-30": 33180, "2024-06-13": 42540,
}


class TestMinuteGrid(unittest.TestCase):

    def test_grid_is_747_minutes_inside_the_window(self):
        grid = minute_grid()
        self.assertEqual(len(grid), N_MINUTES)
        self.assertEqual(grid[0], START_TIME)
        self.assertEqual(grid[-1], 69960)
        self.assertTrue(all(t % 60 == 0 for t in grid))
        self.assertTrue(all(START_TIME <= t < END_TIME for t in grid))


class TestCommittedBaseline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rows = read_baseline(BASELINE)

    def test_is_a_plain_table_pandas_can_read_with_no_arguments(self):
        # Ritun's rule: no metadata, no comment lines -- the manifest carries the hash instead
        import pandas as pd
        df = pd.read_csv(BASELINE)
        self.assertEqual(list(df.columns), BASELINE_COLUMNS)
        self.assertEqual(len(df), N_ROWS)
        self.assertFalse(df.isna().any().any())

    def test_row_count_and_keys(self):
        self.assertEqual(len(self.rows), N_ROWS)
        keys = {(m, t, w) for m, t, w, _mean, _sd in self.rows}
        self.assertEqual(len(keys), N_ROWS, "every (month, minute, window) appears once")
        self.assertEqual({m for m, *_ in self.rows}, set(MONTH_VOCAB))
        self.assertEqual({w for _m, _t, w, *_ in self.rows}, set(WINDOWS_S))
        self.assertEqual({t for _m, t, *_ in self.rows}, set(minute_grid()))

    def test_all_values_finite_and_non_negative(self):
        for m, t, w, mean, sd in self.rows:
            self.assertTrue(np.isfinite(mean) and np.isfinite(sd), (m, t, w))
            self.assertGreaterEqual(mean, 0.0)
            self.assertGreaterEqual(sd, 0.0)

    def test_sd_is_pooled_so_it_does_not_vary_by_month(self):
        # 9-12 days per month are too few to estimate spread, so sd is pooled over all 101
        by_key = {}
        for m, t, w, _mean, sd in self.rows:
            by_key.setdefault((t, w), set()).add(sd)
        offenders = {k: v for k, v in by_key.items() if len(v) != 1}
        self.assertEqual(offenders, {}, "sd must depend only on (minute, window)")

    def test_sixty_minute_means_are_at_least_thirty_minute_means(self):
        means = {(m, t, w): mean for m, t, w, mean, _sd in self.rows}
        for (m, t, w), mean in means.items():
            if w == 1800:
                self.assertLessEqual(mean, means[(m, t, 3600)] + 1e-9, (m, t))

    def test_first_minute_is_empty_by_construction(self):
        # the trailing window at t = start_time lies entirely before the day, so the count is 0
        # on every day; sd is 0 there too, which is what the SD floor exists for
        for w in WINDOWS_S:
            for m in MONTH_VOCAB:
                row = next(r for r in self.rows if r[:3] == (m, START_TIME, w))
                self.assertEqual((row[3], row[4]), (0.0, 0.0))

    def test_rows_are_sorted(self):
        keys = [(m, t, w) for m, t, w, *_ in self.rows]
        self.assertEqual(keys, sorted(keys), "committed rows must be sorted, for a stable diff")


class TestBaselineMatchesAFreshBuild(unittest.TestCase):

    def test_hashes_agree(self):
        rows, _diag = compute_baseline()
        self.assertEqual(baseline_hash(rows), baseline_hash(read_baseline(BASELINE)))

    def test_hash_is_over_parsed_content_not_bytes(self):
        rows = read_baseline(BASELINE)
        with open(BASELINE, newline="") as fh:
            text = fh.read()
        with tempfile.TemporaryDirectory() as tmp:
            crlf = os.path.join(tmp, "crlf.csv")
            with open(crlf, "w", newline="") as fh:
                fh.write(text.replace("\n", "\r\n"))
            with open(crlf, "rb") as a, open(BASELINE, "rb") as b:
                self.assertNotEqual(a.read(), b.read(), "the two files must differ as bytes")
            self.assertEqual(baseline_hash(read_baseline(crlf)), baseline_hash(rows))

    def test_canonical_text_is_idempotent_through_a_write_and_parse(self):
        rows = read_baseline(BASELINE)
        self.assertEqual(canonical_baseline_text(rows),
                         canonical_baseline_text(read_baseline(BASELINE)))

    def test_hash_changes_if_one_value_changes(self):
        rows = read_baseline(BASELINE)
        tampered = list(rows)
        m, t, w, mean, sd = tampered[5000]
        tampered[5000] = (m, t, w, mean + 0.01, sd)
        self.assertNotEqual(baseline_hash(tampered), baseline_hash(rows))


class TestCountsDefinition(unittest.TestCase):
    """`c_d(t, W)` recomputed by hand -- P2.6's first verification."""

    def test_three_days_three_minutes_by_hand(self):
        import pandas as pd
        # genuine training days, spread across the year. 2024-05-15 looks like an obvious
        # pick and is an *evaluation* day -- checked against training_day_files(), not assumed
        days = ["2024-01-09", "2024-05-14", "2024-10-30"]
        probes = [36000, 55800, 61200]
        in_corpus = set(training_day_files())
        self.assertTrue(set(days) <= in_corpus, f"not training days: {set(days) - in_corpus}")
        grid = minute_grid()
        for day in days:
            path = os.path.join(CORPUS, f"{day}.csv")
            df = pd.read_csv(path)
            times = kept_arrival_times(path)
            for w in WINDOWS_S:
                counts = counts_for_day(times, grid, w)
                for t in probes:
                    # by hand, straight from the dataframe, applying both filters inline
                    sel = df[(df.rq_time >= START_TIME) & (df.rq_time < END_TIME)
                             & (df.start != df.end)
                             & (df.rq_time >= t - w) & (df.rq_time < t)]
                    self.assertEqual(counts[grid.index(t)], float(len(sel)),
                                     f"{day} t={t} W={w}")

    def test_window_is_half_open(self):
        times = np.array([1000.0, 2000.0])
        # [t - W, t): an arrival exactly at t is excluded, one exactly at t - W is included
        self.assertEqual(counts_for_day(times, [2000], 1000)[0], 1.0)
        self.assertEqual(counts_for_day(times, [2001], 1000)[0], 1.0)
        self.assertEqual(counts_for_day(times, [3000], 1000)[0], 1.0)


class TestSameNodeRowsAreExcluded(unittest.TestCase):
    """The second filter, which P2.4 found and P2.6's definition now names."""

    def test_the_five_days_are_still_the_five(self):
        import pandas as pd
        found = {}
        for day in training_day_files():
            df = pd.read_csv(os.path.join(CORPUS, f"{day}.csv"))
            win = df[(df.rq_time >= START_TIME) & (df.rq_time < END_TIME)]
            same = win[win.start == win.end]
            if len(same):
                found[day] = int(same.rq_time.iloc[0])
        self.assertEqual(found, SAME_NODE_TRAINING_DAYS)

    def test_kept_times_exclude_the_same_node_row(self):
        import pandas as pd
        for day, rq_time in SAME_NODE_TRAINING_DAYS.items():
            path = os.path.join(CORPUS, f"{day}.csv")
            df = pd.read_csv(path)
            win = df[(df.rq_time >= START_TIME) & (df.rq_time < END_TIME)]
            kept = kept_arrival_times(path)
            self.assertEqual(len(kept), len(win) - 1, day)
            # one fewer arrival at that minute than the raw rows have
            raw_at = int((win.rq_time == rq_time).sum())
            kept_at = int((kept == rq_time).sum())
            self.assertEqual(kept_at, raw_at - 1, day)

    def test_counting_raw_rows_would_inflate_the_mean(self):
        # the effect, measured: a February bucket moves by 1/10 if the row is counted
        import pandas as pd
        day, rq_time = "2024-02-15", 37080
        grid = minute_grid()
        feb = [d for d in training_day_files() if dt.date.fromisoformat(d).month == 2]
        self.assertEqual(len(feb), 10)
        t = rq_time + 60                      # a window that contains the row
        i = grid.index(t)

        def mean_at(use_both_filters):
            vals = []
            for d in feb:
                path = os.path.join(CORPUS, f"{d}.csv")
                df = pd.read_csv(path)
                w = df[(df.rq_time >= START_TIME) & (df.rq_time < END_TIME)]
                if use_both_filters:
                    w = w[w.start != w.end]
                vals.append(counts_for_day(np.sort(w.rq_time.to_numpy()), grid, 1800)[i])
            return float(np.mean(vals))

        correct, inflated = mean_at(True), mean_at(False)
        self.assertAlmostEqual(inflated - correct, 0.1, places=9)
        committed = {(m, tt, ww): mean for m, tt, ww, mean, _sd in read_baseline(BASELINE)}
        self.assertAlmostEqual(committed[(2, t, 1800)], correct, places=9,
                               msg="the committed baseline must use both filters")


class TestMonthOverWeekday(unittest.TestCase):
    """The figures behind "why month, not weekday" (P2.6), re-measured.

    These are the justification for conditioning the baseline on month, so they are asserted
    as a direction rather than as exact values: the ordering must hold even if the corpus is
    ever extended.
    """

    @classmethod
    def setUpClass(cls):
        days = training_day_files()
        grid = minute_grid()
        cls.days = days
        cls.X = np.vstack([counts_for_day(
            kept_arrival_times(os.path.join(CORPUS, f"{d}.csv")), grid, 1800) for d in days])
        cls.months = np.array([dt.date.fromisoformat(d).month for d in days])
        cls.wd = np.array([dt.date.fromisoformat(d).strftime("%a") for d in days])
        cls.grid = grid

    def _spread(self, keys):
        B = np.zeros_like(self.X)
        for k in set(keys):
            sel = keys == k
            B[sel] = self.X[sel].mean(axis=0)
        return float(np.mean(np.std(self.X - B, axis=0, ddof=1)))

    def test_month_baseline_beats_weekday_and_none(self):
        month, weekday = self._spread(self.months), self._spread(self.wd)
        none = float(np.mean(np.std(self.X - self.X.mean(axis=0), axis=0, ddof=1)))
        self.assertLess(month, weekday, "conditioning on month must beat weekday")
        self.assertLess(weekday, none)
        self.assertAlmostEqual(month, 4.75, delta=0.15)
        self.assertAlmostEqual(weekday, 5.20, delta=0.15)
        self.assertAlmostEqual(none, 5.45, delta=0.15)

    def test_weekday_explains_almost_none_of_the_daily_total_variance(self):
        totals = self.X.sum(axis=1)

        def share(keys):
            ss_res = sum(((totals[keys == k] - totals[keys == k].mean()) ** 2).sum()
                         for k in set(keys))
            return 1 - ss_res / ((totals - totals.mean()) ** 2).sum()

        self.assertLess(share(self.wd), 0.05, "weekday must explain essentially nothing")
        self.assertGreater(share(self.months), 0.4, "month must explain a large share")

    def test_weekday_still_matters_at_the_afternoon_peak(self):
        # which is why the weekday one-hot stays in the observation (§5.2)
        i = self.grid.index(55800)
        tue = self.X[self.wd == "Tue", i].mean()
        wed = self.X[self.wd == "Wed", i].mean()
        overall = self.X[:, i].mean()
        self.assertLess(tue, overall)
        self.assertGreater(wed, overall)
        self.assertGreater(wed - tue, 10.0, "the weekday spread at 15:30 is large")

    def test_october_reads_high_against_a_weekday_baseline(self):
        B = np.zeros_like(self.X)
        for k in set(self.wd):
            sel = self.wd == k
            B[sel] = self.X[sel].mean(axis=0)
        res = self.X - B
        sd = np.std(res, axis=0, ddof=1)
        z = res / np.where(sd > 0, sd, 1.0)
        self.assertGreater(z[self.months == 10].mean(), 0.5)


class TestManifest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(MANIFEST) as fh:
            cls.m = json.load(fh)

    def test_constants_match_the_reference_scenario(self):
        self.assertEqual(self.m["start_time"], START_TIME)
        self.assertEqual(self.m["end_time"], END_TIME)
        self.assertEqual(self.m["windows_s"], list(WINDOWS_S))
        self.assertEqual(self.m["sd_floor"], SD_FLOOR)
        self.assertEqual(self.m["month_vocab"], MONTH_VOCAB)
        self.assertEqual(self.m["weekday_vocab"], WEEKDAY_VOCAB)

    def test_file_list_is_the_training_corpus(self):
        self.assertEqual(self.m["n_training_days"], N_TRAINING_DAYS)
        self.assertEqual(len(self.m["files"]), N_TRAINING_DAYS)
        names = [e["filename"] for e in self.m["files"]]
        self.assertEqual(names, sorted(names), "files must be sorted by filename")
        self.assertEqual(names, [f"{d}.csv" for d in training_day_files()])
        self.assertNotIn("2024-06-20.csv", names, "an evaluation day must not be here")

    def test_per_file_hashes_match_the_directory(self):
        self.assertEqual(self.m["files"], corpus_file_hashes())

    def test_corpus_hash_is_over_the_name_and_hash_lines(self):
        self.assertEqual(self.m["corpus_hash"], corpus_hash(self.m["files"]))

    def test_baseline_hash_ties_the_two_artifacts_together(self):
        self.assertEqual(self.m["baseline_hash"], baseline_hash(read_baseline(BASELINE)))
        self.assertEqual(self.m["baseline_file"], os.path.basename(BASELINE))
        self.assertEqual(self.m["baseline_rows"], N_ROWS)

    def test_corpus_hash_changes_if_a_file_is_removed_or_edited(self):
        files = corpus_file_hashes()
        self.assertNotEqual(corpus_hash(files[:-1]), corpus_hash(files))
        edited = [dict(e) for e in files]
        edited[0]["sha256"] = "0" * 64
        self.assertNotEqual(corpus_hash(edited), corpus_hash(files))

    def test_corpus_hash_depends_on_order(self):
        files = corpus_file_hashes()
        self.assertNotEqual(corpus_hash(list(reversed(files))), corpus_hash(files))


class TestBothOrNeither(unittest.TestCase):
    """The builder writes the pair together; neither artifact is written alone."""

    def test_write_both_produces_a_manifest_that_hashes_the_baseline_it_wrote(self):
        rows, _ = compute_baseline()
        with tempfile.TemporaryDirectory() as tmp:
            b = os.path.join(tmp, "arrival_baseline.csv")
            mf = os.path.join(tmp, "corpus_manifest.json")
            manifest = write_both(rows, CORPUS, b, mf)
            self.assertTrue(os.path.isfile(b) and os.path.isfile(mf))
            self.assertEqual(manifest["baseline_hash"], baseline_hash(read_baseline(b)))
            with open(mf) as fh:
                self.assertEqual(json.load(fh)["baseline_hash"], manifest["baseline_hash"])

    def test_a_stale_manifest_is_detected(self):
        # the failure the baseline_hash exists to catch: baseline regenerated, manifest not
        rows, _ = compute_baseline()
        with tempfile.TemporaryDirectory() as tmp:
            b = os.path.join(tmp, "arrival_baseline.csv")
            mf = os.path.join(tmp, "corpus_manifest.json")
            write_both(rows, CORPUS, b, mf)
            tampered = list(rows)
            m, t, w, mean, sd = tampered[0]
            tampered[0] = (m, t, w, mean + 1.0, sd)
            with open(b, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(BASELINE_COLUMNS)
                writer.writerows([(a, bb, c, f"{d:.12g}", f"{e:.12g}")
                                  for a, bb, c, d, e in tampered])
            with open(mf) as fh:
                self.assertNotEqual(json.load(fh)["baseline_hash"],
                                    baseline_hash(read_baseline(b)))

    def test_corpus_with_a_stray_non_csv_entry_is_unaffected(self):
        # §2.9: readers list demand days by *.csv only, because macOS rewrites .DS_Store
        with tempfile.TemporaryDirectory() as tmp:
            corpus = os.path.join(tmp, "corpus")
            shutil.copytree(CORPUS, corpus)
            with open(os.path.join(corpus, ".DS_Store"), "wb") as fh:
                fh.write(b"\x00\x01")
            with open(os.path.join(corpus, "notes.txt"), "w") as fh:
                fh.write("ignore me")
            self.assertEqual(training_day_files(corpus), training_day_files())
            self.assertEqual(corpus_hash(corpus_file_hashes(corpus)),
                             corpus_hash(corpus_file_hashes()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
