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
    FORBIDDEN_MANIFEST_KEYS,
    MANIFEST,
    SD_FLOOR,
    START_TIME,
    WEEKDAY_VOCAB,
    WINDOWS_S,
    _WEEKDAY_OF,
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
#: 4,482 = 3 x 747 x 2 since P2.18, which conditions the baseline on weekday; it was
#: 14,940 = 10 x 747 x 2 while it was conditioned on month.
N_ROWS = N_MINUTES * len(WEEKDAY_VOCAB) * len(WINDOWS_S)     # 4_482

#: P2.18's figures, asserted where the re-centring of the sd is what they pin.
#: 15:24, the minute where the Tue/Wed/Thu spread is widest and the month table's own sd
#: maximum at W = 30 -- which is what makes it the sharpest single check that the sd is taken
#: about the weekday curve and not a month curve.
RECENTRE_PROBE_T = 55440
SD_AT_PROBE_W30 = 7.6205              # weekday-centred; month-centred was 10.1478
SD_MONTH_CENTRED_AT_PROBE_W30 = 10.1478
DAYS_PER_WEEKDAY = {"Tue": 35, "Wed": 33, "Thu": 33}

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
        keys = {(w, t, win) for w, t, win, _mean, _sd in self.rows}
        self.assertEqual(len(keys), N_ROWS, "every (weekday, minute, window) appears once")
        self.assertEqual({w for w, *_ in self.rows}, set(WEEKDAY_VOCAB))
        self.assertEqual({win for _w, _t, win, *_ in self.rows}, set(WINDOWS_S))
        self.assertEqual({t for _w, t, *_ in self.rows}, set(minute_grid()))

    def test_the_weekday_column_holds_names_not_numbers(self):
        # P2.18 stores the weekday as its name, so the artifact reads without a lookup table
        for w, *_rest in self.rows:
            self.assertIsInstance(w, str)
            self.assertIn(w, WEEKDAY_VOCAB)

    def test_all_values_finite_and_non_negative(self):
        for w, t, win, mean, sd in self.rows:
            self.assertTrue(np.isfinite(mean) and np.isfinite(sd), (w, t, win))
            self.assertGreaterEqual(mean, 0.0)
            self.assertGreaterEqual(sd, 0.0)

    def test_sd_is_pooled_so_it_does_not_vary_by_weekday(self):
        # the sd is centred on each day's own weekday curve but pooled over all 101 days, so it
        # depends only on (minute, window) and the three weekday rows of a minute share it
        by_key = {}
        for _w, t, win, _mean, sd in self.rows:
            by_key.setdefault((t, win), set()).add(sd)
        offenders = {k: v for k, v in by_key.items() if len(v) != 1}
        self.assertEqual(offenders, {}, "sd must depend only on (minute, window)")

    def test_sixty_minute_means_are_at_least_thirty_minute_means(self):
        means = {(w, t, win): mean for w, t, win, mean, _sd in self.rows}
        for (w, t, win), mean in means.items():
            if win == 1800:
                self.assertLessEqual(mean, means[(w, t, 3600)] + 1e-9, (w, t))

    def test_first_minute_is_empty_by_construction(self):
        # the trailing window at t = start_time lies entirely before the day, so the count is 0
        # on every day; sd is 0 there too, which is what the SD floor exists for
        for win in WINDOWS_S:
            for w in WEEKDAY_VOCAB:
                row = next(r for r in self.rows if r[:3] == (w, START_TIME, win))
                self.assertEqual((row[3], row[4]), (0.0, 0.0))

    def test_rows_are_in_weekday_vocabulary_order(self):
        """Tue/Wed/Thu, not lexical -- a stable diff that also reads in calendar order (P2.18).

        A lexical file would be Thu/Tue/Wed, which looks like a defect. `canonical_baseline_text`
        still sorts lexically for hashing, so `baseline_hash` needs no ordering table; this is
        the file's order, which is the builder's emission order.
        """
        keys = [(WEEKDAY_VOCAB.index(w), t, win) for w, t, win, *_ in self.rows]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual([w for w, *_ in self.rows][0], "Tue")


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
        # mid-table, derived rather than a literal: a fixed 5000 was inside the 14,940-row month
        # table and is an IndexError against the 4,482-row weekday one (P2.18)
        i = len(rows) // 2
        w, t, win, mean, sd = tampered_row = rows[i]
        tampered = list(rows)
        tampered[i] = (w, t, win, mean + 0.01, sd)
        self.assertNotEqual(tampered[i], tampered_row)
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
        """The effect, measured on the bucket the row actually lands in.

        2024-02-15 is a **Thursday**, so since P2.18 the row inflates the Thursday bucket by
        1/33 rather than a February bucket by 1/10 -- the group it belongs to changed with the
        baseline, the one-sided inflation did not.
        """
        import pandas as pd
        day, rq_time = "2024-02-15", 37080
        grid = minute_grid()
        weekday = _WEEKDAY_OF[dt.date.fromisoformat(day).weekday()]
        self.assertEqual(weekday, "Thu")
        same_weekday = [d for d in training_day_files()
                        if _WEEKDAY_OF[dt.date.fromisoformat(d).weekday()] == weekday]
        self.assertEqual(len(same_weekday), DAYS_PER_WEEKDAY[weekday])
        t = rq_time + 60                      # a window that contains the row
        i = grid.index(t)

        def mean_at(use_both_filters):
            vals = []
            for d in same_weekday:
                path = os.path.join(CORPUS, f"{d}.csv")
                df = pd.read_csv(path)
                w = df[(df.rq_time >= START_TIME) & (df.rq_time < END_TIME)]
                if use_both_filters:
                    w = w[w.start != w.end]
                vals.append(counts_for_day(np.sort(w.rq_time.to_numpy()), grid, 1800)[i])
            return float(np.mean(vals))

        correct, inflated = mean_at(True), mean_at(False)
        self.assertAlmostEqual(inflated - correct, 1.0 / len(same_weekday), places=9)
        committed = {(w, tt, ww): mean for w, tt, ww, mean, _sd in read_baseline(BASELINE)}
        self.assertAlmostEqual(committed[(weekday, t, 1800)], correct, places=9,
                               msg="the committed baseline must use both filters")


class TestWeekdayOverMonth(unittest.TestCase):
    """The figures behind "why weekday, not month" (§8 P2.6, as P2.18 rewrote it).

    These were the justification for conditioning on **month** through P2.17, and the same
    measurements now justify conditioning on **weekday** — because what changed is which of
    them the decision rests on, not what the corpus says. So this class pins both sides,
    including the two that favour month, and the directions are asserted as directions so they
    survive the corpus being extended.

    What is deliberately **not** asserted here is that weekday beats month on fit or on
    prediction. It does not, on either metric, and `TestWeekdayBaselineIsRecentred` plus the
    spec record that as an observation rather than a criterion.
    """

    @classmethod
    def setUpClass(cls):
        days = training_day_files()
        grid = minute_grid()
        cls.days = days
        cls.X = np.vstack([counts_for_day(
            kept_arrival_times(os.path.join(CORPUS, f"{d}.csv")), grid, 1800) for d in days])
        cls.months = np.array([dt.date.fromisoformat(d).month for d in days])
        cls.wd = np.array([_WEEKDAY_OF[dt.date.fromisoformat(d).weekday()] for d in days])
        cls.grid = grid

    def _spread(self, keys):
        B = np.zeros_like(self.X)
        for k in set(keys):
            sel = keys == k
            B[sel] = self.X[sel].mean(axis=0)
        return float(np.mean(np.std(self.X - B, axis=0, ddof=1)))

    def test_whole_day_spread_still_favours_month_and_is_recorded_not_optimised(self):
        """4.75 month, 5.20 weekday, 5.45 none — unchanged, and *against* P2.18's choice.

        Kept because it is true: 3 parameters against 10 leave more unexplained **on average**,
        and the average is not where `arrivals_z_*` does its work. The whole-day number is the
        one figure that most invites re-deriving the old decision, so it is asserted with the
        ordering spelled out rather than quietly dropped.
        """
        month, weekday = self._spread(self.months), self._spread(self.wd)
        none = float(np.mean(np.std(self.X - self.X.mean(axis=0), axis=0, ddof=1)))
        self.assertLess(month, weekday, "month still fits better averaged over the day")
        self.assertLess(weekday, none)
        self.assertAlmostEqual(month, 4.75, delta=0.15)
        self.assertAlmostEqual(weekday, 5.20, delta=0.15)
        self.assertAlmostEqual(none, 5.45, delta=0.15)

    def test_month_explains_the_daily_level_and_weekday_explains_none_of_it(self):
        """56% against 0.2% — and the level is already in the observation as `arrivals_30/60`.

        This is why month is not what the anomaly conditions on: it predicts the *level*, which
        the raw arrival counts carry unlabelled, rather than the within-day shape.
        """
        totals = self.X.sum(axis=1)

        def share(keys):
            ss_res = sum(((totals[keys == k] - totals[keys == k].mean()) ** 2).sum()
                         for k in set(keys))
            return 1 - ss_res / ((totals - totals.mean()) ** 2).sum()

        self.assertLess(share(self.wd), 0.05, "weekday explains essentially no daily total")
        self.assertGreater(share(self.months), 0.4, "month explains a large share")

    def test_weekday_is_the_within_day_shape_at_the_afternoon_peak(self):
        """Tue 21.7 against Wed 37.3 in the window ending 15:30 — the structure P2.18 conditions
        on, and the bias a month-only table carried for about an hour of every day."""
        i = self.grid.index(55800)
        tue = self.X[self.wd == "Tue", i].mean()
        wed = self.X[self.wd == "Wed", i].mean()
        overall = self.X[:, i].mean()
        self.assertLess(tue, overall)
        self.assertGreater(wed, overall)
        self.assertGreater(wed - tue, 10.0, "the weekday spread at 15:30 is large")

    def test_the_accepted_bias_is_a_standing_month_offset(self):
        """October +0.69 SD, August −0.35, against the **shipped** table.

        P2.6 cited this figure as a reason to reject a weekday baseline; P2.18 accepts it, in
        exchange for removing a +1.07 SD Wednesday bias at the afternoon peak (§8 P2.6). It is
        asserted so the trade stays visible and measured rather than remembered.
        """
        B = np.zeros_like(self.X)
        for k in set(self.wd):
            sel = self.wd == k
            B[sel] = self.X[sel].mean(axis=0)
        res = self.X - B
        sd = np.std(res, axis=0, ddof=1)
        z = res / np.maximum(sd, SD_FLOOR)
        self.assertGreater(z[self.months == 10].mean(), 0.5)
        self.assertLess(z[self.months == 8].mean(), -0.2)
        # whole-day weekday bias is gone by construction, which is the point of conditioning
        for w in WEEKDAY_VOCAB:
            self.assertAlmostEqual(float(z[self.wd == w].mean()), 0.0, places=9)


class TestWeekdayBaselineIsRecentred(unittest.TestCase):
    """**The one defect no other check in this repository would catch** (P2.18).

    A weekday mean paired with a month-centred sd passes `--check`, because that compares a
    fresh build against the committed pair and would be comparing the defect with itself. It
    also passes `test_sd_is_pooled_so_it_does_not_vary_by_weekday`, since a month-centred sd
    replicated across three weekday rows is still one sd per `(minute, window)`. It would give
    a wrong `z` at every decision of every episode. So the centring is pinned directly.
    """

    @classmethod
    def setUpClass(cls):
        cls.days = training_day_files()
        cls.grid = minute_grid()
        times = {d: kept_arrival_times(os.path.join(CORPUS, f"{d}.csv")) for d in cls.days}
        cls.counts = {w: {d: counts_for_day(times[d], cls.grid, w) for d in cls.days}
                      for w in WINDOWS_S}
        cls.weekday_of = {d: _WEEKDAY_OF[dt.date.fromisoformat(d).weekday()] for d in cls.days}
        cls.month_of = {d: dt.date.fromisoformat(d).month for d in cls.days}
        cls.committed = {(w, t, win): sd for w, t, win, _mu, sd in read_baseline(BASELINE)}

    def _centred_sd(self, window_s, group_of):
        counts = self.counts[window_s]
        groups = {}
        for d in self.days:
            groups.setdefault(group_of[d], []).append(d)
        means = {g: np.mean([counts[d] for d in ds], axis=0) for g, ds in groups.items()}
        residuals = np.vstack([counts[d] - means[group_of[d]] for d in self.days])
        return residuals.std(axis=0, ddof=1)

    def test_days_per_weekday(self):
        counts = {}
        for d in self.days:
            counts[self.weekday_of[d]] = counts.get(self.weekday_of[d], 0) + 1
        self.assertEqual(counts, DAYS_PER_WEEKDAY)

    def test_committed_sd_is_centred_on_the_weekday_curve(self):
        for window_s in WINDOWS_S:
            want = self._centred_sd(window_s, self.weekday_of)
            for i, t in enumerate(self.grid):
                for weekday in WEEKDAY_VOCAB:
                    self.assertAlmostEqual(self.committed[(weekday, t, window_s)], want[i],
                                           places=9, msg=f"{weekday} t={t} W={window_s}")

    def test_the_probe_minute_distinguishes_the_two_centrings(self):
        """7.6205, not 10.1478, at t = 55440 and W = 30 — the sharpest single check.

        15:24 is where the Tue/Wed/Thu means are furthest apart (20.63 / 38.79 / 23.21) and is
        the month-centred table's own sd maximum. A month-centred sd would read 10.1478 here
        and every other check in the suite would still pass.
        """
        committed = self.committed[("Tue", RECENTRE_PROBE_T, 1800)]
        self.assertAlmostEqual(committed, SD_AT_PROBE_W30, places=4)
        month_centred = self._centred_sd(1800, self.month_of)[self.grid.index(RECENTRE_PROBE_T)]
        self.assertAlmostEqual(month_centred, SD_MONTH_CENTRED_AT_PROBE_W30, places=4)
        self.assertLess(committed, month_centred - 2.0,
                        "the weekday-centred sd must be materially smaller at the probe")

    def test_recentring_lowers_the_peak_and_raises_the_average(self):
        """Both directions, since only reporting one of them would read as a free improvement."""
        for window_s, mean_was, mean_now, max_was, max_now in (
                (1800, 4.750, 5.198, 10.148, 9.192),
                (3600, 7.042, 8.158, 13.616, 12.796)):
            weekday_sd = self._centred_sd(window_s, self.weekday_of)
            month_sd = self._centred_sd(window_s, self.month_of)
            self.assertAlmostEqual(float(month_sd.mean()), mean_was, places=2)
            self.assertAlmostEqual(float(weekday_sd.mean()), mean_now, places=2)
            self.assertAlmostEqual(float(month_sd.max()), max_was, places=2)
            self.assertAlmostEqual(float(weekday_sd.max()), max_now, places=2)
            self.assertGreater(weekday_sd.mean(), month_sd.mean())
            self.assertLess(weekday_sd.max(), month_sd.max())
            # minutes below the SD floor, where the floor is what carries the division
            self.assertEqual(int((weekday_sd < SD_FLOOR).sum()), 2)
            self.assertEqual(float(weekday_sd[self.grid.index(START_TIME)]), 0.0)


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
        self.assertEqual(self.m["weekday_vocab"], WEEKDAY_VOCAB)

    def test_the_committed_manifest_carries_no_month_vocabulary(self):
        """P2.19. The key fed the request block's one-hot and nothing else now reads it."""
        self.assertNotIn("month_vocab", self.m)
        for key, _why in FORBIDDEN_MANIFEST_KEYS:
            self.assertNotIn(key, self.m)

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

    def test_check_refuses_a_manifest_that_still_carries_month_vocab(self):
        """P2.19. **The one thing that tells a P2.18 manifest from a P2.19 one.**

        P2.19 changed neither artifact: `corpus/` is untouched and `arrival_baseline.csv` is
        byte-identical, so `corpus_hash` and `baseline_hash` are the same in both manifests and
        every other `--check` comparison passes on a stale one. Without an explicit
        forbidden-key check a manifest built before P2.19 would therefore pass silently, which
        is the whole reason the check exists.
        """
        import contextlib
        import io

        import src.rl_gym.offline.build_arrival_baseline as builder

        rows, _ = compute_baseline()
        with tempfile.TemporaryDirectory() as tmp:
            b = os.path.join(tmp, "arrival_baseline.csv")
            mf = os.path.join(tmp, "corpus_manifest.json")
            write_both(rows, CORPUS, b, mf)

            def run_check():
                out = io.StringIO()
                argv = ["build_arrival_baseline", "--check", "--corpus", CORPUS,
                        "--baseline", b, "--manifest", mf]
                saved, sys.argv = sys.argv, argv
                try:
                    with contextlib.redirect_stdout(out):
                        code = builder.main()
                finally:
                    sys.argv = saved
                return code, out.getvalue()

            code, text = run_check()
            self.assertEqual(code, 0, f"a fresh pair must pass --check:\n{text}")

            with open(mf) as fh:
                manifest = json.load(fh)
            manifest["month_vocab"] = list(range(1, 11))       # as P2.18 wrote it
            with open(mf, "w") as fh:
                json.dump(manifest, fh, indent=1, sort_keys=False)
                fh.write("\n")

            code, text = run_check()
            self.assertEqual(code, 1, "a manifest carrying month_vocab must fail --check")
            self.assertIn("month_vocab", text)
            self.assertIn("P2.19", text)
            self.assertIn("build_arrival_baseline", text, "name the rebuild command")
            # and the hashes it would otherwise be judged on are genuinely unchanged
            self.assertEqual(manifest["baseline_hash"], baseline_hash(read_baseline(b)))
            self.assertEqual(manifest["corpus_hash"], corpus_hash(corpus_file_hashes(CORPUS)))

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
