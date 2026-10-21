"""The cloud-synced folder guard and the output-row integrity check (P2.20).

P2.19's KPI table did not reproduce because the repository sat inside a OneDrive-synced folder,
and the OneDrive file provider silently dropped a contiguous band of appended rows from
`1_user-stats.csv`. Two things are tested here:

- **the guard**: a path that resolves into a cloud-synced folder is refused, a local one is
  accepted, and every entry point refuses **before** it creates anything;
- **the check**: a file with a missing band raises with both counts and the path, and the KPI
  harness refuses such a file, and a directory with no record at all.

No simulation runs here and **torch is never imported** (§8a): every entry point checks before
its heavy imports. The end-to-end halves -- a real episode's record matching its files, and
one log per episode -- are in `tests/test_rl_gym_env.py`; the record's counts against both
committed baselines are a check of `python -m src.rl_gym.evaluate --verify`.

Run just these with:
    python -m unittest tests.test_rl_gym_output_integrity
"""

import numpy as np  # noqa: F401  -- §8a: numpy before anything that might pull in torch

import csv
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rl_gym import evaluate, kpis  # noqa: E402
from src.rl_gym import output_integrity as oi  # noqa: E402
from src.rl_gym.output_integrity import OutputIntegrityError, SyncedFolderError  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASELINE_USER_STATS = os.path.join(ROOT, "docs", "baseline_user_stats.csv")
BASELINE_OP_STATS = os.path.join(ROOT, "docs", "baseline_op_stats.csv")
SCS = os.path.join(ROOT, "studies", "example_study", "scenarios")

#: the two baselines' data rows, counted with `csv.reader` in P2.20
USER_ROWS = 445
OP_ROWS = 1140

#: makes the repository's own results directory look synced, so an entry point's refusal can be
#: tested without a real synced folder: every run's output path contains this component
RESULTS_LOOK_SYNCED = mock.patch.object(oi, "SYNCED_COMPONENT_MARKERS", ("example_study",))


def make_tmp(test):
    tmp = tempfile.mkdtemp(prefix="p220_")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    return tmp


def synced_dir(test):
    """A real local temporary directory whose path carries a OneDrive component, so the guard
    sees it as synced. Nothing here touches an actual synced folder."""
    path = os.path.join(make_tmp(test), "OneDrive - Test")
    os.makedirs(path)
    return path


# -------------------------------------------------------------------------- #
# the guard
# -------------------------------------------------------------------------- #

class TestSyncedFolderGuard(unittest.TestCase):
    """A path resolving into a cloud-synced folder is refused; a local one is accepted."""

    MARKED = [
        ("OneDrive",),
        ("OneDrive - TUM",),
        ("OneDrive-Personal",),
        ("onedrive",),
        ("Dropbox",),
        ("Google Drive",),
        ("Library", "CloudStorage", "GoogleDrive-someone@example.com", "My Drive"),
        ("Library", "CloudStorage", "Box-Box"),
        ("Library", "Mobile Documents", "com~apple~CloudDocs"),
    ]

    def test_each_marker_is_refused(self):
        tmp = make_tmp(self)
        for components in self.MARKED:
            path = os.path.join(tmp, *components, "repo", "runs", "p220_gate")
            with self.subTest(path=path):
                with self.assertRaises(SyncedFolderError):
                    oi.refuse_synced_path(path, "test output")

    def test_a_local_path_is_accepted(self):
        tmp = make_tmp(self)
        for path in (tmp, os.path.join(tmp, "runs", "not_created_yet")):
            with self.subTest(path=path):
                self.assertEqual(oi.refuse_synced_path(path, "test output"),
                                 os.path.realpath(path))

    def test_a_symlink_into_a_synced_folder_is_refused_after_resolving(self):
        """The path as given names nothing synced; where the bytes go does."""
        tmp = make_tmp(self)
        real = os.path.join(tmp, "OneDrive - Test", "FleetPy_stochasticity")
        os.makedirs(real)
        link = os.path.join(tmp, "code")
        os.symlink(real, link)
        given = os.path.join(link, "runs", "p220_gate")
        self.assertIsNone(oi.synced_folder_match(given.replace(link, "/nowhere/code")),
                          "the link's own path must carry no marker for this test to mean "
                          "anything")
        with self.assertRaises(SyncedFolderError) as cm:
            oi.refuse_synced_path(given, "test output")
        message = str(cm.exception)
        self.assertIn(given, message)
        self.assertIn(os.path.realpath(given), message, "name the resolved path too")

    def test_a_symlink_merely_named_like_a_sync_folder_is_judged_by_its_target(self):
        """`OneDrive` -> a local folder writes locally, so it is accepted."""
        tmp = make_tmp(self)
        target = os.path.join(tmp, "local_target")
        os.makedirs(target)
        os.symlink(target, os.path.join(tmp, "OneDrive"))
        resolved = oi.refuse_synced_path(os.path.join(tmp, "OneDrive", "runs"), "test output")
        self.assertEqual(resolved, os.path.join(os.path.realpath(target), "runs"))

    def test_the_message_names_the_path_what_and_the_fix(self):
        path = os.path.join(make_tmp(self), "Dropbox", "runs")
        with self.assertRaises(SyncedFolderError) as cm:
            oi.refuse_synced_path(path, "simulation output")
        message = str(cm.exception)
        for needle in (path, "simulation output", "'Dropbox'", "P2.20", "~/code/"):
            with self.subTest(needle):
                self.assertIn(needle, message)

    def test_desktop_and_documents_carry_no_marker_and_are_not_refused(self):
        """Ritun's decision in P2.20: only the listed markers, not iCloud's Desktop & Documents
        option, which leaves no trace in the path. A string-only check, so it does not depend
        on how this machine's own home folders are set up."""
        for folder in ("Desktop", "Documents"):
            with self.subTest(folder):
                self.assertIsNone(oi.synced_folder_match(
                    os.path.join("/nonexistent_home_p220", folder, "repo")))


class TestEveryEntryPointRefusesBeforeCreatingAnything(unittest.TestCase):
    """The guard is wired in where output is chosen, and runs first."""

    def test_train_refuses_a_synced_out_dir(self):
        import train_sdpdp
        out_dir = os.path.join(synced_dir(self), "runs", "p220")
        with self.assertRaises(SyncedFolderError):
            train_sdpdp.train(dict(train_sdpdp.DEFAULT_CFG), out_dir=out_dir)
        self.assertFalse(os.path.exists(out_dir), "train() created the folder before refusing")

    def test_phase2_run_refuses_a_synced_out_dir(self):
        from src.rl_gym.tools import phase2_run
        out_dir = os.path.join(synced_dir(self), "p220_gate")
        with mock.patch.object(sys, "argv", ["phase2_run", "--out-dir", out_dir,
                                             "--skip-harness"]):
            with self.assertRaises(SyncedFolderError):
                phase2_run.main()
        self.assertFalse(os.path.exists(out_dir))

    def test_phase2_run_refuses_a_synced_results_directory_when_the_harness_runs(self):
        from src.rl_gym.tools import phase2_run
        out_dir = os.path.join(make_tmp(self), "p220_check")
        with RESULTS_LOOK_SYNCED, mock.patch.object(
                sys, "argv", ["phase2_run", "--out-dir", out_dir, "--harness-only",
                              "--checkpoint", "unused.zip"]):
            with self.assertRaises(SyncedFolderError) as cm:
                phase2_run.main()
        self.assertIn("harness", str(cm.exception))
        self.assertFalse(os.path.exists(out_dir))

    def test_evaluate_refuses_a_synced_report_path_before_any_run(self):
        out = os.path.join(synced_dir(self), "kpi_report.txt")
        argv = ["evaluate", "--validation", "--greedy", "--days", "2024-06-20", "--seeds", "1",
                "--out", out]
        never = mock.Mock(side_effect=AssertionError("a run started before the refusal"))
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(evaluate, "run_greedy_pairs", never):
            with self.assertRaises(SyncedFolderError):
                evaluate.main()
        self.assertFalse(os.path.exists(out))

    def test_evaluate_runners_and_verify_refuse_a_synced_results_directory(self):
        with RESULTS_LOOK_SYNCED:
            with self.subTest("run_greedy_pairs"):
                with self.assertRaises(SyncedFolderError):
                    evaluate.run_greedy_pairs([("2024-06-20", 42)])
            with self.subTest("run_rl_pairs"):
                with self.assertRaises(SyncedFolderError):
                    evaluate.run_rl_pairs([("2024-06-20", 42)], "unused.zip")
            with self.subTest("build_greedy_scenario_parameters"):
                with self.assertRaises(SyncedFolderError):
                    evaluate.build_greedy_scenario_parameters("2024-06-20", 42)
            with self.subTest("verify"):
                with self.assertRaises(SyncedFolderError):
                    evaluate.verify(verbose=False)

    def test_the_env_refuses_at_construction_with_output_on_only(self):
        from src.rl_gym.gym_env import SDPDPAssignmentEnv
        cfg = {
            "constant_cfg_path": os.path.join(SCS, "constant_config_depot_cali_sc_1.csv"),
            "scenario_cfg_path": os.path.join(SCS, "example_depot_cali_sc_1.csv"),
            "K": 8,
        }
        with RESULTS_LOOK_SYNCED:
            with self.assertRaises(SyncedFolderError) as cm:
                SDPDPAssignmentEnv(dict(cfg, skip_output=False))
            self.assertIn("simulation output", str(cm.exception))
            # output off writes nothing, so nothing is checked
            env = SDPDPAssignmentEnv(dict(cfg, skip_output=True))
            env.close()


# -------------------------------------------------------------------------- #
# counting what a simulation writes
# -------------------------------------------------------------------------- #

class _StubDemand:
    """`Demand.save_user_stats`'s condition and its rebinding clear, without a file."""

    BUFFER = 3

    def __init__(self):
        self.user_stat_buffer = []
        self.written = 0

    def save_user_stats(self, force=True):
        n = len(self.user_stat_buffer)
        if (n and force) or n >= self.BUFFER:
            self.written += n
            self.user_stat_buffer = []


class _StubSim:
    """`record_stats`' condition and its in-place clear of the list shared with vehicles."""

    BUFFER = 3

    def __init__(self, n_op=2):
        self.n_op = n_op
        self.demand = _StubDemand()
        self.op_output = {op_id: [] for op_id in range(n_op)}
        self.written = {op_id: 0 for op_id in range(n_op)}

    def record_stats(self, force=True):
        self.demand.save_user_stats(force)
        for op_id in range(self.n_op):
            n = len(self.op_output[op_id])
            if (n and force) or n > self.BUFFER:
                self.written[op_id] += n
                self.op_output[op_id].clear()


class TestRowCounter(unittest.TestCase):

    def test_counts_exactly_the_rows_flushed_by_either_route(self):
        sim = _StubSim()
        shared = sim.op_output[0]
        counter = oi.RowCounter.attach(sim)
        sim.demand.user_stat_buffer += [1, 2]
        sim.op_output[0] += [1]
        sim.record_stats(force=False)           # under both thresholds: nothing written
        self.assertEqual(counter.rows[oi.USER_STATS_FILE], 0)
        self.assertEqual(counter.rows[oi.op_stats_file(0)], 0)
        sim.op_output[1] += [1, 2, 3, 4, 5]
        sim.record_stats()                      # forced, as at the end of every time step
        sim.demand.user_stat_buffer += [3, 4, 5]
        sim.demand.save_user_stats(force=True)  # record_remaining_users' trailing flush
        sim.record_stats()                      # nothing pending: nothing counted
        self.assertEqual(counter.rows, {oi.USER_STATS_FILE: sim.demand.written,
                                        oi.op_stats_file(0): sim.written[0],
                                        oi.op_stats_file(1): sim.written[1]})
        self.assertEqual(counter.rows, {oi.USER_STATS_FILE: 5, oi.op_stats_file(0): 1,
                                        oi.op_stats_file(1): 5})
        self.assertIs(sim.op_output[0], shared,
                      "the list shared with the vehicles must never be rebound (P1.2)")

    def test_wraps_only_the_one_object(self):
        sim, other = _StubSim(), _StubSim()
        oi.RowCounter.attach(sim)
        self.assertIn("record_stats", vars(sim))
        self.assertNotIn("record_stats", vars(other))
        self.assertNotIn("save_user_stats", vars(other.demand))


# -------------------------------------------------------------------------- #
# checking the files
# -------------------------------------------------------------------------- #

def read_rows(path):
    with open(path, newline="") as fh:
        return list(csv.reader(fh))


def write_rows(path, rows):
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)


class TestRowIntegrity(unittest.TestCase):

    def run_dir(self):
        """A checked run: both baselines, and the record a passing check would have written."""
        tmp = make_tmp(self)
        shutil.copy(BASELINE_USER_STATS, os.path.join(tmp, oi.USER_STATS_FILE))
        shutil.copy(BASELINE_OP_STATS, os.path.join(tmp, oi.op_stats_file(0)))
        oi.write_rows_record(tmp, {oi.USER_STATS_FILE: USER_ROWS,
                                   oi.op_stats_file(0): OP_ROWS})
        return tmp

    def drop_band(self, path, first, count):
        """Remove `count` consecutive data rows starting at data row `first` -- the shape
        P2.20's diagnosis found: whole rows, one flush, out of the middle of the file."""
        rows = read_rows(path)
        del rows[1 + first:1 + first + count]
        write_rows(path, rows)

    def test_the_baselines_count_as_measured(self):
        self.assertEqual(oi.count_data_rows(BASELINE_USER_STATS), USER_ROWS)
        self.assertEqual(oi.count_data_rows(BASELINE_OP_STATS), OP_ROWS)

    def test_a_whole_run_passes_and_the_kpi_row_builds(self):
        tmp = self.run_dir()
        self.assertEqual(oi.verify_output_dir(tmp), {oi.USER_STATS_FILE: USER_ROWS,
                                                     oi.op_stats_file(0): OP_ROWS})
        self.assertEqual(kpis.kpi_row(tmp)["requests"], USER_ROWS)

    def test_a_missing_band_raises_with_both_counts_and_the_path(self):
        tmp = self.run_dir()
        path = os.path.join(tmp, oi.USER_STATS_FILE)
        self.drop_band(path, 120, 4)
        with self.assertRaises(OutputIntegrityError) as cm:
            oi.verify_output_dir(tmp)
        message = str(cm.exception)
        for needle in (path, "441 data row(s) in the file", f"wrote {USER_ROWS}",
                       "4 row(s) missing"):
            with self.subTest(needle):
                self.assertIn(needle, message)

    def test_a_missing_band_in_op_stats_raises_too(self):
        tmp = self.run_dir()
        path = os.path.join(tmp, oi.op_stats_file(0))
        self.drop_band(path, 500, 56)
        with self.assertRaises(OutputIntegrityError) as cm:
            oi.verify_output_dir(tmp)
        self.assertIn(path, str(cm.exception))
        self.assertIn(f"{OP_ROWS - 56} data row(s) in the file", str(cm.exception))

    def test_the_kpi_harness_refuses_a_file_with_a_missing_band(self):
        tmp = self.run_dir()
        self.drop_band(os.path.join(tmp, oi.USER_STATS_FILE), 120, 4)
        with self.assertRaises(OutputIntegrityError):
            kpis.kpi_row(tmp)

    def test_the_kpi_harness_refuses_a_directory_with_no_record(self):
        """An abandoned auto-reset episode, a pre-P2.20 run, or one that stopped early."""
        tmp = self.run_dir()
        os.remove(os.path.join(tmp, oi.ROWS_RECORD_FILE))
        with self.assertRaises(OutputIntegrityError) as cm:
            kpis.kpi_row(tmp)
        self.assertIn(oi.ROWS_RECORD_FILE, str(cm.exception))
        self.assertIn("abandoned", str(cm.exception))

    def test_extra_rows_raise(self):
        tmp = self.run_dir()
        path = os.path.join(tmp, oi.USER_STATS_FILE)
        rows = read_rows(path)
        write_rows(path, rows + rows[1:3])
        with self.assertRaises(OutputIntegrityError) as cm:
            oi.verify_output_dir(tmp)
        self.assertIn("2 row(s) extra", str(cm.exception))

    def test_a_repeated_header_raises_even_when_the_count_would_match(self):
        """The file was re-created mid-run: `isfile` chose mode "w" again."""
        tmp = self.run_dir()
        path = os.path.join(tmp, oi.USER_STATS_FILE)
        rows = read_rows(path)
        rows[200] = rows[0]                    # same number of rows, one of them a header
        write_rows(path, rows)
        with self.assertRaises(OutputIntegrityError) as cm:
            oi.verify_output_dir(tmp)
        self.assertIn("header repeats", str(cm.exception))

    def test_a_torn_or_blank_line_raises(self):
        for label, broken in (("torn", lambda r: r[:5]), ("blank", lambda r: [])):
            with self.subTest(label):
                tmp = self.run_dir()
                path = os.path.join(tmp, oi.USER_STATS_FILE)
                rows = read_rows(path)
                rows[300] = broken(rows[300])
                write_rows(path, rows)
                with self.assertRaises(OutputIntegrityError) as cm:
                    oi.verify_output_dir(tmp)
                self.assertIn("field(s) against the header's", str(cm.exception))

    def test_a_missing_file_raises_unless_nothing_was_written_to_it(self):
        tmp = self.run_dir()
        os.remove(os.path.join(tmp, oi.op_stats_file(0)))
        with self.assertRaises(OutputIntegrityError) as cm:
            oi.verify_output_dir(tmp)
        self.assertIn("missing", str(cm.exception))
        oi.verify_rows(tmp, {oi.op_stats_file(0): 0})     # FleetPy never created it: fine

    def test_the_record_is_written_only_when_the_check_passes(self):
        tmp = self.run_dir()
        os.remove(os.path.join(tmp, oi.ROWS_RECORD_FILE))
        counter = oi.RowCounter(n_op=1)
        counter.rows.update({oi.USER_STATS_FILE: USER_ROWS + 4, oi.op_stats_file(0): OP_ROWS})
        with self.assertRaises(OutputIntegrityError):
            oi.check_episode_output(tmp, counter)
        self.assertFalse(os.path.isfile(os.path.join(tmp, oi.ROWS_RECORD_FILE)))
        counter.rows[oi.USER_STATS_FILE] = USER_ROWS
        path = oi.check_episode_output(tmp, counter)
        self.assertEqual(oi.read_rows_record(tmp), counter.rows)
        self.assertEqual(path, os.path.join(tmp, oi.ROWS_RECORD_FILE))


if __name__ == "__main__":
    unittest.main(verbosity=2)
