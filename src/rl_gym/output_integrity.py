"""Where simulation output may be written, and proof that it was written whole (P2.20).

**Why this exists.** P2.19's KPI table did not reproduce: three runs of the same ten pairs gave
three md5s. The simulation was deterministic throughout. `1_user-stats.csv` had **lost whole
rows** -- a contiguous band of complete lines, one or more `save_user_stats` flushes, out of the
middle of a file that only ever receives sequential appends from one thread. The repository was
inside a OneDrive-synced folder, and a storage test at episode pace (400 two-row appends, 0.15 s
apart) lost 4, 0, 4 and 454 rows there, each as one contiguous band, and 0 rows in `/private/tmp`.
**The OneDrive file provider silently discards a band of appends to a file appended to over a
sustained period.** Nothing raised and nothing was logged; the file simply came back shorter.

Two halves, one guarantee -- the CSVs a KPI table is built from hold every row the simulation
handed to the writer:

- **The guard, before a run** (`refuse_synced_path`). A run that would write into a cloud-synced
  folder is refused before anything is created. Judged on the **resolved** path, so a symlink
  into a synced folder is refused and a symlink merely *named* like one, pointing somewhere
  local, is not: what matters is where the bytes go.
- **The check, after a run** (`RowCounter`, `check_episode_output`). Rows are counted where they
  leave the simulation's buffers, compared with the rows in each file once the episode has
  finished, and the counts are written to `rl_output_rows.json` beside the CSVs. `kpis.kpi_row`
  re-counts against that record (`verify_output_dir`) and refuses a directory without one. The
  re-count is not redundant: the lost band in P2.19 disappeared *after* flushes that had written
  correctly, so a file can pass at the end of its episode and be short by the time it is read.

Pure standard library, and nothing here imports FleetPy or the env, so `kpis.py` can use it
without pulling in either.
"""

import csv
import json
import os
from typing import Dict, Optional

#: the two files the check covers -- FleetPy's names, which it builds inline
#: (`FleetSimulationBase.py` ~:253 and ~:646)
USER_STATS_FILE = "1_user-stats.csv"
OP_STATS_TEMPLATE = "2-{op_id}_op-stats.csv"

#: the record of what the simulation wrote, beside the CSVs it describes
ROWS_RECORD_FILE = "rl_output_rows.json"

#: a path component containing any of these (case-insensitively) names a sync client's folder:
#: `OneDrive`, `OneDrive - <org>`, `OneDrive-Personal`, `Dropbox`, `Google Drive`,
#: `GoogleDrive-<account>`
SYNCED_COMPONENT_MARKERS = ("onedrive", "dropbox", "google drive", "googledrive")

#: iCloud Drive's storage folder, `~/Library/Mobile Documents/`, matched as a whole component
ICLOUD_COMPONENT = "mobile documents"

#: the macOS File Provider root, `~/Library/CloudStorage/`, matched as two consecutive
#: components so the rule does not depend on whose home directory it is
CLOUD_STORAGE_COMPONENTS = ("library", "cloudstorage")


class SyncedFolderError(RuntimeError):
    """A run would write inside a cloud-synced folder."""


class OutputIntegrityError(RuntimeError):
    """An output file does not hold the rows the simulation wrote to it."""


def op_stats_file(op_id: int) -> str:
    """`2-<op_id>_op-stats.csv`."""
    return OP_STATS_TEMPLATE.format(op_id=int(op_id))


# -------------------------------------------------------------------------- #
# the guard
# -------------------------------------------------------------------------- #

def resolve(path: str) -> str:
    """Absolute, user-expanded and symlink-resolved. A path that does not exist yet resolves as
    far as it exists, which is what a run directory about to be created needs."""
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))


def synced_folder_match(path: str) -> Optional[str]:
    """Why `path` is inside a cloud-synced folder, or None if it is not. Judged on `resolve(path)`."""
    parts = [p for p in resolve(path).split(os.sep) if p]
    lowered = [p.lower() for p in parts]
    first, second = CLOUD_STORAGE_COMPONENTS
    for i in range(len(lowered) - 1):
        if lowered[i] == first and lowered[i + 1] == second:
            root = os.sep + os.path.join(*parts[:i + 2])
            return (f"it is under {root}, the macOS File Provider root that OneDrive, Google "
                    f"Drive and Dropbox mount into")
    for part, low in zip(parts, lowered):
        if low == ICLOUD_COMPONENT:
            return f"path component {part!r} is iCloud Drive's storage folder"
        for marker in SYNCED_COMPONENT_MARKERS:
            if marker in low:
                return f"path component {part!r} names a sync client's folder ({marker!r})"
    return None


def refuse_synced_path(path: str, what: str) -> str:
    """Raise `SyncedFolderError` if `path` resolves into a cloud-synced folder.

    Call it **before** anything is created -- a refusal should cost no simulation and orphan no
    worker. There is no override.

    :param what: what would be written there, for the message
    :return: the resolved path
    """
    resolved = resolve(path)
    reason = synced_folder_match(resolved)
    if reason is None:
        return resolved
    shown = repr(str(path))
    if resolved != os.path.abspath(os.path.expanduser(str(path))):
        shown += f" (resolves to {resolved!r})"
    raise SyncedFolderError(
        f"refusing to write {what} to {shown}: it is inside a cloud-synced folder -- {reason}. "
        f"A sync client rewrites the files it syncs: in P2.20 the OneDrive file provider "
        f"silently discarded a contiguous band of appended rows from 1_user-stats.csv, with no "
        f"error anywhere, and the KPI table stopped reproducing. Move the repository (or this "
        f"output) to a local folder no sync client watches -- such as ~/code/ -- and run again."
    )


# -------------------------------------------------------------------------- #
# counting what the simulation writes
# -------------------------------------------------------------------------- #

class RowCounter:
    """Rows a simulation handed to the CSV writer, per file, counted at each flush.

    `attach` wraps two methods **on the simulation object itself** -- `sim.demand.save_user_stats`
    and `sim.record_stats` -- which are the only places `1_user-stats.csv` and
    `2-<op>_op-stats.csv` are written (`demand.py` ~:191, `FleetSimulationBase.py` ~:654).
    Every flush reaches them through `self.`, including the trailing `save_user_stats` of
    `record_remaining_users`, so instance attributes see all of them. No FleetPy code is edited
    and the class is untouched: only this one object counts.

    A flush is recognised by its effect: the buffer was non-empty before the call and empty
    after it. That holds for both of FleetPy's clears -- `user_stat_buffer` is rebound to `[]`,
    `op_output[op_id]` is cleared in place (P1.2) -- and an unflushed buffer keeps its length, so
    the two cases cannot be confused. Attach only when output is on: with `skip_output` the same
    buffers are cleared without writing, and the count would describe a file that does not
    exist.
    """

    def __init__(self, n_op: int):
        #: file name -> rows written to it so far
        self.rows: Dict[str, int] = {USER_STATS_FILE: 0}
        for op_id in range(int(n_op)):
            self.rows[op_stats_file(op_id)] = 0

    @classmethod
    def attach(cls, sim) -> "RowCounter":
        """Start counting `sim`'s flushes. Call after construction and before `run()`.

        Nothing is flushed during construction -- `record_stats` and `save_user_stats` are first
        reached from the time loop -- so no row escapes the count.
        """
        counter = cls(sim.n_op)
        demand = sim.demand
        save_user_stats = demand.save_user_stats
        record_stats = sim.record_stats

        def counted_save_user_stats(force=True):
            pending = len(demand.user_stat_buffer)
            save_user_stats(force)
            if pending and not demand.user_stat_buffer:
                counter.rows[USER_STATS_FILE] += pending

        def counted_record_stats(force=True):
            pending = [len(sim.op_output[op_id]) for op_id in range(sim.n_op)]
            record_stats(force)
            for op_id, n in enumerate(pending):
                if n and not sim.op_output[op_id]:
                    counter.rows[op_stats_file(op_id)] += n

        demand.save_user_stats = counted_save_user_stats
        sim.record_stats = counted_record_stats
        return counter


# -------------------------------------------------------------------------- #
# checking the files against the count
# -------------------------------------------------------------------------- #

def count_data_rows(path: str) -> int:
    """Data rows in a CSV written by FleetPy, refusing anything that is not a clean file.

    Counted with `csv.reader`, so a quoted field cannot be mistaken for a row break. A repeated
    header (the file vanished and was re-created mid-run, so `isfile` chose mode `"w"` again)
    and a row whose field count differs from the header's (a torn line) are the same class of
    failure as a lost band and raise here, naming the physical line.
    """
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return 0
        n = 0
        for row in reader:
            if row == header:
                raise OutputIntegrityError(
                    f"{path}: the header repeats at line {reader.line_num}, so the file was "
                    f"re-created part-way through the run. It cannot be shown to hold the rows "
                    f"the simulation wrote (P2.20).")
            if len(row) != len(header):
                raise OutputIntegrityError(
                    f"{path}: line {reader.line_num} has {len(row)} field(s) against the "
                    f"header's {len(header)} -- a torn or blank line. The file cannot be shown "
                    f"to hold the rows the simulation wrote (P2.20).")
            n += 1
    return n


def verify_rows(output_dir: str, recorded: Dict[str, int]) -> None:
    """Raise `OutputIntegrityError` unless each file holds exactly the rows recorded for it."""
    for name, expected in sorted(recorded.items()):
        expected = int(expected)
        path = os.path.join(output_dir, name)
        if not os.path.isfile(path):
            if expected == 0:
                continue        # FleetPy creates the file on the first non-empty flush
            raise OutputIntegrityError(
                f"{path}: missing, but the simulation wrote {expected} row(s) to it (P2.20).")
        found = count_data_rows(path)
        if found != expected:
            gap = expected - found
            raise OutputIntegrityError(
                f"{path}: {found} data row(s) in the file, but the simulation wrote {expected} "
                f"-- {abs(gap)} row(s) {'missing' if gap > 0 else 'extra'}. Something other "
                f"than the simulation changed the file; in P2.20 that was a cloud-sync client "
                f"silently dropping a band of appended rows. No KPI may be computed from it.")


def write_rows_record(output_dir: str, recorded: Dict[str, int]) -> str:
    """Write `rl_output_rows.json`. Only `check_episode_output` should call this on real output,
    after the check has passed; tests use it to build fixture directories."""
    path = os.path.join(output_dir, ROWS_RECORD_FILE)
    with open(path, "w") as fh:
        json.dump({"recorded_rows": {k: int(v) for k, v in recorded.items()}}, fh,
                  indent=2, sort_keys=True)
    return path


def check_episode_output(output_dir: str, counter: RowCounter) -> str:
    """The post-run check: the files against the count, then the record.

    The record is written **only after the check passes**, so a directory with a record has
    passed it once and a directory without one never did.

    :return: the record's path
    """
    verify_rows(output_dir, counter.rows)
    return write_rows_record(output_dir, counter.rows)


def read_rows_record(output_dir: str) -> Dict[str, int]:
    """The recorded counts, raising `OutputIntegrityError` if the directory has no record."""
    path = os.path.join(output_dir, ROWS_RECORD_FILE)
    if not os.path.isfile(path):
        raise OutputIntegrityError(
            f"no {ROWS_RECORD_FILE} in {output_dir}: this run's output was never checked -- an "
            f"episode abandoned before it finished (SubprocVecEnv's auto-reset leaves one per "
            f"worker), a run made before P2.20, or one that stopped early. Its CSVs cannot be "
            f"shown to be whole, so no KPI is computed from them.")
    with open(path) as fh:
        return {str(k): int(v) for k, v in json.load(fh)["recorded_rows"].items()}


def verify_output_dir(output_dir: str) -> Dict[str, int]:
    """Re-count a finished run's files against its record; what `kpis.kpi_row` calls first.

    :return: the recorded counts
    """
    recorded = read_rows_record(output_dir)
    verify_rows(output_dir, recorded)
    return recorded
