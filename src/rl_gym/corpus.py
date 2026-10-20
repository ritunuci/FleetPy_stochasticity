"""The frozen demand corpus: one definition of a training day, and one of its hash.

Everything that needs to know **what the corpus is** reads it from here: D10's day rotation
(`day_rotation.py`), P2.6's offline builder (`offline/build_arrival_baseline.py`), and P2.9's
construction-time check in the env. Three separate implementations of "sorted `*.csv` stems"
and two of `corpus_hash` is exactly how a frozen artifact quietly stops matching the thing it
was computed over.

This module lives under `src/rl_gym/` rather than `offline/` for that reason: `offline/` may
import from here, never the reverse (`offline/__init__.py`), so the runtime check and the
builder cannot drift apart.

**The validation / test split of the held-out days lives here too** (P2.17), for the same
reason: `split_of_day` is the one definition of which half a day belongs to, resolved from the
directory its demand file sits in, and `train_sdpdp.py`, `evaluate.py` and `phase2_run.py` all
go through it rather than carrying day lists of their own.

**Days are listed by `*.csv` only** (§2.9). macOS creates and rewrites `.DS_Store` in these
directories on its own and it is untracked, so it can appear, change or vanish between runs with
no demand file changing. A reader that enumerated directory entries would see a phantom day, and
in the manifest it would put a volatile hash into a frozen artifact and fail every later check
for no real reason.

**`corpus_hash` covers names and contents**, as one `<filename>:<sha256>` line per file in
filename order, joined with `\\n`, hashed as UTF-8. So an edited file is caught as well as a
missing or added one, and the per-file list lets the check say *which* files differ (trap 16).

**`baseline_hash` covers the arrival baseline's parsed rows**, not its bytes, so a line-ending
difference on another machine cannot fail it. It lives here for the same reason `corpus_hash`
does: P2.11 verifies it at env construction, and a second implementation in the runtime path is
exactly the drift the manifest exists to prevent.

**The baseline's first column is the weekday name since P2.18**, not the month number, so
`BASELINE_COLUMNS` and the two hash helpers carry a string there. `verify_baseline` then refuses
every checkpoint-era baseline automatically, since the rows no longer hash to what any earlier
manifest recorded; what refuses an old *checkpoint* against the new table is the
`observation_contract` guard in `train_sdpdp.py`, because the two artifacts agreeing with each
other says nothing about which pair a policy was trained against.
"""

import csv
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEMAND_ROOT = os.path.join(_REPO_ROOT, "data", "demand", "example_demand", "matched",
                            "example_network")

#: the frozen training split (§2.9). 101 days after P2.2.
DEFAULT_CORPUS_DIR = os.path.join(_DEMAND_ROOT, "corpus")

#: P2.17 split the 26 held-out days: 12 validation, which every choice is made on, and 14 test,
#: which only the final results read. Separate directories, so a run's day source names the half
#: it is drawing from and the two cannot be confused by a flag.
TRAIN_DAY_DIR = "corpus"
VAL_DAY_DIR = "corpus_val"
TEST_DAY_DIR = "corpus_eval"

#: `split_of_day`'s two labels, and the directory each lives in. **The single definition of the
#: map**, so nothing derives a directory from a half by string surgery (trap 13).
HELD_OUT_DIRS = {"val": VAL_DAY_DIR, "test": TEST_DAY_DIR}

DEFAULT_VAL_DIR = os.path.join(_DEMAND_ROOT, VAL_DAY_DIR)
DEFAULT_TEST_DIR = os.path.join(_DEMAND_ROOT, TEST_DAY_DIR)

#: P2.17's committed record of the split decision: 26 rows of `day,split`. The *directories* are
#: the runtime truth about where a day's demand file is; this file is the frozen record of which
#: half each day was assigned to, and `check_split_file` binds the two so neither can drift.
#: `docs/RL_GYM_EVAL_SEEDS.csv` is deliberately **not** touched -- P2.14 to P2.16 all cite it as
#: the frozen seed record, and it stays byte-identical at md5
#: `559b836e7c61fb7dadd2e7bd9bc8891d`.
DEFAULT_SPLIT_FILE = os.path.join(_REPO_ROOT, "docs", "RL_GYM_HELDOUT_SPLIT.csv")
SPLIT_COLUMNS = ("day", "split")

#: P2.6's artifacts, beside each other
DEFAULT_MANIFEST = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "data", "rl_gym", "example_network", "corpus_manifest.json")

DEFAULT_BASELINE = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "data", "rl_gym", "example_network", "arrival_baseline.csv")

#: `arrival_baseline.csv`'s header, in order. The first column is the **weekday name** since
#: P2.18 ("Tue" / "Wed" / "Thu"); it was the month number through P2.17.
BASELINE_COLUMNS = ["weekday", "t_seconds", "window_s", "mean", "sd"]
#: float format used when writing and when hashing alike, so parse -> format is idempotent
FLOAT_FMT = "{:.12g}"


def training_day_files(corpus_dir: str = DEFAULT_CORPUS_DIR) -> List[str]:
    """Sorted filename stems of the demand days in `corpus_dir` -- `*.csv` only (§2.9).

    **The single definition of "the training days".** `day_rotation.training_days` and the
    manifest builder both call this, so D10's rotation can never disagree with what the
    manifest hashed.
    """
    if not os.path.isdir(corpus_dir):
        raise ValueError(f"not a directory: {corpus_dir}")
    days = sorted(f[:-4] for f in os.listdir(corpus_dir) if f.endswith(".csv"))
    if not days:
        raise ValueError(f"no *.csv demand days in {corpus_dir}")
    return days


# -------------------------------------------------------------------------- #
# the validation / test split of the held-out days (P2.17)
# -------------------------------------------------------------------------- #

def validation_days(val_dir: str = DEFAULT_VAL_DIR) -> List[str]:
    """The 12 validation days (§2.9, P2.17). Every choice is made on these."""
    return training_day_files(val_dir)


def test_days(test_dir: str = DEFAULT_TEST_DIR) -> List[str]:
    """The 14 test days (§2.9, P2.17). Read only by the final evaluation."""
    return training_day_files(test_dir)


def held_out_days(val_dir: str = DEFAULT_VAL_DIR,
                  test_dir: str = DEFAULT_TEST_DIR) -> List[str]:
    """All 26 held-out days, sorted -- the domain `docs/RL_GYM_EVAL_SEEDS.csv` covers.

    **Sorted over the union, which is why P2.17 did not regenerate the seed file.**
    `evaluate.derive_eval_seeds` consumes one bulk draw row-major over `days` in sorted order,
    so a union of the two halves sorts to exactly the 26-day list the seeds were drawn against
    and reproduces all 260 rows (verified in P2.17). Splitting the directory did not move a
    single seed.
    """
    return sorted(validation_days(val_dir) + test_days(test_dir))


def split_of_day(day: str, val_dir: str = DEFAULT_VAL_DIR,
                 test_dir: str = DEFAULT_TEST_DIR) -> str:
    """`"val"` or `"test"` for a held-out day, resolved from the filesystem (P2.17).

    **The directory a day's file sits in is what decides its half**, never a caller's flag and
    never a parallel list -- which is what makes a final-evaluation command unable to reach a
    validation day even if it is named explicitly (G2/G3).

    Raises when the day is in **neither** half (it is a training day, an excluded day, or a
    typo) and when it is in **both**, which can only mean a day was copied instead of moved and
    is the one failure that would let a day be trained on and reported on at once.
    """
    day = str(day)
    in_val = os.path.isfile(os.path.join(val_dir, f"{day}.csv"))
    in_test = os.path.isfile(os.path.join(test_dir, f"{day}.csv"))
    if in_val and in_test:
        raise ValueError(
            f"demand day {day!r} exists in **both** {os.path.basename(val_dir)} and "
            f"{os.path.basename(test_dir)}, so its half is undefined. §2.9 moves files between "
            f"the splits and never copies them: a day in two places would be selected on and "
            f"reported on at the same time, which is the whole point of holding the test days "
            f"back. Delete the wrong copy -- do not pick one here."
        )
    if in_val:
        return "val"
    if in_test:
        return "test"
    raise ValueError(
        f"demand day {day!r} is not a held-out day: it is in neither "
        f"{os.path.basename(val_dir)} ({len(validation_days(val_dir))} validation days) nor "
        f"{os.path.basename(test_dir)} ({len(test_days(test_dir))} test days). Training days "
        f"live in {TRAIN_DAY_DIR}/ and are never evaluated (§2.9)."
    )


def day_dir_for(day: str, split: Optional[str] = None, val_dir: str = DEFAULT_VAL_DIR,
                test_dir: str = DEFAULT_TEST_DIR) -> str:
    """The `day_dir` name a held-out day's demand file lives in, with the half asserted (G3).

    :param split: when given (`"val"` / `"test"`), the day **must** belong to that half or this
        raises. That is the guard: a `--final-test` run resolves every day with `split="test"`,
        so a validation day cannot be reached; a `--validation` run does the reverse. The two
        directions are one code path, so neither can be fixed while the other rots.
    """
    found = split_of_day(day, val_dir, test_dir)
    if split is not None:
        if split not in HELD_OUT_DIRS:
            raise ValueError(f"split must be one of {sorted(HELD_OUT_DIRS)}, got {split!r}")
        if found != split:
            raise ValueError(
                f"refusing to run {day!r} in {split!r} mode: it is a **{found}** day, in "
                f"{HELD_OUT_DIRS[found]}/. The 14 test days exist to be read once, by the "
                f"final evaluation; the 12 validation days are what every choice is made on "
                f"(P2.17). Mixing them makes the final figures a selection criterion. Run it "
                f"with the {found!r} mode, or drop the day."
            )
    return HELD_OUT_DIRS[found]


def read_split_file(path: str = DEFAULT_SPLIT_FILE) -> List[Tuple[str, str]]:
    """`(day, split)` rows from the committed split record (P2.17)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no held-out split record at {path} (§2.9, P2.17). It carries one `day,split` row "
            f"per held-out day and is what records the split decision independently of where "
            f"the files happen to sit."
        )
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if tuple(header) != SPLIT_COLUMNS:
            raise ValueError(f"{path} header is {header}, expected {list(SPLIT_COLUMNS)}")
        rows = [(r[0], r[1]) for r in reader if r]
    for day, split in rows:
        if split not in HELD_OUT_DIRS:
            raise ValueError(f"{path}: {day} has split {split!r}, "
                             f"expected one of {sorted(HELD_OUT_DIRS)}")
    return rows


def check_split_file(path: str = DEFAULT_SPLIT_FILE, val_dir: str = DEFAULT_VAL_DIR,
                     test_dir: str = DEFAULT_TEST_DIR) -> List[str]:
    """Every label must equal `split_of_day` for that day (G4). Returns the differences.

    This is what stops the committed record and the directories from drifting apart. Without
    it the file would be decoration: a stale label is indistinguishable from a correct one, and
    the guards that read a half would be enforcing a decision nobody had checked still held.
    Writes nothing; called from `evaluate --check-seeds`, `evaluate --verify` and a unit test.
    """
    problems: List[str] = []
    try:
        rows = read_split_file(path)
    except (FileNotFoundError, ValueError) as exc:
        return [str(exc).splitlines()[0]]

    labelled = {day: split for day, split in rows}
    if len(labelled) != len(rows):
        duplicates = sorted({d for d, _ in rows if sum(1 for x, _ in rows if x == d) > 1})
        problems.append(f"duplicate day row(s): {', '.join(duplicates)}")

    on_disk = {}
    for day in held_out_days(val_dir, test_dir):
        on_disk[day] = split_of_day(day, val_dir, test_dir)

    for day in sorted(set(labelled) - set(on_disk)):
        problems.append(f"{day}: labelled {labelled[day]!r} but is in neither half on disk")
    for day in sorted(set(on_disk) - set(labelled)):
        problems.append(f"{day}: on disk in the {on_disk[day]!r} half but carries no label")
    for day in sorted(set(labelled) & set(on_disk)):
        if labelled[day] != on_disk[day]:
            problems.append(f"{day}: labelled {labelled[day]!r} but on disk in "
                            f"{on_disk[day]!r}")
    return problems


def file_sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def corpus_file_hashes(corpus_dir: str = DEFAULT_CORPUS_DIR) -> List[Dict[str, str]]:
    """Every training file, sorted by filename, with the SHA-256 of its bytes."""
    return [{"filename": f"{day}.csv",
             "sha256": file_sha256(os.path.join(corpus_dir, f"{day}.csv"))}
            for day in training_day_files(corpus_dir)]


def corpus_hash(files: Sequence[Dict[str, str]]) -> str:
    """SHA-256 of `<filename>:<sha256>` lines, in order, joined with newlines (§8 P2.6)."""
    text = "\n".join(f"{e['filename']}:{e['sha256']}" for e in files)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_baseline_text(rows: Sequence[Tuple]) -> str:
    """The baseline's rows as canonical text, sorted by `(weekday, t_seconds, window_s)`.

    Hashed instead of the file's bytes so a line-ending or trailing-newline difference on
    another machine cannot change the hash. `FLOAT_FMT` is used both here and when writing, so
    write -> parse -> format is idempotent and the hash round-trips exactly.

    **The sort is lexical on the weekday name, and that is deliberate** (P2.18). The committed
    file is written in Tue/Wed/Thu order instead, by the builder, because a lexical file would
    read Thu/Tue/Wed and look like a defect. Sorting here by the weekday's position in the
    vocabulary would make this function depend on an ordering table, and `baseline_hash` would
    then move silently if that table were ever edited. The hash needs nothing but the rows.
    """
    lines = []
    for weekday, t, window_s, mean, sd in sorted(rows, key=lambda r: (str(r[0]), r[1], r[2])):
        lines.append(f"{weekday},{int(t)},{int(window_s)},"
                     f"{FLOAT_FMT.format(mean)},{FLOAT_FMT.format(sd)}")
    return "\n".join(lines)


def baseline_hash(rows: Sequence[Tuple]) -> str:
    """SHA-256 of `canonical_baseline_text(rows)` (§8 P2.6)."""
    return hashlib.sha256(canonical_baseline_text(rows).encode("utf-8")).hexdigest()


def read_baseline(path: str = DEFAULT_BASELINE) -> List[Tuple]:
    """The committed baseline as parsed rows. A plain table: no comments, no metadata."""
    if not os.path.isfile(path):
        raise ValueError(
            f"no arrival baseline at {path}; build it with "
            f"`python -m src.rl_gym.offline.build_arrival_baseline`"
        )
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != BASELINE_COLUMNS:
            raise ValueError(f"{path}: expected header {BASELINE_COLUMNS}, "
                             f"got {reader.fieldnames}")
        return [(str(r["weekday"]), int(r["t_seconds"]), int(r["window_s"]),
                 float(r["mean"]), float(r["sd"])) for r in reader]


def load_manifest(path: str = DEFAULT_MANIFEST) -> Dict[str, Any]:
    """P2.6's manifest. Raises with the build command if it is missing."""
    if not os.path.isfile(path):
        raise ValueError(
            f"no corpus manifest at {path}; build it with "
            f"`python -m src.rl_gym.offline.build_arrival_baseline`"
        )
    with open(path) as fh:
        manifest = json.load(fh)
    # `month_vocab` was required here through P2.18, for the request block's `month` one-hot.
    # P2.19 removed the one-hot and the key; the builder's `--check` is what refuses a manifest
    # that still carries it, since nothing online reads it any more.
    for key in ("files", "corpus_hash", "weekday_vocab"):
        if key not in manifest:
            raise ValueError(f"{path} has no {key!r}; rebuild it")
    return manifest


def verify_corpus(manifest: Dict[str, Any],
                  corpus_dir: str = DEFAULT_CORPUS_DIR) -> None:
    """Raise unless `corpus_dir` still matches the manifest, naming what differs (trap 16).

    P2.6's baseline, its weekday means and its SD are all computed over one particular set of
    days. A baseline read against a different corpus produces plausible-looking anomalies that
    are simply wrong, and nothing else would notice -- so the env refuses to start (P2.9).

    The message lists the files that are **missing, extra or changed** rather than only
    reporting a hash mismatch, because a hash tells you nothing about what to fix.
    """
    current = corpus_file_hashes(corpus_dir)
    if corpus_hash(current) == manifest["corpus_hash"]:
        return

    expected = {e["filename"]: e["sha256"] for e in manifest["files"]}
    found = {e["filename"]: e["sha256"] for e in current}
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    changed = sorted(n for n in set(expected) & set(found) if expected[n] != found[n])

    detail = []
    if missing:
        detail.append(f"missing {len(missing)}: {', '.join(missing[:8])}"
                      + (" …" if len(missing) > 8 else ""))
    if extra:
        detail.append(f"extra {len(extra)}: {', '.join(extra[:8])}"
                      + (" …" if len(extra) > 8 else ""))
    if changed:
        detail.append(f"changed {len(changed)}: {', '.join(changed[:8])}"
                      + (" …" if len(changed) > 8 else ""))
    if not detail:
        # same names, same contents, different hash: the hash definition itself moved
        detail.append("the file list and contents match but the hash does not -- the "
                      "corpus_hash definition has changed since the manifest was built")

    raise ValueError(
        f"the demand corpus no longer matches {os.path.basename(DEFAULT_MANIFEST)}: "
        + "; ".join(detail)
        + f". The arrival baseline was computed over the manifest's "
          f"{len(expected)} days, so anything trained or evaluated against it is invalid "
          f"(§2.9 freezes the corpus; trap 16). Restore the corpus, or rebuild both artifacts "
          f"with `python -m src.rl_gym.offline.build_arrival_baseline` and retrain."
    )


def verify_baseline(manifest: Dict[str, Any], rows: Sequence[Tuple]) -> None:
    """Raise unless the baseline's parsed rows still hash to the manifest's `baseline_hash`.

    The companion to `verify_corpus`, and the reason both exist is the same (trap 16): the
    manifest ties the two artifacts together, so a baseline edited or rebuilt without its
    manifest — or a manifest restored over a newer baseline — must stop the env rather than
    feed the policy expectations the corpus never produced. P2.11 is the first item that reads
    the baseline at runtime, so it is the first that can check this.

    Compares parsed content, never bytes, so line endings cannot fail it.
    """
    expected = manifest.get("baseline_hash")
    if expected is None:
        raise ValueError(
            f"{os.path.basename(DEFAULT_MANIFEST)} has no 'baseline_hash'; rebuild both "
            f"artifacts with `python -m src.rl_gym.offline.build_arrival_baseline`"
        )
    found = baseline_hash(rows)
    if found == expected:
        return
    n_expected = manifest.get("baseline_rows")
    detail = f"{len(rows)} rows found"
    if n_expected is not None and int(n_expected) != len(rows):
        detail += f", {int(n_expected)} expected"
    raise ValueError(
        f"the arrival baseline does not match {os.path.basename(DEFAULT_MANIFEST)}: "
        f"baseline_hash {found[:12]}… against the manifest's {str(expected)[:12]}… "
        f"({detail}). The manifest records the baseline it was built beside, so one of the two "
        f"has been edited or rebuilt alone and the anomaly features would be wrong in a way "
        f"nothing else would notice (trap 16). Rebuild both with "
        f"`python -m src.rl_gym.offline.build_arrival_baseline`."
    )
