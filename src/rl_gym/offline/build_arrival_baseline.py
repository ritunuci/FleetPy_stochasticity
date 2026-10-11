"""Build the month-conditioned arrival baseline and the corpus manifest (P2.6).

Two artifacts, written together or not at all:

- `arrival_baseline.csv` -- `month, t_seconds, window_s, mean, sd`, the expected arrival count
  and its spread for every (month, minute-of-day, window) the observation needs. A **plain
  table**: no comment lines, no metadata rows, so `pd.read_csv` on it needs no arguments.
- `corpus_manifest.json` -- the training files with their hashes, `corpus_hash`,
  `baseline_hash`, the month and weekday vocabularies, and the window and floor constants.

**Inputs are training days only** (`corpus/`), never evaluation or excluded days.

`c_d(t, W)` counts requests **as `load_demand_file` keeps them** -- both of its filters,
`rq_time` in `[start_time, end_time)` *and* `start` node != `end` node (§2.9). Counting raw file
rows instead would count requests the online side cannot see: P2.7's arrival hook is
`build_assignment_context`, which a same-node row never reaches. Five training days carry one
such row (2024-02-15, 2024-02-20, 2024-03-05, 2024-05-30, 2024-06-13), and counting them
inflates 150 (month, minute) buckets at W = 30 and 300 at W = 60 by up to 0.11 arrivals -- small,
one-sided, and free to avoid.

    python -m src.rl_gym.offline.build_arrival_baseline            # write both artifacts
    python -m src.rl_gym.offline.build_arrival_baseline --check    # verify the committed pair

`--check` compares **parsed content**, not file bytes, so a line-ending difference on another
machine cannot fail it -- the same rule as P2.5's table and for the same reason (§2.9's corpus
twins).

**The manifest is what ties the two artifacts together** (trap 16). `corpus_hash` covers the
training files' names and contents; `baseline_hash` covers the baseline's parsed rows. A baseline
is only meaningful for the corpus it was computed over, so P2.9's construction-time check has one
place to look and the two files cannot drift apart unnoticed.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import statistics
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NETWORK = "example_network"
CORPUS = os.path.join(ROOT, "data", "demand", "example_demand", "matched", NETWORK, "corpus")
OUT_DIR = os.path.join(ROOT, "data", "rl_gym", NETWORK)
BASELINE = os.path.join(OUT_DIR, "arrival_baseline.csv")
MANIFEST = os.path.join(OUT_DIR, "corpus_manifest.json")

#: the reference scenario's window, from `constant_config_depot_cali_sc_1.csv`
START_TIME = 25200
END_TIME = 70000
WINDOWS_S = (1800, 3600)                 # §5.2's 30 and 60 minute trailing windows
SD_FLOOR = 1.0                           # z = (observed - expected) / max(sd, SD_FLOOR)
MINUTE = 60

#: §2.9's vocabularies. An unknown weekday or month raises in P2.9, not here.
MONTH_VOCAB = list(range(1, 11))                     # Jan-Oct
WEEKDAY_VOCAB = ["Tue", "Wed", "Thu"]
_WEEKDAY_OF = {1: "Tue", 2: "Wed", 3: "Thu"}

BASELINE_COLUMNS = ["month", "t_seconds", "window_s", "mean", "sd"]
#: float format used both when writing and when hashing, so parse -> format is idempotent
FLOAT_FMT = "{:.12g}"

EXPECTED_TRAINING_DAYS = 101


def minute_grid() -> List[int]:
    """Every minute `t` in `[start_time, end_time)` -- 747 of them for the reference scenario."""
    return list(range(START_TIME, END_TIME, MINUTE))


def training_day_files(corpus_dir: str = CORPUS) -> List[str]:
    """Training day stems, sorted, `*.csv` only (§2.9)."""
    if not os.path.isdir(corpus_dir):
        raise ValueError(f"no corpus directory at {corpus_dir}")
    days = sorted(f[:-4] for f in os.listdir(corpus_dir) if f.endswith(".csv"))
    if not days:
        raise ValueError(f"no *.csv training days in {corpus_dir}")
    return days


def kept_arrival_times(path: str) -> "object":
    """`rq_time` of the requests `load_demand_file` keeps, sorted. Both filters (§2.9)."""
    import numpy as np
    import pandas as pd
    df = pd.read_csv(path)
    for col in ("rq_time", "start", "end"):
        if col not in df.columns:
            raise ValueError(f"{path} has no {col} column")
    in_window = df[(df["rq_time"] >= START_TIME) & (df["rq_time"] < END_TIME)]
    kept = in_window[in_window["start"] != in_window["end"]]
    return np.sort(kept["rq_time"].to_numpy())


def counts_for_day(times: "object", grid: Sequence[int], window_s: int) -> "object":
    """`c_d(t, W)` over the grid: arrivals with `rq_time` in `[t - W, t)`, half-open (§5.2)."""
    import numpy as np
    left = np.searchsorted(times, np.asarray(grid) - window_s, side="left")
    right = np.searchsorted(times, np.asarray(grid), side="left")
    return (right - left).astype(float)


def compute_baseline(corpus_dir: str = CORPUS) -> Tuple[List[Tuple], Dict[str, object]]:
    """The baseline rows, and the diagnostics the report and the tests want."""
    import numpy as np

    days = training_day_files(corpus_dir)
    grid = minute_grid()
    months = {d: dt.date.fromisoformat(d).month for d in days}
    weekdays = {d: _WEEKDAY_OF[dt.date.fromisoformat(d).weekday()] for d in days}

    unknown_m = sorted({m for m in months.values() if m not in MONTH_VOCAB})
    unknown_w = sorted({w for w in weekdays.values() if w not in WEEKDAY_VOCAB})
    if unknown_m or unknown_w:
        raise ValueError(f"training days outside the vocabularies: months {unknown_m}, "
                         f"weekdays {unknown_w}")

    times = {d: kept_arrival_times(os.path.join(corpus_dir, f"{d}.csv")) for d in days}
    by_month: Dict[int, List[str]] = {}
    for d in days:
        by_month.setdefault(months[d], []).append(d)

    rows: List[Tuple] = []
    diagnostics: Dict[str, object] = {"days": days, "grid": grid,
                                      "days_per_month": {m: len(v) for m, v in
                                                         sorted(by_month.items())}}
    for window_s in WINDOWS_S:
        counts = {d: counts_for_day(times[d], grid, window_s) for d in days}
        # mean over the month's training days
        month_mean = {m: np.mean([counts[d] for d in ds], axis=0)
                      for m, ds in by_month.items()}
        # sd of the month-centred residual, pooled over ALL training days -- 9-12 days per
        # month are too few to estimate spread per month. Sample sd, ddof=1.
        residuals = np.vstack([counts[d] - month_mean[months[d]] for d in days])
        sd = residuals.std(axis=0, ddof=1)
        diagnostics[f"month_mean_{window_s}"] = month_mean
        diagnostics[f"sd_{window_s}"] = sd
        for m in sorted(by_month):
            for i, t in enumerate(grid):
                rows.append((m, t, window_s, float(month_mean[m][i]), float(sd[i])))

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    bad = [r for r in rows if not (np.isfinite(r[3]) and np.isfinite(r[4]))]
    if bad:
        raise ValueError(f"{len(bad)} non-finite baseline values, first {bad[0]}")
    return rows, diagnostics


# ------------------------------------------------------------------------------------ #
# hashing -- parsed content, never bytes
# ------------------------------------------------------------------------------------ #

def canonical_baseline_text(rows: Sequence[Tuple]) -> str:
    """One line per row, sorted, floats at a fixed precision.

    Hashed instead of the file's bytes so a line-ending or trailing-newline difference on
    another machine cannot change the hash. `FLOAT_FMT` is used both here and when writing, so
    write -> parse -> format is idempotent and the hash round-trips exactly.
    """
    lines = []
    for month, t, window_s, mean, sd in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
        lines.append(f"{int(month)},{int(t)},{int(window_s)},"
                     f"{FLOAT_FMT.format(mean)},{FLOAT_FMT.format(sd)}")
    return "\n".join(lines)


def baseline_hash(rows: Sequence[Tuple]) -> str:
    return hashlib.sha256(canonical_baseline_text(rows).encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def corpus_file_hashes(corpus_dir: str = CORPUS) -> List[Dict[str, str]]:
    """Every training file, sorted by filename, with the SHA-256 of its bytes."""
    return [{"filename": f"{d}.csv",
             "sha256": file_sha256(os.path.join(corpus_dir, f"{d}.csv"))}
            for d in training_day_files(corpus_dir)]


def corpus_hash(files: Sequence[Dict[str, str]]) -> str:
    """SHA-256 of `<filename>:<sha256>` lines, in order, joined with newlines (§8 P2.6).

    Covers contents as well as names, so an edited file is caught too, and the per-file list
    lets the env say which files differ.
    """
    text = "\n".join(f"{e['filename']}:{e['sha256']}" for e in files)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------ #
# read / write
# ------------------------------------------------------------------------------------ #

def read_baseline(path: str = BASELINE) -> List[Tuple]:
    """The committed baseline as parsed rows. A plain table: no comments, no metadata."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != BASELINE_COLUMNS:
            raise ValueError(f"{path}: expected header {BASELINE_COLUMNS}, "
                             f"got {reader.fieldnames}")
        return [(int(r["month"]), int(r["t_seconds"]), int(r["window_s"]),
                 float(r["mean"]), float(r["sd"])) for r in reader]


def write_both(rows: Sequence[Tuple], corpus_dir: str, baseline_path: str,
               manifest_path: str) -> Dict[str, object]:
    """Write the baseline and the manifest together. Neither is written alone."""
    files = corpus_file_hashes(corpus_dir)
    manifest = {
        "network": NETWORK,
        "corpus_dir": os.path.relpath(corpus_dir, ROOT),
        "n_training_days": len(files),
        "files": files,
        "corpus_hash": corpus_hash(files),
        "baseline_file": os.path.basename(baseline_path),
        "baseline_hash": baseline_hash(rows),
        "baseline_rows": len(rows),
        "month_vocab": MONTH_VOCAB,
        "weekday_vocab": WEEKDAY_VOCAB,
        "start_time": START_TIME,
        "end_time": END_TIME,
        "windows_s": list(WINDOWS_S),
        "sd_floor": SD_FLOOR,
        "minute_s": MINUTE,
    }
    os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
    # baseline first, then the manifest that hashes it: if the second write fails, --check
    # fails loudly rather than a stale manifest blessing a new baseline
    with open(baseline_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(BASELINE_COLUMNS)
        for month, t, window_s, mean, sd in rows:
            writer.writerow([int(month), int(t), int(window_s),
                             FLOAT_FMT.format(mean), FLOAT_FMT.format(sd)])
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=False)
        fh.write("\n")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the arrival baseline and manifest (P2.6)")
    ap.add_argument("--check", action="store_true",
                    help="verify the committed pair instead of writing; compares parsed "
                         "content, not file bytes")
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--manifest", default=MANIFEST)
    args = ap.parse_args()

    rows, diag = compute_baseline(args.corpus)
    days = diag["days"]
    if len(days) != EXPECTED_TRAINING_DAYS:
        raise SystemExit(f"expected {EXPECTED_TRAINING_DAYS} training days, found {len(days)}; "
                         f"has the corpus changed? (§2.9 freezes it)")
    print(f"training days {len(days)} | minute grid {len(diag['grid'])} | "
          f"windows {list(WINDOWS_S)} | rows {len(rows)}")
    print(f"days per month {diag['days_per_month']}")

    if not args.check:
        manifest = write_both(rows, args.corpus, args.baseline, args.manifest)
        print(f"corpus_hash   {manifest['corpus_hash']}")
        print(f"baseline_hash {manifest['baseline_hash']}")
        print(f"wrote {args.baseline}")
        print(f"wrote {args.manifest}")
        return 0

    problems = []
    committed_rows = read_baseline(args.baseline)
    if baseline_hash(committed_rows) != baseline_hash(rows):
        problems.append("arrival_baseline.csv does not match a fresh build")
        if len(committed_rows) != len(rows):
            problems.append(f"  rows: committed {len(committed_rows)}, rebuilt {len(rows)}")
        else:
            diffs = [(a, b) for a, b in zip(sorted(committed_rows), sorted(rows)) if a != b]
            problems.append(f"  {len(diffs)} differing row(s), first {diffs[0]}"
                            if diffs else "  rows equal but hashes differ")
    with open(args.manifest) as fh:
        manifest = json.load(fh)
    files = corpus_file_hashes(args.corpus)
    if manifest.get("corpus_hash") != corpus_hash(files):
        problems.append("corpus_hash does not match the corpus directory")
        committed = {e["filename"]: e["sha256"] for e in manifest.get("files", [])}
        current = {e["filename"]: e["sha256"] for e in files}
        for name in sorted(set(committed) - set(current)):
            problems.append(f"  missing: {name}")
        for name in sorted(set(current) - set(committed)):
            problems.append(f"  extra:   {name}")
        for name in sorted(set(committed) & set(current)):
            if committed[name] != current[name]:
                problems.append(f"  changed: {name}")
    if manifest.get("baseline_hash") != baseline_hash(committed_rows):
        problems.append("baseline_hash in the manifest does not match arrival_baseline.csv")
    for key, want in (("start_time", START_TIME), ("end_time", END_TIME),
                      ("windows_s", list(WINDOWS_S)), ("sd_floor", SD_FLOOR),
                      ("month_vocab", MONTH_VOCAB), ("weekday_vocab", WEEKDAY_VOCAB)):
        if manifest.get(key) != want:
            problems.append(f"manifest {key}: {manifest.get(key)!r}, expected {want!r}")

    print(f"corpus_hash   {manifest.get('corpus_hash')}")
    print(f"baseline_hash {manifest.get('baseline_hash')}")
    if problems:
        print("MISMATCH:")
        for p in problems:
            print(f"  {p}" if not p.startswith("  ") else p)
        return 1
    print("the committed baseline and manifest match a fresh build, and agree with each other")
    return 0


if __name__ == "__main__":
    sys.exit(main())
