"""The frozen demand corpus: one definition of a training day, and one of its hash.

Everything that needs to know **what the corpus is** reads it from here: D10's day rotation
(`day_rotation.py`), P2.6's offline builder (`offline/build_arrival_baseline.py`), and P2.9's
construction-time check in the env. Three separate implementations of "sorted `*.csv` stems"
and two of `corpus_hash` is exactly how a frozen artifact quietly stops matching the thing it
was computed over.

This module lives under `src/rl_gym/` rather than `offline/` for that reason: `offline/` may
import from here, never the reverse (`offline/__init__.py`), so the runtime check and the
builder cannot drift apart.

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
"""

import csv
import hashlib
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

#: the frozen training split (§2.9). 101 days after P2.2.
DEFAULT_CORPUS_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "data", "demand", "example_demand", "matched", "example_network", "corpus")

#: P2.6's artifacts, beside each other
DEFAULT_MANIFEST = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "data", "rl_gym", "example_network", "corpus_manifest.json")

DEFAULT_BASELINE = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "data", "rl_gym", "example_network", "arrival_baseline.csv")

#: `arrival_baseline.csv`'s header, in order
BASELINE_COLUMNS = ["month", "t_seconds", "window_s", "mean", "sd"]
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
    """The baseline's rows as canonical text, sorted by `(month, t_seconds, window_s)`.

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
        return [(int(r["month"]), int(r["t_seconds"]), int(r["window_s"]),
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
    for key in ("files", "corpus_hash", "month_vocab", "weekday_vocab"):
        if key not in manifest:
            raise ValueError(f"{path} has no {key!r}; rebuild it")
    return manifest


def verify_corpus(manifest: Dict[str, Any],
                  corpus_dir: str = DEFAULT_CORPUS_DIR) -> None:
    """Raise unless `corpus_dir` still matches the manifest, naming what differs (trap 16).

    P2.6's baseline, its month means and its SD are all computed over one particular set of
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
