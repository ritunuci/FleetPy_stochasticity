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
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Sequence

#: the frozen training split (§2.9). 101 days after P2.2.
DEFAULT_CORPUS_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "data", "demand", "example_demand", "matched", "example_network", "corpus")

#: P2.6's artifacts, beside each other
DEFAULT_MANIFEST = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "data", "rl_gym", "example_network", "corpus_manifest.json")


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
