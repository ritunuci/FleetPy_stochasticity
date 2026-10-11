"""KPIs from a simulation output directory, identically for both policies (P2.14).

The RL result is meaningless without the stock-greedy baseline on the same held-out days, and a
comparison is only as good as the sameness of the two measurements. So this module reads **only
the output directory**, with one code path for both policies, and takes nothing from the env, a
tracker or a config.

Three files are read, and the choice of which matters:

- **`1_user-stats.csv`** -- every per-rider figure. It is the file both byte-for-byte gates
  compare, so for a greedy run the numbers here are pinned.
- **`2-0_op-stats.csv`** -- fleet VKT only, as `Σ driven_distance / 1000`, which is exactly how
  `evaluation/standard.py:477` computes `op_total_km` (`G_VR_LEG_DISTANCE = "driven_distance"`).
- **`rl_episode_summary.json`** -- *only* the split between operator rejections and no-candidate
  requests. Both outcomes leave `offers = "0:"`, so the user-stats file identifies the group but
  cannot divide it. Absent (a stock-greedy run) the split is `0` rejections and the whole group
  as no-candidate requests, which is correct by construction: stock `PoolingIRSOnly` takes the
  `argmin` and never rejects.

Two files in the same directory are **deliberately not read**:

- **`3-0_op-dyn_atts.csv`** carries `G_FCTRL_CT_RQU` and `G_FCTRL_CT_DFS`, which are
  `perf_counter` durations. They differ between any two runs of the same scenario, and P1.3's
  build/commit split deliberately recomposed the first of them so that policy-inference time
  stays out of fleet-control CPU time. Measured in P2.14: those two columns are the *only*
  difference between a stock and an env-driven run of the reference day, and no KPI may depend
  on them.
- **`standard_eval.csv`** exists for stock greedy (`evaluate = 1`) and not for an env run (the
  env forces `evaluate = 0`), so reading it would silently give the two policies different
  treatment -- the one thing this module exists to prevent.

**The terminal-outcome counts come from `reward.classify_exit`, not a second ladder.** §6.4's
order is subtle -- `no_show` after `diffusion_cancelled` and gated on `pu_time is None`, no
fallback to rider decline -- and a reimplementation here would drift from the one the reward and
the rolling statistics share (trap 13). `_RowRequest` adapts a CSV row to the handful of
attributes that function reads, so the ladder itself is reused rather than copied.
"""

import json
import math
import os
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from src.rl_gym.reward import classify_exit

#: the three files read, and nothing else in the directory
USER_STATS_FILE = "1_user-stats.csv"
OP_STATS_FILE = "2-0_op-stats.csv"
EPISODE_SUMMARY_FILE = "rl_episode_summary.json"

#: `offers` for a request the operator made no offer to -- an operator rejection or an empty
#: candidate list. `_create_rejection` produces a `Rejection`, which serialises to this.
NO_OFFER = "0:"

#: the column `evaluation/standard.py` sums for `op_total_km` (`G_VR_LEG_DISTANCE`)
VKT_COLUMN = "driven_distance"

#: numeric KPIs, in report order. `aggregate` averages exactly these over seeds.
METRICS = (
    "served",
    "mean_wait_s",
    "mean_detour_ratio",
    "decline_rate",
    "cancellation_rate",
    "no_show_rate",
    "operator_rejections",
    "no_candidate_requests",
    "fleet_vkt_km",
)

#: counts reported beside the metrics, as context for the rates above
COUNTS = ("requests", "offers_made", "accepted_offers", "declines", "cancellations",
          "no_shows", "unserved_at_horizon", "unclassified")


class _RowRequest:
    """Adapts one `1_user-stats.csv` row to the attributes `classify_exit` reads.

    The point is to reuse §6.4's ladder rather than reimplement it (trap 13). Only the six
    attributes that function touches are provided; anything else it might grow would raise
    through `getattr`'s default rather than silently classify wrong.
    """

    __slots__ = ("do_time", "pu_time", "diffusion_cancelled", "rider_declined",
                 "no_show", "chosen_operator_id")

    def __init__(self, row: Any):
        self.do_time = _or_none(row.dropoff_time)
        self.pu_time = _or_none(row.pickup_time)
        self.diffusion_cancelled = bool(row.diffusion_cancelled)
        self.rider_declined = bool(row.rider_declined)
        # the demand-file flag, which is what `rq.no_show` is at runtime too -- a rider
        # attribute, not an outcome, which is why §6.4 gates it on `pu_time is None`
        self.no_show = bool(row.no_show_stat)
        self.chosen_operator_id = _or_none(row.chosen_operator_id)


def _or_none(value: Any) -> Optional[Any]:
    """`None` for a missing cell, so a NaN cannot read as "set" in the ladder."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _as_bool(series: pd.Series) -> pd.Series:
    """A boolean column, whether pandas inferred `bool`, object or strings.

    `1_user-stats.csv` gives `bool` dtype, but a hand-written fixture can give `"True"` strings
    and `== True` would then be silently all-False.
    """
    if series.dtype == bool:
        return series
    return series.map(lambda v: str(v).strip().lower() == "true")


def read_user_stats(output_dir: str) -> pd.DataFrame:
    """`1_user-stats.csv` with the boolean columns normalised."""
    path = os.path.join(output_dir, USER_STATS_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no {USER_STATS_FILE} in {output_dir}")
    df = pd.read_csv(path)
    for column in ("rider_declined", "diffusion_cancelled", "no_show_stat"):
        df[column] = _as_bool(df[column])
    return df


def read_episode_summary(output_dir: str) -> Optional[Dict[str, Any]]:
    """`rl_episode_summary.json`, or `None` for a run that did not write one.

    `None` means a stock-greedy run. It is never a reason to fail: the file is an env artifact
    and the whole design of this module is that a stock run needs no env.
    """
    path = os.path.join(output_dir, EPISODE_SUMMARY_FILE)
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def fleet_vkt_km(output_dir: str) -> float:
    """Total fleet VKT in km: `Σ driven_distance / 1000` over `2-0_op-stats.csv`.

    The same expression as `op_total_km` in `evaluation/standard.py:477`, unfiltered, which is
    right here because the reference scenario has one operator. Reimplemented rather than read
    out of `standard_eval.csv`, which an env run does not write.
    """
    path = os.path.join(output_dir, OP_STATS_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no {OP_STATS_FILE} in {output_dir}")
    return float(pd.read_csv(path)[VKT_COLUMN].sum()) / 1000.0


def classify_user_stats(df: pd.DataFrame) -> Dict[str, int]:
    """§6.4's outcome counts for every row, through `reward.classify_exit`.

    **The no-offer group is handed to the ladder as `no_candidates`**, which is branch 1 and so
    claims those rows before any later branch can. That is faithful: branches 1 and 2 are the
    only ones reachable by a request that got no offer, and §6.4 is first-match-wins, so a
    no-offer row would classify there however it were labelled. The *split* of that group
    between the two branches is not in the file at all -- `kpi_row` takes it from the summary
    JSON -- so this function reports the group under `no_candidates` and leaves the division to
    its caller.
    """
    no_offer_rids = {
        int(r.request_id) for r in df.itertuples()
        if str(r.offers).strip() == NO_OFFER
    }
    counts: Dict[str, int] = {}
    for row in df.itertuples():
        outcome = classify_exit(int(row.request_id), _RowRequest(row),
                                no_offer_rids, frozenset())
        counts[outcome] = counts.get(outcome, 0) + 1
    counts["_no_offer"] = len(no_offer_rids)
    return counts


def kpi_row(output_dir: str, policy: Optional[str] = None, day: Optional[str] = None,
            seed: Optional[int] = None) -> Dict[str, Any]:
    """Every KPI of §8 P2.14 for one run, from its output directory alone.

    :param policy: a label carried through to the table; it selects nothing. Which files exist
        is what distinguishes the two policies, not what the caller says they are.
    """
    df = read_user_stats(output_dir)
    summary = read_episode_summary(output_dir)
    counts = classify_user_stats(df)

    served_mask = df["dropoff_time"].notna()
    served = int(served_mask.sum())
    served_rows = df[served_mask]

    offers_made = int((df["offers"].astype(str).str.strip() != NO_OFFER).sum())
    accepted_offers = int(df["chosen_operator_id"].notna().sum())
    declines = int(counts.get("rider_declined", 0))
    cancellations = int(counts.get("diffusion_cancelled", 0))
    no_shows = int(counts.get("no_show", 0))

    no_offer = int(counts.get("_no_offer", 0))
    rejections, no_candidates = _split_no_offer(no_offer, summary, output_dir)

    row: Dict[str, Any] = {
        "policy": policy,
        "day": day if day is not None else (summary or {}).get("day"),
        "seed": seed if seed is not None else (summary or {}).get("random_seed"),
        "output_dir": output_dir,
        "requests": int(len(df)),
        "offers_made": offers_made,
        "accepted_offers": accepted_offers,
        "served": served,
        "mean_wait_s": _mean(served_rows["pickup_time"] - served_rows["rq_time"]),
        "mean_detour_ratio": _mean(
            (served_rows["dropoff_time"] - served_rows["pickup_time"])
            / served_rows["direct_route_travel_time"]),
        "declines": declines,
        "decline_rate": _ratio(declines, offers_made),
        "cancellations": cancellations,
        "cancellation_rate": _ratio(cancellations, accepted_offers),
        "no_shows": no_shows,
        "no_show_rate": _ratio(no_shows, accepted_offers),
        "operator_rejections": rejections,
        "no_candidate_requests": no_candidates,
        # both must be 0 on a greedy run; `unclassified` non-zero means a §6.4 path was missed
        "unserved_at_horizon": int(counts.get("unserved_at_horizon", 0)),
        "unclassified": int(counts.get("unclassified", 0)),
        "fleet_vkt_km": fleet_vkt_km(output_dir),
    }
    return row


def _split_no_offer(no_offer: int, summary: Optional[Dict[str, Any]],
                    output_dir: str) -> tuple:
    """Divide the no-offer group into operator rejections and no-candidate requests.

    With no summary this is a stock-greedy run: `min(list_tuples, key=...)` cannot reject, so
    every no-offer request had an empty candidate list.

    With a summary, the two counts come from §6.4's own classification of the episode -- and are
    **cross-checked against the file**: if they do not sum to the no-offer rows, the JSON and
    the CSVs describe different runs, which would otherwise show up as a plausible KPI table
    built from mismatched halves.
    """
    if summary is None:
        return 0, no_offer
    rejections = int(summary.get("operator_rejected", 0))
    no_candidates = int(summary.get("no_candidates", 0))
    if rejections + no_candidates != no_offer:
        raise ValueError(
            f"{EPISODE_SUMMARY_FILE} and {USER_STATS_FILE} disagree in {output_dir}: the "
            f"summary reports {rejections} operator rejection(s) and {no_candidates} "
            f"no-candidate request(s), summing to {rejections + no_candidates}, but "
            f"{no_offer} row(s) carry offers == {NO_OFFER!r}. Both outcomes leave that value, "
            f"so the two counts must sum to it; a mismatch means the summary and the CSVs are "
            f"from different runs."
        )
    return rejections, no_candidates


def _mean(series: pd.Series) -> Optional[float]:
    """Mean, or `None` for an empty selection -- never a NaN that would propagate silently."""
    if len(series) == 0:
        return None
    return float(series.mean())


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    """`numerator / denominator`, or `None` when the denominator is 0."""
    if not denominator:
        return None
    return float(numerator) / float(denominator)


# -------------------------------------------------------------------------- #
# aggregation
# -------------------------------------------------------------------------- #

def month_of(day: Optional[str]) -> Optional[str]:
    """`"2024-06-20"` -> `"2024-06"`. `None` for an unparseable stem, never a raise."""
    if not day or len(str(day)) < 7:
        return None
    return str(day)[:7]


def mean_se(values: Sequence[Optional[float]]) -> tuple:
    """`(mean, standard error, n)` over the non-`None` values.

    The standard error is `sd / sqrt(n)` with `ddof = 1`, so **a single seed gives `None`**
    rather than 0.0. One observation carries no information about spread, and printing `± 0.00`
    there would read as a precise measurement.
    """
    present = [float(v) for v in values if v is not None]
    n = len(present)
    if n == 0:
        return None, None, 0
    if n == 1:
        return present[0], None, 1
    return statistics.mean(present), statistics.stdev(present) / math.sqrt(n), n


def aggregate(rows: Iterable[Dict[str, Any]], by: str = "overall") -> List[Dict[str, Any]]:
    """Group `rows` and reduce each `METRICS` entry to `(mean, se, n)` over seeds.

    :param by: `"day"`, `"month"` or `"overall"` -- §8's three levels of the table.
    """
    rows = list(rows)
    if by == "day":
        def key(r): return r.get("day")
    elif by == "month":
        def key(r): return month_of(r.get("day"))
    elif by == "overall":
        def key(r): return "overall"
    else:
        raise ValueError(f"by must be 'day', 'month' or 'overall', got {by!r}")

    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)

    out = []
    for group in sorted(groups, key=lambda g: ("" if g is None else str(g))):
        members = groups[group]
        entry: Dict[str, Any] = {
            "group": group,
            "n_runs": len(members),
            "n_seeds": len({m.get("seed") for m in members}),
        }
        for metric in METRICS:
            entry[metric] = mean_se([m.get(metric) for m in members])
        for count in COUNTS:
            entry[count] = sum(int(m.get(count) or 0) for m in members)
        out.append(entry)
    return out


# -------------------------------------------------------------------------- #
# rendering
# -------------------------------------------------------------------------- #

#: display precision per metric, so a rate and a distance are not printed alike
_FMT = {
    "served": "{:.1f}",
    "mean_wait_s": "{:.2f}",
    "mean_detour_ratio": "{:.4f}",
    "decline_rate": "{:.4f}",
    "cancellation_rate": "{:.4f}",
    "no_show_rate": "{:.4f}",
    "operator_rejections": "{:.2f}",
    "no_candidate_requests": "{:.2f}",
    "fleet_vkt_km": "{:.2f}",
}


def format_mean_se(value: tuple, metric: str) -> str:
    """`mean ± se`, `mean` alone for a single seed, `n/a` for nothing measured."""
    mean, se, n = value
    if n == 0 or mean is None:
        return "n/a"
    fmt = _FMT.get(metric, "{:.4f}")
    if se is None:
        return fmt.format(mean)
    return f"{fmt.format(mean)} ± {fmt.format(se)}"


def format_table(rows: Iterable[Dict[str, Any]], by: str = "overall",
                 title: Optional[str] = None) -> str:
    """One markdown table of `aggregate(rows, by)`, metrics as columns."""
    entries = aggregate(rows, by)
    header = ["group", "n"] + list(METRICS)
    lines = []
    if title:
        lines.append(f"**{title}**")
        lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for entry in entries:
        cells = [str(entry["group"]), str(entry["n_runs"])]
        cells += [format_mean_se(entry[m], m) for m in METRICS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_per_run_table(rows: Iterable[Dict[str, Any]]) -> str:
    """Every run on its own row -- the unaggregated table the comparison rests on."""
    cols = ["policy", "day", "seed"] + list(METRICS) + ["unclassified"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in sorted(list(rows), key=lambda r: (str(r.get("policy")), str(r.get("day")),
                                                 int(r.get("seed") or 0))):
        cells = []
        for col in cols:
            value = row.get(col)
            if col in _FMT and value is not None:
                cells.append(_FMT[col].format(value))
            else:
                cells.append("n/a" if value is None else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
