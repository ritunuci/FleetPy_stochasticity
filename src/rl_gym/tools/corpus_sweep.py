"""Corpus sanity sweep: the scripted greedy policy on every kept demand day (P2.4).

Every Phase 1 figure comes from one day. This runs the P1.10 greedy driver through the Gym API
on each of the 127 kept days -- `corpus/` (101 training) plus the 26 held-out days, which P2.17
split across `corpus_val/` (12) and `corpus_eval/` (14),
never `corpus_excluded/` -- in fixed-day mode with `base_seed` absent, so every day runs at the
scenario row's seed 42. It changes nothing in the env or the simulation.

    # full sweep, 4 processes; caffeinate keeps the Mac awake for the ~20-35 minutes it takes
    caffeinate -i python -m src.rl_gym.tools.corpus_sweep
    # a few days, one process, in the order given
    python -m src.rl_gym.tools.corpus_sweep --days 2024-03-06,2024-10-09,2024-06-20 --processes 1

`caffeinate` is **macOS-only**; drop it elsewhere. On a cluster (HPC3, say) run the module
directly and raise `--processes` to what the node allows -- the 4-process default is sized for a
16 GB laptop, at roughly 0.4 GB per worker.

**Per-process reuse is deliberate and is itself a check.** With `--processes 1` the days run
one after another in this process, each building its own simulation. That is the arrangement
training uses -- many episodes per worker -- so the reference day reproducing P1.10's figures
*after* other days have been simulated in the same process is evidence that nothing carries over
between simulations: global RNG state, routing-engine caches, stochastic travel-time state,
logging handlers.

**The reference-day check is automatic.** 2024-06-20 is the day every Phase 1 figure comes
from, so whenever it is in the sweep set its row is asserted against P1.10's measurements
(`REFERENCE_ROW`). This file carries its own copy of P1.10's instrumentation -- the bucket
classifier and the tail measurement -- rather than importing it from `tests/test_rl_gym.py`,
and that assertion is what keeps the two copies from drifting apart silently.

Writes `docs/RL_GYM_CORPUS_SWEEP.md`. Exits non-zero if any day fails a reconciliation, reports
`unclassified != 0`, or the reference row does not match.
"""

import argparse
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.misc.globals import G_SIM_END_TIME, G_SIM_START_TIME  # noqa: E402
from src.rl_gym.gym_env import SDPDPAssignmentEnv  # noqa: E402
from src.rl_gym.policies import greedy_action  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SCS = os.path.join(ROOT, "studies", "example_study", "scenarios")
DEMAND = os.path.join(ROOT, "data", "demand", "example_demand", "matched", "example_network")
REPORT = os.path.join(ROOT, "docs", "RL_GYM_CORPUS_SWEEP.md")

K = 8
MAX_PROCESSES = 4                 # sized for a 16 GB laptop at ~0.4 GB per worker
#: never corpus_excluded (§2.9). `corpus_val` joined in P2.17, which split the 26 held-out days
#: into 12 validation and 14 test: without it this sweep would find 115 days and the CLI's
#: `expected 127 kept days` check would refuse to run at all. The sweep covers every kept day
#: whatever half it is in -- it is a property measurement of the demand, not an evaluation.
SWEEP_DIRS = ("corpus", "corpus_val", "corpus_eval")

#: `record_remaining_assignments` runs until every vehicle finishes its route, then gives up:
#: `if c_time > self.end_time + 2 * 7200: break` (`FleetSimulationBase.py` ~625 -- written as
#: `2 * 7200`, which is why grepping for 14400 does not find it). Hitting that cap is the only
#: way §6.4's branch 7 can fire, since the loop breaks with vehicles still en route. How close
#: the longest tail comes to it is therefore the branch-7 indicator -- not the horizon margin.
TAIL_CAP_SECONDS = 2 * 7200       # 14_400

#: P1.10's measurements for the reference day. Asserted whenever 2024-06-20 is swept.
REFERENCE_DAY = "2024-06-20"
REFERENCE_ROW = {
    "requests_loaded": 445,
    "steps": 436,
    "empty_candidates": 9,
    "reservation": 0,
    "same_od": 0,
    "duplicate_rid": 0,
    "cand_max": 11,
    "cand_over_k": 13,
    "pickups": 288,
    "declined": 118,
    "cancelled": 28,
    "no_show": 2,
    "unclassified": 0,
    "duplicate_boardings": 1,
}
REFERENCE_MARGIN = 194.74         # horizon_margin, seconds
MARGIN_TOL = 0.1


def kept_days() -> List[Tuple[str, str]]:
    """The 127 `(day_dir, day)` pairs to sweep, `*.csv` only (§2.9), sorted within each dir."""
    pairs = []
    for sub in SWEEP_DIRS:
        path = os.path.join(DEMAND, sub)
        if not os.path.isdir(path):
            raise SystemExit(f"missing demand directory {path}; has P2.2 been applied?")
        pairs += [(sub, f[:-4]) for f in sorted(os.listdir(path)) if f.endswith(".csv")]
    return pairs


def p95(values: List[float]) -> Optional[float]:
    """P1.10's p95, by the same index rule, so the figures stay comparable."""
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(0.95 * len(s)))]


def _file_request_counts(path: str, start_time: float,
                         end_time: float) -> Tuple[int, int, int, int]:
    """`(rows, rows in the time window, same-origin-destination rows dropped, rows loaded)`.

    `load_demand_file` applies **two** filters, not one (`demand.py` ~103-105):

        future_requests = tmp_df[(tmp_df[G_RQ_TIME] >= start_time) & (… < end_time)]
        future_requests = future_requests[future_requests[G_RQ_ORIGIN] != …[G_RQ_DESTINATION]]

    The second one matters and is easy to miss. A request whose `start` node equals its `end`
    node is removed **before** the simulation begins: it never enters `future_requests`, never
    reaches `build_assignment_context`, and never calls `record_user`. So it is correctly absent
    from every reconciliation bucket, and "requests loaded" must exclude it.

    Modelling only the time filter made 7 of the 127 days report a one-request shortfall --
    exactly the 7 days carrying one such row. Worth noting for §1 and P1.10: the
    `o_pos == d_pos` branch inside `user_request` is therefore **not** reached by same-node rows
    in the demand file, and P1.10's `same_origin_destination` bucket stays 0 for that reason
    rather than because the reference day happens to have none.
    """
    import pandas as pd
    df = pd.read_csv(path)
    in_window = df[(df["rq_time"] >= start_time) & (df["rq_time"] < end_time)]
    loaded = in_window[in_window["start"] != in_window["end"]]
    return int(len(df)), int(len(in_window)), int(len(in_window) - len(loaded)), int(len(loaded))


def run_day(task: Tuple[str, str]) -> Dict[str, Any]:
    """One scripted greedy episode on one day, fully instrumented. Picklable for Pool."""
    day_dir, day = task
    t_build = time.perf_counter()
    env = SDPDPAssignmentEnv({
        "constant_cfg_path": os.path.join(SCS, "constant_config_depot_cali_sc_1.csv"),
        "scenario_cfg_path": os.path.join(SCS, "example_depot_cali_sc_1.csv"),
        "K": K,
        "skip_output": True,
        "env_id": 0,
        "day_dir": day_dir,
        "day": day,
        # base_seed absent on purpose: every day runs at the row's seed 42
    })
    start_time = float(env.scenario_parameters[G_SIM_START_TIME])
    end_time = float(env.scenario_parameters[G_SIM_END_TIME])
    n_file, n_window, n_same_od_dropped, n_loaded = _file_request_counts(
        os.path.join(DEMAND, day_dir, f"{day}.csv"), start_time, end_time)

    buckets = {"steps": 0, "reservation": 0, "empty_candidates": 0,
               "same_od": 0, "duplicate_rid": 0}
    cand_lengths: List[int] = []
    tail = {"entered": False, "max_sim_time": end_time, "in_flight_at_end_time": None}
    seen_rids = set()

    obs, info = env.reset()
    op = env.sim.operators[0]
    sim = env.sim

    # --- P1.10's instrumentation ------------------------------------------------------- #
    # build_assignment_context returns None for all three internal-resolution cases without
    # distinguishing them, so the bucket has to be read off the PlanRequest afterwards.
    _build = op.build_assignment_context

    def counting_build(rq, sim_time):
        rid = rq.get_rid_struct()
        if rid in seen_rids:
            buckets["duplicate_rid"] += 1
        seen_rids.add(rid)
        ctx = _build(rq, sim_time)
        if ctx is None:
            prq = op.rq_dict.get(rid)
            if prq is not None and prq.o_pos == prq.d_pos:
                buckets["same_od"] += 1
            elif prq is not None and prq.get_reservation_flag():
                buckets["reservation"] += 1
            else:
                buckets["empty_candidates"] += 1
        return ctx

    op.build_assignment_context = counting_build

    _rra = sim.record_remaining_assignments
    _usf = sim.update_sim_state_fleets

    def counting_usf(last_time, next_time, **kw):
        tail["max_sim_time"] = max(tail["max_sim_time"], float(next_time))
        return _usf(last_time, next_time, **kw)

    def counting_rra(*a, **kw):
        tail["entered"] = True
        tail["in_flight_at_end_time"] = sum(
            1 for rq in sim.demand.rq_db.values()
            if getattr(rq, "pu_time", None) is not None
            and getattr(rq, "do_time", None) is None)
        return _rra(*a, **kw)

    sim.update_sim_state_fleets = counting_usf
    sim.record_remaining_assignments = counting_rra
    # ----------------------------------------------------------------------------------- #

    buckets["steps"] += 1                           # the first decision came from reset()
    cand_lengths.append(info["n_candidates"])

    t_start = time.perf_counter()
    build_seconds = t_start - t_build
    terminated = False
    while not terminated:
        obs, _reward, terminated, truncated, info = env.step(
            greedy_action(obs, env.action_masks(), env.observation_layout))
        if truncated:
            raise AssertionError(f"{day}: episode truncated; D6 requires terminated")
        if not terminated:
            buckets["steps"] += 1
            cand_lengths.append(info["n_candidates"])
    wall = time.perf_counter() - t_start

    summary = info["episode_summary"]
    env.close()

    accounted = sum(buckets.values())
    problems = []
    if accounted != n_loaded:
        problems.append(f"buckets sum to {accounted}, {n_loaded} requests loaded")
    if summary["exits"] != n_loaded:
        problems.append(f"{summary['exits']} exits, {n_loaded} requests loaded")
    if summary["unclassified"] != 0:
        problems.append(f"unclassified = {summary['unclassified']} -- a new §6.4 path exists")
    outcomes = sum(summary[k] for k in (
        "served", "rider_declined", "diffusion_cancelled", "no_show", "no_candidates",
        "operator_rejected", "unserved_at_horizon", "unclassified"))
    if outcomes != n_loaded:
        problems.append(f"§6.4 terminal outcomes sum to {outcomes}, {n_loaded} loaded")

    margin = summary.get("horizon_margin")
    return {
        "day": day,
        "day_dir": day_dir,
        "weekday": summary.get("weekday"),
        "month": summary.get("month"),
        "requests_in_file": n_file,
        "requests_in_window": n_window,
        # dropped by load_demand_file's second filter, before the operator sees anything --
        # distinct from the `same_od` bucket, which counts requests the *operator* resolved
        "same_od_dropped": n_same_od_dropped,
        "requests_loaded": n_loaded,
        "steps": buckets["steps"],
        "empty_candidates": buckets["empty_candidates"],
        "reservation": buckets["reservation"],
        "same_od": buckets["same_od"],
        "duplicate_rid": buckets["duplicate_rid"],
        "unclassified": summary["unclassified"],
        "duplicate_boardings": summary["duplicate_boardings"],
        "cand_max": max(cand_lengths) if cand_lengths else None,
        "cand_p95": p95(cand_lengths),
        "cand_over_k": sum(1 for n in cand_lengths if n > K),
        "last_dropoff": summary.get("last_dropoff"),
        "horizon_margin": margin,
        "tail_sim_seconds": tail["max_sim_time"] - end_time,
        # the cap bound, i.e. the tail broke its loop with vehicles still en route -- the only
        # way §6.4's branch 7 can fire
        "tail_cap_hit": (tail["max_sim_time"] - end_time) > TAIL_CAP_SECONDS,
        "tail_entered": tail["entered"],
        "in_flight_at_end_time": tail["in_flight_at_end_time"],
        "pickups": summary["pickups"],
        "declined": summary["rider_declined"],
        "cancelled": summary["diffusion_cancelled"],
        "no_show": summary["no_show"],
        "served": summary["served"],
        # §6.4's branch 7, unreachable on the reference day only because that day finishes its
        # work before end_time. Reported per day: a negative margin is what makes it reachable,
        # and `w_horizon = 0.0` stops being a placeholder the moment it fires.
        "unserved_at_horizon": summary["unserved_at_horizon"],
        "no_candidates": summary["no_candidates"],
        "operator_rejected": summary["operator_rejected"],
        "episode_reward": summary["episode_reward"],
        "build_seconds": build_seconds,
        "wall_seconds": wall,
        "pid": os.getpid(),
        "problems": problems,
    }


def check_reference(row: Dict[str, Any]) -> List[str]:
    """Mismatches between the reference day's row and P1.10's measurements."""
    bad = []
    for key, want in REFERENCE_ROW.items():
        got = row.get(key)
        if got != want:
            bad.append(f"{key}: got {got}, P1.10 measured {want}")
    margin = row.get("horizon_margin")
    if margin is None or abs(margin - REFERENCE_MARGIN) > MARGIN_TOL:
        bad.append(f"horizon_margin: got {margin}, P1.10 measured ~{REFERENCE_MARGIN}")
    return bad


def _fmt(value: Any, spec: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return format(value, spec) if spec else str(value)


def format_report(rows: List[Dict[str, Any]], processes: int, wall: float,
                  ref_bad: List[str], trial: bool) -> str:
    rows = sorted(rows, key=lambda r: r["day"])
    L: List[str] = []
    L.append("# RL Gym — corpus sanity sweep under greedy (P2.4)\n")
    L.append("Generated by `src/rl_gym/tools/corpus_sweep.py`. Scripted greedy through the Gym "
             "API,\nfixed-day mode with `base_seed` absent, so every day runs at the scenario "
             "row's seed 42.\n")
    L.append("```")
    L.append("caffeinate -i python -m src.rl_gym.tools.corpus_sweep")
    L.append("```")
    L.append("`caffeinate` is macOS-only and only keeps the machine awake; drop it elsewhere. "
             "The\n4-process default is sized for a 16 GB laptop at roughly 0.4 GB per "
             "worker.\n")
    if trial:
        L.append(f"**Trial run — {len(rows)} day(s), not the full sweep.**\n")
    L.append(f"- days swept: **{len(rows)}**")
    L.append(f"- processes: {processes}")
    L.append(f"- wall clock: {wall / 60:.1f} min")
    L.append(f"- distinct worker pids: {len({r['pid'] for r in rows})}")
    L.append("")

    problems = [(r["day"], p) for r in rows for p in r["problems"]]
    L.append("## Reconciliation\n")
    L.append("Per day: gym steps + empty-candidate + reservation + origin=destination + "
             "duplicate-rid\nmust equal the requests loaded, and `episode_summary()['exits']` "
             "must equal it too.\n")
    L.append(f"- days failing a reconciliation: **{len(problems)}**")
    L.append(f"- days with `unclassified != 0`: "
             f"**{sum(1 for r in rows if r['unclassified'])}**")
    L.append("")
    if problems:
        L.append("**Every line here is a defect:**\n")
        for day, p in problems:
            L.append(f"- {day}: {p}")
        L.append("")

    L.append("## Reference day\n")
    if any(r["day"] == REFERENCE_DAY for r in rows):
        if ref_bad:
            L.append(f"**{REFERENCE_DAY} does NOT reproduce P1.10's figures:**\n")
            for b in ref_bad:
                L.append(f"- {b}")
        else:
            L.append(f"`{REFERENCE_DAY}` reproduces P1.10's figures exactly: 445 loaded, 436 "
                     "steps, 9 empty\nlists, max list 11, 13 over `K`, `horizon_margin` "
                     "≈194.7 s, 288 pickups, 118 declines,\n28 cancellations, 2 no-shows, "
                     "`unclassified` 0, 1 duplicate boarding.")
            if processes == 1 and len(rows) > 1:
                order = " → ".join(r["day"] for r in sorted(rows, key=lambda r: r["day"]))
                L.append(f"\nIt ran **last in a single process**, after the other day(s) of "
                         f"this run, so nothing\ncarries over from one simulation to the next: "
                         f"global RNG, routing caches, stochastic\ntravel-time state and "
                         f"logging all start clean per episode. Days present: {order}.")
    else:
        L.append(f"`{REFERENCE_DAY}` was not in this run, so the automatic check did not run.")
    L.append("")

    if rows:
        L.append("## Where `K = 8` binds\n")
        by_k = sorted(rows, key=lambda r: (-r["cand_over_k"], r["day"]))[:10]
        L.append("| day | lists over K | max list | p95 | steps |")
        L.append("|---|---|---|---|---|")
        for r in by_k:
            L.append(f"| {r['day']} | {r['cand_over_k']} | {r['cand_max']} | "
                     f"{_fmt(r['cand_p95'])} | {r['steps']} |")
        over = [r["cand_over_k"] for r in rows]
        maxes = [r["cand_max"] for r in rows if r["cand_max"] is not None]
        p95s = [r["cand_p95"] for r in rows if r["cand_p95"] is not None]
        total_steps = sum(r["steps"] for r in rows)
        p95_over = [r for r in rows if r["cand_p95"] is not None and r["cand_p95"] > K]
        max_over = [r for r in rows if r["cand_max"] is not None and r["cand_max"] > K]
        L.append("")
        L.append("### Corpus-wide\n")
        L.append(f"- **share of all decisions whose candidate list exceeds `K = {K}`: "
                 f"{sum(over)} of {total_steps} = {100 * sum(over) / total_steps:.2f}%**")
        L.append(f"- **days whose p95 list length is above {K}: {len(p95_over)} of "
                 f"{len(rows)}**"
                 + ("" if not p95_over
                    else " — " + ", ".join(f"{r['day']} (p95 {r['cand_p95']:g})"
                                           for r in p95_over[:12])
                    + (", …" if len(p95_over) > 12 else "")))
        L.append(f"- days with any list above {K}: {len(max_over)} of {len(rows)}")
        L.append(f"- per-day p95: min {min(p95s):g}, median {statistics.median(p95s):g}, "
                 f"max {max(p95s):g}")
        L.append(f"- per-day max: min {min(maxes)}, median {statistics.median(maxes):g}, "
                 f"**largest list anywhere {max(maxes)}**")
        L.append(f"- lists over `K` per day: min {min(over)}, median "
                 f"{statistics.median(over):g}, max {max(over)}")
        L.append("")
        L.append(f"A truncated list loses only its tail — the largest `delta_cfv` values, the "
                 f"weakest\ninsertions (D3). The question `K` turns on is whether the policy "
                 f"wants candidates it cannot\nsee, which the slot-selection frequency in "
                 f"P2.13 answers; this is the other half of it.\nDecisions on `K` stay with "
                 f"Ritun, before P2.8.")
        L.append("")

        L.append("## Horizon margin, the tail, and §6.4's branch 7\n")
        with_margin = [r for r in rows if r["horizon_margin"] is not None]
        thin = sorted(with_margin, key=lambda r: r["horizon_margin"])[:10]
        L.append("| day | horizon_margin (s) | last_dropoff | tail sim s | in flight at "
                 "end_time | unserved at horizon | steps |")
        L.append("|---|---|---|---|---|---|---|")
        for r in thin:
            L.append(f"| {r['day']} | {r['horizon_margin']:.1f} | "
                     f"{_fmt(r['last_dropoff'], '.1f')} | {r['tail_sim_seconds']:.1f} | "
                     f"{_fmt(r['in_flight_at_end_time'])} | {r['unserved_at_horizon']} | "
                     f"{r['steps']} |")
        L.append("")
        working_tail = [r for r in rows if r["tail_sim_seconds"] > 0]
        negative = [r for r in with_margin if r["horizon_margin"] < 0]
        stranded = [r for r in rows if r["unserved_at_horizon"]]
        worst_tail = max(rows, key=lambda r: r["tail_sim_seconds"])
        cap_hit = [r for r in rows if r["tail_cap_hit"]]
        L.append(f"- days where the `record_remaining_assignments` tail advanced the simulation "
                 f"at all: **{len(working_tail)} of {len(rows)}**")
        L.append(f"- days with a **negative** margin, i.e. the last dropoff after `end_time`: "
                 f"**{len(negative)} of {len(rows)}**")
        if with_margin:
            margins = sorted(r["horizon_margin"] for r in with_margin)
            L.append(f"- margin across the sweep: min {margins[0]:.1f} s, median "
                     f"{statistics.median(margins):.1f} s, max {margins[-1]:.1f} s")
        L.append("")
        L.append("### The branch-7 indicator is the tail against its cap, not the margin\n")
        L.append(f"- **longest tail across the sweep: {worst_tail['tail_sim_seconds']:.0f} s "
                 f"on {worst_tail['day']}, "
                 f"{100 * worst_tail['tail_sim_seconds'] / TAIL_CAP_SECONDS:.1f}% of the "
                 f"{TAIL_CAP_SECONDS:,} s cap**")
        L.append(f"- days where the cap bound (the tail gave up with vehicles still en route): "
                 f"**{len(cap_hit)}**"
                 + ("" if not cap_hit else " — " + ", ".join(r["day"] for r in cap_hit[:10])))
        L.append(f"- days where **§6.4's branch 7 fired** (accepted but never picked up): "
                 f"**{len(stranded)}**"
                 + ("" if not stranded
                    else " — " + ", ".join(f"{r['day']} ({r['unserved_at_horizon']})"
                                           for r in stranded[:10])))
        L.append("")
        L.append("**A negative margin does not bring branch 7 closer.** "
                 "`record_remaining_assignments` runs\nuntil every vehicle finishes its route "
                 "and only then gives up, at "
                 f"`end_time + {TAIL_CAP_SECONDS:,}`\n(`FleetSimulationBase.py` ~625, written "
                 "`2 * 7200`), where it **breaks the loop with vehicles\nstill en route**. That "
                 "break is the only way a rider who accepted an offer can end the day\nwithout "
                 "a pickup, so **the distance between the longest tail and the cap is the "
                 "indicator**,\nand the margin is not. Negative margins are simply **expected "
                 "on busier days**: more\nrequests arriving late means more trips still running "
                 "at `end_time`, which the tail then\ncompletes. Decisions on `w_horizon` stay "
                 "with Ritun (§6.4, P2.13).")
        L.append("")

        L.append("## Per-day rows\n")
        L.append("| day | dir | wd | mo | file | window | o=d drop | loaded | steps | empty "
                 "| resv | o=d | dup rid "
                 "| uncl | dup brd | max | p95 | >K | last dropoff | margin s | tail s | "
                 "in flt | pickups | served | decl | canc | n/s | unsrv | wall s |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
                 "---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            L.append(
                f"| {r['day']} | {r['day_dir'].replace('corpus', 'c')} | {_fmt(r['weekday'])} "
                f"| {_fmt(r['month'])} | {r['requests_in_file']} | {r['requests_in_window']} "
                f"| {r['same_od_dropped']} | {r['requests_loaded']} "
                f"| {r['steps']} | {r['empty_candidates']} | {r['reservation']} "
                f"| {r['same_od']} | {r['duplicate_rid']} | {r['unclassified']} "
                f"| {r['duplicate_boardings']} | {_fmt(r['cand_max'])} | {_fmt(r['cand_p95'])} "
                f"| {r['cand_over_k']} | {_fmt(r['last_dropoff'], '.1f')} "
                f"| {_fmt(r['horizon_margin'], '.1f')} | {r['tail_sim_seconds']:.1f} "
                f"| {_fmt(r['in_flight_at_end_time'])} "
                f"| {r['pickups']} | {r['served']} | {r['declined']} | {r['cancelled']} "
                f"| {r['no_show']} | {r['unserved_at_horizon']} | {r['wall_seconds']:.1f} |")
        L.append("")
        L.append("`dir`: `c` = `corpus` (training), `c_val` = `corpus_val` (held-out "
                 "validation), `c_eval` = `corpus_eval` (held-out test). The committed report "
                 "predates P2.17's split and shows the 26 held-out days as `c_eval`.\n")
        L.append("**The demand loader applies two filters, in this order** "
                 "(`demand.py` ~103-105): `file` is\nrows in the CSV; `window` keeps "
                 "`rq_time` in `[start_time, end_time)`; `o=d drop` is how many of\nthose are "
                 "then removed because their `start` node equals their `end` node; `loaded` is "
                 "what\nreaches the simulation. A dropped row never enters `future_requests`, "
                 "never reaches\n`build_assignment_context` and never calls `record_user`, so it "
                 "is correctly in no bucket —\nwhich is why it must not be counted as loaded. "
                 "The `o=d` column, by contrast, counts\nrequests the **operator** resolved as "
                 "origin-equals-destination, and is 0 throughout.\n")
        L.append("`margin` is `end_time − last_dropoff`; `tail s` is the simulated seconds "
                 "`record_remaining_assignments`\nadded past `end_time`; `in flt` is the riders "
                 "picked up but not yet dropped off when that\ntail starts; `unsrv` is §6.4's "
                 "branch 7. Asserted per day by the sweep:\n`steps + empty + resv + o=d + dup "
                 "rid = loaded`, `exits = loaded`, and the eight §6.4\nterminal outcomes "
                 "summing to `loaded`.\n")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Corpus sanity sweep under greedy (P2.4)")
    ap.add_argument("--days", default=None,
                    help="comma-separated day stems to sweep instead of all 127, "
                         "**in the order given**")
    ap.add_argument("--processes", type=int, default=MAX_PROCESSES,
                    help=f"worker processes, at most {MAX_PROCESSES} (default {MAX_PROCESSES}); "
                         "1 runs the days sequentially in this process")
    ap.add_argument("--report", default=REPORT)
    args = ap.parse_args()

    if args.processes < 1:
        raise SystemExit("--processes must be >= 1")
    if args.processes > MAX_PROCESSES:
        raise SystemExit(f"--processes is capped at {MAX_PROCESSES} for a 16 GB machine; "
                         f"raise MAX_PROCESSES deliberately to go higher")

    all_pairs = kept_days()
    if args.days:
        wanted = [d.strip() for d in args.days.split(",") if d.strip()]
        index = {day: (sub, day) for sub, day in all_pairs}
        missing = [d for d in wanted if d not in index]
        if missing:
            raise SystemExit(f"not kept days: {missing}")
        tasks = [index[d] for d in wanted]          # the given order, not sorted
        trial = True
    else:
        tasks = all_pairs
        trial = False
        if len(tasks) != 127:
            raise SystemExit(f"expected 127 kept days, found {len(tasks)}")

    print(f"sweeping {len(tasks)} day(s) with {args.processes} process(es)", flush=True)
    if args.processes == 1:
        print("  one process, days in the order given — the reference day reproducing P1.10 "
              "after\n  other days shows nothing carries over between simulations", flush=True)

    t0 = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    if args.processes == 1:
        # deliberately no Pool: one process, days strictly in order
        for i, task in enumerate(tasks, 1):
            row = run_day(task)
            rows.append(row)
            print(f"  [{i}/{len(tasks)}] {row['day']} ({row['day_dir']}) "
                  f"steps {row['steps']} loaded {row['requests_loaded']} "
                  f"uncl {row['unclassified']} margin "
                  f"{_fmt(row['horizon_margin'], '.1f')} s  {row['wall_seconds']:.1f} s "
                  f"pid {row['pid']}", flush=True)
    else:
        from multiprocessing import Pool
        with Pool(processes=args.processes) as pool:
            for i, row in enumerate(pool.imap_unordered(run_day, tasks), 1):
                rows.append(row)
                print(f"  [{i}/{len(tasks)}] {row['day']} ({row['day_dir']}) "
                      f"steps {row['steps']} uncl {row['unclassified']} "
                      f"{row['wall_seconds']:.1f} s pid {row['pid']}", flush=True)
    wall = time.perf_counter() - t0

    ref = next((r for r in rows if r["day"] == REFERENCE_DAY), None)
    ref_bad = check_reference(ref) if ref is not None else []

    with open(args.report, "w") as fh:
        fh.write(format_report(rows, args.processes, wall, ref_bad, trial))

    problems = [(r["day"], p) for r in rows for p in r["problems"]]
    print()
    print(f"days swept {len(rows)} | wall {wall / 60:.1f} min | "
          f"pids {len({r['pid'] for r in rows})}")
    print(f"reconciliation failures : {len(problems)}")
    print(f"unclassified != 0       : {sum(1 for r in rows if r['unclassified'])}")
    if ref is not None:
        print(f"{REFERENCE_DAY} vs P1.10   : "
              f"{'MATCHES' if not ref_bad else 'MISMATCH — ' + '; '.join(ref_bad)}")
    print(f"wrote {args.report}")
    for day, p in problems:
        print(f"  PROBLEM {day}: {p}")
    return 1 if (problems or ref_bad) else 0


if __name__ == "__main__":
    sys.exit(main())
