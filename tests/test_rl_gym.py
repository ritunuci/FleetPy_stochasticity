"""Phase 1 exit gate: a scripted greedy rollout through the Gym API (P1.10).

Drives `SDPDPAssignmentEnv` entirely from outside — `reset()`, then `greedy_action` at every
step until `terminated` — and requires the resulting `1_user-stats.csv` to match
`docs/baseline_user_stats.csv` **byte for byte**. Passing means the generator, observation
extraction, ranking, masking, action translation, reward wiring, seeding and episode
termination are all correct simultaneously, with no RL library involved.

`base_seed` is deliberately absent, so `G_RANDOM_SEED` stays at the row's 42 and the comparison
is against a baseline generated under the same seed (D7). P1.9 appends
`_env0_pid<pid>_ep1` to the scenario name when output is on, so the run writes to its own
directory and cannot clobber the committed results.

**If it nearly matches, do not relax the comparison to a tolerance.** A near-match means the RL
path consumes global RNG differently from the greedy path, which would contaminate every
stochastic result reported afterwards.

Run just this gate with:
    python -m unittest tests.test_rl_gym
"""

import hashlib
import os
import shutil
import statistics
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.misc.globals import G_SIM_END_TIME  # noqa: E402
from src.rl_gym.gym_env import SDPDPAssignmentEnv  # noqa: E402
from src.rl_gym.policies import greedy_action  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCS = os.path.join(ROOT, "studies", "example_study", "scenarios")
RESULTS = os.path.join(ROOT, "studies", "example_study", "results")
BASELINE = os.path.join(ROOT, "docs", "baseline_user_stats.csv")
RESULTS_DOC = os.path.join(ROOT, "docs", "RL_GYM_PHASE1_RESULTS.md")
K = 8
TOTAL_REQUESTS = 445


def md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def describe(values, fmt="{:.4f}"):
    if not values:
        return "n/a"
    s = sorted(values)
    p95 = s[min(len(s) - 1, int(0.95 * len(s)))]
    return (f"min {fmt.format(s[0])} | median {fmt.format(statistics.median(s))} | "
            f"mean {fmt.format(statistics.mean(s))} | p95 {fmt.format(p95)} | "
            f"max {fmt.format(s[-1])}")


def run_scripted_greedy_episode():
    """One full greedy episode through the Gym API, with everything P1.10 must report."""
    env = SDPDPAssignmentEnv({
        "constant_cfg_path": os.path.join(SCS, "constant_config_depot_cali_sc_1.csv"),
        "scenario_cfg_path": os.path.join(SCS, "example_depot_cali_sc_1.csv"),
        "K": K,
        "skip_output": False,      # a user-stats file is required for the comparison
        "env_id": 0,
        # base_seed absent on purpose: keep G_RANDOM_SEED at the row's 42 (D7)
    })
    end_time = float(env.scenario_parameters[G_SIM_END_TIME])

    buckets = {"decision": 0, "reservation_branch": 0, "empty_candidates": 0,
               "same_origin_destination": 0, "duplicate_rid": 0}
    cand_lengths, sim_times, wall_gaps, rewards = [], [], [], []
    tail = {"in_flight_at_end_time": None, "max_sim_time": end_time, "entered": False}
    seen_rids = set()

    obs, info = env.reset()
    op = env.sim.operators[0]
    sim = env.sim

    # classify the requests that never reach the agent -- build_assignment_context returns None
    # for all three internal-resolution cases without distinguishing them
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
                buckets["same_origin_destination"] += 1
            elif prq is not None and prq.get_reservation_flag():
                buckets["reservation_branch"] += 1
            else:
                buckets["empty_candidates"] += 1
        return ctx

    op.build_assignment_context = counting_build

    # the tail: how far past end_time record_remaining_assignments runs, and how many riders
    # are still in flight when it starts
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

    # the first decision arrived from reset(); count it and record its context
    buckets["decision"] += 1
    cand_lengths.append(info["n_candidates"])
    sim_times.append(float(info["sim_time"]))

    t_start = time.perf_counter()
    t_prev = t_start
    terminated = False
    while not terminated:
        action = greedy_action(obs, env.action_masks(), K)
        obs, reward, terminated, truncated, info = env.step(action)
        now = time.perf_counter()
        wall_gaps.append(now - t_prev)
        t_prev = now
        rewards.append(reward)
        assert not truncated, "episode must end terminated, not truncated (D6)"
        if not terminated:
            buckets["decision"] += 1
            cand_lengths.append(info["n_candidates"])
            sim_times.append(float(info["sim_time"]))
    wall_seconds = time.perf_counter() - t_start

    summary = env.episode_summary()
    scenario_name = sim.scenario_name
    env.close()

    sim_gaps = [sim_times[i + 1] - sim_times[i] for i in range(len(sim_times) - 1)]
    return {
        "scenario_name": scenario_name,
        "user_stats": os.path.join(RESULTS, scenario_name, "1_user-stats.csv"),
        "buckets": buckets,
        "summary": summary,
        "rewards": rewards,
        "cand_lengths": cand_lengths,
        "sim_gaps": sim_gaps,
        "wall_gaps": wall_gaps,
        "wall_seconds": wall_seconds,
        "end_time": end_time,
        "tail": tail,
    }


class TestPhase1ExitGate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.r = run_scripted_greedy_episode()

    @classmethod
    def tearDownClass(cls):
        d = os.path.join(RESULTS, cls.r["scenario_name"])
        if os.path.isdir(d):
            shutil.rmtree(d)

    def test_user_stats_matches_baseline_byte_for_byte(self):
        produced = self.r["user_stats"]
        self.assertTrue(os.path.isfile(produced), f"no user-stats written at {produced}")
        with open(BASELINE, "rb") as a, open(produced, "rb") as b:
            base_bytes, run_bytes = a.read(), b.read()
        if base_bytes != run_bytes:
            # do not relax to a tolerance; report the shape of the difference and fail
            base_lines = base_bytes.decode().splitlines()
            run_lines = run_bytes.decode().splitlines()
            differing = [i for i, (x, y) in enumerate(zip(base_lines, run_lines)) if x != y]
            self.fail(
                f"user-stats differs from the baseline: {len(differing)} differing row(s) "
                f"out of {len(base_lines)}; first at line {differing[0] if differing else 'n/a'}; "
                f"length {len(run_lines)} vs {len(base_lines)}. "
                f"md5 {md5(produced)} vs {md5(BASELINE)}")
        self.assertEqual(md5(produced), md5(BASELINE))

    def test_every_request_accounted_for_in_exactly_one_bucket(self):
        b = self.r["buckets"]
        total = (b["decision"] + b["reservation_branch"] + b["empty_candidates"]
                 + b["same_origin_destination"] + b["duplicate_rid"])
        self.assertEqual(total, TOTAL_REQUESTS, f"buckets {b} sum to {total}")

    def test_duplicate_rid_bucket_is_zero(self):
        self.assertEqual(self.r["buckets"]["duplicate_rid"], 0,
                         "a non-zero duplicate-rid count means the source changed (§1)")

    def test_episode_terminated_with_no_unflushed_reward(self):
        s = self.r["summary"]
        self.assertEqual(s["unflushed_reward"], 0.0)
        self.assertAlmostEqual(sum(self.r["rewards"]), s["episode_reward"], places=9)

    def test_reward_counts_reconcile_with_the_baseline(self):
        import pandas as pd
        base = pd.read_csv(BASELINE)
        s = self.r["summary"]
        self.assertEqual(s["pickups"], int(base["pickup_time"].notna().sum()))
        self.assertEqual(s["served"], int(base["dropoff_time"].notna().sum()))
        self.assertEqual(s["no_show"], int((base["no_show_stat"] == True).sum()))  # noqa: E712
        self.assertEqual(s["diffusion_cancelled"],
                         int((base["diffusion_cancelled"] == True).sum()))  # noqa: E712
        self.assertEqual(s["rider_declined"],
                         int((base["rider_declined"] == True).sum()))  # noqa: E712
        self.assertEqual(s["exits"], len(base))
        self.assertEqual(s["unclassified"], 0)
        self.assertEqual(s["operator_rejected"], 0,
                         "scripted greedy never selects slot K")

    def test_seed_was_not_overwritten(self):
        self.assertEqual(self.r["summary"]["random_seed"], 42,
                         "base_seed must stay absent so the byte comparison is like for like")

    def test_results_document_written(self):
        self.assertTrue(os.path.isfile(RESULTS_DOC))


if __name__ == "__main__":
    unittest.main(verbosity=2)
