# RL Gym — Phase 1 results

Figures from the Phase 1 exit gate (P1.10) and the work items behind it.
Reproduce with `python -m unittest tests.test_rl_gym`.

Scenario: `constant_config_depot_cali_sc_1.csv` + `example_depot_cali_sc_1.csv`,
`random_seed = 42`, demand `2024-06-20.csv` (445 requests), fleet `default_vehtype:12`,
`start_time` 25200 → `end_time` 70000, `time_step` 10, `K = 8`.

---

## P1.10 — Phase 1 exit gate

### Byte-for-byte comparison

A scripted greedy policy driven entirely through the Gym API produces
`1_user-stats.csv` **identical to `docs/baseline_user_stats.csv`**.

| | |
|---|---|
| md5 produced | `53879ea79cf211d700f670dee1173dd5` |
| md5 baseline | `53879ea79cf211d700f670dee1173dd5` |
| size | 117,056 bytes |
| differing bytes | **0** |

`base_seed` absent, so `G_RANDOM_SEED` stayed at the row's 42 and the comparison is like for
like (D7). The generator, observation extraction, ranking, masking, action translation, reward
wiring, seeding and episode termination are therefore all correct simultaneously, with no RL
library involved.

### Request reconciliation

Every request lands in exactly one bucket.

| Bucket | Count |
|---|---|
| gym steps (decisions) | 436 |
| reservation branch | 0 |
| empty candidate list | 9 |
| same origin and destination | 0 |
| duplicate rid | 0 |
| **total** | **445** |

The duplicate-rid bucket is expected to be 0 — the parent `user_request` has no such branch
(§1). A non-zero value means the source changed.

### Throughput

| | |
|---|---|
| gym steps | 436 |
| wall clock | 38.00 s |
| steps / second | 11.47 |

### Gaps between consecutive decision epochs

**Simulated seconds** — what bounds how much world each reward window sweeps up, and therefore
how many pickups, declines, cancellations and no-shows can land in a single `flush()`.

| | min | median | mean | p95 | max |
|---|---|---|---|---|---|
| all gaps | 0.0 | 60.0 | 99.2 | 300.0 | 660.0 |
| non-zero gaps only | 60.0 | 120.0 | 136.5 | 360.0 | 660.0 |

**Zero-gap count: 119 of 435 (27.4%)** — reported separately rather than folded into the
distribution, because those are two distinct regimes. A zero gap means two requests arrived in
the same simulation time step, so that reward window carries **no simulated time at all** and
can only contain the action's own immediate consequence. The remaining windows span 60–660
simulated seconds. For reward-weight design (P2.5) it is the shape that matters, not the
pooled median: roughly a quarter of steps see an empty window and the rest see one or two
minutes of simulated world.

**Wall-clock seconds** — throughput planning only.

| min | median | mean | p95 | max |
|---|---|---|---|---|
| 0.0071 | 0.0771 | 0.0872 | 0.1707 | 0.4494 |

### `record_remaining_assignments` tail

| | |
|---|---|
| `end_time` | 70000 |
| max simulated time reached | 70000 |
| **simulated seconds added past `end_time`** | **0** |
| riders in flight at `end_time` | **0** |
| loop cap (`end_time + 2*7200`) | 14400 s — **0% used** |

**The tail is empty on this day, and it is empty for a specific reason.** The last request
arrives at `rq_time = 68400`; the latest pickup is 69064.2 and the latest dropoff 69805.3, both
**before** `end_time = 70000`. Every trip has already completed when the main loop ends, so
`record_remaining_assignments` finds nothing to finish and advances the simulation zero extra
seconds. Confirmed directly: the baseline has no row with a pickup or dropoff past 70000, and
the instrumented run recorded `record_remaining_assignments` as entered with 0 riders in flight.

Two consequences worth carrying forward:

- **The tail contributes nothing to the 38 s episode.** It is not a hidden cost at this scale.
- **The headroom is about 195 simulated seconds** — the gap between the last dropoff (69805.3)
  and `end_time` (70000). That is thin. A later demand tail, a larger fleet load, a longer
  trip, or a policy that defers pickups would push trips past `end_time` and start the tail
  doing real work. The cap is nowhere near binding today, so §6.4's branch 7
  (unserved at horizon) stays inactive and `w_horizon = 0.0` remains a placeholder rather than
  a decision — but that rests on this 195 s margin, not on anything structural.

### Candidate list lengths

| decisions | min | median | mean | max | exceeding `K = 8` |
|---|---|---|---|---|---|
| 436 | 1 | 5 | 4.60 | 11 | 13 |

Reproduces the P1.3 figures exactly. The 13 lists over `K` lose only their tail — the largest
`delta_cfv` values, the weakest insertions — which is what supports `K = 8` (D3).

### `episode_summary()`

| Key | Value |
|---|---|
| pickups (distinct rids) | 288 |
| served | 288 |
| rider_declined | 118 |
| diffusion_cancelled | 28 |
| no_candidates | 9 |
| no_show | 2 |
| operator_rejected | 0 |
| unserved_at_horizon | 0 |
| **unclassified** | **0** |
| exits | 445 |
| boarding_calls | 289 |
| **duplicate_boardings** | **1** |
| labelled_no_candidates | 9 |
| labelled_rejections | 0 |
| episode_reward | 116.510947 |
| unflushed_reward | 0.0 |
| random_seed | 42 |

Terminal outcomes sum to 445. `unclassified = 0` means every request reaching `record_user`
matched a §6.4 branch. `duplicate_boardings = 1` is rid 50, the no-show-adjacent re-boarding
described in §6.4; the pickup count stays 288 distinct rids against 289 boarding calls because
`on_pickup` is last-write-wins.

Reward weights are the §6.1 starting values, untuned — Phase 1 delivers the wiring and the
counts, not an economic design (§6.6).

---

## Environment construction

| | |
|---|---|
| `reset()` cold (first) | 1.068 s |
| `reset()` warm | 0.431 – 0.469 s, median 0.460 s |
| as a share of a 38 s episode | ~1.5% |

Measured under P1.8 over five resets. This is the basis for §10's decision that caching the
routing engine or any simulation object across `reset()` is out of scope.

---

## Test suite

| Suite | Tests |
|---|---|
| `test_rl_gym_spaces.py` | 29 |
| `test_rl_gym_observers.py` | 52 |
| `test_rl_gym_reward.py` | 34 |
| `test_rl_gym_env.py` | 30 |
| `test_rl_gym.py` (exit gate) | 7 |

`check_env` passes through `MaskRespectingShim` with `base_seed` both absent and set.
