# Phase 2 exit gate — short test training (P2.15)

Measured 2026-10-11 on the M4 MacBook (Mac16,1; 10 cores, 4 performance + 6 efficiency; 16 GB),
plugged in, heavy applications closed, the whole run under `caffeinate -i`.

**Result: 22/22 checks passed, in 24.0 minutes.** Every component of Phase 2 ran together with
repositioning off: D10's day rotation over the 101 training days, D7's per-episode reseeding, the
139-feature observation, the rolling statistics, `VecNormalize`, MaskablePPO with action masking,
the evaluation callback over held-out pairs, checkpointing, and P2.14's KPI harness against stock
greedy on identical (day, seed) pairs.

**This is not a performance test and not a learning test** (§8 P2.15). 47 episodes is under half
of one pass over the training days and the reward weights are unswept, so the KPI comparison
below says nothing about whether the policy can learn — only that the machinery produces a
comparable table. The learning curve was logged and deliberately not read.

Reproduce with:

    caffeinate -i python -m src.rl_gym.tools.phase2_run --out-dir runs/p215
    # either half alone
    caffeinate -i python -m src.rl_gym.tools.phase2_run --out-dir runs/p215 --skip-harness
    caffeinate -i python -m src.rl_gym.tools.phase2_run --out-dir runs/p215 --harness-only \
        --checkpoint runs/p215/checkpoints/model_ep00047.zip

Artifacts live under `runs/p215b/` — checkpoints, TensorBoard event files, `kpi_report.txt` and
`p215_figures.json`, the machine-readable source of every figure quoted here. `runs/` is
git-ignored (added in P2.15), so the tables below are reproduced in full rather than referenced.

---

## 1. What was run

| | |
|---|---|
| `base_seed` | **20261015** (fresh; resuming from these checkpoints requires another, P2.13) |
| workers | 4 training (`SubprocVecEnv`, D8) + 2 evaluation |
| `n_steps` | **256** per env, for this run only, so the update path runs ~22 times rather than ~5 |
| `total_timesteps` | **22,528** = 22 rollouts × 1024 → 5,632 steps per worker, **22 updates** |
| episodes | **47** completed: 12 / 12 / 11 / 12 per worker, all inside pass 0 |
| other hyperparameters | P2.13's defaults: `gamma` 0.999, `gae_lambda` 0.95, `learning_rate` 3e-4, `batch_size` 256, `n_epochs` 10, `clip_range` 0.2, `ent_coef` 0.0, `net_arch` [256, 256] |
| reward weights | §6.1's eight, explicitly: `w_pickup` 1.0, `w_wait` 0.05, `w_reject` 1.0, `w_decline` 0.1, `w_cancel` 0.6, `w_noshow` 0.0, `w_horizon` 0.0, `w_no_candidates` 0.0 |
| demand | `day_dir = corpus` (101 training days), D10 shuffled-pass rotation |
| output | `skip_output = True`, `log_level = warning` for training; output on for the harness |
| evaluation callback | 4 pairs over 2 workers (2 each), `eval_freq = 1,700` → evaluations at timesteps 6,800 / 13,600 / 20,400 |
| callback pairs | the **first** seed of four harness days, from `docs/RL_GYM_EVAL_SEEDS.csv`: (2024-01-11, 1620816), (2024-03-19, 408699), (2024-06-20, 2071786), (2024-08-13, 1881929) |
| checkpoints | every 12 episodes → 4: `model_ep00012/24/36/47.zip` and run metadata **(see the correction below)** |
| final harness | 5 days × the first 2 seeds of each (10 pairs), both policies: stock greedy on 4 processes, the `model_ep00047` checkpoint on 5 workers |

2024-10-01 is in the harness but deliberately **not** in the callback: at 545 greedy steps it is
the longest of the five days, and leaving it out keeps three evaluations at ~4 minutes instead of
~6. Four pairs over two workers also gives each worker a 2-pair slice, which is what exercises
P2.13's rotation argument — a 1-pair slice cannot rotate.

> **Correction — P2.16.** The checkpoints row originally read "each with its `vecnormalize.pkl`".
> That was wrong: all four checkpoints shared **one** statistics file, which each save
> overwrote, so only `model_ep00047.zip` kept the statistics it was trained under and the other
> three lost theirs. **No figure in this report is affected.** The harness evaluated
> `model_ep00047.zip`, and the shared file is that checkpoint's own — `ret_rms.count`
> 22,528.0001 against its `num_timesteps` 22,528, and P2.16's gate run produced a
> `model_ep00047_vecnormalize.pkl` **byte-identical** to it. The KPI md5
> `db34f5aea5f8cd59437fb2628c9534e0` stands, and P2.16 reproduced it twice under the new code:
> once through `--harness-only` on this very checkpoint with the shared file passed explicitly,
> and once from a full re-run of the gate at the same `base_seed`. What was lost is the ability
> to evaluate ep12, ep24 or ep36 from these artifacts; their statistics are gone, and the loader
> now says so instead of silently substituting ep47's.

### Machine conditions

Recorded because P2.10 showed throughput figures track machine load rather than the code: the
*same* code measured 11.40 steps/s early in a session and 9.66 two hours later.

| | run B (reported) | run A (first run) |
|---|---|---|
| load average at start | **2.48** (2.59 / 3.20) | **2.10** (2.21 / 2.23) |
| load average at end | **6.96** (5.73 / 5.05) | **7.17** (7.25 / 5.40) |
| load while running | median 4.23, max 17.41, 500+ samples at 2 s | median 3.87, max 9.57 |
| peak RSS per worker | **0.41 GB**, 2.43 GB over 6 FleetPy processes | 0.40 GB, 2.34 GB |

The end-of-run figure is the run's **own** load — the harness holds 9 concurrent simulations at
that point — not background load on the machine; an idle reading a few minutes later was 1.19.
Peak RSS matches P1.11's 0.40 GB per worker, each carrying its own routing engine.

---

## 2. The eight pass criteria

Each line below is one `PASS` from the driver, with what was actually measured. `unclassified`,
`unserved_at_horizon` and the mask audit are the three that could have failed silently, so each is
checked positively rather than by the absence of a crash.

| # | Criterion | How it was checked | Measured |
|---|---|---|---|
| 1 | run completes, exit codes 0, no orphans | `worker_exitcodes` from `venv.unwrapped.processes`; children enumerated after `close()`, helpers excluded **by pid** | exit codes `[0, 0, 0, 0]`, 47 episodes, 17.7 min; no orphaned worker after either the training venv or the harness |
| 2 | `unclassified = 0` in every episode | `EpisodeLogCallback.episode_summaries` for training, and `PairLoggingMaskableEvalCallback.episode_summaries` (added in P2.15) for the evaluation episodes; the key's *absence* fails the check too | **0 in all 47 training and all 12 evaluation episodes**, and 0 in all 20 harness runs. No §6.4 path was missed anywhere |
| 3 | no non-finite observation | P2.12's guard raises from `_observe`, the single site `reset()` and `step()` share; a raise in a worker aborts `learn` | **~22,587 observations** passed the guard (22,528 steps + 59 resets) with no raise |
| 4 | no masked action ever selected | a driver-side callback reads `action_masks` and `actions` out of `collect_rollouts`' locals and checks `mask[i][action]` at **every** step | **22,528 of 22,528 actions legal, 0 violations**; masks present at all 5,632 rollout steps (so masking was never silently off) |
| 5 | no day twice; days exactly D10's | `day_visit_counts()` for repeats; each worker's sequence compared against `day_for_episode(training_days, base_seed, 4, env_id, i)` | **47 distinct days, each visited once**; every worker's sequence matched D10 position for position; all episodes in pass 0 |
| 6 | a reloaded checkpoint is deterministic | `train_sdpdp._check_reload_determinism` on the final checkpoint: two rollouts on a fixed (day, seed) = (2024-06-20, `episode_seed` 42) out of `corpus_eval`, with the statistics saved beside it | identical 15-action sequences, `[0, 1, 3, 0, 1, 2, 1, 1, 3, 0, 3, 3, 3, 3, 1]` — and the *same* sequence in run A |
| 7 | KPI table for both policies on identical pairs | P2.14's harness on the 10 pairs; `(day, seed)` sets compared per policy; `kpi_row`'s own cross-check that each RL run's `operator_rejected + no_candidates` equals its rows carrying `offers = "0:"` | **20 rows, 10 per policy, identical pair sets**; every cross-check passed |
| 8 | every TensorBoard scalar from P2.13 | read back with `EventAccumulator`, never eyeballed | **19 required scalars, both histograms, 9 slot keys** — nothing missing |

Four sizing statements from §8 P2.15 were checked the same way, and all held: the callback fired
**3 times**, each evaluation ran **every pair exactly once** (the order rotated between
evaluations — `[01-11, 06-20, 03-19, 08-13]`, then `[03-19, 08-13, 01-11, 06-20]`, then the first
again — which is P2.13's discarded-episode effect, with coverage intact), the evaluation env kept
`training = False` and `norm_reward = False` after every sync, and 4 checkpoints were written with
their statistics and metadata.

---

## 3. Reported, not judged

### Slot-selection frequency — is `K = 8` binding?

| slot | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | reject |
|---|---|---|---|---|---|---|---|---|---|
| selections | 4,677 | 3,805 | 3,464 | 2,530 | 2,411 | 1,673 | 805 | 433 | 2,730 |
| share | 20.8% | 16.9% | 15.4% | 11.2% | 10.7% | 7.4% | 3.6% | **1.9%** | 12.1% |

Two readings, in tension, and the decision on `K` stays with Ritun:

- **Against `K` binding:** the frequency decays monotonically with slot index and slot 7 is the
  *least*-used candidate slot at 1.9%. There is no pile-up at the last slot, which is the
  signature §8 P2.13 says to watch for.
- **For watching it:** truncation bites far more often off the greedy path. Candidate lists
  averaged **5.88** per decision here against greedy's 4.60 on the reference day, and **55 lists
  per episode** exceeded `K` (median 56, max 88) — about **11.9% of decisions**, against greedy's
  **5.13%** corpus-wide (P2.4). An untrained policy rejecting 12.1% of requests leaves vehicles
  free, which lengthens lists. Per-episode maximum list length was 10–12, so no list came close
  to needing a much larger `K`.

Worth re-reading after the weight sweep, when rejection behaviour is no longer an artifact of an
untrained policy.

### The horizon margin and the tail against its cap

§6.4's branch 7 goes live only when the `record_remaining_assignments` tail's **14,400 s cap**
binds, so the tail is the indicator and the margin is not (P2.4).

| | min | median | mean | max |
|---|---|---|---|---|
| `tail_seconds` | 0.0 | 190.0 | 416.6 | **2,130.0** |
| `horizon_margin` (s) | −2,092.0 | −151.4 | −259.3 | +1,134.4 |

- The worst tail is **2,130 s = 14.8% of the cap**, leaving 6.8× headroom. That is **above
  greedy's worst anywhere in the 127-day corpus** (1,950 s, 13.5%) — the first measured sign that
  a policy lengthens tails, exactly as P2.13 anticipated, and the reason it is logged from the
  first training run rather than discovered later.
- A negative margin is the normal case, not a warning: **31 of 47 episodes**, median −151 s,
  against greedy's 79 of 127 and median −234 s (P2.4).
- **`unserved_at_horizon` was 0 in all 47 training episodes and all 20 harness runs**, so branch 7
  still has not fired and `w_horizon = 0.0` remains a placeholder rather than a decision. What
  would unsettle it is a tail approaching 14,400 s.
- Duplicate boardings: **1 across the whole run**, consistent with §6.4's mechanism (a no-show
  sharing a boarding stop with another rider).

### Throughput

| | run B | run A |
|---|---|---|
| per-worker rate, pause-free | **7.10 steps/s** | 7.44 steps/s |
| per-episode wall, derived from that rate | **65.3 s** | 62.3 s |
| per-episode wall, measured (median) | **65.5 s** | 64.4 s |
| per-episode wall, measured (mean / max) | 87.7 s / 175.9 s | 83.7 s / 160.0 s |
| episode steps (min / median / mean / max) | 387 / 454 / 463.5 / 572 | identical |
| SB3 `time/fps` (median, all 4 workers) | 22 | 23 |
| training wall | **17.7 min** = 13.2 min rollout + 4.4 min in pauses | 16.8 min = 12.6 + 4.2 |
| harness wall | **6.2 min** = greedy 2.7 + RL 3.5 | 5.7 min = 2.5 + 3.2 |
| whole run | **24.0 min** | 22.7 min |

Two things to read carefully here:

- **The median is the honest per-episode figure; the mean is not.** The three pauses the driver
  timed (266.4 s total, max 96.8 s) are the three evaluations, and they land *inside* whichever
  episode is in flight. The derived figure (65.3 s) and the measured median (65.5 s) agree to
  0.3%, which is what makes both trustworthy; the mean of 87.7 s is the same number with an
  evaluation charged to three episodes. See §6 for the logging issue behind this.
- **The 20–25 minute estimate held, but the per-episode estimate behind it did not.** Before the
  run I predicted ~10 steps/s per worker and 14–17 min for training, from P2.4's measured 44.7 s
  per day at 4-way parallelism. The true rate is **7.10**, about 30% slower, and the cause is
  measurable rather than mysterious: P2.4's figures come from the **greedy** driver with the
  Phase 1 observation, whereas this run has longer candidate lists (5.88 against 4.60 per
  decision — more insertion and more candidate-block work per decision), the full 139-feature
  observation, per-step policy inference, 6 FleetPy processes instead of 4, and a higher load. The
  three evaluations cost ~89 s each, within a few seconds of the ~85 s predicted.

**Projected full-length run** (~7,000 episodes, §8 After Phase 2 step 8): **31.7 h** from the
pause-free rate, **31.8 h** from the measured median episode — two independent derivations within
0.3%. Run A gives 30.3 h and 31.3 h. So **about 30–32 h at 4 workers on this laptop**, excluding
the evaluation callback's own time, which would add roughly an hour per 50 evaluations at this
callback size. Two caveats: a trained policy's candidate lists, and so its per-decision cost, need
not match an untrained one's; and full-length training is intended for Linux (HPC3), where §8a's
OpenMP workaround and this machine's 4 performance cores do not apply.

### Episode-level outcomes, across the 47 training episodes

| | min | median | mean | max |
|---|---|---|---|---|
| pickups | 182 | 223 | 224.4 | 267 |
| rider declines | 96 | 142 | 146.1 | 200 |
| post-match cancellations | 21 | 33 | 34.0 | 56 |
| no-shows | 0 | 2 | 2.3 | 7 |
| operator rejections | 15 | 55 | 56.6 | 104 |
| no-candidate requests | 0 | 3 | 3.8 | 12 |
| `episode_reward` | −55.4 | 6.3 | 8.5 | 60.0 |

The reward column is logged and **not read as a learning signal** — 47 episodes over 101 days say
nothing about learning, and §8 P2.15 excludes it from sizing the weight sweep.

---

## 4. The KPI comparison

Stock greedy (D9) and the `model_ep00047` checkpoint on the same 10 (day, seed) pairs, measured by
the same script from the same output files (P2.14). **Performance is not judged in Phase 2**: the
checkpoint has seen 47 episodes at unswept weights, so the right reading of this table is that it
exists, is complete, and is internally consistent — not that greedy is better.

### Per run

| policy | day | seed | served | mean_wait_s | mean_detour_ratio | decline_rate | cancellation_rate | no_show_rate | operator_rejections | no_candidate_requests | fleet_vkt_km | unclassified |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 2024-01-11 | 1186391 | 298.0 | 558.53 | 1.2713 | 0.2411 | 0.0623 | 0.0093 | 0 | 2 | 2274.91 | 0 |
| greedy | 2024-01-11 | 1620816 | 315.0 | 551.07 | 1.2981 | 0.1971 | 0.0592 | 0.0089 | 0 | 4 | 2401.62 | 0 |
| greedy | 2024-03-19 | 408699 | 293.0 | 598.86 | 1.3016 | 0.2090 | 0.0597 | 0.0189 | 0 | 11 | 2354.08 | 0 |
| greedy | 2024-03-19 | 2483245 | 298.0 | 580.59 | 1.2958 | 0.2094 | 0.0530 | 0.0187 | 0 | 7 | 2304.71 | 0 |
| greedy | 2024-06-20 | 1500010 | 305.0 | 602.40 | 1.3417 | 0.2259 | 0.0669 | 0.0061 | 0 | 20 | 2712.69 | 0 |
| greedy | 2024-06-20 | 2071786 | 294.0 | 605.91 | 1.3406 | 0.2713 | 0.0662 | 0.0063 | 0 | 10 | 2644.02 | 0 |
| greedy | 2024-08-13 | 312291 | 313.0 | 594.41 | 1.2908 | 0.2729 | 0.0733 | 0.0088 | 0 | 16 | 2671.00 | 0 |
| greedy | 2024-08-13 | 1881929 | 311.0 | 577.82 | 1.3000 | 0.2462 | 0.0925 | 0.0087 | 0 | 26 | 2662.69 | 0 |
| greedy | 2024-10-01 | 1421286 | 351.0 | 590.08 | 1.3033 | 0.3007 | 0.0941 | 0.0127 | 0 | 24 | 2797.21 | 0 |
| greedy | 2024-10-01 | 2312878 | 343.0 | 583.71 | 1.3402 | 0.2883 | 0.1300 | 0.0125 | 0 | 24 | 2730.19 | 0 |
| rl | 2024-01-11 | 1186391 | 251.0 | 637.38 | 1.3039 | 0.3178 | 0.0896 | 0.0108 | 13 | 3 | 2135.66 | 0 |
| rl | 2024-01-11 | 1620816 | 247.0 | 648.66 | 1.3060 | 0.3244 | 0.1011 | 0.0072 | 14 | 1 | 2045.70 | 0 |
| rl | 2024-03-19 | 408699 | 236.0 | 622.49 | 1.3165 | 0.3268 | 0.1168 | 0.0219 | 5 | 1 | 2001.98 | 0 |
| rl | 2024-03-19 | 2483245 | 239.0 | 618.76 | 1.2956 | 0.3160 | 0.1155 | 0.0217 | 6 | 2 | 2100.58 | 0 |
| rl | 2024-06-20 | 1500010 | 236.0 | 679.06 | 1.3431 | 0.3739 | 0.1282 | 0.0073 | 3 | 6 | 2244.45 | 0 |
| rl | 2024-06-20 | 2071786 | 224.0 | 730.31 | 1.3264 | 0.4092 | 0.1206 | 0.0078 | 4 | 6 | 2196.42 | 0 |
| rl | 2024-08-13 | 312291 | 249.0 | 641.31 | 1.2925 | 0.3318 | 0.1345 | 0.0069 | 43 | 8 | 2311.19 | 0 |
| rl | 2024-08-13 | 1881929 | 259.0 | 596.77 | 1.2896 | 0.3364 | 0.0993 | 0.0137 | 39 | 6 | 2360.76 | 0 |
| rl | 2024-10-01 | 1421286 | 294.0 | 605.38 | 1.2991 | 0.3584 | 0.1648 | 0.0140 | 19 | 9 | 2367.78 | 0 |
| rl | 2024-10-01 | 2312878 | 283.0 | 601.99 | 1.3711 | 0.4113 | 0.1325 | 0.0151 | 12 | 10 | 2336.97 | 0 |

### Overall, mean ± standard error over the 10 pairs

| policy | n | served | mean_wait_s | mean_detour_ratio | decline_rate | cancellation_rate | no_show_rate | operator_rejections | no_candidate_requests | fleet_vkt_km |
|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 10 | 312.1 ± 6.3 | 584.34 ± 5.74 | 1.3083 ± 0.0076 | 0.2462 ± 0.0114 | 0.0757 ± 0.0074 | 0.0111 ± 0.0015 | 0.00 ± 0.00 | 14.40 ± 2.79 | 2555.31 ± 62.56 |
| rl | 10 | 251.8 ± 6.9 | 638.21 ± 12.91 | 1.3144 ± 0.0082 | 0.3506 ± 0.0115 | 0.1203 ± 0.0068 | 0.0126 ± 0.0018 | 15.80 ± 4.51 | 5.20 ± 1.04 | 2210.15 ± 42.57 |

The per-day and per-month aggregations add nothing here — each of the five days sits in its own
month — and are in `runs/p215b/kpi_report.txt`.

One structural observation, which is about the measurement rather than the policy: the RL policy
rejects 15.8 requests per day and meets **fewer** no-candidate requests than greedy (5.2 against
14.4). Rejecting keeps vehicles free, so fewer later requests arrive to an empty candidate list.
The two no-offer buckets trade off, and both leave `offers = "0:"` in `1_user-stats.csv` — which is
exactly why P2.14 reads their split from `rl_episode_summary.json` and cross-checks the sum against
the file. That cross-check passed on all 10 RL runs.

---

## 5. Reproducibility: two independent runs

The gate was run twice at the same `base_seed`, 40 minutes apart (run A to diagnose the defects in
§6, run B as the reported artifact). **Everything the policy and the simulation produced was
identical**:

- `kpi_report.txt` **byte-identical** (md5 `db34f5aea5f8cd59437fb2628c9534e0`) — all 20 runs, both
  policies;
- identical day visits, identical slot-selection counts to the single selection, identical episode
  outcome statistics, identical horizon and tail figures, identical episodes per worker
  (12/12/11/12);
- the same 15-action deterministic sequence from the reloaded checkpoint.

Only wall clock, load and RSS differed: 7.10 against 7.44 steps/s per worker, a 4.6% spread on the
same code, with load medians of 4.23 and 3.87 — a third measurement of P2.10's point that
throughput tracks the machine, and the reason every figure above is quoted with its load.

That D10's rotation, D7's episode seeds, the parent-process seeding (P2.13), the PPO updates and
the evaluation all reproduce exactly across processes is a stronger statement than any single check
in §2, and it is the one that matters for the weight sweep: two configurations can be compared
without the comparison being noise from the run itself.

---

## 6. Defects found, and one left open

**Two in the driver's orphan check, both mine, both found after run A reported them as failures.**
Run A was 20/22; the two failures were the same check, and it was wrong in both directions.

1. **False positive.** It flagged a process whose cmdline was `caffeinate -i python -m
   src.rl_gym.tools.phase2_run`. Diagnosis, reproduced directly: `caffeinate -i <utility>` **execs
   the utility in its own process and forks a child to hold the power assertion**; that child keeps
   the original argv, so it appears as a child of the Python driver. It was never a FleetPy worker.
2. **False negative, which is the serious one.** The filter excluded children whose cmdline
   contained `forkserver`. But a forkserver child is `fork`ed without `exec`, so **every FleetPy
   worker inherits the forkserver's cmdline** —
   `python -c from multiprocessing.forkserver import main; main(7, 9, ['numpy'])`. Measured with
   two workers alive, the check reported **no** live children: it could not have detected a real
   orphan. A guard that cannot fail is decoration (§0, P1.10).

Both are fixed by resolving `multiprocessing`'s two helpers **by pid** — asked of
`forkserver._forkserver._forkserver_pid` and `resource_tracker._resource_tracker._pid` — and
ignoring any child that is not this interpreter. If those internals ever move, the helpers are
reported as orphans and the check fails loudly rather than passing vacuously. Re-verified in both
directions on the real machinery: with a 2-worker `SubprocVecEnv` open it lists **exactly** the two
worker pids; after `close()`, none; a `Pool(maxtasksperchild=1)`'s workers likewise appear and
disappear; and caffeinate's holder is present in the raw child list throughout and correctly
excluded, so the test is not vacuous. Run B then passed 22/22 with the working check.

**One known issue left open deliberately, to be fixed after Phase 2** (Ritun's instruction; not
changed in this item):

`EpisodeLogCallback._log_episode` measures an episode's wall clock as the gap between consecutive
episode completions on that worker, so **any evaluation or PPO update that lands inside an episode
is charged to that episode**. It inflates `episode/wall_seconds` and deflates
`episode/steps_per_second` for the few episodes that straddle a pause. Measured here: mean 87.7 s
against a median of 65.5 s and a derived 65.3 s; in a truncated smoke run where nearly every
episode straddled an evaluation it read **2.56 steps/s against a true 10.47**. It affects only
those two logged diagnostics — never the reward, the policy, the day rotation or any pass criterion
— and the driver sidesteps it by timing its own call gaps, which is where the pause figures above
come from. The fix is to measure an episode's own elapsed time inside the env rather than
differencing completion times in the callback. **It is now §8's After Phase 2 step 7**, placed
before full-length training because that is the run whose logs it would otherwise distort.

---

## 7. What this proves, and what it does not

Proves, simultaneously and end to end: the generator-driven env under MaskablePPO; action masking
reaching the policy (22,528 of 22,528 actions legal, masks present at every step); the full
139-feature observation with no non-finite value; §6.4's classification complete on every one of
~22,000 exits across 59 episodes and 20 harness runs (`unclassified = 0` throughout); D10's
rotation assigning each worker exactly its own days with none repeated; D7's episode seeds;
`VecNormalize` statistics saved and reloaded with a checkpoint that reproduces its actions; the
evaluation callback covering held-out pairs with frozen statistics; and P2.14's harness producing a
complete, cross-checked KPI table for both policies on identical (day, seed) pairs.

Does not prove, and is not meant to: that the policy learns anything (47 episodes, unswept
weights), that `K = 8` is right (§3 reads both ways; the decision is Ritun's), that the reward
weights are sensible (that is the sweep, §8 After Phase 2 step 2), or anything about the simulator
with §2.5's travel-time defect corrected — every figure here is measured on the simulator as it
currently stands, and step 1 of After Phase 2 invalidates them.

---

## 8. Verification run for this item

| | |
|---|---|
| full test suite, after the changes | **591 tests, 200.3 s, OK** (585 before P2.15; +6 for the evaluation-summary recording) |
| P1.10 exit gate | **8/8**, including both byte-for-byte comparisons — `1_user-stats.csv` against `docs/baseline_user_stats.csv` and `2-0_op-stats.csv` against `docs/baseline_op_stats.csv` |
| reference run, before and after everything | `1_user-stats.csv` `53879ea79cf211d700f670dee1173dd5`, `2-0_op-stats.csv` `ded74ef39a274397c921c29f128f6950` — unchanged (trap 1) |
| pre-flight suite, before the changes | 585 tests, 195.4 s, OK |

The suite was re-run **after** the code changes, not only before: the pre-flight preceded the edits
to `callbacks.py` and `train_sdpdp.train()`, so it did not cover them. Neither touches the env, the
trackers or the observation, which is why the gate was expected to pass — not a reason to skip it.
