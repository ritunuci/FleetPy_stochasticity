# SDPDP Gymnasium environment

A Gymnasium environment that exposes FleetPy's ride-pooling **assignment decision** to a
reinforcement-learning policy, plus the training and evaluation code around it. The policy
replaces the greedy `argmin(delta_cfv)` rule in `PoolingIRSOnly`: for each incoming request it
picks one of the top-`K` candidate vehicles FleetPy's insertion heuristic proposes, or rejects
the request.

- **One gym step** = one request for which the operator has an actual choice.
- **One episode** = one full simulated service day.

---

## At a glance

| | |
|---|---|
| Decision epoch | a request with at least one candidate vehicle; requests with none are rejected before the agent is asked |
| Episode | one day, 07:00 to 19:26 (25,200–70,000 s), on a **fresh** simulation every `reset()` |
| Action | `Discrete(K + 1)`, `K = 8`: slots `0…K−1` are candidates in FleetPy's order, slot `K` is reject; empty slots are masked |
| Observation | **129** floats at `K = 8`: candidate block 96, request block 14, global block 19, addressed by name through `env.observation_layout` |
| Reward | weighted rider outcomes (pickup, wait, reject, decline, cancellation, …), charged as they happen |
| Training | `MaskablePPO` with `VecNormalize` over `SubprocVecEnv` — `train_sdpdp.py` at the repo root |
| Evaluation | stock FleetPy greedy against a checkpoint on held-out days; KPIs from output files only |
| Days | 101 training (`corpus/`), 12 validation (`corpus_val/`), 14 test (`corpus_eval/`), Tue/Wed/Thu, Jan–Oct 2024 |

---

## The problem this package solves

`ImmediateDecisionsSimulation.run()` owns the clock. It loops over time steps, calls `step()`,
which calls `user_request()` — and that is where the assignment decision happens, three frames
deep inside a loop nothing outside can pause.

Gymnasium requires the opposite: the agent owns the loop and calls `env.step(action)`.

This package inverts control with a **generator**. FleetPy's call stack suspends at a `yield`
and resumes when the env sends an action back. One flow of control, taking turns — no threads,
no queues, no possibility of deadlock or interleaving.

```
SDPDPAssignmentEnv.step(action)
        │  gen.send(choice)
        ▼
RLImmediateDecisionsSimulation.step_generator
        │  choice = yield ctx          ← the simulation was frozen here
        ▼
RLPoolingIRSOnly.commit_assignment_choice
        │  offer written, rider decides, simulation advances
        ▼  next decision reached
        │  yield ctx
        ▼
SDPDPAssignmentEnv   flush() → reward,  observe() → obs
```

---

## Package map

| Module | Role | Spec |
|---|---|---|
| `fleetctrl_rl.py` | the decision point: `user_request` split around the choice | P1.3 |
| `sim_env_rl.py` | the generator that pauses the simulation at each decision | D1, P1.4 |
| `spaces.py` | action space, candidate truncation, mask | D3, P1.5 |
| `observers.py` | the three observation blocks and the layout map | §5, P2.1–P2.11 |
| `rolling_stats.py` | trailing-window event counts feeding the observation | P2.7 |
| `zones.py` | H3 zone lookup (node → cell, cell → zone set) | P2.5 |
| `corpus.py` | the day splits, the manifest, the arrival baseline and their hashes | §2.9, P2.6, P2.17 |
| `reward.py` | reward accumulation and the outcome ladder | §6, P1.7 |
| `policies.py` | the one scripted greedy policy | P1.10 |
| `gym_env.py` | `SDPDPAssignmentEnv` | P1.8, P1.9, P2.3 |
| `day_rotation.py` | which training day each episode runs | D10 |
| `callbacks.py` | training logs, checkpoints, the evaluation callback | P2.13 |
| `evaluate.py` | the evaluation harness | P2.14, P2.17 |
| `kpis.py` | KPIs from a run's output directory | P2.14 |
| `output_integrity.py` | the cloud-folder guard and the row check | P2.20 |
| `offline/` | builders for the zone table and the arrival baseline | P2.5, P2.6 |
| `tools/` | `phase2_run.py`, `corpus_sweep.py`, `reward_trace.py` | P2.4, P2.15, P1.7 |

`train_sdpdp.py`, at the repository root, holds the training code (P1.11, P2.12–P2.18).

---

## Modules

In the order a decision flows through them, not alphabetical.

### `fleetctrl_rl.py` — the decision point

`RLPoolingIRSOnly(PoolingInsertionHeuristicOnly)` splits `user_request` either side of the line
the RL replaces.

| | |
|---|---|
| `PendingDecision` | carries `prq`, `candidates`, `sim_time`, `rid_struct`, elapsed CPU time across the gap where the policy runs |
| `build_assignment_context(rq, sim_time)` | everything up to the choice; returns `None` when there is nothing to decide |
| `commit_assignment_choice(pending, choice, sim_time)` | `choice` is a candidate index or `None` to reject |
| `user_request(rq, sim_time)` | greedy fallback: build, then commit with choice 0 |
| `set_reward_tracker(tracker)` | optional; `None` by default so non-RL scenarios are unaffected |

Three cases return `None` and generate no gym step: origin equals destination, the reservation
branch, and an empty candidate list. Each reproduces the parent's behaviour exactly, including
`_create_rejection` and the `G_FCTRL_CT_RQU` bookkeeping. An empty candidate list is labelled
for the reward (`note_no_candidates`) but charges nothing.

The class knows nothing about `K` or the action space. It receives a semantic choice, never a
raw action.

### `sim_env_rl.py` — the generator

`RLImmediateDecisionsSimulation(ImmediateDecisionsSimulation)` mirrors the parent's `step()`
and `run()` as generators.

- `step_generator(sim_time)` mirrors `step()` line for line, with `choice = yield ctx` between
  build and commit.
- `run_generator()` mirrors the `run()` loop, delegating with `yield from`, so a `.send()` on
  it reaches the `yield` buried in `step_generator`'s request loop.

`_rl_operator` raises if paired with a stock operator rather than falling back silently: a
silent fallback would run greedy while reporting that it is learning. **Finalisation sits
outside the `try/finally`**, so an abandoned episode does not run
`record_remaining_assignments`, which would advance the simulation up to four hours past
`end_time` on an env being torn down.

Its `__init__` (P2.20) only arranges logging. With output on, each episode gets its own
`00_simulation.log` instead of writing into the first episode's, and the handlers it replaces are
closed. With output off, P1.2's once-per-process `NullHandler` is kept.

These are mirrors. If either parent method changes, they must change with it; the byte-for-byte
gates are what catch a drift.

### `spaces.py` — truncation and masking

Five pure functions, no state. `K` is passed in.

| | |
|---|---|
| `make_action_space(k_max)` | `Discrete(K + 1)` |
| `reject_action(k_max)` | `K` |
| `truncate_candidates(candidates, k_max)` | prefix of at most `K`, as a new list |
| `candidate_slot_validity(n, k_max)` | `bool[K]` |
| `build_action_mask(n, k_max)` | `bool[K + 1]`, reject slot always `True` |

**Never sorts.** `insertion_with_heuristics` returns candidates sorted ascending by `delta_cfv`
via a stable sort, so element 0 is exactly what stock's `min()` selects, and ties are broken
upstream by proximity to the request origin. Re-sorting with any other tie-break would pick a
different vehicle on ties and break the byte gate.

`candidate_slot_validity` is the **single source of truth** for which slots are occupied: the
mask is that array plus one element, and `CandidateObserver` derives `is_valid` from the same
call.

### `observers.py` — the observation

Three observers, concatenated in the order candidate → request → global. Each declares its
features statically in `feature_spec()`; `output_length` is derived from that, and
`build_observation_layout` turns the specs into the name → slice map the env publishes as
`observation_layout`. Nothing hardcodes a total or an offset.

**Candidate block** (`CandidateObserver`), 12 features per slot, feature-major (each feature
is a length-`K` vector, so `delta_cfv` is `obs[8:16]`):

| Feature | Meaning | Scaling |
|---|---|---|
| `is_valid` | a candidate occupies the slot | — |
| `delta_cfv` | change in the control objective | `sign · log1p` |
| `offered_wait` | the wait this plan would quote the rider | ÷ `wait_normalizer_s` |
| `veh_lat`, `veh_lon` | vehicle position (or its offset from the request origin with `position_encoding = "relative"`) | [−1, 1] over the node hull |
| `free_seats` | spare capacity | ÷ vehicle capacity |
| `remaining_stops` | stops in the committed plan | raw |
| `directionality` | cosine between request and vehicle direction | [−1, 1] |
| `lateness_pre`, `lateness_post` | lateness against promises before / after the insertion | ÷ `wait_normalizer_s` |
| `added_distance` | route distance the insertion adds | km |
| `added_pickup_delay` | delay pushed onto riders already waiting | ÷ `wait_normalizer_s` |

**Request block** (`RequestObserver`), 14 features: `o_lat`, `o_lon`, `d_lat`, `d_lon`,
`n_pax`, a 3-wide weekday one-hot `dow` (Tue/Wed/Thu, vocabulary from the manifest), and six
zone counts — arrivals, declines and post-match cancellations over the origin's zone set in
the last 30 and 60 minutes. There is no month feature; the month is parsed
from the filename for reporting only.

**Global block** (`GlobalStateObserver`), 19 features: `time_sin`, `time_cos`, `day_progress`,
`idle_fraction`, fleet-wide `arrivals_30/60`, `arrivals_expected_30/60` from the arrival
baseline for this weekday and minute, `arrivals_z_30/60`, `declines_30/60`, `cancels_30/60`,
`wait_gap_30/60` (realized minus quoted wait of riders picked up in the window), and
`active_now/30/60` (scheduled active vehicles now and 30 / 60 minutes ahead).

Rules this module is bound by:

- **Observation code never mutates simulation state**. Plans are re-timed on copies,
  and the routing cache is restored to exactly what it was.
- **Nothing on this path draws a random number.** Every stochastic component of the simulator
  draws from numpy's one global stream; a single extra draw would shift every later outcome.
- **Every normalisation constant comes from `scenario_parameters`.** `wait_normalizer_s`
  defaults to `op_max_wait_time`; set it explicitly before relaxing the operator's maximum wait,
  or every wait feature is silently rescaled.
- **Padded slots are all zeros**, which is neutral for `VecNormalize`'s running statistics.

### `rolling_stats.py` and `zones.py` — the counts behind the observation

`RollingStatsTracker` keeps four event streams — arrivals (by `rq_time`), declines and
post-match cancellations (by the step clock), pickups (by `pu_time`, carrying the wait gap) —
and answers trailing, half-open windows `[t − W, t)`, so a request never counts itself. Events
are keyed to the rider's origin H3 cell. Exits are classified once, by the same
`reward.classify_exit` the reward uses. The tracker is rebuilt every `reset()`.

`ZoneTable` maps a node to its resolution-7 H3 cell (the `h3_id` column of `nodes.csv`) and a
cell to its zone set (the cell plus its neighbours, from
`data/rl_gym/example_network/h3_neighbours.csv`). Unknown keys raise. `h3` is never imported:
the cells are data. The zones are RL-internal; FleetPy's own zone systems are not involved.

### `corpus.py` — days, manifest and arrival baseline

- **Training days**: `training_day_files` is the single definition, shared by day rotation and
  the baseline builder.
- **The held-out split** (P2.17): `validation_days`, `test_days`, `held_out_days`,
  `split_of_day` and `day_dir_for`. `check_split_file` binds the folders to the frozen record
  `docs/RL_GYM_HELDOUT_SPLIT.csv`. A day in both folders raises.
- **Integrity**: `corpus_hash` covers the names and contents of the training CSVs;
  `baseline_hash` covers the arrival baseline's parsed rows. The env refuses to start if either
  disagrees with `corpus_manifest.json`, and names what differs.

The **arrival baseline** (P2.6, weekday-conditioned since P2.18) is one expected count per
(weekday, minute, window) over the 101 training days, with a day-to-day spread measured around
each day's own weekday curve, floored at `sd_floor`. It is built offline by
`offline/build_arrival_baseline.py`, which writes the baseline and the manifest together.

### `reward.py` — accumulation and classification

`RewardTracker` has four inputs, two of which only label:

| | |
|---|---|
| `on_pickup(rid, rq)` | from `Demand.record_boarding`: charges the pickup term |
| `on_exit(rid, rq)` | from `Demand.record_user`: charges every terminal outcome |
| `note_rejection(rid)` | from `commit_assignment_choice`: labels only |
| `note_no_candidates(rid)` | from `build_assignment_context`: labels only |

The weights, as used for the Phase 2 short test training (the weight sweep after Phase 2
revises them):

| | |
|---|---|
| `w_pickup` | 1.0 per pickup |
| `w_wait` | 0.05 per minute of realized wait, at pickup |
| `w_reject` | 1.0 |
| `w_decline` | 0.1 |
| `w_cancel` | 0.6 (post-match cancellation) |
| `w_noshow`, `w_horizon`, `w_no_candidates` | 0.0 |

`DEFAULT_REWARD_WEIGHTS` holds these, but training requires all eight explicitly
(`train_sdpdp.check_config`).

`on_exit` charges once per rid and is idempotent. `on_pickup` is **last-write-wins**:
`record_boarding` can fire twice for a rider co-located with a no-show, and a repeat charges the
difference. `classify_exit` is the eight-branch ladder, first match wins, with **no fallback to
rider decline**:

```
1. in the no-candidates set          →  -w_no_candidates
2. in the operator-rejection set     →  -w_reject
3. do_time is not None               →  served, no charge
4. diffusion_cancelled               →  -w_cancel
5. rider_declined                    →  -w_decline
6. pu_time is None and no_show       →  -w_noshow
7. pu_time is None and accepted      →  -w_horizon
8. otherwise                         →  unclassified, counted, no charge
```

Served sits third because successful alighting also routes through `record_user`. Branch 8 is a
diagnostic; the gates assert it is zero. `flush()` returns what accumulated since the last call.

### `policies.py` — the scripted greedy policy

`greedy_action(obs, mask, layout)` returns the valid slot with the smallest `delta_cfv`, read
**by name** through the layout map. It drives the P1.10 gate and the corpus sweep. It selects by
value rather than returning 0, even though the sorted list makes 0 always right, so that reading
the wrong part of the observation cannot go unnoticed. It is not the evaluation baseline: that
is stock FleetPy (D9).

### `gym_env.py` — the environment

`SDPDPAssignmentEnv(gymnasium.Env)` owns `K`, the action translation, the observers and the
per-episode wiring.

| Key | Default | Meaning |
|---|---|---|
| `constant_cfg_path`, `scenario_cfg_path` | required | FleetPy configs |
| `scenario_row` | 0 | which row of the scenario CSV |
| `K` | 8 | candidate slots; the action space is `K + 1` |
| `skip_output` | `True` | write no simulation output files |
| `reward_weights` | `DEFAULT_REWARD_WEIGHTS` | the eight weights |
| `env_id` | 0 | worker index; seeds and output names derive from it |
| `base_seed` | none | per-episode reseeding (D7); **absent means the scenario row's own seed** |
| `day_dir` | none | day folder: `corpus` for training, or a held-out folder |
| `day` | none | pin one day |
| `eval_pairs` | none | fixed `(day, seed)` pairs, used by evaluation |
| `episode_seed` | none | pin one seed |
| `n_workers` | none | required for day rotation |
| `wait_normalizer_s` | `op_max_wait_time` | divisor for every wait feature |
| `position_encoding` | `"latlon"` | or `"relative"` |
| `track_events` | `False` | per-event reward breakdown in `info["reward_events"]` |
| `log_level` | `"warning"` | |

**Day sources**, in precedence order: `eval_pairs`, `day`, rotation over `day_dir` (needs
`base_seed` and `n_workers`), and — with no `day_dir` — the scenario row's own day, the path the
byte gates use. Contradictory combinations raise rather than defaulting.

`reset()` builds a **fresh** simulation every episode; nothing is cached across resets, which is
what keeps travel-time, network and RNG state from leaking between episodes. The weekday is
parsed from the filename and checked at `reset()`.

`action_masks()` returns the mask stored with the current observation and never advances the
simulation. Masked actions **assert**. On termination: `terminated=True`, a zeros observation, a
reject-only mask, and `episode_summary()` in `info`.

`observation_contract` records what the observation means — `baseline_hash`, a fingerprint of
the layout (every feature's name and width, in order), the length, `position_encoding` and
`wait_normalizer_s`. Training stores it with every run; evaluation and resume refuse a
checkpoint whose record differs (P2.18).

With output on, the env refuses a results folder inside a cloud-synced directory, attaches a
`RowCounter` to each simulation, and checks the output files when the episode ends (P2.20).

### `day_rotation.py` and `callbacks.py` — training support

**Day rotation** (D10): global shuffled passes over the training days, one day per episode;
worker `env_id` takes positions `env_id`, `env_id + n_workers`, … of each pass. Each pass's
permutation comes from `SeedSequence([base_seed, 1010, p + 1])`, so every worker derives the same
one.

**Callbacks**:

- `EpisodeLogCallback`: per-episode TensorBoard scalars from `info` on the terminal step —
  reward, steps, outcome counts, `tail_seconds`, candidate-list lengths, slot-selection
  frequencies under `slots/` and the `train/day_visits` histogram. Frequent selection of slot
  `K − 1` would mean `K` is binding.
- `EpisodeCheckpointCallback`: a checkpoint every N episodes, each with its own
  `<stem>_vecnormalize.pkl`, plus `run_metadata.json` (base seeds, hyperparameters, reward
  weights, day folder, worker count and the observation contract).
- `PairLoggingMaskableEvalCallback` with `partition_eval_pairs`: the training-time evaluation on
  validation pairs, every pair exactly once, with the normalisation statistics asserted frozen.

### `evaluate.py`, `kpis.py`, `output_integrity.py` — evaluation

`evaluate.py` runs **stock FleetPy greedy** and/or an RL checkpoint on fixed `(day, seed)`
pairs and tabulates both through `kpis.py`. Every `--greedy` or `--rl` run must name its half,
`--validation` or `--final-test`, with no default; each day's folder is derived from the day, and
a day from the other half is refused. Seeds come from `docs/RL_GYM_EVAL_SEEDS.csv` (10 per
held-out day), which is written once and never regenerated.

- **Stock greedy** is built with `load_simulation_environment` and run with `run()` in a fresh
  process per run (`maxtasksperchild=1`), with exactly four overrides: day folder, demand file,
  seed and a unique scenario name. A run that would write into the reference directory raises
  first.
- **RL** checks the checkpoint's statistics pairing and observation contract before any worker
  starts, loads its own `VecNormalize` statistics with `training = False` and
  `norm_reward = False`, and predicts deterministically with masks.
- **The comparison is unpaired.** Every random draw comes from one global stream, so two
  policies on the same seed share a sample path only until their first different decision.

`kpis.kpi_row` reads four files from a run's output directory — `rl_output_rows.json`,
`1_user-stats.csv`, `2-0_op-stats.csv` and `rl_episode_summary.json` — and reports requests,
offers, accepted offers, served, mean wait, mean detour ratio, declines, cancellations,
no-shows (counts and rates), operator rejections, no-candidate requests, unserved at horizon,
unclassified and fleet VKT. `aggregate` gives mean ± standard error per day, per month and
overall.

`output_integrity.py`: `refuse_synced_path` refuses to write inside a cloud-synced folder
(OneDrive, Dropbox, Google Drive, iCloud, `~/Library/CloudStorage`), judged on the resolved
path. `RowCounter` counts rows as they leave FleetPy's buffers; `check_episode_output` compares
the files with that count and writes `rl_output_rows.json`, and `kpi_row` re-counts against it,
raising `OutputIntegrityError` on any difference. Both exist because a OneDrive-synced folder
was silently dropping rows from results files (P2.20).

---

## Using it

Run from the repository root, with the `fleetpy_rl` environment, in a clone **outside any
cloud-synced folder**.

```python
import numpy as np                       # before torch / SB3 (spec §8a)
from src.rl_gym.gym_env import SDPDPAssignmentEnv
from src.rl_gym.policies import greedy_action

SCS = "studies/example_study/scenarios"
env = SDPDPAssignmentEnv({
    "constant_cfg_path": f"{SCS}/constant_config_depot_cali_sc_1.csv",
    "scenario_cfg_path": f"{SCS}/example_depot_cali_sc_1.csv",
    "K": 8,
    "skip_output": True,
})   # no day_dir: the scenario row's own day (thursday/2024-06-20.csv) and seed 42

obs, info = env.reset()
terminated = False
while not terminated:
    action = greedy_action(obs, env.action_masks(), env.observation_layout)
    obs, reward, terminated, truncated, info = env.step(action)
print(info["episode_summary"])
```

### Training

```
python train_sdpdp.py                     # smoke test: workers, no learning
python train_sdpdp.py --train --out-dir runs/<name> --timesteps <n> --workers 4
python train_sdpdp.py --train --resume-from runs/<name>/checkpoints/model_epNNNNN.zip --base-seed <new>
python train_sdpdp.py --verify-run        # 28 checks on a short real run
```

- `MaskablePPO` (`gamma` 0.999, `gae_lambda` 0.95, learning rate 3e-4, `n_steps` 1024 per env,
  batch 256, 10 epochs, clip 0.2, network `[256, 256]`) over `SubprocVecEnv`, never
  `DummyVecEnv` (D8). Each env is built inside its worker, because FleetPy objects cannot be
  pickled.
- `VecNormalize` uses the same `gamma` as PPO and `clip_obs = 10`.
- Three non-overlapping seed streams: `[base_seed, env_id]` for episode seeds (D7),
  `[base_seed, 1010, p + 1]` for day order (D10), `[base_seed, 2020]` for the parent's network
  initialisation and action sampling. `seed=` is never passed to `MaskablePPO`, and resuming
  with a `base_seed` the run already used raises.
- **numpy must be imported before torch**: two OpenMP runtimes otherwise collide (spec §8a).
  `train_sdpdp.py`'s import block is ordered for this; do not reorder it.

### Evaluation

```
python -m src.rl_gym.evaluate --validation --greedy --days 2024-06-20 --seeds 2
python -m src.rl_gym.evaluate --validation --rl --checkpoint runs/<name>/checkpoints/model_epNNNNN.zip
python -m src.rl_gym.evaluate --final-test --greedy --rl --checkpoint <chosen.zip> --out <report.txt>
```

Every choice — checkpoint, reward weights, when to stop — is made on the 12 validation days.
The 14 test days are run once, on the one chosen checkpoint, at the very end.

---

## Data and artifacts

| Path | What |
|---|---|
| `data/demand/example_demand/matched/example_network/corpus/` | 101 training days |
| `…/corpus_val/` | 12 validation days |
| `…/corpus_eval/` | 14 test days |
| `…/corpus_excluded/` | 5 calendar-excluded days, never read |
| `data/rl_gym/example_network/corpus_manifest.json` | training-day hashes, weekday vocabulary, `corpus_hash`, `baseline_hash` |
| `data/rl_gym/example_network/arrival_baseline.csv` | 4,482 rows: 3 weekdays × 747 minutes × 2 windows |
| `data/rl_gym/example_network/h3_neighbours.csv` | the H3 zone table |
| `docs/RL_GYM_EVAL_SEEDS.csv` | 10 evaluation seeds for each of the 26 held-out days |
| `docs/RL_GYM_HELDOUT_SPLIT.csv` | which held-out day is validation and which is test |
| `docs/baseline_user_stats.csv`, `docs/baseline_op_stats.csv` | the byte-gate references (they predate the post-Phase-2 simulator changes) |

Run outputs go to `runs/` and `studies/example_study/results/`; git tracks neither. A pinned
reference lives only as long as its run folder does, so keep the folder.

---

## Touch points in FleetPy

The RL plumbing edits four FleetPy files, 20 lines in all, each marked `# RL-GYM:` (list them
with `grep -rn "RL-GYM:" src --include=*.py | grep -v src/rl_gym`).

| File | What |
|---|---|
| `src/misc/globals.py` | `G_SKIP_OUTPUT`, `G_RL_MODE` |
| `src/FleetSimulationBase.py` | output guards; once-per-process logging in RL mode (`sim_env_rl.py` makes it once per episode when output is on, without editing this file) |
| `src/demand/demand.py` | two optional callbacks, the user-stats write guard |
| `src/fleetctrl/FleetControlBase.py` | dyn-atts write guard |

Registration goes through FleetPy's `dev/` extension hook, `dev/misc/init_modules.py`, so no
core registry was edited.

Two guards are counter-intuitive and deliberate. `record_remaining_assignments` is **not**
guarded despite its name: it advances the simulation past `end_time` and fires pickup callbacks.
Neither is `record_remaining_users`: it is the end-of-day `record_user` sweep, and its only write
is already covered.
