# `src/rl_gym` — SDPDP Gymnasium environment

A Gymnasium environment that exposes FleetPy's ride-pooling **assignment decision** to a
reinforcement-learning policy. The policy replaces the greedy `argmin(delta_cfv)` rule in
`PoolingIRSOnly` that currently chooses which vehicle serves each incoming request.

- **One gym step** = one request for which the operator has an actual choice.
- **One episode** = one full simulation day.

Specification: `docs/SDPDP_GYM_SPEC_v2.md`. Implementation history and findings:
`docs/RL_GYM_PROGRESS.md`. Measurements: `docs/RL_GYM_PHASE1_RESULTS.md`.

Phase 1 (plumbing) is complete. Phase 2 (full observation, reward weights, training) is
specified at topic level in §8 of the spec.

---

## The problem this package solves

`ImmediateDecisionsSimulation.run()` owns the clock. It loops over time steps, calls
`step()`, which calls `user_request()` — and that is where the assignment decision happens,
three frames deep inside a loop nothing outside can pause.

Gymnasium requires the opposite: the agent owns the loop and calls `env.step(action)`.

This package inverts control with a **generator**. FleetPy's call stack suspends at a `yield`
and resumes when the env sends an action back. One flow of control, taking turns — no
threads, no queues, no possibility of deadlock or interleaving.

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

## Modules

Listed in the order they make sense to read, which is the order a decision flows through
them — not alphabetical.

### `fleetctrl_rl.py` — the decision point

`RLPoolingIRSOnly(PoolingInsertionHeuristicOnly)` splits `user_request` either side of the
line the RL replaces.

| | |
|---|---|
| `PendingDecision` | carries `prq`, `candidates`, `sim_time`, `rid_struct`, elapsed CPU time across the gap where the policy runs |
| `build_assignment_context(rq, sim_time)` | everything up to the choice; returns `None` when there was nothing to decide |
| `commit_assignment_choice(pending, choice, sim_time)` | `choice` is a candidate index or `None` to reject |
| `user_request(rq, sim_time)` | greedy fallback — build, then commit with choice 0 |
| `set_reward_tracker(tracker)` | optional; `None` by default so non-RL scenarios are unaffected |

Three cases return `None` and generate no gym step: origin equals destination, the
reservation branch, and an empty candidate list. Each reproduces the parent's behaviour
exactly, including `_create_rejection` and the `G_FCTRL_CT_RQU` bookkeeping.

The class knows nothing about `K` or the action space. It receives a semantic choice, never
a raw action.

### `sim_env_rl.py` — the generator

`RLImmediateDecisionsSimulation(ImmediateDecisionsSimulation)` mirrors the parent's `step()`
and `run()` as generators.

- `step_generator(sim_time)` — mirrors `step()` line for line, with `choice = yield ctx`
  between build and commit
- `run_generator()` — mirrors the `run()` loop, delegating with `yield from`

`yield from` forwards yields outward and sends inward, so a `.send()` on `run_generator`
reaches the `yield` buried in `step_generator`'s request loop.

Two things to know. `_rl_operator` raises if paired with a stock operator rather than
falling back silently — a silent fallback would run greedy while reporting that it is
learning. And **finalisation sits outside the `try/finally`**, so an abandoned episode does
not run `record_remaining_assignments`, which would advance the simulation up to four hours
past `end_time` on an env being torn down.

These are mirrors. If either parent method changes, they must change with it; the
byte-for-byte tests are what catch a drift.

### `spaces.py` — truncation and masking

Five pure functions, no state. `K` is passed in.

| | |
|---|---|
| `make_action_space(k_max)` | `Discrete(K + 1)` |
| `reject_action(k_max)` | `K` |
| `truncate_candidates(candidates, k_max)` | prefix of at most `K`, as a new list |
| `candidate_slot_validity(n, k_max)` | `bool[K]` |
| `build_action_mask(n, k_max)` | `bool[K + 1]`, reject slot always `True` |

**Never sorts.** `insertion_with_heuristics` returns candidates already sorted ascending by
`delta_cfv` via a stable sort, so element 0 is exactly what stock's `min()` selects. The
upstream tie-break is proximity to the request origin, from a backwards Dijkstra — re-sorting
with any other tie-break would pick a different vehicle on ties and break the byte gate.

`candidate_slot_validity` is the **single source of truth** for which slots are occupied.
The action mask is that array plus one element; `CandidateObserver` derives `is_valid` from
the same call. Two independent computations could drift with nothing to raise.

### `observers.py` — the observation

`AbstractObserver` requires `observe(fleetpy_module, ctx) -> dict` and an `output_length`
property. `observe_vector` flattens and asserts the length matches — which is what makes
"the env sums lengths rather than hardcoding a shape" safe.

Phase 1 emits `3 * K + 4` features, 28 at `K = 8`:

- `CandidateObserver` — per slot: `is_valid`, `signed_log1p(delta_cfv)`, offered wait over
  `op_max_wait_time`
- `GlobalStateObserver` — `sin`/`cos` of time-of-day, episode progress, fraction of active
  vehicles idle

Two details worth knowing. Real `delta_cfv` values are **negative**, because the objective
penalises unserved requests, so a more negative delta is a larger improvement. And vehicle
status is compared with `VRL_STATES.OUT_OF_SERVICE`, never `== 5` — `VRL_STATES` is a plain
`Enum` with tuple values, so the three upstream `== 5` comparisons never fire.

All normalisers are read from `scenario_parameters`; no literals.

### `policies.py` — the scripted greedy policy

One function, `greedy_action(obs, mask, k_max)`. Slices `delta_cfv` from `obs[K:2K]`,
restricts to valid slots, returns the argmin.

Used by the P1.10 exit gate, and intended for P2.7's baseline comparison so that the greedy
policy tested and the greedy policy compared against cannot diverge.

It selects by value rather than returning 0, even though the sorted list makes 0 always
correct. Hardcoding would make the function unfalsifiable — it would return the right answer
while reading the wrong block of the observation.

### `reward.py` — accumulation and classification

`RewardTracker` has four inputs, two of which only label:

| | |
|---|---|
| `on_pickup(rid, rq)` | from `Demand.record_boarding` — charges the pickup term |
| `on_exit(rid, rq)` | from `Demand.record_user` — charges every terminal outcome |
| `note_rejection(rid)` | from `commit_assignment_choice` — labels only |
| `note_no_candidates(rid)` | from `build_assignment_context` — labels only |

All charging happens in `on_exit`, once per rid. `on_exit` is idempotent because
`record_alighting_start` and `record_remaining_users` can both fire for a rider mid-alighting
at day's end.

`on_pickup` is **last-write-wins**: `record_boarding` fires twice for a rider co-located with
a no-show, because the no-show cleanup replans the vehicle. FleetPy overwrites `pu_time` and
the output file records the last value, so a repeat charges the delta, leaving a net of one
pickup priced at the final time.

`classify` implements the eight-branch ladder, first match wins, **no fallback to rider
decline**. The order is the design:

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

Served sits third because successful alighting also routes through `record_user` — a
fallback to decline would charge every completed trip a penalty. `no_show` comes after
`diffusion_cancelled` because it is a rider attribute read from the demand file, not an
outcome. Branch 7 uses `chosen_operator_id is not None`, since operator 0 is falsy.

Branch 8 is a diagnostic: a non-zero count means a `record_user` path was missed, and the
exit gate asserts it is zero.

`flush()` returns what accumulated since the last call and resets. Nothing computes a total —
events add themselves as they fire, and the window boundary is enforced by the simulation
being frozen while the agent decides.

### `gym_env.py` — the environment

`SDPDPAssignmentEnv(gymnasium.Env)`. Owns `K`, owns the action translation, and is the only
place either appears.

```python
env = SDPDPAssignmentEnv({
    "constant_cfg_path": ".../constant_config_depot_cali_sc_1.csv",
    "scenario_cfg_path": ".../example_depot_cali_sc_1.csv",
    "K": 8,
    "skip_output": True,
    "base_seed": 20260913,     # omit to keep the scenario row's own seed
    "env_id": 0,
})
```

| key | default | meaning |
|---|---|---|
| `constant_cfg_path`, `scenario_cfg_path` | required | FleetPy configs |
| `scenario_row` | 0 | which row of the scenario CSV |
| `K` | 8 | candidate slots; action space is `K + 1` |
| `skip_output` | `True` | suppress all file writes |
| `reward_weights` | `None` | falls back to `DEFAULT_REWARD_WEIGHTS` |
| `env_id` | 0 | worker index; seeds and output names derive from it |
| `base_seed` | `None` | **absent means no per-episode reseeding** |
| `log_level` | `"warning"` | |
| `scenario_pool` | `None` | accepted, unused until Phase 2 |

`reset()` builds a **fresh** simulation each episode — 0.46 s, about 1.2% of an episode.
Nothing is cached across resets, which is what keeps stochastic travel-time state, dynamic
network state and RNG state from leaking between episodes.

`base_seed` absent is the reproducibility-safe default and the training-unsafe one. The
byte-for-byte gates require the scenario's own seed; training requires reseeding.
`train_sdpdp.py` raises if it is missing.

`action_masks()` returns the mask stored alongside the current observation. It must not
advance the simulation — `MaskablePPO` calls it between `step()` and the next action.

Masked actions **assert**. `check_env` samples uniformly and ignores masks, so the tests run
it through a shim rather than weakening the env.

---

## Using it

```python
obs, info = env.reset()
terminated = False
while not terminated:
    action = greedy_action(obs, env.action_masks(), K)     # or model.predict(...)
    obs, reward, terminated, truncated, info = env.step(action)
```

On termination: `terminated=True`, `truncated=False`, a zeros observation, a reject-only
mask, and `episode_summary()` in `info`.

---

## Touch points in FleetPy

Four files, ~140 lines, every edit marked `# RL-GYM:`.

| | |
|---|---|
| `src/misc/globals.py` | `G_SKIP_OUTPUT`, `G_RL_MODE` |
| `src/FleetSimulationBase.py` | output guards, once-per-process logging in RL mode |
| `src/demand/demand.py` | two optional callbacks, the user-stats write guard |
| `src/fleetctrl/FleetControlBase.py` | dyn-atts write guard |

Registration goes through FleetPy's `dev/` extension hook — `dev/misc/init_modules.py` — so
no core registry was edited.

Two guards are counter-intuitive and deliberate. `record_remaining_assignments` is **not**
guarded despite its name: it advances the simulation past `end_time` and fires pickup
callbacks. Neither is `record_remaining_users`: it is the end-of-day `record_user` sweep, and
its only write is already covered.

---

## Reference measurements

Seed 42, `2024-06-20.csv`, 445 requests.

| | |
|---|---|
| decisions | 436 (9 empty-candidate, 0 reservation) |
| episode | 38.0 s, 11.5 steps/s |
| candidates per decision | min 1, median 5, mean 4.60, max 11; 13 over `K = 8` |
| outcomes | 288 served, 118 declined, 28 cancelled, 9 no-candidates, 2 no-show, 0 unclassified |
| simulated gap between decisions | median 60 s, mean 99 s, max 660 s; 119 of 435 are zero |
| tail past `end_time` | 0 s — last dropoff 69,805 against 70,000 |
| workers | 0.40 GB each; 4 workers cost 1.6× the wall clock of 1 |
