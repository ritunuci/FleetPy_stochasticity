"""Offline builders for the RL gym's frozen data artifacts. Not imported at runtime.

Each module here produces a file under `data/rl_gym/` that the env then reads: P2.5's H3
neighbour table, P2.6's arrival baseline and corpus manifest. They run once, by hand, and their
output is committed and reviewed.

Nothing in `src/rl_gym/` may import from this package -- the dependency runs one way, so a
builder can never be pulled into a training run. The runtime readers live beside the env
(`zones.py` for P2.5's table) and depend only on the committed file.
"""
