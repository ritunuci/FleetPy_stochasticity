"""Diagnostic drivers for the RL gym. Not imported by the env, the trackers or training.

Each module here is a standalone entry point that drives `SDPDPAssignmentEnv` from outside and
writes a report under `docs/`. Nothing in `src/rl_gym/` may import from this package -- the
dependency runs one way, so a broken tool can never reach a training run or a gate.
"""
