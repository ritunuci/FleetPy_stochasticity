"""RL gym environment for the SDPDP assignment decision of PoolingIRSOnly.

See docs/SDPDP_GYM_SPEC_v2.md. Modules are imported lazily by
src/misc/init_modules.py through the dev extension package, so this file stays
empty of imports: pulling in fleetctrl_rl or sim_env_rl here would make every
`import src.rl_gym` drag in FleetPy's fleet control stack.
"""
