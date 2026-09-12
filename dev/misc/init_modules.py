"""Development-content module registry.

``src/misc/init_modules.py`` imports this module inside a
``try/except ModuleNotFoundError`` and calls every one of the ten ``add_*``
functions below from inside its ``get_src_*()`` getters. Registering RL modules
here keeps ``get_src_fleet_control_modules()`` and
``get_src_simulation_environments()`` unedited.

Two invariants this file must preserve:

1. **No imports, no module-level work.** A ``ModuleNotFoundError`` raised while
   importing this module is swallowed by that ``except`` clause, ``dev_content``
   silently becomes ``None``, and the failure resurfaces much later as
   ``IOError: Fleet control module RLPoolingIRSOnly is invalid!``.
2. **All ten functions stay defined.** The calls live inside the getters, not at
   import time, so a missing one raises ``AttributeError`` partway into a
   simulation rather than at startup.

Dict values are ``(module path, class name)`` tuples, dereferenced lazily by
``load_module``, so targets may name modules that do not exist yet.
"""


# -------------------------------------------------------------------------- #
# RL modules
# ----------

def add_dev_simulation_environments():
    """Simulation environments, merged into ``get_src_simulation_environments()``."""
    return {
        "RLImmediateDecisionsSimulation": ("src.rl_gym.sim_env_rl", "RLImmediateDecisionsSimulation"),
    }


def add_fleet_control_modules():
    """Fleet control modules, merged into ``get_src_fleet_control_modules()``."""
    return {
        "RLPoolingIRSOnly": ("src.rl_gym.fleetctrl_rl", "RLPoolingIRSOnly"),
    }


# -------------------------------------------------------------------------- #
# Hooks required by the getters but unused by the RL work
# -------------------------------------------------------

def add_dev_routing_engines():
    return {}


def add_request_models():
    return {}


def add_repositioning_modules():
    return {}


def add_charging_strategy_modules():
    return {}


def add_dynamic_pricing_strategy_modules():
    return {}


def add_dynamic_fleetsizing_strategy_modules():
    return {}


def add_reservation_strategy_modules():
    return {}


def add_ride_pooling_batch_optimizer_modules():
    return {}
