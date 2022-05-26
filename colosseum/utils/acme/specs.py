from typing import Any, NamedTuple

import dm_env
from dm_env import specs

Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray


class EnvironmentSpec(NamedTuple):
    """Full specification of the domains used by a given environment."""

    # TODO(b/144758674): Use NestedSpec type here.
    observations: Any
    actions: Any
    rewards: Any
    discounts: Any


def make_environment_spec(environment: dm_env.Environment) -> EnvironmentSpec:
    """Returns an `EnvironmentSpec` describing values used by an environment."""
    return EnvironmentSpec(
        observations=environment.observation_spec(),
        actions=environment.action_spec(),
        rewards=environment.reward_spec(),
        discounts=environment.discount_spec(),
    )
