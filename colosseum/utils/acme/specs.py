from typing import TYPE_CHECKING, Any, NamedTuple, Tuple

import numpy as np
from dm_env import specs

from colosseum.emission_maps import EmissionMap

if TYPE_CHECKING:
    from colosseum.mdp import BaseMDP

Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray


class MDPSpec(NamedTuple):
    """Full specification of the domains used by a given environment."""

    observations: Any
    actions: Any
    rewards: Any
    discounts: Any
    time_horizon: Any
    rewards_range: Tuple[float, float]
    emission_map: "EmissionMap"
    n_states: int


def make_mdp_spec(mdp: "BaseMDP") -> MDPSpec:
    """Returns an `MDPSpec` describing values used by an environment."""
    return MDPSpec(
        observations=mdp.observation_spec(),
        actions=mdp.action_spec(),
        rewards=mdp.reward_spec(),
        discounts=mdp.discount_spec(),
        time_horizon=mdp.H if mdp.is_episodic() else np.inf,
        rewards_range=mdp.rewards_range,
        emission_map=mdp.emission_map,
        n_states=mdp.n_states,
    )
