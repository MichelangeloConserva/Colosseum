"""
The actor components of reinforcement learning agents handle the interactions with the MDPs.
Their main role is to provide a mapping from a `BaseMDPModel`, which contains the knowledge of the agent regarding the
MDP, to a policy.
"""

from typing import Union

from colosseum.agent.actors.base import BaseActor
from colosseum.agent.actors.Q_values_actor import QValuesActor
from colosseum.agent.actors.random import RandomActor

ACTOR_TYPES = Union[BaseActor, QValuesActor]
