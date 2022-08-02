from typing import Union

from colosseum.agent.actors.base import BaseActor
from colosseum.agent.actors.Q_values_actor import QValuesActor
from colosseum.agent.actors.random import RandomActor

ACTOR_TYPES = Union[BaseActor, QValuesActor]
