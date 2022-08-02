import abc
import random
from typing import TYPE_CHECKING

import dm_env
import numpy as np

from colosseum.utils.acme.specs import MDPSpec

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE, OBSERVATION_TYPE


class BaseActor(abc.ABC):
    @abc.abstractmethod
    def __init__(self, seed: int, environment_spec: MDPSpec):
        """
        Parameters
        ----------
        seed : int
            is the random seed.
        environment_spec : MDPSpec
            provides the full specification of the MDP.
        """
        self._environment_spec = environment_spec
        self._seed = seed

        self._rng = np.random.RandomState(seed)
        self._rng_fast = random.Random(seed)

    @abc.abstractmethod
    def select_action(
        self, ts: dm_env.TimeStep, time_step: int
    ) -> "ACTION_TYPE":
        """

        Parameters
        ----------
        ts : dm_env.TimeStep
            the observation for which the agent is required to calculate the next action.
        time_step : int
            the current time step of the environment. In the episodic case, this refers to the in-episode time step,
            whereas in the continuous case this refers to the number of previous interactions.

        Returns
        -------
        action : ACTION_TYPE
            the action that the agent suggests to take given the observation and the time step.
        """
