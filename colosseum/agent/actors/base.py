import abc
import random
from typing import TYPE_CHECKING

import dm_env
import numpy as np

from colosseum.utils.acme.specs import MDPSpec

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE


class BaseActor(abc.ABC):
    """
    The `BaseActor` class is the abstract class for the actor component of a reinforcement learning agent, which
    handles the interactions with the MDPs.
    """

    @abc.abstractmethod
    def __init__(self, seed: int, mdp_specs: MDPSpec):
        """
        Parameters
        ----------
        seed : int
            The random seed.
        mdp_specs : MDPSpec
            The full specification of the MDP.
        """
        self._mdp_spec = mdp_specs
        self._seed = seed

        self._rng = np.random.RandomState(seed)
        self._rng_fast = random.Random(seed)

    @abc.abstractmethod
    def select_action(self, ts: dm_env.TimeStep, time: int) -> "ACTION_TYPE":
        """
        Parameters
        ----------
        ts : dm_env.TimeStep
            The TimeStep for which the agent is required to calculate the next action.
        time : int
            The current time of the environment. In the episodic case, this refers to the in-episode time, whereas in
            the continuous case this refers to the total number of previous interactions.

        Returns
        -------
        action : ACTION_TYPE
            The action that the agent suggests to take given the observation and the time step.
        """
