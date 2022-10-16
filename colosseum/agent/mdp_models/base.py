import abc

import dm_env
import numpy as np

from colosseum.mdp import ACTION_TYPE
from colosseum.utils.acme.specs import MDPSpec


class BaseMDPModel(abc.ABC):
    """
    The `BaseMDPModel` is the base class for MDP models.
    """

    def __init__(self, seed: int, mdp_specs: MDPSpec):
        """
        Parameters
        ----------
        seed : int
            The random seed.
        mdp_specs : MDPSpec
            The full specification of the MDP.
        """

        self._seed = seed
        self._n_states = mdp_specs.observations.num_values
        self._n_actions = mdp_specs.actions.num_values
        self._reward_range = mdp_specs.rewards_range
        self._H = mdp_specs.time_horizon
        self._rng = np.random.RandomState(seed)

    @abc.abstractmethod
    def step_update(
        self,
        ts_t: dm_env.TimeStep,
        a_t: "ACTION_TYPE",
        ts_tp1: dm_env.TimeStep,
        time: int,
    ):
        """
        updates the model with the transition in input.

        Parameters
        ----------
        ts_t : dm_env.TimeStep
            The TimeStep at time t.
        a_t : "ACTION_TYPE"
            The action taken by the agent at time t.
        ts_tp1 : dm_env.TimeStep
            The TimeStep at time t + 1.
        time : int
            The current time of the environment. In the episodic case, this refers to the in-episode time, whereas in
            the continuous case this refers to the total number of previous interactions.
        """
