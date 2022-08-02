import abc

import dm_env
import numpy as np

from colosseum.mdp import ACTION_TYPE
from colosseum.utils.acme.specs import MDPSpec


class BaseMDPModel(abc.ABC):
    def __init__(self, seed: int, environment_spec: MDPSpec):
        self._seed = seed
        self._n_states = environment_spec.observations.num_values
        self._n_actions = environment_spec.actions.num_values
        self._reward_range = environment_spec.rewards_range
        self._H = environment_spec.time_horizon
        self._rng = np.random.RandomState(seed)

    @abc.abstractmethod
    def step_update(
        self, ts_t: dm_env.TimeStep, a_t: "ACTION_TYPE", ts_tp1: dm_env.TimeStep, h: int
    ):
        """
        updates the MDP model with the _compute_transition information given in input. This must be done withing the
        given maximum time. Note that the implementation of the function must make sure that the limit is respected.
        """
