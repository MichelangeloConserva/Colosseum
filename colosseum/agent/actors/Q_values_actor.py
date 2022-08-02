from typing import TYPE_CHECKING, Callable, Union

import dm_env
import numpy as np

from colosseum.agent.actors import BaseActor
from colosseum.utils.acme.specs import MDPSpec

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE, OBSERVATION_TYPE


class QValuesActor(BaseActor):
    def __init__(
        self,
        seed: int,
        environment_spec: MDPSpec,
        epsilon_greedy: Union[float, Callable] = None,
        boltzmann_temperature: Union[float, Callable] = None,
    ):
        """
        returns an actor that implements the standard action selection techniques based on q values.

        Parameters
        ----------
        epsilon_greedy : Union[float, Callable], optional
            is the probability of selecting an action at random. It can be provided as a float or as a function of the
            total number of interactions.
        boltzmann_temperature : Union[float, Callable], optional
            is the parameter that control the Boltzmann exploration. It can be provided as a float or as a function of
            the total number of interactions.
        """
        super(QValuesActor, self).__init__(seed, environment_spec)

        if epsilon_greedy is not None:
            if type(epsilon_greedy) == float:
                epsilon_greedy = lambda t: epsilon_greedy
        if boltzmann_temperature is not None:
            if type(boltzmann_temperature) == float:
                boltzmann_temperature = lambda t: boltzmann_temperature

        self._epsilon_greedy = epsilon_greedy
        self._boltzmann_temperature = boltzmann_temperature
        self._total_interactions = 0
        self._q_values = None
        self._n_states = self._environment_spec.observations.num_values
        self._n_actions = self._environment_spec.actions.num_values

    def set_q_values(self, Q: np.ndarray):
        self._q_values = Q
        self._episodic = Q.ndim == 3

    def select_action(
        self, ts: dm_env.TimeStep, time_step: int
    ) -> "ACTION_TYPE":
        assert self._q_values is not None, "The q values have not been initialized."

        self._total_interactions += 1

        # Epsilon greedy policy
        if self._epsilon_greedy is not None:
            if self._rng_fast.random() < self._epsilon_greedy(self._total_interactions):
                return self._rng_fast.randint(0, self._n_actions - 1)
        q = self._q_values[(time_step, ts.observation) if self._episodic else ts.observation]

        # Boltzmann exploration
        if self._boltzmann_temperature is not None:
            q = np.exp(self._boltzmann_temperature(self._total_interactions) * q)
            return self._rng.choice(
                range(self._n_actions), replace=False, p=q / q.sum()
            )

        # Greedy selection
        return self._rng.choice(np.where(q == q.max())[0])
