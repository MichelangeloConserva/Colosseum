from typing import TYPE_CHECKING, Callable, Union

import dm_env
import numpy as np

from colosseum.agent.actors import BaseActor
from colosseum.utils.acme.specs import MDPSpec

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE


class QValuesActor(BaseActor):
    """
    The `QValuesActor` is an actor component that selects actions that maximises a given q-values estimate. It also
    supports the epsilon-greedy and the Boltzmann action selection strategies to boost exploration.
    """

    def __init__(
        self,
        seed: int,
        mdp_specs: MDPSpec,
        epsilon_greedy: Union[float, Callable[[int], float]] = None,
        boltzmann_temperature: Union[float, Callable[[int], float]] = None,
    ):
        """
        Parameters
        ----------
        seed : int
            The random seed.
        mdp_specs : MDPSpec
            The full specification of the MDP.
        epsilon_greedy : Union[float, Callable], optional
            The probability of selecting an action at random. It can be provided as a float or as a function of the
            total number of interactions. By default, the probability is set to zero.
        boltzmann_temperature : Union[float, Callable], optional
            The parameter that controls the Boltzmann exploration. It can be provided as a float or as a function of
            the total number of interactions. By default, Boltzmann exploration is disabled.
        """
        super(QValuesActor, self).__init__(seed, mdp_specs)

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
        self._n_states = self._mdp_spec.observations.num_values
        self._n_actions = self._mdp_spec.actions.num_values

    def set_q_values(self, Q: np.ndarray):
        """
        updates the q-values estimates of the component with the one given in input.
        Parameters
        ----------
        Q : np.ndarray
            The q-values estimates.
        """
        self._q_values = Q
        self._episodic = Q.ndim == 3

    def select_action(self, ts: dm_env.TimeStep, time: int) -> "ACTION_TYPE":
        assert self._q_values is not None, "The q values have not been initialized."

        self._total_interactions += 1

        # Epsilon greedy policy
        if self._epsilon_greedy is not None:
            if self._rng_fast.random() < self._epsilon_greedy(self._total_interactions):
                return self._rng_fast.randint(0, self._n_actions - 1)

        # Retrieve the q-values
        q = self._q_values[(time, ts.observation) if self._episodic else ts.observation]

        # Boltzmann exploration
        if self._boltzmann_temperature is not None:
            q = np.exp(self._boltzmann_temperature(self._total_interactions) * q)
            return self._rng.choice(
                range(self._n_actions), replace=False, p=q / q.sum()
            )

        # Greedy selection
        return self._rng.choice(np.where(q == q.max())[0])
