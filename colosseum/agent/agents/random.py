import abc
from typing import Any, Dict, TYPE_CHECKING

import numpy as np
from ray import tune

from colosseum.agent.actors import RandomActor
from colosseum.agent.agents.base import BaseAgent
from colosseum.utils.acme.specs import DiscreteArray, MDPSpec

if TYPE_CHECKING:
    from colosseum.emission_maps import EmissionMap


class RandomAgent(BaseAgent, abc.ABC):
    """
    The `RandomAgent` implements a uniformly randomly acting agent.
    """

    @staticmethod
    def is_emission_map_accepted(emission_map: "EmissionMap") -> bool:
        return True

    @staticmethod
    def get_hyperparameters_search_spaces() -> Dict[str, tune.sample.Domain]:
        return {}

    @staticmethod
    def produce_gin_file_from_parameters(parameters: Dict[str, Any], index: int = 0):
        return ""

    @property
    def current_optimal_stochastic_policy(self) -> np.ndarray:
        return self._policy

    def __init__(self, seed: int, mdp_specs: MDPSpec):
        """
        Parameters
        ----------
        seed : int
            The random seed.
        mdp_specs : MDPSpec
            The full specification of the MDP.
        """
        super(RandomAgent, self).__init__(
            seed,
            mdp_specs,
            mdp_model=None,
            actor=RandomActor(seed, mdp_specs),
            optimization_horizon=0,
        )

        if type(self._mdp_spec.observations) == DiscreteArray:
            if type(self._mdp_spec.actions) == DiscreteArray:
                n_s = self._mdp_spec.observations.num_values
                n_a = self._mdp_spec.actions.num_values

                self._policy = (
                    np.ones(
                        (n_s, n_a)
                        if self._time_horizon == np.inf
                        else (self._time_horizon, n_s, n_a)
                    )
                    / n_a
                )
        else:
            raise NotImplementedError(
                "The RandomAgent is implemented for discrete MDP only."
            )

    def episode_end_update(self):
        pass

    def before_start_interacting(self):
        pass


class RandomAgentEpisodic(RandomAgent):
    @staticmethod
    def get_agent_instance_from_parameters(
        seed: int,
        optimization_horizon: int,
        mdp_specs: MDPSpec,
        parameters: Dict[str, Any],
    ) -> "BaseAgent":
        return RandomAgentEpisodic(seed, mdp_specs)

    @staticmethod
    def is_episodic() -> bool:
        return True


class RandomAgentContinuous(RandomAgent):
    @staticmethod
    def get_agent_instance_from_parameters(
        seed: int,
        optimization_horizon: int,
        mdp_specs: MDPSpec,
        parameters: Dict[str, Any],
    ) -> "BaseAgent":
        return RandomAgentContinuous(seed, mdp_specs)

    @staticmethod
    def is_episodic() -> bool:
        return False
