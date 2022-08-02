import abc
from typing import Any, Dict

import numpy as np
from ray import tune

from colosseum.agent.actors import RandomActor
from colosseum.agent.agents.base import BaseAgent
from colosseum.utils.acme.specs import DiscreteArray, MDPSpec


class RandomAgent(BaseAgent, abc.ABC):
    @staticmethod
    def get_hyperparameters_search_spaces() -> Dict[str, tune.sample.Domain]:
        pass

    @property
    def current_optimal_stochastic_policy(self) -> np.ndarray:
        return self._policy

    def __init__(self, seed: int, environment_spec: MDPSpec):
        super(RandomAgent, self).__init__(
            seed,
            environment_spec,
            mdp_model=None,
            actor=RandomActor(seed, environment_spec),
            optimization_horizon=0,
        )

        if type(self._environment_spec.observations) == DiscreteArray:
            if type(self._environment_spec.actions) == DiscreteArray:
                n_s = self._environment_spec.observations.num_values
                n_a = self._environment_spec.actions.num_values

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
    def get_agent_instance_from_hyperparameters(
        seed: int,
        optimization_horizon: int,
        mdp_specs: MDPSpec,
        hyperparameters: Dict[str, Any],
    ) -> "BaseAgent":
        return RandomAgentEpisodic(seed, mdp_specs)

    @staticmethod
    def is_episodic() -> bool:
        return True


class RandomAgentContinuous(RandomAgent):
    @staticmethod
    def get_agent_instance_from_hyperparameters(
        seed: int,
        optimization_horizon: int,
        mdp_specs: MDPSpec,
        hyperparameters: Dict[str, Any],
    ) -> "BaseAgent":
        return RandomAgentContinuous(seed, mdp_specs)

    @staticmethod
    def is_episodic() -> bool:
        return False
