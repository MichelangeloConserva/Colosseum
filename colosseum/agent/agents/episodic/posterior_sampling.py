from typing import Any, Callable, Dict, Union

import gin
import numpy as np
from ray import tune

from colosseum.dynamic_programming import episodic_value_iteration
from colosseum.dynamic_programming.utils import get_policy_from_q_values
from colosseum.agent.actors import QValuesActor
from colosseum.agent.agents.base import BaseAgent
from colosseum.agent.mdp_models.bayesian_model import BayesianMDPModel
from colosseum.agent.mdp_models.bayesian_models import (
    RewardsConjugateModel,
    TransitionsConjugateModel,
)
from colosseum.utils.acme.specs import MDPSpec


@gin.configurable
class PSRLEpisodic(BaseAgent):
    @staticmethod
    def produce_gin_file_from_hyperparameters(
        hyperparameters: Dict[str, Any], index: int = 0
    ):
        return (
            "from colosseum.agent.mdp_models import bayesian_models\n"
            f"prms_{index}/PSRLEpisodic.reward_prior_model = %bayesian_models.RewardsConjugateModel.N_NIG\n"
            f"prms_{index}/PSRLEpisodic.transitions_prior_model = %bayesian_models.TransitionsConjugateModel.M_DIR\n"
            f"prms_{index}/PSRLEpisodic.rewards_prior_prms = [{hyperparameters['prior_mean']}, 1, 1, 1]\n"
            f"prms_{index}/PSRLEpisodic.transitions_prior_prms = [{hyperparameters['transition_prior']}]"
        )

    @staticmethod
    def is_episodic() -> bool:
        return True

    @staticmethod
    def get_hyperparameters_search_spaces() -> Dict[str, tune.sample.Domain]:
        return dict(
            prior_mean=tune.uniform(0.001, 2.0), transition_prior=tune.uniform(0.001, 2)
        )

    @staticmethod
    def get_agent_instance_from_hyperparameters(
        seed: int,
        optimization_horizon: int,
        mdp_spec: MDPSpec,
        hyperparameters: Dict[str, Any],
    ) -> "BaseAgent":
        return PSRLEpisodic(
            environment_spec=mdp_spec,
            seed=seed,
            optimization_horizon=optimization_horizon,
            reward_prior_model=RewardsConjugateModel.N_NIG,
            transitions_prior_model=TransitionsConjugateModel.M_DIR,
            rewards_prior_prms=[hyperparameters["prior_mean"], 1, 1, 1],
            transitions_prior_prms=[hyperparameters["transition_prior"]],
        )

    @property
    def current_optimal_stochastic_policy(self) -> np.ndarray:
        T_map, R_map = self._mdp_model.get_map_estimate()
        Q, _ = episodic_value_iteration(self._time_horizon, T_map, R_map)
        return get_policy_from_q_values(Q, True)

    def __init__(
        self,
        seed: int,
        environment_spec: MDPSpec,
        optimization_horizon: int,
        # MDP model hyperparameters
        reward_prior_model: RewardsConjugateModel = None,
        transitions_prior_model: TransitionsConjugateModel = None,
        rewards_prior_prms=None,
        transitions_prior_prms=None,
        # Actor hyperparameters
        epsilon_greedy: Union[float, Callable] = None,
        boltzmann_temperature: Union[float, Callable] = None,
    ):
        mdp_model = BayesianMDPModel(
            seed,
            environment_spec,
            reward_prior_model=reward_prior_model,
            transitions_prior_model=transitions_prior_model,
            rewards_prior_prms=rewards_prior_prms,
            transitions_prior_prms=transitions_prior_prms,
        )
        actor = QValuesActor(
            seed, environment_spec, epsilon_greedy, boltzmann_temperature
        )

        super(PSRLEpisodic, self).__init__(
            seed,
            environment_spec,
            mdp_model,
            actor,
            optimization_horizon,
        )

    def episode_end_update(self):
        Q, _ = episodic_value_iteration(self._time_horizon, *self._mdp_model.sample())
        self._actor.set_q_values(Q)

    def before_start_interacting(self):
        self.episode_end_update()
