from typing import Any, Callable, Dict, Union

import gin
import numpy as np

from ray import tune

from colosseum.agent.actors import QValuesActor
from colosseum.agent.agents.base import BaseAgent
from colosseum.agent.mdp_models.bayesian_model import BayesianMDPModel
from colosseum.agent.mdp_models.bayesian_models import RewardsConjugateModel
from colosseum.agent.mdp_models.bayesian_models import TransitionsConjugateModel
from colosseum.dynamic_programming import episodic_value_iteration
from colosseum.dynamic_programming.utils import get_policy_from_q_values
from colosseum.emission_maps import EmissionMap
from colosseum.utils.acme.specs import MDPSpec


@gin.configurable
class PSRLEpisodic(BaseAgent):
    """
    The posterior sampling for reinforcement learning algorithm.

    Osband, Ian, Daniel Russo, and Benjamin Van Roy. "(More) efficient reinforcement learning via posterior sampling."
    Advances in Neural Information Processing Systems 26 (2013).
    """

    @staticmethod
    def is_emission_map_accepted(emission_map: "EmissionMap") -> bool:
        return emission_map.is_tabular

    @staticmethod
    def produce_gin_file_from_parameters(
        parameters: Dict[str, Any], index: int = 0
    ):
        return (
            "from colosseum.agent.mdp_models import bayesian_models\n"
            f"prms_{index}/PSRLEpisodic.reward_prior_model = %bayesian_models.RewardsConjugateModel.N_NIG\n"
            f"prms_{index}/PSRLEpisodic.transitions_prior_model = %bayesian_models.TransitionsConjugateModel.M_DIR\n"
            f"prms_{index}/PSRLEpisodic.rewards_prior_prms = [{parameters['prior_mean']}, 1, 1, 1]\n"
            f"prms_{index}/PSRLEpisodic.transitions_prior_prms = [{parameters['transition_prior']}]"
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
    def get_agent_instance_from_parameters(
        seed: int,
        optimization_horizon: int,
        mdp_specs: MDPSpec,
        parameters: Dict[str, Any],
    ) -> "BaseAgent":
        return PSRLEpisodic(
            mdp_specs=mdp_specs,
            seed=seed,
            optimization_horizon=optimization_horizon,
            reward_prior_model=RewardsConjugateModel.N_NIG,
            transitions_prior_model=TransitionsConjugateModel.M_DIR,
            rewards_prior_prms=[parameters["prior_mean"], 1, 1, 1],
            transitions_prior_prms=[parameters["transition_prior"]],
        )

    @property
    def current_optimal_stochastic_policy(self) -> np.ndarray:
        T_map, R_map = self._mdp_model.get_map_estimate()
        Q, _ = episodic_value_iteration(self._time_horizon, T_map, R_map)
        return get_policy_from_q_values(Q, True)

    def __init__(
        self,
        seed: int,
        mdp_specs: MDPSpec,
        optimization_horizon: int,
        # MDP model parameters
        reward_prior_model: RewardsConjugateModel = None,
        transitions_prior_model: TransitionsConjugateModel = None,
        rewards_prior_prms=None,
        transitions_prior_prms=None,
        # Actor parameters
        epsilon_greedy: Union[float, Callable] = None,
        boltzmann_temperature: Union[float, Callable] = None,
    ):
        """

        Parameters
        ----------
        seed : int
            The random seed.
        mdp_specs : MDPSpec
            The full specification of the MDP.
        optimization_horizon : int
            The total number of interactions that the agent is expected to have with the MDP.
        reward_prior_model : RewardsConjugateModel, optional
            The reward priors.
        transitions_prior_model : TransitionsConjugateModel, optional
            The transitions priors.
        rewards_prior_prms : Any
            The reward prior parameters.
        transitions_prior_prms : Any
            The transitions prior parameters.
        epsilon_greedy : Union[float, Callable], optional
            The probability of selecting an action at random. It can be provided as a float or as a function of the
            total number of interactions. By default, the probability is set to zero.
        boltzmann_temperature : Union[float, Callable], optional
            The parameter that controls the Boltzmann exploration. It can be provided as a float or as a function of
            the total number of interactions. By default, Boltzmann exploration is disabled.
        """

        mdp_model = BayesianMDPModel(
            seed,
            mdp_specs,
            reward_prior_model=reward_prior_model,
            transitions_prior_model=transitions_prior_model,
            rewards_prior_prms=rewards_prior_prms,
            transitions_prior_prms=transitions_prior_prms,
        )
        actor = QValuesActor(seed, mdp_specs, epsilon_greedy, boltzmann_temperature)

        super(PSRLEpisodic, self).__init__(
            seed,
            mdp_specs,
            mdp_model,
            actor,
            optimization_horizon,
        )

    def episode_end_update(self):
        Q, _ = episodic_value_iteration(self._time_horizon, *self._mdp_model.sample())
        self._actor.set_q_values(Q)

    def before_start_interacting(self):
        self.episode_end_update()
