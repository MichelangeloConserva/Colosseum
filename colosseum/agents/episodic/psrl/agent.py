import gin
import toolz
from ray import tune

from colosseum.agents.base import FiniteHorizonBaseActor, ValueBasedAgent
from colosseum.agents.bayes_tools.conjugate_rewards import PRIOR_TYPE as R_PRIOR_TYPE
from colosseum.agents.bayes_tools.conjugate_rewards import RewardsConjugateModel
from colosseum.agents.bayes_tools.conjugate_transitions import (
    PRIOR_TYPE as T_PRIOR_TYPE,
)
from colosseum.agents.bayes_tools.conjugate_transitions import TransitionsConjugateModel
from colosseum.dp.episodic import value_iteration
from colosseum.utils.acme.specs import EnvironmentSpec
from colosseum.utils.random_vars import state_occurencens_to_counts


@gin.configurable
class PSRLEpisodic(FiniteHorizonBaseActor, ValueBasedAgent):
    """
    The Posterior Sampling for Reinforcement Learning agent for finite horizon MDP.
    """

    search_space = {"a": tune.uniform(0.0001, 1.1), "b": tune.uniform(0.1, 2)}

    def __init__(
        self,
        environment_spec: EnvironmentSpec,
        seed: int,
        H: int,
        r_max: float,
        T: int,
        reward_prior_model: RewardsConjugateModel,
        transitions_prior_model: TransitionsConjugateModel,
        rewards_prior_prms: R_PRIOR_TYPE,
        transitions_prior_prms: T_PRIOR_TYPE,
    ):
        """

        Parameters
        ----------
        environment_spec : EnvironmentSpec
            encodes the specification of the MDP in terms of number of states and actions.
        seed : int
            the seed for the agent
        H : int
            the MDP horizon.
        r_max : float
            the maximum reward that the MDP can yield.
        T : int
            the optimization horizon.
        reward_prior_model : RewardsConjugateModel
            check the RewardsConjugateModel class to see which Bayesian conjugate models are available.
        transitions_prior_model : TransitionsConjugateModel
            check the TransitionsConjugateModel class to see which Bayesian conjugate models are available.
        rewards_prior_prms : Union[List, Dict[Tuple[int, int], List]]
            the reward prior can either be a list of hyperparameters that are set identical for each state-action pair
            or it can be a dictionary with the state action pair as key and a list of hyperparameters as value.
        transitions_prior_prms : T_PRIOR_TYPE
            the transition prior can either be a list of hyperparameters that are set identical for each state-action
            pair, or it can be a dictionary with the state action pair as key and a list of hyperparameters as value.
        """

        super(PSRLEpisodic, self).__init__(environment_spec, seed, H, r_max, T)

        # Instantiate Bayesian models
        self.rewards_model = reward_prior_model.get_class()(
            self.num_states, self.num_actions, rewards_prior_prms, seed
        )
        self.transitions_model = transitions_prior_model.get_class()(
            self.num_states, self.num_actions, transitions_prior_prms, seed
        )

        # First policy optimistic_sampling
        self.update_policy()

    def update_policy(self):
        """
        Samples a single MDP from the posterior and solve for optimal Q values.
        """
        # Sample the MDP
        R_samp = self.rewards_model.sample_MDP()
        P_samp = self.transitions_model.sample_MDP()

        # Solve the MDP via value iteration
        self.Q, self.V = value_iteration(self.H, P_samp, R_samp)

    def _before_new_episode(self):
        self.update_policy()

    def update_models(self):
        """
        Updates the Bayesian MDP model of the agent using the Bayes rule and the latest gather data.
        """
        self.transitions_model.update(
            toolz.valmap(
                lambda x: state_occurencens_to_counts(x, self.num_states),
                self.episode_transition_data,
            )
        )
        self.rewards_model.update(self.episode_reward_data)
