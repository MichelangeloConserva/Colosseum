from typing import TYPE_CHECKING, Tuple

import dm_env
import numpy as np

from colosseum.agent.mdp_models.base import BaseMDPModel
from colosseum.agent.mdp_models.bayesian_models import RewardsConjugateModel
from colosseum.agent.mdp_models.bayesian_models import TransitionsConjugateModel
from colosseum.utils.acme.specs import MDPSpec

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE


class BayesianMDPModel(BaseMDPModel):
    """
    The `BayesianMDPModel` is the wrapper for Bayesian tabular MDP models.
    """

    def __init__(
        self,
        seed: int,
        mdp_specs: MDPSpec,
        reward_prior_model: RewardsConjugateModel = None,
        transitions_prior_model: TransitionsConjugateModel = None,
        rewards_prior_prms=None,
        transitions_prior_prms=None,
    ):
        """
        Parameters
        ----------
        seed : int
            The random seed.
        mdp_specs : MDPSpec
            The full specification of the MDP.
        reward_prior_model : RewardsConjugateModel, optional
            The reward priors.
        transitions_prior_model : TransitionsConjugateModel, optional
            The transitions priors.
        rewards_prior_prms : Any
            The reward prior parameters.
        transitions_prior_prms : Any
            The transitions prior parameters.
        """

        super(BayesianMDPModel, self).__init__(seed, mdp_specs)

        if reward_prior_model is None:
            reward_prior_model = RewardsConjugateModel.N_NIG
            rewards_prior_prms = [self._reward_range[1], 1, 1, 1]
        if transitions_prior_model is None:
            transitions_prior_model = TransitionsConjugateModel.M_DIR
            transitions_prior_prms = [1.0 / self._n_states]

        self._rewards_model = reward_prior_model.get_class()(
            self._n_states, self._n_actions, rewards_prior_prms, seed
        )
        self._transitions_model = transitions_prior_model.get_class()(
            self._n_states, self._n_actions, transitions_prior_prms, seed
        )

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns an MDP model in terms of transitions probabilities matrix and rewards matrix.
        """
        return self._transitions_model.sample(), self._rewards_model.sample()

    def sample_T(self):
        return self._transitions_model.sample()

    def sample_R(self):
        return self._rewards_model.sample()

    def get_map_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self._transitions_model.get_map_estimate(),
            self._rewards_model.get_map_estimate(),
        )

    def step_update(
        self,
        ts_t: dm_env.TimeStep,
        a_t: "ACTION_TYPE",
        ts_tp1: dm_env.TimeStep,
        time: int,
    ):
        self._rewards_model.update_single_transition(
            ts_t.observation, a_t, ts_tp1.reward
        )
        if not ts_tp1.last():
            self._transitions_model.update_single_transition(
                ts_t.observation, a_t, ts_tp1.observation
            )
