from typing import TYPE_CHECKING, Any, Callable, Dict, Union

import dm_env
import gin
import numpy as np
from ray import tune

from colosseum.agent.actors import QValuesActor
from colosseum.agent.agents.base import BaseAgent
from colosseum.agent.mdp_models.bayesian_model import BayesianMDPModel
from colosseum.agent.mdp_models.bayesian_models import RewardsConjugateModel
from colosseum.agent.mdp_models.bayesian_models import TransitionsConjugateModel
from colosseum.dynamic_programming import discounted_value_iteration
from colosseum.dynamic_programming.utils import get_policy_from_q_values
from colosseum.emission_maps import EmissionMap
from colosseum.utils.acme.specs import MDPSpec

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE


def get_psi(n_states, n_actions, T, p) -> float:
    r"""
    computes the theoretical value of the :math:`\psi` parameter.

    Parameters
    ----------
    n_states : int
        The number of states.
    n_actions : int
        The number of actions.
    T : int
        The optimization horizon.
    p : float
        The probability of failure.

    Returns
    -------
    float
        The theoretical value of the :math:`\psi` parameter.
    """
    return n_states * np.log(n_states * n_actions / p)


def get_omega(n_states, n_actions, T, p) -> float:
    r"""
    computes the theoretical value of the :math:`\omega` parameter.

    Parameters
    ----------
    n_states : int
        The number of states.
    n_actions : int
        The number of actions.
    T : int
        The optimization horizon.
    p : float
        The probability of failure.

    Returns
    -------
    float
        The theoretical value of the :math:`\omega` parameter.
    """
    return np.log(T / p)


def get_kappa(n_states, n_actions, T, p) -> float:
    r"""
    computes the theoretical value of the :math:`\kappa` parameter.

    Parameters
    ----------
    n_states : int
        The number of states.
    n_actions : int
        The number of actions.
    T : int
        The optimization horizon.
    p : float
        The probability of failure.

    Returns
    -------
    float
        The theoretical value of the :math:`\kappa` parameter.
    """
    return np.log(T / p)


def get_eta(n_states, n_actions, T, p, omega) -> float:
    r"""
    computes the theoretical value of the :math:`\eta` parameter.

    Parameters
    ----------
    n_states : int
        The number of states.
    n_actions : int
        The number of actions.
    T : int
        The optimization horizon.
    p : float
        The probability of failure.
    omega : float
        The omega parameter.

    Returns
    -------
    float
        The theoretical value of the :math:`\eta` parameter.
    """
    return np.sqrt(T * n_states / n_actions) + 12 * omega * n_states ** 4


@gin.configurable
class PSRLContinuous(BaseAgent):
    """
    The posterior sampling for reinforcement learning algorithm with optimism.

    Agrawal, Shipra, and Randy Jia. "Posterior sampling for reinforcement learning: worst-case regret bounds." arXiv
    preprint arXiv:1705.07041 (2017).
    """

    @staticmethod
    def is_emission_map_accepted(emission_map: "EmissionMap") -> bool:
        return emission_map.is_tabular

    @staticmethod
    def produce_gin_file_from_parameters(parameters: Dict[str, Any], index: int = 0):
        return (
            "from colosseum.agent.mdp_models import bayesian_models\n"
            f"prms_{index}/PSRLContinuous.reward_prior_model = %bayesian_models.RewardsConjugateModel.N_NIG\n"
            f"prms_{index}/PSRLContinuous.rewards_prior_prms = [{parameters['rewards_prior_mean']}, 1, 1, 1]\n"
            f"prms_{index}/PSRLContinuous.psi_weight = {parameters['psi_weight']}\n"
            f"prms_{index}/PSRLContinuous.omega_weight = {parameters['omega_weight']}\n"
            f"prms_{index}/PSRLContinuous.kappa_weight = {parameters['kappa_weight']}\n"
            f"prms_{index}/PSRLContinuous.eta_weight = {parameters['eta_weight']}"
        )

    @staticmethod
    def is_episodic() -> bool:
        return False

    @staticmethod
    def get_hyperparameters_search_spaces() -> Dict[str, tune.sample.Domain]:
        return {
            "psi_weight": tune.uniform(0.001, 0.1),
            "omega_weight": tune.uniform(0.0001, 1),
            "kappa_weight": tune.uniform(0.2, 4),
            "eta_weight": tune.uniform(1e-10, 1e-6),
            "rewards_prior_mean": tune.uniform(0.0, 1.2),
        }

    @staticmethod
    def get_agent_instance_from_parameters(
        seed: int,
        optimization_horizon: int,
        mdp_specs: MDPSpec,
        parameters: Dict[str, Any],
    ) -> "BaseAgent":
        return PSRLContinuous(
            mdp_specs=mdp_specs,
            seed=seed,
            optimization_horizon=optimization_horizon,
            reward_prior_model=RewardsConjugateModel.N_NIG,
            rewards_prior_prms=[parameters["rewards_prior_mean"], 1, 1, 1],
            psi_weight=parameters["psi_weight"],
            omega_weight=parameters["omega_weight"],
            kappa_weight=parameters["kappa_weight"],
            eta_weight=parameters["eta_weight"],
        )

    @property
    def current_optimal_stochastic_policy(self) -> np.ndarray:
        T_map, R_map = self._mdp_model.get_map_estimate()
        Q, _ = discounted_value_iteration(T_map, R_map)
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
        psi_weight: float = 1.0,
        omega_weight: float = 1.0,
        kappa_weight: float = 1.0,
        eta_weight: float = 1.0,
        get_psi: Callable[[int, int, int, float], float] = get_psi,
        get_omega: Callable[[int, int, int, float], float] = get_omega,
        get_kappa: Callable[[int, int, int, float], float] = get_kappa,
        get_eta: Callable[[int, int, int, float, float], float] = get_eta,
        p: float = 0.05,
        no_optimistic_sampling: bool = False,
        truncate_reward_with_max: bool = False,
        min_steps_before_new_episode: int = 0,
        max_psi: int = 60,
    ):
        r"""
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
        psi_weight : float
            The coefficient for which the theoretical value of the :math:`\psi` parameter is multiplied for. By default,
            it is set to one.
        omega_weight : float
            The coefficient for which the theoretical value of the :math:`\omega` parameter is multiplied for. By default,
            it is set to one.
        kappa_weight : float
            The coefficient for which the theoretical value of the :math:`\kappa` parameter is multiplied for. By default,
            it is set to one.
        eta_weight : float
            The coefficient for which the theoretical value of the :math:`\eta` parameter is multiplied for. By default,
            it is set to one.
        get_psi : Callable[[int, int, int, float], float]
            The function that computes the value of the :math:`\psi` parameter given number of states, number of action,
             optimization horizon, and probability of failure. By default, it is set to the theoretical value.
        get_omega : Callable[[int, int, int, float], float]
            The function that computes the value of the :math:`\omega` parameter given number of states, number of action,
             optimization horizon, and probability of failure. By default, it is set to the theoretical value.
        get_kappa : Callable[[int, int, int, float], float]
            The function that computes the value of the :math:`\kappa` parameter given number of states, number of action,
             optimization horizon, and probability of failure. By default, it is set to the theoretical value.
        get_eta : Callable[[int, int, int, float, float], float]
            The function that computes the value of the :math:`\eta` parameter given number of states, number of action,
             optimization horizon, probability of failure, and the omega parameter. By default, it is set to the
             theoretical value.
        p : float
            The probability of failure. By default, it is set to :math:`0.05`.
        no_optimistic_sampling : bool
            If True the optimistic sampling procedure is disabled.
        truncate_reward_with_max : bool
            If True, the sampled rewards are truncated to the maximum possible value of the reward. By default, it is
            set to False.
        min_steps_before_new_episode : int
            The minimum interval length between artificial episodes. By default, it is set to zero.
        max_psi : int
            The maximum value of the :math:`\psi` parameter. By default, it is set to :math:`60`.
        """

        self._n_states = mdp_specs.observations.num_values
        self._n_actions = mdp_specs.actions.num_values

        self.truncate_reward_with_max = truncate_reward_with_max
        self.no_optimistic_sampling = (
            no_optimistic_sampling
            or (self._n_states ** 2 * self._n_actions) > 6_000_000
        )

        self.p = p
        self.psi = min(
            max_psi,
            max(
                2,
                int(
                    psi_weight
                    * get_psi(self._n_states, self._n_actions, optimization_horizon, p)
                ),
            ),
        )
        self.omega = omega_weight * get_omega(
            self._n_states, self._n_actions, optimization_horizon, p
        )
        self.kappa = kappa_weight * get_kappa(
            self._n_states, self._n_actions, optimization_horizon, p
        )
        self.eta = max(
            5,
            min(
                10 * self._n_states,
                eta_weight
                * get_eta(
                    self._n_states,
                    self._n_actions,
                    optimization_horizon,
                    p,
                    self.omega,
                ),
            ),
        )

        self._n_states = mdp_specs.observations.num_values
        self.episode = 0
        self.min_steps_before_new_episode = min_steps_before_new_episode
        self.last_change = 0

        self.M = np.zeros(
            (self._n_states, self._n_actions, self._n_states), dtype=np.float32
        )
        self.N = np.zeros(
            (self._n_states, self._n_actions, self._n_states), dtype=np.int32
        )
        q_shape = (
            (self._n_states, self._n_actions, self._n_states)
            if no_optimistic_sampling
            else (self.psi, self._n_states, self._n_actions, self._n_states)
        )
        self.Q = np.zeros(q_shape, dtype=np.float32)
        self.nu_k = np.zeros((self._n_states, self._n_actions), dtype=np.int8)
        self.episode_transition_data = dict()

        mdp_model = BayesianMDPModel(
            seed,
            mdp_specs,
            reward_prior_model=reward_prior_model,
            transitions_prior_model=transitions_prior_model,
            rewards_prior_prms=rewards_prior_prms,
            transitions_prior_prms=transitions_prior_prms,
        )

        super(PSRLContinuous, self).__init__(
            seed,
            mdp_specs,
            mdp_model,
            QValuesActor(seed, mdp_specs, epsilon_greedy, boltzmann_temperature),
            optimization_horizon,
        )

    def is_episode_end(
        self,
        ts_t: dm_env.TimeStep,
        a_t: "ACTION_TYPE",
        ts_tp1: dm_env.TimeStep,
        time: int,
    ) -> bool:
        if time - self.last_change < self.min_steps_before_new_episode:
            return False
        self.last_change = time
        nu_k = len(self.episode_transition_data[ts_t.observation, a_t])
        N_tau = self.N[ts_t.observation, a_t].sum()
        return N_tau >= 2 * (N_tau - nu_k)

    def episode_end_update(self):
        if self.no_optimistic_sampling:
            T = self._mdp_model.sample_T()
        else:
            self.optimistic_sampling()
            T = np.moveaxis(self.Q, 0, 2)
            T = T.reshape((self._n_states, -1, self._n_states))

        R = self._mdp_model.sample_R()
        if self.truncate_reward_with_max:
            R = np.maximum(self.r_max, R)
        if not self.no_optimistic_sampling:
            R = np.tile(R, (1, self.psi))

        Q, _ = discounted_value_iteration(T, R)
        self._actor.set_q_values(Q)

        self.episode_transition_data = dict()

    def before_start_interacting(self):
        self._actor.set_q_values(
            self._rng.randn(self._n_states, self._n_actions * self.psi)
        )
        self.episode_end_update()

    def select_action(self, ts: dm_env.TimeStep, time: int) -> "ACTION_TYPE":
        return self.extended_action_to_real(
            super(PSRLContinuous, self).select_action(ts, time)
        )

    def step_update(
        self, ts_t: dm_env.TimeStep, a_t: "ACTION_TYPE", ts_tp1: dm_env.TimeStep, h: int
    ):
        super(PSRLContinuous, self).step_update(ts_t, a_t, ts_tp1, h)

        self.M[ts_t.observation, a_t, ts_tp1.observation] = (
            self.N[ts_t.observation, a_t, ts_tp1.observation] + self.omega
        ) / self.kappa
        self.N[ts_t.observation, a_t, ts_tp1.observation] += 1

        if (ts_t.observation, a_t) in self.episode_transition_data:
            if not ts_tp1.last():
                self.episode_transition_data[ts_t.observation, a_t].append(
                    ts_tp1.observation
                )
        else:
            if not ts_tp1.last():
                self.episode_transition_data[ts_t.observation, a_t] = [
                    ts_tp1.observation
                ]

    def optimistic_sampling(self):
        """
        performs the optimistic sampling procedure.
        """
        Nsum = self.N.sum(-1)
        cond = Nsum < self.eta
        indices_2 = list(np.where(cond))
        indices_1 = list(np.where(~cond))

        do_simple_sampling = len(indices_2[0]) > 0
        do_posterior_sampling = len(indices_1[0]) > 0
        if do_simple_sampling:
            P_hat = self.N / np.maximum(Nsum[..., None], 1)
            N = np.maximum(self.N, 1)
            P_minus = P_hat - np.minimum(
                np.sqrt(3 * P_hat * np.log(4 * self._n_states) / N)
                + 3 * np.log(4 * self._n_states) / N,
                P_hat,
            )

        for psi in range(self.psi):
            if do_posterior_sampling:
                self.Q[
                    tuple([np.array([psi] * len(indices_1[0]))] + indices_1)
                ] = self._mdp_model._transitions_model.sample_sa(tuple(indices_1))
            if do_simple_sampling:
                z = self._rng.randint(self._n_states)
                summing = 1 - P_minus.sum(-1)
                P_minus[:, :, z] += summing
                self.Q[
                    tuple([np.array([psi] * len(indices_2[0]))] + indices_2)
                ] = P_minus[tuple(indices_2)]
                P_minus[:, :, z] -= summing

    def extended_action_to_real(self, action) -> int:
        """
        transform the extended action used to induce optimistic to a real action of the MDP.
        """
        if self.no_optimistic_sampling:
            return action
        psi, real_action = action % self.psi, int(action / self.psi)
        return real_action
