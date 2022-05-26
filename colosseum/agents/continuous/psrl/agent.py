from typing import Callable

import dm_env
import gin
import numpy as np
import toolz

from colosseum.agents.base import ContinuousHorizonBaseActor
from colosseum.agents.bayes_tools.conjugate_rewards import PRIOR_TYPE as R_PRIOR_TYPE
from colosseum.agents.bayes_tools.conjugate_rewards import RewardsConjugateModel
from colosseum.agents.bayes_tools.conjugate_transitions import M_DIR
from colosseum.agents.policy import ContinuousPolicy
from colosseum.dp.continuous import get_policy, value_iteration
from colosseum.utils.acme.specs import EnvironmentSpec
from colosseum.utils.random_vars import state_occurencens_to_counts


def get_psi(num_states, num_actions, T, p) -> float:
    return num_states * np.log(num_states * num_actions / p)


def get_omega(num_states, num_actions, T, p) -> float:
    return np.log(T / p)


def get_kappa(num_states, num_actions, T, p) -> float:
    return np.log(T / p)


def get_eta(num_states, num_actions, T, p, omega) -> float:
    return np.sqrt(T * num_states / num_actions) + 12 * omega * num_states ** 4


@gin.configurable
class PSRLContinuous(ContinuousHorizonBaseActor):
    """
    The Posterior Sampling for Reinforcement Learning agent for infinite horizon MDP.
    """

    def __init__(
        self,
        environment_spec: EnvironmentSpec,
        seed: int,
        T: int,
        r_max: float,
        reward_prior_model: RewardsConjugateModel,
        rewards_prior_prms: R_PRIOR_TYPE,
        psi_weight: float = 1.0,
        omega_weight: float = 1.0,
        kappa_weight: float = 1.0,
        eta_weight: float = 1.0,
        get_psi: Callable = get_psi,
        get_omega: Callable = get_omega,
        get_kappa: Callable = get_kappa,
        get_eta: Callable = get_eta,
        p: float = 0.05,
        no_optimistic_sampling: bool = False,
        truncate_reward_with_max: bool = False,
        min_steps_before_new_episode: int = 0,
    ):
        """

        Parameters
        ----------
        environment_spec : EnvironmentSpec
            encodes the specification of the MDP in terms of number of states and actions.
        seed : int
            the seed for the agent
        T : int
            the optimization horizon.
        r_max : float
            the maximum reward that the MDP can yield.
        reward_prior_model : RewardsConjugateModel
            check the RewardsConjugateModel class to see which Bayesian conjugate models are available
        rewards_prior_prms : Union[List, Dict[Tuple[int, int], List]]
            the reward prior can either be a list of hyperparameters that are set identical for each state-action pair,
            or it can be a dictionary with the state action pair as key and a list of hyperparameters as value.
        psi_weight : float, optional
            linear coefficient for the psi hyperparameter.
            By default, it is set to 1.
        omega_weight : float, optional
            linear coefficient for the omega hyperparameter.
            By default, it is set to 1.
        kappa_weight : float, optional
            linear coefficient for the kappa hyperparameter.
            By default, it is set to 1.
        eta_weight : float, optional
            linear coefficient for the eta hyperparameter.
            By default, it is set to 1.
        get_psi : Callable, optional
            represents the number of optimistic sampling that the algorithm does.
            By default, it is calculated according to its theoretical guarantees value.
        get_omega : Callable, optional
            is the prior for the dirichlet in the transition conjugate model.
            By default, it is calculated according to its theoretical guarantees value.
        get_kappa : Callable, optional
            is a positive-valued hyperparameter of the algorithm.
            By default, it is calculated according to its theoretical guarantees value.
        get_eta : Callable, optional
            is a positive-valued hyperparameter of the algorithm.
            By default, it is calculated according to its theoretical guarantees value.
        p : float
            1 - the probability of the theoretical upper bound holding.
        no_optimistic_sampling : bool, optional
            removes the optimistic sampling procedures.
        truncate_reward_with_max : bool, optional
            truncates the sampling of the reward using the known value of the maximum obtainable reward.
        min_steps_before_new_episode : int, optional
            sets a minimum number of time step before a new artificial episode can start.
        """

        super(PSRLContinuous, self).__init__(environment_spec, seed, r_max, T)

        self.truncate_reward_with_max = truncate_reward_with_max
        self.no_optimistic_sampling = (
            no_optimistic_sampling
            or (self.num_states ** 2 * self.num_actions) > 6_000_000
        )

        self.p = p
        self.psi = min(
            80,
            max(2, int(psi_weight * get_psi(self.num_states, self.num_actions, T, p))),
        )
        self.omega = omega_weight * get_omega(self.num_states, self.num_actions, T, p)
        self.kappa = kappa_weight * get_kappa(self.num_states, self.num_actions, T, p)
        self.eta = max(
            5,
            min(
                10 * self.num_states,
                eta_weight
                * get_eta(self.num_states, self.num_actions, T, p, self.omega),
            ),
        )

        self.episode = 0
        self.min_steps_before_new_episode = min_steps_before_new_episode
        self.last_change = 0

        self.M = np.zeros(
            (self.num_states, self.num_actions, self.num_states), dtype=np.float32
        )
        q_shape = (
            (self.num_states, self.num_actions, self.num_states)
            if no_optimistic_sampling
            else (self.psi, self.num_states, self.num_actions, self.num_states)
        )
        self.Q = np.zeros(q_shape, dtype=np.float32)
        self.nu_k = np.zeros((self.num_states, self.num_actions), dtype=np.int8)

        # Instantiate Bayesian models
        self.rewards_model = reward_prior_model.get_class()(
            self.num_states, self.num_actions, rewards_prior_prms, seed
        )
        self.transitions_model = M_DIR(
            self.num_states, self.num_actions, [self.omega / self.num_states], seed
        )

        self.episode_reward_data = dict()
        self.episode_transition_data = dict()

    @property
    def current_policy(self) -> ContinuousPolicy:
        return ContinuousPolicy(
            {s: self.select_action(s) for s in range(self.num_states)}, self.num_actions
        )

    def is_episode_end(
        self, timestep: dm_env.TimeStep, action: int, new_timestep: dm_env.TimeStep
    ) -> bool:
        """
        Checks whether the artificial episode of the agent has ended.
        """
        if self.steps - self.last_change < self.min_steps_before_new_episode:
            return False
        self.last_change = self.steps
        nu_k = len(self.episode_transition_data[timestep.observation, action])
        N_tau = self.N[timestep.observation, action].sum()
        return N_tau >= 2 * (N_tau - nu_k)

    def select_action(self, observation: int) -> int:
        a = self._fast_rng.choice(
            np.where(self.policy.pi(observation) == self.policy.pi(observation).max())[
                0
            ]
        )
        return self.extended_action_to_real(a)

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
        # max_r = max(max(x) for x in self.episode_reward_data.values())
        self.rewards_model.update(self.episode_reward_data)

        if not self.no_optimistic_sampling:
            for (s_tm1, action), s_ts in self.episode_transition_data.items():
                for s_t in s_ts:
                    self.M[s_tm1, action, s_t] = (
                        self.N[self.s_tm1, action, s_t] + self.omega
                    ) / self.kappa

    def extended_action_to_real(self, action) -> int:
        """
        Transform the extended action used to induce optimistic to a real action of the MDP.
        """
        if self.no_optimistic_sampling:
            return action
        psi, real_action = action % self.psi, int(action / self.psi)
        return real_action

    def _before_new_episode(self):
        self.calculate_optimal_policy()

        self.nu_k = self.N.copy() - 1
        self.episode += 1

    def calculate_optimal_policy(self):
        if self.no_optimistic_sampling:
            T = self.transitions_model.sample_MDP()
        else:
            self.optimistic_sampling()
            T = np.moveaxis(self.Q, 0, 2)
            T = T.reshape(self.num_states, -1, self.num_states)

        R = self.rewards_model.sample_MDP()
        if self.truncate_reward_with_max:
            R = np.maximum(self.r_max, R)
        if not self.no_optimistic_sampling:
            R = np.tile(R, (1, self.psi))

        Q, _ = value_iteration(T, R)
        self.policy = get_policy(Q, self._fast_rng)

    def optimistic_sampling(self):
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
                np.sqrt(3 * P_hat * np.log(4 * self.num_states) / N)
                + 3 * np.log(4 * self.num_states) / N,
                P_hat,
            )

        for psi in range(self.psi):
            if do_posterior_sampling:
                self.Q[
                    tuple([np.array([psi] * len(indices_1[0]))] + indices_1)
                ] = self.transitions_model.sample_sa(tuple(indices_1))
            if do_simple_sampling:
                z = self._rng.randint(self.num_states)
                summing = 1 - P_minus.sum(-1)
                P_minus[:, :, z] += summing
                self.Q[
                    tuple([np.array([psi] * len(indices_2[0]))] + indices_2)
                ] = P_minus[tuple(indices_2)]
                P_minus[:, :, z] -= summing
