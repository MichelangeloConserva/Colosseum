import math
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Union

import dm_env
import gin
import numpy as np
from ray import tune

from colosseum.agent.actors import QValuesActor
from colosseum.agent.agents.base import BaseAgent
from colosseum.dynamic_programming import discounted_value_iteration
from colosseum.dynamic_programming.infinite_horizon import extended_value_iteration
from colosseum.dynamic_programming.utils import get_policy_from_q_values
from colosseum.emission_maps import EmissionMap

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE
    from colosseum.utils.acme.specs import MDPSpec


def _chernoff(it, N, delta, sqrt_C, log_C, range=1.0):
    ci = range * np.sqrt(sqrt_C * math.log(log_C * (it + 1) / delta) / np.maximum(1, N))
    return ci


def bernstein(scale_a, log_scale_a, scale_b, log_scale_b, alpha_1, alpha_2):
    A = scale_a * math.log(log_scale_a)
    B = scale_b * math.log(log_scale_b)
    return alpha_1 * np.sqrt(A) + alpha_2 * B

@gin.configurable
class UCRL2Continuous(BaseAgent):
    """
    The second version of upper confidence for reinforcement learning algorithm.

    Auer, Peter, Thomas Jaksch, and Ronald Ortner. "Near-optimal regret bounds for reinforcement learning." Advances in
    neural information processing systems 21 (2008).

    Fruit, Ronan, Matteo Pirotta, and Alessandro Lazaric. "Improved analysis of ucrl2 with empirical bernstein inequality."
    arXiv preprint arXiv:2007.05456 (2020).
    """

    @staticmethod
    def is_emission_map_accepted(emission_map: "EmissionMap") -> bool:
        return emission_map.is_tabular

    @staticmethod
    def produce_gin_file_from_parameters(parameters: Dict[str, Any], index: int = 0):
        string = f"prms_{index}/UCRL2Continuous.bound_type_p='bernstein'\n"
        for k, v in parameters.items():
            string += f"prms_{index}/UCRL2Continuous.{k} = {v}\n"
        return string[:-1]

    @staticmethod
    def is_episodic() -> bool:
        return False

    @staticmethod
    def get_hyperparameters_search_spaces() -> Dict[str, tune.sample.Domain]:
        return {"alpha_p": tune.uniform(0.1, 3), "alpha_r": tune.uniform(0.1, 3)}

    @staticmethod
    def get_agent_instance_from_parameters(
        seed: int,
        optimization_horizon: int,
        mdp_specs: "MDPSpec",
        parameters: Dict[str, Any],
    ) -> "BaseAgent":
        return UCRL2Continuous(
            mdp_specs=mdp_specs,
            seed=seed,
            optimization_horizon=optimization_horizon,
            alpha_p=parameters["alpha_p"],
            alpha_r=parameters["alpha_r"],
            bound_type_p="bernstein",
        )

    @property
    def current_optimal_stochastic_policy(self) -> np.ndarray:
        Q, _ = discounted_value_iteration(self.P, self.estimated_rewards)
        return get_policy_from_q_values(Q, True)

    def __init__(
        self,
        seed: int,
        mdp_specs: "MDPSpec",
        optimization_horizon: int,
        # MDP model parameters
        alpha_r=1.0,
        alpha_p=1.0,
        bound_type_p="_chernoff",
        bound_type_rew="_chernoff",
        # Actor parameters
        epsilon_greedy: Union[float, Callable] = None,
        boltzmann_temperature: Union[float, Callable] = None,
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
        alpha_r : float
            The :math:`\alpha` parameter for the rewards. By default, it is set to one.
        alpha_p : float
            The :math:`\alpha` parameter for the transitions. By default, it is set to one.
        bound_type_p : str
            The upper confidence bound type for the transitions. It can either be '_chernoff' or 'bernstein'. By default,
            it is set to '_chernoff'.
        bound_type_rew : str
            The upper confidence bound type for the rewards. It can either be '_chernoff' or 'bernstein'. By default,
            it is set to '_chernoff'.
        epsilon_greedy : Union[float, Callable], optional
            The probability of selecting an action at random. It can be provided as a float or as a function of the
            total number of interactions. By default, the probability is set to zero.
        boltzmann_temperature : Union[float, Callable], optional
            The parameter that controls the Boltzmann exploration. It can be provided as a float or as a function of
            the total number of interactions. By default, Boltzmann exploration is disabled.
        """

        n_states = self._n_states = mdp_specs.observations.num_values
        n_actions = self._n_actions = mdp_specs.actions.num_values
        self.reward_range = mdp_specs.rewards_range

        assert bound_type_p in ["_chernoff", "bernstein"]
        assert bound_type_rew in ["_chernoff", "bernstein"]

        self.alpha_p = alpha_p
        self.alpha_r = alpha_r

        # initialize matrices
        self.policy = np.zeros((n_states,), dtype=np.int_)
        self.policy_indices = np.zeros((n_states,), dtype=np.int_)

        # initialization
        self.iteration = 0
        self.episode = 0
        self.delta = 1.0  # confidence
        self.bound_type_p = bound_type_p
        self.bound_type_rew = bound_type_rew

        self.P = np.ones((n_states, n_actions, n_states), np.float32) / n_states

        self.estimated_rewards = (
            np.ones((n_states, n_actions), np.float32) * mdp_specs.rewards_range[1]
        )
        self.variance_proxy_reward = np.zeros((n_states, n_actions), np.float32)
        self.estimated_holding_times = np.ones((n_states, n_actions), np.float32)
        self.N = np.zeros((n_states, n_actions, n_states), dtype=np.int32)

        self.current_state = None
        self.artificial_episode = 0
        self.episode_reward_data = dict()
        self.episode_transition_data = dict()

        super(UCRL2Continuous, self).__init__(
            seed,
            mdp_specs,
            None,
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
        nu_k = len(self.episode_transition_data[ts_t.observation, a_t])
        return nu_k >= max(1, self.N[ts_t.observation, a_t].sum() - nu_k)

    def episode_end_update(self):
        self.episode += 1
        self.delta = 1 / math.sqrt(self.iteration + 1)

        new_sp = self.solve_optimistic_model()
        if new_sp is not None:
            self.span_value = new_sp / self.reward_range[1]

        if len(self.episode_transition_data) > 0:
            self.model_update()
            self.episode_reward_data = dict()
            self.episode_transition_data = dict()

    def before_start_interacting(self):
        self.episode_end_update()

    def step_update(
        self, ts_t: dm_env.TimeStep, a_t: "ACTION_TYPE", ts_tp1: dm_env.TimeStep, h: int
    ):
        self.N[ts_t.observation, a_t, ts_tp1.observation] += 1

        if (ts_t.observation, a_t) in self.episode_reward_data:
            self.episode_reward_data[ts_t.observation, a_t].append(ts_tp1.reward)
            if not ts_tp1.last():
                self.episode_transition_data[ts_t.observation, a_t].append(
                    ts_tp1.observation
                )
        else:
            self.episode_reward_data[ts_t.observation, a_t] = [ts_tp1.reward]
            if not ts_tp1.last():
                self.episode_transition_data[ts_t.observation, a_t] = [
                    ts_tp1.observation
                ]

    def model_update(self):
        """
        updates the model given the transitions obtained during the artificial episode.
        """
        for (s_tm1, action), r_ts in self.episode_reward_data.items():
            # updated observations
            scale_f = self.N[s_tm1, action].sum()
            for r in r_ts:
                # update the number of total iterations
                self.iteration += 1

                # update reward and variance estimate
                scale_f += 1
                old_estimated_reward = self.estimated_rewards[s_tm1, action]
                self.estimated_rewards[s_tm1, action] *= scale_f / (scale_f + 1.0)
                self.estimated_rewards[s_tm1, action] += r / (scale_f + 1.0)
                self.variance_proxy_reward[s_tm1, action] += (
                    r - old_estimated_reward
                ) * (r - self.estimated_rewards[s_tm1, action])

                # update holding time
                self.estimated_holding_times[s_tm1, action] *= scale_f / (scale_f + 1.0)
                self.estimated_holding_times[s_tm1, action] += 1 / (scale_f + 1)

        for (s_tm1, action) in set(self.episode_transition_data.keys()):
            self.P[s_tm1, action] = self.N[s_tm1, action] / self.N[s_tm1, action].sum()

    def beta_r(self, nb_observations) -> np.ndarray:
        """
        calculates the confidence bounds on the reward.
        Returns
        -------
        np.array
            The vector of confidence bounds on the reward function (|S| x |A|)
        """
        S = self._n_states
        A = self._n_actions
        if self.bound_type_rew != "bernstein":
            ci = _chernoff(
                it=self.iteration,
                N=nb_observations,
                range=self.reward_range[1],
                delta=self.delta,
                sqrt_C=3.5,
                log_C=2 * S * A,
            )
            return self.alpha_r * ci
        else:
            N = np.maximum(1, nb_observations)
            Nm1 = np.maximum(1, nb_observations - 1)
            var_r = self.variance_proxy_reward / Nm1
            log_value = 2.0 * S * A * (self.iteration + 1) / self.delta
            beta = bernstein(
                scale_a=14 * var_r / N,
                log_scale_a=log_value,
                scale_b=49.0 * self.r_max / (3.0 * Nm1),
                log_scale_b=log_value,
                alpha_1=math.sqrt(self.alpha_r),
                alpha_2=self.alpha_r,
            )
            return beta

    def beta_p(self, nb_observations) -> np.ndarray:
        """
        calculates the confidence bounds on the transition probabilities.
        Returns
        -------
        np.array
            The vector of confidence bounds on the reward function (|S| x |A|)
        """
        S = self._n_states
        A = self._n_actions
        if self.bound_type_p != "bernstein":
            beta = _chernoff(
                it=self.iteration,
                N=nb_observations,
                range=1.0,
                delta=self.delta,
                sqrt_C=14 * S,
                log_C=2 * A,
            )
            return self.alpha_p * beta.reshape([S, A, 1])
        else:
            N = np.maximum(1, nb_observations)
            Nm1 = np.maximum(1, nb_observations - 1)
            var_p = self.P * (1.0 - self.P)
            log_value = 2.0 * S * A * (self.iteration + 1) / self.delta
            beta = bernstein(
                scale_a=14 * var_p / N[:, :, np.newaxis],
                log_scale_a=log_value,
                scale_b=49.0 / (3.0 * Nm1[:, :, np.newaxis]),
                log_scale_b=log_value,
                alpha_1=math.sqrt(self.alpha_p),
                alpha_2=self.alpha_p,
            )
            return beta

    def solve_optimistic_model(self) -> Union[None, float]:
        """
        solves the optimistic value iteration.
        Returns
        -------
        Union[None, float]
            The span value of the estimates from the optimistic value iteration or None if no solution has been found.
        """
        nb_observations = self.N.sum(-1)

        beta_r = self.beta_r(nb_observations)  # confidence bounds on rewards
        beta_p = self.beta_p(
            nb_observations
        )  # confidence bounds on transition probabilities

        T = self.P
        estimated_rewards = self.estimated_rewards

        assert np.isclose(T.sum(-1), 1.0).all()
        try:
            res = extended_value_iteration(
                T, estimated_rewards, beta_r, beta_p, self.reward_range[1]
            )
        except SystemError:
            # Debug logs if the optimistic value iteration fails
            os.makedirs(f"tmp{os.sep}error_ext_vi", exist_ok=True)
            for i in range(100):
                if not os.path.isfile(f"tmp{os.sep}error_ext_vi{os.sep}T{i}.npy"):
                    np.save(f"tmp{os.sep}error_ext_vi{os.sep}T{i}.npy", T)
                    np.save(
                        f"tmp{os.sep}error_ext_vi{os.sep}estimated_rewards{i}.npy",
                        estimated_rewards,
                    )
                    np.save(f"tmp{os.sep}error_ext_vi{os.sep}beta_r.npy{i}", beta_r)
                    np.save(f"tmp{os.sep}error_ext_vi{os.sep}beta_p.npy{i}", beta_p)
                    break
            res = None

        if res is not None:
            span_value_new, self.Q, self.V = res
            span_value = span_value_new
            self._actor.set_q_values(self.Q)

            assert span_value >= 0, "The span value cannot be lower than zero"
            assert np.abs(span_value - span_value_new) < 1e-8

            return span_value
        return None
