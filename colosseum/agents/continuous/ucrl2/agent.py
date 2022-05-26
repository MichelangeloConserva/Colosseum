import math as m
from typing import Union

import dm_env
import gin
import numpy as np
from ray import tune

from colosseum.agents.base import ContinuousHorizonBaseActor, ValueBasedAgent
from colosseum.dp.continuous import extended_value_iteration
from colosseum.utils.acme.specs import EnvironmentSpec


def chernoff(it, N, delta, sqrt_C, log_C, range=1.0):
    ci = range * np.sqrt(sqrt_C * m.log(log_C * (it + 1) / delta) / np.maximum(1, N))
    return ci


def bernstein(scale_a, log_scale_a, scale_b, log_scale_b, alpha_1, alpha_2):
    A = scale_a * m.log(log_scale_a)
    B = scale_b * m.log(log_scale_b)
    return alpha_1 * np.sqrt(A) + alpha_2 * B


@gin.configurable
class UCRL2Continuous(ContinuousHorizonBaseActor, ValueBasedAgent):
    """
    The UCRL agent for infinite horizon MDP.
    """

    search_space = {"a": tune.uniform(0.0001, 1), "b": tune.uniform(0.001, 1)}

    def __init__(
        self,
        environment_spec: EnvironmentSpec,
        seed: int,
        r_max: float,
        T: int,
        alpha_r=None,
        alpha_p=None,
        bound_type_p="chernoff",
        bound_type_rew="chernoff",
    ):
        """

        Parameters
        ----------
        environment_spec : EnvironmentSpec
            encodes the specification of the MDP in terms of number of states and actions.
        seed : int
            the seed for the agent
        r_max : float
            the maximum reward that the MDP can yield.
        T : int
            the optimization horizon.
        alpha_r : float, optional
            multiplicative factor for the concentration bound on rewards. By default, it is set to 1.
        alpha_p : float, optional
            multiplicative factor for the concentration bound on transition probabilities. By default, it is set to 1.
        bound_type_p : str, optional
            the type of bound to use for the upper confidence levels of the transition probabilities. It can be either
            chernoff or bernstein. By default, it is set to chernoff.
        bound_type_rew : str, optional
            the type of bound to use for the upper confidence levels of the rewards. It can be either chernoff or
            bernstein. By default, it is set to chernoff.
        """
        assert bound_type_p in ["chernoff", "bernstein"]
        assert bound_type_rew in ["chernoff", "bernstein"]

        super(UCRL2Continuous, self).__init__(environment_spec, seed, r_max, T)

        self.alpha_p = 1.0 if alpha_p is None else alpha_p
        self.alpha_r = 1.0 if alpha_r is None else alpha_r

        # initialize matrices
        self.policy = np.zeros((self.num_states,), dtype=np.int_)
        self.policy_indices = np.zeros((self.num_states,), dtype=np.int_)

        # initialization
        self.iteration = 0
        self.episode = 0
        self.delta = 1.0  # confidence
        self.bound_type_p = bound_type_p
        self.bound_type_rew = bound_type_rew

        self.local_random = self._rng
        self.P = (
            np.ones((self.num_states, self.num_actions, self.num_states), np.float32)
            / self.num_states
        )

        self.estimated_rewards = (
            np.ones((self.num_states, self.num_actions), np.float32) * r_max
        )
        self.variance_proxy_reward = np.zeros(
            (self.num_states, self.num_actions), np.float32
        )
        self.estimated_holding_times = np.ones(
            (self.num_states, self.num_actions), np.float32
        )

        self.current_state = None
        self.artificial_episode = 0

    def select_action(self, observation: int) -> int:
        Q = self.Q[observation]
        action = self._fast_rng.choice(np.where(Q == Q.max())[0])
        return int(action)

    def _before_new_episode(self):
        self.episode += 1
        self.delta = 1 / m.sqrt(self.iteration + 1)

        new_sp = self.solve_optimistic_model()
        if new_sp is not None:
            self.span_value = new_sp / self.r_max

    def update_models(self):
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
                self.estimated_holding_times[self.s_tm1, action] *= scale_f / (
                    scale_f + 1.0
                )
                self.estimated_holding_times[self.s_tm1, action] += 1 / (scale_f + 1)

        for (s_tm1, action) in set(self.episode_transition_data.keys()):
            self.P[s_tm1, action] = self.N[s_tm1, action] / self.N[s_tm1, action].sum()

        super(UCRL2Continuous, self).update_models()

    def is_episode_end(
        self, timestep: dm_env.TimeStep, action: int, new_timestep: dm_env.TimeStep
    ) -> bool:
        nu_k = len(self.episode_transition_data[timestep.observation, action])
        return nu_k >= max(1, self.N[timestep.observation, action].sum() - nu_k)

    def beta_r(self, nb_observations) -> np.ndarray:
        """
        Calculates the confidence bounds on the reward.
        Returns:
            np.array: the vector of confidence bounds on the reward function (|S| x |A|)
        """

        S = self.num_states
        A = self.num_actions
        if self.bound_type_rew != "bernstein":
            ci = chernoff(
                it=self.iteration,
                N=nb_observations,
                range=self.r_max,
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
                alpha_1=m.sqrt(self.alpha_r),
                alpha_2=self.alpha_r,
            )
            return beta

    def beta_p(self, nb_observations) -> np.ndarray:
        """
        Calculates the confidence bounds on the transition probabilities.
        Returns:
            np.array: the vector of confidence bounds on the reward function (|S| x |A|)
        """
        S = self.num_states
        A = self.num_actions
        if self.bound_type_p != "bernstein":
            beta = chernoff(
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
                alpha_1=m.sqrt(self.alpha_p),
                alpha_2=self.alpha_p,
            )
            return beta

    def solve_optimistic_model(self) -> Union[None, float]:
        """
        Solves the optimistic value iteration.
        Returns
        -------
            The span value of the estimates from the optimistic value iteration or None if no solution has been found.
        """
        nb_observations = self.N.sum(-1)

        beta_r = self.beta_r(nb_observations)  # confidence bounds on rewards
        beta_p = self.beta_p(
            nb_observations
        )  # confidence bounds on transition probabilities

        T = self.P
        estimated_rewards = self.estimated_rewards
        r_max = self.r_max

        assert np.isclose(T.sum(-1), 1.0).all()
        res = extended_value_iteration(T, estimated_rewards, beta_r, beta_p, r_max)
        if res is not None:
            span_value_new, self.Q, self.V = res
            span_value = span_value_new

            assert span_value >= 0, "The span value cannot be lower than zero"
            assert np.abs(span_value - span_value_new) < 1e-8

            return span_value
        return None
