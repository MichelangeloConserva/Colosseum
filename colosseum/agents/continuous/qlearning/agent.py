from typing import Callable

import dm_env
import gin
import numpy as np

from colosseum.agents.base import ContinuousHorizonBaseActor, ValueBasedAgent
from colosseum.utils.acme.specs import EnvironmentSpec


def get_H(num_states, num_actions, T, span_approx, confidence):
    return min(
        np.sqrt(span_approx * T / num_states / num_actions),
        (T / num_states / num_actions / np.log(4 * T / confidence)) ** 0.333,
    )


@gin.configurable
class QLearningContinuous(ContinuousHorizonBaseActor, ValueBasedAgent):
    """
    The Q-learning agent for infinite horizon MDP.
    """

    def is_episode_end(
        self, timestep: dm_env.TimeStep, action: int, new_timestep: dm_env.TimeStep
    ) -> bool:
        """
        Returns False as there are no artificial episodes for Q-learning.
        """
        return False

    def __init__(
        self,
        environment_spec: EnvironmentSpec,
        seed: int,
        T: int,
        r_max: float,
        min_at: float = 0,
        confidence: float = 0.95,
        h_weigh: float = 1,
        span_approx_weight: float = 1,
        get_span_approx: Callable[[int, int], float] = None,
        get_H: Callable[[int, int, int, float, float], float] = get_H,
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
        min_at : float, optional
            minimum value for the update coefficient of the q values.
            By default, it is set to 0.
        confidence : float, optional
            the confidence level of the theoretical guarantees in the (0, 1) interval.
        h_weigh : float, optional
            linear coefficient for the H hyperparameter.
            By default, it is set to 1.
        span_approx_weight : float, optional
            linear coefficient for the span_approximation hyperparameter.
            By default, it is set to 1.
        get_span_approx : Callable, optional
            provides an upper bound approximation of the span of the value function of the MDP.
            By default, it is set to 1.
        get_H : Callable, optional, optional
            a hyperparameter of the agent.
            By default, it is calculated according to its theoretical guarantees value.
        """
        super(QLearningContinuous, self).__init__(
            environment_spec, seed, r_max, T, False, False
        )

        self.min_at = min_at if min_at > 0.009 else 0
        self.span_approx = span_approx_weight
        if get_span_approx is not None:
            self.span_approx *= get_span_approx(self.num_states, self.num_actions)

        self.confidence = confidence
        self.H = h_weigh * get_H(
            self.num_states, self.num_actions, T, self.span_approx, confidence
        )
        self.gamma = 1 - 1 / self.H

        self.N = np.zeros((self.num_states, self.num_actions), np.int32)
        self.Q = np.zeros((self.num_states, self.num_actions), np.float32) + self.H
        self.Q_main = np.zeros((self.num_states, self.num_actions), np.float32) + self.H
        self.V = np.zeros((self.num_states,), np.float32) + self.H

    def _before_new_episode(self):
        pass

    def observe(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        next_timestep: dm_env.TimeStep,
    ):
        s_tm1 = timestep.observation
        s_t = next_timestep.observation

        self.N[s_tm1, action] += 1
        a_t = max(self.min_at, (self.H + 1) / (self.H + self.N[s_tm1, action]))
        b_t = (
            4
            * self.span_approx
            * np.sqrt(
                self.H / self.N[s_tm1, action] * np.log(2 * self.T / self.confidence)
            )
        )

        self.Q_main[s_tm1, action] = (1 - a_t) * self.Q[s_tm1, action] + a_t * (
            next_timestep.reward + self.gamma * self.V[s_t] + b_t
        )
        self.Q[s_tm1, action] = min(self.Q[s_tm1, action], self.Q_main[s_tm1, action])
        self.V[s_t] = self.Q[s_t].max()

        super(QLearningContinuous, self).observe(timestep, action, next_timestep)

    def update_models(self):
        pass
