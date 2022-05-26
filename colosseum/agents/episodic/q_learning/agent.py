import dm_env
import gin
import numpy as np

from colosseum.agents.base import FiniteHorizonBaseActor, ValueBasedAgent
from colosseum.utils.acme.specs import EnvironmentSpec


@gin.configurable
class QLearningEpisodic(FiniteHorizonBaseActor, ValueBasedAgent):
    """
    The Q-learning agent for infinite horizon MDP.
    """

    def __init__(
        self,
        environment_spec: EnvironmentSpec,
        seed: int,
        H: int,
        r_max: float,
        T: int,
        p: float,
        c_1: float,
        c_2: float = None,
        min_at: float = 0,
        UCB_type="hoeffding",
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
        T : int
            the optimization horizon.
        r_max : float
            the maximum reward that the MDP can yield.
        p : float
            1 - the probability of the regret bound holding.
        c_1 : float
            positive-valued hyperparameter of the algorithm.
        c_2 : float, optional
            positive-valued hyperparameters of the algorithm when the ucb type is bernstein.
        min_at : float, optional
            minimum value for the update coefficient of the q values.
            By default, it is set to 0.
        UCB_type : str
            the type of bound to use for the upper confidence levels. It can be either "hoeffding" or "bernstein".
            By default, it is set to hoeffding.
        """

        super(QLearningEpisodic, self).__init__(
            environment_spec, seed, H, r_max, T, False, False
        )

        assert 0 <= min_at < 0.99
        assert 0 < p < 1
        assert c_1 > 0
        assert UCB_type.lower() in ["hoeffding", "bernstein"]
        if UCB_type.lower() == "bernstein":
            assert c_2 is not None and c_2 > 0

        self.p = p
        self.c_1 = c_1
        self.c_2 = c_2
        self.min_at = min_at if min_at > 0.009 else 0
        self.UCB_type = UCB_type.lower()

        self.i = np.log(self.num_states * self.num_actions * T / p)
        self.N = np.zeros((self.H, self.num_states, self.num_actions), np.int32)
        self.Q = np.zeros((self.H, self.num_states, self.num_actions), np.float32) + H
        self.V = np.zeros((self.H + 1, self.num_states), np.float32)

        if self.UCB_type == "bernstein":
            self.mu = np.zeros((self.H, self.num_states, self.num_actions), np.float32)
            self.sigma = np.zeros(
                (self.H, self.num_states, self.num_actions), np.float32
            )
            self.beta = np.zeros(
                (self.H, self.num_states, self.num_actions), np.float32
            )

    def _before_new_episode(self):
        pass

    def observe(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        next_timestep: dm_env.TimeStep,
    ):
        self.N[self.h, timestep.observation, action] += 1

        t = self.N[self.h, timestep.observation, action]
        a_t = max(self.min_at, (self.H + 1) / (self.H + t))

        if self.UCB_type == "hoeffding":
            b_t = self.c_1 * np.sqrt(self.H ** 3 * self.i / t)
        else:
            self.mu[self.h, timestep.observation, action] += self.V[
                self.h + 1, next_timestep.observation
            ]
            self.sigma[self.h, timestep.observation, action] += (
                self.V[self.h + 1, next_timestep.observation] ** 2
            )
            old_beta = self.beta[self.h, timestep.observation, action]
            self.beta[self.h, timestep.observation, action] = min(
                self.c_1
                * (
                    np.sqrt(
                        (
                            self.H
                            * (
                                (
                                    self.sigma[self.h, timestep.observation, action]
                                    - self.mu[self.h, timestep.observation, action]
                                )
                                ** 2
                            )
                            / t ** 2
                            + self.H
                        )
                        * self.i
                    )
                    + np.sqrt(self.H ** 7 * self.num_states * self.num_actions)
                    * self.i
                    / t
                ),
                self.c_2 * np.sqrt(self.H ** 3 * self.i / t),
            )
            b_t = (
                (self.beta[self.h, timestep.observation, action] - (1 - a_t) * old_beta)
                / 2
                / a_t
            )

        self.Q[self.h, timestep.observation, action] = a_t * self.Q[
            self.h, timestep.observation, action
        ] + (1 - a_t) * (
            next_timestep.reward + self.V[self.h + 1, next_timestep.observation] + b_t
        )
        self.V[self.h, timestep.observation] = min(
            self.H, self.Q[self.h, timestep.observation].max()
        )
        super(QLearningEpisodic, self).observe(timestep, action, next_timestep)

    def update_models(self):
        pass
