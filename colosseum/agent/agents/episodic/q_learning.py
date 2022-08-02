from typing import TYPE_CHECKING, Any, Callable, Dict, Union

import dm_env
import gin
import numpy as np
from ray import tune

from colosseum.dynamic_programming.utils import get_policy_from_q_values
from colosseum.agent.actors import QValuesActor
from colosseum.agent.agents.base import BaseAgent
from colosseum.agent.mdp_models.base import BaseMDPModel

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE
    from colosseum.utils.acme.specs import MDPSpec


class _QValuesModel(BaseMDPModel):
    def __init__(
        self,
        seed: int,
        environment_spec: "MDPSpec",
        optimization_horizon: int,
        p: float,
        c_1: float,
        c_2: float = None,
        min_at: float = 0,
        UCB_type="hoeffding",
    ):
        super(_QValuesModel, self).__init__(seed, environment_spec)

        self._UCB_type = UCB_type
        self._min_at = min_at
        self._c_1 = c_1
        self._c_2 = c_2
        self._p = p

        self.i = np.log(self._n_states * self._n_actions * optimization_horizon / p)
        self.N = np.ones((self._H, self._n_states, self._n_actions), np.int32)
        self.Q = (
            np.zeros((self._H, self._n_states, self._n_actions), np.float32) + self._H
        )
        self.V = np.zeros((self._H + 1, self._n_states), np.float32)

        if UCB_type == "bernstein":
            self.mu = np.zeros((self._H, self._n_states, self._n_actions), np.float32)
            self.sigma = np.zeros(
                (self._H, self._n_states, self._n_actions), np.float32
            )
            self.beta = np.zeros((self._H, self._n_states, self._n_actions), np.float32)

    def step_update(
        self, ts_t: dm_env.TimeStep, a_t: "ACTION_TYPE", ts_tp1: dm_env.TimeStep, h: int
    ):
        s_t = ts_t.observation
        s_tp1 = ts_tp1.observation

        self.N[h, s_t, a_t] += 1

        t = self.N[h, s_t, a_t]
        self._alpha_t = max(self._min_at, (self._H + 1) / (self._H + t))

        if self._UCB_type == "hoeffding":
            b_t = self._c_1 * np.sqrt(self._H ** 3 * self.i / t)
        else:
            self.mu[h, s_t, a_t] += self.V[h + 1, s_tp1]
            self.sigma[h, s_t, a_t] += self.V[h + 1, s_tp1] ** 2
            old_beta = self.beta[h, s_t, a_t]
            self.beta[h, s_t, a_t] = min(
                self._c_1
                * (
                    np.sqrt(
                        (
                            self._H
                            * ((self.sigma[h, s_t, a_t] - self.mu[h, s_t, a_t]) ** 2)
                            / t ** 2
                            + self._H
                        )
                        * self.i
                    )
                    + np.sqrt(self._H ** 7 * self._n_states * self._n_actions)
                    * self.i
                    / t
                ),
                self._c_2 * np.sqrt(self._H ** 3 * self.i / t),
            )
            b_t = (
                (self.beta[h, s_t, a_t] - (1 - self._alpha_t) * old_beta)
                / 2
                / self._alpha_t
            )

        self.Q[h, s_t, a_t] = self._alpha_t * self.Q[h, s_t, a_t] + (
            1 - self._alpha_t
        ) * (ts_tp1.reward + self.V[h + 1, s_tp1] + b_t)
        self.V[h, s_t] = min(self._H, self.Q[h, s_t].max())


@gin.configurable
class QLearningEpisodic(BaseAgent):
    @staticmethod
    def produce_gin_file_from_hyperparameters(
        hyperparameters: Dict[str, Any], index: int = 0
    ):
        string = (
            f"prms_{index}/QLearningEpisodic.p=0.05\n"
            f'prms_{index}/QLearningEpisodic.UCB_type="Bernstein"\n'
        )
        for k, v in hyperparameters.items():
            string += f"prms_{index}/QLearningEpisodic.{k} = {v}\n"
        return string[:-1]

    @staticmethod
    def is_episodic() -> bool:
        return True

    @staticmethod
    def get_hyperparameters_search_spaces() -> Dict[str, tune.sample.Domain]:
        return {
            "c_1": tune.uniform(0.001, 1.1),
            "c_2": tune.uniform(0.001, 1.1),
            "min_at": tune.uniform(0.001, 0.2),
        }

    @staticmethod
    def get_agent_instance_from_hyperparameters(
        seed: int,
        optimization_horizon: int,
        mdp_specs: "MDPSpec",
        hyperparameters: Dict[str, Any],
    ) -> "BaseAgent":
        return QLearningEpisodic(
            environment_spec=mdp_specs,
            seed=seed,
            optimization_horizon=optimization_horizon,
            min_at=hyperparameters["min_at"],
            c_1=hyperparameters["c_1"],
            c_2=hyperparameters["c_2"],
            UCB_type="Bernstein",
            p=0.05,
        )

    @property
    def current_optimal_stochastic_policy(self) -> np.ndarray:
        return get_policy_from_q_values(self._mdp_model.Q, True)

    def __init__(
        self,
        seed: int,
        environment_spec: "MDPSpec",
        optimization_horizon: int,
        # MDP model hyperparameters
        p: float,
        c_1: float,
        c_2: float = None,
        min_at: float = 0,
        UCB_type="hoeffding",
        # Actor hyperparameters
        epsilon_greedy: Union[float, Callable] = None,
        boltzmann_temperature: Union[float, Callable] = None,
    ):
        UCB_type = UCB_type.lower()
        assert 0 <= min_at < 0.99
        assert 0 < p < 1
        assert c_1 > 0
        assert UCB_type in ["hoeffding", "bernstein"]
        if UCB_type == "bernstein":
            assert c_2 is not None and c_2 > 0

        super(QLearningEpisodic, self).__init__(
            seed,
            environment_spec,
            _QValuesModel(
                seed,
                environment_spec,
                optimization_horizon,
                p,
                c_1,
                c_2,
                min_at,
                UCB_type,
            ),
            QValuesActor(seed, environment_spec, epsilon_greedy, boltzmann_temperature),
            optimization_horizon,
        )

    def episode_end_update(self):
        pass

    def before_start_interacting(self):
        self._actor.set_q_values(self._mdp_model.Q)

    def step_update(
        self, ts_t: dm_env.TimeStep, a_t: "ACTION_TYPE", ts_tp1: dm_env.TimeStep, h: int
    ):
        super(QLearningEpisodic, self).step_update(ts_t, a_t, ts_tp1, h)
        self._actor.set_q_values(self._mdp_model.Q)
