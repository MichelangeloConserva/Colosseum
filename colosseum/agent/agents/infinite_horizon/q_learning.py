from typing import Any, Callable, Dict, Union

import dm_env
import gin
import numpy as np
from ray import tune
from typing_extensions import TYPE_CHECKING

from colosseum.agent.actors import QValuesActor
from colosseum.agent.agents.base import BaseAgent
from colosseum.agent.mdp_models.base import BaseMDPModel
from colosseum.dynamic_programming.utils import get_policy_from_q_values
from colosseum.emission_maps import EmissionMap

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE
    from colosseum.utils.acme.specs import MDPSpec


def get_H(n_states, n_actions, T, span_approx, confidence):
    """
    computes the theoretical value for the horizon approximation value.
    Parameters
    ----------
    n_states : int
        The number of states.
    n_actions : int
        The number of actions.
    T : int
        The optimization horizon.
    span_approx : float
        The span approximation value.
    confidence : float
        One minus the probability of failure.

    Returns
    -------
    float
        The theoretical value for the horizon approximation value.
    """
    return min(
        np.sqrt(span_approx * T / n_states / n_actions),
        (T / n_states / n_actions / np.log(4 * T / confidence)) ** 0.333,
    )


class _QValuesModel(BaseMDPModel):
    def __init__(
        self,
        seed: int,
        mdp_specs: "MDPSpec",
        optimization_horizon: int,
        min_at: float,
        confidence: float,
        span_approx_weight: float,
        get_span_approx: Callable[[int, int], float],
        h_weight: float,
        get_H: Callable[[int, int, int, float, float], float],
    ):
        super(_QValuesModel, self).__init__(seed, mdp_specs)

        self.min_at = min_at if min_at > 0.009 else 0
        self.span_approx = span_approx_weight
        if get_span_approx is not None:
            self.span_approx *= get_span_approx(self._n_states, self._n_actions)

        self.confidence = confidence
        self.optimization_horizon = optimization_horizon

        self.H = h_weight * get_H(
            self._n_states,
            self._n_actions,
            optimization_horizon,
            self.span_approx,
            confidence,
        )
        self.gamma = 1 - 1 / self.H

        self.N = np.zeros((self._n_states, self._n_actions), np.int32)
        self.Q = np.zeros((self._n_states, self._n_actions), np.float32) + self.H
        self.Q_main = np.zeros((self._n_states, self._n_actions), np.float32) + self.H
        self.V = np.zeros((self._n_states,), np.float32) + self.H

    def step_update(
        self,
        ts_t: dm_env.TimeStep,
        a_t: "ACTION_TYPE",
        ts_tp1: dm_env.TimeStep,
        time: int,
    ):
        s_t = ts_t.observation
        s_tp1 = ts_tp1.observation

        self.N[s_t, a_t] += 1
        alpha_t = max(self.min_at, (self.H + 1) / (self.H + self.N[s_t, a_t]))
        b_t = (
            4
            * self.span_approx
            * np.sqrt(
                self.H
                / self.N[s_t, a_t]
                * np.log(2 * self.optimization_horizon / self.confidence)
            )
        )

        self.Q_main[s_t, a_t] = (1 - alpha_t) * self.Q[s_t, a_t] + alpha_t * (
            ts_tp1.reward + self.gamma * self.V[s_tp1] + b_t
        )
        self.Q[s_t, a_t] = min(self.Q[s_t, a_t], self.Q_main[s_t, a_t])
        self.V[s_tp1] = self.Q[s_tp1].max()


@gin.configurable
class QLearningContinuous(BaseAgent):
    """
    The q-learning algorithm with optimism.

    Wei, Chen-Yu, et al. "Model-free reinforcement learning in infinite-horizon average-reward markov decision processes."
    International conference on machine learning. PMLR, 2020.
    """

    @staticmethod
    def is_emission_map_accepted(emission_map: "EmissionMap") -> bool:
        return emission_map.is_tabular

    @staticmethod
    def produce_gin_file_from_parameters(parameters: Dict[str, Any], index: int = 0):
        string = ""
        for k, v in parameters.items():
            string += f"prms_{index}/QLearningContinuous.{k} = {v}\n"
        return string[:-1]

    @staticmethod
    def is_episodic() -> bool:
        return False

    @staticmethod
    def get_hyperparameters_search_spaces() -> Dict[str, tune.search.sample.Domain]:
        return {
            "h_weight": tune.uniform(0.001, 1.1),
            "span_approx_weight": tune.uniform(0.001, 1.1),
            "min_at": tune.uniform(0.001, 0.2),
        }

    @staticmethod
    def get_agent_instance_from_parameters(
        seed: int,
        optimization_horizon: int,
        mdp_specs: "MDPSpec",
        parameters: Dict[str, Any],
    ) -> "BaseAgent":
        return QLearningContinuous(
            mdp_specs=mdp_specs,
            seed=seed,
            optimization_horizon=optimization_horizon,
            min_at=parameters["min_at"],
            h_weight=parameters["h_weight"],
            span_approx_weight=parameters["span_approx_weight"],
        )

    @property
    def current_optimal_stochastic_policy(self) -> np.ndarray:
        return get_policy_from_q_values(self._mdp_model.Q, True)

    def __init__(
        self,
        seed: int,
        mdp_specs: "MDPSpec",
        optimization_horizon: int,
        # MDP model parameters
        min_at: float = 0,
        confidence: float = 0.95,
        span_approx_weight: float = 1,
        get_span_approx: Callable[[int, int], float] = None,
        h_weight: float = 1,
        get_H: Callable[[int, int, int, float, float], float] = get_H,
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
        min_at : float
            The minimum value for the alpha coefficient. By default, it is set to zero.
        confidence : float
            One minus the probability of failure. By default, it is set to :math:`0.95`.
        span_approx_weight : float
            The weight given to the value of the span approximation.
        get_span_approx : Callable[[int, int], float]
            The function that computes the value for the span approximation given number of states and number of actions.
            By default, the theoretical value is used.
        h_weight : float
            The weight given to the value of approximate horizon. By default, it is set to one.
        get_H : Callable[[int, int, int, float, float], float]
            The function that computes the approximate horizon given number of states, number of actions, optimization
            horizon, the value of the span approximation, and the confidence. By default, the theoretical value is used.
        epsilon_greedy : Union[float, Callable], optional
            The probability of selecting an action at random. It can be provided as a float or as a function of the
            total number of interactions. By default, the probability is set to zero.
        boltzmann_temperature : Union[float, Callable], optional
            The parameter that controls the Boltzmann exploration. It can be provided as a float or as a function of
            the total number of interactions. By default, Boltzmann exploration is disabled.
        """

        assert 0 <= min_at < 0.99
        assert 0 < confidence < 1
        assert span_approx_weight > 0
        assert h_weight > 0

        super(QLearningContinuous, self).__init__(
            seed,
            mdp_specs,
            _QValuesModel(
                seed,
                mdp_specs,
                optimization_horizon,
                min_at,
                confidence,
                span_approx_weight,
                get_span_approx,
                h_weight,
                get_H,
            ),
            QValuesActor(seed, mdp_specs, epsilon_greedy, boltzmann_temperature),
            optimization_horizon,
        )

    def episode_end_update(self):
        pass

    def before_start_interacting(self):
        self._actor.set_q_values(self._mdp_model.Q)

    def step_update(
        self, ts_t: dm_env.TimeStep, a_t: "ACTION_TYPE", ts_tp1: dm_env.TimeStep, h: int
    ):
        super(QLearningContinuous, self).step_update(ts_t, a_t, ts_tp1, h)
        self._actor.set_q_values(self._mdp_model.Q)

    def get_q_value_estimate(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            The q-values estimate.
        """
        return self._mdp_model.Q
