import abc
import random
from typing import TYPE_CHECKING, Any, Dict, Union

import dm_env
import numpy as np
from ray import tune

from colosseum.utils.acme.specs import MDPSpec

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE, OBSERVATION_TYPE
    from colosseum.agent.actors import ACTOR_TYPES
    from colosseum.agent.mdp_models import MODEL_TYPES


class BaseAgent(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def is_episodic() -> bool:
        """
        returns whether the agent is suited for the episodic setting.
        """

    @abc.abstractmethod
    def __init__(
        self,
        seed: int,
        environment_spec: "MDPSpec",
        mdp_model: Union[None, "MODEL_TYPES"],
        actor: "ACTOR_TYPES",
        optimization_horizon: int,
    ):
        """
        Parameters
        ----------
        seed : int
            is the random seed.
        environment_spec : MDPSpec
            provides the full specification of the MDP.
        mdp_model : BaseMDP_Model
            is the component of the agent that encapsulates the knowledge acquired from the interactions with
            the MDP.
        actor : BaseActor
            calculates an action given the corresponding MDP model.
        optimization_horizon : int
            is the total number of interactions that the agent is expected to have with the MDP.
        """
        self._environment_spec = environment_spec
        self._mdp_model = mdp_model
        self._actor = actor
        self._optimization_horizon = optimization_horizon
        self._time_horizon = environment_spec.time_horizon

        self._rng = np.random.RandomState(seed)
        self._rng_fast = random.Random(seed)

    @abc.abstractmethod
    def episode_end_update(self):
        """
        is in charge of any change that should happen when an episode ends. In the infinite horizon case, we refer to
        artificial episodes.
        """

    @abc.abstractmethod
    def before_start_interacting(self):
        """
        is called before the agent starts interacting with the MDP.
        """

    def is_episode_end(
        self,
        ts_t: dm_env.TimeStep,
        a_t: "ACTION_TYPE",
        ts_tp1: dm_env.TimeStep,
        time_step: int,
    ) -> bool:
        """
        checks whether the episode is terminated. By default, this checks whether the current time step
        exceeds the time horizon. In the continuous case, this can be used to define artificial episodes.
        """
        return ts_tp1.last()

    def select_action(
        self, ts: dm_env.TimeStep, time_step: int
    ) -> "ACTION_TYPE":
        """

        Parameters
        ----------
        ts : dm_env.TimeStep
            the observation for which the agent is required to calculate the next action.
        time_step : int
            the current time step of the environment. In the episodic case, this refers to the in-episode time step,
            whereas in the continuous case this refers to the number of previous interactions.

        Returns
        -------
        action : ACTION_TYPE
            the action that the agent suggests to take given the observation and the time step.
        """
        return self._actor.select_action(ts, time_step)

    def step_update(
        self, ts_t: dm_env.TimeStep, a_t: "ACTION_TYPE", ts_tp1: dm_env.TimeStep, h: int
    ):
        """
        updates the MDP model with the _compute_transition information given in input.
        """
        if self._mdp_model:
            self._mdp_model.step_update(ts_t, a_t, ts_tp1, h)

    def agent_logs(self):
        """
        is called during the agent MDP interaction at lagging time. It can be used to log additional information.
        """
        pass

    @property
    @abc.abstractmethod
    def current_optimal_stochastic_policy(self) -> np.ndarray:
        """
        returns the estimates of the best optimal policy given the current knowledge of the agent. It must be return in
        the stochastic.
        """

    @staticmethod
    @abc.abstractmethod
    def get_hyperparameters_search_spaces() -> Dict[str, tune.sample.Domain]:
        """
        returns a dictionary with key value pairs corresponding to hyperparameter name and ray.tune sampler.
        """

    @staticmethod
    @abc.abstractmethod
    def produce_gin_file_from_hyperparameters(
        hyperparameters: Dict[str, Any], index: int = 0
    ):
        pass

    @staticmethod
    @abc.abstractmethod
    def get_agent_instance_from_hyperparameters(
        seed: int,
        optimization_horizon: int,
        mdp_specs: MDPSpec,
        hyperparameters: Dict[str, Any],
    ) -> "BaseAgent":
        """
        returns an agent instance for the mdp specification and agent hyperparameters given in input.
        """
