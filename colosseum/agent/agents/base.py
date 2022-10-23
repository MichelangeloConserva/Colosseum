import abc
import random
from typing import TYPE_CHECKING, Any, Dict, Union

import dm_env
import numpy as np
from ray import tune

from colosseum.emission_maps import EmissionMap
from colosseum.utils.acme.specs import MDPSpec

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE
    from colosseum.agent.actors import ACTOR_TYPES
    from colosseum.agent.mdp_models import MODEL_TYPES


class BaseAgent(abc.ABC):
    """
    The base class for Colosseum agents.
    """

    @staticmethod
    @abc.abstractmethod
    def is_emission_map_accepted(emission_map: "EmissionMap") -> bool:
        """
        Returns
        -------
        bool
            True if the agent class accepts the emission map.
        """

    @staticmethod
    @abc.abstractmethod
    def is_episodic() -> bool:
        """
        Returns
        -------
        bool
            True if the agent is suited for the episodic setting.
        """

    @staticmethod
    @abc.abstractmethod
    def get_hyperparameters_search_spaces() -> Dict[str, tune.sample.Domain]:
        """
        Returns
        -------
        Dict[str, tune.sample.Domain]
            The dictionary with key value pairs corresponding to hyperparameter names and corresponding `ray.tune` samplers.
        """

    @staticmethod
    @abc.abstractmethod
    def produce_gin_file_from_parameters(
        parameters: Dict[str, Any], index: int = 0
    ) -> str:
        """
        produces a string containing the gin config file corresponding to the parameters given in input.

        Parameters
        ----------
        parameters : Dict[str, Any]
            The dictionary containing the parameters of the agent.
        index : int
            The index assigned to the gin configuration.

        Returns
        -------
        gin_config : str
            The gin configuration file.
        """

    @staticmethod
    @abc.abstractmethod
    def get_agent_instance_from_parameters(
        seed: int,
        optimization_horizon: int,
        mdp_specs: MDPSpec,
        parameters: Dict[str, Any],
    ) -> "BaseAgent":
        """
        returns an agent instance for the mdp specification and agent parameters given in input.

        Parameters
        ----------
        seed : int
            The random seed.
        optimization_horizon : int
            The total number of interactions that the agent is expected to have with the MDP.
        mdp_specs : MDPSpec
            The full specification of the MDP.
        parameters : Dict[str, Any]
            The dictionary containing the parameters of the agent.

        Returns
        -------
        BaseAgent
            The agent instance.
        """

    @abc.abstractmethod
    def __init__(
        self,
        seed: int,
        mdp_specs: "MDPSpec",
        mdp_model: Union[None, "MODEL_TYPES"],
        actor: "ACTOR_TYPES",
        optimization_horizon: int,
    ):
        """
        Parameters
        ----------
        seed : int
            The random seed.
        mdp_specs : MDPSpec
            The full specification of the MDP.
        mdp_model : BaseMDPModel
            The component of the agent that contains the knowledge acquired from the interactions with
            the MDP.
        actor : BaseActor
            The component of the agent that provide a mapping from MDP estimates to actions.
        optimization_horizon : int
            The total number of interactions that the agent is expected to have with the MDP.
        """
        self._mdp_spec = mdp_specs
        self._mdp_model = mdp_model
        self._actor = actor
        self._optimization_horizon = optimization_horizon
        self._time_horizon = mdp_specs.time_horizon

        self._rng = np.random.RandomState(seed)
        self._rng_fast = random.Random(seed)

    @property
    @abc.abstractmethod
    def current_optimal_stochastic_policy(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            The estimates of the best optimal policy given the current knowledge of the agent in the form of
            distribution over actions.
        """

    @abc.abstractmethod
    def episode_end_update(self):
        """
        is called when an episode ends. In the infinite horizon case, we refer to artificial episodes.
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
        time: int,
    ) -> bool:
        """
        checks whether the episode is terminated. By default, this checks whether the current time step exceeds the time
        horizon. In the continuous case, this can be used to define artificial episodes.

        Parameters
        ----------
        ts_t : dm_env.TimeStep
            The TimeStep at time t.
        a_t : "ACTION_TYPE"
            The action taken by the agent at time t.
        ts_tp1 : dm_env.TimeStep
            The TimeStep at time t + 1.
        time : int
            The current time of the environment. In the episodic case, this refers to the in-episode time, whereas in
            the continuous case this refers to the total number of previous interactions.

        Returns
        -------
        bool
            True if the episode terminated at time t+1.
        """
        return ts_tp1.last()

    def select_action(self, ts: dm_env.TimeStep, time: int) -> "ACTION_TYPE":
        """
        Parameters
        ----------
        ts : dm_env.TimeStep
            The TimeStep for which the agent is required to calculate the next action.
        time : int
            The current time of the environment. In the episodic case, this refers to the in-episode time, whereas in
            the continuous case this refers to the total number of previous interactions.

        Returns
        -------
        action : ACTION_TYPE
            The action that the agent suggests to take given the observation and the time step.
        """
        return self._actor.select_action(ts, time)

    @abc.abstractmethod
    def step_update(
        self,
        ts_t: dm_env.TimeStep,
        a_t: "ACTION_TYPE",
        ts_tp1: dm_env.TimeStep,
        time: int,
    ):
        """
        adds the transition in input to the MDP model.

        Parameters
        ----------
        ts_t : dm_env.TimeStep
            The TimeStep at time t.
        a_t : "ACTION_TYPE"
            The action taken by the agent at time t.
        ts_tp1 : dm_env.TimeStep
            The TimeStep at time t + 1.
        time : int
            The current time of the environment. In the episodic case, this refers to the in-episode time, whereas in
            the continuous case this refers to the total number of previous interactions.
        """
        if self._mdp_model:
            self._mdp_model.step_update(ts_t, a_t, ts_tp1, time)

    def agent_logs(self):
        """
        is called during the agent MDP interaction at lagging time. It can be used to log additional information.
        """
