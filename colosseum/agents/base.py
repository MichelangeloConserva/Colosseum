import abc
import random
from abc import ABC, abstractmethod
from itertools import product
from typing import Union

import dm_env
import numpy as np

from colosseum.agents.policy import ContinuousPolicy, EpisodicPolicy
from colosseum.utils.acme.specs import EnvironmentSpec


class Agent(ABC):
    """
    Base class for agents.
    """

    @staticmethod
    @abstractmethod
    def is_episodic() -> bool:
        """
        Returns whether the agent is episodic.
        """

    @property
    @abstractmethod
    def current_policy(self) -> Union[ContinuousPolicy, EpisodicPolicy]:
        """
        Returns the current policy of the agent.
        """

    @abstractmethod
    def __init__(
        self,
        environment_spec: EnvironmentSpec,
        seed: int,
        r_max: float,
        T: int,
        store_counts: bool,
        save_episode_data: bool,
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
        store_counts : bool
            checks whether the visitation counts should be stored.
        save_episode_data : bool
            check whether the samples from state action pairs visitation should be stored.
        """
        self.T = T
        self.seed = seed
        self.r_max = r_max
        self.store_counts = store_counts
        self.save_episode_data = save_episode_data

        self.steps = 0
        self.s_tm1 = None
        self.episode_reward_data = dict()
        self.episode_transition_data = dict()

        self._rng = np.random.RandomState(seed)
        self._fast_rng = random.Random(seed)
        self.num_states = environment_spec.observations.num_values
        self.num_actions = environment_spec.actions.num_values

        if store_counts:
            self.N = np.zeros(
                (self.num_states, self.num_actions, self.num_states), dtype=np.int32
            )

    def debug_info(self):
        """
        This methods is called at every log of the MDPLoop run function.
        It can be used for debugging purposes.
        """

    def store_episode_data(self, s_tm1, action, r, s_t):
        """
        Store the transition and update the counts.
        """

        if self.store_counts:
            self.N[s_tm1, action, s_t] += 1

        if (s_tm1, action) in self.episode_reward_data:
            self.episode_reward_data[s_tm1, action].append(r)
            if s_t != -1:
                self.episode_transition_data[s_tm1, action].append(s_t)
        else:
            self.episode_reward_data[s_tm1, action] = [r]
            self.episode_transition_data[s_tm1, action] = [s_t] if s_t != -1 else []

    @abc.abstractmethod
    def select_action(self, observation: int) -> int:
        """Samples from the policy and returns an action."""

    def observe(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        next_timestep: dm_env.TimeStep,
    ):
        """
        Allows the agent to gather information on the interactions between the agent and the MDP.
        Note that it may be useful to overwrite it if the agent changes its policy at every interaction.
        See the implementation of Q-learning for the episodic case.
        """
        if self.save_episode_data:
            self.store_episode_data(
                timestep.observation,
                action,
                next_timestep.reward,
                next_timestep.observation,
            )
        self.s_tm1 = next_timestep.observation

        self.h += 1
        self.steps += 1

    def before_new_episode(self):
        """
        Called before a new episode starts. In the infinite horizon setting,
        this is called when a new artificial starts if such functionality is implemented.
        """
        self.h = 0

        if len(self.episode_transition_data) > 0:
            self.update_models()
            self.episode_reward_data = dict()
            self.episode_transition_data = dict()

        self._before_new_episode()

    @abstractmethod
    def update_models(self):
        """
        Called before a new episode (or artificial episode starts).
        Its main purpose it to let the agent update its model for transition and rewards using the data collected
        during the interactions in the previous episode, i.e. self.episode_reward_data and self.episode_transition_data.
        """

    @abstractmethod
    def _before_new_episode(self):
        """
        Called before a new episode starts. It is useful to perform the calculations of the new policy or to update
        other important quantities from the data collected during the interactions in the previous episode.
        """

    @abstractmethod
    def is_episode_end(
        self, timestep: dm_env.TimeStep, action: int, new_timestep: dm_env.TimeStep
    ) -> bool:
        """
        Has to be implemented only for continuous mdp and should return True if the artificial episode is ended.
        Note that after the artificial episode is ended 'before_new_episode' will be called.
        """


class ContinuousHorizonBaseActor(Agent, ABC):
    """
    The Base class for infinite horizon agents.
    """

    @staticmethod
    def is_episodic():
        return False

    @abstractmethod
    def __init__(
        self,
        environment_spec: EnvironmentSpec,
        seed: int,
        r_max: float,
        T: int,
        store_counts=True,
        save_episode_data=True,
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
        store_counts : bool, optional
            checks whether the visitation counts should be stored. By default, it is set to True.
        save_episode_data : bool, optional
            check whether the samples from state action pairs visitation should be stored. By default, it is set to True.
        """
        super(ContinuousHorizonBaseActor, self).__init__(
            environment_spec, seed, r_max, T, store_counts, save_episode_data
        )


class FiniteHorizonBaseActor(Agent, ABC):
    """
    The Base class for finite horizon agents.
    """

    @staticmethod
    def is_episodic():
        return True

    def __init__(
        self,
        environment_spec: EnvironmentSpec,
        seed: int,
        H: int,
        r_max: float,
        T: int,
        store_counts=True,
        save_episode_data=True,
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
        store_counts : bool, optional
            checks whether the visitation counts should be stored. By default, it is set to True.
        save_episode_data : bool, optional
            check whether the samples from state action pairs visitation should be stored. By default, it is set to True.
        """
        super(FiniteHorizonBaseActor, self).__init__(
            environment_spec, seed, r_max, T, store_counts, save_episode_data
        )
        self.H = H
        self.h = 0

    def is_episode_end(
        self, timestep: dm_env.TimeStep, action: int, new_timestep: dm_env.TimeStep
    ):
        return new_timestep.last()


class ValueBasedAgent(Agent, ABC):
    """
    Utility class for agents that acts based on estimates of the value function.
    """

    @property
    def current_policy(self) -> Union[ContinuousPolicy, EpisodicPolicy]:
        if self.is_episodic():
            return EpisodicPolicy(
                {
                    (h, s): self.select_action(s, h)
                    for h, s in product(range(self.H), range(self.num_states))
                },
                self.num_actions,
                self.H,
            )
        return ContinuousPolicy(
            {s: self.select_action(s) for s in range(self.num_states)}, self.num_actions
        )

    def select_action(self, observation: int, h: int = None) -> int:
        if self.is_episodic() and h is None:
            h = self.h
        Q = self.Q[h, observation] if self.is_episodic() else self.Q[observation]
        try:
            action = self._fast_rng.choice(np.where(Q == Q.max())[0])
        except:
            raise ValueError(np.where(Q == Q.max())[0], Q.max())
        return int(action)
