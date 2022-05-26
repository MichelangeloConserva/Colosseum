from typing import Dict, Tuple, Union

import numpy as np


class ContinuousPolicy:
    """
    stores the policy of an infinite horizon agent.
    """

    def __init__(self, policy: Dict[int, Union[int, np.ndarray]], num_actions: int):
        """

        Parameters
        ----------
        policy : Dict[int, Union[int, np.ndarray]]
            the deterministic or stochastic policy of the agent.
        num_actions : int
            the number of actions.
        """
        if type(list(policy.values())[0]) in [
            int,
            np.int8,
            np.int16,
            np.int64,
            np.int32,
        ]:
            self.is_deterministic = True
        else:
            self.is_deterministic = False
            assert type(list(policy.values())[0]) == np.ndarray

        self._policy = policy
        self._num_actions = num_actions
        self._num_states = len(policy.keys())
        self._pi_matrix = None
        self._pi = dict()

    def pi(self, s: int) -> np.array:
        """
        Returns the policy for a given state.
        """
        if s not in self._pi:
            if self.is_deterministic:
                p = np.zeros(self._num_actions, np.float32)
                p[self._policy[s]] = 1.0
                self._pi[s] = p
            self._pi[s] = self._policy[s]
        return self._pi[s]

    @property
    def pi_matrix(self) -> np.ndarray:
        """
        Returns a matrix |S| x |A| containing the policy of the agent.
        """
        if self._pi_matrix is None:
            self._pi_matrix = np.zeros(
                (self._num_states, self._num_actions), np.float32
            )
            for s in self._policy:
                self._pi_matrix[s] = self.pi(s)
        return self._pi_matrix

    def __hash__(self) -> int:
        return hash(tuple(self._pi_matrix.tolist()))


class EpisodicPolicy:
    """
    Stores the policy of a finite horizon agent.
    """

    def __init__(
        self, policy: Dict[Tuple[int, int], Union[int, np.ndarray]], num_actions, H
    ):
        """

        Parameters
        ----------
        policy : Dict[int, Union[int, np.ndarray]]
            the deterministic or stochastic policy of the agent.
        num_actions : int
            the number of actions.
        H : int
            the horizon of the MDP.
        """
        if type(list(policy.values())[0]) in [int, np.int64, np.int32]:
            self.is_deterministic = True
        else:
            self.is_deterministic = False
            assert type(list(policy.values())[0]) == np.ndarray

        self._policy = policy
        self._num_actions = num_actions
        self._H = H
        self._num_states = len(policy.keys())
        self._pi_matrix = None
        self._pi = dict()

    def pi(self, h: int, s: int) -> np.array:
        """
        Returns the policy for a given state and time step.
        """
        if (h, s) not in self._pi:
            if self.is_deterministic:
                p = np.zeros(self._num_actions, np.float32)
                p[self._policy[h, s]] = 1.0
                self._pi[h, s] = p
            else:
                self._pi[h, s] = self._policy[h, s]
        return self._pi[h, s]

    @property
    def pi_matrix(self) -> np.ndarray:
        """
        Returns a matrix |H| x |S| x |A| containing the policy of the agent.
        """
        if self._pi_matrix is None:
            self._pi_matrix = np.zeros(
                (self._H, self._num_states, self._num_actions), np.float32
            )
            for h, s in self._policy:
                self._pi_matrix[h, s] = self.pi(h, s)
        return self._pi_matrix

    def __hash__(self) -> int:
        return hash(tuple(self._pi_matrix.tolist()))
