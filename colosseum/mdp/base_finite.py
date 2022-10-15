import abc
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import networkx as nx
import numpy as np

from colosseum.dynamic_programming import discounted_policy_iteration
from colosseum.dynamic_programming import discounted_value_iteration
from colosseum.dynamic_programming import episodic_policy_evaluation
from colosseum.dynamic_programming import episodic_value_iteration
from colosseum.dynamic_programming.utils import get_policy_from_q_values
from colosseum.mdp import BaseMDP
from colosseum.mdp.utils.mdp_creation import (
    get_continuous_form_episodic_transition_matrix_and_rewards,
)
from colosseum.mdp.utils.mdp_creation import get_episodic_graph
from colosseum.mdp.utils.mdp_creation import get_episodic_transition_matrix_and_rewards


if TYPE_CHECKING:
    from colosseum.mdp import NODE_TYPE


class EpisodicMDP(BaseMDP, abc.ABC):
    """
    The base class for episodic MDPs.
    """

    @staticmethod
    def is_episodic() -> bool:
        return True

    @property
    def H(self) -> int:
        """
        Returns
        -------
        int
            The episode length.
        """
        if self._H is None:
            self._set_time_horizon(self._input_H)
        return self._H


    @property
    def random_policy_cf(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            The random policy for the continuous form the episodic MDP.
        """
        if self._random_policy_cf is None:
            self._random_policy_cf = (
                np.ones(
                    (len(self.get_episodic_graph(True).nodes), self.n_actions),
                    np.float32,
                )
                / self.n_actions
            )
        return self._random_policy_cf

    @property
    def random_policy(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            The random uniform policy.
        """
        if self._random_policy is None:
            self._random_policy = (
                np.ones((self.H, self.n_states, self.n_actions), np.float32)
                / self.n_actions
            )
        return self._random_policy

    def __init__(self, H: int = None, **kwargs):
        super(EpisodicMDP, self).__init__(**kwargs)

        # Computing the time horizon
        self._input_H = H
        self._H = None

        # Episodic setting specific caching variables
        self._reachable_states = None
        self._episodic_graph = dict()
        self._continuous_form_episodic_transition_matrix_and_rewards = None
        self._episodic_transition_matrix_and_rewards = None
        self._optimal_policy_cf = dict()
        self._worst_policy_cf = dict()
        self._optimal_value_cf = None
        self._worst_value_cf = None
        self._random_value_cf = None
        self._eoar = None
        self._woar = None
        self._roar = None
        self._random_policy_cf = None
        self._random_policy = None
        self._average_optimal_episodic_reward = None
        self._average_worst_episodic_reward = None
        self._average_random_episodic_reward = None


    def _set_time_horizon(self, H: int) -> int:
        """
        sets a meaningful minimal horizon for the MDP.
        """
        if "Taxi" in str(type(self)):
            # it is complicated to give the same horizon to different seed of the same MDP instance
            # for the Taxi MDP
            minimal_H = int(1.5 * self._size ** 2)
        else:
            minimal_H = (
                max(
                    max(nx.shortest_path_length(self.G, sn).values())
                    for sn in self._possible_starting_nodes
                )
                + 1
            )
        if H is None:
            self._H = self._H = minimal_H
        else:
            self._H = self._H = max(minimal_H, H)

    def _vi(self, *args):
        return episodic_value_iteration(self.H, *args)

    def _pe(self, *args):
        return episodic_policy_evaluation(self.H, *args)

    @property
    def parameters(self) -> Dict[str, Any]:
        parameters = super(EpisodicMDP, self).parameters
        if not self._exclude_horizon_from_parameters:
            parameters["H"] = self.H
        return parameters

    @property
    def reachable_states(self) -> List[Tuple[int, "NODE_TYPE"]]:
        """
        Returns
        -------
        List[Tuple[int, "NODE_TYPE"]]
            The pairs of in episode time step and states that is possible to reach with the given episode time.
        """
        if self._reachable_states is None:
            self._reachable_states = [
                (h, self.node_to_index[n])
                for h, n in self.get_episodic_graph(False).nodes
            ]
        return self._reachable_states

    @property
    def T_cf(self) -> np.ndarray:
        """
        is an alias for the continuous form of the transition matrix.
        """
        return self.continuous_form_episodic_transition_matrix_and_rewards[0]

    @property
    def R_cf(self) -> np.ndarray:
        """
        is an alias for the continuous form of the rewards matrix.
        """
        return self.continuous_form_episodic_transition_matrix_and_rewards[1]

    @property
    def optimal_value_continuous_form(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        np.ndarray
            The q-value function of the optimal policy for the continuous form of the MDP.
        np.ndarray
            The state-value function of the optimal policy for the continuous form of the MDP.
        """
        if self._optimal_value_cf is None:
            self._optimal_value_cf = discounted_value_iteration(self.T_cf, self.R_cf)
        return self._optimal_value_cf

    @property
    def worst_value_continuous_form(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            The q-value function of the worst policy for the continuous form of the MDP.
        np.ndarray
            The state-value function of the worst policy for the continuous form of the MDP.
        """
        if self._worst_value_cf is None:
            self._worst_value_cf = discounted_value_iteration(self.T_cf, -self.R_cf)
        return self._worst_value_cf

    @property
    def random_value_continuous_form(self):
        """
        Returns
        -------
        np.ndarray
            The q-value function of the random uniform policy for the continuous form of the MDP.
        np.ndarray
            The state-value function of the random uniform policy for the continuous form of the MDP.
        """
        if self._random_value_cf is None:
            self._random_value_cf = discounted_policy_iteration(
                self.T_cf, self.R_cf, self.random_policy_cf
            )
        return self._random_value_cf


    @property
    def episodic_optimal_average_reward(self) -> float:
        """
        Returns
        -------
        float
            The average episodic reward for the optimal policy.
        """
        if self._eoar is None:
            _eoar = 0.0
            for sn, p in self._starting_node_sampler.next_nodes_and_probs:
                _eoar += p * self.get_optimal_policy_starting_value(sn)
            self._eoar = _eoar / self.H
        return self._eoar


    @property
    def episodic_worst_average_reward(self) -> float:
        """
        Returns
        -------
        float
            The average episodic reward for the worst policy.
        """
        if self._woar is None:
            _woar = 0.0
            for sn, p in self._starting_node_sampler.next_nodes_and_probs:
                _woar += p * self.get_worst_policy_starting_value(sn)
            self._woar = _woar / self.H
        return self._woar

    @property
    def episodic_random_average_reward(self) -> float:
        """
        Returns
        -------
        float
            The average episodic reward for the random uniform policy.
        """
        if self._roar is None:
            _roar = 0.0
            for sn, p in self._starting_node_sampler.next_nodes_and_probs:
                _roar += p * self.get_random_policy_starting_value(sn)
            self._roar = _roar / self.H
        return self._roar

    @property
    def continuous_form_episodic_transition_matrix_and_rewards(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        np.ndarray
            The transition 3d array for the continuous form of the MDP.
        np.ndarray
            The reward matrix for the continuous form of the MDP.
        """
        if self._continuous_form_episodic_transition_matrix_and_rewards is None:
            self._continuous_form_episodic_transition_matrix_and_rewards = (
                get_continuous_form_episodic_transition_matrix_and_rewards(
                    self.H,
                    self.get_episodic_graph(True),
                    *self.transition_matrix_and_rewards,
                    self._starting_node_sampler,
                    self.node_to_index,
                )
            )
        return self._continuous_form_episodic_transition_matrix_and_rewards

    @property
    def episodic_transition_matrix_and_rewards(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        np.ndarray
            The transition 3d array for the MDP.
        np.ndarray
            The reward matrix for the MDP.
        """
        if self._episodic_transition_matrix_and_rewards is None:
            self._episodic_transition_matrix_and_rewards = (
                get_episodic_transition_matrix_and_rewards(
                    self.H,
                    *self.transition_matrix_and_rewards,
                    self._starting_node_sampler,
                    self.node_to_index,
                )
            )
        return self._episodic_transition_matrix_and_rewards

    def get_optimal_policy_continuous_form(self, stochastic_form: bool) -> np.ndarray:
        """
        Returns
        ------
        np.ndarray
            The optimal policy computed for the continuous form.
        """
        if stochastic_form not in self._optimal_policy_cf:
            self._optimal_policy_cf[stochastic_form] = get_policy_from_q_values(
                self.optimal_value_continuous_form[0], stochastic_form
            )
        return self._optimal_policy_cf[stochastic_form]

    def get_worst_policy_continuous_form(self, stochastic_form) -> np.ndarray:
        """
        Returns
        ------
        np.ndarray
            The worst policy computed for the continuous form.
        """
        if stochastic_form not in self._worst_policy_cf:
            self._worst_policy_cf[stochastic_form] = get_policy_from_q_values(
                self.worst_value_continuous_form[0], stochastic_form
            )
        return self._worst_policy_cf[stochastic_form]

    def get_random_policy_continuous_form(self, stochastic_form) -> np.ndarray:
        """
        Returns
        ------
        np.ndarray
            The random uniform policy computed for the continuous form.
        """
        if stochastic_form not in self._worst_policy_cf:
            self._random_policy_cf[stochastic_form] = get_policy_from_q_values(
                self.random_value_continuous_form[0], stochastic_form
            )
        return self._random_policy_cf[stochastic_form]

    def get_minimal_regret_for_starting_node(self, node: "NODE_TYPE") -> float:
        """
        Returns
        -------
        float
            The minimal possible regret obtained from the given starting state.
        """
        return self.get_optimal_policy_starting_value(
            node
        ) - self.get_worst_policy_starting_value(node)

    def get_optimal_policy_starting_value(self, node: "NODE_TYPE") -> float:
        """
        Returns
        -------
        float
            The value of the given state at in episode time step zero for the optimal policy.
        """
        return self.optimal_value_functions[1][0, self.node_to_index[node]]

    def get_worst_policy_starting_value(self, node: "NODE_TYPE"):
        """
        Returns
        -------
        float
            The value of the given state at in episode time step zero for the worst policy.
        """
        return self.worst_value_functions[1][0, self.node_to_index[node]]

    def get_random_policy_starting_value(self, node: "NODE_TYPE"):
        """
        Returns
        -------
        float
            The value of the given state at in episode time step zero for the random uniform policy.
        """
        return self.random_value_functions[1][0, self.node_to_index[node]]

    def get_episodic_graph(self, remove_labels: bool) -> nx.DiGraph:
        """
        Returns
        -------
        The graph corresponding the state space augmented with the in episode time step. It is possible to remove
        the labels that mark the nodes.
        """
        if remove_labels not in self._episodic_graph:
            self._episodic_graph[remove_labels] = get_episodic_graph(
                self.G, self.H, self.node_to_index, self.starting_nodes, remove_labels
            )
        return self._episodic_graph[remove_labels]

    def get_grid_representation(self, node: "NODE_TYPE", h: int = None) -> np.array:
        if h is None:
            h = self.h
        grid = self._get_grid_representation(node)
        while grid.shape[1] < 2 + len(str(self.h)):
            adder = np.zeros((grid.shape[1], 1), dtype=str)
            adder[:] = "X"
            grid = np.hstack((grid, adder))
        title = np.array(
            [" "] * grid.shape[1] + ["_"] * grid.shape[1], dtype=str
        ).reshape(2, -1)
        title[0, 0] = "H"
        title[0, 1] = "="
        for i, l in enumerate(str(h)):
            title[0, 2 + i] = l
        return np.vstack((title, grid))
