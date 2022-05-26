from abc import ABC
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
import toolz

try:
    from functools import cached_property
except:
    from backports.cached_property import cached_property

from scipy.stats import rv_continuous

from colosseum.mdps import MDP
from colosseum.mdps.base_mdp import NextStateSampler
from colosseum.utils.random_vars import deterministic


@dataclass(frozen=True)
class CustomNode:
    ID: int

    def __str__(self):
        return str(self.ID + 1)


class CustomMDP(MDP, ABC):
    @staticmethod
    def testing_parameters() -> Dict[str, List]:
        t_params = MDP.testing_parameters()

        num_states = 4
        num_actions = 2

        T = np.zeros((num_states, num_actions, num_states), dtype=np.float32)
        T[0, 0, 1] = 1.0
        T[0, 1, 2] = 1.0

        T[1, 0, 2] = T[1, 0, 3] = 0.5
        T[1, 1, 2] = T[1, 1, 3] = 0.1
        T[1, 1, 1] = 0.8

        T[2, 0, 1] = T[2, 0, 3] = 0.5
        T[2, 1, 1] = T[2, 1, 3] = 0.1
        T[2, 1, 2] = 0.8

        T[3, 0, 0] = 0.5
        T[3, 0, 1] = T[3, 0, 2] = 0.25
        T[3, 1, 0] = 0.1
        T[3, 1, 1] = T[3, 1, 2] = 0.1
        T[3, 1, 3] = 0.7
        np.random.seed(42)
        R = np.random.rand(num_states, num_actions)
        T_0 = {0: 1.0}

        t_params["T_0"] = (T_0,)
        t_params["T"] = (T,)
        t_params["R"] = (R,)

        return t_params

    @staticmethod
    def get_node_class() -> Type[CustomNode]:
        return CustomNode

    def __init__(
        self,
        seed: int,
        T_0: Dict[int, float],
        T: np.ndarray,
        R: Union[np.ndarray, Dict[Tuple[int, int], rv_continuous]],
        lazy: Union[None, float] = None,
        randomize_actions: bool = True,
        **kwargs,
    ):
        """

        Parameters
        ----------
        seed : int
            the seed used for sampling rewards and next states.
        randomize_actions : bool, optional
            whether the effect of the actions changes for every node. It is particularly important to set this value to
             true when doing experiments to avoid immediately reaching highly rewarding states in some MDPs by just
             selecting the same action repeatedly. By default, it is set to true.
        lazy : float
            the probability of an action not producing any effect on the MDP.
        T_0 : Dict[int, float]
            the starting distribution. Note that the values of the dictionary should sum to one.
        T : np.ndarray
            the |S| x |A| x |S| transition distribution matrix.
        R : Union[np.ndarray, Dict[Tuple[int, int], rv_continuous]]
            the rewards can be either passed as a |S| x |A| array filled with deterministic values or with a dictionary of
            state action pairs and rv_continuous objects.
        """

        if type(R) == dict:
            self.R = {(CustomNode(ID=s), a): d for (s, a), d in R.items()}
        elif type(R) == np.ndarray:
            self.R = R
        else:
            raise NotImplementedError(
                f"The type of R, {type(R)}, is not accepted as input."
            )
        if type(T_0) == np.ndarray:
            self.T_0 = {CustomNode(ID=i): p for i, p in enumerate(T_0) if T_0[i] > 0}
        elif type(T_0) == dict:
            self.T_0 = toolz.keymap(lambda x: CustomNode(ID=x), T_0)
        else:
            raise NotImplementedError(
                f"The type of T_0, {type(T_0)}, is not accepted as input."
            )
        self.T = T
        self._num_actions = self.T.shape[1]

        super(CustomMDP, self).__init__(
            seed=seed,
            randomize_actions=randomize_actions,
            lazy=lazy,
            **kwargs,
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return super(CustomMDP, self).parameters

    @property
    def num_actions(self):
        return self._num_actions

    @cached_property
    def possible_starting_nodes(self) -> List[CustomNode]:
        return list(self.T_0.keys())

    def _calculate_next_nodes_prms(
        self, node: Any, action: int
    ) -> Tuple[Tuple[dict, float], ...]:
        return tuple(
            (dict(ID=next_node), self.T[node.ID, action, next_node])
            for next_node in range(len(self.T))
            if self.T[node.ID, action, next_node] > 0.0
        )

    def _calculate_reward_distribution(
        self, node: Any, action: IntEnum, next_node: Any
    ) -> rv_continuous:
        if type(self.R) == dict:
            return self.R[node, action]
        return deterministic(self.R[node.ID, action])

    def _check_input_parameters(self):
        super(CustomMDP, self)._check_input_parameters()

        assert self.T.ndim == 3
        assert type(self.R) in [dict, np.ndarray]
        assert np.isclose(np.sum(list(self.T_0.values())), 1)

        for s in range(len(self.T)):
            for a in range(self.T.shape[1]):
                assert np.isclose(self.T[s, a].sum(), 1), (
                    f"The transition kernel associated with state {s} and action {a} "
                    f"is not a well defined probability distribution."
                )

    def _instantiate_starting_node_sampler(self) -> NextStateSampler:
        return NextStateSampler(
            next_states=self.possible_starting_nodes,
            probs=list(self.T_0.values()),
            seed=self._next_seed(),
        )

    def merge_grid(self, grid, axis):
        indices = np.where(
            (grid == -1).sum(1 if axis == 0 else 0)
            == grid.shape[1 if axis == 0 else 0] - 1
        )[0][::2]
        for ind in indices:
            if axis == 1:
                grid = grid.T
            grid[ind + 1 : ind + 2][grid[ind : ind + 1] != -1] = grid[ind : ind + 1][
                grid[ind : ind + 1] != -1
            ]
            if axis == 1:
                grid = grid.T
        return np.delete(grid, indices, axis)

    @cached_property
    def grid(self) -> np.ndarray:
        coo = np.array(list(self.graph_layout.values()))

        X = sorted(coo[:, 0])
        Y = sorted(coo[:, 1])

        grid = np.zeros((len(coo), len(coo)), dtype=int) - 1

        for ind, (x, y) in enumerate(coo):
            grid[np.where(X == x)[0][0], np.where(Y == y)[0][0]] = ind

        has_changed = True
        while has_changed:
            has_changed = False
            if any((grid == -1).sum(0) == grid.shape[0] - 1):
                grid = self.merge_grid(grid, 1)
                has_changed = True
            if any((grid == -1).sum(1) == grid.shape[1] - 1):
                grid = self.merge_grid(grid, 0)
                has_changed = True

        return grid

    @lru_cache()
    def get_node_pos_in_grid(self, node) -> Tuple[int, int]:
        x, y = np.where((self.str_grid_node_order[node] == self.grid))
        return x[0], y[0]

    @cached_property
    def str_grid_node_order(self):
        return dict(zip(self.graph_layout.keys(), range(self.num_states)))

    def calc_grid_repr(self, node: Any) -> np.ndarray:
        str_grid = np.zeros(self.grid.shape[:2], dtype=str)
        str_grid[self.grid == -1] = "X"
        str_grid[self.grid != -1] = " "
        x, y = self.get_node_pos_in_grid(node)
        str_grid[x, y] = "A"
        return str_grid[::-1, :]
