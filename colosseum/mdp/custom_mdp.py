import abc
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Type, Union, TYPE_CHECKING

import numpy as np
import toolz
from scipy.stats import rv_continuous

from colosseum.mdp import BaseMDP, EpisodicMDP, ContinuousMDP
from colosseum.mdp.utils.custom_samplers import NextStateSampler
from colosseum.utils.miscellanea import deterministic

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE, NODE_TYPE


@dataclass(frozen=True)
class CustomNode:
    """
    The node for the CustomMDP.
    """

    ID: int
    """The id associated to the node."""

    def __str__(self):
        return str(self.ID + 1)


def _merge_grid(grid, axis):
    indices = np.where(
        (grid == -1).sum(1 if axis == 0 else 0) == grid.shape[1 if axis == 0 else 0] - 1
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


class CustomMDP(BaseMDP, abc.ABC):
    """
    The base class for the Custom MDP.
    """

    @staticmethod
    def get_unique_symbols() -> List[str]:
        return ["X", " ", "A"]

    @staticmethod
    def does_seed_change_MDP_structure() -> bool:
        raise NotImplementedError(
            "does_seed_change_MDP_structure is not implemented for the Custom MDP."
        )

    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "sample_parameters is not implemented for the Custom MDP."
        )

    @staticmethod
    def sample_mdp_parameters(
        n: int, is_episodic: bool, seed: int = None
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "sample_mdp_parameters is not implemented for the Custom MDP."
        )

    @staticmethod
    def get_node_class() -> Type["NODE_TYPE"]:
        return CustomNode

    @property
    def n_actions(self) -> int:
        return self._num_actions

    def _get_next_nodes_parameters(
        self, node: "NODE_TYPE", action: "ACTION_TYPE"
    ) -> Tuple[Tuple[dict, float], ...]:
        return tuple(
            (dict(ID=next_node), self.T[node.ID, action, next_node])
            for next_node in range(len(self.T))
            if self.T[node.ID, action, next_node] > 0.0
        )

    def _get_reward_distribution(
        self, node: "NODE_TYPE", action: "ACTION_TYPE", next_node: "NODE_TYPE"
    ) -> rv_continuous:
        if type(self.R) == dict:
            return self.R[node, action]
        return deterministic(self.R[node.ID, action])

    def _get_starting_node_sampler(self) -> NextStateSampler:
        return NextStateSampler(
            next_nodes=self._possible_starting_nodes,
            probs=list(self.T_0.values()),
            seed=self._produce_random_seed(),
        )

    def _check_parameters_in_input(self):
        super(CustomMDP, self)._check_parameters_in_input()

        assert self.T.ndim == 3
        assert type(self.R) in [dict, np.ndarray]
        assert np.isclose(np.sum(list(self.T_0.values())), 1)

        for s in range(len(self.T)):
            for a in range(self.T.shape[1]):
                assert np.isclose(self.T[s, a].sum(), 1), (
                    f"The transition kernel associated with state {s} and action {a} "
                    f"is not a well defined probability distribution."
                )

    def get_gin_parameters(self, index: int) -> str:
        raise NotImplementedError()

    @property
    def str_grid_node_order(self):
        if self._str_grid_node_order is None:
            self._str_grid_node_order = dict(
                zip(self.graph_layout.keys(), range(self.n_states))
            )
        return self._str_grid_node_order

    def get_node_pos_in_grid(self, node) -> Tuple[int, int]:
        """
        Returns
        -------
        Tuple[int, int]
            The position of the node in the visualization grid.
        """
        if node not in self._nodes_pos_in_grid:
            x, y = np.where((self.str_grid_node_order[node] == self.grid))
            self._nodes_pos_in_grid[node] = x[0], y[0]
        return self._nodes_pos_in_grid[node]

    @property
    def grid(self) -> np.ndarray:
        if self._grid is None:
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
                    grid = _merge_grid(grid, 1)
                    has_changed = True
                if any((grid == -1).sum(1) == grid.shape[1] - 1):
                    grid = _merge_grid(grid, 0)
                    has_changed = True

            self._grid = grid
        return self._grid

    def _get_grid_representation(self, node: "NODE_TYPE") -> np.ndarray:
        str_grid = np.zeros(self.grid.shape[:2], dtype=str)
        str_grid[self.grid == -1] = "X"
        str_grid[self.grid != -1] = " "
        x, y = self.get_node_pos_in_grid(node)
        str_grid[x, y] = "A"
        return str_grid[::-1, :]

    @property
    def _possible_starting_nodes(self) -> List["NODE_TYPE"]:
        return list(self.T_0.keys())

    @property
    def parameters(self) -> Dict[str, Any]:
        return super(CustomMDP, self).parameters

    def __init__(
        self,
        seed: int,
        T_0: Dict[int, float],
        T: np.ndarray,
        R: Union[np.ndarray, Dict[Tuple[int, int], rv_continuous]],
        **kwargs,
    ):
        """
        Parameters
        ----------
        seed : int
            the seed used for sampling rewards and next states.
        T_0 : Dict[int, float]
            the starting distribution. Note that the values of the dictionary should sum to one.
        T : np.ndarray
            the |S| x |A| x |S| transition distribution matrix.
        R : Union[np.ndarray, Dict[Tuple[int, int], rv_continuous]]
            the rewards can be either passed as a |S| x |A| array filled with deterministic values or with a dictionary of
            state action pairs and rv_continuous objects.
        """

        self.n_states, self._num_actions, _ = T.shape

        if type(R) == dict:
            _R = np.zeros((self.n_states, self._num_actions), np.float32)
            for (s, a), d in R.items():
                _R[s, a] = d.mean()
        elif type(R) == np.ndarray:
            pass
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

        self._transition_matrix_and_rewards = T, _R
        self._str_grid_node_order = None
        self._grid = None
        self._nodes_pos_in_grid = dict()

        super(CustomMDP, self).__init__(seed=seed, **kwargs)


class CustomEpisodic(CustomMDP, EpisodicMDP):
    """
    The episodic Custom MDP.
    """


class CustomContinuous(CustomMDP, ContinuousMDP):
    """
    The continuous Custom MDP.
    """
