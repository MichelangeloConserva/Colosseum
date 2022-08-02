import abc
import dataclasses
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
from dm_env.specs import BoundedArray

if TYPE_CHECKING:
    from colosseum.mdp import NODE_TYPE
    from colosseum.mdp.base import BaseMDP


class StateRepresentationType(IntEnum):
    ONEHOT_STATE = 0
    LINEAR_STATE_OPTIMAL = 1
    LINEAR_STATE_RAND_PI = 2
    NON_LINEAR_VECTOR = 3
    NON_LINEAR_MATRIX = 4
    NON_LINEAR_TENSOR = 5


class RepresentationMapping(abc.ABC):
    @property
    @abc.abstractmethod
    def representation_type(self) -> StateRepresentationType:
        """
        returns the representation type of the mapping.
        """

    @abc.abstractmethod
    def _node_to_observation(self, node: "NODE_TYPE", h: int = None) -> np.ndarray:
        """
        returns the representation corresponding to the node given in input. Episodic MDPs also requires the current
        in-episode time step.
        """

    def __init__(self, mdp: "BaseMDP"):
        self._mdp = mdp
        self._cached_obs = dict()

    def get_observation(self, node: "NODE_TYPE", h: int = None) -> np.ndarray:
        if self._mdp.is_episodic() and h is None:
            h = self._mdp.h
        if not self._mdp.is_episodic():
            h = None
        if node not in self._cached_obs:
            self._cached_obs[h, node] = self._node_to_observation(node, h)
        return self._cached_obs[h, node]


class OneHotEncoding(RepresentationMapping):
    @property
    def representation_type(self) -> StateRepresentationType:
        return StateRepresentationType.ONEHOT_STATE

    def _node_to_observation(self, node: "NODE_TYPE", h: int = None) -> np.ndarray:
        index = self._mdp.node_to_index[node]
        obs = np.zeros(self._mdp.n_states, np.float32)
        obs[index] = 1.0
        return obs


class StateLinear(RepresentationMapping, abc.ABC):
    def __init__(self, mdp: "BaseMDP", d: int):
        super(StateLinear, self).__init__(mdp)
        self._d = d
        self._features = None

    @property
    @abc.abstractmethod
    def V(self):
        pass

    def _sample_features(self):
        self._features = _sample_linear_value_features(
            self.V, self._d, self._mdp.H if self._mdp.is_episodic() else None
        ).astype(np.float32)

    def _node_to_observation(self, node: "NODE_TYPE", h: int = None) -> np.ndarray:
        if self._features is None:
            self._sample_features()
        if h is not None:
            return self._features[h, self._mdp.node_to_index[node]]
        return self._features[self._mdp.node_to_index[node]]


class StateLinearOptimal(StateLinear):
    @property
    def representation_type(self) -> StateRepresentationType:
        return StateRepresentationType.LINEAR_STATE_OPTIMAL

    @property
    def V(self):
        return self._mdp.optimal_value[1].ravel()


class StateLinearRandom(StateLinear):
    @property
    def representation_type(self) -> StateRepresentationType:
        return StateRepresentationType.LINEAR_STATE_RAND_PI

    @property
    def V(self):
        return self._mdp.random_value[1].ravel()


class NodeInfo(RepresentationMapping):
    @property
    def representation_type(self) -> StateRepresentationType:
        return StateRepresentationType.NON_LINEAR_VECTOR

    def _node_to_observation(self, node: "NODE_TYPE", h: int = None) -> np.ndarray:
        return np.array(dataclasses.astuple(node)).astype(np.float32)


class GridMatrix(RepresentationMapping):
    @property
    def representation_type(self) -> StateRepresentationType:
        return StateRepresentationType.NON_LINEAR_MATRIX

    def __init__(self, mdp: "BaseMDP"):
        super(GridMatrix, self).__init__(mdp)
        self._symbol_mapping = None

    def _node_to_observation(self, node: "NODE_TYPE", h: int = None) -> np.ndarray:
        if self._symbol_mapping is None:
            symbols = np.unique(self._mdp.get_grid_representation(node, h))
            self._symbol_mapping = dict(zip(symbols, range(len(symbols))))

        grid = self._mdp.get_grid_representation(node, h)
        return np.array(
            list(map(np.vectorize(lambda x: self._symbol_mapping[x]), grid))
        ).astype(np.float32)


class TensorMatrix(RepresentationMapping):
    @property
    def representation_type(self) -> StateRepresentationType:
        return StateRepresentationType.NON_LINEAR_TENSOR

    def __init__(self, mdp: "BaseMDP"):
        super(TensorMatrix, self).__init__(mdp)
        self._symbol_mapping = None

    def _node_to_observation(self, node: "NODE_TYPE", h: int = None) -> np.ndarray:
        if self._symbol_mapping is None:
            symbols = np.unique(self._mdp.get_grid_representation(node, h))
            self._symbol_mapping = dict(zip(symbols, range(len(symbols))))

        grid = self._mdp.get_grid_representation(node, h)
        obs = np.zeros((*grid.shape, len(self._symbol_mapping)), dtype=np.float32)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                obs[i, j, self._symbol_mapping[grid[i, j]]] = 1
        return obs


def _sample_linear_value_features(v: np.ndarray, d: int, H: int = None):
    psi = np.random.randn(v.size, d)
    psi[:, 0] = 1
    psi[:, 1] = v

    P = psi @ np.linalg.inv(psi.T @ psi) @ psi.T

    W = np.random.randn(v.size, d)
    W[:, 0] = 1

    W_p = P @ W
    features = W_p / np.linalg.norm(W_p, axis=0, keepdims=True)
    if H is not None:
        features = features.reshape(H + 1, -1, d)
    return features
