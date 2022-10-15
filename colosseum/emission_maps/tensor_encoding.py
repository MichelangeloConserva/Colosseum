from typing import TYPE_CHECKING, Type, Dict, Any

import numpy as np

from colosseum.emission_maps.base import EmissionMap
from colosseum.emission_maps.base import _get_symbol_mapping


if TYPE_CHECKING:
    from colosseum.mdp import BaseMDP, NODE_TYPE
    from colosseum.noises.base import Noise


class TensorEncoding(EmissionMap):
    """
    The `TensorEncoding` emission map creates a non-tabular 3d numpy array representation that one-hot encodes the
    visual elements in the visual representation of the MDP.
    """

    @property
    def is_tabular(self) -> bool:
        return False

    def __init__(
        self,
        mdp: "BaseMDP",
        noise_class: Type["Noise"],
        noise_kwargs: Dict[str, Any],
    ):
        self._symbol_mapping = None

        super(TensorEncoding, self).__init__(mdp, noise_class, noise_kwargs)

    def node_to_observation(self, node: "NODE_TYPE", in_episode_time: int = None) -> np.ndarray:
        if self._symbol_mapping is None:
            self._symbol_mapping = _get_symbol_mapping(self._mdp)

        grid = self._mdp.get_grid_representation(node, in_episode_time)
        if self._mdp.is_episodic():
            grid = grid[2:]

        obs = np.zeros((*grid.shape, len(self._symbol_mapping)), dtype=np.float32)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                obs[i, j, self._symbol_mapping[grid[i, j]]] = 1

        if self._mdp.is_episodic():
            return np.concatenate(
                (obs, np.zeros((*grid.shape, 1), np.float32) + in_episode_time), axis=-1
            )
        return obs
