from typing import TYPE_CHECKING, Type, Dict, Any

import numpy as np

from colosseum.emission_maps.base import EmissionMap
from colosseum.emission_maps.base import _get_symbol_mapping


if TYPE_CHECKING:
    from colosseum.mdp import BaseMDP, NODE_TYPE
    from colosseum.noises.base import Noise


class ImageEncoding(EmissionMap):
    """
    The `ImageEncoding` emission map creates a non-tabular matrix representation that encodes the visual representation
    of the MDP.
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

        super(ImageEncoding, self).__init__(mdp, noise_class, noise_kwargs)

    def node_to_observation(self, node: "NODE_TYPE", in_episode_time: int = None) -> np.ndarray:
        if self._symbol_mapping is None:
            self._symbol_mapping = _get_symbol_mapping(self._mdp)

        grid = self._mdp.get_grid_representation(node, in_episode_time)
        if self._mdp.is_episodic():
            grid = grid[2:]

        obs = np.array(
            list(map(np.vectorize(lambda x: self._symbol_mapping[x]), grid))
        ).astype(np.float32)
        if self._mdp.is_episodic():
            x = in_episode_time + np.zeros(obs.shape[1])
            return np.vstack((x, obs))
        return obs
