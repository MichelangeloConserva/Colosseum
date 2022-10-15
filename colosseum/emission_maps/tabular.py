from typing import TYPE_CHECKING

import numpy as np

from colosseum.emission_maps.base import EmissionMap

if TYPE_CHECKING:
    from colosseum.mdp import BaseMDP, NODE_TYPE


class Tabular(EmissionMap):
    """
    The `Tabular` emission map is just the tabular mapping.
    """

    @property
    def is_tabular(self) -> bool:
        return True

    def __init__(self, mdp: "BaseMDP"):
        super(Tabular, self).__init__(mdp, None, None)
        self._mdp = mdp
        self._cached_obs = dict()

    def node_to_observation(self, node: "NODE_TYPE", in_episode_time: int = None) -> np.ndarray:
        raise NotImplementedError()

    def get_observation(self, state: "NODE_TYPE", in_episode_time: int = None) -> np.ndarray:
        raise NotImplementedError()
