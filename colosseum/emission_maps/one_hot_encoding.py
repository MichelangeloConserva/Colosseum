from typing import TYPE_CHECKING

import numpy as np

from colosseum.emission_maps.base import EmissionMap

if TYPE_CHECKING:
    from colosseum.mdp import NODE_TYPE


class OneHotEncoding(EmissionMap):
    """
    The `OneHotEncoding` emission map creates a non-tabular vector filled with zero except the index corresponding to
    the state.
    """

    @property
    def is_tabular(self) -> bool:
        return False

    def node_to_observation(
        self, node: "NODE_TYPE", in_episode_time: int = None
    ) -> np.ndarray:
        index = self._mdp.node_to_index[node]
        obs = np.zeros(self._mdp.n_states, np.float32)
        obs[index] = 1.0
        return obs
