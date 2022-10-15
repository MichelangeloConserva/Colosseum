import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from colosseum.emission_maps.base import EmissionMap

if TYPE_CHECKING:
    from colosseum.mdp import NODE_TYPE


class StateInfo(EmissionMap):
    """
    The `StateInfo` emission map creates a non-tabular vector containing uniquely identifying information about the
    state.
    """

    @property
    def is_tabular(self) -> bool:
        return False

    def node_to_observation(self, node: "NODE_TYPE", in_episode_time: int = None) -> np.ndarray:
        if self._mdp.is_episodic():
            in_episode_time = 0 if in_episode_time is None else in_episode_time
            return np.array((in_episode_time, *dataclasses.astuple(node))).astype(np.float32)
        return np.array(dataclasses.astuple(node)).astype(np.float32)
