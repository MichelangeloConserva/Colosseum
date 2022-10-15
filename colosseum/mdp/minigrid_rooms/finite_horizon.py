from typing import Any, Dict, List

import gin

from colosseum.mdp import EpisodicMDP
from colosseum.mdp.minigrid_rooms.base import MiniGridRoomsMDP


@gin.configurable
class MiniGridRoomsEpisodic(EpisodicMDP, MiniGridRoomsMDP):
    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return MiniGridRoomsMDP.sample_mdp_parameters(n, True, seed)


