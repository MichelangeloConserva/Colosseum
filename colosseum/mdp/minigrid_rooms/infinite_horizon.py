from typing import Any, Dict, List

import gin

from colosseum.mdp import ContinuousMDP
from colosseum.mdp.minigrid_rooms.base import MiniGridRoomsMDP


@gin.configurable
class MiniGridRoomsContinuous(ContinuousMDP, MiniGridRoomsMDP):
    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return MiniGridRoomsMDP.sample_mdp_parameters(n, False, seed)

