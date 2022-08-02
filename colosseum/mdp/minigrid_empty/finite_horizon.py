from typing import Any, Dict, List

import gin

from colosseum.mdp import EpisodicMDP
from colosseum.mdp.minigrid_empty.base import MiniGridEmptyMDP


@gin.configurable
class MiniGridEmptyEpisodic(EpisodicMDP, MiniGridEmptyMDP):
    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return MiniGridEmptyMDP._sample_parameters(n, True, seed)


MDPClass = MiniGridEmptyEpisodic
