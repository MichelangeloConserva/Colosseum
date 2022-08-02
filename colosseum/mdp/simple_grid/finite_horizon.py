from typing import Any, Dict, List

import gin

from colosseum.mdp import EpisodicMDP
from colosseum.mdp.simple_grid.base import SimpleGridMDP


@gin.configurable
class SimpleGridEpisodic(EpisodicMDP, SimpleGridMDP):
    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return SimpleGridMDP._sample_parameters(n, True, seed)


MDPClass = SimpleGridEpisodic
