from typing import Any, Dict, List

import gin

from colosseum.mdp import ContinuousMDP
from colosseum.mdp.minigrid_empty.base import MiniGridEmptyMDP


@gin.configurable
class MiniGridEmptyContinuous(ContinuousMDP, MiniGridEmptyMDP):
    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return MiniGridEmptyMDP._sample_parameters(n, False, seed)


MDPClass = MiniGridEmptyContinuous
