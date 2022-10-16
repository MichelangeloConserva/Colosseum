from typing import Any, Dict, List

import gin

from colosseum.mdp import ContinuousMDP
from colosseum.mdp.minigrid_empty.base import MiniGridEmptyMDP


@gin.configurable
class MiniGridEmptyContinuous(ContinuousMDP, MiniGridEmptyMDP):
    """
    The continuous MiniGridEmpty MDP.
    """

    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return MiniGridEmptyMDP.sample_mdp_parameters(n, False, seed)
