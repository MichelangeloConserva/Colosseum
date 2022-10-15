from typing import Any, Dict, List

import gin

from colosseum.mdp import EpisodicMDP
from colosseum.mdp.minigrid_empty.base import MiniGridEmptyMDP


@gin.configurable
class MiniGridEmptyEpisodic(EpisodicMDP, MiniGridEmptyMDP):
    """
    The episodic MiniGridEmpty MDP.
    """

    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return MiniGridEmptyMDP.sample_mdp_parameters(n, True, seed)


