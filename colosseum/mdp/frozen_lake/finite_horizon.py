from typing import Any, Dict, List

import gin

from colosseum.mdp import EpisodicMDP
from colosseum.mdp.frozen_lake.base import FrozenLakeMDP


@gin.configurable
class FrozenLakeEpisodic(EpisodicMDP, FrozenLakeMDP):
    """
    The FrozenLake episodic MDP.
    """

    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return FrozenLakeMDP.sample_mdp_parameters(n, True, seed)
