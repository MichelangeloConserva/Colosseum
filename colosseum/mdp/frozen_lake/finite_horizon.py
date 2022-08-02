from typing import Any, Dict, List

import gin

from colosseum.mdp import EpisodicMDP
from colosseum.mdp.frozen_lake.base import FrozenLakeMDP


@gin.configurable
class FrozenLakeEpisodic(EpisodicMDP, FrozenLakeMDP):
    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return FrozenLakeMDP._sample_parameters(n, True, seed)


MDPClass = FrozenLakeEpisodic
