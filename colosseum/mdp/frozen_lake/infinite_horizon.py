from typing import Any, Dict, List

import gin

from colosseum.mdp import ContinuousMDP
from colosseum.mdp.frozen_lake.base import FrozenLakeMDP


@gin.configurable
class FrozenLakeContinuous(ContinuousMDP, FrozenLakeMDP):
    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return FrozenLakeMDP._sample_parameters(n, False, seed)


MDPClass = FrozenLakeContinuous
