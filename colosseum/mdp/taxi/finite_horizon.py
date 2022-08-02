from typing import Any, Dict, List

import gin

from colosseum.mdp import EpisodicMDP
from colosseum.mdp.taxi.base import TaxiMDP


@gin.configurable
class TaxiEpisodic(EpisodicMDP, TaxiMDP):
    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return TaxiMDP._sample_parameters(n, True, seed)


MDPClass = TaxiEpisodic
