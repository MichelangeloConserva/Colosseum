from typing import Any, Dict, List

import gin

from colosseum.mdp import ContinuousMDP
from colosseum.mdp.taxi.base import TaxiMDP


@gin.configurable
class TaxiContinuous(ContinuousMDP, TaxiMDP):
    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return TaxiMDP.sample_mdp_parameters(n, False, seed)
