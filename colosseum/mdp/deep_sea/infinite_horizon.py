from typing import Any, Dict, List

import gin

from colosseum.mdp import ContinuousMDP
from colosseum.mdp.deep_sea.base import DeepSeaMDP


@gin.configurable
class DeepSeaContinuous(ContinuousMDP, DeepSeaMDP):
    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return DeepSeaMDP._sample_parameters(n, False, seed)

    def custom_graph_layout(self):
        return {node: tuple(node) for node in self.G}


MDPClass = DeepSeaContinuous
