from typing import Any, Dict, List

import gin

from colosseum.mdp import EpisodicMDP
from colosseum.mdp.deep_sea.base import DeepSeaMDP


@gin.configurable
class DeepSeaEpisodic(EpisodicMDP, DeepSeaMDP):
    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return DeepSeaMDP._sample_parameters(n, True, seed)

    def custom_graph_layout(self):
        return {node: (node.X, node.Y) for node in self.G}

    def __init__(self, *args, **kwargs):
        if "size" in kwargs:
            H = kwargs["size"]
        else:
            raise NotImplementedError(
                "The 'size' parameter should be given as a keyword parameter."
            )

        super(DeepSeaEpisodic, self).__init__(*args, H=H, **kwargs)


MDPClass = DeepSeaEpisodic
