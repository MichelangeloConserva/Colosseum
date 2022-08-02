from typing import Any, Dict, List

import gin

from colosseum.mdp import EpisodicMDP
from colosseum.mdp.river_swim.base import RiverSwimMDP


@gin.configurable
class RiverSwimEpisodic(EpisodicMDP, RiverSwimMDP):
    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return RiverSwimMDP._sample_parameters(n, True, seed)

    def custom_graph_layout(self):
        return {node: tuple(node) for node in self.G}


MDPClass = RiverSwimEpisodic
