from typing import Any, Dict, List, Tuple

import gin

from colosseum.mdp import ContinuousMDP
from colosseum.mdp.deep_sea.base import DeepSeaMDP, DeepSeaNode


@gin.configurable
class DeepSeaContinuous(ContinuousMDP, DeepSeaMDP):
    """
    The continuous DeepSea MDP class.
    """

    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return DeepSeaMDP.sample_mdp_parameters(n, False, seed)

    def custom_graph_layout(self) -> Dict[DeepSeaNode, Tuple[int, int]]:
        """
        Returns
        -------
        Dict[DeepSeaNode, Tuple[int, int]]
            The custom layout to draw a nx.Graph.
        """
        return {node: tuple(node) for node in self.G}
