from typing import Any, Dict, List, Tuple

import gin

from colosseum.mdp import EpisodicMDP
from colosseum.mdp.deep_sea.base import DeepSeaMDP, DeepSeaNode


@gin.configurable
class DeepSeaEpisodic(EpisodicMDP, DeepSeaMDP):
    """
    The episodic DeepSea MDP class.
    """

    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return DeepSeaMDP.sample_mdp_parameters(n, True, seed)

    def custom_graph_layout(self) -> Dict[DeepSeaNode, Tuple[int, int]]:
        """
        Returns
        -------
        Dict[DeepSeaNode, Tuple[int, int]]
            The custom layout to draw a nx.Graph.
        """
        return {node: (node.X, node.Y) for node in self.G}

    def __init__(self, *args, **kwargs):
        if "size" in kwargs:
            H = kwargs["size"]
        else:
            raise NotImplementedError(
                "The 'size' parameter should be given as a keyword parameter."
            )

        super(DeepSeaEpisodic, self).__init__(*args, H=H, **kwargs)


