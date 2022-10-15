from typing import Any, Dict, List, Tuple

import gin

from colosseum.mdp import ContinuousMDP
from colosseum.mdp.frozen_lake.base import FrozenLakeMDP, FrozenLakeNode


@gin.configurable
class FrozenLakeContinuous(ContinuousMDP, FrozenLakeMDP):
    """
    The FrozenLake continuous MDP.
    """

    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return FrozenLakeMDP.sample_mdp_parameters(n, False, seed)

    def custom_graph_layout(self) -> Dict[FrozenLakeNode, Tuple[int, int]]:
        """
        Returns
        -------
        Dict[FrozenLakeNode, Tuple[int, int]]
            The custom layout to draw a nx.Graph.
        """
        return {node: tuple(node) for node in self.G}


