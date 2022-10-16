from typing import Any, Dict, List, Tuple

import gin

from colosseum.mdp import ContinuousMDP
from colosseum.mdp.river_swim.base import RiverSwimMDP, RiverSwimNode


@gin.configurable
class RiverSwimContinuous(ContinuousMDP, RiverSwimMDP):
    """
    The continuous RiverSwim MDP.
    """

    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return RiverSwimMDP.sample_mdp_parameters(n, False, seed)

    def custom_graph_layout(self) -> Dict[RiverSwimNode, Tuple[int, int]]:
        """
        Returns
        -------
        Dict[RiverSwimNode, Tuple[int, int]]
            The custom layout to draw a nx.Graph.
        """
        return {node: tuple(node) for node in self.G}
