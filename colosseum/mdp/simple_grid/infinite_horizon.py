from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import gin

from colosseum.mdp import ContinuousMDP
from colosseum.mdp.simple_grid.base import SimpleGridMDP

if TYPE_CHECKING:
    from colosseum.mdp import NODE_TYPE

@gin.configurable
class SimpleGridContinuous(ContinuousMDP, SimpleGridMDP):
    @staticmethod
    def sample_parameters(n: int, seed: int = None) -> List[Dict[str, Any]]:
        return SimpleGridMDP.sample_mdp_parameters(n, False, seed)

    def custom_graph_layout(self) -> Dict["NODE_TYPE", Tuple[float, float]]:
        return {node: list(node) for node in self.G}

