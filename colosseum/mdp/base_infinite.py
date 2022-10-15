import abc
from typing import TYPE_CHECKING

from colosseum.mdp import BaseMDP

if TYPE_CHECKING:
    from colosseum.mdp import NODE_TYPE


class ContinuousMDP(BaseMDP, abc.ABC):
    """
    The base class for continuous MDPs.
    """

    @staticmethod
    def is_episodic() -> bool:
        return False

    def get_grid_representation(self, node: "NODE_TYPE", h: int=None):
        return super(ContinuousMDP, self)._get_grid_representation(node)
