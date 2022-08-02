import abc

from colosseum.mdp import BaseMDP


class ContinuousMDP(BaseMDP, abc.ABC):
    @staticmethod
    def is_episodic() -> bool:
        return False

    def get_grid_representation(self, node: "NODE_TYPE", h : int):
        return super(ContinuousMDP, self)._get_grid_representation(node)