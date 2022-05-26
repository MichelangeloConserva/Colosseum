import gin
import matplotlib.pyplot as plt
import networkx as nx

from colosseum.loops import human_loop
from colosseum.mdps import ContinuousMDP
from colosseum.mdps.river_swim.river_swim import RiverSwimMDP


@gin.configurable
class RiverSwimContinuous(ContinuousMDP, RiverSwimMDP):
    @property
    def _graph_layout(self):
        return {node: tuple(node) for node in self.G}


if __name__ == "__main__":

    mdp = RiverSwimContinuous(
        seed=42,
        randomize_actions=False,
        size=32,
        lazy=0.1,
        random_action_p=0.1,
        make_reward_stochastic=True,
    )

    print(mdp)

    nx.draw_networkx(mdp.G, mdp.graph_layout)
    plt.show()

    # random_loop(mdp, 50, verbose=True)
    human_loop(mdp)
