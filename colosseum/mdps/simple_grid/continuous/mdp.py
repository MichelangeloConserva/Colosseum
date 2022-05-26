import gin
import matplotlib.pyplot as plt
import networkx as nx

from colosseum.loops import human_loop
from colosseum.mdps import ContinuousMDP
from colosseum.mdps.simple_grid.simple_grid import SimpleGridMDP, SimpleGridReward


@gin.configurable
class SimpleGridContinuous(ContinuousMDP, SimpleGridMDP):
    @property
    def _graph_layout(self):
        return {node: tuple(node) for node in self.G}


if __name__ == "__main__":
    mdp = SimpleGridContinuous(
        reward_type=SimpleGridReward.XOR,
        seed=42,
        size=10,
        randomize_actions=False,
        make_reward_stochastic=True,
        lazy=0.01,
        number_starting_states=1,
    )

    print(mdp.num_states)

    nx.draw_networkx(mdp.G, mdp.graph_layout)
    plt.show()
    # random_loop(mdp, 50, verbose=False)
    human_loop(mdp)
