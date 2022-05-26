import gin
import matplotlib.pyplot as plt
import networkx as nx

from colosseum.loops import human_loop
from colosseum.mdps import ContinuousMDP
from colosseum.mdps.deep_sea.deep_sea import DeepSeaMDP


@gin.configurable
class DeepSeaContinuous(ContinuousMDP, DeepSeaMDP):
    @property
    def _graph_layout(self):
        return {node: tuple(node) for node in self.G}


if __name__ == "__main__":
    mdp = DeepSeaContinuous(
        seed=42,
        size=5,
        randomize_actions=True,
        make_reward_stochastic=False,
        random_action_p=0.1,
    )
    print(mdp)

    nx.draw_networkx(mdp.G, mdp.graph_layout)
    plt.show()

    # random_loop(mdp, 50, verbose=True)
    human_loop(mdp)
