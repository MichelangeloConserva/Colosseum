import gin

from colosseum.loops import human_loop
from colosseum.mdps import ContinuousMDP
from colosseum.mdps.minigrid_empty.minigrid_empty import MiniGridEmptyMDP


@gin.configurable
class MiniGridEmptyContinuous(ContinuousMDP, MiniGridEmptyMDP):
    pass


if __name__ == "__main__":
    mdp = MiniGridEmptyContinuous(
        seed=42,
        randomize_actions=False,
        make_reward_stochastic=False,
        lazy=0.01,
        size=4,
    )

    # nx.draw_networkx(mdp.G, mdp.graph_layout)
    # plt.show()

    # random_loop(mdp, 50, verbose=True)
    human_loop(mdp)
