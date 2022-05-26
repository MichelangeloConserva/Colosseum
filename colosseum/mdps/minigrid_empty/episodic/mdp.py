import time

import gin

from colosseum.mdps import EpisodicMDP
from colosseum.mdps.minigrid_empty.minigrid_empty import MiniGridEmptyMDP


@gin.configurable
class MiniGridEmptyEpisodic(EpisodicMDP, MiniGridEmptyMDP):
    pass


if __name__ == "__main__":
    start = time.time()
    mdp = MiniGridEmptyEpisodic(
        seed=42,
        size=6,
        randomize_actions=False,
        make_reward_stochastic=True,
        lazy=0.01,
    )
    print((isinstance(mdp, MiniGridEmptyEpisodic)))

    print(mdp.num_states, mdp.H)
    print(time.time() - start)
    print(mdp)
    print(time.time() - start)

    # nx.draw_networkx(mdp.G, mdp.graph_layout)
    # plt.show()

    # random_loop(mdp, 50, verbose=True)
    # human_loop(mdp)
