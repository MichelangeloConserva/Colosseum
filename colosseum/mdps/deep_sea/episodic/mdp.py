import time

import gin

from colosseum.loops import human_loop
from colosseum.mdps import EpisodicMDP
from colosseum.mdps.deep_sea.deep_sea import DeepSeaMDP


@gin.configurable
class DeepSeaEpisodic(EpisodicMDP, DeepSeaMDP):
    @property
    def _graph_layout(self):
        return {node: (node.X, node.Y) for node in self.G}

    def __init__(self, *args, **kwargs):
        if "size" in kwargs:
            H = kwargs["size"]
        else:
            raise NotImplementedError(
                "The 'size' parameter should be given as a keyword parameter."
            )

        super(DeepSeaEpisodic, self).__init__(*args, H=H, **kwargs)


if __name__ == "__main__":

    start = time.time()
    mdp = DeepSeaEpisodic(
        seed=42,
        size=20,
        randomize_actions=False,
        make_reward_stochastic=True,
        # random_action_p=0.1,
        force_single_thread=True
        # H=12,
    )
    print(mdp.H)
    print(time.time() - start)
    print(mdp)
    print(time.time() - start)

    # nx.draw_networkx(mdp.G, mdp.graph_layout)
    # plt.show()

    # random_loop(mdp, 50, verbose=True)
    human_loop(mdp)
