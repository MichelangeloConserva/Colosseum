import time

import gin

from colosseum.mdps import EpisodicMDP
from colosseum.mdps.frozen_lake.frozen_lake import FrozenLakeMDP


@gin.configurable
class FrozenLakeEpisodic(EpisodicMDP, FrozenLakeMDP):
    pass


if __name__ == "__main__":

    start = time.time()
    mdp = FrozenLakeEpisodic(
        seed=42,
        randomize_actions=True,
        make_reward_stochastic=False,
        lazy=None,
        size=12,
        p_frozen=0.9,
    )
    print(mdp.num_states, mdp.H)
    print(time.time() - start)
    print(mdp)
    print(time.time() - start)
    #
    # print(mdp)

    # random_loop(mdp, 50, verbose=False)
    # human_loop(mdp)
