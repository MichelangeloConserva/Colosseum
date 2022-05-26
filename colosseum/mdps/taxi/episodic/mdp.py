import time

import gin

from colosseum.loops import human_loop
from colosseum.mdps import EpisodicMDP
from colosseum.mdps.taxi.taxi import TaxiMDP


@gin.configurable
class TaxiEpisodic(EpisodicMDP, TaxiMDP):
    pass


if __name__ == "__main__":
    start = time.time()
    mdp = TaxiEpisodic(
        seed=42,
        randomize_actions=False,
        make_reward_stochastic=True,
        lazy=None,
        size=6,
        length=2,
        width=2,
        space=2,
        n_locations=2 ** 2,
    )
    # print(mdp.num_states, mdp.H)
    # print(time.time() - start)
    # print(mdp)
    # print(time.time() - start)

    human_loop(mdp)
