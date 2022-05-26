import time

import gin

from colosseum.mdps import EpisodicMDP
from colosseum.mdps.minigrid_doorkey.minigrid_doorkey import MiniGridDoorKeyMDP


@gin.configurable
class MiniGridDoorKeyEpisodic(EpisodicMDP, MiniGridDoorKeyMDP):
    pass


if __name__ == "__main__":
    start = time.time()
    mdp = MiniGridDoorKeyEpisodic(
        # H=30,
        seed=1,
        randomize_actions=True,
        make_reward_stochastic=True,
        lazy=0.05,
        size=5,
        random_action_p=0.1,
        verbose=True,
        force_single_thread=True,
    )

    mdp.diameter

    # random_loop(mdp, 50, verbose=False)
    # human_loop(mdp)
