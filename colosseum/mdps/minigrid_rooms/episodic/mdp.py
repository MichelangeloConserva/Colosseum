import time

import gin

from colosseum.loops import human_loop
from colosseum.mdps import EpisodicMDP
from colosseum.mdps.minigrid_rooms.minigrid_rooms import MiniGridRoomsMDP


@gin.configurable
class MiniGridRoomsEpisodic(EpisodicMDP, MiniGridRoomsMDP):
    pass


if __name__ == "__main__":
    start = time.time()
    mdp = MiniGridRoomsEpisodic(
        seed=42 + 1,
        randomize_actions=True,
        make_reward_stochastic=False,
        random_action_p=0.1,
        room_size=4,
        n_rooms=9,
    )
    print(mdp.num_states, mdp.H)
    print(time.time() - start)
    print(mdp)
    print(time.time() - start)

    # random_loop(mdp, 50, verbose=True)
    human_loop(mdp)
