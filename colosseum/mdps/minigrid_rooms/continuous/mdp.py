import gin

from colosseum.loops import human_loop
from colosseum.mdps import ContinuousMDP
from colosseum.mdps.minigrid_rooms.minigrid_rooms import MiniGridRoomsMDP


@gin.configurable
class MiniGridRoomsContinuous(ContinuousMDP, MiniGridRoomsMDP):
    pass


if __name__ == "__main__":
    mdp = MiniGridRoomsContinuous(
        seed=0,
        randomize_actions=False,
        make_reward_stochastic=True,
        random_action_p=0.1,
        room_size=4,
        n_rooms=3 ** 2,
    )
    print(mdp)

    # random_loop(mdp, 50, verbose=True)
    human_loop(mdp)
