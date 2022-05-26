import gin

from colosseum.mdps import ContinuousMDP
from colosseum.mdps.minigrid_doorkey.minigrid_doorkey import MiniGridDoorKeyMDP


@gin.configurable
class MiniGridDoorKeyContinuous(ContinuousMDP, MiniGridDoorKeyMDP):
    pass


if __name__ == "__main__":
    mdp = MiniGridDoorKeyContinuous(
        seed=42 + 1,
        randomize_actions=True,
        make_reward_stochastic=True,
        lazy=0.05,
        size=6,
        random_action_p=0.1,
    )
    print(mdp)

    # human_loop(mdp)
