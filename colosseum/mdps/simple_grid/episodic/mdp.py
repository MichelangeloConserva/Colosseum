import gin

from colosseum.loops import human_loop
from colosseum.mdps import EpisodicMDP
from colosseum.mdps.simple_grid.simple_grid import SimpleGridMDP, SimpleGridReward


@gin.configurable
class SimpleGridEpisodic(EpisodicMDP, SimpleGridMDP):
    pass


if __name__ == "__main__":
    mdp = SimpleGridEpisodic(
        reward_type=SimpleGridReward.AND,
        seed=42,
        size=5,
        randomize_actions=False,
        make_reward_stochastic=True,
        lazy=0.01,
        number_starting_states=1,
    )

    # random_loop(mdp, 50, verbose=False)
    human_loop(mdp)
