import gin

from colosseum.loops import human_loop
from colosseum.mdps import ContinuousMDP
from colosseum.mdps.taxi.taxi import TaxiMDP


@gin.configurable
class TaxiContinuous(ContinuousMDP, TaxiMDP):
    pass


if __name__ == "__main__":

    mdp = TaxiContinuous(
        seed=42,
        randomize_actions=False,
        make_reward_stochastic=False,
        lazy=None,
        size=16,
        length=1,
        width=1,
        space=1,
        n_locations=2 ** 2,
    )

    human_loop(mdp)
