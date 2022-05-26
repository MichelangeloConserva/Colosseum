import gin

from colosseum.loops import human_loop
from colosseum.mdps import EpisodicMDP
from colosseum.mdps.river_swim.river_swim import RiverSwimMDP


@gin.configurable
class RiverSwimEpisodic(EpisodicMDP, RiverSwimMDP):
    @property
    def _graph_layout(self):
        return {node: tuple(node) for node in self.G}


if __name__ == "__main__":

    mdp = RiverSwimEpisodic(
        seed=42,
        randomize_actions=False,
        size=15,
        lazy=0.01,
        random_action_p=0.1,
        make_reward_stochastic=True,
    )

    # random_loop(mdp, 50, verbose=True)
    human_loop(mdp)
