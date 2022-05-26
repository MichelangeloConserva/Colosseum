import time

import gin

from colosseum.mdps import ContinuousMDP
from colosseum.mdps.frozen_lake.frozen_lake import FrozenLakeMDP


@gin.configurable
class FrozenLakeContinuous(ContinuousMDP, FrozenLakeMDP):
    @property
    def _graph_layout(self):
        return {node: tuple(node) for node in self.G}


if __name__ == "__main__":
    start = time.time()
    mdp = FrozenLakeContinuous(
        seed=42,
        randomize_actions=True,
        make_reward_stochastic=False,
        lazy=None,
        size=16,
        p_frozen=0.8,
        # random_action_p=0.
    )
    print(time.time() - start)
    print(mdp)
    print(time.time() - start)

    # random_loop(mdp, 50, verbose=True)
    # human_loop(mdp)
