import gin
import numpy as np

from colosseum.loops import human_loop
from colosseum.mdps import ContinuousMDP
from colosseum.mdps.custom.custom import CustomMDP


@gin.configurable
class CustomContinuous(ContinuousMDP, CustomMDP):
    pass


if __name__ == "__main__":
    num_states = 4
    num_actions = 2

    T = np.zeros((num_states, num_actions, num_states), dtype=np.float32)
    T[0, 0, 1] = 1.0
    T[0, 1, 2] = 1.0

    T[1, 0, 2] = T[1, 0, 3] = 0.5
    T[1, 1, 2] = T[1, 1, 3] = 0.1
    T[1, 1, 1] = 0.8

    T[2, 0, 1] = T[2, 0, 3] = 0.5
    T[2, 1, 1] = T[2, 1, 3] = 0.1
    T[2, 1, 2] = 0.8

    T[3, 0, 0] = 0.5
    T[3, 0, 1] = T[3, 0, 2] = 0.25
    T[3, 1, 0] = 0.1
    T[3, 1, 1] = T[3, 1, 2] = 0.1
    T[3, 1, 3] = 0.7
    np.random.seed(42)
    R = np.random.rand(num_states, num_actions)
    T_0 = {0: 1.0}

    mdp = CustomContinuous(
        seed=42,
        randomize_actions=True,
        lazy=None,
        random_action_p=None,
        T_0=T_0,
        T=T,
        R=R,
    )

    human_loop(mdp)
