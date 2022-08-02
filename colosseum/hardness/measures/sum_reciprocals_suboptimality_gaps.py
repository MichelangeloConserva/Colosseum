from typing import List, Tuple

import numpy as np


def get_sum_reciprocals_suboptimality_gaps(
    Q: np.ndarray,
    V: np.ndarray,
    reachable_states: List[Tuple[int, int]] = None,
    regularization: float = 0.1,
):
    """
    returns the sum of the reciprocals of the sub-optimality gaps. The reachable_states parameter is necessary in the
    episodic setting and it should be a list of tuple with in episode time step and state for each feasible combination
    of in episode time step and state.
    """
    is_episodic = V.ndim == 2
    gaps = V[..., None] - Q
    if is_episodic:
        assert reachable_states is not None, (
            "For the episodic setting, it is necessary to provide the set of nodes that are reachable for any given"
            "in episode time step."
        )
        gaps = np.vstack([gaps[h, s] for h, s in reachable_states])
    return (1 / (gaps + regularization)).sum()
