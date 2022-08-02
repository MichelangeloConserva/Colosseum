from typing import Tuple

import numba
import numpy as np

from colosseum.dynamic_programming import DP_MAX_ITERATION
from colosseum.dynamic_programming.utils import (
    DynamicProgrammingMaxIterationExceeded,
    argmax_3d,
)


@numba.njit()
def episodic_value_iteration(
    H: int, T: np.ndarray, R: np.ndarray, max_value: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    num_states, num_actions, _ = T.shape

    Q = np.zeros((H + 1, num_states, num_actions), dtype=np.float32)
    V = np.zeros((H + 1, num_states), dtype=np.float32)
    for i in range(H):
        h = H - i - 1
        for s in range(num_states):
            Q[h, s] = R[s] + T[s] @ V[h + 1]
            V[h, s] = Q[h, s].max()
            if max_value is not None and V[h, s] > max_value:
                return None
    return Q, V


@numba.njit()
def episodic_policy_evaluation(
    H: int, T: np.ndarray, R: np.ndarray, policy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    num_states, num_actions, _ = T.shape

    Q = np.zeros((H + 1, num_states, num_actions), dtype=np.float32)
    V = np.zeros((H + 1, num_states), dtype=np.float32)
    for i in range(H):
        h = H - i - 1
        for s in range(num_states):
            Q[h, s] = R[s] + T[s] @ V[h + 1]
            V[h, s] = (Q[h, s] * policy[h, s]).sum()
    return Q, V


def episodic_policy_iteration(T: np.ndarray, R: np.ndarray, gamma=0.99, epsilon=1e-7):
    H, num_states, num_actions, _ = T.shape

    Q = np.random.rand(H, num_states, num_actions)
    pi = argmax_3d(Q)
    for t in range(DP_MAX_ITERATION):
        old_pi = pi.copy()
        Q, V = episodic_policy_evaluation(T, R, pi, gamma, epsilon)
        pi = argmax_3d(Q)
        if (pi != old_pi).sum() == 0:
            return Q, V, pi
    raise DynamicProgrammingMaxIterationExceeded()
