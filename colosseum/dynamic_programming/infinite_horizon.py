from typing import Tuple, Union

import numba
import numpy as np
import sparse

from colosseum.dynamic_programming import DP_MAX_ITERATION
from colosseum.dynamic_programming.utils import (
    DynamicProgrammingMaxIterationExceeded,
    argmax_2d,
)


def discounted_value_iteration(
    T: Union[np.ndarray, sparse.COO],
    R: np.ndarray,
    gamma=0.99,
    epsilon=1e-3,
    max_abs_value: float = None,
    sparse_n_states_threshold: int = 300 * 3 * 300,
    sparse_nnz_per_threshold: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    num_states, num_actions, _ = T.shape

    if type(T) == sparse.COO:
        return _discounted_value_iteration_sparse(T, R, gamma, epsilon, max_abs_value)

    if T.size > sparse_n_states_threshold:
        T_sparse = sparse.COO(T)
        if T_sparse.nnz / T.size < sparse_nnz_per_threshold:
            return _discounted_value_iteration_sparse(
                T_sparse, R, gamma, epsilon, max_abs_value
            )

    try:
        res = _discounted_value_iteration(T, R, gamma, epsilon, max_abs_value)
    except:
        # Failure of discounted value iteration may randomly happen when using multiprocessing.
        # If that happens, we can simply resort to sparse value iteration.
        T_sparse = sparse.COO(T)
        res = _discounted_value_iteration_sparse(
            T_sparse, R, gamma, epsilon, max_abs_value
        )
    return res


def discounted_policy_evaluation(
    T: Union[np.ndarray, sparse.COO],
    R: np.ndarray,
    pi: np.ndarray,
    gamma=0.99,
    epsilon=1e-7,
    sparse_n_states_threshold: int = 200,
    sparse_nnz_per_threshold: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    num_states, num_actions, _ = T.shape

    if type(T) == sparse.COO:
        return _discounted_policy_evaluation_sparse(T, R, pi, gamma, epsilon)
    if num_states > sparse_n_states_threshold:
        T_sparse = sparse.COO(T)
        if T_sparse.nnz / T.size < sparse_nnz_per_threshold:
            return _discounted_policy_evaluation_sparse(T_sparse, R, pi, gamma, epsilon)
    return _discounted_policy_evaluation(T, R, pi, gamma, epsilon)


@numba.njit()
def extended_value_iteration(
    T: np.ndarray,
    estimated_rewards: np.ndarray,
    beta_r: np.ndarray,
    beta_p: np.ndarray,
    r_max: float,
    epsilon=1e-3,
) -> Union[None, Tuple[float, np.ndarray, np.ndarray]]:
    """
    if successful, it returns the span of the value function, the Q matrix and the V matrix. It returns None when it was
    not possible to complete the extended value iteration procedure.
    """
    num_states, num_actions = beta_r.shape

    Q = np.zeros((num_states, num_actions), dtype=np.float32)
    V = np.zeros((num_states,), dtype=np.float32)

    u1 = np.zeros(num_states, np.float32)
    sorted_indices = np.arange(num_states)
    u2 = np.zeros(num_states, np.float32)
    vec = np.zeros(num_states, np.float32)

    for _ in range(DP_MAX_ITERATION):
        for s in range(num_states):
            first_action = True
            for a in range(num_actions):
                vec = _max_proba(
                    T[s, a], sorted_indices, beta_p[s, a], num_states, num_actions
                )
                vec[s] -= 1
                r_optimal = min(
                    np.float32(r_max),
                    estimated_rewards[s, a] + beta_r[s, a],
                )
                v = r_optimal + np.dot(vec, u1)
                Q[s, a] = v
                if (
                    first_action
                    or v + u1[s] > u2[s]
                    or np.abs(v + u1[s] - u2[s]) < epsilon
                ):  # optimal policy = argmax
                    u2[s] = np.float32(v + u1[s])
                first_action = False
            V[s] = np.max(Q[s])
        if np.ptp(u2 - u1) < epsilon:  # stopping condition of EVI
            return np.ptp(u1), Q, V
        else:
            u1 = u2
            u2 = np.empty(num_states, np.float32)
            sorted_indices = np.argsort(u1)
    return None


@numba.njit()
def _discounted_value_iteration(
    T: np.ndarray, R: np.ndarray, gamma=0.99, epsilon=1e-3, max_abs_value: float = None
) -> Tuple[np.ndarray, np.ndarray]:

    num_states, num_actions, _ = T.shape
    gamma = np.float32(gamma)

    V = np.zeros(num_states, dtype=np.float32)
    Q = np.zeros((num_states, num_actions), dtype=np.float32)
    for _ in range(DP_MAX_ITERATION):
        V_old = V.copy()
        for s in range(num_states):
            Q[s] = R[s] + gamma * T[s] @ V
            V[s] = Q[s].max()
            if max_abs_value is not None:
                if np.abs(V[s]) > max_abs_value:
                    return None
        diff = np.abs(V_old - V).max()
        if diff < epsilon:
            return Q, V
    raise DynamicProgrammingMaxIterationExceeded()


def _discounted_value_iteration_sparse(
    T: sparse.COO, R: np.ndarray, gamma=0.99, epsilon=1e-3, max_abs_value: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    num_states, num_actions, _ = T.shape
    gamma = np.float32(gamma)

    V = np.zeros(num_states, dtype=np.float32)
    for _ in range(DP_MAX_ITERATION):
        V_old = V.copy()
        Q = R + gamma * (T @ V).squeeze()
        V = Q.max(1, keepdims=True)

        if max_abs_value is not None:
            if V.abs() > max_abs_value:
                return None

        diff = np.abs(V_old.squeeze() - V.squeeze()).max()
        if diff < epsilon:
            return Q, V.squeeze()
    raise DynamicProgrammingMaxIterationExceeded()


@numba.njit()
def _discounted_policy_evaluation(
    T: np.ndarray, R: np.ndarray, pi: np.ndarray, gamma=0.99, epsilon=1e-7
) -> Tuple[np.ndarray, np.ndarray]:
    num_states, num_actions, _ = T.shape
    gamma = np.array([gamma], dtype=np.float32)

    V = np.zeros(num_states, dtype=np.float32)
    Q = np.zeros((num_states, num_actions), dtype=np.float32)
    for _ in range(DP_MAX_ITERATION):
        V_old = V.copy()
        for s in range(num_states):
            Q[s] = R[s] + gamma * T[s] @ V
            V[s] = (Q[s] * pi[s]).sum()
        diff = np.abs(V_old.squeeze() - V.squeeze()).max()
        if diff < epsilon:
            return Q, V
    raise DynamicProgrammingMaxIterationExceeded()


def _discounted_policy_evaluation_sparse(
    T: Union[np.ndarray, sparse.COO], R: np.ndarray, pi: np.ndarray, gamma=0.99, epsilon=1e-7
) -> Tuple[np.ndarray, np.ndarray]:
    num_states, num_actions, _ = T.shape
    gamma = np.array([gamma], dtype=np.float32)

    V = np.zeros(num_states, dtype=np.float32)
    for _ in range(DP_MAX_ITERATION):
        V_old = V.copy()
        Q = R + gamma * T @ V
        V = (Q * pi).sum(1)
        diff = np.abs(V_old - V).max()
        if diff < epsilon:
            return Q, V
    raise DynamicProgrammingMaxIterationExceeded()


def discounted_policy_iteration(T: np.ndarray, R: np.ndarray, gamma=0.99, epsilon=1e-7):
    num_states, num_actions, _ = T.shape

    Q = np.random.rand(num_states, num_actions)
    pi = argmax_2d(Q)
    for t in range(DP_MAX_ITERATION):
        old_pi = pi.copy()
        Q, V = discounted_policy_evaluation(T, R, pi, gamma, epsilon)
        pi = argmax_2d(Q)
        if (pi != old_pi).sum() == 0:
            return Q, V, pi
    raise DynamicProgrammingMaxIterationExceeded()


@numba.njit()
def _max_proba(
    p: np.ndarray,
    sorted_indices: np.ndarray,
    beta: np.ndarray,
    num_states: int,
    num_actions: int,
) -> np.ndarray:
    min1 = min(1.0, (p[sorted_indices[num_states - 1]] + beta / 2)[0])
    if min1 == 1:
        p2 = np.zeros(num_states, np.float32)
        p2[sorted_indices[num_states - 1]] = 1
    else:
        sorted_p = p[sorted_indices]
        support_sorted_p = np.nonzero(sorted_p)[0]
        restricted_sorted_p = sorted_p[support_sorted_p]
        support_p = sorted_indices[support_sorted_p]
        p2 = np.zeros(num_states, np.float32)
        p2[support_p] = restricted_sorted_p
        p2[sorted_indices[num_states - 1]] = min1
        s = 1 - p[sorted_indices[num_states - 1]] + min1
        s2 = s
        for i, proba in enumerate(restricted_sorted_p):
            max1 = max(0, 1 - s + proba)
            s2 += max1 - proba
            p2[support_p[i]] = max1
            s = s2
            if s <= 1:
                break
    return p2
