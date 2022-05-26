import random
from typing import Tuple, Union

import numba
import numpy as np

from colosseum.agents.policy import ContinuousPolicy
from colosseum.utils.miscellanea import argmax_2D


@numba.njit()
def value_iteration(
    T: np.ndarray, R: np.ndarray, gamma=0.99, epsilon=1e-3
) -> Tuple[np.ndarray, np.ndarray]:
    num_states, num_actions, _ = T.shape
    gamma = np.float32(gamma)

    V = np.zeros(num_states, dtype=np.float32)
    Q = np.zeros((num_states, num_actions), dtype=np.float32)
    for _ in range(int(1e9)):
        V_old = V.copy()
        for s in range(num_states):
            Q[s] = R[s] + gamma * T[s] @ V
            V[s] = Q[s].max()
        diff = np.abs(V_old - V).max()
        if diff < epsilon:
            break
    return Q, V


@numba.njit()
def policy_evaluation(
    T: np.ndarray, R: np.ndarray, pi: np.ndarray, gamma=0.99, epsilon=1e-7
) -> Tuple[np.ndarray, np.ndarray]:
    num_states, num_actions, _ = T.shape
    gamma = np.array([gamma], dtype=np.float32)

    V = np.zeros(num_states, dtype=np.float32)
    Q = np.zeros((num_states, num_actions), dtype=np.float32)
    for _ in range(int(1e9)):
        V_old = V.copy()
        for s in range(num_states):
            Q[s] = R[s] + gamma * T[s] @ V
            V[s] = (Q[s] * pi[s]).sum()
        diff = np.abs(V_old - V).max()
        if diff < epsilon:
            break
    return Q, V


def policy_iteration(T: np.ndarray, R: np.ndarray, gamma=0.99, epsilon=1e-7):
    num_states, num_actions, _ = T.shape

    Q = np.random.rand(num_states, num_actions)
    pi = argmax_2D(Q)

    for t in range(1_000_000):
        old_pi = pi.copy()
        Q, V = policy_evaluation(T, R, pi, gamma, epsilon)
        pi = argmax_2D(Q)
        if (pi != old_pi).sum() == 0:
            break
    return Q, V, pi


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
    if successful, it returns the span of the value function, the Q matrix and the V matrix.
    It returns None when it was not possible to complete the extended value iteration procedure.
    """
    num_states, num_actions = beta_r.shape

    Q = np.zeros((num_states, num_actions), dtype=np.float32)
    V = np.zeros((num_states,), dtype=np.float32)

    u1 = np.zeros(num_states, np.float32)
    sorted_indices = np.arange(num_states)
    u2 = np.zeros(num_states, np.float32)
    vec = np.zeros(num_states, np.float32)

    for _ in range(250_000):
        for s in range(num_states):
            first_action = True
            for a in range(num_actions):
                vec = max_proba(
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
def max_proba(
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


def get_policy(Q: np.ndarray, rng: random.Random) -> ContinuousPolicy:
    num_states, num_actions = Q.shape
    policy = dict()
    for s in range(num_states):
        q = Q[s]
        action = rng.choice(np.where(q == q.max())[0])
        policy[s] = action
    return ContinuousPolicy(policy, num_actions)


def get_random_policy(num_states: int, num_actions: int) -> ContinuousPolicy:
    return ContinuousPolicy(
        {
            s: (np.ones(num_actions, np.float32) / num_actions)
            for s in range(num_states)
        },
        num_actions,
    )


def get_constant_policy(
    num_states: int, num_actions: int, c: float
) -> ContinuousPolicy:
    return ContinuousPolicy(
        {s: c for s in range(num_states)},
        num_actions,
    )
