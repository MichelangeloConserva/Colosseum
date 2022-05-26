import random
from itertools import product
from typing import Tuple

import numba
import numpy as np

from colosseum.agents.policy import EpisodicPolicy
from colosseum.utils.miscellanea import argmax_3D


@numba.njit()
def value_iteration(
    H: int, T: np.ndarray, R: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    num_states, num_actions, _ = T.shape

    Q = np.zeros((H + 1, num_states, num_actions), dtype=np.float32)
    V = np.zeros((H + 1, num_states), dtype=np.float32)
    for i in range(H):
        h = H - i - 1
        for s in range(num_states):
            Q[h, s] = R[s] + T[s] @ V[h + 1]
            V[h, s] = Q[h, s].max()
    return Q, V


@numba.njit()
def policy_evaluation(
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


def policy_iteration(H: int, T: np.ndarray, R: np.ndarray):
    num_states, num_actions, _ = T.shape

    Q = np.random.rand(H + 1, num_states, num_actions)
    pi = argmax_3D(Q)
    for t in range(1_000_000):
        old_pi = pi.copy()
        Q, V = policy_evaluation(H, T, R, pi)
        pi = argmax_3D(Q)
        if (pi[:-1] != old_pi[:-1]).sum() == 0:
            break
    return Q, V, pi


@numba.njit()
def optimistic_value_iteration(
    H: int, T: np.ndarray, R: np.ndarray, R_bonus: np.ndarray, P_bonus: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    num_states, num_actions, _ = T.shape

    Q = np.zeros((H + 1, num_states, num_actions), dtype=np.float32)
    V = np.zeros((H + 1, num_states), dtype=np.float32)
    for i in range(H):
        h = H - i - 1
        for s in range(num_states):
            Q[h, s] = R[s] + T[s] @ V[h + 1] + P_bonus[s] * h + R_bonus[s]
            V[h, s] = Q[h, s].max()
    return Q, V


def get_policy(Q: np.ndarray, rng: random.Random) -> EpisodicPolicy:
    H, num_states, num_actions = Q.shape
    policy = dict()
    for h in range(H):
        for s in range(num_states):
            q = Q[h, s]
            action = rng.choice(np.where(q == q.max())[0])
            policy[h, s] = action
    return EpisodicPolicy(policy, num_actions, H)


def get_random_policy(H: int, num_states: int, num_actions: int) -> EpisodicPolicy:
    return EpisodicPolicy(
        {
            (h, s): (np.ones(num_actions, np.float32) / num_actions)
            for h, s in product(range(H), range(num_states))
        },
        num_actions,
        H,
    )


def get_constant_policy(
    H: int, num_states: int, num_actions: int, c: float
) -> EpisodicPolicy:
    return EpisodicPolicy(
        {(h, s): c for h, s in product(range(H), range(num_states))},
        num_actions,
        H,
    )
