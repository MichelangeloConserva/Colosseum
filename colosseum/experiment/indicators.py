from typing import Tuple

import numpy as np

from colosseum.dynamic_programming import episodic_value_iteration
from colosseum.dynamic_programming import episodic_policy_evaluation


def get_episodic_regret_at_time_zero(
    H: int,
    T: np.ndarray,
    R: np.ndarray,
    policy: np.ndarray,
    optimal_value: np.ndarray = None,
) -> np.ndarray:
    """
    Returns
    -------
    np.ndarray
        The regret for the states at in-episode time of zero.
    """
    assert T.ndim == 3, "We don't need the episodic transition matrix here."
    _, V = episodic_policy_evaluation(H, T, R, policy)
    if optimal_value is None:
        _, optimal_value = episodic_value_iteration(H, T, R)
    return optimal_value[0] - V[0]


def get_episodic_regrets_and_average_reward_at_time_zero(
    H, T, R, policy, starting_state_distribution, optimal_value: np.ndarray = None
) -> Tuple[np.ndarray, float]:
    """
    Returns
    -------
    np.ndarray
        The regret for the states at in-episode time of zero.
    float
        The average value at time zero
    """
    _, V = episodic_policy_evaluation(H, T, R, policy)
    episodic_agent_average_reward = sum(V[0] * starting_state_distribution)
    if optimal_value is None:
        _, optimal_value = episodic_value_iteration(H, T, R)
    regret_at_time_zero = np.maximum(optimal_value[0] - V[0], 0.0)
    return regret_at_time_zero, episodic_agent_average_reward
