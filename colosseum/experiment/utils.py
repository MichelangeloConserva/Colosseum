import numpy as np

from colosseum.dynamic_programming import episodic_policy_evaluation, episodic_value_iteration


def get_episodic_regret_for_starting_states(
    H: int,
    T: np.ndarray,
    R: np.ndarray,
    policy: np.ndarray,
    optimal_value: np.ndarray = None,
):
    assert T.ndim == 3, "We don't need the episodic transition matrix here."
    _, V = episodic_policy_evaluation(H, T, R, policy)
    if optimal_value is None:
        _, optimal_value = episodic_value_iteration(H, T, R)
    return optimal_value[0] - V[0]


def get_episodic_regrets_and_average_reward(
    H, T, R, policy, starting_distribution, optimal_value: np.ndarray = None
):
    _, V = episodic_policy_evaluation(H, T, R, policy)
    episodic_agent_average_reward = sum(V[0] * starting_distribution)
    if optimal_value is None:
        _, optimal_value = episodic_value_iteration(H, T, R)
    regrets_for_starting_states = np.maximum(optimal_value[0] - V[0], 0.0)
    return regrets_for_starting_states, episodic_agent_average_reward
