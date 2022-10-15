import time

import numpy as np
from tqdm import trange

from colosseum import config
from colosseum.dynamic_programming import discounted_value_iteration
from colosseum.dynamic_programming.infinite_horizon import discounted_policy_evaluation
from colosseum.dynamic_programming.utils import get_policy_from_q_values
from colosseum.mdp.utils.markov_chain import get_average_rewards
from colosseum.mdp.utils.markov_chain import get_transition_probabilities


def get_value_norm(
    T: np.ndarray,
    R: np.ndarray,
    discount: bool,
    policy: np.ndarray = None,
) -> float:
    """
    computes the environmental value norm.

    Parameters
    ----------
    T : np.ndarray
        is the transition matrix.
    R : np.ndarray
        is the reward matrix.
    discount : bool
        checks whether to compute the environmental value norm in the discounted or undiscounted form.
    policy : np.ndarray, optional
        is the policy for which it computes the environmental value norm. By default, it uses the optimal policy.

    Returns
    -------
    float
        The environmental value norm value.
    """

    if discount:
        Q, V = (
            discounted_value_iteration(T, R)
            if policy is None
            else discounted_policy_evaluation(T, R, policy)
        )
        return calculate_norm_discounted(T, V)

    if policy is None:
        policy = get_policy_from_q_values(discounted_value_iteration(T, R)[0], True)
    tps = get_transition_probabilities(T, policy)
    ars = get_average_rewards(R, policy)
    return calculate_norm_average(T, tps, ars)


def _expected_value(f, ni):
    if np.isclose(ni, 0).mean() > 0.9:
        import sparse

        ni_sparse = sparse.COO(ni)
        return ni_sparse @ f
    return np.einsum("iaj,j->ia", ni, f)


def _calculate_gain(tps, average_rewards, steps):
    P_star = np.linalg.matrix_power(tps, steps)
    return P_star @ average_rewards


def _calculate_bias(tps, average_rewards, steps=1000):
    n_states = len(tps)

    gain = _calculate_gain(tps, average_rewards, steps)

    h = np.zeros((n_states,))
    P_i = np.eye(n_states)
    start = time.time()
    for i in trange(steps, desc="gain") if config.VERBOSE_LEVEL > 0 else range(steps):
        h += P_i @ (average_rewards - gain)
        P_i = P_i @ tps
        if time.time() - start > 60:
            break
    return h


def calculate_norm_discounted(T, V):
    Ev = _expected_value(V, T)
    return np.sqrt(np.einsum("iaj,ja->ia", T, (V.reshape(-1, 1) - Ev) ** 2)).max()


def calculate_norm_average(T, tps, average_rewards, steps=1000):
    h = _calculate_bias(tps, average_rewards, steps)
    Eh = _expected_value(h, T)
    return np.sqrt(np.einsum("iaj,ja->ia", T, (h.reshape(-1, 1) - Eh) ** 2)).max()
