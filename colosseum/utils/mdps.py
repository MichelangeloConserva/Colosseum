from copy import deepcopy
from typing import List, Union

import numpy as np
from scipy.stats import rv_continuous

from colosseum.agents.policy import ContinuousPolicy
from colosseum.mdps.markov_chain import get_markov_chain, transition_probabilities
from colosseum.utils.random_vars import deterministic_gen


def mdp_string(name, summary):
    string = name + "\n"
    m_l = 0
    for k, v in summary.items():
        m_l = max(m_l, len(max(v.keys(), key=len)) + 4)
    for k, v in summary.items():
        string += "\t" + k + "\n"
        for kk, vv in v.items():
            string += f"\t\t{kk}{' ' * (m_l - len(kk))}:\t{vv}\n"
    return string


def check_distributions(dists: List[Union[rv_continuous, None]], are_stochastic: bool):
    """
    Checks that the distribution given in input respects the necessary conditions.
    :param dists: a list of distributions
    :param are_stochastic: to check if the the distributions are stochastic.
    """
    # You either define all or none of the distribution
    assert dists.count(None) in [0, len(dists)]

    # Double check that the distributions in input matches the stochasticity of the reward parameter
    if dists[0] is not None:
        if are_stochastic:
            assert all(type(dist.dist) != deterministic_gen for dist in dists)
        else:
            assert all(type(dist.dist) == deterministic_gen for dist in dists)


def get_average_rewards(R: np.ndarray, policy: ContinuousPolicy):
    return np.einsum("sa,sa->s", R, policy.pi_matrix)


def get_average_reward(
    average_rewards: np.ndarray, T: np.ndarray, policy: ContinuousPolicy, cur_state=None
):
    if cur_state is None:
        MC = get_markov_chain(transition_probabilities(T, policy))
        sd = MC.pi[0]
    else:
        P = transition_probabilities(T, policy)
        sd = np.zeros((1, len(T)), np.float32)
        sd[:, cur_state] = 1
        for _ in range(1_000_000):
            old_sd = deepcopy(sd)
            sd = sd @ P
            if np.isclose(np.abs(sd - old_sd).sum(), 0, atol=1e-3, rtol=1e-3):
                break
    return (average_rewards * sd).sum()
