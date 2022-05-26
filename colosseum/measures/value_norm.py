import time

import numpy as np
from tqdm import trange


def expected_value(f, ni):
    return np.einsum("iaj,j->ia", ni, f)


def calculate_gain(transition_probabilities, average_rewards, steps):
    P_star = np.linalg.matrix_power(transition_probabilities, steps)
    return P_star @ average_rewards


def calculate_bias(
    transition_probabilities, average_rewards, steps=1000, verbose=False
):
    num_states = len(transition_probabilities)

    gain = calculate_gain(transition_probabilities, average_rewards, steps)

    h = np.zeros((num_states,))
    P_i = np.eye(num_states)
    start = time.time()
    for i in trange(steps, desc="gain") if verbose else range(steps):
        h += P_i @ (average_rewards - gain)
        P_i = P_i @ transition_probabilities
        if time.time() - start > 60:
            break
    return h


def calculate_norm_discounted(T, V):
    Ev = expected_value(V, T)
    return np.sqrt(np.einsum("iaj,ja->ia", T, (V.reshape(-1, 1) - Ev) ** 2)).max()


def calculate_norm_average(
    T, transition_probabilities, average_rewards, steps=1000, verbose=False
):
    h = calculate_bias(transition_probabilities, average_rewards, steps, verbose)
    Eh = expected_value(h, T)
    return np.sqrt(np.einsum("iaj,ja->ia", T, (h.reshape(-1, 1) - Eh) ** 2)).max()
