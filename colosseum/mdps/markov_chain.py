from typing import List

import numpy as np
from pydtmc import MarkovChain

from colosseum.agents.policy import ContinuousPolicy


def transition_probabilities(T: np.ndarray, policy: ContinuousPolicy) -> np.ndarray:
    return np.einsum("saj,sa->sj", T, policy.pi_matrix)


def get_markov_chain(
    transition_probabilities: np.ndarray, labels: List = None
) -> MarkovChain:
    return MarkovChain(transition_probabilities, labels)
