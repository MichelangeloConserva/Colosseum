"""
The module containing some implementations of Bayesian models.
"""

from enum import IntEnum

import gin

from colosseum.agent.mdp_models.bayesian_models.base_conjugate import ConjugateModel
from colosseum.agent.mdp_models.bayesian_models.conjugate_rewards import N_N, N_NIG
from colosseum.agent.mdp_models.bayesian_models.conjugate_transitions import M_DIR


@gin.constants_from_enum
class RewardsConjugateModel(IntEnum):
    """
    An enumeration of the available rewards conjugate models.
    """

    N_NIG = 0
    N_N = 1

    def get_class(self):
        if self == self.N_NIG:
            return N_NIG
        if self == self.N_N:
            return N_N


@gin.constants_from_enum
class TransitionsConjugateModel(IntEnum):
    """
    An enumeration of the available transition conjugate models.
    """
    M_DIR = 0

    def get_class(self):
        if self == self.M_DIR:
            return M_DIR
