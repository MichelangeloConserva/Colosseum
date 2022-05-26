from enum import IntEnum
from typing import List, Tuple, Union

import gin
import numpy as np

from colosseum.agents.bayes_tools.conjugate_base import ConjugateModel


@gin.constants_from_enum
class TransitionsConjugateModel(IntEnum):
    M_DIR = 0

    def get_class(self):
        if self == self.M_DIR:
            return M_DIR


PRIOR_TYPE = Union[
    List[
        float,
    ],
    List[
        List[
            float,
        ],
    ],
]


class M_DIR(ConjugateModel):
    """
    Multinomial-Dirichlet conjugate model.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        hyper_params: Union[
            List[float],
            List[
                List[
                    float,
                ]
            ],
        ],
        seed: int,
    ):
        """

        Parameters
        ----------
        num_states : int
            the number of states of the MDP.
        num_actions : int
            the number of action of the
        hyper_params : Union[List[float],List[List[float]]]
            the prior hyperparameters can either be a list of hyperparameters that are set identical for each
            state-action pair, or it can be a dictionary with the state action pair as key and a list of hyperparameters
            as value.
        seed : int
            the seed for sampling.
        """
        super(M_DIR, self).__init__(num_states, num_actions, hyper_params, seed)
        if self.hyper_params.shape == (num_states, num_actions, 1):
            self.hyper_params = np.tile(self.hyper_params, (1, 1, num_states))
        assert self.hyper_params.shape == (num_states, num_actions, num_states)

    def _update_sa(self, s: int, a: int, xs: List[int]):
        """
        updates the beliefs of the given state action pair.
        Parameters
        ----------
        s : int
            the state to update.
        a : int
            the action to update.
        xs : List
            the occurrences of states obtained from state action pair (s,a).
        """
        self.hyper_params[s, a] += np.array(xs)

    def sample(self, hyper_params: np.ndarray) -> np.ndarray:
        r = self._rng.standard_gamma(hyper_params).astype(np.float32)
        return r / (1e-5 + r.sum(-1, keepdims=True))

    def sample_MDP(self) -> np.ndarray:
        r = self.sample(
            self.hyper_params.reshape(self.num_states * self.num_actions, -1)
        )
        return r.reshape(self.num_states, self.num_actions, -1)

    def sample_sa(self, sa: Tuple[int, int]) -> np.ndarray:
        return self.sample(self.hyper_params[sa])
