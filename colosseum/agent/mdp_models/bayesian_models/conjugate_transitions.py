from typing import List, Tuple, Union

import numpy as np

from colosseum.agent.mdp_models.bayesian_models import ConjugateModel
from colosseum.utils.miscellanea import state_occurencens_to_counts

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
        xs = [state_occurencens_to_counts(x, self.num_states) for x in xs]
        self.hyper_params[s, a] += np.array(xs).sum(0)

    def _sample(self, hyper_params: np.ndarray, n: int) -> np.ndarray:
        r = (
            self._rng.standard_gamma(hyper_params, (n, *hyper_params.shape))
            .astype(np.float32)
            .squeeze()
        )
        return r / (1e-5 + r.sum(-1, keepdims=True))

    def sample(self, n: int = 1) -> np.ndarray:
        r = self._sample(
            self.hyper_params.reshape(self.num_states * self.num_actions, -1), n
        )
        return r.reshape((self.num_states, self.num_actions, -1))

    def sample_sa(self, sa: Tuple[int, int]) -> np.ndarray:
        return self._sample(self.hyper_params[sa], 1)

    def get_map_estimate(self) -> np.ndarray:
        return self.hyper_params / self.hyper_params.sum(-1, keepdims=True)
