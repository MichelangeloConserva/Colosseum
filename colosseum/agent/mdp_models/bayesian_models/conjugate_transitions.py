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
        n_states: int,
        n_actions: int,
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
        super(M_DIR, self).__init__(n_states, n_actions, hyper_params, seed)
        if self.hyper_params.shape == (n_states, n_actions, 1):
            self.hyper_params = np.tile(self.hyper_params, (1, 1, n_states))
        assert self.hyper_params.shape == (n_states, n_actions, n_states)

    def update_sa(self, s: int, a: int, xs: List[int]):
        xs = [state_occurencens_to_counts(x, self.n_states) for x in xs]
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
            self.hyper_params.reshape(self.n_states * self.n_actions, -1), n
        )
        return r.reshape((self.n_states, self.n_actions, -1))

    def sample_sa(self, sa: Tuple[int, int]) -> np.ndarray:
        return self._sample(self.hyper_params[sa], 1)

    def get_map_estimate(self) -> np.ndarray:
        return self.hyper_params / self.hyper_params.sum(-1, keepdims=True)
