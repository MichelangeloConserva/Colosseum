from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np


class ConjugateModel(ABC):
    """
    Base class for Bayesian conjugate models.
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

        self.num_actions = num_actions
        self.num_states = num_states
        self._rng = np.random.RandomState(seed)

        if type(hyper_params[0]) in [int, float] or "numpy.flo" in str(
            type(hyper_params[0])
        ):
            # same priors for each state action pair
            self.hyper_params = np.tile(
                hyper_params, (num_states, num_actions, 1)
            ).astype(np.float32)
        elif type(hyper_params[0]) in [list, tuple, np.ndarray]:
            # each state action pair has a different prior
            self.hyper_params = np.array(hyper_params, np.float32)
        else:
            raise ValueError(
                f"Received incorrect hyperparameters  with type "
                f"{type(hyper_params), type(hyper_params[0])}"
            )

    @abstractmethod
    def _update_sa(self, s: int, a: int, xs: List):
        """
        updates the beliefs of the given state action pair.
        Parameters
        ----------
        s : int
            the state to update.
        a : int
            the action to update.
        xs : List
            the samples obtained from state action pair (s,a).
        """

    @abstractmethod
    def sample_MDP(self) -> np.ndarray:
        """
        samples form the model belief.
        Returns
        -------
            a np.ndarray.
        """

    def update(self, data: Dict[Tuple[int, int], List[float]]):
        """

        Parameters
        ----------
        data : Dict[Tuple[int, int], List[float]]
            the data to be used to update the model using Bayes rule.
        """
        for (s, a), xs in data.items():
            self._update_sa(s, a, xs)
