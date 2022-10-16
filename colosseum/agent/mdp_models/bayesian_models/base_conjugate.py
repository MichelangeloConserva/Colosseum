from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np


class ConjugateModel(ABC):
    """
    Base class for Bayesian conjugate models.
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
        """
        Parameters
        ----------
        n_states : int
            The number of states of the MDP.
        n_actions : int
            The number of action of the MDP.
        hyper_params : Union[List[float],List[List[float]]]
            The prior parameters can either be a list of parameters that are set identical for each
            state-action pair, or it can be a dictionary with the state action pair as key and a list of parameters
            as value.
        seed : int
            The random seed.
        """

        self.n_actions = n_actions
        self.n_states = n_states
        self._rng = np.random.RandomState(seed)

        if type(hyper_params[0]) in [int, float] or "numpy.flo" in str(
            type(hyper_params[0])
        ):
            # same priors for each state action pair
            self.hyper_params = np.tile(hyper_params, (n_states, n_actions, 1)).astype(
                np.float32
            )
        elif type(hyper_params[0]) in [list, tuple, np.ndarray]:
            # each state action pair has a different prior
            self.hyper_params = np.array(hyper_params, np.float32)
        else:
            raise ValueError(
                f"Received incorrect parameters  with type "
                f"{type(hyper_params), type(hyper_params[0])}"
            )

    @abstractmethod
    def update_sa(self, s: int, a: int, xs: List):
        """
        updates the beliefs of the given state action pair.
        Parameters
        ----------
        s : int
            The state to update.
        a : int
            The action to update.
        xs : List
            The samples obtained from state action pair (s,a).
        """

    @abstractmethod
    def sample(self, n: int = 1) -> np.ndarray:
        """
        samples from the posterior
        Parameters
        ----------
        n : int
            The number of samples. By default, it is set to one.

        Returns
        -------
        np.ndarray
            The n samples from the posterior.
        """

    @abstractmethod
    def get_map_estimate(self) -> np.ndarray:
        """
        computes the maximum a posterior estimate.
        Returns
        -------
        np.ndarray
            The maximum a posteriori estimates
        """

    def update_single_transition(self, s: int, a: int, x: Any):
        """
        updates the posterior for a single transition.
        Parameters
        ----------
        s : int
            The state to update.
        a : int
            The action to update.
        x : Any
            A sample obtained from state action pair (s,a).
        """
        self.update_sa(s, a, [x])

    def update(self, data: Dict[Tuple[int, int], List[float]]):
        """
        updates the Bayesian model.
        Parameters
        ----------
        data : Dict[Tuple[int, int], List[float]]
            the data to be used to update the model using Bayes rule.
        """
        for (s, a), xs in data.items():
            self.update_sa(s, a, xs)
