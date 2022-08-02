from typing import Dict, List, Tuple, Union

import numpy as np

from colosseum.agent.mdp_models.bayesian_models import ConjugateModel

PRIOR_TYPE = Union[
    List[
        float,
    ],
    Dict[
        Tuple[int, int],
        List[
            float,
        ],
    ],
]


class N_NIG(ConjugateModel):
    """
    The Normal-Normal Inverse Gamma conjugate model.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        hyper_params: Union[List[float], List[List[float]]],
        seed: int,
        interpretable_parameters: bool = True,
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
        interpretable_parameters : bool
            checks if the parameters are given in the natural way of speaking of NIG hyperparameters.
        """
        super(N_NIG, self).__init__(num_states, num_actions, hyper_params, seed)

        assert self.hyper_params.shape == (num_states, num_actions, 4)

        if interpretable_parameters:
            for i in range(self.hyper_params.shape[0]):
                for j in range(self.hyper_params.shape[1]):
                    mu, n_mu, tau, n_tau = self.hyper_params[i, j]
                    self.hyper_params[i, j] = (
                        mu,
                        n_mu,
                        n_tau * 0.5,
                        (0.5 * n_tau) / tau,
                    )

    def _update_sa(self, s: int, a: int, rs: List[float]):
        """
        updates the beliefs of the given state action pair.
        Parameters
        ----------
        s : int
            the state to update.
        a : int
            the action to update.
        xs : List
            the rewards obtained from state action pair (s,a).
        """
        # Unpack the prior
        (mu0, lambda0, alpha0, beta0) = self.hyper_params[s, a]

        n = len(rs)
        y_bar = np.mean(rs)

        # Updating normal component
        lambda1 = lambda0 + n
        mu1 = (lambda0 * mu0 + n * y_bar) / lambda1

        # Updating Inverse-Gamma component
        alpha1 = alpha0 + (n * 0.5)
        ssq = n * np.var(rs)
        prior_disc = lambda0 * n * ((y_bar - mu0) ** 2) / lambda1
        beta1 = beta0 + 0.5 * (ssq + prior_disc)

        self.hyper_params[s, a] = (mu1, lambda1, alpha1, beta1)

    def sample(self, n: int = 1) -> np.ndarray:
        # Unpack the prior
        (mu, lambda0, alpha, beta) = self.hyper_params.reshape(
            self.num_states * self.num_actions, -1
        ).T

        # Sample scaling tau from a gamma distribution
        tau = self._rng.gamma(shape=alpha, scale=1.0 / beta).astype(np.float32)
        var = 1.0 / (lambda0 * tau)

        # Sample mean from normal mean mu, var
        mean = self._rng.normal(loc=mu, scale=np.sqrt(var), size=(n, *mu.shape)).astype(
            np.float32
        )

        return mean.reshape(self.num_states, self.num_actions).squeeze()

    def get_map_estimate(self) -> np.ndarray:
        return self.hyper_params[:, :, 0]


class N_N(ConjugateModel):
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        hyper_params: Union[List[float], List[List[float]]],
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
        super(N_N, self).__init__(num_states, num_actions, hyper_params, seed)

        assert self.hyper_params.shape == (num_states, num_actions, 2)

    def _update_sa(self, s: int, a: int, xs: List[float]):
        """
        updates the beliefs of the given state action pair.
        Parameters
        ----------
        s : int
            the state to update.
        a : int
            the action to update.
        xs : List
            the rewards obtained from state action pair (s,a).
        """
        for r in xs:
            mu0, tau0 = self.hyper_params[s, a]
            tau1 = tau0 + 1
            mu1 = (mu0 * tau0 + r * 1) / tau1
            self.hyper_params[s, a] = (mu1, tau1)

    def sample(self, n: int = 1) -> np.ndarray:
        return (
            self._rng.normal(
                loc=self.hyper_params[:, :, 0], scale=self.hyper_params[:, :, 1], size=n
            )
            .astype(np.float32)
            .squeeze()
        )

    def get_map_estimate(self) -> np.ndarray:
        return self.hyper_params[:, :, 0]
