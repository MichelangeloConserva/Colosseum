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
        n_states: int,
        n_actions: int,
        hyper_params: Union[List[float], List[List[float]]],
        seed: int,
        interpretable_parameters: bool = True,
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
        interpretable_parameters : bool
            If True, the parameters are given in the natural way of speaking of NIG parameters.
        """

        super(N_NIG, self).__init__(n_states, n_actions, hyper_params, seed)

        assert self.hyper_params.shape == (n_states, n_actions, 4)

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

    def update_sa(self, s: int, a: int, rs: List[float]):
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
            self.n_states * self.n_actions, -1
        ).T

        # Sample scaling tau from a gamma distribution
        tau = self._rng.gamma(shape=alpha, scale=1.0 / beta).astype(np.float32)
        var = 1.0 / (lambda0 * tau)

        # Sample mean from normal mean mu, var
        mean = self._rng.normal(loc=mu, scale=np.sqrt(var), size=(n, *mu.shape)).astype(
            np.float32
        )

        return mean.reshape(self.n_states, self.n_actions).squeeze()

    def get_map_estimate(self) -> np.ndarray:
        return self.hyper_params[:, :, 0]


class N_N(ConjugateModel):
    """
    The Normal-Normal conjugate model.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        hyper_params: Union[List[float], List[List[float]]],
        seed: int,
    ):
        super(N_N, self).__init__(n_states, n_actions, hyper_params, seed)

        assert self.hyper_params.shape == (n_states, n_actions, 2)

    def update_sa(self, s: int, a: int, xs: List[float]):
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
