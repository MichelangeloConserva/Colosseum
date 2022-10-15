from typing import Callable, List

import numpy as np
from scipy.stats import wishart, multivariate_normal

from colosseum.noises.base import Noise


class GaussianCorrelated(Noise):
    """
    The class that creates Gaussian correlated noise.
    """

    def _sample_noise(self, n: int) -> np.ndarray:
        if self.rv is None:
            W = wishart(scale=[self._scale] * np.prod(self.shape)).rvs(1, self._rng)
            self.rv = multivariate_normal(cov=W)
        return self.rv.rvs(n, self._rng).reshape(n, *self.shape)

    def __init__(self, seed: int, shape_f: Callable[[], List[int]], scale: float = 0.1):
        """
        Parameters
        ----------
        seed : int
            The random seed.
        shape_f : Callable[[], List[int]]
            The function that returns the shape of the emission map.
        scale : float
            The scale parameter for the Wishart distribution for the covariance matrix. By default, it is 0.1.
        """

        super(GaussianCorrelated, self).__init__(seed, shape_f)

        self._scale = scale
        self.rv = None
