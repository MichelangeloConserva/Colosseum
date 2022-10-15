from typing import Callable, List

import numpy as np

from colosseum.noises.base import Noise


class GaussianUncorrelated(Noise):
    """
    The class that creates Gaussian uncorrelated noise.
    """

    def _sample_noise(self, n: int) -> np.ndarray:
        return self._rng.normal(loc=0, scale=self._scale, size=(n, *self.shape))

    def __init__(self, seed: int, shape_f: Callable[[], List[int]], scale: float = 0.1):
        """
        Parameters
        ----------
        seed : int
            The random seed.
        shape_f : Callable[[], List[int]]
            The function that returns the shape of the emission map.
        scale : float
            The scale parameter for Gaussian noise. By default, it is 0.1.
        """
        super(GaussianUncorrelated, self).__init__(seed, shape_f)

        self._scale = scale
