from typing import Callable, List

import numpy as np

from colosseum.noises.base import Noise


class StudentTUncorrelated(Noise):
    """
    The class that creates Student's t uncorrelated noise.
    """

    def _sample_noise(self, n: int) -> np.ndarray:
        return self._rng.standard_t(self._df, *self.shape)

    def __init__(self, seed: int, shape_f: Callable[[], List[int]], df: float = 3):
        """
        Parameters
        ----------
        seed : int
            The random seed.
        shape_f : Callable[[], List[int]]
            The function that returns the shape of the emission map.
        df : int
            The degree of freedom of the Student's t distribution.
        """
        super(StudentTUncorrelated, self).__init__(seed, shape_f)

        self._df = df
