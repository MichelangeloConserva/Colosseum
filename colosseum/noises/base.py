import abc
from typing import List, Callable, Tuple

import numpy as np

from colosseum import config


class Noise(abc.ABC):
    """
    The base class for the noise to be applied to the emission maps.
    """

    @abc.abstractmethod
    def _sample_noise(self, n: int) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            The n samples of the noise_class.
        """

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        Tuple[int, ...]
            The shape of the emission map to which the noise is applied.
        """
        if self._shape is None:
            self._shape = self._shape_f()
        return self._shape

    def __init__(self, seed: int, shape_f: Callable[[], List[int]]):
        """
        Parameters
        ----------
        seed : int
            The random seed.
        shape_f : Callable[[], List[int]]
            The function that returns the shape of the emission map.
        """

        self._rng = np.random.RandomState(seed)
        self._shape_f = shape_f

        self._shape = None
        self._cached_samples = []

    def __next__(self):
        if len(self._cached_samples) == 0:
            self._cached_samples = list(
                self._sample_noise(config.get_size_cache_noise()).astype(np.float32)
            )
        return self._cached_samples.pop(0)
