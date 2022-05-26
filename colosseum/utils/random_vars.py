import importlib
from typing import List

import numpy as np
from scipy.stats import rv_continuous


def get_dist(dist_name, args):
    return importlib.import_module(f"scipy.stats").__getattribute__(dist_name)(*args)


class deterministic_gen(rv_continuous):
    def _cdf(self, x):
        return np.where(x < 0, 0.0, 1.0)

    def _stats(self):
        return 0.0, 0.0, 0.0, 0.0

    def _rvs(self, size=None, random_state=None):
        return np.zeros(shape=size)


deterministic = deterministic_gen(name="deterministic")


def state_occurencens_to_counts(occurences: List[int], N: int) -> np.ndarray:
    x = np.zeros(N)
    for s, c in dict(zip(*np.unique(occurences, return_counts=True))).items():
        x[s] = c
    return x
