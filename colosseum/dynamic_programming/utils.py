import numba
import numpy as np

ARGMAX_SEED = 42
_rng = np.random.RandomState(ARGMAX_SEED)


class DynamicProgrammingMaxIterationExceeded(Exception):
    pass


@numba.njit()
def argmax_2d(A: np.ndarray) -> np.ndarray:
    """
    returns an array of same dimensionality of A with ones corresponding to the max across rows and zeros otherwise.
    """
    np.random.seed(ARGMAX_SEED)
    X = np.zeros_like(A, np.float32)
    for s in range(len(A)):
        i = np.random.choice(np.where(A[s] == A[s].max())[0])
        X[s, i] = 1
    return X


@numba.njit()
def argmax_3d(A: np.ndarray) -> np.ndarray:
    """
    implements a vectorized version of argmax_2d.
    """
    np.random.seed(ARGMAX_SEED)
    X = np.zeros(A.shape, np.float32)
    for h in range(len(A)):
        for s in range(A.shape[1]):
            i = np.random.choice(np.where(A[h, s] == A[h, s].max())[0])
            X[h, s, i] = 1.0
    return X


@numba.njit()
def get_deterministic_policy_from_q_values(Q: np.ndarray) -> np.ndarray:
    """
    returns the infinite horizon optimal deterministic policy for the given q values.
    """
    np.random.seed(ARGMAX_SEED)
    X = np.zeros(Q.shape[:-1], np.int32)
    for s in range(len(Q)):
        i = np.random.choice(np.where(Q[s] == Q[s].max())[0])
        X[s] = np.int32(i)
    return X


@numba.njit()
def get_deterministic_policy_from_q_values_finite_horizon(Q: np.ndarray) -> np.ndarray:
    """
    returns the finite horizon optimal deterministic policy for the given q values.
    """
    np.random.seed(ARGMAX_SEED)
    X = np.zeros(Q.shape[:-1], np.int32)
    for h in range(len(Q)):
        for s in range(Q.shape[1]):
            i = np.random.choice(np.where(Q[h, s] == Q[h, s].max())[0])
            X[h, s] = np.int32(i)
    return X


def get_policy_from_q_values(Q: np.ndarray, stochastic_form=False) -> np.ndarray:
    """
    returns the deterministic policy derived from the q_values given in input. When stochastic_form is False, it returns
    an array containing the integers corresponding to the optimal actions, whereas when stochastic_form is True, for
    each it returns an array containing a vector representing the deterministic probability distribution for the optimal
    action, i.e. [0., 0., 0., 1.] instead of 3.
    """
    # Episodic case
    if Q.ndim == 3:
        if stochastic_form:
            return argmax_3d(Q)
        return get_deterministic_policy_from_q_values_finite_horizon(Q)

    # Infinite horizon case
    if stochastic_form:
        return argmax_2d(Q)
    return get_deterministic_policy_from_q_values(Q)
