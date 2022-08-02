import os
from typing import Iterable, List, Optional, Tuple

import networkx as nx
import numba
import numpy as np
import scipy
from pydtmc import MarkovChain
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix


def get_average_reward(
    T: np.ndarray,
    R: np.ndarray,
    policy: np.ndarray,
    next_states_and_probs: Optional,
    sparse_threshold_size: int = 500 * 500,
) -> float:
    """
    returns the expected average reward when following policy for the MDP defined by the given transition matrix and
    rewards matrix.
    """
    average_rewards = get_average_rewards(R, policy)
    tps = get_transition_probabilities(T, policy)
    sd = get_stationary_distribution(tps, next_states_and_probs, sparse_threshold_size)
    return (average_rewards * sd).sum()


def get_average_rewards(R: np.ndarray, policy: np.ndarray) -> np.ndarray:
    """
    returns the expected rewards for each state when following the given policy.
    """
    return np.einsum("sa,sa->s", R, policy)


def get_transition_probabilities(T: np.ndarray, policy: np.ndarray) -> np.ndarray:
    """
    returns the transition probability matrix of the Markov chain yielded by the given policy.
    """
    return np.minimum(1.0, np.einsum("saj,sa->sj", T, policy))


def get_markov_chain(transition_probabilities: np.ndarray) -> MarkovChain:
    """
    returns a Markov chain object from the pydtmc package.
    """
    return MarkovChain(transition_probabilities)


def get_stationary_distribution(
    tps: np.ndarray,
    starting_states_and_probs: Iterable[Tuple[int, float]],
    sparse_threshold_size: int = 500 * 500,
) -> np.ndarray:
    """
    returns the stationary distribution of the transition matrix. If there are multiple recurrent classes, and so
    multiple stationary distribution, the return stationary distribution is the average of the stationary distributions
    weighted using the starting state distribution.

    Parameters
    ----------
    tps : np.ndarray
        is the transition probabilities matrix.
    starting_states_and_probs : List[Tuple[int, float]]
        is an iterable over the starting states and their corresponding probabilities.
    sparse_threshold_size : int
        is a threshold for the size of the transition probabilities matrix that flags whether it is better to use sparse
        matrices.
    """
    if tps.size > sparse_threshold_size:
        G = nx.DiGraph(coo_matrix(tps))
    else:
        G = nx.DiGraph(tps)

    # Obtain the recurrent classes
    recurrent_classes = list(map(tuple, nx.attracting_components(G)))
    if len(recurrent_classes) == 1 and len(recurrent_classes[0]) < len(tps):
        sd = np.zeros(len(tps), np.float32)
        if len(recurrent_classes[0]) == 1:
            sd[recurrent_classes[0][0]] = 1
        else:
            sd[list(recurrent_classes[0])] = _get_stationary_distribution(
                tps[np.ix_(recurrent_classes[0], recurrent_classes[0])],
                sparse_threshold_size,
            )
        return sd

    elif len(recurrent_classes) > 1 and len(recurrent_classes[0]) < len(tps):

        # Calculate the stationary distribution for each recurrent class
        recurrent_classes_sds = dict()
        for recurrent_class in recurrent_classes:
            recurrent_classes_sds[recurrent_class] = _get_stationary_distribution(
                tps[np.ix_(recurrent_class, recurrent_class)], sparse_threshold_size
            )

        sd = np.zeros(len(tps))
        if len(recurrent_classes) > 1:
            # Weight the stationary distribution of the recurrent classes with starting states distribution
            for ss, p in starting_states_and_probs:
                for recurrent_class in recurrent_classes:
                    try:
                        # this means that the starting state ss is connected to recurrent_class
                        nx.shortest_path_length(G, ss, recurrent_class[0])
                        sd[list(recurrent_class)] += (
                            p * recurrent_classes_sds[recurrent_class]
                        )
                        break
                    except nx.exception.NetworkXNoPath:
                        pass
        else:
            # No need to weight with the starting state distribution since there is only one recurrent class
            sd[list(recurrent_class)] += recurrent_classes_sds[recurrent_class]

        return sd

    sd = _get_stationary_distribution(tps)
    return sd


@numba.njit()
def _gth_solve_numba(tps: np.ndarray) -> np.ndarray:
    """
    returns the stationary distribution of a transition probabilities matrix with a single recurrent class using the
    GTH method.
    """
    a = np.copy(tps).astype(np.float64)
    n = a.shape[0]

    for i in range(n - 1):
        scale = np.sum(a[i, i + 1 : n])

        if scale <= 0.0:  # pragma: no cover
            n = i + 1
            break

        a[i + 1 : n, i] /= scale
        a[i + 1 : n, i + 1 : n] += np.outer(
            a[i + 1 : n, i : i + 1], a[i : i + 1, i + 1 : n]
        )

    x = np.zeros(n, np.float64)
    x[n - 1] = 1.0
    x[n - 2] = a[n - 1, n - 2]
    for i in range(n - 3, -1, -1):
        x[i] = np.sum(x[i + 1 : n] * a[i + 1 : n, i])
    x /= np.sum(x)
    return x


def _convertToRateMatrix(tps: csr_matrix):
    """
    converts the initial matrix to a rate matrix. We make all rows in Q sum to zero by subtracting the row sums from the
    diagonal.
    """
    rowSums = tps.sum(axis=1).getA1()
    idxRange = np.arange(tps.shape[0])
    Qdiag = coo_matrix((rowSums, (idxRange, idxRange)), shape=tps.shape).tocsr()
    return tps - Qdiag


def _eigen_method(tps, tol=1e-8, maxiter=1e5):
    """
    returns the stationary distribution of a transition probabilities matrix with a single recurrent class using the
    eigenvalue method.
    """
    Q = _convertToRateMatrix(tps)
    size = Q.shape[0]
    guess = np.ones(size, dtype=float)
    w, v = scipy.sparse.linalg.eigs(
        Q.T, k=1, v0=guess, sigma=1e-6, which="LM", tol=tol, maxiter=maxiter
    )
    pi = v[:, 0].real
    pi /= pi.sum()
    return np.maximum(pi, 0.0)


def _get_stationary_distribution(
    tps: np.ndarray, sparse_threshold_size: int = 500 * 500
) -> np.ndarray:
    if tps.size > sparse_threshold_size:
        sd = _eigen_method(csr_matrix(tps))
        if np.isnan(sd).any() or not np.isclose(sd.sum(), 1.0):
            # sometimes the eigen method fails so we use gth that is slower but more reliable
            os.makedirs("tmp/sd_failures", exist_ok=True)
            for i in range(1000):
                if not os.path.isfile(f"tmp/sd_failures/tps{i}.npy"):
                    np.save(f"tmp/sd_failures/tps{i}.npy", tps)
                    break

            sd = _gth_solve_numba(tps)
            assert not (np.isnan(sd).any() or not np.isclose(sd.sum(), 1.0)), np.save(
                "tmp/tps.npy", tps
            )
    else:
        sd = _gth_solve_numba(tps)
        assert not (np.isnan(sd).any() or not np.isclose(sd.sum(), 1.0)), np.save(
            "tmp/tps.npy", tps
        )
    return sd
