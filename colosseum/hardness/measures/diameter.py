from io import StringIO
from multiprocessing import Pool
from typing import TYPE_CHECKING, Iterable

import networkx as nx
import numba
import numpy as np
import sparse
from tqdm import tqdm, trange

from colosseum import config
from colosseum.dynamic_programming import discounted_value_iteration
from colosseum.dynamic_programming import episodic_value_iteration
from colosseum.utils import get_loop

if TYPE_CHECKING:
    from colosseum.mdp import NODE_TYPE


def get_diameter(T: np.ndarray, is_episodic: bool, max_value: float = None) -> float:
    """
    Returns
    -------
    float
        The diameter for the transition matrix given in input. The is_episodic is only necessary to check whether
    the dimensionality of the transition matrix is effectively correct. Note that, for the episodic setting, this
    computes the diameter for the augmented state space.
    """
    assert (is_episodic and T.ndim == 4) or (not is_episodic and T.ndim == 3)
    if is_episodic:
        if config.get_available_cores() >= 3:
            return _multi_thread_episodic_diameter_calculation(T, max_value=max_value)
        return _single_thread_episodic_diameter_calculation(T, max_value=max_value)

    if config.get_available_cores() >= 3:
        return _continuous_multi_thread_diam(T, max_value=max_value)
    if T.shape[-1] > 1000:
        return _get_sparse_diameter(T, max_value=max_value)
    return _continuous_single_thread_diam(T, max_value=max_value)


def get_in_episodic_diameter(
    H: int,
    T: np.ndarray,
    reachable_node: Iterable["NODE_TYPE"],
    max_value: float = None,
) -> float:
    """
    Returns
    -------
    float
        The diameter for the in episodic formulation. Note that in this case, the diameter will always be less than
    the time horizon.
    """
    if config.get_available_cores() >= 3:
        return _episodic_multi_thread_diam(H, T, reachable_node, max_value)
    return _episodic_single_thread_diam(H, T, reachable_node, max_value)


def get_diameter_for_determinsitic_MDPs(G: nx.DiGraph) -> float:
    """
    Returns
    -------
    float
        The diameter for the given graph that represents an MDP. Note that this can be considerably slower than
    the dynamic programming implementation we propose.
    """
    A = nx.to_numpy_array(G, nonedge=np.inf)
    n, m = A.shape
    np.fill_diagonal(A, 0)  # diagonal elements should be zero
    for i in get_loop(range(n)):
        A = np.minimum(A, A[i, :][np.newaxis, :] + A[:, i][:, np.newaxis])
    return np.max(A, where=A != 0, initial=-np.inf)


def _continuous_diam_calculation(
    es: int, T: np.ndarray, max_value: float = None
) -> float:
    """
    Returns
    -------
    float
        The highest optimal expected number of time step to reach es among all other states in the continuous setting.
    """
    T_es = T.copy()
    T_es[es] = 0
    T_es[es, :, es] = 1

    R_es = np.zeros(T.shape[:2], np.float32) - 1.0
    R_es[es] = 0
    res = discounted_value_iteration(T_es, R_es, 1, max_abs_value=max_value)
    if res is None:
        return None
    _, V = res
    return -V.min()


def _continuous_single_thread_diam(T: np.ndarray, max_value: float = None) -> float:
    diameter = 0
    for es in get_loop(range(len(T))):
        diameter = max(
            diameter, _continuous_diam_calculation(es, T, max_value=max_value)
        )
        if max_value is not None and diameter > max_value:
            return None
    return diameter


def _continuous_multi_thread_diam(T: np.ndarray, max_value: float = None) -> float:
    loop = get_loop(range(len(T)))
    diameter = 0
    with Pool(processes=config.get_available_cores()) as p:
        for d in p.starmap(
            _continuous_diam_calculation, [[es, T, max_value] for es in range(len(T))]
        ):
            if max_value is not None and d is None:
                diameter = None
                break
            diameter = max(diameter, d)
            if config.VERBOSE_LEVEL > 0:
                loop.set_description(f"Max diam: {diameter:.2f}")
                loop.update()
                loop.refresh()
    return diameter


def _episodic_diam_calculation(
    es: int,
    T: np.ndarray,
    H: int,
    reachable_states: Iterable["NODE_TYPE"],
    max_value: float = None,
) -> float:
    """
    returns the highest optimal expected number of time step to reach es among all other states in the episodic setting
    """
    T_es = T.copy()
    T_es[es] = 0
    T_es[es, :, es] = 1

    R_es = np.zeros(T.shape[:2], np.float32) - 1.0
    R_es[es] = 0
    res = episodic_value_iteration(H, T_es, R_es, max_abs_value=max_value)
    if res is None:
        return None
    _, V = res
    return max(-V[h, s] for h, s in reachable_states)


def _episodic_single_thread_diam(
    H: int,
    T: np.ndarray,
    reachable_states: Iterable["NODE_TYPE"],
    max_value: float = None,
) -> float:
    assert (
        len(T.shape) == 3
    ), "The transition matrix is incorrect. Please provide the non episodic transition matrix."
    diam = 0
    for es in get_loop(range(len(T))):
        diam = max(
            diam,
            _episodic_diam_calculation(es, T, H, reachable_states, max_value=max_value),
        )
        if max_value is not None and diam > max_value:
            return None
    return diam


def _episodic_multi_thread_diam(
    H: int,
    T: np.ndarray,
    reachable_states: Iterable["NODE_TYPE"],
    max_value: float = None,
) -> float:
    loop = get_loop(range(len(T)))
    diam = 0
    with Pool(processes=config.get_available_cores()) as p:
        for d in p.starmap(
            _episodic_diam_calculation,
            [[es, T, H, reachable_states, max_value] for es in range(len(T))],
        ):
            diam = max(diam, d)
            if max_value is not None and diam > max_value:
                return None
            if config.VERBOSE_LEVEL > 0:
                loop.set_description(f"Max diam: {diam:.2f}")
                loop.update()
                loop.refresh()
    return diam


def _single_thread_episodic_diameter_calculation(
    T: np.ndarray,
    states: Iterable["NODE_TYPE"] = None,
    epsilon=0.001,
    max_value: float = None,
) -> float:
    if states is None:
        n_states = T.shape[-1]
        states = range(n_states)

    diameter = -np.inf
    states = list(reversed(states))

    if config.VERBOSE_LEVEL != 0:
        if type(config.VERBOSE_LEVEL) == int:
            loop = tqdm(states, desc="Diameter calculation", mininterval=5)
        else:
            s = StringIO()
            loop = tqdm(states, desc="Diameter calculation", file=s, mininterval=5)
    else:
        loop = states

    dima_f = (
        _continuous_diameter_calculation
        if T.ndim == 3
        else _episodic_diameter_calculation
    )
    for es in loop:
        diameter = dima_f(es, T, diameter, epsilon, max_value)
        if max_value is not None and diameter > max_value:
            return None
        if config.VERBOSE_LEVEL != 0 and type(config.VERBOSE_LEVEL) == str:
            with open(config.VERBOSE_LEVEL, "a") as f:
                f.write(
                    s.getvalue()
                    .split("\x1b")[0][2:]
                    .replace("\x00", "")
                    .replace("\n\r", "")
                    + "\n"
                )
            s.truncate(0)
    return diameter


def _d_imap(args):
    T, state_set, epsilon, sparse_T, max_value = args
    if sparse_T:
        n_states, num_sa = T.shape[-2:]
        diam = 0
        for es in state_set:
            diam = _sparse_diameter_calculation(
                es, T, int(num_sa / n_states), n_states, diam, epsilon, max_value
            )
        return diam
    return _single_thread_episodic_diameter_calculation(
        T, state_set, epsilon, max_value
    )


def _multi_thread_episodic_diameter_calculation(
    T: np.ndarray,
    epsilon: float = 0.001,
    force_sparse: bool = False,
    max_value: float = None,
) -> float:
    n_actions, n_states = T.shape[-2:]
    diameter = -np.inf

    sparse_T = False
    if force_sparse or (n_states > 500 and T.ndim == 3):
        T = np.moveaxis(T, -1, 0).reshape(n_states, n_states * n_actions)
        T = sparse.COO(T)
        sparse_T = True

    states = np.arange(n_states)[::-1]
    # np.random.shuffle(states)
    states_sets = np.array_split(states, config.get_available_cores())
    inputs = [(T, state_set, epsilon, sparse_T, max_value) for state_set in states_sets]
    if config.VERBOSE_LEVEL > 0:
        loop = trange(len(inputs), desc="Diameter calculation", mininterval=5)
    with Pool(processes=config.get_available_cores()) as p:
        for d in p.imap_unordered(_d_imap, inputs):
            if max_value is not None and d is None:
                diameter = None
                break
            diameter = max(diameter, d)
            if config.VERBOSE_LEVEL > 0:
                loop.update()
                loop.set_description(f"Max diam: {diameter:.2f}")
    return diameter


@numba.njit()
def _episodic_diameter_calculation(
    es: int,
    T: np.ndarray,
    max_diam: float,
    epsilon: float = 0.001,
    max_value: float = None,
) -> float:
    H, n_states, n_actions, _ = T.shape
    ETs = np.zeros((H, n_states), dtype=np.float32)
    ET_minh = np.zeros((n_states,), dtype=np.float32)
    for t in range(1_000_000):
        ETs_old = ETs.copy()
        ETs[-1] = T[-1, 0, 0] @ (1 + ETs[0])
        for hh in range(1, H):
            h = H - hh
            for j in range(n_states):
                if j != es:
                    s = np.zeros(n_actions, np.float32)
                    for ns in range(n_states):
                        if ns != es:
                            s += T[h - 1, j, :, ns] * (1 + ETs[h, ns])
                    ETs[h - 1, j] = np.min(T[h - 1, j, :, es] + s)
                    if max_value is not None and ETs[h - 1, j] > max_value:
                        return None

        diff = np.abs(ETs_old - ETs).max()
        for ss in range(n_states):
            ccc = ETs[:, ss]
            ET_minh[ss] = np.min(ccc[ccc > 0])
        cur_diam = ET_minh.max()
        if diff < epsilon or (diff < 0.01 and cur_diam - 1 < max_diam):
            break
    return max(max_diam, cur_diam)


@numba.njit()
def _continuous_diameter_calculation(
    es: "NODE_TYPE",
    T: np.ndarray,
    max_diam: float,
    epsilon: float = 0.001,
    max_value: float = None,
) -> float:
    n_states, n_actions, _ = T.shape
    ETs = np.zeros(n_states, dtype=np.float32)
    for t in range(1_000_000):
        ETs_old = ETs.copy()
        for j in range(n_states):
            if j != es:
                s = np.zeros(n_actions, np.float32)
                for ns in range(n_states):
                    if ns != es:
                        s += T[j, :, ns] * (1 + ETs[ns])
                ETs[j] = np.min(T[j, :, es] + s)
                if max_value is not None and ETs[j].max() > max_value:
                    return None

        diff = np.abs(ETs_old - ETs).max()
        if diff < epsilon or (diff < 0.05 and ETs.max() - 1 < max_diam):
            break
    return max(max_diam, ETs.max())


def _sparse_diameter_calculation(
    es: "NODE_TYPE",
    T: np.ndarray,
    n_actions: int,
    n_states: int,
    max_diam: float,
    epsilon: float = 0.001,
    max_value: float = None,
) -> float:
    ETs = np.zeros(n_states - 1, dtype=np.float32)

    next_ets = (
        lambda TT, ET: (TT * ET.reshape((-1, 1)))
        .reshape((n_states - 1, n_states, n_actions))
        .sum(0)
        .todense()
    )
    selector = np.ones(n_states, dtype=bool)
    selector[es] = False
    Te = T[es].reshape((n_states, n_actions))
    T_me = T[selector]

    for j in range(1_000_000):
        ETs_old = ETs.copy()
        ETs = (Te + next_ets(T_me, 1 + ETs)).min(1)[selector]
        diff = np.abs(ETs_old - ETs).max()
        if diff < epsilon or (diff < 0.05 and ETs.max() - 1 < max_diam):
            break
        if max_value is not None and ETs.max() > max_value:
            return None
    return max(max_diam, ETs.max())


def _get_sparse_diameter(
    T: np.ndarray, epsilon: float = 0.001, max_value: float = None
) -> float:
    n_states, n_actions, _ = T.shape

    T = np.moveaxis(T, -1, 0).reshape(n_states, n_states * n_actions)
    T = sparse.COO(T)
    next_ets = (
        lambda TT, ET: (TT * ET.reshape(-1, 1))
        .reshape((n_states - 1, n_states, n_actions))
        .sum(0)
        .todense()
    )

    diameter = -np.inf
    loop = (
        trange(n_states, desc="Calculating diameter", mininterval=5)
        if config.VERBOSE_LEVEL > 0
        else range(n_states)
    )
    selector = np.ones(n_states, dtype=bool)
    for i in loop:
        selector[i] = False
        selector[i - 1] = True
        Te = T[i].reshape((n_states, n_actions))
        T_me = T[selector]

        ETs = np.zeros(n_states - 1)
        diff = np.inf
        for j in range(1_000_000):
            ETs_old = ETs.copy()
            ETs = (Te + next_ets(T_me, 1 + ETs)).min(1)[selector]
            diff = np.abs(ETs_old - ETs).max()
            if diff < epsilon or (diff < 0.05 and ETs.max() - 1 < diameter):
                break
        diameter = max(diameter, ETs.max())
        if max_value is not None and diameter > max_value:
            return None
    return diameter
