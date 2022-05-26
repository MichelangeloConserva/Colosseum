from io import StringIO
from multiprocessing import Pool, cpu_count

import networkx as nx
import numba
import numpy as np
import sparse
from tqdm import tqdm, trange

from colosseum.dp.continuous import value_iteration as c_value_iteration
from colosseum.dp.episodic import value_iteration as e_value_iteration
from colosseum.utils.miscellanea import get_loop


def deterministic_diameter(G, verbose) -> float:
    A = nx.to_numpy_array(G, nonedge=np.inf)
    n, m = A.shape
    np.fill_diagonal(A, 0)  # diagonal elements should be zero
    for i in get_loop(range(n), verbose):
        A = np.minimum(A, A[i, :][np.newaxis, :] + A[:, i][:, np.newaxis])
    return np.max(A, where=A != 0, initial=-np.inf)


def numba_continuous_diameter(T, force_single_thread=False, verbose=False):
    if not force_single_thread:
        return continuous_multi_thread_diam(T, verbose)
    if T.shape[-1] > 1000:
        return sparse_diameter_calculation(T, verbose=verbose)
    return continuous_single_thread_diam(T, verbose)


def numba_episodic_diameter(H, T, reachable_nodes, force_single_thread, verbose=False):
    if not force_single_thread:
        return episodic_multi_thread_diam(H, T, reachable_nodes, verbose=verbose)
    return episodic_single_thread_diam(H, T, reachable_nodes, verbose=verbose)


def continuous_diam_calculation(es, T):
    T_es = T.copy()
    T_es[es] = 0
    T_es[es, :, es] = 1

    R_es = np.zeros(T.shape[:2], np.float32) - 1.0
    R_es[es] = 0
    _, V = c_value_iteration(T_es, R_es, 1)

    return -V.min()


def episodic_diam_calculation(es, T, H, reachable_nodes):
    T_es = T.copy()
    T_es[es] = 0
    T_es[es, :, es] = 1

    R_es = np.zeros(T.shape[:2], np.float32) - 1.0
    R_es[es] = 0
    _, V = e_value_iteration(H, T_es, R_es)
    return max(-V[h, s] for h, s in reachable_nodes)


def continuous_single_thread_diam(T, verbose=False):
    diam = 0
    for es in get_loop(range(len(T)), verbose):
        diam = max(diam, continuous_diam_calculation(es, T))
    return diam


def episodic_single_thread_diam(H, T, reachable_nodes, verbose=False):
    diam = 0
    for es in get_loop(range(len(T)), verbose):
        diam = max(diam, episodic_diam_calculation(es, T, H, reachable_nodes))
    return diam


def continuous_multi_thread_diam(T, verbose=False):
    loop = get_loop(range(len(T)), verbose)
    n_cores = cpu_count() - 2
    diam = 0
    with Pool(processes=n_cores) as p:
        for d in p.starmap(
            continuous_diam_calculation, [[es, T] for es in range(len(T))]
        ):
            diam = max(diam, d)
            if verbose:
                loop.set_description(f"Max diam: {diam:.2f}")
                loop.update()
                loop.refresh()
    return diam


def episodic_multi_thread_diam(H, T, reachable_nodes, verbose=False):
    loop = get_loop(range(len(T)), verbose)
    n_cores = cpu_count() - 2
    diam = 0
    with Pool(processes=n_cores) as p:
        for d in p.starmap(
            episodic_diam_calculation,
            [[es, T, H, reachable_nodes] for es in range(len(T))],
        ):
            diam = max(diam, d)
            if verbose:
                loop.set_description(f"Max diam: {diam:.2f}")
                loop.update()
                loop.refresh()
    return diam


def single_thread_diameter_calculation(T, states=None, epsilon=0.001, verbose=False):
    if states is None:
        num_states = T.shape[-1]
        states = range(num_states)

    diameter = -np.inf
    states = list(reversed(states))

    if verbose:
        if type(verbose) == bool:
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
        diameter = dima_f(es, T, diameter, epsilon)
        if verbose and type(verbose) == str:
            with open(verbose, "a") as f:
                f.write(
                    s.getvalue()
                    .split("\x1b")[0][2:]
                    .replace("\x00", "")
                    .replace("\n\r", "")
                    + "\n"
                )
            s.truncate(0)
    return diameter


def d_imap(args):
    T, state_set, epsilon, sparse_T = args
    if sparse_T:
        num_states, num_sa = T.shape[-2:]
        diam = 0
        for es in state_set:
            diam = _sparse_diameter_calculation(
                es, T, int(num_sa / num_states), num_states, diam, epsilon
            )
        return diam
    return single_thread_diameter_calculation(T, state_set, epsilon)


def multi_thread_diameter_calculation(T, epsilon=0.001, verbose=False):
    num_actions, num_states = T.shape[-2:]
    diameter = -np.inf

    sparse_T = False
    if num_states > 500 and T.ndim == 3:
        T = np.moveaxis(T, -1, 0).reshape(num_states, num_states * num_actions)
        T = sparse.COO(T)
        sparse_T = True

    n_cores = cpu_count() - 2
    states = np.arange(num_states)[::-1]
    # np.random.shuffle(states)
    states_sets = np.array_split(states, n_cores)
    inputs = [(T, state_set, epsilon, sparse_T) for state_set in states_sets]
    if verbose:
        loop = trange(len(inputs), desc="Diameter calculation", mininterval=5)
    with Pool(processes=n_cores) as p:
        for d in p.imap_unordered(d_imap, inputs):
            diameter = max(diameter, d)
            if verbose:
                loop.update()
                loop.set_description(f"Max diam: {diameter:.2f}")
    return diameter


@numba.njit()
def _episodic_diameter_calculation(es, T, max_diam, epsilon=0.001):
    H, num_states, num_actions, _ = T.shape
    ETs = np.zeros((H, num_states), dtype=np.float32)
    ET_minh = np.zeros((num_states,), dtype=np.float32)
    for t in range(1_000_000):
        ETs_old = ETs.copy()
        ETs[-1] = T[-1, 0, 0] @ (1 + ETs[0])
        for hh in range(1, H):
            h = H - hh
            for j in range(num_states):
                if j != es:
                    s = np.zeros(num_actions, np.float32)
                    for ns in range(num_states):
                        if ns != es:
                            s += T[h - 1, j, :, ns] * (1 + ETs[h, ns])
                    ETs[h - 1, j] = np.min(T[h - 1, j, :, es] + s)

        diff = np.abs(ETs_old - ETs).max()

        for ss in range(num_states):
            ccc = ETs[:, ss]
            ET_minh[ss] = np.min(ccc[ccc > 0])
        cur_diam = ET_minh.max()
        if diff < epsilon or (diff < 0.01 and cur_diam - 1 < max_diam):
            break
    return max(max_diam, cur_diam)


@numba.njit()
def _continuous_diameter_calculation(es, T, max_diam, epsilon=0.001):
    num_states, num_actions, _ = T.shape
    ETs = np.zeros(num_states, dtype=np.float32)
    for t in range(1_000_000):
        ETs_old = ETs.copy()
        for j in range(num_states):
            if j != es:
                s = np.zeros(num_actions, np.float32)
                for ns in range(num_states):
                    if ns != es:
                        s += T[j, :, ns] * (1 + ETs[ns])
                ETs[j] = np.min(T[j, :, es] + s)

        diff = np.abs(ETs_old - ETs).max()
        if diff < epsilon or (diff < 0.05 and ETs.max() - 1 < max_diam):
            break
    return max(max_diam, ETs.max())


def _sparse_diameter_calculation(
    es, T, num_actions, num_states, max_diam, epsilon=0.001
):
    ETs = np.zeros(num_states - 1, dtype=np.float32)

    next_ets = (
        lambda TT, ET: (TT * ET.reshape((-1, 1)))
        .reshape((num_states - 1, num_states, num_actions))
        .sum(0)
        .todense()
    )
    selector = np.ones(num_states, dtype=bool)
    selector[es] = False
    Te = T[es].reshape((num_states, num_actions))
    T_me = T[selector]

    for j in range(1_000_000):
        ETs_old = ETs.copy()
        ETs = (Te + next_ets(T_me, 1 + ETs)).min(1)[selector]
        diff = np.abs(ETs_old - ETs).max()
        if diff < epsilon or (diff < 0.05 and ETs.max() - 1 < max_diam):
            break
    return max(max_diam, ETs.max())


def sparse_diameter_calculation(T, epsilon=0.001, verbose=False):
    num_states, num_actions, _ = T.shape

    T = np.moveaxis(T, -1, 0).reshape(num_states, num_states * num_actions)
    T = sparse.COO(T)
    next_ets = (
        lambda TT, ET: (TT * ET.reshape(-1, 1))
        .reshape((num_states - 1, num_states, num_actions))
        .sum(0)
        .todense()
    )

    diameter = -np.inf
    loop = (
        trange(num_states, desc="Calculating diameter", mininterval=5)
        if verbose
        else range(num_states)
    )
    selector = np.ones(num_states, dtype=bool)
    for i in loop:
        selector[i] = False
        selector[i - 1] = True
        Te = T[i].reshape((num_states, num_actions))
        T_me = T[selector]

        ETs = np.zeros(num_states - 1)
        for j in range(1_000_000):
            ETs_old = ETs.copy()
            ETs = (Te + next_ets(T_me, 1 + ETs)).min(1)[selector]
            diff = np.abs(ETs_old - ETs).max()
            if diff < epsilon or (diff < 0.05 and ETs.max() - 1 < diameter):
                break
        diameter = max(diameter, ETs.max())
    return diameter
