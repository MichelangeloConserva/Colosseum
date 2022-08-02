from __future__ import annotations

import warnings
from copy import deepcopy
from enum import IntEnum
from functools import reduce
from typing import TYPE_CHECKING, Iterable

import networkx as nx
import numba
import numpy as np
from numba import bool_, types
from numba.core.errors import NumbaTypeSafetyWarning
from numba.typed import Dict
from tqdm import trange

import colosseum.config as config

warnings.simplefilter("ignore", category=NumbaTypeSafetyWarning)


if TYPE_CHECKING:
    from colosseum.mdp import NODE_TYPE


class MDPCommunicationClass(IntEnum):
    ERGODIC = 0
    COMMUNICATING = 1
    WEAKLY_COMMUNICATING = 2


def get_recurrent_nodes_set(
    communication_type: MDPCommunicationClass, G: nx.DiGraph
) -> Iterable[NODE_TYPE]:
    """
    returns the recurrent node set. Note that for ergodic and communicating MDPs this corresponds to the state space.
    """
    if communication_type == MDPCommunicationClass.WEAKLY_COMMUNICATING:
        c = nx.condensation(G)
        leaf_nodes = [x for x in c.nodes() if c.out_degree(x) == 0]
        assert len(leaf_nodes) == 1
        return c.nodes(data="members")[leaf_nodes[0]]
    return G.nodes


def get_communication_class(T: np.ndarray, G: nx.DiGraph):
    """
    returns the communication class for the MDP.
    """
    if T.ndim == 4:  # episodic MDP
        assert (
            len(list(G.nodes)[0]) == 2
        ), "For an episodic MDP, you must input a episodic graph form."
        return _get_episodic_MDP_class(T, G)
    return _get_continuous_MDP_class(T)


def _get_episodic_MDP_class(T, episodic_graph: nx.DiGraph):
    G = episodic_graph.copy()
    for (h, u), (hp1, v) in episodic_graph.edges():
        if not (T[h, u, :, v] > 0).all():
            G.remove_edge((h, u), (hp1, v))

    if _check_ergodicity(G, T, True):
        return MDPCommunicationClass.ERGODIC
    return (
        MDPCommunicationClass.COMMUNICATING
    )  # if an episodic MDP is not ergodic is, by definition, communicating


def _get_continuous_MDP_class(T):
    return _calculate_MDP_class(T)


def _calculate_MDP_class(T) -> MDPCommunicationClass:
    G_1 = nx.DiGraph(np.all(T > 0, axis=1))
    if _check_ergodicity(G_1, T, False):
        return MDPCommunicationClass.ERGODIC

    G_2 = nx.DiGraph(np.any(T > 0, axis=1))
    G_2.remove_edges_from(nx.selfloop_edges(G_2))
    sccs = list(nx.strongly_connected_components(G_2))
    if len(sccs) == 1:
        return MDPCommunicationClass.COMMUNICATING

    m = 0
    T_ = set()
    R = []
    for C_k in sccs:
        is_closed = not np.any(
            np.delete(
                T[
                    list(C_k),
                ],
                list(C_k),
                axis=-1,
            )
            > 0
        )
        if is_closed:
            m += 1
            R.append(C_k)
        else:
            T_ = T_.union(C_k)

    if m == 1:
        return MDPCommunicationClass.WEAKLY_COMMUNICATING

    return MDPCommunicationClass.NON_WEAKLY_COMMUNICATING


def _condense_mpd_graph_old(G_ccs, T):
    adj = np.zeros((len(G_ccs.keys()), len(G_ccs.keys())), dtype=bool)
    for k in G_ccs:
        for l in G_ccs:
            if k == l:
                continue
            if (
                T[np.array(G_ccs[k]).reshape(-1, 1), :, np.array(G_ccs[l])]
                .sum(1)
                .min(1)
                .max()
                > 0
            ):
                adj[k, l] = True
    return adj


@numba.njit()
def _condense_mpd_graph(G_ccs, T, d):
    _, num_actions, _ = T.shape
    adj = np.zeros((d, d), dtype=bool_)
    for k in G_ccs:
        for l in G_ccs:
            if k == l:
                continue
            M = np.zeros(len(G_ccs[k]), np.float32)
            for i, r in enumerate(G_ccs[k]):
                min_a = np.inf
                for a in range(num_actions):
                    summation = 0.0
                    for s in G_ccs[l]:
                        summation += T[r, a, s]
                        if summation > min_a:
                            break
                    min_a = min(min_a, summation)
                M[i] = min_a
            if M.max() > 0:
                adj[k, l] = True
    return adj


@numba.njit()
def _condense_mpd_graph_episodic(G_ccs, T, d):
    H, _, num_actions, _ = T.shape
    adj = np.zeros((d, d), dtype=bool_)
    for k in G_ccs:
        for l in G_ccs:
            if k == l:
                continue
            M = np.zeros(len(G_ccs[k]), np.float32)
            for i, (hr, r) in enumerate(G_ccs[k]):
                min_a = np.inf
                for a in range(num_actions):
                    summation = 0.0
                    for hs, s in G_ccs[l]:
                        if hr + 1 == hs or (hr + 1 == H and hs == 0):
                            summation += T[hr, r, a, s]
                            if summation > min_a:
                                break
                    min_a = min(min_a, summation)
                M[i] = min_a
            if M.max() > 0:
                adj[k, l] = True
    return adj


def _get_ultimate_condensation(G, T, is_episodic=False):
    mapping = {i: tuple(cc) for i, cc in enumerate(nx.strongly_connected_components(G))}

    loop = (
        trange(1_000_000, desc="Communication class calculation", mininterval=5)
        if config.VERBOSE_LEVEL > 0
        else range(1_000_000)
    )
    for _ in loop:
        old_mapping = deepcopy(mapping)
        if is_episodic:
            d = Dict.empty(
                key_type=types.int16, value_type=types.Array(types.int16, 2, "A")
            )
            for k, v in mapping.items():
                d[k] = np.array(v).reshape(-1, 2).astype(np.int16)
            new_G_c = nx.DiGraph(_condense_mpd_graph_episodic(d, T, len(mapping)))
        else:
            d = Dict.empty(
                key_type=types.int16, value_type=types.Array(types.int16, 1, "A")
            )
            for k, v in mapping.items():
                d[k] = np.array(v).astype(np.int16)
            new_G_c = nx.DiGraph(_condense_mpd_graph(d, T, len(mapping)))

        new_mapping = {
            i: reduce(lambda x, y: x + y, (mapping[c] for c in cc))
            for i, cc in enumerate(nx.strongly_connected_components(new_G_c))
        }
        if old_mapping == new_mapping:
            return new_mapping
        mapping = new_mapping


def _check_ergodicity(G_1, T, is_episodic):
    G_1.remove_edges_from(nx.selfloop_edges(G_1))
    G_1_c_star_mapping = _get_ultimate_condensation(G_1, T, is_episodic=is_episodic)
    if len(G_1_c_star_mapping.keys()) == 1:
        return True
    return False
