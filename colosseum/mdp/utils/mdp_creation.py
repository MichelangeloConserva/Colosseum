import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

import networkx as nx
import numpy as np
import sparse
from numpy.core._exceptions import _ArrayMemoryError
from scipy.stats import rv_continuous
from tqdm import tqdm

from colosseum import config
from colosseum.mdp.utils.custom_samplers import NextStateSampler

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE, NODE_TYPE, BaseMDP


@dataclass()
class NodeInfoClass:
    """
    The data class containing some quantities related to the nodes.
    """
    transition_distributions: Dict[int, NextStateSampler]
    """A dictionary that maps actions to next state distributions."""
    actions_visitation_count: Dict[int, int]
    """The dictionary that keeps the count of how many times each action has been selected for the current node."""
    state_visitation_count: int = 0
    """The counter for the number of times the node has been visited."""

    def update_visitation_counts(self, action: int = None):
        self.state_visitation_count += 1
        if action is not None:
            self.actions_visitation_count[action] += 1

    def sample_next_state(self, action: int):
        return self.transition_distributions[action].sample()


def get_transition_matrix_and_rewards(
    n_states: int,
    n_actions: int,
    G: nx.DiGraph,
    get_info_class: Callable[["NODE_TYPE"], NodeInfoClass],
    get_reward_distribution: Callable[
        ["NODE_TYPE", "ACTION_TYPE", "NODE_TYPE"], rv_continuous
    ],
    node_to_index: Dict["NODE_TYPE", int],
    is_sparse: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    np.ndarray
        The transition 3d array of the MDP.
    np.ndarray
        The reward matrix of the MDP.
    """
    if not is_sparse:
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        is_sparse = n_states ** 2 * n_actions * np.float32().itemsize > 0.1 * mem_bytes
    if is_sparse:
        T = dict()
    else:
        T = np.zeros(
            (n_states, n_actions, n_states),
            dtype=np.float32,
        )

    R = np.zeros((n_states, n_actions), dtype=np.float32)
    for i, node in enumerate(G.nodes()):
        for action, td in get_info_class(node).transition_distributions.items():
            r = 0
            for next_state, prob in zip(td.next_nodes, td.probs):
                r += prob * get_reward_distribution(node, action, next_state).mean()
                if is_sparse and (i, action, node_to_index[next_state]) not in T:
                    T[i, action, node_to_index[next_state]] = prob
                    continue
                T[i, action, node_to_index[next_state]] += prob
            R[i, action] = r

    if is_sparse:
        coords = [[], [], []]
        data = []
        for k, v in T.items():
            coords[0].append(k[0])
            coords[1].append(k[1])
            coords[2].append(k[2])
            data.append(np.float32(v))
        T = sparse.COO(coords, data, shape=(n_states, n_actions, n_states))

    assert np.isclose(T.sum(-1).todense() if is_sparse else T.sum(-1), 1).all()
    assert np.isnan(R).sum() == 0
    return T, R


def get_episodic_transition_matrix_and_rewards(
    H: int,
    T: np.ndarray,
    R: np.ndarray,
    starting_node_sampler: NextStateSampler,
    node_to_index: Dict["NODE_TYPE", int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    np.ndarray
        The episodic transition 4d array of the MDP.
    np.ndarray
        The 3d episodic reward array of the MDP.
    """
    n_states, n_actions = R.shape
    T_epi = np.zeros(
        (H, n_states, n_actions, n_states),
        dtype=np.float32,
    )
    for sn, p in starting_node_sampler.next_nodes_and_probs:
        sn = node_to_index[sn]
        T_epi[0, sn] = T[sn]
        T_epi[H - 1, :, :, sn] = p
    for h in range(1, H - 1):
        for s in range(len(T)):
            if T_epi[h - 1, :, :, s].sum() > 0:
                T_epi[h, s] = T[s]
    R = np.tile(R, (H, 1, 1))
    R[-1] = 0.0
    return T_epi, R


def get_continuous_form_episodic_transition_matrix_and_rewards(
    H: int,
    G: nx.DiGraph,
    T: np.ndarray,
    R: np.ndarray,
    starting_node_sampler: NextStateSampler,
    node_to_index: Dict["NODE_TYPE", int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    np.ndarray
        The transition 3d array for the continuous form of the MDP.
    np.ndarray
        The reward matrix for the continuous form of the MDP.
    """
    _, n_action = R.shape

    try:
        T_epi = np.zeros((len(G.nodes), n_action, len(G.nodes)), np.float32)
        R_epi = np.zeros((len(G.nodes), n_action), np.float32)
    except _ArrayMemoryError:
        raise ValueError(
            "It is not possible calculate the value for this MDP. Its continuous form is too large."
        )

    nodes = list(G.nodes)
    for h, n in (
        tqdm(
            G.nodes,
            desc="Continuous form episodic transition matrix and rewards",
        )
        if config.VERBOSE_LEVEL > 0
        else G.nodes
    ):
        if h == H - 1:
            for sn, p in starting_node_sampler.next_nodes_and_probs:
                T_epi[nodes.index((h, n)), :, node_to_index[sn]] = p
                R_epi[nodes.index((h, n))] = R[n]
        else:
            for hp1, nn in G.successors((h, n)):
                T_epi[nodes.index((h, n)), :, nodes.index((hp1, nn))] = T[n, :, nn]
                R_epi[nodes.index((h, n))] = R[n]

    assert np.isclose(T_epi.sum(-1), 1.0).all()
    return T_epi, R_epi


def get_episodic_graph(
    G: nx.DiGraph,
    H: int,
    node_to_index: Dict["NODE_TYPE", int],
    starting_nodes,
    remove_label=False,
) -> nx.DiGraph:
    """
    Returns
    -------
    nx.DiGraph
        The graph of the MDP augmented with the time step in the state space.
    """

    def add_successors(n, h):
        n_ = node_to_index[n] if remove_label else n
        if h < H - 1:
            successors = G.successors(n)
        else:
            successors = starting_nodes
        for succ in successors:
            succ_ = node_to_index[succ] if remove_label else succ
            next_h = (h + 1) if h + 1 != H else 0
            G_epi.add_edge((h, n_), (next_h, succ_))
            if h < H - 1 and len(list(G_epi.successors((next_h, succ_)))) == 0:
                add_successors(succ, next_h)

    G_epi = nx.DiGraph()
    for sn in starting_nodes:
        add_successors(sn, 0)
    return G_epi


def instantiate_transitions(mdp: "BaseMDP", node: "NODE_TYPE"):
    """
    recursively instantiate the transitions of MDPs.
    """
    if not mdp.G.has_node(node) or len(list(mdp.G.successors(node))) == 0:
        transition_distributions = dict()
        for a in range(mdp.n_actions):
            td = _instantiate_individual_transition(mdp, node, a)

            if not td.is_deterministic:
                mdp._are_all_transition_deterministic = False

            for ns in td.next_nodes:
                instantiate_transitions(mdp, ns)
            transition_distributions[mdp._inverse_action_mapping(node, a)] = td

        assert all(
            action in transition_distributions.keys() for action in range(mdp.n_actions)
        )
        _add_node_info_class(mdp, node, transition_distributions)


def _compute_transition(mdp: "BaseMDP", next_states, probs, node, action, next_node, p):
    next_states.append(next_node)
    probs.append(p)
    if (
        mdp._are_all_rewards_deterministic
        and mdp.get_reward_distribution(node, action, next_node).dist.name
        != "deterministic"
    ):
        mdp._are_all_rewards_deterministic = False
    mdp.G.add_edge(node, next_node)


def _add_node_info_class(
    mdp: "BaseMDP",
    n: "NODE_TYPE",
    transition_distributions: Dict[int, NextStateSampler],
):
    """
    adds a container class (NodeInfoClass) in the state n containing the transition distributions.

    Parameters
    ----------
    n : NodeType
        the state to which it adds the NodeInfoClass
    transition_distributions : Dict[int, NextStateSampler]
        the dictionary containing the transition distributions.
    """
    mdp.G.nodes[n]["info_class"] = NodeInfoClass(
        transition_distributions=transition_distributions,
        actions_visitation_count=dict.fromkeys(range(mdp.n_actions), 0),
    )


def _get_next_node(
    mdp: "BaseMDP", node: "NODE_TYPE", action: "ACTION_TYPE"
) -> List[Tuple["NODE_TYPE", float]]:
    return [
        (mdp.get_node_class()(**node_prms), prob)
        for node_prms, prob in mdp._get_next_nodes_parameters(node, action)
    ]


def _instantiate_individual_transition(
    mdp: "BaseMDP", node: "NODE_TYPE", action: "ACTION_TYPE"
) -> NextStateSampler:
    next_nodes = []
    probs = []
    for next_node, p in _get_next_node(mdp, node, action):
        p1_lazy = 1.0 if mdp._p_lazy is None else (1 - mdp._p_lazy)
        p = p1_lazy * p
        p = (
            p
            if mdp._p_rand is None
            else ((1 - mdp._p_rand) * p + p * mdp._p_rand / mdp.n_actions)
        )
        _compute_transition(mdp, next_nodes, probs, node, action, next_node, p)
    if mdp._p_lazy is not None:
        next_node = node
        _compute_transition(
            mdp, next_nodes, probs, node, action, next_node, mdp._p_lazy
        )
    if mdp._p_rand is not None:
        for a in range(mdp.n_actions):
            if a == action:
                continue
            for next_node, p in _get_next_node(mdp, node, a):
                p = p1_lazy * mdp._p_rand * p / mdp.n_actions
                # p = p1_lazy * p
                _compute_transition(mdp, next_nodes, probs, node, action, next_node, p)

    assert np.isclose(sum(probs), 1.0)

    return NextStateSampler(
        next_nodes=next_nodes,
        probs=probs,
        seed=mdp._produce_random_seed(),
    )
