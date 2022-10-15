from typing import Dict, Tuple, Union, TYPE_CHECKING, List

import networkx as nx
import numpy as np
import seaborn as sns
import toolz
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from colosseum.mdp.utils import MDPCommunicationClass

if TYPE_CHECKING:
    from colosseum.mdp import ContinuousMDP, EpisodicMDP, NODE_TYPE

custom_dark = sns.color_palette("dark")
p = custom_dark.pop(1)


def plot_MDP_graph(
    mdp: Union["ContinuousMDP", "EpisodicMDP"],
    node_palette: List[Tuple[float, float, float]] = sns.color_palette("bright"),
    action_palette: List[Tuple[float, float, float]] = custom_dark,
    save_file: str = None,
    ax: plt.Axes = None,
    figsize: Tuple[int, int] = None,
    node_labels: Union[bool, Dict["NODE_TYPE", Union[float, str]]] = None,
    action_labels: Union[bool, Dict[Tuple["NODE_TYPE", int], Union[float, str]]] = None,
    int_labels_offset_x: int = 10,
    int_labels_offset_y: int = 10,
    continuous_form: bool = True,
    prog: str = "neato",
    ncol: int = 4,
    title: str = None,
    legend_fontsize: int = None,
    font_color_state_labels: str = "k",
    font_color_state_actions_labels: str = "k",
    cm_state_labels: LinearSegmentedColormap = None,
    cm_state_actions_labels: LinearSegmentedColormap = None,
    no_written_state_labels=True,
    no_written_state_action_labels=True,
    node_size=150,
):
    """
    In the MDP graph representation, states are associated to round nodes and actions to square nodes. Each state is
    connected to action nodes for all the available actions. Each action node is connected to the state node such that
    the probability of transitioning to them when playing the corresponding action from the corresponding state is
    greater than one.

    Parameters
    ----------
    mdp : Union["ContinuousMDP", "EpisodicMDP"]
        The MDP to be visualised.
    node_palette : List[Tuple[float, float, float]]
        The color palette for the nodes corresponding to states. By default, the seaborn 'bright' palette is used.
    action_palette : List[Tuple[float, float, float]]
        The color palette for the nodes corresponding to actions. By default, the seaborn 'dark' palette is used.
    save_file : str
        The path file where the visualization will be store. Be default, the visualization will not be stored.
    ax : plt.Axes
        The ax object where the plot will be put. By default, a new axis is created.
    figsize : Tuple[int, int]
        The size of the figures in the grid of plots. By default, None is provided.
    node_labels : Union[bool, Dict["NODE_TYPE", Union[float, str]]]
        The dictionary mapping state nodes to labels (either a float or a str). If provided as bool, then the default labels
        are used. By default, no label is plotted.
    action_labels : Union[bool, Dict[Tuple["NODE_TYPE", int], Union[float, str]]]
        The dictionary mapping action nodes to labels (either a float or a str). If provided as bool, then the default labels
        are used. By default, no label is plotted.
    int_labels_offset_x : int
        x-axis offset for the node labels. By default, the offset is set to 10.
    int_labels_offset_y : int
        y-axis offset for the node labels. By default, the offset is set to 10.
    continuous_form : bool
        If True and the MDP is episodic, the continuous form of the MDP is used. By default, the continuous form is
        used.
    prog : str
        The default program to create the graph layout to be passed to the networkx library. By default, 'neato' is used.
    ncol : int
        The number of columns in the legend. By default, four columns are used.
    title : str
        The title to be given to the plot. By default, no title is used.
    legend_fontsize : int
        The font size of the legend. Vy default, it is not specified.
    font_color_state_labels : str
        The color code for the labels of the state nodes. By default, it is set to 'k'.
    font_color_state_actions_labels : str
        The color code for the labels of the action nodes. By default, it is set to 'k'.
    cm_state_labels : LinearSegmentedColormap
        The matplotlib color map for the labels of the state nodes. By default, it is not specified.
    cm_state_actions_labels : LinearSegmentedColormap
        The matplotlib color map for the labels of the action nodes. By default, it is not specified.
    no_written_state_labels : bool
        If True, the labels for the node states are not shown. By default, it is set to True.
    no_written_state_action_labels : bool
        If True, the labels for the node actions are not shown. By default, it is set to True.
    node_size : int
        The size of the node in the plot. By default, it is set to 100.
    """
    show = ax is None

    sns.reset_defaults()
    G, probs = (
        _create_epi_MDP_graph(mdp)
        if mdp.is_episodic() and not continuous_form
        else _create_MDP_graph(mdp)
    )
    T, R = mdp.transition_matrix_and_rewards

    layout = nx.nx_agraph.graphviz_layout(G, prog=prog)

    if ax is None:
        ax = _create_ax(layout, figsize)

    if mdp.is_episodic() and not continuous_form:
        if node_labels is not None and cm_state_labels is not None:
            node_color = [
                cm_state_labels(node_labels[node] / max(node_labels.values()))
                for node in mdp.get_episodic_graph(False).nodes
            ]
        else:
            node_color = [
                node_palette[5]  # brown
                if node[1] in mdp.starting_nodes and node[0] == 0
                else node_palette[2]  # green
                if R[mdp.node_to_index(node[1])].max() == R.max()
                else node_palette[-2]  # yellow
                if node[1] in mdp.recurrent_nodes_set
                else node_palette[-3]  # grey
                for node in mdp.get_episodic_graph(False).nodes
            ]
    else:
        if node_labels is not None and cm_state_labels is not None:
            node_color = [
                cm_state_labels(node_labels[node] / max(node_labels.values()))
                for node in mdp.G.nodes
            ]
        else:
            node_color = [
                node_palette[5]  # brown
                if node in mdp.starting_nodes
                else node_palette[2]  # green
                if R[mdp.node_to_index[node]].max() == R.max()
                else node_palette[-2]  # yellow
                if node in mdp.recurrent_nodes_set
                else node_palette[-3]  # grey
                for node in mdp.G.nodes
            ]

    # Lazy way to create nice legend handles
    x, y = list(layout.values())[0]
    if cm_state_labels is None:
        ax.scatter(x, y, color=node_palette[2], label="Highly rewarding state")
        ax.scatter(x, y, color=node_palette[-2], label="State")
        if mdp.communication_class == MDPCommunicationClass.WEAKLY_COMMUNICATING:
            ax.scatter(x, y, color=node_palette[-3], label="Transient state")
        ax.scatter(x, y, color=node_palette[5], label="Starting state")
    ax.plot(x, y, color=node_palette[-3], label="Transition probability")
    if cm_state_actions_labels is None:
        for a in range(mdp.n_actions):
            ax.plot(x, y, color=action_palette[a], label=f"Action: {a}", marker="s")

    G_nodes = (
        mdp.get_episodic_graph(False).nodes
        if mdp.is_episodic() and not continuous_form
        else mdp.G.nodes
    )
    nx.draw_networkx_nodes(
        G,
        layout,
        G_nodes,
        ax=ax,
        node_color=node_color,
        edgecolors="black",
        node_size=node_size,
    )
    for a in range(mdp.n_actions):
        na_nodes = (
            [an for an in G.nodes if type(an[1]) == int and an[-1] == a]
            if mdp.is_episodic() and not continuous_form
            else [an for an in G.nodes if type(an) == tuple and an[-1] == a]
        )
        nx.draw_networkx_nodes(
            G,
            layout,
            na_nodes,
            node_shape="s",
            ax=ax,
            node_size=node_size,
            node_color=[action_palette[a]]
            if cm_state_actions_labels is None
            else [
                cm_state_actions_labels(action_labels[an] / max(action_labels.values()))
                for an in na_nodes
            ],
            edgecolors="black",
        )
        nx.draw_networkx_edges(
            G,
            layout,
            edgelist=[e for e in G.edges if type(e[0][0]) != tuple and e[1][1] == a]
            if mdp.is_episodic() and not continuous_form
            else [e for e in G.edges if type(e[0]) != tuple and e[1][1] == a],
            ax=ax,
            edge_color=action_palette[a],
        )
    nx.draw_networkx_edges(
        G,
        layout,
        edgelist=[e for e in G.edges if type(e[0][0]) == tuple]
        if mdp.is_episodic() and not continuous_form
        else [e for e in G.edges if type(e[0]) == tuple],
        ax=ax,
        edge_color=action_palette[-3],
        width=[probs[e] for e in G.edges if type(e[0][0]) == tuple]
        if mdp.is_episodic() and not continuous_form
        else [probs[e] for e in G.edges if type(e[0]) == tuple],
    )
    ax.legend(ncol=ncol, fontsize=legend_fontsize)
    if node_labels is not None and not no_written_state_labels:
        if type(node_labels) == bool and node_labels:
            node_labels = {
                n: (
                    f"h={n[0]},{n[1]}"
                    if mdp.is_episodic() and not continuous_form
                    else str(n)
                )
                for n in G_nodes
            }
        assert all(n in G.nodes for n in node_labels)
        nx.draw_networkx_labels(
            G,
            toolz.valmap(
                lambda x: [x[0] + int_labels_offset_x, x[1] + int_labels_offset_y],
                layout,
            ),
            node_labels,
            font_color=font_color_state_labels,
            ax=ax,
            verticalalignment="center_baseline",
        )
    if action_labels is not None and not no_written_state_action_labels:
        if type(action_labels) == bool and action_labels:
            action_labels = {
                n: str(n[1])
                for n in (
                    (an for an in G.nodes if type(an[1]) == int)
                    if mdp.is_episodic() and not continuous_form
                    else (an for an in G.nodes if type(an) == tuple)
                )
            }
        assert all(n in G.nodes and type(n[1]) == int for n in action_labels)
        nx.draw_networkx_labels(
            G,
            toolz.valmap(
                lambda x: [x[0] + int_labels_offset_x, x[1] + int_labels_offset_y],
                layout,
            ),
            action_labels,
            font_color=font_color_state_actions_labels,
            ax=ax,
            verticalalignment="center_baseline",
        )

    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    if save_file is not None:
        plt.savefig(save_file, bbox_inches="tight")
    if show:
        plt.show()


def plot_MCGraph(
    mdp: Union["ContinuousMDP", "EpisodicMDP"],
    node_palette: List[Tuple[float, float, float]] = sns.color_palette("deep"),
    labels: Union[bool, Dict["NODE_TYPE", Union[float, str]]] = {},
    font_color_labels="k",
    save_file: str = None,
    ax: plt.Axes = None,
    figsize: Tuple[int, int] = None,
    prog: str = None,
    legend_fontsize: int = None,
    node_size=100,
    cm_state_labels=None,
    no_written_state_labels=True,
):
    """
    In the Markov chain-based representation, states are associated to nodes and actions are not visualized. Each state
    is connected to states such that there exists at least one action with corresponding transition probability greater
    than zero.

    Parameters
    ----------
    mdp : Union["ContinuousMDP", "EpisodicMDP"]
        The MDP to be visualized.
    node_palette : List[Tuple[float, float, float]]
        The color palette for the nodes. By default, the seaborn 'bright' palette is used.
    labels : Union[bool, Dict["NODE_TYPE", Union[float, str]]]
        The dictionary mapping nodes to labels (either a float or a str). If provided as bool, then the default labels
        are used. By default, no label is plotted.
    font_color_labels
    save_file : str
        The path file where the visualization will be store. Be default, the visualization will not be stored.
    ax : plt.Axes
        The ax object where the plot will be put. By default, a new axis is created.
    figsize : Tuple[int, int]
        The size of the figures in the grid of plots. By default, None is provided.
    prog : str
        The default program to create the graph layout to be passed to the networkx library. By default, it is not
        specified.
    legend_fontsize : int
        The font size of the legend. Vy default, it is not specified.
    node_size : int
        The size of the node in the plot. By default, it is set to 100.
    cm_state_labels : LinearSegmentedColormap
        The matplotlib color map for the labels of the state nodes. By default, it is not specified.
    no_written_state_labels : bool
        If True, the labels for the nodes are not shown. By default, it is set to True.

    """

    show = ax is None

    _, R = mdp.transition_matrix_and_rewards

    if cm_state_labels is not None:
        node_color = [
            cm_state_labels(labels[n] / max(labels.values())) for n in mdp.G.nodes
        ]
    else:
        node_color = [
            node_palette[0]  # brown
            if node in mdp.starting_nodes
            else node_palette[2]  # green
            if R[mdp.node_to_index[node]].max() == R.max()
            else node_palette[1]  # yellow
            if node in mdp.recurrent_nodes_set
            else node_palette[-1]  # grey
            for node in mdp.G.nodes
        ]

    if ax is None:
        ax = _create_ax(mdp.graph_layout, figsize)

    if cm_state_labels is None:
        x, y = list(mdp.graph_layout.values())[0]
        ax.scatter(x, y, color=node_palette[2], label="Highly rewarding state")
        ax.scatter(x, y, color=node_palette[1], label="State")
        if mdp.communication_class == MDPCommunicationClass.WEAKLY_COMMUNICATING:
            ax.scatter(x, y, color=node_palette[-1], label="Transient state")
        ax.scatter(x, y, color=node_palette[0], label="Starting state")

    nx.draw(
        mdp.G,
        mdp.graph_layout
        if prog is None
        else nx.nx_agraph.graphviz_layout(mdp.G, prog=prog),
        node_color=node_color,
        node_size=node_size,
        edgecolors="black",
        edge_color=node_palette[-3],
        labels={}
        if cm_state_labels is not None and no_written_state_labels
        else labels,
        font_color=font_color_labels,
        ax=ax,
    )

    if cm_state_labels is None:
        ax.legend(fontsize=legend_fontsize)
    if save_file is not None:
        plt.savefig(save_file)
    if show:
        plt.show()


def _create_ax(layout, figsize: Tuple[int, int] = None):
    if figsize is None:
        positions = np.array(list(layout.values()))
        max_distance = max(
            np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
            for i in range(len(layout))
            for j in range(i + 1, len(layout))
        )
        figsize = max(6, min(20, int(max_distance / 70)))
        figsize = (figsize, figsize)
    plt.figure(None, figsize)
    cf = plt.gcf()
    cf.set_facecolor("w")
    ax = cf.add_axes((0, 0, 1, 1)) if cf._axstack() is None else cf.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    return ax


def _create_MDP_graph(mdp: Union["ContinuousMDP", "EpisodicMDP"]):
    T, R = mdp.transition_matrix_and_rewards

    probs = dict()
    G = nx.DiGraph()
    for s in range(mdp.n_states):
        n = mdp.index_to_node[s]
        if s not in G.nodes:
            G.add_node(n)
        for a in range(mdp.n_actions):
            an = tuple((n, a))
            G.add_edge(n, an)
            for nn in np.where(T[s, a] > 0)[0]:
                G.add_edge(an, mdp.index_to_node[nn])
                probs[an, mdp.index_to_node[nn]] = T[s, a, nn]

    return G, probs


def _create_epi_MDP_graph(mdp: "EpisodicMDP"):
    G_epi = mdp.get_episodic_graph(False)
    T, R = mdp.episodic_transition_matrix_and_rewards

    probs = dict()
    G = nx.DiGraph()
    for n in G_epi.nodes:
        if n not in G.nodes:
            G.add_node(n)

        for a in range(mdp.n_actions):
            an = tuple((n, a))
            G.add_edge(n, an)
            for nn in G_epi.successors(n):
                G.add_edge(an, nn)
                probs[an, nn] = T[
                    n[0], mdp.node_to_index(n[1]), a, mdp.node_to_index(nn[1])
                ]

    return G, probs
