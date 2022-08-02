import os
from typing import TYPE_CHECKING, Callable, List, Tuple, Union, Dict, Type, Iterable

import gin
import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from colosseum import config
from colosseum.benchmark.analysis.utils import (
    get_available_mdps_agents_prms_and_names,
    get_formatted_name,
    group_by_mdp_individual_plot,
)
from colosseum.experiment.agent_mdp_interaction import MDPLoop
from colosseum.experiment.experiment.folder_structuring import (
    get_experiment_config,
    retrieve_mdp_classes_agent_classes_gin_config,
)
from colosseum.experiment.experiment.utils import apply_gin_config
from colosseum.hardness.analysis.utils import compute_hardness_measure
from colosseum.agent.agents.base import BaseAgent
from colosseum.utils.formatter import clear_agent_mdp_class_name

if TYPE_CHECKING:
    from colosseum.mdp import BaseMDP

sns.set_theme()


def get_index(x):
    return clear_agent_mdp_class_name(x[0].__name__), x[1]

def agent_performances_per_mdp_plot(
    experiment_result_folder_path: str,
    indicator: str,
    figsize_scale: int = 8,
    save_plot_to_file: bool = True,
    path_to_save_file_to: str = None,
    color_pallete: List[str] = matplotlib.colors.TABLEAU_COLORS.keys(),
):
    """
    produces a plot in which the performance indicator of the agents is shown for each MDP for the given experiment results.

    Parameters
    ----------
    experiment_result_folder_path : str
        is the folder that contains the experiment logs, MDP configurations and agent configurations.
    indicator : str
        is a string representing the performance indicator that is shown in the plot. Check
        MDPLoop.get_available_indicators() to get a list of the available indicators.
    figsize_scale : int, optional
        is the scale for figsize in the resulting plot. The default value is 8.
    save_plot_to_file : bool, optional
        checks whether to save the plot to a file. The default value is True.
    path_to_save_file_to : str, optional
        is the path to which the save will be saved. The default value is the 'tmp' folder in the current working
        directory.
    """
    assert (
        indicator in MDPLoop.get_available_indicators()
    ), f"Please check that the indicator given in input is one from {MDPLoop.get_available_indicators()}."

    available_mdps, available_agents = get_available_mdps_agents_prms_and_names(
        experiment_result_folder_path
    )
    colors_dict_agents = dict(zip(available_agents, color_pallete))

    n_plots = len(available_mdps)
    h = int(np.ceil(n_plots ** 0.5))
    w = int(np.ceil(n_plots / h))
    fig, axes = plt.subplots(h, w, figsize=(h * figsize_scale, w * figsize_scale))
    for i, available_mdp in enumerate(sorted(available_mdps, key=lambda x: x[::-1])):
        ax = axes.ravel()[i]
        mdp_formatted_name = get_formatted_name(*available_mdp)
        group_by_mdp_individual_plot(
            experiment_result_folder_path,
            ax,
            indicator,
            *available_mdp,
            available_agents,
            colors_dict_agents,
            add_random=True,
        )
        ax.set_title(mdp_formatted_name)
        ax.legend()

    for j in range(i + 1, len(axes.ravel())):
        fig.delaxes(axes.ravel()[j])

    plt.tight_layout()
    if save_plot_to_file:
        if path_to_save_file_to is None:
            exp_name = experiment_result_folder_path[
                experiment_result_folder_path.rfind(os.sep) + 1 :
            ]
            path_to_save_file_to = f"tmp{os.sep}{indicator}-for-{exp_name}.pdf"
            print(f"The plot has been saved to {path_to_save_file_to}")
        save_folder = path_to_save_file_to[: path_to_save_file_to.rfind(os.sep)]
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(path_to_save_file_to, bbox_inches="tight")
    plt.show()


def get_hardness_measures_from_experiment_folder(
    experiment_result_folder_path: str,
    hardness_measures: Iterable[str] = ("diameter", "value_norm", "suboptimal_gaps"),
    reduce_seed: Callable[[List[float]], float] = np.mean,
) -> Dict[Tuple[Type["BaseMDP"], str], Dict[str, float]]:
    """
    for each mdp and mdp gin config in the experiment folder, it retrieves the given measures of hardness.

    Parameters
    ----------
    experiment_result_folder_path : str
        is the path of the folder that contains the experiments.
    hardness_measures : Iterable[str]
        is the list containing the measures of hardness to compute.
    reduce_seed : Callable[[List[float]], float], optional
        is the function that reduces the values of the measures for different seed to a single scalar. By default, the
        mean function is employed.
    """
    (
        mdp_classes_scopes,
        agent_classes_scopes,
        gin_config_files_paths,
    ) = retrieve_mdp_classes_agent_classes_gin_config(experiment_result_folder_path)
    exp_config = get_experiment_config(experiment_result_folder_path, None)

    res = dict()
    for mdp_class, mdp_scopes in tqdm(
        mdp_classes_scopes.items(), desc=os.path.basename(experiment_result_folder_path)
    ):
        for mdp_scope in mdp_scopes:
            apply_gin_config(gin_config_files_paths)
            with gin.config_scope(mdp_scope):
                res[mdp_class, mdp_scope] = {
                    hm: reduce_seed(
                        [
                            compute_hardness_measure(
                                mdp_class,
                                dict(seed=seed),
                                hm,
                                config.get_hardness_measures_cache_folder()
                                + mdp_class.__name__
                                + os.sep,
                            )
                            for seed in range(exp_config.n_seeds)
                        ]
                    )
                    for hm in hardness_measures
                }
    return res


def plot_labels_on_benchmarks_hardness_space(
    experiment_result_folder_path: str,
    text_f: Callable[[Tuple[str, str]], str],
    color_f: Callable[[Tuple[str, str]], str] = lambda x: None,
    label_f: Callable[[Tuple[str, str]], str] = lambda x: None,
    ax: plt.Axes = None,
    multiplicative_factor_xlim=1.0,
    multiplicative_factor_ylim=1.0,
    legend_ncol=1,
    underneath_x_label: str = None,
    set_ylabel=True,
    set_legend=True,
    xaxis_measure: Union[str, Tuple[str, Callable[["BaseMDP"], float]]] = "diameter",
    yaxis_measure: Union[str, Tuple[str, Callable[["BaseMDP"], float]]] = "value_norm",
    fontsize : int = 22,
    fontsize_xlabel : int =32
):
    """
    Locates the value obtain from 'label_f' in the location corresponding to xaxis_measure and yaxis_measure.

    Parameters
    ----------
    experiment_result_folder_path : str.
        A directory path containing the 'mdp_configs' folder with the MDPs configurations to be analyzed.
    text_f : Callable.
        A function that assigns text to each tuple of MDP class name and parameter.
        For example, ('DeepSeaEpisodic', 'prms_0') -> 0.
    color_f : Callable, optional.
        A function that assigns a color to each tuple of MDP class name and parameter.
        For example, ('DeepSeaEpisodic', 'prms_0') -> "red".
    label_f : Callable, optional.
        A function that assigns a label to each tuple of MDP class name and parameter.
        For example, ('DeepSeaEpisodic', 'prms_0') -> "DeepSea".
    ax : plt.Axes, optional.
        A matplotlib axes on which the values are positioned.
    multiplicative_factor_xlim : float, optional.
        Additional space to add on the right side of the figure. May be useful to add space for the legend.
    multiplicative_factor_ylim : float, optional.
        Additional space to add on the top side of the figure. May be useful to add space for the legend.
    legend_ncol : int, optional.
        The number of columns in the legend.
    underneath_x_label : str, optional.
        Text to be added underneath the x_label.
    set_ylabel : bool, optional.
        Whether to set the y_label or not.
    set_legend : bool, optional.
        Whether to set the legend or not.
    xaxis_measure : Union[str, Tuple[str, Callable[[Union[EpisodicMDP,ContinuousMDP]], float]]], optional.
        The hardness measure for the x axis. If given as a string, it should be one of those implemented in Colosseum.
        Otherwise it can be given as a tuple with the first element being the name of the measure and the second element
        being a function that takes in input an MDP and returns a value. The default value is the diameter.
    yaxis_measure : Union[str, Tuple[str, Callable[[Union[EpisodicMDP,ContinuousMDP]], float]]], optional.
        The hardness measure for the y axis. If given as a string, it should be one of those implemented in Colosseum.
        Otherwise it can be given as a tuple with the first element being the name of the measure and the second element
        being a function that takes in input an MDP and returns a value. The default value is the environmental value norm.
    """

    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        set_ylabel = True

    hardness_measures = get_hardness_measures_from_experiment_folder(
        experiment_result_folder_path, (xaxis_measure, yaxis_measure)
    )
    texts = []
    for k, r in hardness_measures.items():
        texts.append(
            ax.text(
                r[xaxis_measure],
                r[yaxis_measure],
                text_f(k),
                fontdict=dict(fontsize=fontsize),
            )
        )
        ax.scatter(
            r[xaxis_measure], r[yaxis_measure], 100, color=color_f(k), label=label_f(k)
        )

    ax.tick_params(labelsize=22)
    if set_ylabel:
        ax.set_ylabel(
            yaxis_measure.capitalize().replace("_", " "),
            fontdict=dict(fontsize=fontsize),
            labelpad=10,
        )
    ax.set_xlabel(
        xaxis_measure.capitalize().replace("_", " "),
        fontdict=dict(fontsize=fontsize),
        labelpad=15,
        ha="center",
    )

    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] * multiplicative_factor_xlim)
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] * multiplicative_factor_ylim)

    if type(underneath_x_label) == str:
        ax.text(
            np.mean(ax.get_xlim()),
            ylim[0] - 0.28 * (ylim[1] - ylim[0]),
            underneath_x_label,
            fontdict=dict(fontsize=fontsize_xlabel),
            ha="center",
        )
    # adjust_text(texts)
    if set_legend:
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(ncol=legend_ncol)
    if show:
        plt.tight_layout()
        plt.show()
