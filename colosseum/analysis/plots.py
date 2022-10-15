import os
import re
from string import ascii_lowercase
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Tuple,
    Union,
    Dict,
    Type,
    Iterable,
    Optional,
)

import gin
import matplotlib
import numpy as np
import seaborn as sns
from adjustText import adjust_text
from matplotlib import pyplot as plt
from tqdm import tqdm

from colosseum import config
from colosseum.analysis.tables import get_latex_table_of_average_indicator
from colosseum.analysis.utils import get_available_mdps_agents_prms_and_names
from colosseum.analysis.utils import get_logs_data, add_time_exceed_sign_to_plot
from colosseum.analysis.utils import get_formatted_name
from colosseum.experiment.agent_mdp_interaction import MDPLoop
from colosseum.experiment.folder_structuring import get_experiment_config
from colosseum.experiment.folder_structuring import get_mdp_agent_gin_configs
from colosseum.experiment.utils import apply_gin_config
from colosseum.hardness.analysis import compute_hardness_measure
from colosseum.utils import ensure_folder
from colosseum.utils.formatter import clear_agent_mdp_class_name

if TYPE_CHECKING:
    from colosseum.mdp import BaseMDP
    from matplotlib.figure import Figure

sns.set_theme()


def _get_index(x):
    return clear_agent_mdp_class_name(x[0].__name__), x[1]


def agent_performances_per_mdp_plot(
    experiment_folder: str,
    indicator: str,
    figsize_scale: int = 8,
    standard_error: bool = False,
    color_palette: List[str] = matplotlib.colors.TABLEAU_COLORS.keys(),
    n_rows=None,
    savefig_folder=None,
    baselines=MDPLoop.get_baselines(),
) -> "Figure":
    """
    produces a plot in which the performance indicator of the agents is shown for each MDP for the given experiment
    results.

    Parameters
    ----------
    experiment_folder : str
        The folder that contains the experiment logs, MDP configurations and agent configurations.
    indicator : str
        The code name of the performance indicator that will be shown in the plot. Check `MDPLoop.get_indicators()` to
        get a list of the available indicators.
    figsize_scale : int
        The scale for size of the figure in the resulting plot. The default value is 8.
    standard_error : bool
        If True standard errors are computed instead of the bootstrapping estimates in seaborn.
    color_palette : List[str]
        The colors to be assigned to the agents. By default, the tableau colors are used.
    n_rows : int
        The number of rows for the grid of plots. By default, it is computed to create a square grid.
    savefig_folder : str
        The folder where the figure will be saved. By default, the figure it is not saved.
    baselines : List[str]
        The baselines to be included in the plot. Check `MDPLoop.get_baselines()` to get a list of the available
        baselines. By default, all baselines are shown.

    Returns
    -------
    Figure
        The matplotlib figure.
    """

    # Check the inputs
    assert (
        indicator in MDPLoop.get_indicators()
    ), f"Please check that the indicator given in input is one from {MDPLoop.get_indicators()}."
    assert all(
        b in MDPLoop.get_baselines() for b in baselines
    ), f"Please check that the baselines given in input are available."

    # Retrieve the MDPs and agents configurations from the experiment folder
    available_mdps, available_agents = get_available_mdps_agents_prms_and_names(
        experiment_folder
    )

    # Variables for the plots
    colors_dict_agents = dict(zip(available_agents, color_palette))
    n_plots = len(available_mdps)
    h = int(np.ceil(n_plots ** 0.5)) if n_rows is None else n_rows
    w = int(np.ceil(n_plots / h))
    fig, axes = plt.subplots(
        h,
        w,
        figsize=(w * figsize_scale, h * figsize_scale),
        sharex=True,
        # If the indicator is normalized we can also share the indicator axis
        sharey="normaliz" in indicator,
    )
    if config.VERBOSE_LEVEL != 0:
        available_mdps = tqdm(
            sorted(available_mdps, key=lambda x: "".join(x)),
            desc="Plotting the results",
        )
    else:
        available_mdps = sorted(available_mdps, key=lambda x: "".join(x))

    for i, available_mdp in enumerate(available_mdps):
        ax = axes.ravel()[i]
        mdp_formatted_name = get_formatted_name(*available_mdp)
        group_by_mdp_individual_plot(
            experiment_folder,
            ax,
            indicator,
            *available_mdp,
            available_agents,
            colors_dict_agents,
            standard_error=standard_error,
            baselines=baselines,
        )
        ax.set_title(mdp_formatted_name)
        ax.legend()
        ax.ticklabel_format(style="sci", scilimits=(0, 4))

    # Remove unused axes
    for j in range(i + 1, len(axes.ravel())):
        fig.delaxes(axes.ravel()[j])

    # Last touches
    plt.ticklabel_format(style="sci", scilimits=(0, 4))
    plt.tight_layout()

    if savefig_folder is not None:
        os.makedirs(savefig_folder, exist_ok=True)
        exp_name = os.path.basename(os.path.dirname(ensure_folder(experiment_folder)))
        plt.savefig(
            f"{ensure_folder(savefig_folder)}{indicator}-for-{exp_name}.pdf",
            bbox_inches="tight",
        )

    plt.show()

    return fig


def get_hardness_measures_from_experiment_folder(
    experiment_folder: str,
    hardness_measures: Iterable[str] = ("diameter", "value_norm", "suboptimal_gaps"),
    reduce_seed: Callable[[List[float]], float] = np.mean,
) -> Dict[Tuple[Type["BaseMDP"], str], Dict[str, float]]:
    """
    retrieves the given measures of hardness for each mdp and mdp gin config in the experiment folder.

    Parameters
    ----------
    experiment_folder : str
        The folder that contains the experiment logs, MDP configurations and agent configurations.
    hardness_measures : Iterable[str]
        The list containing the measures of hardness to compute.
    reduce_seed : Callable[[List[float]], float], optional
        The function that reduces the values of the measures for different seed to a single scalar. By default, the
        mean function is employed.

    Returns
    -------
    Dict[Tuple[Type["BaseMDP"], str], Dict[str, float]]
        The dictionary that assigns to each MDP class and gin config index the corresponding dictionary containing the
        hardness measures names and values.
    """

    # Retrieve the gin configurations of the agents and MDPs
    (
        mdp_classes_scopes,
        agent_classes_scopes,
        gin_config_files_paths,
    ) = get_mdp_agent_gin_configs(experiment_folder)

    # Retrieve the number of seeds
    n_seeds = get_experiment_config(experiment_folder).n_seeds

    res = dict()
    for mdp_class, mdp_scopes in tqdm(
        mdp_classes_scopes.items(), desc=os.path.basename(experiment_folder)
    ):
        for mdp_scope in mdp_scopes:
            apply_gin_config(gin_config_files_paths)
            with gin.config_scope(mdp_scope):
                res[mdp_class, mdp_scope] = {
                    hm: reduce_seed(
                        [
                            compute_hardness_measure(mdp_class, dict(seed=seed), hm)
                            for seed in range(n_seeds)
                        ]
                    )
                    for hm in hardness_measures
                }
    return res


def plot_labels_on_benchmarks_hardness_space(
    experiment_folder: str,
    text_f: Callable[[Tuple[str, str]], str],
    color_f: Callable[[Tuple[str, str]], Union[str, None]] = lambda x: None,
    label_f: Callable[[Tuple[str, str]], Union[str, None]] = lambda x: None,
    ax: plt.Axes = None,
    multiplicative_factor_xlim=1.0,
    multiplicative_factor_ylim=1.0,
    legend_ncol=1,
    underneath_x_label: str = None,
    set_ylabel=True,
    set_legend=True,
    xaxis_measure: Union[str, Tuple[str, Callable[["BaseMDP"], float]]] = "diameter",
    yaxis_measure: Union[str, Tuple[str, Callable[["BaseMDP"], float]]] = "value_norm",
    fontsize: int = 22,
    fontsize_xlabel_underneath: int = 32,
    text_label_fontsize=16,
):
    """
    for each agent configuration in the experiment folder, it produces a plot such that it is possible to place a text
    label in the position corresponding to the value of the x-axis measure and indicator-axis measure. In addition to the text,
    it is also possible choose the color assigned to the point in such position.

    Parameters
    ----------
    experiment_folder : str
        The folder that contains the experiment logs, MDP configurations and agent configurations.
    text_f : Callable[[Tuple[str, str]], str]
        The function that returns a text label for a given MDP class name and gin config index. For example,
        ('DeepSeaEpisodic', 'prms_0') -> "DeepSeaEpisodic (0)".
    color_f : Callable[[Tuple[str, str]], str]
        The function that returns the color for the point in the position corresponding to a given MDP class name and
        gin config index. For example, ('DeepSeaEpisodic', 'prms_0') -> "DeepSeaEpisodic (0)". By default, no particular
        color is specified.
    label_f : Callable[[Tuple[str, str]], str]
        The function that returns the label to be put in the legend for the point in the position corresponding to a
        given MDP class name and gin config index. For example, ('DeepSeaEpisodic', 'prms_0') -> "DeepSea family".
        By default, the legend is not included in the plot.
    ax : plt.Axes
        The ax object where the plot will be put. By default, a new axis is created.
    multiplicative_factor_xlim : float
        The additional space to add on the right side of the figure. It can be useful to add space for the legend. By
        default, it is set to one.
    multiplicative_factor_ylim : float
        The additional space to add on the top side of the figure. It can be useful to add space for the legend. By
        default, it is set to one.
    legend_ncol : int
        The number of columns in the legend. By default, it is set to one.
    underneath_x_label : str
        Text to be added underneath the x_label. By default, no text is added.
    set_ylabel : bool
        If True, the indicator-label is set to the name of the indicator-axis measure. By default, the indicator-label is set.
    set_legend : bool
        If True, the legend is set. By default, the legend is set.
    xaxis_measure : str
        The code name of the hardness measures available in the package. Check `BaseMDP.get_available_hardness_measures()`
        to get to know the available ones. By default, it is set to the diameter.
    yaxis_measure : str
        The code name of the hardness measures available in the package. Check `BaseMDP.get_available_hardness_measures()`.
        to get to know the available ones. By default, it is set to the value norm.
    fontsize : int
        The font size for x and indicator labels. By default, it is set to :math:`22`.
    fontsize_xlabel_underneath :
        The font size for the text below the x label. By default, it is set to :math:`32`.
    text_label_fontsize : int
        The font size for the text labels of the points. By default, it is set to :math:`16`.
    """

    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        set_ylabel = True

    hardness_measures = get_hardness_measures_from_experiment_folder(
        experiment_folder, (xaxis_measure, yaxis_measure)
    )
    texts = []
    for k, r in hardness_measures.items():
        texts.append(
            ax.text(
                r[xaxis_measure],
                r[yaxis_measure],
                text_f(k),
                fontdict=dict(fontsize=text_label_fontsize),
            )
        )
        ax.scatter(
            r[xaxis_measure],
            r[yaxis_measure],
            500,
            color=color_f(k),
            label=label_f(k),
            edgecolor="black",
            linewidths=0.5,
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
            fontdict=dict(fontsize=fontsize_xlabel_underneath),
            ha="center",
        )

    if set_legend:
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(ncol=legend_ncol)

    plt.tight_layout()
    adjust_text(
        texts,
        ax=ax,
        expand_text=(1.05, 1.8),
        expand_points=(1.05, 1.5),
        only_move={"points": "indicator", "text": "xy"},
        precision=0.0001,
        lim=1000,
    )

    if show:
        plt.tight_layout()
        plt.show()


def plot_indicator_in_hardness_space(
    experiment_folder: str,
    indicator: str = "normalized_cumulative_regret",
    fontsize: int = 22,
    cmap: str = "Reds",
    fig_size=8,
    text_label_fontsize=14,
    savefig_folder: Optional[str] = "tmp",
) -> "Figure":
    """
    for each agent config, it produces a plot that places the given indicator obtained by the agent config for each MDP
    config in the position corresponding to the diameter and value norm of the MDP.

    Parameters
    ----------
    experiment_folder : str
        The path of the directory containing the experiment results.
    indicator : str
        is a string representing the performance indicator that is shown in the plot. Check `MDPLoop.get_indicators()`
        to get a list of the available indicators. By default, the 'normalized_cumulative_regret' is used.
    fontsize : int
        The font size for x and indicator labels. By default, it is set to :math:`22`.
    cmap : str
        The code name for the color map to be used when plotting the indicator values. By default,
        the 'Reds' color map is used.
    fig_size : int
        The size of the figures in the grid of plots. By default, it is set to :math:`8`.
    text_label_fontsize : int
        The font size for the text labels of the points. By default, it is set to :math:`14`.
    savefig_folder : str
        The folder where the figure will be saved. By default, the figure it is saved in a local folder with name 'tmp'.
        If the directory does not exist, it is created.

    Returns
    -------
    Figure
        The matplotlib figure.
    """

    color_map = matplotlib.cm.get_cmap(cmap)
    _, df = get_latex_table_of_average_indicator(
        experiment_folder,
        indicator,
        show_prm=True,
        return_table=True,
        mdps_on_row=False,
    )
    df_numerical = df.applymap(lambda s: float(re.findall("\d+\.\d+", s)[0]))
    fig, axes = plt.subplots(
        1, len(df.index), figsize=(len(df.index) * fig_size + 1, fig_size), sharey=True
    )
    if len(df.index) == 1:
        axes = np.array([axes])
    for i, (a, ax) in enumerate(zip(df.index, axes.tolist())):
        plot_labels_on_benchmarks_hardness_space(
            experiment_folder,
            label_f=lambda x: None,
            color_f=lambda x: color_map(
                df_numerical.loc[a, _get_index(x)] / df_numerical.loc[a].max()
            ),
            text_f=lambda x: f"{_get_index(x)[0].replace('MiniGrid', 'MG-')} "
            f"({(_get_index(x)[1].replace('prms_', ''))})",
            # text_f=lambda x: "",
            ax=ax,
            fontsize=fontsize,
            text_label_fontsize=text_label_fontsize,
            underneath_x_label=f"({ascii_lowercase[i]}) {a[0]}",
        )
        # ax.set_title(
        #     f"({ascii_lowercase[i]}) {a[0]}",
        #     fontdict=dict(legend_fontsize=legend_fontsize + 4),
        #     indicator=-0.28,
        # )

    plt.tight_layout()

    if savefig_folder is not None:
        os.makedirs(savefig_folder, exist_ok=True)
        exp_name = os.path.basename(os.path.dirname(ensure_folder(experiment_folder)))
        plt.savefig(
            f"{ensure_folder(savefig_folder)}{indicator}_in_hard_space_{exp_name}.pdf",
            bbox_inches="tight",
        )
    plt.show()

    return fig


def group_by_mdp_individual_plot(
    experiment_folder: str,
    ax,
    measure: str,
    mdp_class_name: str,
    mdp_prms: str,
    available_agents: List[Tuple[str, str]],
    colors_dict_agents: Dict[Tuple[str, str], str],
    standard_error: bool = False,
    baselines=MDPLoop.get_baselines(),
):
    """
    plots the measure for the given agents and experiment fold in the given axes.

    Parameters
    ----------
    experiment_folder : str
        is the folder that contains the experiment logs, MDP configurations and agent configurations.
    ax : plt.Axes
        is where the plot will be shown.
    measure : str
        is a string representing the performance measure that is shown in the plot. Check
        MDPLoop.get_indicators() to get a list of the available indicators.
    mdp_prms : str
        is a string that contains the mdp parameter gin config parameter, i.e. 'prms_0'.
    mdp_class_name : str
        is a string that contains the mdp class name.
    available_agents : List[Tuple[str, str]]
        is a list containing the agent gin config parameters and the agent class names.
    colors_dict_agents : Dict[Tuple[str, str], str]
        is a dict that assign to each agent gin config parameter and agent class name a different color.
    """
    mdp_code = mdp_prms + config.EXPERIMENT_SEPARATOR_PRMS + mdp_class_name

    for available_agent in available_agents:
        agent_code = (
            available_agent[1] + config.EXPERIMENT_SEPARATOR_PRMS + available_agent[0]
        )
        agent_formatted_name = get_formatted_name(*available_agent)
        df, n_seeds = get_logs_data(
            experiment_folder, mdp_class_name, mdp_prms, *available_agent
        )

        for b in baselines:
            y = measure.replace("cumulative_reward", "cumulative_expected_reward")
            if b + "_" + y in MDPLoop.get_baseline_indicators():
                sns.lineplot(
                    x="steps",
                    y= b + "_" + y,
                    label= b.capitalize() + " agent",
                    data=df,
                    ax=ax,
                    errorbar=None,
                    color=MDPLoop.get_baselines_color_dict()[b],
                    linestyle=MDPLoop.get_baselines_style_dict()[b],
                    linewidth=2,
                )

        # We plot the baselines only once
        baselines = []

        add_time_exceed_sign_to_plot(
            ax,
            df,
            colors_dict_agents[available_agent],
            measure,
            n_seeds,
            experiment_folder,
            mdp_code,
            agent_code,
        )
        sns_ax = sns.lineplot(
            x="steps",
            y=measure,
            label=agent_formatted_name,
            data=df,
            ax=ax,
            errorbar="se" if standard_error else ("ci", 95),
            color=colors_dict_agents[available_agent],
        )
        sns_ax.set_ylabel(" ".join(map(lambda x: x.capitalize(), measure.split("_"))))
