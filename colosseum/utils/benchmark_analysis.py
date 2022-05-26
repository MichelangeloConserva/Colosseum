import os
from copy import deepcopy
from glob import glob
from typing import Callable, Dict, List, Tuple, Type, Union

import gin
import numpy as np
import yaml
from adjustText import adjust_text
from matplotlib import pyplot as plt
from tqdm import tqdm

from colosseum.experiments.experiment import Experiment
from colosseum.experiments.utils import apply_gin_config
from colosseum.mdps import ContinuousMDP, EpisodicMDP
from colosseum.utils.miscellanea import (
    get_mdp_class_from_name,
    get_n_seeds_of_experiment, ensure_folder,
)


def get_average_measures_of_hardness(
    prms: str,
    mdp_class: Union[Type[EpisodicMDP], Type[ContinuousMDP], str],
    n_seeds: int,
    force_retrivial_from_file: bool,
    gin_files: List[str],
    hardness_measures: Union[List[Union[Tuple[str, Callable], str]]] = None,
) -> Dict[str, float]:
    """
    Given a parameter configuration, e.g. 'prms_0' and an MDP class (or name), it returns a dictionary containing the
    average measures of hardness given in input or the default ones for.

    Parameters
    ----------
    prms : str.
        The string identifying the configuration of the MDP, e.g. 'prms_0'.
    mdp_class : Union[Type[EpisodicMDP],Type[ContinuousMDP], str].
        The MDP class (or name) for which we want to calculate the average measure of hardness.
    n_seeds : int.
        The number of seed to calculate the average.
    force_retrivial_from_file : bool.
        Whether to force the retrivial of the measures of hardness from cached files.
    gin_files : List[str].
        The list containing all the paths to the Gin configuration files corresponding to the MDP class.
    hardness_measures : Union[List[Union[Tuple[str, Callable], str]]], optional.
        A list containing the measures of hardness to be computed. If given as a string, it should be one of those
        implemented in Colosseum. Otherwise it can be given as a tuple with the first element being the name of the
        measure and the second element being a function that takes in input an MDP and returns a value.

    Returns
    -------
        A dictionary with the name of the measure of hardness as the key and its average value as the value.
    """
    if type(mdp_class) == str:
        mdp_class = get_mdp_class_from_name(mdp_class)

    apply_gin_config(gin_files)
    with gin.config_scope(prms):
        hardness_dict = dict()
        for seed in range(n_seeds):
            mdp = mdp_class(seed=seed, verbose=True)
            if force_retrivial_from_file:
                assert (
                    mdp.hardness_report
                ), f"The hardness report for {mdp_class.__name__} {prms} is absent."

            for m in hardness_measures:
                if type(m) == str:
                    v = mdp.get_measure_from_name(m)
                elif type(m) == tuple:
                    m, f = m
                    v = f(mdp)
                else:
                    raise ValueError(
                        f"{type(m)} is invalid."
                        f"The valid type are 'Tuple[str, Callable]' and 'str'"
                    )
                if m not in hardness_dict:
                    hardness_dict[m] = 0
                hardness_dict[m] += v / n_seeds
    return hardness_dict


def retrieve_hardness_report_of_mdp_in_benchmark(
    exp_fold: str,
    hardness_measures: List[Union[str, Tuple[str, Callable]]] = (
        "diameter",
        "value_norm",
        "suboptimal_gaps",
    ),
    verbose=True,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    exp_fold = ensure_folder(exp_fold)

    hardness_measures_dict = dict()
    if os.path.exists(exp_fold + "cached_measures.yml"):
        with open(exp_fold + "cached_measures.yml", "r") as f:
            hardness_measures_dict = yaml.load(f, yaml.Loader)

    gin_files = glob(exp_fold + "mdp_configs" + os.sep + "*")

    n_seeds = get_n_seeds_of_experiment(exp_fold)

    prms_mdps = [
        (
            x[: x.find(Experiment.SEPARATOR_PRMS)],
            x[
                x.find(Experiment.SEPARATOR_PRMS)
                + len(Experiment.SEPARATOR_PRMS) : x.find(
                    Experiment.SEPARATOR_MDP_AGENT
                )
            ],
        )
        for x in os.listdir(exp_fold + "logs" + os.sep)
    ]

    for prms, mdp_name in tqdm(prms_mdps) if verbose else prms_mdps:

        hardness_measures_to_calculate = deepcopy(hardness_measures)
        if (mdp_name, prms) in hardness_measures_dict:
            for m in hardness_measures:
                # Removing the measures of hardness that are cached
                if type(m) == str and m in hardness_measures_dict[mdp_name, prms]:
                    hardness_measures_to_calculate.remove(m)
        else:
            hardness_measures_dict[mdp_name, prms] = dict()
        if len(hardness_measures_to_calculate) == 0:
            continue

        hardness_measures_dict[mdp_name, prms].update(
            get_average_measures_of_hardness(
                prms,
                get_mdp_class_from_name(mdp_name),
                n_seeds,
                False,
                gin_files,
                hardness_measures_to_calculate,
            )
        )

    return hardness_measures_dict


def plot_labels_on_benchmarks_hardness_space(
    exp_to_show,
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
    xaxis_measure: Union[
        str, Tuple[str, Callable[[Union[EpisodicMDP, ContinuousMDP]], float]]
    ] = "diameter",
    yaxis_measure: Union[
        str, Tuple[str, Callable[[Union[EpisodicMDP, ContinuousMDP]], float]]
    ] = "value_norm",
):
    """
    Locates the value obtain from 'label_f' in the location corresponding to xaxis_measure and yaxis_measure.

    Parameters
    ----------
    exp_to_show : str.
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
        The hardness measure for the x axis. If given as a string, it should be one of those implemented in Colosseum.
        Otherwise it can be given as a tuple with the first element being the name of the measure and the second element
        being a function that takes in input an MDP and returns a value. The default value is the environmental value norm.
    """
    assert "mdp_configs" in os.listdir(
        exp_to_show
    ), f"The directory folder {exp_to_show} does not contain 'mdp_configs' folders."
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        set_ylabel = True

    hardness_reports = retrieve_hardness_report_of_mdp_in_benchmark(
        exp_to_show, [xaxis_measure, yaxis_measure]
    )

    if type(xaxis_measure) == tuple:
        xaxis_measure = xaxis_measure[0]
    if type(yaxis_measure) == tuple:
        yaxis_measure = yaxis_measure[0]

    texts = []
    for k, r in hardness_reports.items():
        texts.append(
            ax.text(
                r[xaxis_measure],
                r[yaxis_measure],
                text_f(k),
                fontdict=dict(fontsize=22),
            )
        )
        ax.scatter(
            r[xaxis_measure], r[yaxis_measure], 100, color=color_f(k), label=label_f(k)
        )

    ax.tick_params(labelsize=22)
    if set_ylabel:
        ax.set_ylabel(
            yaxis_measure.capitalize().replace("_", " "),
            fontdict=dict(fontsize=22),
            labelpad=10,
        )
    ax.set_xlabel(
        xaxis_measure.capitalize().replace("_", " "),
        fontdict=dict(fontsize=22),
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
            fontdict=dict(fontsize=32),
            ha="center",
        )
    adjust_text(texts)
    if set_legend:
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(ncol=legend_ncol)
    if show:
        plt.tight_layout()
        plt.show()
