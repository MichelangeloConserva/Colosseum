import os
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns

from colosseum import config


def group_by_mdp_individual_plot(
    experiment_result_folder_path: str,
    ax,
    measure: str,
    mdp_class_name: str,
    mdp_prms: str,
    available_agents: List[Tuple[str, str]],
    colors_dict_agents: Dict[Tuple[str, str], str],
    add_random : bool = False
):
    """
    plots the measure for the given agents and experiment fold in the given axes.

    Parameters
    ----------
    experiment_result_folder_path : str
        is the folder that contains the experiment logs, MDP configurations and agent configurations.
    ax : plt.Axes
        is where the plot will be shown.
    measure : str
        is a string representing the performance measure that is shown in the plot. Check
        MDPLoop.get_available_indicators() to get a list of the available indicators.
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
            available_agent[0] + config.EXPERIMENT_SEPARATOR_PRMS + available_agent[1]
        )
        agent_formatted_name = get_formatted_name(*available_agent)
        df, n_seeds = get_logs_data(
            experiment_result_folder_path, mdp_class_name, mdp_prms, *available_agent
        )
        _add_time_exceed_sign_to_plot(
            ax,
            df,
            colors_dict_agents[available_agent],
            measure,
            n_seeds,
            experiment_result_folder_path,
            mdp_code,
            agent_code,
        )
        sns.lineplot(
            x="steps",
            y=measure,
            label=agent_formatted_name,
            data=df,
            ax=ax,
        )
    if add_random:
        sns.lineplot(
            x="steps",
            y="random_" + measure,
            label="Random agent",
            data=df,
            ax=ax,
        )


def get_formatted_name(mdp_or_agent_class_name: str, mdp_or_agent_prms: str) -> str:
    """
    returns a nicely formatted name for the MDP and the agent specification in input.
    """
    return (
        " (".join((mdp_or_agent_class_name, mdp_or_agent_prms)).replace("prms_", "")
        + ")"
    )


def format_indicator_name(indicator: str) -> str:
    return indicator.replace("_", " ").replace("normalized", "norm.").capitalize()


def get_available_mdps_agents_prms_and_names(
    experiment_result_folder_path: str,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    returns the tuples of gin config parameters and MDPs/agents class names in the given experiment folder.
    """
    logs_folders = os.listdir(f"{experiment_result_folder_path}{os.sep}logs{os.sep}")
    available_mdps, available_agents = set(), set()
    for logs_folder in logs_folders:
        mdp_code, agent_code = logs_folder.split(config.EXPERIMENT_SEPARATOR_MDP_AGENT)
        available_mdps.add(
            tuple(mdp_code.split(config.EXPERIMENT_SEPARATOR_PRMS)[::-1])
        )
        available_agents.add(
            tuple(agent_code.split(config.EXPERIMENT_SEPARATOR_PRMS)[::-1])
        )
    return sorted(available_mdps, key=lambda x: "".join(x)), sorted(
        available_agents, key=lambda x: "".join(x)
    )


def get_logs_data(
    exp_path: str,
    mdp_class_name: str,
    mdp_prms: str,
    agent_class_name: str,
    agent_prms: str,
) -> Tuple[pd.DataFrame, int]:
    mdp_code = mdp_prms + config.EXPERIMENT_SEPARATOR_PRMS + mdp_class_name
    agent_code = agent_prms + config.EXPERIMENT_SEPARATOR_PRMS + agent_class_name
    log_seed_files = glob(
        f"{exp_path}{os.sep}logs{os.sep}{mdp_code}*{agent_code}{os.sep}*.csv"
    )
    assert len(log_seed_files), f"No logs files found for {mdp_code}___{agent_code}"
    dfs = []
    for log_seed_file in log_seed_files:
        seed = int(log_seed_file[-10])
        df = pd.read_csv(log_seed_file)
        df.loc[:, "seed"] = seed
        df.loc[-1] = {
            c: seed
            if c == "seed"
            else (df.loc[0, c] if c == "steps_per_second" else 0.0)
            for c in df.columns
        }
        dfs.append(df.sort_index())
    return pd.concat(dfs).reset_index(drop=True), len(log_seed_files)


def get_n_failed_interactions(
    experiment_result_folder_path: str,
    mdp_class_name: str,
    mdp_prms: str,
    agent_class_name: str,
    agent_prms: str,
) -> int:
    mdp_code = mdp_prms + config.EXPERIMENT_SEPARATOR_PRMS + mdp_class_name
    agent_code = agent_prms + config.EXPERIMENT_SEPARATOR_PRMS + agent_class_name
    time_exceeded_file = (
        f"{experiment_result_folder_path}{os.sep}logs{os.sep}{mdp_code}"
        f"{config.EXPERIMENT_SEPARATOR_MDP_AGENT}{agent_code}{os.sep}time_exceeded.txt"
    )
    if os.path.isfile(time_exceeded_file):
        with open(time_exceeded_file, "r") as f:
            failed = set(f.readlines())
        return len(failed)
    return 0


def _add_time_exceed_sign_to_plot(
    ax,
    df: pd.DataFrame,
    color: str,
    measure: str,
    n_seeds: int,
    experiment_result_folder_path,
    prm_mdp: str,
    prm_agent: str,
):
    time_exceeded_file = (
        f"{experiment_result_folder_path}{os.sep}logs{os.sep}{prm_mdp}"
        f"{config.EXPERIMENT_SEPARATOR_MDP_AGENT}{prm_agent}{os.sep}time_exceeded.txt"
    )
    if os.path.isfile(time_exceeded_file):
        with open(time_exceeded_file, "r") as f:
            failed = set(f.readlines())

        # Getting the average time step at which the training stopped for the different seeds
        mean_time_step = 0
        for fail in failed:
            time_step = int(fail[fail.find("(") + 1 : fail.find(")")])
            mean_time_step += time_step / n_seeds

        # Plot the stopped training symbol on the plot for the given measure
        mean_time_step = df.loc[:, "steps"].tolist()[
            np.argmin(np.abs(df.loc[:, "steps"] - mean_time_step))
        ]
        ax.text(
            mean_time_step,
            df[df.loc[:, "steps"] == mean_time_step].loc[:, measure].mean(),
            "}",
            fontdict=dict(size=27),
            verticalalignment="center",
            color=color,
        )
