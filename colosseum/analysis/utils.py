import os
import re
from glob import glob
from typing import List, Tuple

import numpy as np
import pandas as pd

from colosseum import config
from colosseum.utils import ensure_folder
from colosseum.utils.formatter import clear_agent_mdp_class_name


def get_formatted_name(mdp_or_agent_class_name: str, mdp_or_agent_prms: str) -> str:
    """
    provides a nice formatting for the name of an agent or MDP class and its corresponding gin config index.

    Parameters
    ----------
    mdp_or_agent_class_name : str
        The str containing the name of an agent or MDP class.
    mdp_or_agent_prms : str
        The str containing the gin config prms of an agent or MDP.

    Returns
    -------
    str
        A nicely formatted name for the MDP and the agent specification in input.
    """
    return (
        clear_agent_mdp_class_name(mdp_or_agent_class_name)
        + f" ({1 + int(re.findall('[0-9]+', mdp_or_agent_prms)[0])})"
    )


def format_indicator_name(indicator: str) -> str:
    """
    provides a nice formatting to the indicator code name in input.

    Parameters
    ----------
    indicator : str
        The indicator code name.

    Returns
    -------
    str
        A nicely formatted string.
    """
    return indicator.replace("_", " ").replace("normalized", "norm.").capitalize()


def get_available_mdps_agents_prms_and_names(
    experiment_folder: str,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    retrieves the gin configurations for MDPs and agents from the experiment folder.

    Parameters
    ----------
    experiment_folder : str
        The folder where the experiments are.

    Returns
    -------
    Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]
        The tuples of gin config parameters and MDPs/agents class names in the given experiment folder.
    """
    logs_folders = os.listdir(f"{ensure_folder(experiment_folder)}logs{os.sep}")
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
    experiment_folder: str,
    mdp_class_name: str,
    mdp_prm: str,
    agent_class_name: str,
    agent_prm: str,
) -> Tuple[pd.DataFrame, int]:
    """
    retrieves the logging data of the experiment folder for the given mdp class, mdp gin config, agent class, and agent
    gin config.

    Parameters
    ----------
    experiment_folder : str
        The folder that contains the experiment logs, MDP configurations and agent configurations.
    mdp_class_name : str
        The name of the MDP class.
    mdp_prm : str
        The gin configuration parameter of the MDP.
    agent_class_name
        The name of the agent class.
    agent_prm : str
        The gin configuration parameter of the agent.

    Returns
    -------
    pd.DataFrame
        The logging data stored as a pd.DataFrame.
    int
        The number of seeds of used in the experiment.
    """

    mdp_code = mdp_prm + config.EXPERIMENT_SEPARATOR_PRMS + mdp_class_name
    agent_code = agent_prm + config.EXPERIMENT_SEPARATOR_PRMS + agent_class_name
    log_seed_files = glob(
        f"{experiment_folder}{os.sep}logs{os.sep}{mdp_code}*{agent_code}{os.sep}*.csv"
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
    experiment_folder: str,
    mdp_class_name: str,
    mdp_prm: str,
    agent_class_name: str,
    agent_prm: str,
) -> int:
    """
    retrieves the number of times the agent config has failed to complete the total number of interaction in time in the
    given MDP config.

    Parameters
    ----------
    experiment_folder : str
        The folder that contains the experiment logs, MDP configurations and agent configurations.
    mdp_class_name : str
        The name of the MDP class.
    mdp_prm : str
        The gin configuration parameter of the MDP.
    agent_class_name
        The name of the agent class.
    agent_prm : str
        The gin configuration parameter of the agent.

    Returns
    -------
    int
        The number of time the agent broke the computational time limit.
    """

    mdp_code = mdp_prm + config.EXPERIMENT_SEPARATOR_PRMS + mdp_class_name
    agent_code = agent_prm + config.EXPERIMENT_SEPARATOR_PRMS + agent_class_name
    time_exceeded_file = (
        f"{ensure_folder(experiment_folder)}logs{os.sep}{mdp_code}"
        f"{config.EXPERIMENT_SEPARATOR_MDP_AGENT}{agent_code}{os.sep}time_exceeded.txt"
    )
    if os.path.isfile(time_exceeded_file):
        with open(time_exceeded_file, "r") as f:
            failed = set(f.readlines())
        return len(failed)
    return 0


def add_time_exceed_sign_to_plot(
    ax,
    df: pd.DataFrame,
    color: str,
    indicator: str,
    n_seeds: int,
    experiment_folder,
    mdp_prm: str,
    agent_prm: str,
):
    """
    adds the '}' symbol to the plot in the time corresponding to the average time the agent broke the computational time
    limit.

    Parameters
    ----------
    ax : plt.Axes
        The ax object where the symbol will be put.
    df : pd.DataFrame
        The logging data of the experiment for a given agent config and MDP config.
    color : str
        The code name for the color of the symbol.
    indicator : str
        is a string representing the performance indicator that is shown in the plot. Check `MDPLoop.get_indicators()`
        to get a list of the available indicators. By default, the 'normalized_cumulative_regret' is used.
    n_seeds : int
        The total number of seed used in the experiment.
    experiment_folder : str
        The path of the directory containing the experiment results.
    mdp_prm : str
        The gin configuration parameter of the MDP.
    agent_prm : str
        The gin configuration parameter of the agent.
    """

    time_exceeded_file = (
        f"{ensure_folder(experiment_folder)}logs{os.sep}{mdp_prm}"
        f"{config.EXPERIMENT_SEPARATOR_MDP_AGENT}{agent_prm}{os.sep}time_exceeded.txt"
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
            df[df.loc[:, "steps"] == mean_time_step].loc[:, indicator].mean(),
            "}",
            fontdict=dict(size=27),
            verticalalignment="center",
            color=color,
        )
