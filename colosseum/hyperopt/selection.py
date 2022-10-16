import csv
import os
from glob import glob
from typing import Dict, Type, Iterable, Callable, Collection

import numpy as np

from colosseum import config
from colosseum.agent.agents.base import BaseAgent
from colosseum.benchmark.utils import retrieve_agent_configs
from colosseum.utils.miscellanea import ensure_folder


def retrieve_best_agent_config_from_hp_folder(
    agent_classes: Iterable[Type["BaseAgent"]] = None,
    indicator="normalized_cumulative_regret",
    reduce_seeds: Callable[[Collection], float] = np.mean,
    folder: str = None,
) -> Dict[Type["BaseAgent"], str]:
    """
    retrieve the best agents configurations from a folder with the results of a hyperparameter optimization procedure.
    Note that. by default, the indicator is minimized. If you want to maximise the indicator you can pass a
    `reduce_seeds` function that inverts the sign of the indicators, e.g. `lambda x : -np.mean(x)`.

    Parameters
    ----------
    agent_classes : Iterable[Type["BaseAgent"]]
        The agent classes for which the function retrieves the best config. By default, the agent classes are retrieved
         from the hyper_opt folder.
    indicator : str
        The code name of the performance indicator that will be used in the choice of the best parameters. Check
        `MDPLoop.get_indicators()` to get a list of the available indicators. By default, the indicator is the
        'normalized_cumulative_regret'.
    reduce_seeds : Callable[[Collection], float]
        The function that reduces the values of different seeds. By default, the average is used.
    folder : str
        The folder where the parameters optimization results are stored. By default, the one configured in the
        package is used.

    Returns
    -------
    Dict[Type["BaseAgent"], str]
        A dictionary that associates to each agent class its best configuration.
    """

    if folder is None:
        folder = config.get_hyperopt_folder()
    else:
        folder = ensure_folder(folder)

    latest_hyprms_folder = folder + "latest_hyprms" + os.sep

    # Retrive the agent classes from the folder if no agent classes is given
    if agent_classes is None:
        agent_classes = []
        if os.path.isdir(folder + "hyperopt_continuous"):
            agent_classes += list(
                retrieve_agent_configs(folder + "hyperopt_continuous").keys()
            )
        if os.path.isdir(folder + "hyperopt_episodic"):
            agent_classes += list(
                retrieve_agent_configs(folder + "hyperopt_episodic").keys()
            )
        assert len(agent_classes) > 0, f"No agent classes found in the {folder}"

    agent_config = dict()
    for agent_class in agent_classes:
        current_hp_folder = (
            folder
            + "hyperopt_"
            + ("episodic" if agent_class.is_episodic() else "continuous")
            + os.sep
        )

        if os.path.isfile(latest_hyprms_folder + agent_class.__name__ + ".gin"):
            with open(latest_hyprms_folder + agent_class.__name__ + ".gin", "r") as f:
                agent_config[agent_class] = f.read()
        elif os.path.isdir(current_hp_folder):
            agent_config[agent_class] = get_best_agent_gin_config(
                agent_class, current_hp_folder, indicator, reduce_seeds
            )
        else:
            raise FileNotFoundError(
                f"The hyperoptimization folder for {agent_class.__name__} could not be found "
                f"neither in the cache ({latest_hyprms_folder}) nor in the full logs ({current_hp_folder})"
            )
    return agent_config


def get_best_agent_gin_config(
    agent_class: Type["BaseAgent"],
    hp_folder: str,
    indicator="normalized_cumulative_regret",
    reduce_seeds: Callable[[Collection], float] = np.mean,
) -> str:
    """
    retrieve the best agents configurations from the folder with the results of a hyperparameter optimization procedure
    given an indicator. Note that. by default, the indicator is minimized. If you want to maximise the indicator you
    can pass a `reduce_seeds` function that inverts the sign of the indicators, e.g. `lambda x : -np.mean(x)`.

    Parameters
    ----------
    agent_class : Type["BaseAgent"]
        The agent class for which the function retrieves the best config.
    hp_folder : str
        The folder where the results of the parameters optimization procedure are located.
    indicator : str
        The code name of the performance indicator that will be used in the choice of the best parameters. Check
        `MDPLoop.get_indicators()` to get a list of the available indicators. By default, the indicator is the
        'normalized_cumulative_regret'.
    reduce_seeds : Callable[[Collection], float]
        The function that reduces the values of different seeds. By default, the average is used.
    Returns
    -------
    str
        The gin config of the best parameters.
    """
    agents_configs = retrieve_agent_configs(hp_folder, False)[agent_class]

    prms_scores = dict()
    for prm in agents_configs:
        agent_prm_logs = glob(
            hp_folder
            + f"logs/*{prm}{config.EXPERIMENT_SEPARATOR_PRMS}{agent_class.__name__}/*.csv",
            recursive=True,
        )

        scores = []
        for l_f in agent_prm_logs:
            with open(l_f) as f:
                reader = csv.DictReader(f)
                for p in reader:
                    pass
                scores.append(float(p[indicator]))
        score = reduce_seeds(scores)
        prms_scores[prm] = score
    best_prms = min(prms_scores, key=lambda k: prms_scores[k])

    return agents_configs[best_prms]
