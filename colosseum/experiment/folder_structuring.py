import os
import re
import shutil
from glob import glob
from tempfile import gettempdir
from typing import Dict, List, Optional, Set, Tuple, Type

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from colosseum import config
from colosseum.agent.agents.base import BaseAgent
from colosseum.emission_maps import get_emission_map_from_name
from colosseum.experiment import ExperimentConfig
from colosseum.experiment.experiment_instance import ExperimentInstance
from colosseum.mdp import BaseMDP
from colosseum.utils import ensure_folder
from colosseum.utils.miscellanea import get_agent_class_from_name
from colosseum.utils.miscellanea import get_mdp_class_from_name


def get_mdp_agent_gin_configs(
    experiment_folder: str,
) -> Tuple[
    Dict[Type["BaseMDP"], Set[str]], Dict[Type["BaseAgent"], Set[str]], List[str]
]:
    """
    Returns
    -------
    Dict[Type["BaseMDP"], Set[str]]
        The dictionary that associated to each MDP class the set of gin configuration indices found in the experiment
        folder.
    Dict[Type["BaseAgent"], Set[str]]
        The dictionary that associated to each agent class the set of gin configuration indices found in the experiment
        folder.
    List[str]
        The gin configuration file paths found in the experiment_folder.
    """
    gin_config_files_paths = []

    mdp_classes_scopes = dict()
    for mdp_config_file in glob(
        f"{ensure_folder(experiment_folder)}mdp_configs{os.sep}*"
    ):
        with open(mdp_config_file, "r") as f:
            f_ = f.read()
        mdp_scopes = set(re.findall(r"prms_\d+", f_))
        mdp_class_name = re.findall(r"prms_\d+/(.*?)\.", f_)[0]
        mdp_class = get_mdp_class_from_name(mdp_class_name)
        mdp_classes_scopes[mdp_class] = mdp_scopes
        gin_config_files_paths.append(mdp_config_file)

    agent_classes_scopes = dict()
    for agent_config_file in glob(
        f"{ensure_folder(experiment_folder)}agents_configs{os.sep}*"
    ):
        with open(agent_config_file, "r") as f:
            f_ = f.read()
        agent_scopes = set(re.findall(r"prms_\d+", f_))
        agent_class_name = re.findall(r"prms_\d+/(.*?)\.", f_)[0]
        agent_class = get_agent_class_from_name(agent_class_name)
        agent_classes_scopes[agent_class] = agent_scopes
        gin_config_files_paths.append(agent_config_file)

    classes = list(mdp_classes_scopes.keys()) + list(agent_classes_scopes.keys())
    assert sum([c.is_episodic() for c in classes]) in [0, len(classes)], (
        f"Episodic and infinite horizon agents and/or MDP instances should not be mixed."
        f"Please check the configuration files of {experiment_folder}."
    )

    return mdp_classes_scopes, agent_classes_scopes, gin_config_files_paths


def _get_experiment_mdp_agent_couples(
    experiment_config: ExperimentConfig,
    experiment_cur_folder: str,
    mdp_classes_scopes: Dict[Type["BaseMDP"], Set[str]],
    agent_classes_scopes: Dict[Type["BaseAgent"], Set[str]],
    gin_config_files_paths: List[str],
) -> List[ExperimentInstance]:
    experiment_mdp_agent_couples = []
    for seed in range(experiment_config.n_seeds):
        for mdp_class, mdp_scopes in mdp_classes_scopes.items():
            for mdp_scope in mdp_scopes:
                for (
                    agent_class,
                    agent_scopes,
                ) in agent_classes_scopes.items():
                    for agent_scope in agent_scopes:
                        exp_inst = ExperimentInstance(
                            seed,
                            mdp_class,
                            mdp_scope,
                            agent_class,
                            agent_scope,
                            experiment_cur_folder,
                            gin_config_files_paths,
                            experiment_config,
                        )
                        if not exp_inst.does_log_file_exists:
                            experiment_mdp_agent_couples.append(exp_inst)
    return experiment_mdp_agent_couples


def get_experiment_config(experiment_folder: str) -> ExperimentConfig:
    """
    Returns
    -------
    ExperimentConfig
        The `ExperimentConfig` corresponding to the experiment folder.
    """

    config_file = ensure_folder(experiment_folder) + "experiment_config.yml"
    with open(config_file, "r") as f:
        experiment_config = yaml.load(f, yaml.Loader)
    return ExperimentConfig(
        n_seeds=experiment_config["n_seeds"],
        n_steps=experiment_config["n_steps"],
        max_interaction_time_s=experiment_config["max_interaction_time_s"],
        log_performance_indicators_every=experiment_config[
            "log_performance_indicators_every"
        ],
        emission_map=get_emission_map_from_name(
            experiment_config["emission_map"]
            if "emission_map" in experiment_config
            else "Tabular"
        ),
    )


def _clean_time_exceeded_records(log_file: str):
    """
    checks if the log file has been classified as an experiment that exceeded the time limit and, if so, it cleans the
    record.
    """
    time_exceeded_experiment_record = (
        log_file[: log_file.rfind(os.sep)] + os.sep + "time_exceeded.txt"
    )
    if os.path.exists(time_exceeded_experiment_record):
        with open(time_exceeded_experiment_record, "r") as ff:
            te = ff.readlines()
        for tee in te:
            if log_file in tee:
                te.remove(tee)
                break
        if len(te) > 0:
            with open(time_exceeded_experiment_record, "w") as ff:
                ff.write("".join(te))
        else:
            os.remove(time_exceeded_experiment_record)


def remove_corrupted_log_files(
    experiment_folder: str,
    experiment_config: ExperimentConfig = None,
) -> List[str]:
    """
    checks if there are any inconsistencies in the log files of an experiment and removes them.
    """

    assert experiment_config is not None or os.path.isfile(
        ensure_folder(experiment_folder) + "experiment_config.yml"
    )
    if not os.path.isdir(ensure_folder(experiment_folder) + "logs"):
        return

    if experiment_config is None:
        with open(ensure_folder(experiment_folder) + "experiment_config.yml", "r") as f:
            experiment_config = ExperimentConfig(**yaml.load(f, yaml.Loader))

    file_paths = glob(f"{experiment_folder}{os.sep}**{os.sep}*.csv", recursive=True)
    if config.VERBOSE_LEVEL != 0:
        file_paths = tqdm(file_paths, desc="Checking for corrupted log files")

    corrupted_files = []
    for f in file_paths:
        with open(f, "r") as ff:
            len_f = len(ff.readlines())
        logged_steps = [] if len_f <= 1 else pd.read_csv(f).steps.tolist()
        if (
            len_f <= 1
            or
            # checks whether there are any inconsistencies in the order of the logs
            any(np.diff(pd.read_csv(f).steps) < 0)
            # checks that all the steps have been logged
            or not (
                all(
                    t in logged_steps
                    for t in range(1, experiment_config.n_steps)
                    if t % experiment_config.log_performance_indicators_every == 0
                )
                and (experiment_config.n_steps - 1) in logged_steps
            )
        ):
            # If it was registered that this instance failed due to the time constrain, we remove that record since we
            # are going to run this agent/MDP interaction from scratch.
            _clean_time_exceeded_records(f)

            # Moving the file to the temporary file folder just in case we want to double-check the formatting error.
            shutil.move(
                f,
                gettempdir()
                + f"{os.sep}_{len(corrupted_files)}_"
                + f[f.rfind(os.sep) + 1 :],
            )
            corrupted_files.append(f)
            tqdm.write(
                f"The file {f} has been moved to tmp as it has some formatting errors."
            )

    if config.VERBOSE_LEVEL != 0:
        print(corrupted_files)

    return corrupted_files
