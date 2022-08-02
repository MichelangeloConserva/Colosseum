import os
import re
import shutil
import warnings
from glob import glob
from tempfile import gettempdir
from typing import Dict, List, Optional, Set, Tuple, Type

import numpy as np
import pandas as pd
import yaml

from colosseum import config
from colosseum.experiment.experiment import ExperimentConfig, ExperimentInstance
from colosseum.experiment.experiment.utils import check_same_experiment
from colosseum.mdp import BaseMDP
from colosseum.agent.agents.base import BaseAgent
from colosseum.utils import ensure_folder
from colosseum.utils.miscellanea import (
    get_agent_class_from_name,
    get_mdp_class_from_name,
)


def retrieve_experiment_instances(
    experiment_configs: Dict[str, ExperimentConfig] = None
) -> List[ExperimentInstance]:
    """
    returns a list of ExperimentInstance that can be processed independently. If experiment_configs are not provided, it
    is assumed that an experiment_config.yml file is provided inside the experiment configuration folders.
    """
    assert len(glob(config.get_experiment_to_run_folder() + "**")) > 0, (
        "No experiment configuration folder has been found in"
        f"{config.get_experiment_to_run_folder()}"
    )

    print("Retrieving experiments:")
    experiments_prms = []
    for experiment_folder in glob(config.get_experiment_to_run_folder() + "**"):
        experiment_name = experiment_folder[experiment_folder.rfind(os.sep) + 1 :]
        if experiment_name[0] == "_":
            continue
        experiment_config = get_experiment_config(experiment_folder, experiment_configs)
        experiment_results_folder = (
            config.get_experiment_result_folder() + experiment_name
        )

        print(f"\t- {experiment_name}")

        is_merging = _prepare_result_folder(
            experiment_folder, experiment_results_folder, experiment_config
        )

        (
            mdp_classes_scopes,
            agent_classes_scopes,
            gin_config_files_paths,
        ) = retrieve_mdp_classes_agent_classes_gin_config(
            experiment_folder, experiment_results_folder
        )
        experiments_prms += _get_experiment_mdp_agent_couples(
            experiment_config,
            experiment_results_folder,
            mdp_classes_scopes,
            agent_classes_scopes,
            gin_config_files_paths,
            is_merging,
        )

        if experiment_config.remove_experiment_configuration_folder:
            shutil.rmtree(experiment_folder)

    return experiments_prms


def retrieve_mdp_classes_agent_classes_gin_config(
    experiment_folder: str, result_folder: str = None
) -> Tuple[
    Dict[Type["BaseMDP"], Set[str]], Dict[Type["BaseAgent"], Set[str]], List[str]
]:
    """
    returns the mdp classes, the agent classes and the gin configuration file that correspond to the experiments in the
    given folder.
    """
    if result_folder is None:
        result_folder = experiment_folder

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
        gin_config_files_paths.append(
            mdp_config_file.replace(experiment_folder, result_folder)
        )

    agent_classes_scopes = dict()
    for agent_config_file in glob(
        f"{ensure_folder(experiment_folder)}agent_configs{os.sep}*"
    ):
        with open(agent_config_file, "r") as f:
            f_ = f.read()
        agent_scopes = set(re.findall(r"prms_\d+", f_))
        agent_class_name = re.findall(r"prms_\d+/(.*?)\.", f_)[0]
        agent_class = get_agent_class_from_name(agent_class_name)
        agent_classes_scopes[agent_class] = agent_scopes
        gin_config_files_paths.append(
            agent_config_file.replace(experiment_folder, result_folder)
        )

    classes = list(mdp_classes_scopes.keys()) + list(agent_classes_scopes.keys())
    assert sum([c.is_episodic() for c in classes]) in [0, len(classes)], (
        f"Episodic and infinite horizon agents and/or MDP instances should not be mixed."
        f"Please check the configuration files of {experiment_folder}."
    )

    return mdp_classes_scopes, agent_classes_scopes, gin_config_files_paths


def _get_experiment_mdp_agent_couples(
    experiment_config: ExperimentConfig,
    result_folder: str,
    mdp_classes_scopes: Dict[Type["BaseMDP"], Set[str]],
    agent_classes_scopes: Dict[Type["BaseAgent"], Set[str]],
    gin_config_files_paths: List[str],
    is_merging: bool,
) -> List[ExperimentInstance]:
    """
    returns all the ExperimentInstances that corresponds to the experiment to be run. If an ExperimentInstance has been
    previously run then it is not added to the returned list.
    """
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
                            result_folder,
                            gin_config_files_paths,
                            experiment_config,
                        )
                        if not is_merging or not exp_inst.does_log_file_exists:
                            experiment_mdp_agent_couples.append(exp_inst)
    return experiment_mdp_agent_couples


def _prepare_result_folder(
    experiment_folder, experiment_results_folder, experiment_config: ExperimentConfig
) -> bool:
    """
    creates the folder structure in the config.get_experiment_results_folder() and returns whether a merging with logs
    already available in the folder is possible.
    """
    is_merging = False
    if os.path.isdir(experiment_results_folder):
        if experiment_config.overwrite_previous_experiment:
            shutil.rmtree(experiment_results_folder)
        else:
            # We need to check whether the current experiment correspond to the previously run one
            is_same_experiment = check_same_experiment(
                experiment_folder, experiment_results_folder
            )
            if not is_same_experiment:
                raise ValueError(
                    "When overwrite_experiment is set to False and there is a previously run experiment with the same "
                    "name in the results folder, they must be the same. This allows to save experiment running progress."
                )

            # If they are the same then, it may be the case that we are resuming the benchmark evaluation so we
            # need to remove possible corrupted log files.
            _remove_corrupted_log_files(experiment_results_folder, experiment_config)
            is_merging = True
    if not is_merging:
        shutil.copytree(experiment_folder, experiment_results_folder)
    return is_merging


def get_experiment_config(
    experiment_folder: str, experiment_configs: Optional[Dict[str, ExperimentConfig]]
) -> ExperimentConfig:
    """
    returns a ExperimentConfig corresponding to the experiment folder. If it is not provided in the given dictionary, it
    is looked for in the folder as experiment_config.yml file.
    """
    experiment_name = experiment_folder[experiment_folder.rfind(os.sep) + 1 :]
    if experiment_configs and experiment_name in experiment_configs:
        return experiment_configs[experiment_name]
    config_file = ensure_folder(experiment_folder) + "experiment_config.yml"
    with open(config_file, "r") as f:
        experiment_config = yaml.load(f, yaml.Loader)
    return ExperimentConfig(
        remove_experiment_configuration_folder=experiment_config[
            "remove_experiment_configuration_folder"
        ],
        n_seeds=experiment_config["n_seeds"],
        overwrite_previous_experiment=experiment_config[
            "overwrite_previous_experiment"
        ],
        n_steps=experiment_config["n_steps"],
        max_agent_mdp_interaction_time_s=experiment_config[
            "max_agent_mdp_interaction_time_s"
        ],
        log_performance_indicator_every=experiment_config[
            "log_performance_indicators_every"
        ],
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


def _remove_corrupted_log_files(
    experiment_results_folder: str,
    experiment_config: ExperimentConfig,
):
    """
    checks if there are any inconsistencies in the log files of an experiment and removes them.
    """
    o = 0
    for f in glob(
        f"{experiment_results_folder}{os.sep}**{os.sep}*.csv", recursive=True
    ):
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
            shutil.move(f, gettempdir() + f"{os.sep}_{o}_" + f[f.rfind(os.sep) + 1 :])
            o += 1
            warnings.warn(
                f"The file {f} has been moved to tmp as it has some formatting errors."
            )
