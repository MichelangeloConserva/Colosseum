import os
from glob import glob
from typing import TYPE_CHECKING, Dict, List, Optional

import yaml

from colosseum import config
from colosseum.experiment.experiment import ExperimentConfig
from colosseum.utils import ensure_folder
from colosseum.utils.miscellanea import (
    get_colosseum_agent_classes,
    get_colosseum_mdp_classes,
)

if TYPE_CHECKING:
    pass


def apply_gin_config(gin_config_files_paths: List[str]):
    """
    binds the gin files configuration to the corresponding objects.
    """

    import gin

    gin.clear_config()

    get_colosseum_mdp_classes()
    get_colosseum_agent_classes()

    for config_file in gin_config_files_paths:
        gin.parse_config_file(config_file)


def check_same_experiment(experiment_folder: str, result_folder: str):
    """
    checks whether all the files from the experiment folder perfectly match the configuration files in the result folder.
    """
    same_experiment = True
    for f in glob(ensure_folder(experiment_folder) + "**", recursive=True):
        if os.path.isfile(f):
            if os.path.isfile(f.replace(experiment_folder, result_folder)):
                with open(f, "r") as fr:
                    exp_fold_file = fr.read()
                with open(f.replace(experiment_folder, result_folder), "r") as fr:
                    res_fold_file = fr.read()
                if exp_fold_file.replace("\n", "") != res_fold_file.replace("\n", ""):
                    same_experiment = False
                    break
            else:
                same_experiment = False
                break
        if os.path.isdir(f):
            if not os.path.isdir(f.replace(experiment_folder, result_folder)):
                same_experiment = False
                break
    return same_experiment


def check_experiment_folders_formatting(
    cwd: str, experiment_configs: Optional[Dict[str, ExperimentConfig]]
):
    """
    checks whether there are problems in the structure of the experiments folders.
    """
    experiment_folders = glob(
        ensure_folder(cwd) + config.get_experiment_to_run_folder() + "**"
    )
    assert (
        len(experiment_folders) > 0
    ), f"No experiment folder found in {config.get_experiment_to_run_folder()}."

    for experiment_folder in experiment_folders:
        assert os.path.isdir(
            experiment_folder
        ), f"The file {experiment_folder} is not a directory, please remove it."

        assert "agent_configs" in os.listdir(
            experiment_folder
        ), f"The experiment folder {experiment_folder} is missing the agent_configs folder."
        assert "mdp_configs" in os.listdir(
            experiment_folder
        ), f"The experiment folder {experiment_folder} is missing the mdp_configs folder."
        b_config_file = "experiment_config.yml" in os.listdir(experiment_folder)

        experiment_name = experiment_folder[experiment_folder.rfind(os.sep) + 1 :]
        assert (b_config_file and not experiment_configs) or (
            not b_config_file and experiment_name in experiment_configs
        ), "The experiment configuration should be either provided as a .yml file or as an ExperimentConfig object."
        if b_config_file:
            config_file = ensure_folder(experiment_folder) + "experiment_config.yml"
            with open(config_file, "r") as f:
                experiment_config = yaml.load(f, yaml.Loader)
            assert (
                "remove_experiment_configuration_folder" in experiment_config.keys()
            ), f"remove_experiment_configuration_folder missing from {config_file}."
            assert (
                "overwrite_previous_experiment" in experiment_config.keys()
            ), f"overwrite_previous_experiment missing from {config_file}."
            assert (
                "n_seeds" in experiment_config.keys()
            ), f"n_seeds missing from {config_file}."
            assert (
                "n_steps" in experiment_config.keys()
            ), f"n_steps missing from {config_file}."
            assert (
                "max_agent_mdp_interaction_time_s" in experiment_config.keys()
            ), f"max_agent_mdp_interaction_time_s missing from {config_file}."
            assert (
                "log_performance_indicators_every" in experiment_config.keys()
            ), f"log_performance_indicators_every missing from {config_file}."
        else:
            assert (
                type(experiment_configs[experiment_name]) == ExperimentConfig
            ), "The experiment configuration should be given as an ExperimentConfig object."
