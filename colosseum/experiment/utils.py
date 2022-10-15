import os
from glob import glob
from typing import Dict, List, Optional, Type, Union

import gin
import yaml
from tqdm import tqdm

from colosseum import config
from colosseum.agent.agents.base import BaseAgent
from colosseum.agent.utils import sample_n_agent_hyperparameters
from colosseum.experiment import ExperimentConfig
from colosseum.experiment.folder_structuring import (
    get_mdp_agent_gin_configs,
    get_experiment_config,
)
from colosseum.mdp import BaseMDP
from colosseum.utils import ensure_folder
from colosseum.utils.miscellanea import get_colosseum_agent_classes
from colosseum.utils.miscellanea import get_colosseum_mdp_classes


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


def check_same_experiment(folder_1: str, folder_2: str, exclude_config: bool = False):
    """
    checks whether all the files from folder_1 perfectly match the configuration files in folder_2.
    """

    # Check same experiment configuration
    if not exclude_config:
        with open(ensure_folder(folder_1) + "experiment_config.yml", "r") as f:
            config_1 = yaml.load(f, yaml.Loader)
        with open(ensure_folder(folder_2) + "experiment_config.yml", "r") as f:
            config_2 = yaml.load(f, yaml.Loader)
        if config_1 != config_2:
            return False

    # Check same MDP configs
    mdp_configs_1 = set(os.listdir(ensure_folder(folder_1) + "mdp_configs"))
    mdp_configs_2 = set(os.listdir(ensure_folder(folder_2) + "mdp_configs"))
    if mdp_configs_1 != mdp_configs_2:
        return False
    for mdp_config in mdp_configs_1:
        with open(ensure_folder(folder_1) + "mdp_configs" + os.sep + mdp_config) as f:
            mdp_config_1 = f.read()
        with open(ensure_folder(folder_2) + "mdp_configs" + os.sep + mdp_config) as f:
            mdp_config_2 = f.read()
        if mdp_config_1 != mdp_config_2:
            return False

    # Check same agent configs
    if "agents_configs" in os.listdir(ensure_folder(folder_1)):
        agent_configs_1 = set(os.listdir(ensure_folder(folder_1) + "agents_configs"))
        agent_configs_2 = set(os.listdir(ensure_folder(folder_2) + "agents_configs"))
        if agent_configs_1 != agent_configs_2:
            return False
        for agent_config in agent_configs_1:
            with open(
                ensure_folder(folder_1) + "agents_configs" + os.sep + agent_config
            ) as f:
                agent_config_1 = f.read()
            with open(
                ensure_folder(folder_2) + "agents_configs" + os.sep + agent_config
            ) as f:
                agent_config_2 = f.read()
            if agent_config_1 != agent_config_2:
                return False

    return True


def check_experiments_to_run_folders_formatting(
    experiment_configs: Optional[Dict[str, ExperimentConfig]] = None
):
    """
    checks whether there are problems in the structure of the experiments folders.
    """
    experiment_folders = glob(config.get_experiments_folder() + "**")
    assert (
        len(experiment_folders) > 0
    ), f"No experiment folder found in {config.get_experiment_to_run_folder()}."

    for experiment_folder in experiment_folders:
        check_experiment_folder(experiment_folder, experiment_configs)


def check_experiment_folder(
    experiment_folder: str,
    experiment_config: Union[str, ExperimentConfig] = None,
):
    """
    checks the integrity of the experiment folder.
    """

    assert os.path.isdir(
        experiment_folder
    ), f"The file {experiment_folder} is not a directory, please remove it."

    assert "agents_configs" in os.listdir(
        experiment_folder
    ), f"The experiment folder {experiment_folder} is missing the agents_configs folder."
    assert "mdp_configs" in os.listdir(
        experiment_folder
    ), f"The experiment folder {experiment_folder} is missing the mdp_configs folder."
    b_config_file = "experiment_config.yml" in os.listdir(experiment_folder)

    experiment_name = experiment_folder[experiment_folder.rfind(os.sep) + 1 :]
    assert (
        experiment_config is not None or b_config_file
    ), "The experiment configuration should be either provided as a .yml file or as an ExperimentConfig object."
    if b_config_file:
        config_file = ensure_folder(experiment_folder) + "experiment_config.yml"
        with open(config_file, "r") as f:
            experiment_config = yaml.load(f, yaml.Loader)
        assert (
            "n_seeds" in experiment_config.keys()
        ), f"n_seeds missing from {config_file}."
        assert (
            "n_steps" in experiment_config.keys()
        ), f"n_steps missing from {config_file}."
        assert (
            "max_interaction_time_s" in experiment_config.keys()
        ), f"max_interaction_time_s missing from {config_file}."
        assert (
            "log_performance_indicators_every" in experiment_config.keys()
        ), f"log_performance_indicators_every missing from {config_file}."
    else:
        assert (
            type(experiment_config) == ExperimentConfig
        ), "The experiment configuration should be given as an ExperimentConfig object."

    from colosseum.experiment.folder_structuring import remove_corrupted_log_files

    remove_corrupted_log_files(experiment_folder, ExperimentConfig(**experiment_config))


def instantiate_gin_files(
    dest_folder: str,
    agent_classes: List[Type["BaseAgent"]],
    mdp_classes: List[Type["BaseMDP"]],
    n_samples_agents: int,
    n_samples_mdps: int,
    seed: int,
) -> List[str]:
    """
    produces and instantiates gin files from samples of the given MDP and Agent classes in the destination folder.
    Returns
    -------
    List[str]
        The file paths of the gin files
    """
    os.makedirs(dest_folder, exist_ok=True)

    # Store gin files for MDPs
    gin_files = []
    for mdp_class in mdp_classes:
        fp = (
            ensure_folder(dest_folder)
            + "mdp_configs"
            + os.sep
            + mdp_class.__name__
            + ".gin"
        )
        gin_files.append(fp)
        with open(fp, "w") as f:
            f.write(
                "\n\n".join(
                    mdp_class.produce_gin_file_from_mdp_parameters(
                        mdp_hyperparameters, mdp_class.__name__, i
                    )
                    for i, mdp_hyperparameters in enumerate(
                        mdp_class.sample_parameters(n_samples_mdps, seed)
                    )
                )
            )

    # Store gin file for agents
    for agent_class in agent_classes:
        fp = (
            ensure_folder(dest_folder)
            + "agents_configs"
            + os.sep
            + agent_class.__name__
            + ".gin"
        )
        gin_files.append(fp)
        with open(fp, "w") as f:
            f.write(
                "\n\n".join(
                    agent_class.produce_gin_file_from_parameters(
                        agent_hyperparameter, i
                    )
                    for i, agent_hyperparameter in enumerate(
                        sample_n_agent_hyperparameters(n_samples_agents, agent_class, seed)
                    )
                )
            )

    return gin_files


def instantiate_mdps_from_experiment_folder(
    experiment_folder: str, exclude_horizon_from_parameters=False
) -> List["BaseMDP"]:
    """
    Returns
    -------
    List["BaseMDP"]
        The MDP instances corresponding to the experiment folder in input.
    """
    (
        mdp_classes_scopes,
        agent_classes_scopes,
        gin_config_files_paths,
    ) = get_mdp_agent_gin_configs(experiment_folder)
    exp_config = get_experiment_config(experiment_folder, None)

    loop = mdp_classes_scopes.items()
    if config.VERBOSE_LEVEL != 0:
        loop = tqdm(loop, desc=os.path.basename(experiment_folder))

    mdps = []
    for mdp_class, mdp_scopes in loop:
        for mdp_scope in mdp_scopes:
            apply_gin_config(gin_config_files_paths)
            with gin.config_scope(mdp_scope):
                for seed in range(exp_config.n_seeds):
                    mdps.append(
                        mdp_class(
                            seed=seed,
                            exclude_horizon_from_parameters=exclude_horizon_from_parameters,
                        )
                    )

    return mdps
