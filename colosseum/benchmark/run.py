import os
import shutil
from typing import TYPE_CHECKING, Dict, List, Type, Union

from colosseum import config
from colosseum.benchmark import ColosseumBenchmarks
from colosseum.benchmark.utils import obtain_hyperparameters
from colosseum.experiment.experiment import run_experiments_from_folders
from colosseum.experiment.hyperopt.base import HyperparameterOptimizationConfiguration, \
    DEFAULT_HYPERPARAMETERS_OPTIMIZATION_CONFIGURATION

if TYPE_CHECKING:
    from colosseum.agent.agents.base import BaseAgent


def run_benchmark(
    agent_classes_to_benchmark_with_hyperparameters: Union[
        List[Type["BaseAgent"]], Dict[Type["BaseAgent"], str]
    ],
    n_cores: int,
    load_colosseum_cached_hyperparams: bool = False,
    benchmark_to_run: ColosseumBenchmarks = ColosseumBenchmarks.ALL,
    use_ray: bool = False,
    overwrite_previous_benchmark_config=True,
    verbose_hyperopt: bool = True,
    hpoc: HyperparameterOptimizationConfiguration = DEFAULT_HYPERPARAMETERS_OPTIMIZATION_CONFIGURATION,
):
    """
    runs the Colosseum benchmark for the given agent classes. If no hyperparameters are provided then the default
    hyperparameters optimization procedure is run.

    Parameters
    ----------
    agent_classes_to_benchmark_with_hyperparameters : Union[List[Type["BaseAgent"]], Dict[Type["BaseAgent"], str]
        is the list of agent classes or a dictionary whose key are agent classes and values are the corresponding
        hyperparameters.
    n_cores : int
        is the number of cores that are made available to run the benchmark and the hyperparameter optimization when
        necessary.
    load_colosseum_cached_hyperparams : bool, optional
        checks whether to load the hyperparameters of the default agents from Colosseum cache. By default, it is set to
        False.
    benchmark_to_run : ColosseumBenchmarks, optional
        is the type of Colosseum benchmark to run. By default, it is set to the benchmark for all the different MDP
        settings.
    use_ray : bool, optional
        checks whether to use ray for the hyperparameters optimization when necessary. By default, it is set to False.
    overwrite_previous_benchmark_config : bool, optional
        checks whether to remove any previous benchmark configuration from the config.get_experiments_to_run() folder.
        By default, it is set to True.
    verbose_hyperopt : bool, optional
        checks whether the hyperparameters optimization procedure is verbose or not.
    hpoc: HyperparameterOptimizationConfiguration, optional
        is the configuration for the hyperparameters optimization procedure. By default, the default one is used.
    """

    # Obtain the gin config for each agent class
    agent_classes_to_benchmark_with_hyperparameters = obtain_hyperparameters(
        agent_classes_to_benchmark_with_hyperparameters,
        n_cores,
        load_colosseum_cached_hyperparams,
        verbose_hyperopt,
        hpoc
    )

    stopped_existing_experiments = []
    for existing_experiment in os.listdir(config.get_experiment_to_run_folder()):
        # Removing previous configuration of the benchmark
        if (
            overwrite_previous_benchmark_config
            and existing_experiment in benchmark_to_run.experiment_names()
        ):
            shutil.rmtree(config.get_experiment_to_run_folder() + existing_experiment)
        # Ensure that the other experiments in the folder are not run together
        elif existing_experiment[0] != "_":
            stopped_existing_experiments.append(existing_experiment)
            shutil.move(
                config.get_experiment_to_run_folder() + existing_experiment,
                config.get_experiment_to_run_folder() + "_" + existing_experiment,
            )

    # Create the folder structure for the mdp_configs of the benchmark
    benchmark_to_run.get_copy_benchmark_to_folder(config.get_experiment_to_run_folder())

    # Copy the obtained hyperparameters into the experiment config folder
    for (
        agent_class,
        hyper_gin_config,
    ) in agent_classes_to_benchmark_with_hyperparameters.items():
        benchmark_to_run.add_agent_config(
            config.get_experiment_to_run_folder(), hyper_gin_config, agent_class
        )

    run_experiments_from_folders(n_cores, use_ray=use_ray)

    # Restoring the previously stopped experiments
    for stopped_existing_experiment in stopped_existing_experiments:
        shutil.move(
            config.get_experiment_to_run_folder() + "_" + stopped_existing_experiment,
            config.get_experiment_to_run_folder() + stopped_existing_experiment,
        )
