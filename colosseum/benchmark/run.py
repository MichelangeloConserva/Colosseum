import os
import shutil
from typing import TYPE_CHECKING, Dict, List, Type, Tuple, Iterable

from colosseum import config
from colosseum.benchmark.benchmark import ColosseumBenchmark
from colosseum.benchmark.utils import (
    instantiate_benchmark_folder,
    instantiate_agent_configs,
)
from colosseum.experiment.experiment_instance import ExperimentInstance
from colosseum.experiment.experiment_instances import (
    get_experiment_instances_from_folder,
)
from colosseum.utils import ensure_folder

if TYPE_CHECKING:
    from colosseum.agent.agents.base import BaseAgent


def instantiate_agents_and_benchmark(
    agents_configs: Dict[Type["BaseAgent"], str],
    benchmark: ColosseumBenchmark,
    overwrite_previous_experiment: bool = False,
    experiment_folder: str = None,
) -> str:
    """
    instantiate the benchmark and the agents configs locally.

    Parameters
    ----------
    agents_configs : Dict[Type["BaseAgent"], str]
        The dictionary associates agent classes to their gin config files.
    benchmark : ColosseumBenchmark
        The benchmark to be instantiated.
    overwrite_previous_experiment : bool
        If True the destination folder is cleared before instantiating the MDP configs. If False, it raises an error if
        it finds a different set of agents configs, MDP configs, or `ExperimentalConfig` in the destination folder.
    experiment_folder : str
        The folder where to instantiate the benchmark and the agents configs. By default, it is taken from the package
        configurations.

    Returns
    -------
    str
        The folder where the benchmark and the agents configs have been instantiated.
    """

    # Avoid mixing episodic and continuous settings
    assert all(
        agent_class.is_episodic() == list(agents_configs)[0].is_episodic()
        for agent_class in agents_configs
    )
    assert all(
        mdp_configs.is_episodic() == list(agents_configs)[0].is_episodic()
        for mdp_configs in benchmark.mdps_gin_configs
    )

    # Set the current experiment folder of the given benchmark
    benchmark_folder = (
        config.get_experiments_folder()
        if experiment_folder is None
        else ensure_folder(experiment_folder)
    ) + benchmark.name

    if overwrite_previous_experiment:
        # Remove any previous experiment directory
        shutil.rmtree(benchmark_folder, ignore_errors=True)
        os.makedirs(benchmark_folder)

    # Instantiate the mdp configs
    instantiate_benchmark_folder(benchmark, benchmark_folder)

    # Instantiate the agent configs
    instantiate_agent_configs(agents_configs, benchmark_folder)

    return benchmark_folder


def instantiate_and_get_exp_instances_from_benchmark(
    agents_configs: Dict[Type["BaseAgent"], str],
    benchmark: ColosseumBenchmark,
    overwrite_previous_experiment: bool = False,
    experiment_folder: str = None,
) -> List[ExperimentInstance]:
    """
    instantiate the benchmark and the agents configs locally, and creates the corresponding `ExperimentInstance`.

    Parameters
    ----------
    agents_configs : Dict[Type["BaseAgent"], str]
        The dictionary associates agent classes to their gin config files.
    benchmark : ColosseumBenchmark
        The benchmark to be instantiated.
    overwrite_previous_experiment : bool
        If True the destination folder is cleared before instantiating the MDP configs. If False, it raises an error if
        it finds a different set of agents configs, MDP configs, or `ExperimentalConfig` in the destination folder.
    experiment_folder : str
        The folder where to instantiate the benchmark and the agents configs. By default, it is taken from the package
        configurations.

    Returns
    -------
    List[ExperimentInstance]
        The `ExperimentInstance`s corresponding to the benchmark and the agents configs.
    """

    # Instantiate the local directories for the benchmark
    benchmark_folder = instantiate_agents_and_benchmark(
        agents_configs, benchmark, overwrite_previous_experiment, experiment_folder
    )

    # Create the ExperimentInstance objects
    return get_experiment_instances_from_folder(benchmark_folder)


def instantiate_and_get_exp_instances_from_agents_and_benchmarks(
    agents_and_benchmarks: Iterable[
        Tuple[Dict[Type["BaseAgent"], str], ColosseumBenchmark]
    ],
    overwrite_previous_experiment: bool = False,
    experiment_folder: str = None,
) -> List[ExperimentInstance]:
    """
    instantiate the benchmarks_and_agents and the agents configs locally, and creates the corresponding `ExperimentInstance`.

    Parameters
    ----------
    agents_and_benchmarks : Iterable[Tuple[Dict[Type["BaseAgent"], str], ColosseumBenchmark]]
        The agent configs and benchmarks_and_agents to be instantiated.
    overwrite_previous_experiment : bool
        If True the destination folder is cleared before instantiating the MDP configs. If False, it raises an error if
        it finds a different set of agents configs, MDP configs, or `ExperimentalConfig` in the destination folder.
    experiment_folder : str
        The folder where to instantiate the benchmark and the agents configs. By default, it is taken from the package
        configurations.

    Returns
    -------
    List[ExperimentInstance]
        The `ExperimentInstance`s corresponding to the benchmarks_and_agents and the agents configs.
    """

    experiment_instances = []
    for agents_configs, benchmark in agents_and_benchmarks:
        experiment_instances += instantiate_and_get_exp_instances_from_benchmark(
            agents_configs, benchmark, overwrite_previous_experiment, experiment_folder
        )
    return experiment_instances


def instantiate_and_get_exp_instances_from_agents_and_benchmarks_for_hyperopt(
    agents_and_benchmarks: Iterable[
        Tuple[Dict[Type["BaseAgent"], str], ColosseumBenchmark]
    ],
    overwrite_previous_experiment: bool = False,
) -> List[ExperimentInstance]:
    """
    instantiate the benchmarks_and_agents and the agents configs for the parameters optimization locally (by
    checking the parameters optimization folder in the package configurations), and creates the corresponding
    `ExperimentInstance`.

    Parameters
    ----------
    agents_and_benchmarks : Iterable[Tuple[Dict[Type["BaseAgent"], str], ColosseumBenchmark]]
        The agent configs and benchmarks_and_agents to be instantiated.
    overwrite_previous_experiment : bool
        If True the destination folder is cleared before instantiating the MDP configs. If False, it raises an error if
        it finds a different set of agents configs, MDP configs, or `ExperimentalConfig` in the destination folder.

    Returns
    -------
    List[ExperimentInstance]
        The `ExperimentInstance`s corresponding to the benchmarks_and_agents and the agents configs.
    """
    return instantiate_and_get_exp_instances_from_agents_and_benchmarks(
        agents_and_benchmarks,
        overwrite_previous_experiment,
        config.get_hyperopt_folder(),
    )
