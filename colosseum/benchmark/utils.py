import os
from typing import TYPE_CHECKING, Dict, List, Type, Union

from colosseum.benchmark import BENCHMARKS_DIRECTORY
from colosseum.experiment.hyperopt.base import (
    DEFAULT_HYPERPARAMETERS_OPTIMIZATION_CONFIGURATION,
    HyperparameterOptimizationConfiguration,
)
from colosseum.experiment.hyperopt.multiprocessing import MPRegretOptimizer

if TYPE_CHECKING:
    from colosseum.agent.agents.base import BaseAgent


def obtain_hyperparameters(
    agent_classes_to_benchmark_with_hyperparameters: Union[
        List[Type["BaseAgent"]], Dict[Type["BaseAgent"], Union[None, str]]
    ],
    n_cores: int,
    load_colosseum_cached_hyperparams: bool = False,
    verbose=True,
    hpoc: HyperparameterOptimizationConfiguration = DEFAULT_HYPERPARAMETERS_OPTIMIZATION_CONFIGURATION,
) -> Dict[Type["BaseAgent"], str]:
    """
    returns a dictionary containing the hyperparameter for the given agents by either reading them from files or by
    running the default hyperparameters optimization procedure.

    Parameters
    ----------
    agent_classes_to_benchmark_with_hyperparameters : Union[List[Type["BaseAgent"]], Dict[Type["BaseAgent"], Union[None, str]]]
        is the list of agent classes or a dictionary whose key are agent classes and values are the corresponding
        hyperparameters.
    n_cores : int
        is the number of cores that are made available to run the benchmark and the hyperparameter optimization when
        necessary.
    load_colosseum_cached_hyperparams : bool, optional
        checks whether to load the hyperparameters of the default agents from Colosseum cache. By default, it is set to
        False.
    verbose : bool, optional
        checks whether the hyperparameters optimization procedure is verbose or not. By default, it is set to True.
    hpoc: HyperparameterOptimizationConfiguration, optional
        is the configuration for the hyperparameters optimization procedure. By default, the default one is used.
    """

    if type(agent_classes_to_benchmark_with_hyperparameters) == list:
        agent_classes_to_benchmark_with_hyperparameters = dict(
            zip(
                agent_classes_to_benchmark_with_hyperparameters,
                [None] * len(agent_classes_to_benchmark_with_hyperparameters),
            )
        )

    for (
        agent_class,
        hyper_gin_config,
    ) in agent_classes_to_benchmark_with_hyperparameters.items():
        if hyper_gin_config is None:

            # If the config is not given, first look if this agent class is a default one
            if load_colosseum_cached_hyperparams:
                cache_dir = (
                    BENCHMARKS_DIRECTORY
                    + "cached_hyperparameters"
                    + os.sep
                    + "agent_configs"
                    + os.sep
                )
                if os.path.isfile(cache_dir + agent_class.__name__ + ".gin"):
                    with open(cache_dir + agent_class.__name__ + ".gin", "r") as f:
                        agent_classes_to_benchmark_with_hyperparameters[
                            agent_class
                        ] = f.read()
                    continue

            # If it is not then run the hyperparameters optimization procedure
            opt = MPRegretOptimizer(agent_class=agent_class, hpoc=hpoc)
            hyper_parameters, score = opt.optimize(n_cores, verbose)
            agent_classes_to_benchmark_with_hyperparameters[
                agent_class
            ] = agent_class.produce_gin_file_from_hyperparameters(hyper_parameters)

        elif type(hyper_gin_config) == str:
            if os.path.isfile(hyper_gin_config):
                with open(hyper_gin_config, "r") as f:
                    agent_classes_to_benchmark_with_hyperparameters[agent_class] = f.read()
            elif "prms_" in hyper_gin_config:
                agent_classes_to_benchmark_with_hyperparameters[agent_class] = hyper_gin_config
            else:
                raise ValueError(
                    f"There is an error with hyper_gin_config. The received value is {hyper_gin_config}"
                )
        else:
            raise ValueError(
                "The agent configuration should either be None or a file path."
            )
        # if verbose:
        #     print(agent_class.__name__)
        #     print(agent_classes_to_benchmark_with_hyperparameters[agent_class])
        #     print("---------------------")

        os.makedirs(f"tmp{os.sep}hyperopt", exist_ok=True)
        with open(
            f"tmp{os.sep}hyperopt{os.sep}{agent_class.__name__}_latest_hyperparameter_config.gin",
            "w",
        ) as f:
            f.write(agent_classes_to_benchmark_with_hyperparameters[agent_class])

    return agent_classes_to_benchmark_with_hyperparameters
