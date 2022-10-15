import dataclasses
import os
from enum import IntEnum
from typing import TYPE_CHECKING, Type, Dict

import yaml.reader

import colosseum
from colosseum import config
from colosseum.experiment import ExperimentConfig
from colosseum.utils import ensure_folder
from colosseum.utils.miscellanea import compare_gin_configs

if TYPE_CHECKING:
    from colosseum.mdp import BaseMDP

BENCHMARKS_DIRECTORY = (
    os.path.dirname(colosseum.__file__) + os.sep + "benchmark" + os.sep
)


@dataclasses.dataclass(frozen=True)
class ColosseumBenchmark:
    """
    The `ColosseumBenchmark` encodes a benchmark, which is composed of MDP configurations and an experimental
    configuration.
    """

    name: str
    """The name to assign to the benchmark."""
    mdps_gin_configs: Dict[Type["BaseMDP"], str]
    """The gin config files of MDPs in the benchmark."""
    experiment_config: ExperimentConfig
    """The experiment configuration for the benchmark."""

    def __eq__(self, other):
        if type(other) != ColosseumBenchmark:
            return False
        return (
            self.experiment_config == other.experiment_config
            and compare_gin_configs(self.mdps_gin_configs, other.mdps_gin_configs)
        )

    def instantiate(self, benchmark_folder: str):
        """
        instantiates the benchmark locally.

        Parameters
        ----------
        benchmark_folder : str
            The folder where the benchmark will be instantiated.
        """
        os.makedirs(ensure_folder(benchmark_folder) + "mdp_configs", exist_ok=True)
        for mdp_cl, gin_configs in self.mdps_gin_configs.items():
            with open(
                ensure_folder(benchmark_folder)
                + "mdp_configs"
                + os.sep
                + mdp_cl.__name__
                + ".gin",
                "w",
            ) as f:
                f.write(gin_configs)

        self.experiment_config.store_at(benchmark_folder)

    def get_experiments_benchmark_log_folder(self) -> str:
        """
        Returns
        -------
        str
            The experiment folder corresponding to the benchmark given the experiments folder configuration.
        """
        return self.get_log_folder(config.get_experiments_folder())

    def get_hyperopt_benchmark_log_folder(self):
        """
        Returns
        -------
        str
            The parameters optimization folder corresponding to the benchmark given the experiments huperopt
            configuration.
        """
        return self.get_log_folder(config.get_hyperopt_folder())

    def get_log_folder(self, benchmark_folder: str):
        """
        Parameters
        ----------
        benchmark_folder : str
            The folder where the benchmark could have been instantiated.

        Returns
        -------
        str
            The folder that correspond to the benchmark if it was instantiated in the given benchmark folder.
        """
        return ensure_folder(benchmark_folder + self.name)


class ColosseumDefaultBenchmark(IntEnum):
    CONTINUOUS_ERGODIC = 0
    """The default benchmark for the continuous ergodic setting."""
    CONTINUOUS_COMMUNICATING = 1
    """The default benchmark for the continuous communicating setting."""
    EPISODIC_ERGODIC = 2
    """The default benchmark for the episodic ergodic setting."""
    EPISODIC_COMMUNICATING = 3
    """The default benchmark for the episodic communicating setting."""
    EPISODIC_QUICK_TEST = 4
    """A quick benchmark for the episodic setting."""
    CONTINUOUS_QUICK_TEST = 5
    """A quick benchmark for the continuous setting."""

    @staticmethod
    def get_default_experiment_config() -> ExperimentConfig:
        """
        Returns
        -------
        ExperimentConfig
            The default experiment configuration proposed by the package for the tabular setting.
        """
        with open(BENCHMARKS_DIRECTORY + "experiment_config.yml", "r") as f:
            experimental_config = yaml.load(f, yaml.Loader)
        return ExperimentConfig(**experimental_config)

    @staticmethod
    def get_default_non_tabular_experiment_config() -> ExperimentConfig:
        """
        Returns
        -------
        ExperimentConfig
            The default experiment configuration proposed by the package for the non-tabular setting.
        """
        from colosseum.emission_maps import StateInfo

        default_experiment_config = ColosseumDefaultBenchmark.get_default_experiment_config()
        default_experiment_config = dataclasses.asdict(default_experiment_config)
        default_experiment_config["emission_map"] = StateInfo
        return ExperimentConfig(**default_experiment_config)

    def get_benchmark(
        self, postfix="", experiment_config: ExperimentConfig = None, non_tabular : bool = False
    ) -> ColosseumBenchmark:
        """
        creates a `ColosseumBenchmark` corresponding to the default benchmark the object encodes.

        Parameters
        ----------
        postfix : str
            A postfix string to add to the default name of the default benchmark.
        experiment_config : ExperimentConfig
            The experiment config to be substituted to the default one. By default, no substitution happens.
        non_tabular : bool
            If True, the default non-tabular experimental configuration is used.
        Returns
        -------
        ColosseumBenchmark
            The default benchmark the object encodes.
        """

        exp_folder = (
            BENCHMARKS_DIRECTORY
            + f"benchmark_"
            + self.name.lower()
        )
        from colosseum.benchmark.utils import retrieve_benchmark

        if experiment_config is None and "QUICK" not in self.name:
            if non_tabular:
                experiment_config = ColosseumDefaultBenchmark.get_default_non_tabular_experiment_config()
            else:
                experiment_config = ColosseumDefaultBenchmark.get_default_experiment_config()

        benchmark = retrieve_benchmark(
            exp_folder,
            experiment_config,
            f"{'_' if len(str(postfix)) > 0 else ''}{postfix}",
        )
        return benchmark
