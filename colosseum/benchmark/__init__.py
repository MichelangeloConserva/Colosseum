import os
import shutil
from enum import IntEnum
from typing import TYPE_CHECKING, List, Type

import colosseum
from colosseum.utils import ensure_folder

if TYPE_CHECKING:
    from colosseum.agent.agents.base import BaseAgent

BENCHMARKS_DIRECTORY = (
    colosseum.__file__[: colosseum.__file__.rfind(os.sep)]
    + os.sep
    + "benchmark"
    + os.sep
)


class ColosseumBenchmarks(IntEnum):
    CONTINUOUS_ERGODIC = 0
    CONTINUOUS_COMMUNICATING = 1
    CONTINUOUS = 2
    EPISODIC_ERGODIC = 3
    EPISODIC_COMMUNICATING = 4
    EPISODIC = 5
    ALL = 6
    EPISODIC_QUICK_TEST = 7
    CONTINUOUS_QUICK_TEST = 8

    def experiment_names(self) -> List[str]:
        if self == ColosseumBenchmarks.EPISODIC_QUICK_TEST:
            return ["benchmark_episodic_quick_test"]
        if self == ColosseumBenchmarks.CONTINUOUS_QUICK_TEST:
            return ["benchmark_continuous_ergodic"]
        if self == ColosseumBenchmarks.CONTINUOUS_ERGODIC:
            return ["benchmark_continuous_ergodic"]
        if self == ColosseumBenchmarks.CONTINUOUS_COMMUNICATING:
            return ["benchmark_continuous_communicating"]
        if self == ColosseumBenchmarks.CONTINUOUS:
            return (
                ColosseumBenchmarks.CONTINUOUS_ERGODIC.experiment_names()
                + ColosseumBenchmarks.CONTINUOUS_COMMUNICATING.experiment_names()
            )
        if self == ColosseumBenchmarks.EPISODIC_ERGODIC:
            return ["benchmark_episodic_ergodic"]
        if self == ColosseumBenchmarks.EPISODIC_COMMUNICATING:
            return ["benchmark_episodic_communicating"]
        if self == ColosseumBenchmarks.EPISODIC:
            return (
                ColosseumBenchmarks.EPISODIC_ERGODIC.experiment_names()
                + ColosseumBenchmarks.EPISODIC_COMMUNICATING.experiment_names()
            )
        if self == ColosseumBenchmarks.ALL:
            return (
                ColosseumBenchmarks.CONTINUOUS.experiment_names()
                + ColosseumBenchmarks.EPISODIC.experiment_names()
            )

    def get_copy_benchmark_to_folder(self, folder_path: str):
        for en in self.experiment_names():
            if os.path.isdir(ensure_folder(folder_path) + en):
                shutil.rmtree(ensure_folder(folder_path) + en)
            shutil.copytree(BENCHMARKS_DIRECTORY + en, ensure_folder(folder_path) + en)

    def add_agent_config(
        self, folder_path: str, agent_gin_config: str, agent_class: Type["BaseAgent"]
    ):
        def check_mdp_config_folder(setting: str) -> str:
            folder = ensure_folder(folder_path) + setting + os.sep
            assert os.path.isdir(
                folder + "mdp_configs"
            ), "Please ensure that the benchmark folder with the mdp configuration has been created."
            return folder

        def write_gin_file(_folder):
            os.makedirs(_folder + "agent_configs", exist_ok=True)
            with open(
                _folder + f"agent_configs{os.sep}{agent_class.__name__}.gin", "w"
            ) as f:
                f.write(agent_gin_config)

        if self == ColosseumBenchmarks.CONTINUOUS_QUICK_TEST:
            folder = check_mdp_config_folder("benchmark_continuous_quick_test")
            if not agent_class.is_episodic():
                write_gin_file(folder)

        if self == ColosseumBenchmarks.EPISODIC_QUICK_TEST:
            folder = check_mdp_config_folder("benchmark_episodic_quick_test")
            if agent_class.is_episodic():
                write_gin_file(folder)

        if self == ColosseumBenchmarks.CONTINUOUS_ERGODIC:
            folder = check_mdp_config_folder("benchmark_continuous_ergodic")
            if not agent_class.is_episodic():
                write_gin_file(folder)

        if self == ColosseumBenchmarks.CONTINUOUS_COMMUNICATING:
            folder = check_mdp_config_folder("benchmark_continuous_communicating")
            if not agent_class.is_episodic():
                write_gin_file(folder)

        if self == ColosseumBenchmarks.CONTINUOUS:
            ColosseumBenchmarks.CONTINUOUS_ERGODIC.add_agent_config(
                folder_path, agent_gin_config, agent_class
            )
            ColosseumBenchmarks.CONTINUOUS_COMMUNICATING.add_agent_config(
                folder_path, agent_gin_config, agent_class
            )

        if self == ColosseumBenchmarks.EPISODIC_ERGODIC:
            folder = check_mdp_config_folder("benchmark_episodic_ergodic")
            if agent_class.is_episodic():
                write_gin_file(folder)

        if self == ColosseumBenchmarks.EPISODIC_COMMUNICATING:
            folder = check_mdp_config_folder("benchmark_episodic_communicating")
            if agent_class.is_episodic():
                write_gin_file(folder)

        if self == ColosseumBenchmarks.EPISODIC:
            ColosseumBenchmarks.EPISODIC_ERGODIC.add_agent_config(
                folder_path, agent_gin_config, agent_class
            )
            ColosseumBenchmarks.EPISODIC_COMMUNICATING.add_agent_config(
                folder_path, agent_gin_config, agent_class
            )
        if self == ColosseumBenchmarks.ALL:
            ColosseumBenchmarks.CONTINUOUS_ERGODIC.add_agent_config(
                folder_path, agent_gin_config, agent_class
            )
            ColosseumBenchmarks.CONTINUOUS_COMMUNICATING.add_agent_config(
                folder_path, agent_gin_config, agent_class
            )
            ColosseumBenchmarks.EPISODIC_ERGODIC.add_agent_config(
                folder_path, agent_gin_config, agent_class
            )
            ColosseumBenchmarks.EPISODIC_COMMUNICATING.add_agent_config(
                folder_path, agent_gin_config, agent_class
            )
