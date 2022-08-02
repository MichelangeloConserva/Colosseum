import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Type

from colosseum import config

if TYPE_CHECKING:
    from colosseum.mdp import BaseMDP
    from colosseum.agent.agents.base import BaseAgent


@dataclass()
class ExperimentConfig:
    def __init__(
        self,
        remove_experiment_configuration_folder: bool,
        overwrite_previous_experiment: bool,
        n_seeds: int,
        n_steps: int,
        max_agent_mdp_interaction_time_s: int,
        log_performance_indicator_every: int,
    ):
        """
        returns an experiment configuration.

        Parameters
        ----------
        remove_experiment_configuration_folder : bool
            checks whether to remove the folder containing the experiment configuration.
        overwrite_previous_experiment : bool
            checks whether to overwrite the files corresponding to the same experiment configuration.
        n_seeds : int
            is the number of seed each agent/MDP interaction is repeated.
        n_steps : int
            is the optimization horizon.
        max_agent_mdp_interaction_time_s : int
            is the maximum training time given to the agent in seconds.
        log_performance_indicator_every : int
            is the number of step after which the performance indicators are calculated and stored.
        """
        self.remove_experiment_configuration_folder = (
            remove_experiment_configuration_folder
        )
        self.n_seeds = n_seeds
        self.overwrite_previous_experiment = overwrite_previous_experiment
        self.n_steps = n_steps
        self.max_agent_mdp_interaction_time_s = max_agent_mdp_interaction_time_s
        self.log_performance_indicators_every = log_performance_indicator_every


@dataclass(frozen=True)
class ExperimentInstance:
    seed: int
    mdp_class: Type["BaseMDP"]
    mdp_scope: str
    agent_class: Type["BaseAgent"]
    agent_scope: str
    result_folder: str
    gin_config_files: List[str]
    experiment_config: ExperimentConfig

    @property
    def experiment_name(self) -> str:
        return self.result_folder[self.result_folder.rfind(os.sep) + 1 :]

    @property
    def experiment_label(self) -> str:
        return (
            f"{self.mdp_scope}{config.EXPERIMENT_SEPARATOR_PRMS}{self.mdp_class.__name__}"
            + f"{config.EXPERIMENT_SEPARATOR_MDP_AGENT}"
            + f"{self.agent_scope}{config.EXPERIMENT_SEPARATOR_PRMS}{self.agent_class.__name__}"
        )

    @property
    def does_log_file_exists(self) -> bool:
        lf = (
            self.result_folder
            + os.sep
            + "logs"
            + os.sep
            + self.experiment_label
            + f"{os.sep}seed{self.seed}_logs.csv"
        )
        return os.path.exists(lf)

    def __str__(self):
        return f"{self.experiment_name} for seed:{self.seed}, " + self.experiment_label

    def __repr__(self):
        return str(self)
