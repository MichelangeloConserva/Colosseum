import os
from dataclasses import dataclass
from typing import Type, List

from colosseum import config
from colosseum.agent.agents.base import BaseAgent
from colosseum.emission_maps import EmissionMap
from colosseum.experiment import ExperimentConfig
from colosseum.mdp import BaseMDP
from colosseum.utils import ensure_folder


@dataclass(frozen=True)
class ExperimentInstance:
    seed: int
    """The random seed for agent/MDP interaction."""
    mdp_class: Type["BaseMDP"]
    """The class of the MDP."""
    mdp_scope: str
    """The gin parameter for the MDP."""
    agent_class: Type["BaseAgent"]
    """The class of the agent."""
    agent_scope: str
    """The gin parameter for the agent."""
    result_folder: str
    """The folder where the results of the interactions will be stored."""
    gin_config_files: List[str]
    """The paths to the gin configuration files for the agent and MDPs"""
    experiment_config: ExperimentConfig
    """The experiment configuration of the agent/MDP interaction."""

    @property
    def emission_map(self) -> Type[EmissionMap]:
        """
        Returns
        -------
        Type[EmissionMap]
            The emission map class of the experiment configuration.
        """
        return self.experiment_config.emission_map

    @property
    def experiment_name(self) -> str:
        """
        Returns
        -------
        str
            The folder where the results are stored.
        """
        return self.result_folder[self.result_folder.rfind(os.sep) + 1 :]

    @property
    def experiment_label(self) -> str:
        """
        Returns
        -------
        str
            The label for the experiment, which identifies the agent class, agent gin config, MDP class, and MDP gin
            config.
        """
        return (
            f"{self.mdp_scope}{config.EXPERIMENT_SEPARATOR_PRMS}{self.mdp_class.__name__}"
            + f"{config.EXPERIMENT_SEPARATOR_MDP_AGENT}"
            + f"{self.agent_scope}{config.EXPERIMENT_SEPARATOR_PRMS}{self.agent_class.__name__}"
        )

    @property
    def does_log_file_exists(self) -> bool:
        """
        Returns
        -------
        bool
            True if the csv log file where the results of the interaction were supposed to be stored exists.
        """
        lf = (
            ensure_folder(self.result_folder)
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
