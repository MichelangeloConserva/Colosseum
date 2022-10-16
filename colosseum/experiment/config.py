from dataclasses import dataclass
from typing import Type

import yaml

from colosseum.emission_maps import EmissionMap
from colosseum.utils import ensure_folder


@dataclass(frozen=True)
class ExperimentConfig:
    n_seeds: int
    """is the number of seed each agent/MDP interaction is repeated."""
    n_steps: int
    """is the optimization horizon."""
    max_interaction_time_s: float
    """is the maximum training time given to the agent in seconds."""
    log_performance_indicators_every: int
    """is the number of step after which the performance indicators are calculated and stored."""
    emission_map: Type["EmissionMap"] = None
    """is the emission map for the MDP."""

    def store_at(self, dest_folder: str):
        with open(ensure_folder(dest_folder) + "experiment_config.yml", "w") as f:
            conf = {
                "n_seeds": self.n_seeds,
                "n_steps": self.n_steps,
                "max_interaction_time_s": self.max_interaction_time_s,
                "log_performance_indicators_every": self.log_performance_indicators_every,
            }
            if self.emission_map is not None:
                conf["emission_map"] = self.emission_map.__name__

            yaml.dump(conf, f)
