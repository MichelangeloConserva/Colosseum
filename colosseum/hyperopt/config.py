from dataclasses import dataclass
from typing import Type, List, TYPE_CHECKING

from colosseum.emission_maps import EmissionMap, StateInfo
from colosseum.experiment import ExperimentConfig

if TYPE_CHECKING:
    from colosseum.mdp.base import BaseMDP


@dataclass(frozen=True)
class HyperOptConfig:
    seed: int
    """The seed that controls the parameters optimization procedure."""
    n_timesteps: int
    """The number of time step for the Agent/MDP interaction."""
    max_interaction_time_s: float
    """The maximum amount of time the agent is allowed to interact with the MDP."""
    n_samples_agents: int
    """The number of samples from the Agent hyperparameter space."""
    n_samples_mdps: int
    """The number of samples from the MDP parameters spaces defined to provide an interesting but mild challenge."""
    log_every: int
    """The number of time steps between each time the performance metrics are computed."""
    emission_map: Type[EmissionMap] = None
    """The emission map that will be use to provide a state/observation to the Agent. By default, it is tabular."""
    mdp_classes: List[Type["BaseMDP"]] = None
    """The MDP classes to be used in the hyperparameter optimization procedure. By default, we use all the available ones."""
    n_seeds: int = 3
    """The number of times each Agent/MDP interaction is repeated."""

    @property
    def experiment_config(self) -> ExperimentConfig:
        """
        Returns
        -------
        ExperimentConfig
            The experiment configuration associated to the parameters optimization procedure.
        """
        return ExperimentConfig(
            n_seeds=self.n_seeds,
            n_steps=self.n_timesteps,
            max_interaction_time_s=self.max_interaction_time_s,
            log_performance_indicators_every=self.log_every,
        )


DEFAULT_HYPEROPT_CONF = HyperOptConfig(
    seed=42,
    n_timesteps=250_000,
    max_interaction_time_s=5 * 60,
    n_samples_agents=50,
    n_samples_mdps=5,
    log_every=100_000,
)
"""The default parameters optimization configuration for the tabular setting."""

SMALL_HYPEROPT_CONF = HyperOptConfig(
    seed=42,
    n_timesteps=30_000,
    max_interaction_time_s=120,
    n_samples_agents=2,
    n_samples_mdps=2,
    log_every=10_000,
    n_seeds=1,
)
"""The default small scale parameters optimization configuration for the tabular setting."""

DEFAULT_HYPEROPT_CONF_NONTABULAR = HyperOptConfig(
    seed=42,
    n_timesteps=250_000,
    max_interaction_time_s=10 * 60,
    n_samples_agents=50,
    n_samples_mdps=5,
    log_every=50_000,
    emission_map=StateInfo,
)
"""The default small scale parameters optimization configuration for the non-tabular setting."""

SMALL_HYPEROPT_CONF_NONTABULAR = HyperOptConfig(
    seed=42,
    n_timesteps=50_000,
    max_interaction_time_s=1 * 60,
    n_samples_agents=2,
    n_samples_mdps=2,
    log_every=10_000,
    emission_map=StateInfo,
    n_seeds=1,
)
"""The default parameters optimization configuration for the non-tabular setting."""
