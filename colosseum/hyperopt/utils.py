from typing import List, Type, Dict, Tuple

from colosseum.agent.agents.base import BaseAgent
from colosseum.agent.utils import sample_agent_gin_configs_file
from colosseum.benchmark.benchmark import ColosseumBenchmark
from colosseum.hyperopt import HyperOptConfig
from colosseum.utils import get_colosseum_mdp_classes
from colosseum.utils.miscellanea import sample_mdp_gin_configs_file


def sample_agent_configs_and_benchmarks_for_hyperopt(
    agent_cls: List[Type[BaseAgent]], hpoc: HyperOptConfig
) -> List[Tuple[Dict[Type[BaseAgent], str], ColosseumBenchmark]]:
    """
    samples the agent configurations from the agents parameters sample spaces and the MDP configuration from the
    MDP sampling functions to be used in the parameters optimization procedures.

    Parameters
    ----------
    agent_cls : Type["BaseAgent"]
        The agent classes to be optimized.
    hpoc : HyperOptConfig
        The parameters optimization procedure configuration.
    Returns
    -------
    List[Tuple[Dict[Type[BaseAgent], str], ColosseumBenchmark]]
        The agents configurations and the benchmarks with the MDP configs for the parameters optimization procedure.
    """

    agents_and_benchmarks = []

    episodic_benchmark_name = f"hyperopt_episodic"
    episodic_agents_configs = dict()

    continuous_benchmark_name = f"hyperopt_continuous"
    continuous_agents_configs = dict()

    for agent_cl in agent_cls:

        # First sample the agent parameters
        agent_samples = sample_agent_gin_configs_file(
            agent_cl, hpoc.n_samples_agents, hpoc.seed
        )

        if agent_cl.is_episodic():
            episodic_agents_configs[agent_cl] = agent_samples
        else:
            continuous_agents_configs[agent_cl] = agent_samples

    # Sampling the episodic MDPs
    if len(episodic_agents_configs) > 0:
        episodic_mdps_configs = dict()
        for cl in get_colosseum_mdp_classes(True):
            episodic_mdps_configs[cl] = sample_mdp_gin_configs_file(
                cl, hpoc.n_samples_mdps, hpoc.seed
            )
        episodic_benchmark = ColosseumBenchmark(
            episodic_benchmark_name, episodic_mdps_configs, hpoc.experiment_config
        )
        agents_and_benchmarks.append((episodic_agents_configs, episodic_benchmark))

    # Sampling the continuous MDPs
    if len(continuous_agents_configs) > 0:
        continuous_mdps_configs = dict()
        for cl in get_colosseum_mdp_classes(False):
            continuous_mdps_configs[cl] = sample_mdp_gin_configs_file(
                cl, hpoc.n_samples_mdps, hpoc.seed
            )
        continuous_benchmark = ColosseumBenchmark(
            continuous_benchmark_name, continuous_mdps_configs, hpoc.experiment_config
        )
        agents_and_benchmarks.append((continuous_agents_configs, continuous_benchmark))

    return agents_and_benchmarks
