from typing import Type, TYPE_CHECKING, Any, Dict, List

import numpy as np

from colosseum.utils.miscellanea import rounding_nested_structure

if TYPE_CHECKING:
    from colosseum.agent.agents.base import BaseAgent


def sample_agent_hyperparameters(
    agent_class: Type["BaseAgent"], seed: int
) -> Dict[str, Any]:
    """
    samples parameters from the agent class sample spaces.

    Parameters
    ----------
    agent_class : Type["BaseAgent"]
        The agent class for which we are sampling from.
    seed : int
        The random seed.

    Returns
    -------
    Dict[str, Any]
        The parameters sampled from the agent hyperparameter spaces.
    """
    np.random.seed(seed)
    search_spaces = agent_class.get_hyperparameters_search_spaces()
    return rounding_nested_structure({k: v.sample() for k, v in search_spaces.items()})


def sample_n_agent_hyperparameters(
    n: int, agent_class: Type["BaseAgent"], seed: int
) -> List[Dict[str, Any]]:
    """
    samples n parameters from the agent class sample spaces.

    Parameters
    ----------
    n : int
        The number of samples.
    agent_class : Type["BaseAgent"]
        The agent class for which we are sampling from.
    seed : int
        The random seed.

    Returns
    -------
    List[Dict[str, Any]]
        The list of n parameters sampled from the agent hyperparameter spaces.
    """
    return [sample_agent_hyperparameters(agent_class, seed + i) for i in range(n)]


def sample_agent_gin_configs(
    agent_class: Type["BaseAgent"], n: int = 1, seed: int = 42
) -> List[str]:
    """
    samples gin configurations from the agent class sample spaces.

    Parameters
    ----------
    agent_class : Type["BaseAgent"]
        The agent class for which we are sampling from.
    n : int
        The number of samples. By default, it is set to one.
    seed : int
        The random seed. By default, it is set to :math:`42`.

    Returns
    -------
    List[str]
        The list containing the sampled gin configs.
    """
    return [
        agent_class.produce_gin_file_from_parameters(params, i)
        for i, params in enumerate(sample_n_agent_hyperparameters(n, agent_class, seed))
    ]


def sample_agent_gin_configs_file(
    agent_class: Type["BaseAgent"], n: int = 1, seed: int = 42
) -> str:
    """
    samples gin configurations from the agent class sample spaces and store them in a string that can be used to create
    a gin config file.

    Parameters
    ----------
    agent_class : Type["BaseAgent"]
        The agent class for which we are sampling from.
    n : int
        The number of samples. By default, it is set to one.
    seed : int
        The random seed. By default, it is set to :math:`42`.

    Returns
    -------
    str
        The gin configuration file.
    """
    return "\n".join(sample_agent_gin_configs(agent_class, n, seed))
