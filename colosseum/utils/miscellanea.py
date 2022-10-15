import cProfile
import collections
import importlib
import inspect
import numbers
import os
from glob import glob
from io import StringIO
from typing import TYPE_CHECKING, Iterable, List, Type, Union, Any, Dict

import dm_env
import numpy as np
from scipy.stats import rv_continuous
from tqdm import tqdm

import colosseum
import colosseum.config as config

if TYPE_CHECKING:
    from colosseum.mdp import BaseMDP
    from colosseum.agent.agents.base import BaseAgent


def rounding_nested_structure(x: Dict):
    """
    https://stackoverflow.com/questions/7076254/rounding-decimals-in-nested-data-structures-in-python
    """
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return type(x)(
            (key, rounding_nested_structure(value)) for key, value in x.items()
        )
    if isinstance(x, collections.Container):
        return type(x)(rounding_nested_structure(value) for value in x)
    if isinstance(x, numbers.Number):
        return round(x, config.get_n_floating_sampling_hyperparameters())
    return x


def compare_gin_configs(
    gin_configs1: Dict[Union[Type["BaseMDP"], Type["BaseAgent"]], str],
    gin_configs2: Dict[Union[Type["BaseMDP"], Type["BaseAgent"]], str],
) -> bool:
    """
    Returns
    -------
    bool
        True, if the two gin configs are identical.
    """
    if set(gin_configs1) != set(gin_configs2):
        return False

    gin_configs1 = set(
        map(lambda x: x.replace(" ", "").replace("\n", ""), gin_configs1.values())
    )
    gin_configs2 = set(
        map(lambda x: x.replace(" ", "").replace("\n", ""), gin_configs2.values())
    )
    return gin_configs1 == gin_configs2


def sample_mdp_gin_configs(
    mdp_class: Type["BaseMDP"], n: int = 1, seed: int = 42
) -> List[str]:
    """
    Parameters
    ----------
    mdp_class : Type["BaseMDP"]
        The MDP class to sample from.
    n : int
        The number of samples. By default, one sample is taken.
    seed : int
        The random seed. By default, it is set to 42.

    Returns
    -------
    List[str]
        The n sampled gin configs.
    """
    return [
        mdp_class.produce_gin_file_from_mdp_parameters(params, mdp_class.__name__, i)
        for i, params in enumerate(mdp_class.sample_parameters(n, seed))
    ]


def sample_mdp_gin_configs_file(
    mdp_class: Type["BaseMDP"], n: int = 1, seed: int = 42
) -> str:
    """
    Parameters
    ----------
    mdp_class : Type["BaseMDP"]
        The MDP class to sample from.
    n : int
        The number of samples. By default, one sample is taken.
    seed : int
        The random seed. By default, it is set to 42.

    Returns
    -------
    str
        The n sampled gin configs as a single file string.
    """
    return "\n".join(sample_mdp_gin_configs(mdp_class, n, seed))


def get_empty_ts(state: Any) -> dm_env.TimeStep:
    return dm_env.TimeStep(dm_env.StepType.MID, 0, 0, state)


def profile(file_path):
    def decorator(f):
        print(f"Profiling {f}")

        def inner(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            f(*args, **kwargs)
            pr.disable()
            # after your program ends
            pr.dump_stats(file_path)

        return inner

    return decorator


def get_colosseum_mdp_classes(episodic: bool = None) -> List[Type["BaseMDP"]]:
    """
    Returns
    -------
    List[Type["BaseMDP"]]
        All available MDP classes in the package.
    """
    if episodic is None:
        return _get_colosseum_mdp_classes() + _get_colosseum_mdp_classes(False)
    if episodic:
        return _get_colosseum_mdp_classes()
    return _get_colosseum_mdp_classes(False)


def _get_colosseum_mdp_classes(episodic=True) -> List[Type["BaseMDP"]]:
    import colosseum

    kw = "Episodic" if episodic else "Continuous"
    mdp_path = "finite_horizon" if episodic else "infinite_horizon"
    return [
        next(
            filter(
                lambda x: kw in x[0] and "MDP" not in x[0],
                importlib.import_module(
                    mdp_file[mdp_file.find("colosseum") :].replace(os.sep, ".")[:-3]
                ).__dict__.items(),
            )
        )[1]
        for mdp_file in glob(
            f"{os.path.dirname(inspect.getfile(colosseum))}{os.sep}mdp{os.sep}**{os.sep}{mdp_path}.py",
            recursive=True,
        )
    ]


def get_mdp_class_from_name(mdp_class_name: str) -> Type["BaseMDP"]:
    """
    Returns
    -------
    Type["BaseMDP"]
        The MDP class corresponding to the name in input.
    """
    try:
        return next(
            filter(lambda c: c.__name__ == mdp_class_name, get_colosseum_mdp_classes())
        )
    except StopIteration:
        raise ModuleNotFoundError(
            f"The MDP class {mdp_class_name} was not found in colosseum. Please check the correct spelling and the result of "
            f"get_colosseum_mdp_classes()"
        )


def get_colosseum_agent_classes(episodic: bool = None) -> List[Type["BaseAgent"]]:
    """
    Returns
    -------
    List[Type["BaseAgent"]]
        All available agent classes in the package.
    """
    if episodic is None:
        return _get_colosseum_agent_classes(True) + _get_colosseum_agent_classes(False)
    if episodic:
        return _get_colosseum_agent_classes(True)
    return _get_colosseum_agent_classes(False)


def _get_colosseum_agent_classes(episodic: bool) -> List[Type["BaseAgent"]]:
    agent_path = "episodic" if episodic else "infinite_horizon"
    kw = "Episodic" if episodic else "Continuous"
    return [
        next(
            filter(
                lambda x: kw in x[0],
                importlib.import_module(
                    agent_file[agent_file.find("colosseum") :].replace(os.sep, ".")[:-3]
                ).__dict__.items(),
            )
        )[1]
        for agent_file in glob(
            f"{os.path.dirname(inspect.getfile(colosseum))}{os.sep}agent{os.sep}agents{os.sep}{agent_path}"
            f"{os.sep}**{os.sep}[a-z]*.py",
            recursive=True,
        )
    ]


def get_agent_class_from_name(agent_class_name: str) -> Type["BaseAgent"]:
    """
    Returns
    -------
    Type["BaseAgent"]
        The agent class corresponding to the name in input.
    """
    return next(
        filter(
            lambda c: c.__name__ == agent_class_name,
            get_colosseum_agent_classes() + config.get_external_agent_classes(),
        )
    )

    try:
        return next(
            filter(
                lambda c: c.__name__ == agent_class_name,
                get_colosseum_agent_classes() + config.get_external_agent_classes(),
            )
        )
    except StopIteration:
        raise ModuleNotFoundError(
            f"The agent class {agent_class_name} was not found in colosseum. The available classes are {get_colosseum_agent_classes() + config.get_external_agent_classes()}"
        )


def ensure_folder(path: str) -> str:
    """
    Returns
    -------
    str
        The path with the os.sep at the end.
    """
    return path if path[-1] == os.sep else (path + os.sep)


def get_dist(dist_name, args):
    if dist_name == "deterministic":
        return deterministic(*args)
    return importlib.import_module(f"scipy.stats").__getattribute__(dist_name)(*args)


class deterministic_gen(rv_continuous):
    def _cdf(self, x):
        return np.where(x < 0, 0.0, 1.0)

    def _stats(self):
        return 0.0, 0.0, 0.0, 0.0

    def _rvs(self, size=None, random_state=None):
        return np.zeros(shape=size)


deterministic = deterministic_gen(name="deterministic")


def state_occurencens_to_counts(occurences: List[int], N: int) -> np.ndarray:
    x = np.zeros(N)
    for s, c in dict(zip(*np.unique(occurences, return_counts=True))).items():
        x[s] = c
    return x


def check_distributions(dists: List[Union[rv_continuous, None]], are_stochastic: bool):
    """
    checks that the distribution given in input respects the necessary conditions.

    Parameters
    ----------
    dists : List[Union[rv_continuous, None]]
        is the list of distributions.
    are_stochastic : bool
        whether the distributions are supposed to be stochastic.
    """
    # You either define all or none of the distribution
    assert dists.count(None) in [0, len(dists)]

    # Double check that the distributions in input matches the stochasticity of the reward parameter
    if dists[0] is not None:
        if are_stochastic:
            assert all(type(dist.dist) != deterministic_gen for dist in dists)
        else:
            assert all(type(dist.dist) == deterministic_gen for dist in dists)


def get_loop(x: Iterable) -> Iterable:
    """
    Returns
    -------
    Iterable
        An iterable that respects the current level of verbosity.
    """
    if config.VERBOSE_LEVEL != 0:
        if type(config.VERBOSE_LEVEL) == int:
            return tqdm(x, desc="Diameter calculation", mininterval=5)
        s = StringIO()
        return tqdm(x, desc="Diameter calculation", file=s, mininterval=5)
    return x
