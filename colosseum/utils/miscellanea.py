import cProfile
import importlib
import inspect
import os
from glob import glob
from io import StringIO
from typing import TYPE_CHECKING, Iterable, List, Type, Union, Any

import dm_env
import numpy as np
from scipy.stats import rv_continuous
from tqdm import tqdm

import colosseum.config as config

if TYPE_CHECKING:
    from colosseum.mdp import BaseMDP


def get_empty_ts(state : Any) -> dm_env.TimeStep:
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
    if episodic is None:
        return _get_colosseum_mdp_classes() + _get_colosseum_mdp_classes(False)
    if episodic:
        return _get_colosseum_mdp_classes()
    return _get_colosseum_mdp_classes(False)


def _get_colosseum_mdp_classes(episodic=True):
    import colosseum

    mdp_path = "finite_horizon" if episodic else "infinite_horizon"
    return [
        importlib.import_module(
            mdp_file[mdp_file.find("colosseum") : mdp_file.find(mdp_path) - 1].replace(
                os.sep, "."
            )
            + "."
            + mdp_path
        ).MDPClass
        for mdp_file in glob(
            f"{os.path.dirname(inspect.getfile(colosseum))}{os.sep}mdp{os.sep}**{os.sep}{mdp_path}.py",
            recursive=True,
        )
    ]


def get_mdp_class_from_name(mdp_class_name: str):
    try:
        return next(
            filter(lambda c: c.__name__ == mdp_class_name, get_colosseum_mdp_classes())
        )
    except StopIteration:
        raise ModuleNotFoundError(
            f"The {mdp_class_name} was not found in colosseum. Please check the correct spelling and the result of "
            f"get_colosseum_mdp_classes()"
        )


def get_colosseum_agent_classes(episodic=None):
    from colosseum.agent.agents.episodic import EPISODIC_AGENT_CLASSES
    from colosseum.agent.agents.infinite_horizon import (
        INFINITE_HORIZON_AGENT_CLASSES,
    )

    if episodic is None:
        return EPISODIC_AGENT_CLASSES + INFINITE_HORIZON_AGENT_CLASSES
    if episodic:
        return EPISODIC_AGENT_CLASSES
    return INFINITE_HORIZON_AGENT_CLASSES


def _get_colosseum_agent_classes(episodic=True):
    from colosseum.agent.agents.episodic import EPISODIC_AGENT_CLASSES
    from colosseum.agent.agents.infinite_horizon import (
        INFINITE_HORIZON_AGENT_CLASSES,
    )

    if episodic:
        return EPISODIC_AGENT_CLASSES
    return INFINITE_HORIZON_AGENT_CLASSES


def get_agent_class_from_name(agent_class_name: str):
    try:
        return next(
            filter(
                lambda c: c.__name__ == agent_class_name,
                get_colosseum_agent_classes() + config.get_external_agent_classes(),
            )
        )
    except StopIteration:
        raise ModuleNotFoundError(
            f"The {agent_class_name} was not found in colosseum. Please check the correct spelling and the result of "
            f"get_colosseum_agent_classes(). If you have implemented the {agent_class_name}, please make sure to update"
            f"the lists of agents in colosseum/agent/agents/episodic/__init__.py and "
            f"colosseum/agent/agents/continuous/__init__.py"
        )


def ensure_folder(path: str) -> str:
    return path if path[-1] == os.sep else (path + os.sep)


def get_dist(dist_name, args):
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


def get_loop(x: Iterable):
    """
    returns an iterable that respects the current level of verbosity.
    """
    if config.VERBOSE_LEVEL != 0:
        if type(config.VERBOSE_LEVEL) == int:
            return tqdm(x, desc="Diameter calculation", mininterval=5)
        s = StringIO()
        return tqdm(x, desc="Diameter calculation", file=s, mininterval=5)
    return x
