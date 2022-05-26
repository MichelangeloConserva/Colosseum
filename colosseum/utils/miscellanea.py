import importlib
import inspect
import os
import random
from glob import glob
from io import StringIO
from typing import TYPE_CHECKING, Iterable, List, Type

import numba
import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from colosseum.agents.base import Agent
    from colosseum.mdps import MDP


def clear_th(x):
    return x.replace("Continuous", "").replace("Episodic", "").replace("QL", "Q-l")


def ensure_folder(x):
    return x + (os.sep if x[-1] != os.sep else "")


@numba.njit()
def argmax_2D(a):
    X = np.zeros_like(a)
    for s in range(len(a)):
        i = random.choice(np.where(a[s] == a[s].max())[0])
        X[s, i] = 1
    return X


# @numba.njit()
def argmax_3D(a):
    X = np.zeros_like(a)
    for h in range(len(a)):
        for s in range(a.shape[1]):
            i = random.choice(np.where(a[h, s] == a[h, s].max())[0])
            # i = np.where(a[h, s] == a[h, s].max())[0][0]
            X[h, s, i] = 1
    return X


def get_loop(x: Iterable, verbose: bool):
    if verbose:
        if type(verbose) == bool:
            return tqdm(x, desc="Diameter calculation", mininterval=5)
        s = StringIO()
        return tqdm(x, desc="Diameter calculation", file=s, mininterval=5)
    return x


def get_all_mdp_classes() -> List[Type["MDP"]]:
    """
    returns all the MDP classes available in Colosseum.
    """
    import colosseum

    return [
        importlib.import_module(
            mdp_file[mdp_file.find("colosseum") : mdp_file.find("mdp.py") - 1].replace(
                os.sep, "."
            )
        ).MDP
        for mdp_file in glob(
            f"{os.path.dirname(inspect.getfile(colosseum))}{os.sep}**{os.sep}mdp.py",
            recursive=True,
        )
    ]


def get_all_agent_classes() -> List[Type["Agent"]]:
    """
    returns all the agent classes available in Colosseum.
    """
    import colosseum

    return [
        importlib.import_module(
            mdp_file[
                mdp_file.find("colosseum") : mdp_file.find("agent.py") - 1
            ].replace(os.sep, ".")
        ).Agent
        for mdp_file in glob(
            f"{os.path.dirname(inspect.getfile(colosseum))}{os.sep}**{os.sep}agent.py",
            recursive=True,
        )
    ]


def normalize(l):
    ll = np.array(l)
    return (ll - ll.min()) / (ll.max() - ll.min())


def get_mdp_class_from_name(mdp_name):
    return next(x for x in get_all_mdp_classes() if x.__name__ == mdp_name)


def get_n_seeds_of_experiment(exp_fold):
    ff = glob(ensure_folder(exp_fold) + "logs" + os.sep + "*")[0]
    return len(glob(ensure_folder(ff) + "seed*csv"))
