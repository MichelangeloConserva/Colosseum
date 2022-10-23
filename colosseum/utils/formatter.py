from typing import Union

import numpy as np
import toolz


def clear_agent_mdp_class_name(class_name: str) -> str:
    """
    Returns
    -------
    str
        A nicely formatted version of the class name.
    """
    return (
        class_name.replace("Continuous", "")
        .replace("Episodic", "")
        .replace("QL", "Q-l")
    )


def cleaner(x) -> Union[float, str]:
    """
    Returns
    -------
    Union[float, str]
        A cleaned version of the object given in input that is ready to be transformed to string.
    """
    from colosseum.mdp.simple_grid.base import SimpleGridReward
    from colosseum.mdp.utils import MDPCommunicationClass

    if "numpy" in str(type(x)) and "bool" not in str(type(x)):
        return float(np.round(x, 5))
    if "scipy" in str(type(x)) and "frozen" in str(type(x)):
        return (
            f"{x.dist.name.capitalize()}"
            f"({', '.join(map(str, map(lambda y : float(np.round(y, 2)), x.args)))})"
        )
    if type(x) == float:
        return float(np.round(x, 5))
    if type(x) in [MDPCommunicationClass, SimpleGridReward]:
        return x.name
    return x


def clean_for_storing(inp: Union[dict, list]) -> Union[dict, list]:
    """
    Returns
    -------
    Union[dict, list]
        The list or the values of the dictionary given in input to a form that is ready to be stored.
    """
    if type(inp) == dict:
        return toolz.valmap(cleaner, inp)
    elif type(inp) == list:
        return list(map(cleaner, inp))
    raise NotImplementedError(
        f"'clean_for_storing' not implemented for type {type(inp)}."
    )


def clean_for_file_path(s: str) -> str:
    """
    Returns
    -------
    str
        The string given in input with all symbols that are not allowed for file names removed.
    """
    return (
        s.replace("_", "-")
        .replace(".", "_")
        .replace(" ", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "__")
    )
