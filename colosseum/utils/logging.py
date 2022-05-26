from typing import Union

import numpy as np
import toolz


def make_bold(string: str) -> str:
    return "\033[1m" + string + "\033[0m"


def cleaner(x) -> Union[float, str]:
    from colosseum.mdps.mdp_classification import MDPClass
    from colosseum.mdps.simple_grid import SimpleGridReward

    if "numpy" in str(type(x)):
        return float(np.round(x, 5))
    if "rv_frozen" in str(type(x)):
        return (
            f"{x.dist.name.capitalize()}"
            f"({', '.join(map(str, map(lambda y : float(np.round(y, 2)), x.args)))})"
        )
    if type(x) == float:
        return float(np.round(x, 5))
    if type(x) in [MDPClass, SimpleGridReward]:
        return x.name
    return x


def clean_for_storing(inp: Union[dict, list]) -> Union[dict, list]:
    if type(inp) == dict:
        return toolz.valmap(cleaner, inp)
    elif type(inp) == list:
        return list(map(cleaner, inp))
    raise NotImplementedError(
        f"'clean_for_storing' not implemented for type {type(inp)}."
    )


def enumerate_options(options):
    print("\n".join(f"{i}) {x}" for i, x in enumerate(options)))


def extract_option(options):
    options = list(options)
    enumerate_options(options)
    option = input(
        "Please select between the following options or "
        "enter any other character to return to the previous menu."
    )
    try:
        option = int(option)
        assert option in range(len(list(options)))
        return option
    except:
        return -1
