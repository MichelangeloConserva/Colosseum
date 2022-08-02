import os
from glob import glob
from typing import TYPE_CHECKING, Any, Dict, Type, Union

import yaml

from colosseum.utils import cleaner

if TYPE_CHECKING:
    from colosseum.mdp import BaseMDP


def find_hardness_report_file(
    mdp: "BaseMDP", hardness_reports_folder="hardness_reports"
) -> Union[str, None]:
    """
    tries to find a previously calculated hardness report for an MDP instance.

    Parameters
    ----------
    mdp : MDP
        the MDP instance for which it will be looking for the hardness report.
    hardness_reports_folder : str
        the folder in which the hardness reports are stored
    """
    seed_report = glob(f"{hardness_reports_folder}{os.sep}{type(mdp).__name__}_*.yml")
    for existing_report in seed_report:
        if type(mdp).__name__ not in existing_report:
            continue
        with open(existing_report, "r") as f:
            report = yaml.load(f, yaml.Loader)
        same_mdp = True
        for k, v in report["MDP parameters"].items():
            if not same_mdp:
                break
            same_mdp = cleaner(getattr(mdp, k)) == v
        for k, v in report["MDP graph metrics"].items():
            if not same_mdp:
                break
            same_mdp = mdp.graph_metrics[k] == v
        if same_mdp:
            return existing_report
    return None


def get_average_measure_values(
    mdp_class: Type["BaseMDP"],
    mdp_kwargs: Dict[str, Any],
    n_seeds: int = None,
    measure_names_max_values: Dict[str, float] = None,
) -> Union[None, Dict[str, float]]:
    """
    returns the values of the measures given in inputs averaged across several seed if necessary. Note that this is only
    necessary when the MDP structure changes for different seeds. If any of the measure produces a value that is higher
    than the limit, the function returns None.

    Parameters
    ----------
    mdp_class : Type["BaseMDP"]
        is the MDP class for which the measures will be computed.
    mdp_kwargs : Dict[str, Any]
        is the parameters that are used to instantiate the MDP.
    n_seeds : int, optional
        is the number of seeds for the average. By default, it is set to five when necessary.
    measure_names_max_values : Dict[str, float], optional
        is the dictionary whose keys are the names of the measure of hardness and the values are the maximum value that
        we allow them to be. By default, it is set to 'dict(diameter=200, value_norm=5.0)'.

    """
    if mdp_class.does_seed_change_MDP_structure():
        n_seeds = 5 if n_seeds is None else n_seeds
        assert n_seeds > 1
    else:
        n_seeds = 1 if n_seeds is None else n_seeds
        assert n_seeds == 1

    if measure_names_max_values is None:
        measure_names_max_values = dict(diameter=200, value_norm=5.0)
    average_values = dict(
        zip(measure_names_max_values.keys(), [0] * len(measure_names_max_values))
    )

    for seed in range(n_seeds):
        mdp = mdp_class(seed=seed, **mdp_kwargs)
        for m_name, m_max_value in measure_names_max_values.items():
            m_value = mdp.get_measure_from_name(m_name)
            if m_value is None or m_value > m_max_value:
                return None
            average_values[m_name] += m_value / n_seeds
    return average_values
