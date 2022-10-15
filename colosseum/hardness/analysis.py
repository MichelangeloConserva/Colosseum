import os
from copy import deepcopy
from dataclasses import dataclass
from multiprocessing import Pool
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Type,
    Tuple,
    Callable,
    Union,
    Collection,
)

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from colosseum import config
from colosseum.agent.agents.episodic import PSRLEpisodic
from colosseum.agent.agents.infinite_horizon import UCRL2Continuous
from colosseum.mdp.base import BaseMDP

sns.set_theme()

if TYPE_CHECKING:
    from colosseum.agent.agents.base import BaseAgent


@dataclass()
class SingleInstanceHyperOptConfig:
    mdp_class: Type["BaseMDP"]
    """The class of the MDP with respect to whom we are performing the parameters optimization."""
    mdp_parameters: Dict[str, Any]
    """The dictionary containing the parameters for the MDP class with respect to whom we are performing the parameters optimization."""
    n_seed: int
    """The number of seed the agent/MDP interaction is repeated."""
    optimization_horizon: int
    """The optimization horizon of the agent/MDP interaction."""
    num_samples: int
    """The number of samples of the agent parameters."""
    max_interaction_s: float
    """The maximum amount of seconds allocated to training the agent."""
    log_every: int
    """The length of the interval between logging of the performance indicators."""
    episodic_near_optimal_agent_class: Type["BaseAgent"] = PSRLEpisodic
    """The class of the near optimal agent for the episodic setting. By default, it is ``PSRLEpisodic``."""
    continuous_near_optimal_agent_class: Type["BaseAgent"] = UCRL2Continuous
    """The class of the near optimal agent for the continuous setting. By default, it is ``UCRL2Continuous``."""


@dataclass()
class HardnessAnalysisParams:
    mdp_class: Type["BaseMDP"]
    """The class of the MDP whose hardness we are studying."""
    varying_params_name: str
    """The name of the parameter being varied."""
    varying_params_values: Iterable
    """The values of the parameter being varied."""
    fixed_params: Dict[str, Any]
    """The dictionary containing the names and values for the parameters being kept fixed."""
    n_seeds_mdp: int
    """The number of seeds used when instantiating the MDP."""
    hardness_measures: Collection[Union[str, Callable[[Type["BaseMDP"]], float]]] = (
        "diameter",
        "value_norm",
    )
    """An iterable containing either the code name of an available measure of hardness or a function that takes an MDP
    object as input and returns a value."""
    near_optimal_agent_hyperopt_config: SingleInstanceHyperOptConfig = None
    """The parameters optimization configuration for the near optimal agent. By default, it is None, which means
    that the regret of the near optimal agent with tuned parameters is not computed and used as proxy for a 
    complete measure of hardness."""
    varying_params_name_clean: str = None
    """The name of the parameter being varied in a clean format."""
    retrieve_from_cache: bool = True
    """If ture, the ``config.get_hardness_measures_cache_folder()`` is searched for a cached value of the measure.
    By default, it is True."""

    @property
    def clean_varying_prm_name(self) -> str:
        """
        Returns
        -------
        str
            A nicely formatted name for the varying parameter.
        """
        if self.varying_params_name_clean is None:
            return self.varying_params_name
        return self.varying_params_name_clean


def run_scenario_analysis(
    hap: HardnessAnalysisParams,
    ax=None,
):
    """
    runs a hardness analysis scenario.

    Parameters
    ----------
    hap : HardnessAnalysisParams
        The hardness analysis scenario to run.
    ax : plt.Axes
        The ax object where the plot will be put. By default, a new axis is created.
    """

    show = ax is None
    if ax is None:
        fig, ax = plt.subplots()

    dfs = get_varying_parameter_dfs(hap, normalize_measures=True)

    for measure_name, df in dfs.items():
        sns.lineplot(
            x=hap.clean_varying_prm_name,
            y=measure_name,
            data=df,
            ax=ax,
            label=measure_name,
        )
    plt.ylabel("Hardness measure value")
    plt.legend()

    if show:
        plt.show()


def get_varying_parameter_dfs(
    hap: HardnessAnalysisParams,
    normalize_measures: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    computes the hardness measures for the scenarios and stored them in a `pd.DataFrame`s.

    Parameters
    ----------
    hap : HardnessAnalysisParams
        The hardness analysis scenario to run.
    normalize_measures : bool
        If True, the values of the hardness measures are normalized.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary that associated the name of a hardness measure to its corresponding pd.DataFrame.
    """

    if hap.near_optimal_agent_hyperopt_config is not None:
        raise NotImplementedError(
            "The computation of the regret of the near optimal agent as a proxy for a complete measure of hardness is "
            "being refactored at the moment."
        )

    measure_results = dict()
    if config.get_available_cores() > 1:
        inputs = [
            (
                hap.mdp_class,
                hap.fixed_params,
                hap.varying_params_name,
                varying_value,
                seed,
                measure,
                hap.retrieve_from_cache,
            )
            for varying_value in hap.varying_params_values
            for measure in hap.hardness_measures
            for seed in range(hap.n_seeds_mdp)
        ]

        with Pool(processes=config.get_available_cores()) as p:
            for measure_name, varying_value, seed, measure_value in p.starmap_async(
                _compute_hardness_measure, inputs
            ).get():
                _add_result(
                    measure_results, measure_name, varying_value, seed, measure_value
                )
    else:
        for seed in range(hap.n_seeds_mdp):
            for measure in hap.hardness_measures:
                for varying_value in hap.varying_params_values:
                    out = compute_hardness_measure_for_varying_prm(
                        hap.mdp_class,
                        hap.fixed_params,
                        hap.varying_params_name,
                        varying_value,
                        seed,
                        measure,
                        force_single_core=True,
                        retrieve_from_cache=hap.retrieve_from_cache,
                        return_n_states=hap.varying_params_name == "size",
                    )
                    if hap.varying_params_name == "size":
                        measure_name, measure_value, n_states = out
                    else:
                        measure_name, measure_value = out

                    _add_result(
                        measure_results,
                        measure_name,
                        varying_value
                        if hap.varying_params_name != "size"
                        else n_states,
                        seed,
                        measure_value,
                    )

    for measure_name in measure_results:
        # Create a Pandas DataFrame
        df = pd.DataFrame.from_dict(measure_results[measure_name])

        # Normalize the values if required
        if normalize_measures:
            min_value = df.loc[:, measure_name].min()
            max_value = df.loc[:, measure_name].max()
            if max_value > min_value + 1e-4:
                df.loc[:, measure_name] = (df.loc[:, measure_name] - min_value) / (
                    max_value - min_value
                )
            else:
                df.loc[:, measure_name] = 0.5

        # Clean the varying parameter name
        df = df.rename(columns={"Varying value": hap.clean_varying_prm_name}).set_index(
            hap.clean_varying_prm_name
        )

        measure_results[measure_name] = df

    return measure_results


def _compute_hardness_measure(
    mdp_class,
    fixed_params,
    varying_params_name,
    varying_value,
    seed,
    measure,
    retrieve_from_cache,
    return_n_states: bool = False,
):
    measure_name, measure_value = compute_hardness_measure_for_varying_prm(
        mdp_class,
        fixed_params,
        varying_params_name,
        varying_value,
        seed,
        measure,
        force_single_core=True,
        retrieve_from_cache=retrieve_from_cache,
    )
    return measure_name, varying_value, seed, measure_value


def compute_hardness_measure_for_varying_prm(
    mdp_class: Type["BaseMDP"],
    fixed_params: Dict[str, Any],
    varying_params_name: str,
    varying_value: Any,
    seed: int,
    measure: Union[str, Callable[[BaseMDP], float]],
    force_single_core: bool = False,
    retrieve_from_cache: bool = True,
    folder: str = None,
    return_n_states: bool = False,
) -> Tuple[str, float, int]:
    """
    computes the hardness measure for varying values of the parameter.

    Parameters
    ----------
    mdp_class : Type["BaseMDP"]
        The MDP class for which the measures will be computed.
    fixed_params : Dict[str, Any]
        The parameters of the MDP that are being kept fixed.
    varying_params_name : str
        The name of the varying parameter.
    varying_value : Any
        The value of the parameter that is varying.
    seed : int
        The random seed.
    measure : Union[str, Callable[[BaseMDP], float]]
        The measure to be computed. It can be given as a function from MDP instances to float or as a string. If given
        as a string, it will be looked for in the ones available in the package
    force_single_core : bool
        If True, the computation of the measure is forced to use only a single core. Note that this is not enforced when
        the measure is given as a function. By default, single processing is not enforced.
    retrieve_from_cache : bool
        If True, the function will look for cached values of the measure. Note that this also hold if the measure is
        given as a function.
    folder : str
        The folder where cached values are looked for. By default, it is the `config.get_hardness_measures_cache_folder()`.
    return_n_states : bool
        If True, the number of states is returned.

    Returns
    -------
    str
        The nicely formatted name of the measure.
    float
        The value of the measure.
    int, optional
        The number of states.
    """

    # Instantiate the MDP parameters
    mdp_kwargs = deepcopy(fixed_params)
    mdp_kwargs["seed"] = seed
    mdp_kwargs[varying_params_name] = varying_value

    return compute_hardness_measure(
        mdp_class,
        mdp_kwargs,
        measure,
        force_single_core,
        retrieve_from_cache,
        folder,
        True,
        return_n_states,
    )


def compute_hardness_measure(
    mdp_class: Type["BaseMDP"],
    mdp_params: Dict[str, Any],
    measure: Union[str, Callable[[BaseMDP], float]],
    force_single_core: bool = False,
    retrieve_from_cache: bool = True,
    folder: str = None,
    return_measure_name: bool = False,
    return_n_states: bool = False,
) -> Union[float, Tuple[str, float], Tuple[float, int], Tuple[str, float, int]]:
    """

    Parameters
    ----------
    mdp_class : Type["BaseMDP"]
        The MDP class for which the measures will be computed.
    mdp_params : Dict[str, Any]
        The parameters for the MDP.
    measure : Union[str, Callable[[BaseMDP], float]]
        The measure to be computed. It can be given as a function from MDP instances to float or as a string. If given
        as a string, it will be looked for in the ones available in the package
    force_single_core : bool
        If True, the computation of the measure is forced to use only a single core. Note that this is not enforced when
        the measure is given as a function. By default, single processing is not enforced.
    retrieve_from_cache : bool
        If True, the function will look for cached values of the measure. Note that this also hold if the measure is
        given as a function.
    folder : str
        The folder where cached values are looked for. By default, it is the `config.get_hardness_measures_cache_folder()`.
    return_measure_name : bool
        If True, a nicely formatted name for the measure is returned.
    return_n_states : bool
        If True, the number of states is returned.

    Returns
    -------
    str, optional
        The nicely formatted name of the measure.
    float
        The value of the measure.
    int, optional
        The number of states.
    """

    # Obtain the name of the measure and the function to compute it
    measure_name, measure_f = _process_measure(measure)

    # Check if the measure has already been computed
    if retrieve_from_cache:
        mdp_shell = mdp_class(
            **mdp_params, instantiate_mdp=False, exclude_horizon_from_parameters=True
        )
        if folder is None:
            folder = (
                config.get_hardness_measures_cache_folder()
                + mdp_class.__name__
                + os.sep
            )

        measure_file_path = f"{folder}{measure_name}_{mdp_shell.hash}.txt"
        if os.path.isfile(measure_file_path):
            with open(measure_file_path, "r") as f:
                measure_value = float(f.read())

            out = [measure_value]
            if return_measure_name:
                out.insert(0, measure_name)
            if return_n_states:
                mdp_shell.instantiate_MDP()
                out.append(mdp_shell.n_states)
            return out

    # Possible forcing the computation to avoid using multiple cores
    if force_single_core and config.get_available_cores() > 1:
        available_cores = config.get_available_cores()
        config.disable_multiprocessing()
        mdp = mdp_class(**mdp_params)
        measure_value = measure_f(mdp)
        config.set_available_cores(available_cores)
    else:
        mdp = mdp_class(**mdp_params)
        measure_value = measure_f(mdp)

    # Caching the value of the measure (only in case we were supposed to look for it in the first place)
    if retrieve_from_cache:
        os.makedirs(os.path.dirname(measure_file_path), exist_ok=True)
        with open(measure_file_path, "w") as f:
            f.write(str(measure_value))

    out = [measure_value]
    if return_measure_name:
        out.insert(0, measure_name)
    if return_n_states:
        out.append(mdp.n_states)
    return out


def _process_measure(
    measure: Union[str, Callable[[BaseMDP], float]]
) -> Tuple[str, Callable[[BaseMDP], float]]:
    if type(measure) == str:
        measure_name = measure
        if measure_name not in BaseMDP.get_available_hardness_measures():
            raise ValueError(
                f"{measure} is not a valid hardness measure, the available ones are: "
                f"{BaseMDP.get_available_hardness_measures()}"
            )
        return measure, lambda mdp: mdp.get_measure_from_name(measure)
    elif callable(measure):
        return measure.__name__, measure
    else:
        raise ValueError(
            f"The measure should either be a string or a Callable, {type(measure)} received."
        )


def _add_result(measure_results, measure_name, varying_value, seed, measure_value):
    measure_name = measure_name.capitalize().replace("_", " ")

    if measure_name not in measure_results:
        measure_results[measure_name] = {
            "Varying value": [],
            measure_name: [],
            "Seed": [],
        }
    measure_results[measure_name]["Varying value"].append(varying_value)
    measure_results[measure_name]["Seed"].append(seed)
    measure_results[measure_name][measure_name].append(measure_value)
