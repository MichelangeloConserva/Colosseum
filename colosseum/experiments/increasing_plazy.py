import os
from typing import Union, TYPE_CHECKING, Type, Dict, Any

import numpy as np
import yaml
from tqdm import tqdm

from colosseum.experiments.optimal_agent_approx import get_optimal_hyperparams
from colosseum.experiments.utils import calculate_values, instantiate
from colosseum.experiments.visualisation import plot_results_hardness_analysis
from colosseum.mdps.simple_grid import SimpleGridReward

if TYPE_CHECKING:
    from colosseum.agents.base import Agent
    from colosseum.mdps import ContinuousMDP, EpisodicMDP, NodeType

def moving_plazy(
    prandom : np.ndarray,
    n_seeds : int,
    mdp_class : Union[Type["EpisodicMDP"], Type["ContinuousMDP"]],
    mdp_kwargs : Dict[str, Any],
    approximate_regret : bool,
    from_strach=False,
    save_fig: str = None,
    save_folder: Union[str, None] = f"hardness_analysis{os.sep}data{os.sep}",
):
    """
    Parameters
    ----------
    prandom : np.ndarray.
        the values for the p_lazy parameter.
    n_seeds : int.
        the number of seed to be used to calculate the average measures of hardness for each p_lazy value.
    mdp_class : Union[Type["EpisodicMDP"], Type["ContinuousMDP"]],
        the class of the MDP to be investigated.
    mdp_kwargs : Dict[str, Any].
        additional parameters that the MDP class may require. For example, "size".
    approximate_regret : bool.
        whether to calculate the cumulative regret of the near-optimal tuned agent.
    from_strach : bool, optional.
        whether to calculate the measures or to look for cached results in the save_folder.
    save_fig : str, optional.
        if a string is given, the result of the investigation will be stored as a file. Please do provide the file extension (e.g. png or svg).
    save_folder : str, optional
        the folder where the cached values will be looked into or where the calculated values will be stored.
    """


    N = len(prandom)
    diam_path = f"{save_folder}{mdp_class.__name__}_" f"plazy_" f"diameter_values.npy"
    vnorm_path = f"{save_folder}{mdp_class.__name__}_" f"plazy_" f"valuenorm_values.npy"
    gaps_path = f"{save_folder}{mdp_class.__name__}_" f"plazy_" f"gaps_values.npy"
    cumulative_regret_path = (
        f"{save_folder}{mdp_class.__name__}_" f"plazy_" f"cumulativeregret_values.npy"
    )
    load_d = load_v = load_g = load_r = not from_strach
    (
        diameter_values,
        valuenorm_values,
        gaps_values,
        cum_reg_values,
        d_loaded,
        v_loaded,
        g_loaded,
        r_loaded,
    ) = instantiate(
        diam_path,
        vnorm_path,
        gaps_path,
        cumulative_regret_path,
        N,
        n_seeds,
        load_d,
        load_v,
        load_g,
        load_r,
    )

    for i, prand in enumerate(tqdm(prandom)):
        mdp_kwargs.update(
            dict(
                seed=0,
                randomize_actions=True,
                make_reward_stochastic=True,
                lazy=prand,
                reward_type=SimpleGridReward.AND,
                p_frozen=0.85,
            )
        )
        if approximate_regret and not r_loaded:
            # Calculating the best hyperparameter for this MDP instance
            optimal_config_path = (
                f"{save_folder}{mdp_class.__name__}_"
                f"plazy_"
                f"optimal_config_{prand}_plazy.yml"
            )
            if os.path.isfile(optimal_config_path):
                with open(optimal_config_path, "r") as f:
                    config = yaml.load(f, yaml.Loader)
            else:
                config = get_optimal_hyperparams(
                    mdp_class,
                    mdp_kwargs,
                    T=200_000,
                    n_sample_per_cpu_count=4,
                    max_time=60 * 2,
                    n_seeds=3,
                    verbose=f"temp_multiprocess{os.sep}_{prand}_{i}_tmp{os.sep}",
                )
                with open(optimal_config_path, "w") as f:
                    yaml.dump(config, f)

        for seed in range(n_seeds):
            mdp_kwargs["seed"] = seed
            mdp = mdp_class(**mdp_kwargs)
            calculate_values(
                mdp,
                diameter_values,
                valuenorm_values,
                gaps_values,
                cum_reg_values,
                i,
                seed,
                d_loaded,
                v_loaded,
                g_loaded,
                r_loaded,
                approximate_regret,
                mdp_class,
                mdp_kwargs,
                config if approximate_regret and not r_loaded else None,
            )
    if save_folder is not None:
        np.save(diam_path, diameter_values)
        np.save(vnorm_path, valuenorm_values)
        np.save(gaps_path, gaps_values)
        if approximate_regret:
            np.save(cumulative_regret_path, cum_reg_values)

    plot_results_hardness_analysis(
        mdp,
        None,
        n_seeds,
        diameter_values,
        valuenorm_values,
        gaps_values,
        cum_reg_values,
        prandom,
        "Probability of lazy action",
        approximate_regret,
        save_folder=save_folder,
        save_fig=save_fig,
    )
    return (
        diameter_values,
        valuenorm_values,
        gaps_values,
        (cum_reg_values if approximate_regret else None),
    )
