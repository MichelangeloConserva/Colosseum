import json
import os
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, NamedTuple, Type

import numpy as np
import pandas as pd
import seaborn as sns
from frozendict import frozendict
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

from colosseum import config
from colosseum.experiment.hyperopt.base import agent_hyperparameters_generator
from colosseum.experiment.hyperopt.hyperopt_measures import get_regret_score
from colosseum.agent.agents.episodic import PSRLEpisodic
from colosseum.agent.agents.infinite_horizon import UCRL2Continuous
from colosseum.utils import clean_for_storing, ensure_folder, make_environment_spec
from colosseum.utils.formatter import clean_for_file_path

sns.set_theme()

if TYPE_CHECKING:
    from colosseum.mdp import BaseMDP


class SingleInstanceHyperparamOptParameters(NamedTuple):
    mdp_class: "BaseMDP"
    mdp_parameters: Dict[str, Any]
    n_cores: int
    n_seed: int
    num_samples: int
    optimization_horizon: int
    max_interaction_s: float
    log_every: int
    verbose: bool

@dataclass()
class HardnessAnalysisParams:
    mdp_base_params: Dict[str, Any]
    sizes: Iterable[int]
    ps: Iterable[float]
    n_seeds: int
    mdp_class_name: str = None
    optimal_agent_optimization: Dict = None

def plot_hardness_analysis(
    varying_parameter_name: str,
    varying_parameter_values: Iterable,
    mdp_class,
    base_mdp_parameters,
    n_seeds: int,
    look_for_cache: bool = True,
    cache_folder: str = config.get_hardness_measures_cache_folder(),
    hyper_opt_hyper_params=None,
    # frozendict(
    #     n_cores=5,
    #     n_seed=2,
    #     num_samples=5,
    #     optimization_horizon=100_000,
    #     max_interaction_s=1 * 60,
    #     log_every=int(100_000 / 3),
    #     verbose=False,
    # ),
    ax=None,
):
    show = False
    if ax is None:
        show = True
        fig, ax = plt.subplots()

    base_mdp_parameters = deepcopy(base_mdp_parameters)

    if hyper_opt_hyper_params is not None:
        hyper_opt_hyper_params = SingleInstanceHyperparamOptParameters(
            mdp_class, base_mdp_parameters, **hyper_opt_hyper_params
        )

    df = get_varying_parameter_df(
        varying_parameter_name,
        varying_parameter_values,
        mdp_class,
        base_mdp_parameters,
        n_seeds,
        look_for_cache=look_for_cache,
        cache_folder=cache_folder,
        hyper_opt_hyper_params=hyper_opt_hyper_params,
    )

    measures = ["diameter", "value_norm", "suboptimal_gaps", "optimal_regret"]
    df.fillna(0, inplace=True)
    for m in measures:
        if np.isclose(df.loc[:, m], df.loc[:, m].max(), atol=1e-3).all():
            df.loc[:, m] = 0.5
    df.loc[:, measures] = (df.loc[:, measures] - df.loc[:, measures].min(0)) / (
        df.loc[:, measures].max(0) - df.loc[:, measures].min(0)
    )
    df.fillna(0.5, inplace=True)

    linewidth = 2.5
    marker_size = 10
    sns.lineplot(
        x=varying_parameter_name,
        y="diameter",
        label="Diameter",
        data=df,
        ax=ax,
        marker="o",
        markersize=marker_size,
        linewidth=linewidth,
    )
    sns.lineplot(
        x=varying_parameter_name,
        y="value_norm",
        label="Environmental value norm",
        data=df,
        ax=ax,
        marker="^",
        markersize=marker_size,
        linewidth=linewidth,
    )
    sns.lineplot(
        x=varying_parameter_name,
        y="suboptimal_gaps",
        label="Sum  of  the  reciprocals \nof  the  sub-optimality  gaps",
        data=df,
        ax=ax,
        marker="D",
        markersize=marker_size,
        linewidth=linewidth,
    )
    if hyper_opt_hyper_params is not None:
        sns.lineplot(
            x=varying_parameter_name,
            y="optimal_regret",
            label="Cumulative regret",
            data=df,
            ax=ax,
            marker="X",
            markersize=marker_size,
            linewidth=linewidth,
        )
    ax.set_ylabel("Normalized measure of hardness")
    ax.set_xlabel(varying_parameter_name.capitalize())
    if show:
        plt.tight_layout()
        plt.show()


def get_varying_parameter_df(
    varying_parameter_name: str,
    varying_parameter_values: List,
    mdp_class,
    base_mdp_parameters,
    n_seeds: int,
    look_for_cache: bool = True,
    cache_folder: str = None,
    hyper_opt_hyper_params: SingleInstanceHyperparamOptParameters = None,
) -> pd.DataFrame:
    base_mdp_parameters["randomize_actions"] = False

    df = pd.DataFrame(
        columns=[
            "MDP",
            varying_parameter_name,
            "diameter",
            "value_norm",
            "suboptimal_gaps",
            "optimal_regret",
            "seed",
        ]
    )
    if cache_folder:
        cache_folder = f"{ensure_folder(cache_folder)}{ensure_folder(mdp_class.__name__)}"
        os.makedirs(cache_folder, exist_ok=True)
    for varying_p in tqdm(
        varying_parameter_values,
        desc=f"{mdp_class.__name__} for varying {varying_parameter_name}.",
    ):
        base_mdp_parameters[varying_parameter_name] = varying_p

        # Compute the hyperparameters for the mdp class and parameters
        if hyper_opt_hyper_params is not None:
            agent_hyper_params_f = f"{cache_folder}optimal_agent_hprs_{varying_parameter_name}_{varying_p}_{clean_for_file_path(str(clean_for_storing(base_mdp_parameters).items()))}.json"
            agent_hyper_params_f = re.sub(r"[^\w\-/_\. ]", "_", agent_hyper_params_f)
            if look_for_cache and os.path.isfile(agent_hyper_params_f):
                with open(agent_hyper_params_f, "r") as f:
                    agent_hyper_params = json.load(f)
            else:
                agent_hyper_params, _ = get_optimal_hyperparameters_for_single_instance(
                    *hyper_opt_hyper_params
                )
                # print(agent_hyper_params)
                with open(agent_hyper_params_f, "w") as f:
                    json.dump(agent_hyper_params, f)
            # print(agent_hyper_params_f)

        for seed in range(n_seeds if mdp_class.does_seed_change_MDP_structure() else 1):
            diameter = compute_hardness_measure(
                mdp_class, dict(seed=seed, **base_mdp_parameters), "diameter", cache_folder, look_for_cache
            )
            value_norm = compute_hardness_measure(
                mdp_class, dict(seed=seed, **base_mdp_parameters), "value_norm", cache_folder, look_for_cache
            )
            suboptimal_gaps = compute_hardness_measure(
                mdp_class, dict(seed=seed, **base_mdp_parameters), "suboptimal_gaps", cache_folder, look_for_cache
            )
            optimal_regret = None
            if hyper_opt_hyper_params is not None:
                optimal_regret = compute_hardness_measure(
                    mdp_class, dict(seed=seed, **base_mdp_parameters),
                    "optimal_regret",
                    cache_folder,
                    look_for_cache,
                    agent_hyper_params,
                    hyper_opt_hyper_params,
                )
            df.loc[len(df)] = [
                mdp_class.__name__,
                varying_p,
                diameter,
                value_norm,
                suboptimal_gaps,
                optimal_regret,
                seed,
            ]
    return df


def compute_hardness_measure(
    mdp_class: Type["BaseMDP"],
    mdp_kwargs : Dict[str, Any],
    measure: str,
    folder: str = None,
    look_for_cache: bool = True,
    agent_hyper_params: Dict[str, Any] = None,
    hyper_opt_hyper_params: SingleInstanceHyperparamOptParameters = None,
):
    """
    returns the measure of hardness.

    Parameters
    ----------
    mdp : BaseMDP
        is the MDP for which the measure of hardness is computed.
    measure : str
        is the name of the measure of hardness.
    folder : str, optional
        is the folder where cached values are looked for and stored. By default, it is set to None.
    look_for_cache : bool, optional
        checks whether to looked for cached values in the folder.
    """
    mdp = mdp_class(**mdp_kwargs, instantiate_mdp=mdp_class.is_episodic())
    measure_f = f"{folder}{measure}_{mdp.hash}.txt"
    if look_for_cache and os.path.isfile(measure_f):
        with open(measure_f, "r") as f:
            measure = float(f.read())
        return measure

    if not mdp_class.is_episodic():
        mdp.instantiate_MDP()
    if measure == "optimal_regret":
        assert agent_hyper_params is not None
        agent_class = PSRLEpisodic if mdp_class.is_episodic() else UCRL2Continuous
        agent = agent_class.get_agent_instance_from_hyperparameters(
            0,
            hyper_opt_hyper_params.optimization_horizon,
            make_environment_spec(mdp),
            agent_hyper_params,
        )
        compute_measure_f = lambda: get_regret_score(
            mdp,
            agent,
            hyper_opt_hyper_params.optimization_horizon,
            hyper_opt_hyper_params.max_interaction_s,
            log_every=hyper_opt_hyper_params.log_every,
            enforce_time_constraint=False,
        )
    else:
        mdp = mdp_class(**mdp_kwargs)
        compute_measure_f = lambda: mdp.get_measure_from_name(measure)

    if folder is None:
        return compute_measure_f()


    measure = compute_measure_f()
    os.makedirs(os.path.dirname(measure_f), exist_ok=True)
    with open(measure_f, "w") as f:
        f.write(str(measure))
    return measure


# def tune_run(agent_next_hyperparams):
#     scores = []
#     for seed in range(n_seed):
#         mdp_parameters["seed"] = seed
#         mdp = mdp_class(**mdp_parameters)
#         agent = agent_class.get_agent_instance_from_hyperparameters(
#             seed,
#             optimization_horizon,
#             make_environment_spec(mdp),
#             agent_next_hyperparams,
#         )
#         scores.append(
#             get_regret_score(
#                 mdp,
#                 agent,
#                 optimization_horizon,
#                 max_interaction_s,
#                 log_every=log_every,
#                 enforce_time_constraint=False,
#             )
#         )
#     # tune.report(regret=np.mean(scores))
#     return agent_next_hyperparams, np.mean(scores)


def run(x):
    (
        agent_next_hyperparams,
        n_seed,
        mdp_parameters,
        mdp_class,
        agent_class,
        optimization_horizon,
        max_interaction_s,
        log_every,
    ) = x

    scores = []
    for seed in range(n_seed):
        mdp_parameters["seed"] = seed
        mdp = mdp_class(**mdp_parameters)
        agent = agent_class.get_agent_instance_from_hyperparameters(
            seed,
            optimization_horizon,
            make_environment_spec(mdp),
            agent_next_hyperparams,
        )
        scores.append(
            get_regret_score(
                mdp,
                agent,
                optimization_horizon,
                max_interaction_s,
                log_every=log_every,
                enforce_time_constraint=False,
            )
        )
    # tune.report(regret=np.mean(scores))
    return agent_next_hyperparams, np.mean(scores)


def get_optimal_hyperparameters_for_single_instance(
    mdp_class,
    mdp_parameters,
    n_cores: int,
    n_seed: int,
    num_samples: int,
    optimization_horizon: int,
    max_interaction_s: float,
    log_every: int,
    verbose: bool,
):

    agent_class = PSRLEpisodic if mdp_class.is_episodic() else UCRL2Continuous
    # ray.init(num_cpus=n_cores, log_to_driver=False)
    # analysis = tune.run(
    #     tune_run,
    #     config=agent_class.get_hyperparameters_search_spaces(),
    #     num_samples=num_samples,
    #     # verbose=verbose,
    # )
    # best_trial = analysis.get_best_trial("regret", mode="min")
    # ray.shutdown()

    gen = agent_hyperparameters_generator(
        42, agent_class.get_hyperparameters_search_spaces()
    )
    if verbose:
        loop = trange(num_samples, desc="Single instance hyperparameters optimization")
    best_score = np.inf
    best_hyprms = None
    rng = random.Random(42)
    with Pool(processes=n_cores) as p:
        for hprms, score in p.imap_unordered(
            run,
            list(
                (
                    next(gen),
                    n_seed,
                    mdp_parameters,
                    mdp_class,
                    agent_class,
                    optimization_horizon,
                    max_interaction_s,
                    log_every,
                )
                for _ in range(num_samples)
            ),
        ):
            if score < best_score:
                best_hyprms = hprms
                best_score = score
            elif np.isclose(score, best_score):
                if rng.random() > 0.5:
                    best_hyprms = hprms
                    best_score = score
            if verbose:
                loop.update(1)
                loop.set_description(f"Current best score: {best_score:.2f}")

    # return best_trial.config, best_trial.last_result["regret"]
    return best_hyprms, best_score
