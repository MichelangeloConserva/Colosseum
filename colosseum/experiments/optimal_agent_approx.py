import os
from multiprocessing import Pool
from tempfile import gettempdir
from typing import Dict, Union, Type, Any, Tuple, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ray import tune
from tqdm import trange

from colosseum.agents.bayes_tools.conjugate_rewards import RewardsConjugateModel
from colosseum.agents.bayes_tools.conjugate_transitions import TransitionsConjugateModel
from colosseum.agents.continuous.ucrl2 import UCRL2Continuous
from colosseum.agents.episodic.psrl import PSRLEpisodic
from colosseum.experiments.experiment import MDPLoop
from colosseum.mdps import EpisodicMDP, ContinuousMDP
from colosseum.utils.acme.in_memory_logger import InMemoryLogger
from colosseum.utils.acme.specs import make_environment_spec
from colosseum.utils.miscellanea import ensure_folder


def run(
    config: Dict[str, Any],
    n_seeds: int,
    mdp_kwargs: Dict[str, Any],
    mdp_class : Union[Type["EpisodicMDP"], Type["ContinuousMDP"]],
    max_time: float,
    T: int,
    verbose: Union[bool, str, None],
    folder: Union[None, str],
    i: int,
) -> Tuple[Dict[str, Any], List[float]]:
    """

    Parameters
    ----------
    config: Dict[str, Any].
        the hyperparameters of the agent.
    n_seeds: int.
        the number of seeds over which the agent/MDP interaction is averaged.
    mdp_kwargs: Dict[str, Any].
        the parameters of the MDP.
    mdp_class : Union[Type["EpisodicMDP"], Type["ContinuousMDP"]].
        the class of the MDP.
    max_time: float.
        the maximum training time for the agent.
    T: int.
        the total number of time steps for the agent/MDP interaction.
    verbose: Union[bool, str, None].
        when it is set to bool it prints directly on the console and when it is set to a string it saves the outputs in
        a file with such name.
    folder: Union[None, str].
        if given, the logs of the agent/MDP interaction will be stored in this folder.
    i: int.
        an integer id to assign to the logging file.

    Returns
    -------
    a tuple containing the hyperparameters of the agent and the cumulative regrets for the different seeds.
    """

    regrets = []
    for seed in range(n_seeds):
        mdp_kwargs["seed"] = seed
        mdp_kwargs["force_single_thread"] = True
        mdp = mdp_class(**mdp_kwargs)
        if issubclass(mdp_class, EpisodicMDP):
            agent = PSRLEpisodic(
                environment_spec=make_environment_spec(mdp),
                seed=seed,
                H=mdp.H,
                r_max=mdp.r_max,
                T=T,
                reward_prior_model=RewardsConjugateModel.N_NIG,
                transitions_prior_model=TransitionsConjugateModel.M_DIR,
                rewards_prior_prms=[config["a"], 1, 1, 1],
                transitions_prior_prms=[config["b"]],
            )
        else:
            agent = UCRL2Continuous(
                environment_spec=make_environment_spec(mdp),
                seed=seed,
                r_max=mdp.r_max,
                T=T,
                alpha_p=config["a"],
                alpha_r=config["b"],
                bound_type_p="bernstein",
            )

        loop = MDPLoop(mdp, agent, logger=InMemoryLogger())
        loop.run(T=T, verbose=verbose, max_time=max_time)
        df_e = pd.DataFrame(loop.logger.data)
        loop.close()
        if folder:
            df_e.to_csv(
                f"{ensure_folder(folder)}_{type(mdp).__name__}_{i}_run_{config}.csv",
                index=False,
            )
        regrets.append(df_e.normalized_cumulative_regret.tolist()[-1])
    return config, regrets


def get_optimal_hyperparams(
    mdp_class : Union[Type["EpisodicMDP"], Type["ContinuousMDP"]],
    mdp_kwargs: Dict[str, Any],
    T : int,
    n_sample_per_cpu_count : int,
    max_time : float,
    n_seeds : int,
    verbose : Union[bool, str, None],
    folder=None,
) -> Dict[str, Any]:
    """

    Parameters
    ----------
    mdp_class : Union[Type["EpisodicMDP"], Type["ContinuousMDP"]].
        the class of the MDP.
    mdp_kwargs: Dict[str, Any].
        the parameters of the MDP.
    T: int.
        the total number of time steps for the agent/MDP interaction.
    n_sample_per_cpu_count : int
        the number of samples in the hyperparameters random search for each cpu core available.
    max_time: float.
        the maximum training time for the agent.
    n_seeds: int.
        the number of seeds over which the agent/MDP interaction is averaged.
    verbose: Union[bool, str, None].
        when it is set to bool it prints directly on the console and when it is set to a string it saves the outputs in
        a file with such name.
    folder: Union[None, str].
        if given, the logs of the agent/MDP interaction will be stored in this folder.

    Returns
    -------
    the best hyperparameters configuration.
    """

    n_cores = os.cpu_count() - 2 # to ensure that the machine does not crush.
    num_samples = n_cores * n_sample_per_cpu_count
    agent_class = (
        PSRLEpisodic if issubclass(mdp_class, EpisodicMDP) else UCRL2Continuous
    )

    mdp = mdp_class(**mdp_kwargs)
    search_space = (
        {
            "a": tune.uniform(
                0.000000000001 * mdp.num_states ** 3, 0.0000005 * mdp.num_states ** 3
            ),
            "b": tune.uniform(
                0.000000000001 * mdp.num_states ** 3, 0.0000005 * mdp.num_states ** 3
            ),
        }
        if agent_class == UCRL2Continuous
        else {
            "a": tune.uniform(
                0.000000000001 * mdp.num_states ** 3, 0.0000005 * mdp.num_states ** 3
            ),
            "b": tune.uniform(0.001, 2),
        }
    )

    hp_samples = []
    while len(hp_samples) < num_samples:
        hp_samples.append(
            (
                {k: v.sample() for k, v in search_space.items()},
                n_seeds,
                mdp_kwargs,
                mdp_class,
                max_time,
                T,
                verbose,
                folder,
                len(hp_samples),
            )
        )

    result = []
    loop = trange(len(hp_samples), desc="Hyperparameter random search")
    ### Single thread version for debugging purposes
    # for args in hp_samples:
    #     result.append(run(*args))
    #     loop.update()
    #     loop.refresh()
    with Pool(processes=n_cores) as p:
        for hps, regrets in p.starmap(run, hp_samples):
            result.append((hps, regrets))
            loop.update()
            loop.refresh()
    return min(result, key=lambda x: np.mean(x[1]))[0]


def get_optimal_regret(
    mdp_class,
    mdp_kwargs,
    T,
    max_time=60 * 2,
    n_seeds=3,
    n_sample_per_cpu_count=4,
    agent_config=None,
    seed=42,
    plot_regret=False,
    verbose=False,
) -> float:
    """

    Parameters
    ----------
    mdp_class : Union[Type["EpisodicMDP"], Type["ContinuousMDP"]].
        the class of the MDP.
    mdp_kwargs: Dict[str, Any].
        the parameters of the MDP.
    T: int.
        the total number of time steps for the agent/MDP interaction.
    max_time: float.
        the maximum training time for the agent.
    n_seeds: int.
        the number of seeds over which the agent/MDP interaction is averaged.
    n_sample_per_cpu_count : int
        the number of samples in the hyperparameters random search for each cpu core available.
    agent_config : Union[None, Dict[str, Any].
        if the hyperparameters are not given, then a hyperparameter random search will be carried out.
    seed : int.
        the seed for the agent/MDP interaction that calculates the cumulative regret using the optimal hyperparameters.
    verbose: Union[bool, str, None].
        when it is set to bool it prints directly on the console and when it is set to a string it saves the outputs in
        a file with such name.
    plot_regret : bool.
        whether to plot the cumulative regret curve of the tuned near-optimal agent.

    Returns
    -------
    the cumulative regret of the tuned near-optimal agent.
    """

    if agent_config is None:
        agent_config = get_optimal_hyperparams(
            mdp_class,
            mdp_kwargs,
            T,
            n_sample_per_cpu_count=n_sample_per_cpu_count,
            max_time=max_time,
            n_seeds=n_seeds,
            verbose=verbose,
        )
    mdp_kwargs["seed"] = seed
    mdp = mdp_class(**mdp_kwargs)
    if issubclass(mdp_class, EpisodicMDP):
        agent = PSRLEpisodic(
            environment_spec=make_environment_spec(mdp),
            seed=seed,
            H=mdp.H,
            r_max=mdp.r_max,
            T=T,
            reward_prior_model=RewardsConjugateModel.N_NIG,
            transitions_prior_model=TransitionsConjugateModel.M_DIR,
            rewards_prior_prms=[agent_config["a"], 1, 1, 1],
            transitions_prior_prms=[agent_config["b"]],
        )
    else:
        agent = UCRL2Continuous(
            environment_spec=make_environment_spec(mdp),
            seed=seed,
            r_max=mdp.r_max,
            T=T,
            alpha_p=agent_config["a"],
            alpha_r=agent_config["b"],
            bound_type_p="bernstein",
        )
    loop = MDPLoop(mdp, agent, logger=InMemoryLogger())
    loop.run(T, verbose=verbose, max_time=2 * max_time)
    if plot_regret:
        loop.plot(["normalized_cumulative_regret", "random_normalized_cumulative_regret"])
    df_e = pd.DataFrame(loop.logger.data)
    loop.close()
    return df_e.normalized_cumulative_regret.tolist()[-1]
