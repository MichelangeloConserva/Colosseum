import os
import shutil
import warnings
from glob import glob
from shutil import copytree, rmtree
from tempfile import gettempdir
from typing import Any, Dict, List, Tuple, Type

import gin
import numpy as np
import pandas as pd
import yaml

from colosseum.agents.base import Agent
from colosseum.agents.bayes_tools.conjugate_rewards import RewardsConjugateModel
from colosseum.agents.bayes_tools.conjugate_transitions import TransitionsConjugateModel
from colosseum.agents.episodic.psrl import PSRLEpisodic
from colosseum.experiments.experiment import Experiment, MDPLoop, experiment_label
from colosseum.experiments.hardness_reports import mean_metrics_report
from colosseum.experiments.optimal_agent_approx import get_optimal_regret
from colosseum.mdps import MDP
from colosseum.utils.acme.in_memory_logger import InMemoryLogger
from colosseum.utils.acme.specs import make_environment_spec
from colosseum.utils.miscellanea import ensure_folder


def group_experiments_by(experiments_prms: List, by_mdp: bool) -> List[Tuple[Any, Any]]:
    experiments_dict = dict()
    for mdp_class, mdp_prms, agent_class, agent_prms in experiments_prms:

        cls_prms1 = mdp_class, mdp_prms
        cls_prms2 = agent_class, agent_prms
        if not by_mdp:
            cls_prms1, cls_prms2 = cls_prms2, cls_prms1

        if cls_prms1 in experiments_dict:
            experiments_dict[cls_prms1].add(cls_prms2)
        else:
            experiments_dict[cls_prms1] = {cls_prms2}

    return sorted(experiments_dict.items(), key=lambda x: x[0][0].__name__)


def get_config_files(experiment: str) -> List[str]:
    """
    returns the gin config files of a given experiment.
    """
    gin_config_files_paths = []
    for mdp_config_file in glob(f"{ensure_folder(experiment)}mdp_configs{os.sep}*"):
        gin_config_files_paths.append(mdp_config_file)
    for agent_config_file in glob(f"{ensure_folder(experiment)}agent_configs{os.sep}*"):
        gin_config_files_paths.append(agent_config_file)
    return gin_config_files_paths


def retrieve_parameters(
    mdp_name: str,
    mdp_prms: str,
    agent_name: str,
    agent_prms: str,
    config_files: List[str],
) -> Tuple[List[str], List[str]]:
    """

    Parameters
    ----------
    mdp_name: str.
        the name of the MDP family.
    mdp_prms: str.
        the string identifying the configuration of the MDP, e.g. 'prms_0'.
    agent_name: str.
        the name of the agent class.
    agent_prms: str.
        the string identifying the configuration of the agent, e.g. 'prms_0'.
    config_files: List[str].
        the gin configuration files paths.

    Returns
    -------
    the gin configuration of the MDP and the agent.
    """
    with open(next(filter(lambda x: mdp_name in x, config_files)), "r") as f:
        mdp_parameters = list(
            map(
                lambda x: x.replace(mdp_prms + "/", "")
                .replace("\n", "")
                .replace(mdp_name + ".", ""),
                filter(lambda x: mdp_prms in x, f.readlines()),
            )
        )
    with open(next(filter(lambda x: agent_name in x, config_files)), "r") as f:
        agent_parameters = list(
            map(
                lambda x: x.replace(agent_prms + "/", "")
                .replace("\n", "")
                .replace(agent_name + ".", ""),
                filter(lambda x: agent_prms in x, f.readlines()),
            )
        )
    return mdp_parameters, agent_parameters


def get_log_folder(
    mdp_name: str,
    mdp_prms: str,
    agent_name: str,
    agent_prms: str,
    logs_folders : List[str]) -> str:
    """

    Parameters
    ----------
    mdp_name: str.
        the name of the MDP family.
    mdp_prms: str.
        the string identifying the configuration of the MDP, e.g. 'prms_0'.
    agent_name: str.
        the name of the agent class.
    agent_prms: str.
        the string identifying the configuration of the agent, e.g. 'prms_0'.
    logs_folders : List[str]
        the folders corresponding to all the agent/MDP interaction in an experiment.
    Returns
    -------
        the logging folder corresponding the mdp and agent given in input.
    """
    return next(
        filter(
            lambda x: f"{mdp_prms}{Experiment.SEPARATOR_PRMS}{mdp_name}"
            f"{Experiment.SEPARATOR_MDP_AGENT}"
            f"{agent_prms}{Experiment.SEPARATOR_PRMS}{agent_name}" in x,
            logs_folders,
        )
    )


def apply_gin_config(gin_config_files_paths : List[str]):
    """
    applies the gin configurations that are in the given files.
    """
    import gin

    gin.clear_config()

    from colosseum.utils.miscellanea import get_all_agent_classes, get_all_mdp_classes

    get_all_mdp_classes()
    get_all_agent_classes()

    for config_file in gin_config_files_paths:
        gin.parse_config_file(config_file)


def retrieve_experiment_prms(result_folder: str, num_seeds: int, merge, num_steps : int):
    (
        experiments_mdp_classes_and_gin_scopes,
        experiment_agents_classes_and_gin_scopes,
        gin_config_files_paths,
    ) = retrieve_experiment_classes_and_gin_scopes(result_folder)

    return (
        get_experiment_mdp_agent_couples(
            experiments_mdp_classes_and_gin_scopes,
            experiment_agents_classes_and_gin_scopes,
            num_seeds,
            result_folder,
            gin_config_files_paths,
            merge,
            num_steps,
        ),
        (result_folder, experiments_mdp_classes_and_gin_scopes, gin_config_files_paths),
    )


def retrieve_experiments_prms(
    num_steps: int,
    num_seeds: int,
    overwrite_experiment: bool,
    remove_experiment_folder: bool,
    experiment_config_dir: str = "experiments_to_run",
    seed=None,
):

    experiments_folders = []
    experiments = []
    experiments_mdp_configurations = []
    from colosseum.utils.logging import make_bold

    print(make_bold("Retrieving experiments:"))
    for experiment in glob(f"{ensure_folder(experiment_config_dir)}**"):
        if f"run{os.sep}_" in experiment:
            continue
        print(f"\t- experiment {experiment.split(os.sep)[1]}")

        result_folder, merge = prepare_folders(experiment, overwrite_experiment)

        experiments_folders.append(result_folder)

        o = 0
        for f in glob(f"{ensure_folder(result_folder)}**{os.sep}*.csv", recursive=True):
            with open(f, "r") as ff:
                l = ff.readlines()
            if (
                len(l) < 10
                or any(np.diff(pd.read_csv(f).steps) < 0)
                or pd.read_csv(f).steps.tolist()[-1] < num_steps * 0.95
            ):
                f_te = f[: f.rfind(os.sep)] + os.sep + "time_exceeded.txt"
                if os.path.exists(f_te):
                    with open(f_te, "r") as ff:
                        te = ff.readlines()
                    for tee in te:
                        if f in tee:
                            te.remove(tee)
                            break
                    if len(te) > 0:
                        with open(f_te, "w") as ff:
                            ff.write("".join(te))
                    else:
                        os.remove(f_te)

                shutil.move(
                    f, gettempdir() + f"{os.sep}_{o}_" + f[f.rfind(os.sep) + 1 :]
                )
                o += 1
                warnings.warn(
                    f"The file {f} has been moved to tmp as it has some formatting errors."
                )

        cur_experiments, mdp_configurations = retrieve_experiment_prms(
            result_folder, num_seeds, merge, num_steps
        )

        experiments += cur_experiments
        experiments_mdp_configurations.append(mdp_configurations)

        if remove_experiment_folder:
            rmtree(experiment)

    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(experiments)

    checks = [
        ",".join([str(x[0]), x[1].__name__, x[2], x[3].__name__, x[4], x[5]])
        for x in experiments
    ]
    assert len(experiments) == len(list(set(checks)))
    return experiments, experiments_mdp_configurations, experiments_folders


def check_existance(path, N, n_seeds, load):
    if load and os.path.exists(path):
        return np.load(path), True
    return np.zeros((N, n_seeds)), False


def instantiate(
    diam_path,
    vnorm_path,
    gaps_path,
    cumulative_regret_path,
    N,
    n_seeds,
    load_d=True,
    load_v=True,
    load_g=True,
    load_r=True,
):
    diameter_values, d_loaded = check_existance(diam_path, N, n_seeds, load_d)
    valuenorm_values, v_loaded = check_existance(vnorm_path, N, n_seeds, load_v)
    gaps_values, g_loaded = check_existance(gaps_path, N, n_seeds, load_g)
    cum_reg_values, r_loaded = check_existance(
        cumulative_regret_path, N, n_seeds, load_r
    )

    return (
        diameter_values,
        valuenorm_values,
        gaps_values,
        cum_reg_values,
        d_loaded,
        v_loaded,
        g_loaded,
        r_loaded,
    )


def calculate_values(
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
    agent_config=None,
    T=200_000,
    max_time=60 * 2,
    n_seeds=2,
    n_sample_per_cpu_count=4,
    verbose=False,
):
    if not d_loaded:
        diameter_values[i, seed] = mdp.diameter
    if not v_loaded:
        valuenorm_values[i, seed] = mdp.get_value_norm_policy(True)
    if not g_loaded:
        gaps_values[i, seed] = mdp.suboptimal_gaps
    if approximate_regret and not r_loaded:
        cum_reg_values[i, seed] = get_optimal_regret(
            mdp_class,
            mdp_kwargs,
            T,
            max_time,
            n_seeds,
            n_sample_per_cpu_count,
            agent_config,
            seed,
            verbose=verbose,
        )


def c(x):
    return str(x).replace(".", "")


def episodic_agent(mdp, seed, i, cum_reg_values, samp_compl_values, T=1_000_000):
    agent = PSRLEpisodic(
        environment_spec=make_environment_spec(mdp),
        seed=seed,
        H=mdp.H,
        r_max=mdp.r_max,
        T=T,
        reward_prior_model=RewardsConjugateModel.N_NIG,
        transitions_prior_model=TransitionsConjugateModel.M_DIR,
        rewards_prior_prms=[0.33, 1, 1, 1],
        transitions_prior_prms=[0.017],
    )
    loop = MDPLoop(mdp, agent, logger=InMemoryLogger())
    loop.run(T=T, verbose=True)
    df_e = pd.DataFrame(loop.logger.data)
    loop.close()
    cumulative_regret = df_e.loc[:, "cumulative_regret"].tolist()[-1]
    total_steps = df_e.loc[:, "steps"].tolist()[-1]
    assert total_steps < T - 100
    cum_reg_values[i, seed] = cumulative_regret
    samp_compl_values[i, seed] = total_steps


def retrieve_n_seed(logs_folders: List[str]) -> int:
    return len(
        np.unique(
            list(
                int(x[x.find("seed") + 4 : x.find("_logs.csv")])
                for x in glob(ensure_folder(logs_folders[0]) + "seed*")
                if "_logs.csv" in x
            )
        )
    )


def unite_seed(cur_logs_folder) -> pd.DataFrame:
    info_logs = glob(f"{ensure_folder(cur_logs_folder)}**logs.csv")
    df = pd.DataFrame()
    for i, info_log in enumerate(info_logs):
        with open(info_log, "r") as f:
            if len(f.readlines()) == 0:
                raise ValueError(f"{info_log} is empty.")
        df_cur = pd.read_csv(info_log)
        df_cur.insert(0, "seed", i)
        df = df.append(df_cur)
    return df.reset_index(inplace=False)


def retrieve_available_measures(logs_folders) -> List:
    with open(glob(f"{logs_folders[0]}{os.sep}**.yml")[0], "r") as f:
        m = yaml.load(f, yaml.Loader)
        measures = sorted(
            list(m["MDP graph metrics"].keys())
            + list(m["MDP measure of hardness"].keys())
        )
    return measures


def retrieve_measure_value(
    measure, mdp_class, mdp_prms, agent_class, agent_prms, logs_folders=None
):
    from colosseum.experiments.utils import get_log_folder

    cur_logs_folder = get_log_folder(
        mdp_class.__name__,
        mdp_prms,
        agent_class.__name__,
        agent_prms,
        logs_folders,
    )
    return get_measure_value(measure, cur_logs_folder)


def get_measure_value(measure, cur_logs_folder):
    reports = glob(f"{ensure_folder(cur_logs_folder)}**report.yml")
    mean_report, _ = mean_metrics_report(reports)

    if measure in mean_report["MDP measure of hardness"]:
        return mean_report["MDP measure of hardness"][measure]
    return mean_report["MDP graph metrics"][measure]


def prepare_folders(
    experiment_folder: str, overwrite_experiment: bool = False
) -> Tuple[str, bool]:
    result_folder = "experiments_done" + os.sep + experiment_folder.split(os.sep)[-1]

    if overwrite_experiment:
        rmtree(result_folder, ignore_errors=True)
    else:
        if os.path.isdir(result_folder):
            same_experiment = True
            for f in glob(ensure_folder(experiment_folder) + "**", recursive=True):
                if os.path.isfile(f):
                    if os.path.isfile(f.replace(experiment_folder, result_folder)):
                        with open(f, "r") as fr:
                            exp_fold_file = fr.read()
                        with open(
                            f.replace(experiment_folder, result_folder), "r"
                        ) as fr:
                            res_fold_file = fr.read()
                        if exp_fold_file != res_fold_file:
                            same_experiment = False
                            break
                if os.path.isdir(f):
                    if not os.path.isdir(f.replace(experiment_folder, result_folder)):
                        same_experiment = False
                        break

            if not same_experiment:
                raise ValueError(
                    "If you set overwrite_experiment to False you have to make sure that there are not other different previously run experiments with the same name."
                )
            return result_folder, True

    copytree(experiment_folder, result_folder)
    return result_folder, False


def check_for_log_file(result_folder, seed, mc, ms, ac, asc, num_steps) -> bool:
    # mc, ms, ac, asc = mdp_class, mdp_scope, agent_class, agent_scope
    lf = (
        ensure_folder(result_folder)
        + "logs"
        + os.sep
        + experiment_label(mc, ms, ac, asc)
        + f"{os.sep}seed{seed}_logs.csv"
    )
    if os.path.exists(lf):
        with open(lf, "r") as olff:
            olfff = olff.read()
        if len(olfff) > 10 and pd.read_csv(lf).steps.tolist()[-1] * 1.05 > num_steps:
            return True
        return False
    return False


def get_experiment_mdp_agent_couples(
    experiments_mdp_classes_and_gin_scopes,
    experiment_agents_classes_and_gin_scopes,
    num_seeds,
    result_folder,
    gin_config_files_paths,
    merge,
    num_steps,
) -> List[Tuple[int, Type["MDP"], str, Type["Agent"], str, str, List[str]]]:
    experiment_mdp_agent_couples = []
    for seed in range(num_seeds):
        for mdp_class, mdp_scopes in experiments_mdp_classes_and_gin_scopes.items():
            for mdp_scope in mdp_scopes:
                for (
                    agent_class,
                    agent_scopes,
                ) in experiment_agents_classes_and_gin_scopes.items():
                    for agent_scope in agent_scopes:
                        if not (
                            merge
                            and check_for_log_file(
                                result_folder,
                                seed,
                                mdp_class,
                                mdp_scope,
                                agent_class,
                                agent_scope,
                                num_steps,
                            )
                        ):
                            experiment_mdp_agent_couples.append(
                                (
                                    seed,
                                    mdp_class,
                                    mdp_scope,
                                    agent_class,
                                    agent_scope,
                                    result_folder,
                                    gin_config_files_paths,
                                )
                            )
    return experiment_mdp_agent_couples


def retrieve_experiment_classes_and_gin_scopes(experiment: str):
    from colosseum.utils.miscellanea import get_all_agent_classes, get_all_mdp_classes

    gin_config_files_paths = []
    mdp_classes = get_all_mdp_classes()
    agent_classes = get_all_agent_classes()

    experiments_mdp_classes: Dict[Type["MDP"], set] = {}
    for mdp_config_file in glob(f"{ensure_folder(experiment)}mdp_configs{os.sep}*"):
        gin.parse_config_file(mdp_config_file)
        with open(mdp_config_file, "r") as f:
            scopes = set(
                map(
                    lambda x: x.split(os.sep)[0],
                    filter(lambda x: os.sep in x, f.readlines()),
                )
            )
        mdp_class_name = mdp_config_file.split(f"mdp_configs{os.sep}")[1].split(".")[0]
        experiments_mdp_classes[
            next(filter(lambda c: c.__name__ == mdp_class_name, mdp_classes))
        ] = scopes
        gin_config_files_paths.append(mdp_config_file)

    experiments_agents_classes: Dict[Type["Agent"], set] = {}
    for agent_config_file in glob(f"{ensure_folder(experiment)}agent_configs{os.sep}*"):
        gin.parse_config_file(agent_config_file)
        with open(agent_config_file, "r") as f:
            scopes = set(
                map(
                    lambda x: x.split(os.sep)[0],
                    filter(lambda x: os.sep in x, f.readlines()),
                )
            )
        agent_class_name = agent_config_file.split(f"agent_configs{os.sep}")[1].split(
            "."
        )[0]
        experiments_agents_classes[
            next(filter(lambda c: c.__name__ == agent_class_name, agent_classes))
        ] = scopes
        gin_config_files_paths.append(agent_config_file)

    # We do not mix and match episodic mdps and/or agents
    episodic_checks = (
        c.is_episodic()
        for c in (
            list(experiments_mdp_classes.keys())
            + list(experiments_agents_classes.keys())
        )
    )
    assert all(episodic_checks) or all(
        not c for c in episodic_checks
    ), "We do not mix and match episodic mdps and/or agents. Please check the config files."

    return experiments_mdp_classes, experiments_agents_classes, gin_config_files_paths
