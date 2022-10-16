import os
import pickle
import shutil
from multiprocessing import Pool
from typing import List, Union, TYPE_CHECKING

import numpy as np
import ray
from ray.util.multiprocessing import Pool as RayPool
from tqdm import tqdm

from colosseum import config
from colosseum.benchmark.utils import retrieve_experiment_config
from colosseum.experiment.agent_mdp_interaction import MDPLoop
from colosseum.experiment.experiment_instance import ExperimentInstance
from colosseum.experiment.folder_structuring import _get_experiment_mdp_agent_couples
from colosseum.experiment.folder_structuring import get_mdp_agent_gin_configs
from colosseum.experiment.utils import apply_gin_config
from colosseum.experiment.utils import check_experiment_folder
from colosseum.utils import ensure_folder, make_mdp_spec
from colosseum.utils.acme import CSVLogger

if TYPE_CHECKING:
    pass


def get_experiment_instances_from_folder(
    experiment_folder: str,
) -> List[ExperimentInstance]:
    """
    Returns
    -------
    List[ExperimentInstance]
        The `ExperimentInstance`s associated with the experiment folder.
    """

    if config.VERBOSE_LEVEL != 0:
        print(f"\t- {experiment_folder}")

    # Retrieve experiment configuration from dictionary in input or from .yml file
    experiment_config = retrieve_experiment_config(experiment_folder)

    # Check that every experiment folder is well-structured
    check_experiment_folder(experiment_folder, experiment_config)

    (
        mdp_classes_scopes,
        agent_classes_scopes,
        gin_config_files_paths,
    ) = get_mdp_agent_gin_configs(experiment_folder)

    assert (
        len(mdp_classes_scopes) > 0
    ), f"No MDP gin configurations find in {experiment_folder}"
    assert (
        len(agent_classes_scopes) > 0
    ), f"No agent gin configurations find in {experiment_folder}"
    assert (
        len(gin_config_files_paths) > 0
    ), f"No gin configurations find in {experiment_folder}"

    return _get_experiment_mdp_agent_couples(
        experiment_config,
        experiment_folder,
        mdp_classes_scopes,
        agent_classes_scopes,
        gin_config_files_paths,
    )


def save_instances_to_folder(
    experiment_instances: List[ExperimentInstance],
    store_instances_folder: str,
    overwrite=False,
) -> List[str]:
    """
    Parameters
    ----------
    experiment_instances : List[ExperimentInstance]
        The `ExperimentInstance`s to be store locally.
    store_instances_folder : str
        The folder where the `ExperimentInstance`s are to be stored.
    overwrite : bool
        If True, any file in store_instances_folder is removed. If False, it raises an error if store_instances_folder
        contains some files.

    Returns
    -------
    List[str]
        The paths of the pickled `ExperimentInstance`s.
    """

    # Prepare the folder to store the pickled experiment instances
    if (
        os.path.isdir(store_instances_folder)
        and len(os.listdir(store_instances_folder)) > 0
    ):
        if overwrite:
            shutil.rmtree(store_instances_folder)
        else:
            raise ValueError(
                f"The store_instances_folder is not empty, {store_instances_folder}"
            )
    os.makedirs(store_instances_folder, exist_ok=True)

    # Pickle the experiments instances to be run later
    experiment_instance_paths = []
    for i, exp_ins in enumerate(experiment_instances):
        fp = ensure_folder(store_instances_folder) + f"exp_inst_{i+1}.pkl"
        experiment_instance_paths.append(fp)
        with open(fp, "wb") as f:
            pickle.dump(exp_ins, f)

    return experiment_instance_paths


def run_experiment_instances(
    experiment_instances: List[Union[ExperimentInstance, str]],
    use_ray=False,
):
    """
    runs the `ExperimentInstance`s locally. If multiprocessing is enabled in the package configuration, multiple cores
    will be used to simultaneously run several `ExperimentInstance`s.

    Parameters
    ----------
    experiment_instances : List[Union[ExperimentInstance, str]]
        The `ExperimentInstance`s to run. They can either be `ExperimentInstance` objects or paths to the pickled
        `ExperimentInstance`s.
    use_ray : bool
        If True, it uses `ray` to handle the multiprocessing. By default, the Python default `multiprocessing` module is
        used.
    """

    if len(experiment_instances) == 0:
        return

    # Shuffle can improve speed of multiprocessing
    np.random.RandomState(42).shuffle(experiment_instances)

    exp_done = 0
    if config.VERBOSE_LEVEL != 0:
        tqdm.write(f"Completed: {exp_done}/{len(experiment_instances)}")
    if len(experiment_instances) >= config.get_available_cores() > 1:
        # Ensure that Colosseum does not use multiprocessing while we run the experiments
        colosseum_cores_config = config.get_available_cores()
        config.disable_multiprocessing()

        if use_ray:
            ray.init(num_cpus=colosseum_cores_config)
            with RayPool(processes=colosseum_cores_config) as p:
                for exp_ins in p.imap_unordered(
                    run_experiment_instance, experiment_instances
                ):
                    exp_done += 1
                    if config.VERBOSE_LEVEL != 0:
                        tqdm.write(f"Completed: {exp_done}/{len(experiment_instances)}")
            ray.shutdown()
        else:
            with Pool(processes=colosseum_cores_config) as p:
                for exp_ins in p.imap_unordered(
                    run_experiment_instance, experiment_instances
                ):
                    exp_done += 1
                    if config.VERBOSE_LEVEL != 0:
                        tqdm.write(f"Completed: {exp_done}/{len(experiment_instances)}")

        # Restore the previous configuration
        config.set_available_cores(colosseum_cores_config)
    else:
        for experiment_instance in experiment_instances:
            run_experiment_instance(experiment_instance)
            exp_done += 1
            if config.VERBOSE_LEVEL != 0:
                tqdm.write(f"Completed: {exp_done}/{len(experiment_instances)}")


def run_experiment_instance(exp_ins: Union[ExperimentInstance, str]):
    """
    runs a single `ExperimentInstance`, which can be passed as `ExperimentInstance` object or as a path to a pickled
    `ExperimentInstance` object.
    """

    # Load the experiment instance if a path is given
    if type(exp_ins) == str:
        with open(exp_ins, "rb") as f:
            exp_ins = pickle.load(f)

    import gin

    apply_gin_config(exp_ins.gin_config_files)

    with gin.config_scope(exp_ins.mdp_scope):
        mdp = exp_ins.mdp_class(
            seed=exp_ins.seed,
            emission_map=exp_ins.emission_map,
        )
    with gin.config_scope(exp_ins.agent_scope):
        agent = exp_ins.agent_class(
            seed=exp_ins.seed,
            mdp_specs=make_mdp_spec(mdp),
            optimization_horizon=exp_ins.experiment_config.n_steps,
        )

    logger = CSVLogger(
        exp_ins.result_folder,
        add_uid=False,
        label=exp_ins.experiment_label,
        file_name=f"seed{exp_ins.seed}_logs",
    )
    loop = MDPLoop(mdp, agent, logger)
    last_training_step, _ = loop.run(
        exp_ins.experiment_config.n_steps,
        exp_ins.experiment_config.log_performance_indicators_every,
        exp_ins.experiment_config.max_interaction_time_s,
    )

    if last_training_step != -1:
        with open(f"{logger._directory}{os.sep}time_exceeded.txt", "a") as f:
            f.write(
                f"last training step at ({last_training_step}) for {logger.file_path}\n"
            )
    return exp_ins
