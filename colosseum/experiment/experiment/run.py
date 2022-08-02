import os
from multiprocessing import Pool
from typing import Dict

import numpy as np
import ray
from ray.util.multiprocessing import Pool as RayPool
from tqdm import tqdm, trange

from colosseum import (
    config,
    disable_multiprocessing,
    get_available_cores,
    set_available_cores,
)
from colosseum.experiment.agent_mdp_interaction import MDPLoop
from colosseum.experiment.experiment import ExperimentConfig, ExperimentInstance
from colosseum.experiment.experiment.folder_structuring import (
    retrieve_experiment_instances,
)
from colosseum.experiment.experiment.utils import (
    apply_gin_config,
    check_experiment_folders_formatting,
)
from colosseum.utils import make_environment_spec
from colosseum.utils.acme import CSVLogger


def run_experiments_from_folders(
    n_cores: int, experiment_configs: Dict[str, ExperimentConfig] = None, use_ray=False
):
    """
    takes the experiments from the config.get_experiment_to_run_folder() to run folder and store the results in the
    config.get_experiment_results_folder(). The experiment configuration can be provided as experiment_config.yml files in
    the corresponding directories or as a dictionary in input to this function.
    """

    # Check that every experiment folder is well-structured
    check_experiment_folders_formatting(os.getcwd(), experiment_configs)

    # Retrieve experiment instances from the default experiment configuration folder
    experiment_instances = retrieve_experiment_instances(experiment_configs)
    np.random.RandomState(42).shuffle(experiment_instances)

    if len(experiment_instances) >= n_cores > 1:
        # Ensure that Colosseum does not use multiprocessing while we perform hyperparameter optimization
        colosseum_cores_config = get_available_cores()
        disable_multiprocessing()
        vl = config.VERBOSE_LEVEL
        if type(config.VERBOSE_LEVEL) == int:
            config.disable_verbose_logging()

        loop = trange(len(experiment_instances))
        if use_ray:
            ray.init(num_cpus=n_cores)
            with RayPool(processes=n_cores) as p:
                for exp_ins in p.imap_unordered(
                    run_experiment_instance, experiment_instances
                ):
                    loop.update(1)
                    loop.set_description(f"{exp_ins.experiment_label} completed.")
            ray.shutdown()
        else:
            with Pool(processes=n_cores) as p:
                for exp_ins in p.imap_unordered(
                    run_experiment_instance, experiment_instances
                ):
                    loop.update(1)
                    loop.set_description(f"{exp_ins.experiment_label} completed.")

        # Reactive previous multiprocessing configuration
        set_available_cores(colosseum_cores_config)
        config.VERBOSE_LEVEL = vl
    else:
        vl = config.VERBOSE_LEVEL
        if type(config.VERBOSE_LEVEL) == int:
            config.disable_verbose_logging()
        for experiment_instance in tqdm(experiment_instances):
            run_experiment_instance(experiment_instance)
        config.VERBOSE_LEVEL = vl


def run_experiment_instance(exp_ins: ExperimentInstance):
    """
    runs the experiment instance given in input with the corresponding gin config applied.
    """
    import gin

    apply_gin_config(exp_ins.gin_config_files)

    with gin.config_scope(exp_ins.mdp_scope):
        mdp = exp_ins.mdp_class(seed=exp_ins.seed, randomize_actions=True)
    with gin.config_scope(exp_ins.agent_scope):
        agent = exp_ins.agent_class(
            seed=exp_ins.seed,
            environment_spec=make_environment_spec(mdp),
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
        exp_ins.experiment_config.max_agent_mdp_interaction_time_s,
    )

    if last_training_step != -1:
        with open(f"{logger._directory}{os.sep}time_exceeded.txt", "a") as f:
            f.write(
                f"last training step at ({last_training_step}) for {logger.file_path}\n"
            )
    return exp_ins
