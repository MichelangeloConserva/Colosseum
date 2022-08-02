import datetime
import time
from typing import Any, Dict, List, Tuple, Type

import numpy as np
from ray import tune

from colosseum.experiment.agent_mdp_interaction import MDPLoop
from colosseum.mdp import BaseMDP
from colosseum.agent.agents.base import BaseAgent
from colosseum.utils import clean_for_storing
from colosseum.utils.acme import InMemoryLogger
from colosseum.utils.acme.specs import make_environment_spec


def get_regret_score(
    mdp: "BaseMDP",
    agent: "BaseAgent",
    optimization_horizon: int,
    max_time: float,
    log_every: int,
    performance_indicator: str = "normalized_cumulative_regret",
    enforce_time_constraint: bool = True,
) -> float:
    """
    returns the performance indicator for a single agent/MDP interaction.
    """
    loop = MDPLoop(
        mdp,
        agent,
        InMemoryLogger(),
        enforce_time_constraint=enforce_time_constraint,
    )
    t, logs = loop.run(optimization_horizon, log_every=log_every, max_time=max_time)
    return logs[performance_indicator]


def compute_regret(
    agent_hyperparameters: Dict[str, Any],
    debug_file_path: str,
    mdp_classes: List[Type["BaseMDP"]],
    n_mdp_parameter_samples_from_class: int,
    mdp_parameters_sampler_seed: int,
    optimization_horizon: int,
    n_seeds: int,
    agent_class: Type["BaseAgent"],
    max_interaction_s: float,
    report_ray: bool,
    enforce_time_constraint: bool,
    log_every: int,
) -> Tuple[Dict[str, Any], float]:
    """
    returns the average normalized cumulative regret divided by the optimization horizon computed over samples from the
    MDP classes given in input.

    Parameters
    ----------
    agent_hyperparameters : Dict[str, Any]
        is the set of hyperparameters for the agent.
    debug_file_path : str
        is the file in which debug logs will be stored. This is not used when the optimization procedure is carried out
        with ray.tune.
    mdp_classes : List[Type["BaseMDP"]]
        is the list of MDP classes that are used in the optimization procedure.
    n_mdp_parameter_samples_from_class
        is the number of parameters sampled from the MDP classes and that results in MDP instance that the agent.
        will interact with during the hyperparameter optimization.
    mdp_parameters_sampler_seed : int
        is the seed used for sampling the MDP parameter to create the MDP instance over which the optimization is carried
        out.
    optimization_horizon : int
        is the optimization horizon that will be used in the agent/MDP interactions.
    n_seeds : int
        is the number of seeds for which each agent/MDP interaction is repeated.
    agent_class : Type["BaseAgent"]
        is the agent class for which the hyperparameters will be optimized.
    max_interaction_s : float
        is the maximum amount of training time given to the agent for each agent/MDp interaction.
    report_ray : bool
        checks whether to report the result of the procedure to ray or to return.
    enforce_time_constraint : bool
        checks whether to actively enforce the time limit constraints in the agent/MDP interaction. This must be set to
        False when using ray.
    """
    assert not report_ray or (report_ray and enforce_time_constraint)

    start = time.time()
    regrets = dict()
    for mdp_class in mdp_classes:
        for i, mdp_parameter in enumerate(
            mdp_class.sample_parameters(
                n_mdp_parameter_samples_from_class, mdp_parameters_sampler_seed
            )
        ):
            with open(
                debug_file_path.replace(".txt", "_" + mdp_class.__name__ + ".txt"),
                "a",
            ) as f:
                f.write(f"{mdp_class.__name__}\n")
                for k, v in clean_for_storing(mdp_parameter).items():
                    f.write(f"\t{k}:\t\t\t\t{v}\n")
                f.write(f"{agent_class.__name__}\n")
                for k, v in clean_for_storing(agent_hyperparameters).items():
                    f.write(f"\t{k}:\t\t\t\t{v}\n")

            for seed in range(n_seeds):
                mdp_parameter["randomize_actions"] = True
                mdp = mdp_class(seed=seed, **mdp_parameter)
                agent = agent_class.get_agent_instance_from_hyperparameters(
                    seed,
                    optimization_horizon,
                    make_environment_spec(mdp),
                    agent_hyperparameters,
                )
                normalized_cumulative_regret = get_regret_score(
                    mdp,
                    agent,
                    optimization_horizon,
                    max_interaction_s,
                    log_every=log_every,
                    enforce_time_constraint=enforce_time_constraint,
                )
                r = normalized_cumulative_regret / optimization_horizon
                regrets[str(clean_for_storing(mdp_parameter))] = r
                if not report_ray:
                    with open(
                        debug_file_path.replace(
                            ".txt", "_" + mdp_class.__name__ + ".txt"
                        ),
                        "a",
                    ) as f:
                        f.write(
                            f"Time: {datetime.timedelta(seconds=int(time.time() - start))},"
                            f" regret : {r:3.5f},"
                            f" MDP class: {mdp_class.__name__}"
                            f" parameter:{i + 1}/{n_mdp_parameter_samples_from_class} "
                            f" seed:{seed + 1}/{n_seeds}\n"
                        )
    score = float(np.mean(list(regrets.values())))
    if not report_ray:
        return agent_hyperparameters, score
    tune.report(regret=score)
