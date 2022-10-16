import os
import re
import shutil
from glob import glob
from typing import List, Dict, TYPE_CHECKING, Type, Union

import yaml

from colosseum.benchmark.benchmark import ColosseumBenchmark
from colosseum.emission_maps import EmissionMap
from colosseum.experiment import ExperimentConfig
from colosseum.utils import ensure_folder
from colosseum.utils.miscellanea import (
    get_agent_class_from_name,
    get_mdp_class_from_name,
    compare_gin_configs,
)

if TYPE_CHECKING:
    from colosseum.mdp.base import BaseMDP
    from colosseum.agent.agents.base import BaseAgent


def get_mdps_configs_from_mdps(mdps: List["BaseMDP"]) -> List[str]:
    """
    Returns
    -------
    List[str]
        The gin configs of the MDPs
    """
    mdp_configs = dict()
    for mdp in mdps:
        if not type(mdp) in mdp_configs:
            mdp_configs[type(mdp)] = []
        mdp_configs[type(mdp)].append(mdp.get_gin_config(len(mdp_configs[type(mdp)])))
    return mdp_configs


def instantiate_agent_configs(
    agents_configs: Dict[Type["BaseAgent"], str],
    benchmark_folder: str,
):
    """
    instantiates the gin agent configurations in the given folder.

    Parameters
    ----------
    agents_configs : Dict[Type["BaseAgent"], str]
        The dictionary associates agent classes to their gin config files.
    benchmark_folder : str
        The folder where the corresponding benchmark is located.
    """

    if os.path.isdir(benchmark_folder + "agents_configs" + os.sep):
        try:
            local_agent_configs = retrieve_agent_configs(benchmark_folder)
            if not compare_gin_configs(agents_configs, local_agent_configs):
                raise ValueError(
                    f"The existing agent configs in {benchmark_folder} are different from the one in input."
                )

        # If the local folder is corrupted, we eliminate it
        except AssertionError:
            shutil.rmtree(benchmark_folder + "agents_configs")
    else:
        os.makedirs(ensure_folder(benchmark_folder) + "agents_configs", exist_ok=True)
        for mdp_cl, gin_config in agents_configs.items():
            with open(
                ensure_folder(benchmark_folder)
                + "agents_configs"
                + os.sep
                + mdp_cl.__name__
                + ".gin",
                "w",
            ) as f:
                f.write(gin_config)


def instantiate_benchmark_folder(benchmark: ColosseumBenchmark, benchmark_folder: str):
    """
    instantiates the benchmark locally. If a local benchmark is found then it merges iff they are the same in terms of
    MDP configs and experiment configurations.

    Parameters
    ----------
    benchmark : ColosseumBenchmark
        The benchmark to instantiate.
    benchmark_folder : str
        The folder where the corresponding benchmark is located.
    """

    # Check whether the experiment folder has already been created
    if os.path.isdir(benchmark_folder) and len(os.listdir(benchmark_folder)) > 0:
        try:
            local_benchmark = retrieve_benchmark(benchmark_folder)
            if local_benchmark != benchmark:
                raise ValueError(
                    f"The experiment folder {benchmark_folder} is already occupied."
                )

        # If the local folder is corrupted, we eliminate it
        except AssertionError:
            shutil.rmtree(benchmark_folder)
    else:
        benchmark.instantiate(benchmark_folder)


def retrieve_benchmark(
    benchmark_folder: str, experiment_config: ExperimentConfig = None, postfix: str = ""
) -> ColosseumBenchmark:
    """
    retrieves a benchmark from a folder.

    Parameters
    ----------
    benchmark_folder : ColosseumBenchmark
        The folder where the benchmark is located.
    experiment_config : ExperimentConfig
        The experiment config to be substituted to the default one. By default, no substitution happens.
    postfix : str
        The postfix to add to the name of the benchmark. By default, no postfix is added.

    Returns
    -------
    ColosseumBenchmark
        The retrieved benchmark.
    """
    benchmark = ColosseumBenchmark(
        os.path.basename(ensure_folder(benchmark_folder)[:-1]) + postfix,
        retrieve_mdp_configs(benchmark_folder),
        retrieve_experiment_config(benchmark_folder)
        if experiment_config is None
        else experiment_config,
    )
    return benchmark


def update_emission_map(benchmark_folder: str, emission_map: EmissionMap):
    """
    substitutes the emission map in the experiment config of the given experiment folder with the one given in input.
    """
    config_fp = ensure_folder(benchmark_folder) + "experiment_config.yml"
    assert os.path.isfile(
        config_fp
    ), f"The folder {benchmark_folder} does not contain a configuration file."

    with open(config_fp, "r") as f:
        config_file = yaml.load(f, yaml.Loader)
    config_file["emission_map"] = emission_map.__name__
    with open(config_fp, "w") as f:
        yaml.dump(config_file, f)


def retrieve_experiment_config(benchmark_folder: str) -> ExperimentConfig:
    """
    Returns
    -------
    ExperimentConfig
        The experiment config from the given benchmark folder.
    """
    config_fp = ensure_folder(benchmark_folder) + "experiment_config.yml"
    assert os.path.isfile(
        config_fp
    ), f"The folder {benchmark_folder} does not contain a configuration file."

    with open(config_fp, "r") as f:
        exp_config = yaml.load(f, yaml.Loader)
    return ExperimentConfig(**exp_config)


def retrieve_mdp_configs(
    benchmark_folder: str, return_string=True
) -> Union[Dict[Type["BaseMDP"], str], Dict[Type["BaseMDP"], Dict[str, str]],]:
    """
    retrieves the MDP gin configs of a benchmark.

    Parameters
    ----------
    benchmark_folder : ColosseumBenchmark
        The folder where the benchmark is located.
    return_string : bool
        If False, the gin configs are returned as a list of strings. If True, the list is joined in a singles string. By
        default, the single string format is used.

    Returns
    -------
    Union[
        Dict[Type["BaseMDP"], str],
        Dict[Type["BaseMDP"], Dict[str, str]],
    ]
        The dictionary that for each MDP name contains a list of gin configs obtained from the given benchmark folder.
    """
    return retrieve_gin_configs(
        ensure_folder(benchmark_folder) + "mdp_configs" + os.sep, return_string
    )


def retrieve_agent_configs(
    benchmark_folder: str, return_string=True
) -> Union[Dict[Type["BaseAgent"], str], Dict[Type["BaseAgent"], Dict[str, str]],]:
    """
    retrieves the agent gin configs of a benchmark.

    Parameters
    ----------
    benchmark_folder : ColosseumBenchmark
        The folder where the benchmark is located.
    return_string : bool
        If False, the gin configs are returned as a list of strings. If True, the list is joined in a singles string. By
        default, the single string format is used.

    Returns
    -------
    Union[
        Dict[Type["BaseAgent"], str],
        Dict[Type["BaseAgent"], Dict[str, str]],
    ]
        The dictionary that for each MDP name contains a list of gin configs obtained from the given benchmark folder.
    """
    return retrieve_gin_configs(
        ensure_folder(benchmark_folder) + "agents_configs" + os.sep, return_string
    )


def retrieve_gin_configs(
    gin_config_folder: str, return_string: bool
) -> Dict[Union[Type["BaseMDP"], Type["BaseAgent"]], str]:
    """
    retrieves the gin configs from a folder.

    Parameters
    ----------
    gin_config_folder : ColosseumBenchmark
        The folder where the gin configs are stored.
    return_string : bool
        If False, the gin configs are returned as a list of strings. If True, the list is joined in a singles string. By
        default, the single string format is used.

    Returns
    -------
    Dict[Union[Type["BaseMDP"], Type["BaseAgent"]], str]
        The dictionary that for each MDP and agent name contains a list of gin configs obtained from the given folder.
    """

    gin_config_folder = ensure_folder(gin_config_folder)

    configs = glob(gin_config_folder + "*.gin")
    assert (
        len(configs) > 0
    ), f"The folder {gin_config_folder} does not contain config files"

    gin_configs = dict()
    for f in configs:
        name = os.path.basename(f).replace(".gin", "")
        cl = (
            get_agent_class_from_name(name)
            if "agent" in os.path.basename(gin_config_folder[:-1])
            else get_mdp_class_from_name(name)
        )

        gin_configs[cl] = [] if return_string else dict()
        with open(f, "r") as ff:
            gin_config_file = ff.read() + "\n"
        for config_prms in sorted(
            set(re.findall(r"prms_[0-9]+/", gin_config_file)),
            # Ascending order based on the parameter index
            key=lambda x: int(x.replace("prms_", "")[:-1]),
        ):
            imports = set(re.findall("from.+?import.+?\n", gin_config_file))
            prms_configs = "".join(re.findall(config_prms + ".+?\n", gin_config_file))
            if len(imports) > 0:
                prms_configs = "".join(imports) + prms_configs

            if return_string:
                gin_configs[cl].append(prms_configs)
            else:
                gin_configs[cl][config_prms[:-1]] = prms_configs

        if return_string:
            gin_configs[cl] = "\n".join(gin_configs[cl])

    return gin_configs
