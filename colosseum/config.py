import os
import shutil
from glob import glob
from typing import Type, TYPE_CHECKING, List

from colosseum.benchmark import BENCHMARKS_DIRECTORY
from colosseum.utils import ensure_folder

if TYPE_CHECKING:
    from colosseum.agent.agents.base import BaseAgent


# EXPERIMENT VARIABLES
EXPERIMENT_SEPARATOR_PRMS = "-"
EXPERIMENT_SEPARATOR_MDP_AGENT = "____"


# MULTIPROCESSING
_AVAILABLE_CORES = 1


def enable_multiprocessing():
    """
    sets the number of cores available to the number of cores in the machine minus two.
    """
    set_available_cores(os.cpu_count() - 2)


def disable_multiprocessing():
    """
    disables multiprocessing. Note that multiprocessing is disabled by default.
    """
    set_available_cores(1)


def set_available_cores(n: int):
    """
    sets the available core to n.
    """
    assert os.cpu_count() >= n >= 1, (
        f"The input value for the number of cores should be larger than zero and less than the number of cores in the "
        f"machine, {n} received."
    )

    global _AVAILABLE_CORES
    n = (os.cpu_count() - 2) if n is None else n
    _AVAILABLE_CORES = n


def get_available_cores() -> int:
    """
    Returns
    -------
    int
        The number of cores available to the package.
    """
    return _AVAILABLE_CORES


# VERBOSITY
VERBOSE_LEVEL = 0
"""The level of verbose output."""


def enable_verbose_logging():
    """
    enables verbose loggings.
    """
    global VERBOSE_LEVEL
    VERBOSE_LEVEL = 1


def set_verbose_logs_file(file_path: str):
    """
    redirects verbose logging to the file in input.
    """
    _check_log_file_path(file_path)
    global VERBOSE_LEVEL
    VERBOSE_LEVEL = file_path


def disable_verbose_logging():
    """
    disables verbosity.
    """
    global VERBOSE_LEVEL
    VERBOSE_LEVEL = 0


# NUMERICAL VARIABLES
_N_FLOATING_SAMPLING_HYPERPARAMETERS = 4
_SIZE_NOISE_CACHE = 5000
_MIN_LINEAR_FEATURE_DIMENSIONALITY = 10


def set_n_floating_sampling_hyperparameters(n: int):
    """
    sets the number of floating points to keep when rounding float samples from the tune.ray hyperparameter spaces.
    By default, it is set to 4.
    """
    assert 1 < n < 10
    global _N_FLOATING_SAMPLING_HYPERPARAMETERS
    _N_FLOATING_SAMPLING_HYPERPARAMETERS = n


def get_n_floating_sampling_hyperparameters() -> int:
    """
    Returns
    -------
    int
        The number of floating points to keep when rounding float samples from the tune.ray hyperparameter spaces.
    """
    return _N_FLOATING_SAMPLING_HYPERPARAMETERS


def set_size_cache_noise(x: int):
    """
    sets the size of the cached for the `Noise` objects. This reduces the computational burden of the noise sampling
    procedure. By default, it is set to 5000.
    """
    global _SIZE_NOISE_CACHE
    assert type(x) == int and x > 0
    _SIZE_NOISE_CACHE = x


def get_size_cache_noise() -> int:
    """
    Returns
    -------
    int
        The size of the cached for the `Noise` objects.
    """
    return _SIZE_NOISE_CACHE


def set_min_linear_feature_dim(x: int):
    """
    sets the minimum dimension for the `StateLinear` emission maps. By default, it is set to two.
    """
    global _MIN_LINEAR_FEATURE_DIMENSIONALITY
    assert type(x) == int and x > 0
    _MIN_LINEAR_FEATURE_DIMENSIONALITY = x


def get_min_linear_feature_dim() -> int:
    """
    Returns
    -------
    int
        The minimum dimension for the `StateLinear` emission maps.
    """
    return _MIN_LINEAR_FEATURE_DIMENSIONALITY


# PATHS
_HYPEROPT_FOLDER = "default_hyperopt" + os.sep + "hyperopt_experiments" + os.sep
_EXPERIMENTS_FOLDER = "experiments" + os.sep
_HARDNESS_MEASURES_CACHE_FOLDER = os.path.join("cached_hardness_measures", "")

_HARDNESS_MEASURES_CACHE_FOLDER_COLOSSEUM = os.path.join(
    BENCHMARKS_DIRECTORY, "cached_hardness_measures", ""
)
_HARDNESS_MDPS_CACHE_FOLDER_COLOSSEUM = os.path.join(
    BENCHMARKS_DIRECTORY, "cached_mdps", ""
)


def get_cached_hardness_benchmark_folder() -> str:
    """
    Returns
    -------
    str
        The folder where hardness of measures are cached.
    """
    return _HARDNESS_MEASURES_CACHE_FOLDER_COLOSSEUM


def get_cached_mdps_benchmark_folder() -> str:
    """
    Returns
    -------
    str
        The folder where the benchmark environments have been cached.
    """
    return _HARDNESS_MDPS_CACHE_FOLDER_COLOSSEUM


def set_hyperopt_folder(experiment_folder: str, experiment_name: str):
    """
    sets the folder for the hyperparameters optimization procedure. The resulting folder has the following structure,
    `experiment_folder/hyperopt/experiment_name`.

    Parameters
    ----------
    experiment_folder : str
        The folder for the results of the hyperparameters optimization procedure.
    experiment_name : str
        The name of the experiment related to the hyperparameter optimization procedure.
    """
    global _HYPEROPT_FOLDER
    _HYPEROPT_FOLDER = (
        ensure_folder(experiment_folder)
        + "hyperopt"
        + os.sep
        + ensure_folder(experiment_name)
    )
    os.makedirs(_HYPEROPT_FOLDER, exist_ok=True)


def get_hyperopt_folder() -> str:
    """
    Returns
    -------
    str
        The folder for the hyperparameters optimization procedure.
    """
    return _HYPEROPT_FOLDER


def set_experiments_folder(experiment_folder: str, experiment_name: str):
    """
    sets the folder for the results of the benchmarking procedure. The resulting folder has the following structure,
    `experiment_folder/benchmarking/experiment_name`.

    Parameters
    ----------
    experiment_folder : str
        The folder for the results of the benchmarking procedure.
    experiment_name : str
        The name of the experiment related to the benchmarking procedure.
    """
    global _EXPERIMENTS_FOLDER
    _EXPERIMENTS_FOLDER = (
        ensure_folder(experiment_folder)
        + "benchmarking"
        + os.sep
        + ensure_folder(experiment_name)
    )
    os.makedirs(_EXPERIMENTS_FOLDER, exist_ok=True)


def get_experiments_folder() -> str:
    """
    Returns
    -------
    str
        The folder for the results of the benchmarking procedure.
    """
    return _EXPERIMENTS_FOLDER


def set_up_hardness_measures_cache_folder():
    """
    copies the cached measures of hardness from the colosseum package to the local hardness measures cache folder.
    """

    os.makedirs(_HARDNESS_MEASURES_CACHE_FOLDER, exist_ok=True)

    # Copy the Colosseum cached hardness reports
    SRC_DIR = BENCHMARKS_DIRECTORY + "cached_hardness_measures"
    TARG_DIR = _HARDNESS_MEASURES_CACHE_FOLDER

    for mdp_dir in os.listdir(SRC_DIR):
        os.makedirs(TARG_DIR + os.sep + mdp_dir, exist_ok=True)
        for file in glob(SRC_DIR + os.sep + mdp_dir + os.sep + "*"):
            if os.path.basename(file) not in map(
                os.path.basename, glob(os.path.join(TARG_DIR, mdp_dir + os.sep + "*"))
            ):
                shutil.copy(file, TARG_DIR + os.sep + mdp_dir)


def set_hardness_measures_cache_folder(path: str):
    """
    sets the hardness measures cache folder to the path in input.
    """
    global _HARDNESS_MEASURES_CACHE_FOLDER
    _HARDNESS_MEASURES_CACHE_FOLDER = ensure_folder(path)


def get_hardness_measures_cache_folder() -> str:
    """
    Returns
    -------
    str
        The hardness measures cache folder. Note that if the folder does not exist, it is created and filled with the
        cached measures from the package.
    """
    if not os.path.isdir(_HARDNESS_MEASURES_CACHE_FOLDER):
        set_up_hardness_measures_cache_folder()
    return _HARDNESS_MEASURES_CACHE_FOLDER



# AGENTS
_REGISTERED_EXTERNAL_AGENT_CLASSES = list()


def register_agent_class(agent_class: Type["BaseAgent"]):
    """
    makes the package knows that the agent class in input can be used for hyperparameters optimization and benchmarking.
    """
    _REGISTERED_EXTERNAL_AGENT_CLASSES.append(agent_class)


def get_external_agent_classes() -> List[Type["BaseAgent"]]:
    """
    Returns
    -------
    List[Type["BaseAgent"]]
        The agent classes that have been registered to the package.
    """
    return _REGISTERED_EXTERNAL_AGENT_CLASSES


def _check_log_file_path(file_path: str):
    assert ".txt", "The file extension should be txt."
    if not os.path.isfile(file_path):
        with open(file_path, "w") as f:
            pass


#### Work in progress

_DEBUG_LEVEL = 0
_DEBUG_FILE = None


def process_debug_output(debug_output):
    """
    work in progress.
    """
    if _DEBUG_FILE:
        with open(_DEBUG_FILE, "a") as f:
            f.write(debug_output + "\n")
    else:
        if _DEBUG_LEVEL > 0:
            print(debug_output)


def set_debug_logs_file(file_path: str):
    """
    work in progress.
    """
    _check_log_file_path(file_path)
    global _DEBUG_FILE, _DEBUG_LEVEL
    _DEBUG_FILE = file_path
    _DEBUG_LEVEL = 1


def activate_debug():
    """
    work in progress.
    """
    set_debug_level(1)


def set_debug_level(n: int):
    """
    work in progress.
    """
    global _DEBUG_LEVEL
    _DEBUG_LEVEL = n


def deactivate_debugs():
    """
    work in progress.
    """
    global _DEBUG_LEVEL, _DEBUG_FILE
    if type(_DEBUG_FILE) == str and os.path.isfile(_DEBUG_FILE):
        os.rmdir(_DEBUG_FILE)
    _DEBUG_FILE = None
    _DEBUG_LEVEL = 0
