import os
import shutil
from collections import namedtuple
from glob import glob
from typing import Type, TYPE_CHECKING, List

from colosseum.benchmark import BENCHMARKS_DIRECTORY
from colosseum.utils import ensure_folder

if TYPE_CHECKING:
    from colosseum.agent.agents.base import BaseAgent

_AVAILABLE_CORES = os.cpu_count() - 2
VERBOSE_LEVEL = 0
DEBUG_LEVEL = 0
DEBUG_FILE = None

_EXPERIMENTS_TO_RUN_FOLDER = "experiments_to_run" + os.sep
_EXPERIMENTS_RESULTS_FOLDER = "experiments_results" + os.sep
_HARDNESS_RESULTS_FOLDER = "hardness_analysis" + os.sep
_HARDNESS_MEASURES_CACHE_FOLDER = "hardness_measures_cache" + os.sep

EXPERIMENT_SEPARATOR_PRMS = "-"
EXPERIMENT_SEPARATOR_MDP_AGENT = "____"

_REGISTERED_EXTERNAL_AGENT_CLASSES = list()


def register_agent_class(agent_class: Type["BaseAgent"]):
    _REGISTERED_EXTERNAL_AGENT_CLASSES.append(agent_class)

def get_external_agent_classes() -> List[Type["BaseAgent"]]:
    return _REGISTERED_EXTERNAL_AGENT_CLASSES


def set_up_hardness_measures_cache_folder():
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
    global _HARDNESS_MEASURES_CACHE_FOLDER
    _HARDNESS_MEASURES_CACHE_FOLDER = ensure_folder(path)


def get_hardness_measures_cache_folder() -> str:
    if not os.path.isdir(_HARDNESS_MEASURES_CACHE_FOLDER):
        set_up_hardness_measures_cache_folder()
    return _HARDNESS_MEASURES_CACHE_FOLDER


def set_hardness_analysis_folder(path: str):
    global _HARDNESS_RESULTS_FOLDER
    _HARDNESS_RESULTS_FOLDER = ensure_folder(path)


def get_hardness_analysis_folder() -> str:
    os.makedirs(_HARDNESS_RESULTS_FOLDER, exist_ok=True)
    return _HARDNESS_RESULTS_FOLDER


def set_experiment_to_run_folder(path: str):
    global _EXPERIMENTS_TO_RUN_FOLDER
    _EXPERIMENTS_TO_RUN_FOLDER = ensure_folder(path)


def get_experiment_to_run_folder() -> str:
    os.makedirs(_EXPERIMENTS_TO_RUN_FOLDER, exist_ok=True)
    return _EXPERIMENTS_TO_RUN_FOLDER


def set_experiment_results_folder(path: str):
    global _EXPERIMENTS_RESULTS_FOLDER
    _EXPERIMENTS_RESULTS_FOLDER = ensure_folder(path)


def get_experiment_result_folder() -> str:
    os.makedirs(_EXPERIMENTS_RESULTS_FOLDER, exist_ok=True)
    return _EXPERIMENTS_RESULTS_FOLDER


def disable_multiprocessing():
    set_available_cores(-1)


def set_available_cores(n: int):
    global _AVAILABLE_CORES
    _AVAILABLE_CORES = n


def get_available_cores() -> int:
    return _AVAILABLE_CORES


def process_debug_output(debug_output):
    if DEBUG_FILE:
        with open(DEBUG_FILE, "a") as f:
            f.write(debug_output + "\n")
    else:
        if DEBUG_LEVEL > 0:
            print(debug_output)


def set_debug_logs_file(file_path: str):
    _check_log_file_path(file_path)
    global DEBUG_FILE, DEBUG_LEVEL
    DEBUG_FILE = file_path
    DEBUG_LEVEL = 1


def activate_debug():
    set_debug_level(1)


def set_debug_level(n: int):
    global DEBUG_LEVEL
    DEBUG_LEVEL = n


def deactivate_debugs():
    global DEBUG_LEVEL, DEBUG_FILE
    if type(DEBUG_FILE) == str and os.path.isfile(DEBUG_FILE):
        os.rmdir(DEBUG_FILE)
    DEBUG_FILE = None
    DEBUG_LEVEL = 0


def activate_verbose_logging():
    global VERBOSE_LEVEL
    VERBOSE_LEVEL = 1


def set_verbose_logs_file(file_path: str):
    _check_log_file_path(file_path)
    global VERBOSE_LEVEL
    VERBOSE_LEVEL = file_path


def disable_verbose_logging():
    global VERBOSE_LEVEL
    VERBOSE_LEVEL = 0


def _check_log_file_path(file_path: str):
    assert ".txt", "The file extension should be txt."
    if not os.path.isfile(file_path):
        with open(file_path, "w") as f:
            pass
