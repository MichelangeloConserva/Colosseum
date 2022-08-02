import abc
import datetime
import os
import shutil
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Tuple, Type

import numpy as np
import toolz

from colosseum.utils import clean_for_storing, ensure_folder, get_colosseum_mdp_classes

if TYPE_CHECKING:
    from ray.tune.sample import Domain

    from colosseum.mdp import BaseMDP
    from colosseum.agent.agents.base import BaseAgent


class HyperparametersOptimizationFailed(Exception):
    pass


def agent_hyperparameters_generator(
    seed: int, agent_hyperparameters_sampler: Dict[str, "Domain"]
) -> Generator:
    """
    returns a generator that yields a dictionary containing an instance of agent hyperparameters sampled from the
    agent hyperparameters sampler given in input.
    """
    np.random.seed(seed)
    while True:
        yield toolz.valmap(lambda x: x.sample(), agent_hyperparameters_sampler)


@dataclass()
class HyperparameterOptimizationConfiguration:
    optimization_horizon: int
    max_agent_interaction_s: float
    max_optimization_time_budget_s: float
    n_seeds_per_agent_mdp_interaction: int
    n_mdp_parameter_samples_from_mdp_classes: int
    log_every: int


DEFAULT_HYPERPARAMETERS_OPTIMIZATION_CONFIGURATION = (
    HyperparameterOptimizationConfiguration(
        optimization_horizon=250_000,
        max_agent_interaction_s=2 * 60,
        max_optimization_time_budget_s=3 * 60 * 60,
        n_seeds_per_agent_mdp_interaction=3,
        n_mdp_parameter_samples_from_mdp_classes=3,
        log_every=100_000,
    )
)

SMALL_HYPERPARAMETERS_OPTIMIZATION_CONFIGURATION = (
    HyperparameterOptimizationConfiguration(
        optimization_horizon=30_000,
        max_agent_interaction_s=0.7 * 60,
        max_optimization_time_budget_s=2 * 60,
        n_seeds_per_agent_mdp_interaction=2,
        n_mdp_parameter_samples_from_mdp_classes=1,
        log_every=10_000,
    )
)


class BaseAgentHyperparametersOptimizer:
    def __init__(
        self,
        agent_class: Type["BaseAgent"],
        hpoc: HyperparameterOptimizationConfiguration = DEFAULT_HYPERPARAMETERS_OPTIMIZATION_CONFIGURATION,
        seed: int = 0,
        log_folder: str = f"tmp{os.sep}hyperopt{os.sep}",
        overwrite_previous_logs: bool = True,
        mdp_classes: List["BaseMDP"] = None,
    ):
        """
        is the base class for the agent hyperparameter optimizer.

        Parameters
        ----------
        agent_class : Type["BaseAgent"]
            is the agent class for which the hyperparameters will be optimized.
        hpoc : HyperparameterOptimizationConfiguration
            is the hyperparameter optimization configuration with the following parameters:
            optimization_horizon : int, optional
                is the optimization horizon that will be used in the agent/MDP interactions.
            max_agent_interaction_s : float, optional
                is the maximum amount of training time given to the agent for each agent/MDp interaction.
            max_optimization_time_budget_s : float, optional
                is the maximum amount of time to search the best hyperparameters.
            n_seeds_per_agent_mdp_interaction : int, optional
                is the number of seeds for which each agent/MDP interaction is repeated.
            n_mdp_parameter_samples_from_mdp_classes : int, optional
                is the number of parameters sampled from the MDP classes and that results in MDP instance that the agent.
                will interact with during the hyperparameter optimization.
        seed : int, optional
            is the seed used for sampling the MDP parameters and the agent hyperparameters.
        log_folder : str, optional
            is the folder where the logs from the hyperparameter optimization procedure can be stored.
        overwrite_previous_logs : bool, optional
            checks whether to remove the previous logs in the given folder.
        """

        assert (
            0.7 * hpoc.max_optimization_time_budget_s
            >= hpoc.max_agent_interaction_s
            * hpoc.n_seeds_per_agent_mdp_interaction
            * hpoc.n_mdp_parameter_samples_from_mdp_classes
        ), (
            "You are not giving enough total optimization time for the selected number of interactions. "
            "Make sure that at least 0.5 * max_optimization_time_budget_s > max_agent_interaction_s * "
            "n_seeds_per_agent_mdp_interaction * n_mdp_parameter_samples_from_mdp_classes so that you can obtain two"
            "full samples of the regrets from all the MDPs for each of your cores."
        )

        # We add 30 seconds to allow for computing the quantities related to the final score like the optimal
        # average reward
        hpoc.max_agent_interaction_s += 30

        # The total time necessary to compute a measure given the number of seeds and the number of MDP parameters to be
        # sampled
        total_time_to_compute_single_measure = (
            hpoc.max_agent_interaction_s
            * hpoc.n_seeds_per_agent_mdp_interaction
            * hpoc.n_mdp_parameter_samples_from_mdp_classes
        )

        # The effective amount of time required to compute the maximum number of scores given the total time constraint.
        # All the additional time would be wasted since it would not be possible to compute the score.
        max_optimization_time_budget_s = total_time_to_compute_single_measure * max(
            1,
            int(
                hpoc.max_optimization_time_budget_s
                / total_time_to_compute_single_measure
            ),
        )

        self._agent_class = agent_class
        self._mdp_classes = (
            get_colosseum_mdp_classes(self._agent_class.is_episodic())
            if mdp_classes is None
            else mdp_classes
        )
        self._optimization_horizon = hpoc.optimization_horizon
        self._max_agent_interaction_s = hpoc.max_agent_interaction_s
        self._max_optimization_time_budget_s = max_optimization_time_budget_s
        self._n_seeds_per_agent_mdp_interaction = hpoc.n_seeds_per_agent_mdp_interaction
        self._n_mdp_parameter_samples_from_mdp_classes = (
            hpoc.n_mdp_parameter_samples_from_mdp_classes
        )
        self._seed = seed
        self._log_every = hpoc.log_every

        self._rng = np.random.RandomState(seed)
        self._log_folder = ensure_folder(log_folder) + agent_class.__name__ + os.sep
        self._rng.shuffle(self._mdp_classes)

        if overwrite_previous_logs and os.path.isdir(self._log_folder):
            shutil.rmtree(self._log_folder)
        os.makedirs(self._log_folder, exist_ok=True)

    def _get_debug_file(self, agent_next_hyperparams: Dict[str, Any]) -> str:
        """
        returns a txt file path corresponding to the given agent hyperparameter instance where the logs can be stored.
        """
        agent_next_hyperparams_str = map(
            lambda x: "_".join(map(str, x[0])),
            zip(clean_for_storing(agent_next_hyperparams).items()),
        )
        agent_next_hyperparams_str = "__".join(agent_next_hyperparams_str)
        debug_file_path = self._log_folder + agent_next_hyperparams_str + ".txt"
        if os.path.isfile(debug_file_path):
            os.remove(debug_file_path)
        return debug_file_path

    @abc.abstractmethod
    def _multi_process_optimization(
        self,
        n_cores: int,
        gen: Generator[Dict[str, "Domain"], None, None],
        compute_measure: Callable[
            [
                Dict[str, Any],
                str,
                List[Type["BaseMDP"]],
                int,
                int,
                int,
                int,
                Type["BaseAgent"],
                float,
                bool,
                bool,
                int,
            ],
            Tuple[Dict[str, Any], float],
        ],
        verbose: bool,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        performs the hyperparameter optimization procedure using n_cores.

        Parameters
        ----------
        n_cores : int
            is the number of processes available for the hyperparameters optimization procedure.
        gen : Generator[Dict[str, "Domain"], None, None]
            is a generator that yield instances of agent hyperparameters.
        compute_measure : Callable
            is the function that is optimized in the hyperparameter optimization procedure. See "compute_regret" in
            colosseum.experiments.hyperopt.utils for an example of such function.
        Returns
        -------
        returns a list of tuples containing hyperparameters and their corresponding scores.
        """

    def single_process_optimization(
        self,
        gen: Generator[Dict[str, "Domain"], None, None],
        compute_measure: Callable[
            [
                Dict[str, Any],
                str,
                List[Type["BaseMDP"]],
                int,
                int,
                int,
                int,
                Type["BaseAgent"],
                float,
                bool,
                bool,
                int,
            ],
            Tuple[Dict[str, Any], float],
        ],
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        performs the hyperparameter optimization procedure in single thread.

        Parameters
        ----------
        gen : Generator[Dict[str, "Domain"], None, None]
            is a generator that yield instances of agent hyperparameters.
        compute_measure : Callable
            is the function that is optimized in the hyperparameter optimization procedure. See "compute_regret" in
            colosseum.experiments.hyperopt.utils for an example of such function.
        Returns
        -------
        returns a list of tuples containing hyperparameters and their corresponding scores.
        """

        results = []
        start = time.time()
        remaining_time = self._max_optimization_time_budget_s - (time.time() - start)
        while remaining_time > 0:
            print(
                "\r",
                f"{datetime.timedelta(seconds=int(remaining_time))} time left with {len(results)} completed measures.",
                end="",
            )
            agent_next_hyperparams = next(gen)
            agent_hyperparameters, score = compute_measure(
                agent_next_hyperparams,
                self._get_debug_file(agent_next_hyperparams),
                self._mdp_classes,
                self._n_mdp_parameter_samples_from_mdp_classes,
                self._seed,
                self._optimization_horizon,
                self._n_seeds_per_agent_mdp_interaction,
                self._agent_class,
                self._max_agent_interaction_s,
                False,
                False,
                self._log_every,
            )
            results.append((agent_hyperparameters, score))
            remaining_time = self._max_optimization_time_budget_s - (
                time.time() - start
            )
        return results

    @abc.abstractmethod
    def optimize(
        self, n_cores: int, verbose: bool
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        returns a tuples containing the best hyperparameters and the corresponding score. When implementing this function,
        you should internally define the function that computes the score and call the self._optimize function.

        Parameters
        ----------
        n_cores : int
            is the number of processes available for the hyperparameters optimization procedure. Value less than one
            correspond to single thread execution.
        verbose : bool
            checks whether the progress in the hyperparameters optimization procedure are printed in the console.
        """

    def _optimize(
        self,
        n_cores: int,
        compute_measure: Callable[
            [
                Dict[str, Any],
                str,
                List[Type["BaseMDP"]],
                int,
                int,
                int,
                int,
                Type["BaseAgent"],
                float,
            ],
            Tuple[Dict[str, Any], float],
        ],
        verbose: bool,
    ) -> Tuple[Dict[str, Any], float]:
        """
        returns a tuples containing the best hyperparameters and the corresponding score.
        """
        agent_hyperparams_gen = agent_hyperparameters_generator(
            self._seed, self._agent_class.get_hyperparameters_search_spaces()
        )

        from colosseum import (
            disable_multiprocessing,
            get_available_cores,
            set_available_cores,
        )

        # Ensure that Colosseum does not use multiprocessing while we perform hyperparameter optimization
        colosseum_cores_config = get_available_cores()
        disable_multiprocessing()

        if verbose:
            print(f"Random search started for {self._agent_class.__name__}.")
        start = time.time()
        if n_cores > 1:
            results = self._multi_process_optimization(
                n_cores, agent_hyperparams_gen, compute_measure, verbose
            )
        else:
            results = self.single_process_optimization(
                agent_hyperparams_gen, compute_measure
            )

        # Reactive previous multiprocessing configuration
        set_available_cores(colosseum_cores_config)

        if len(results) == 0:
            raise HyperparametersOptimizationFailed()

        best_hyperparams_and_score = max(results, key=lambda x: -x[1])
        if verbose:
            print(
                f"Search took {datetime.timedelta(seconds=int(time.time() - start))} "
                f"and computed {len(results)} full agent/MDP interactions."
            )
            print(
                f"The best hyperparameters are: {toolz.valmap(lambda x: np.round(x, 3), best_hyperparams_and_score[0])} "
                f"with a total regret of {min(x[1] for x in results):.5f}"
            )
            print(
                f"The worst hyperparameters obtained {max(x[1] for x in results):.5f} regret "
                f"and the average over all the regrets is "
                f"{np.mean([x[1] for x in results]):.5f} plus minus {np.std([x[1] for x in results]):.5f}."
            )
        return best_hyperparams_and_score
