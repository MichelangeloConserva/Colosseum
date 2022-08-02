import abc
import concurrent
import datetime
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Tuple, Type

from pebble import ProcessPool

from colosseum.experiment.hyperopt.base import BaseAgentHyperparametersOptimizer
from colosseum.utils.miscellanea import get_colosseum_mdp_classes

if TYPE_CHECKING:
    from ray.tune.sample import Domain

    from colosseum.mdp import BaseMDP
    from colosseum.agent.agents.base import BaseAgent


class BaseMultiProcessingOptimizer(BaseAgentHyperparametersOptimizer, abc.ABC):
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
    ):
        results = []
        processes_pool = []
        start = time.time()
        with ProcessPool(max_workers=n_cores) as p:
            remaining_time = self._max_optimization_time_budget_s - (
                time.time() - start
            )
            while remaining_time > 0:
                if verbose:
                    print(
                        "\r",
                        f"{datetime.timedelta(seconds=int(remaining_time))} time left with {len(results)} completed measures.",
                        end="",
                    )
                while len(processes_pool) < n_cores:
                    agent_next_hyperparams = next(gen)
                    args = (
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
                        True,
                        self._log_every,
                    )
                    processes_pool.append(p.schedule(compute_measure, args))

                # time.sleep(5)
                for i, p_res in enumerate(processes_pool):
                    if p_res.done():
                        # try:
                        agent_hyperparameters, score = p_res.result()
                        results.append((agent_hyperparameters, score))
                        processes_pool.remove(p_res)
                        # except pebble.common.ProcessExpired or pebble.common.ProcessExpired:
                        #     pass

                remaining_time = self._max_optimization_time_budget_s - (
                    time.time() - start
                )

            # One last chance
            for p_res in processes_pool:
                try:
                    agent_hyperparameters, score = p_res.result(timeout=0.1)
                    results.append((agent_hyperparameters, score))
                except concurrent.futures.TimeoutError:
                    p_res.cancel()
            if verbose:
                print(f"Search terminated with {len(results)} samples.")
        return results
