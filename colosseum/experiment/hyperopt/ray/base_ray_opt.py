import abc
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Tuple, Type

import ray
from ray import tune

from colosseum.experiment.hyperopt.base import BaseAgentHyperparametersOptimizer

if TYPE_CHECKING:
    from ray.tune.sample import Domain

    from colosseum.mdp import BaseMDP
    from colosseum.agent.agents.base import BaseAgent


class BaseRayOptimizer(BaseAgentHyperparametersOptimizer, abc.ABC):
    @abc.abstractmethod
    def get_function_to_optimize(self) -> Callable[[Dict[str, "Domain"]], float]:
        pass

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

        tune_run = self.get_function_to_optimize()

        ray.init(num_cpus=n_cores, verbose=verbose)
        analysis = tune.run(
            tune_run,
            config=self._agent_class.get_hyperparameters_search_spaces(),
            num_samples=int(1e9),
            time_budget_s=self._max_optimization_time_budget_s,
        )
        ray.shutdown()

        results = []
        for t in analysis.trials:
            if "regret" in t.last_result:
                results.append((t.last_result["config"], t.last_result["regret"]))
        return results
