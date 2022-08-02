from typing import Callable, Dict

from colosseum.experiment.hyperopt.base import BaseAgentHyperparametersOptimizer
from colosseum.experiment.hyperopt.hyperopt_measures import compute_regret
from colosseum.experiment.hyperopt.ray.base_ray_opt import BaseRayOptimizer


class RayRegretOptimizer(BaseRayOptimizer, BaseAgentHyperparametersOptimizer):
    def get_function_to_optimize(self) -> Callable[[Dict[str, "Domain"]], float]:
        def tune_run(agent_next_hyperparams):
            return compute_regret(
                agent_next_hyperparams,
                self._get_debug_file(agent_next_hyperparams),
                self._mdp_classes,
                self._n_mdp_parameter_samples_from_mdp_classes,
                self._seed,
                self._optimization_horizon,
                self._n_seeds_per_agent_mdp_interaction,
                self._agent_class,
                self._max_agent_interaction_s,
                True,
                False,
                self._log_every,
            )

        return tune_run

    def optimize(self, n_cores: int, verbose: bool):
        return super(RayRegretOptimizer, self)._optimize(
            n_cores, compute_regret, verbose
        )
