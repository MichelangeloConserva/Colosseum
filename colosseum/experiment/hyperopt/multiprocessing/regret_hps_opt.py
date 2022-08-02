from colosseum.experiment.hyperopt.base import BaseAgentHyperparametersOptimizer
from colosseum.experiment.hyperopt.hyperopt_measures import compute_regret
from colosseum.experiment.hyperopt.multiprocessing.base_mp_opt import (
    BaseMultiProcessingOptimizer,
)


class MPRegretOptimizer(
    BaseMultiProcessingOptimizer, BaseAgentHyperparametersOptimizer
):
    def optimize(self, n_cores: int, verbose: bool):
        return super(MPRegretOptimizer, self)._optimize(
            n_cores, compute_regret, verbose
        )
