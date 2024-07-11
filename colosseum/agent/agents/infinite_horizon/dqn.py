from typing import Any, Dict, TYPE_CHECKING

import gin
import numpy as np
import sonnet as snt
import tensorflow as tf
from bsuite.baselines.tf.dqn import DQN
from ray import tune

from colosseum.dynamic_programming.utils import argmax_2d
from colosseum.utils.non_tabular.bsuite import NonTabularBsuiteAgentWrapper

if TYPE_CHECKING:
    from colosseum.agent.agents.base import BaseAgent
    from colosseum.utils.acme.specs import MDPSpec


@gin.configurable
class DQNContinuous(NonTabularBsuiteAgentWrapper):
    """
    The wrapper for the `DQN` agent from `bsuite`.
    """

    @staticmethod
    def produce_gin_file_from_parameters(parameters: Dict[str, Any], index: int = 0):
        string = ""
        for k, v in parameters.items():
            string += f"prms_{index}/DQNContinuous.{k} = {v}\n"
        return string[:-1]

    @staticmethod
    def is_episodic() -> bool:
        return False

    @staticmethod
    def get_hyperparameters_search_spaces() -> Dict[str, tune.search.sample.Domain]:
        return {
            "network_width": tune.choice([64, 128, 256]),
            "network_depth": tune.choice([2, 4]),
            "batch_size": tune.choice([32, 64, 128]),
            "sgd_period": tune.choice([1, 4, 8]),
            "target_update_period": tune.choice([4, 16, 32]),
            "epsilon": tune.choice([0.01, 0.05, 0.1]),
        }

    @staticmethod
    def get_agent_instance_from_parameters(
        seed: int,
        optimization_horizon: int,
        mdp_specs: "MDPSpec",
        parameters: Dict[str, Any],
    ) -> "BaseAgent":

        return DQNContinuous(
            seed,
            mdp_specs,
            optimization_horizon,
            parameters["network_width"],
            parameters["network_depth"],
            parameters["batch_size"],
            parameters["sgd_period"],
            parameters["target_update_period"],
            parameters["epsilon"],
        )

    @property
    def current_optimal_stochastic_policy(self) -> np.ndarray:
        qvals = tf.stop_gradient(
            self._agent._forward(self.emission_map.all_observations)
        ).numpy()
        return argmax_2d(qvals)

    def __init__(
        self,
        seed: int,
        mdp_specs: "MDPSpec",
        optimization_horizon: int,
        # MDP model parameters
        network_width: int,
        network_depth: int,
        batch_size: int,
        sgd_period: int,
        target_update_period: int,
        # Actor parameters
        epsilon: float,
    ):
        r"""
        Parameters
        ----------
        seed : int
            The random seed.
        mdp_specs : MDPSpec
            The full specification of the MDP.
        optimization_horizon : int
            The total number of interactions that the agent is expected to have with the MDP.
        network_width : int
            The width of the neural networks of the agent.
        network_depth : int
            The depth of the neural networks of the agent.
        batch_size : int
            The batch size for training the agent.
        sgd_period : int
            The stochastic gradient descent update period.
        target_update_period : int
            The interval length between updating the target network.
        epsilon : Callable[[int], float]]
            The :math:`\epsilon` greedy probability as a function of the time.
        """

        tf.random.set_seed(seed)
        np.random.seed(seed)

        network = snt.Sequential(
            [
                snt.Flatten(),
                snt.nets.MLP(
                    [network_width] * network_depth + [mdp_specs.actions.num_values]
                ),
            ]
        )
        optimizer = snt.optimizers.Adam(learning_rate=1e-3)

        agent = DQN(
            action_spec=mdp_specs.actions,
            network=network,
            batch_size=batch_size,
            discount=0.99,
            replay_capacity=10000,
            min_replay_size=100,
            sgd_period=sgd_period,
            target_update_period=target_update_period,
            optimizer=optimizer,
            epsilon=epsilon,
            seed=seed,
        )

        super(DQNContinuous, self).__init__(seed, agent, mdp_specs)
