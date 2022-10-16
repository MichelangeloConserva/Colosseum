from typing import Dict, Any, TYPE_CHECKING

import gin
import numpy as np
import sonnet as snt
import tensorflow as tf
from bsuite.baselines.tf.actor_critic import PolicyValueNet, ActorCritic
from ray import tune

from colosseum.utils.non_tabular.bsuite import NonTabularBsuiteAgentWrapper

if TYPE_CHECKING:
    from colosseum.agent.agents.base import BaseAgent
    from colosseum.utils.acme.specs import MDPSpec


@gin.configurable
class ActorCriticContinuous(NonTabularBsuiteAgentWrapper):
    """
    The wrapper for the `ActorCritic` agent from `bsuite`.
    """

    @staticmethod
    def produce_gin_file_from_parameters(parameters: Dict[str, Any], index: int = 0):
        string = ""
        for k, v in parameters.items():
            string += f"prms_{index}/ActorCriticContinuous.{k} = {v}\n"
        return string[:-1]

    @staticmethod
    def is_episodic() -> bool:
        return False

    @staticmethod
    def get_hyperparameters_search_spaces() -> Dict[str, tune.sample.Domain]:
        return {
            "network_width": tune.choice([64, 128, 256]),
            "network_depth": tune.choice([2, 4]),
            "max_sequence_length": tune.choice([16, 32, 64, 128]),
            "td_lambda": tune.choice([0.7, 0.8, 0.9]),
        }

    @staticmethod
    def get_agent_instance_from_parameters(
        seed: int,
        optimization_horizon: int,
        mdp_specs: "MDPSpec",
        parameters: Dict[str, Any],
    ) -> "BaseAgent":

        return ActorCriticContinuous(
            seed,
            mdp_specs,
            optimization_horizon,
            parameters["network_width"],
            parameters["network_depth"],
            parameters["max_sequence_length"],
            parameters["td_lambda"],
        )

    @property
    def current_optimal_stochastic_policy(self) -> np.ndarray:
        logits = tf.stop_gradient(
            self._agent._network(
                tf.convert_to_tensor(self.emission_map.all_observations)
            )[0].logits
        ).numpy()

        return (logits >= logits.max(-1, keepdims=True)).astype(np.float32)

    def __init__(
        self,
        seed: int,
        mdp_specs: "MDPSpec",
        optimization_horizon: int,
        # MDP model parameters
        network_width: int,
        network_depth: int,
        max_sequence_length: int,
        td_lambda: float,
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
        max_sequence_length : int
            The maximum sequence length for training the agent.
        td_lambda : float
            The TD(:math:`\lambda`) parameter for training the agent.
        """

        tf.random.set_seed(seed)
        np.random.seed(seed)

        network = PolicyValueNet(
            hidden_sizes=[network_width] * network_depth,
            action_spec=mdp_specs.actions,
        )
        agent = ActorCritic(
            obs_spec=mdp_specs.observations,
            action_spec=mdp_specs.actions,
            network=network,
            optimizer=snt.optimizers.Adam(learning_rate=3e-3),
            max_sequence_length=max_sequence_length,
            td_lambda=td_lambda,
            discount=0.99,
            seed=seed,
        )
        super(ActorCriticContinuous, self).__init__(seed, agent, mdp_specs)
