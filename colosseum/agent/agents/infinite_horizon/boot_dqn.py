from typing import Any, Dict, Callable, TYPE_CHECKING

import dm_env
import gin
import numpy as np
import sonnet as snt
import tensorflow as tf
from bsuite.baselines.tf.boot_dqn import BootstrappedDqn, make_ensemble
from ray import tune

from colosseum.dynamic_programming.utils import argmax_2d
from colosseum.utils.non_tabular.bsuite import NonTabularBsuiteAgentWrapper

if TYPE_CHECKING:
    from colosseum.agent.agents.base import BaseAgent
    from colosseum.utils.acme.specs import MDPSpec
    from colosseum.mdp import ACTION_TYPE


@gin.configurable
class BootDQNContinuous(NonTabularBsuiteAgentWrapper):
    """
    The wrapper for the `BootDQN` agent from `bsuite`.
    """

    @staticmethod
    def produce_gin_file_from_parameters(parameters: Dict[str, Any], index: int = 0):
        string = ""
        for k, v in parameters.items():
            string += f"prms_{index}/BootDQNContinuous.{k} = {v}\n"
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
            "mask_prob": tune.choice([0.8, 0.9, 1.0]),
            "noise_scale": tune.choice([0.0, 0.05, 0.1]),
            "n_ensemble": tune.choice([8, 16, 20]),
        }

    @staticmethod
    def get_agent_instance_from_parameters(
        seed: int,
        optimization_horizon: int,
        mdp_specs: "MDPSpec",
        parameters: Dict[str, Any],
    ) -> "BaseAgent":
        return BootDQNContinuous(
            seed,
            mdp_specs,
            optimization_horizon,
            parameters["network_width"],
            parameters["network_depth"],
            parameters["batch_size"],
            parameters["sgd_period"],
            parameters["target_update_period"],
            parameters["mask_prob"],
            parameters["noise_scale"],
            parameters["n_ensemble"],
        )

    @property
    def current_optimal_stochastic_policy(self) -> np.ndarray:
        qvals = tf.stop_gradient(
            self._agent._forward[self._agent._active_head](
                self.emission_map.all_observations
            )
        ).numpy()
        policy = argmax_2d(qvals)
        assert np.isclose(policy.sum(-1).mean(), 1)
        return policy

    def step_update(
        self, ts_t: dm_env.TimeStep, a_t: "ACTION_TYPE", ts_tp1: dm_env.TimeStep, h: int
    ):
        if self._rng_fast.random() < self._switch_prob:
            self._agent._active_head = self._rng.randint(self._agent._num_ensemble)
        super(BootDQNContinuous, self).step_update(ts_t, a_t, ts_tp1, h)

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
        mask_prob: float,
        noise_scale: float,
        n_ensemble: int,
        switch_prob: float = 0.1,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        replay_capacity: int = 10000,
        epsilon_fn: Callable[[int], float] = lambda t: 0,  # lambda t: 10 / (10 + t)
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
        mask_prob : float
            The masking probability for the bootstrapping procedure.
        noise_scale : float
            The scale of the Gaussian noise_class added to the value estimates.
        n_ensemble : int
            The number of ensembles.
        switch_prob : float
            The probability of changing the ensemble whose q-estimates are used to select actions. By default, it is set
            to :math:`0.1`.
        learning_rate : float
            The learning rate of the optimizer. By default, it is set to 1e-3.
        discount : float
            The discount factor.
        replay_capacity : int
            The maximum capacity of the replay buffer. By default, it is set to 10 000.
        epsilon_fn : Callable[[int], float]]
            The :math:`\epsilon` greedy probability as a function of the time. By default, it is set to zero.
        """

        self._switch_prob = switch_prob

        tf.random.set_seed(seed)
        np.random.seed(seed)

        ensemble = make_ensemble(
            mdp_specs.actions.num_values, n_ensemble, network_depth, network_width
        )
        optimizer = snt.optimizers.Adam(learning_rate=learning_rate)

        agent = BootstrappedDqn(
            obs_spec=mdp_specs.observations,
            action_spec=mdp_specs.actions,
            ensemble=ensemble,
            batch_size=batch_size,
            discount=discount,
            replay_capacity=replay_capacity,
            min_replay_size=batch_size,
            sgd_period=sgd_period,
            target_update_period=target_update_period,
            optimizer=optimizer,
            mask_prob=mask_prob,
            noise_scale=noise_scale,
            seed=seed,
            epsilon_fn=epsilon_fn,
        )
        super(BootDQNContinuous, self).__init__(seed, agent, mdp_specs)
