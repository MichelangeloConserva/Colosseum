import abc
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, Union

import numpy as np
from scipy.stats import beta, rv_continuous

from colosseum.mdp import BaseMDP
from colosseum.mdp.utils.custom_samplers import NextStateSampler
from colosseum.utils.miscellanea import (
    check_distributions,
    deterministic,
    get_dist,
    rounding_nested_structure,
)

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE, NODE_TYPE


@dataclass(frozen=True)
class DeepSeaNode:
    """
    The node for the DeepSea MDP.
    """

    X: int
    """x coordinate."""
    Y: int
    """y coordinate."""

    def __str__(self):
        return f"X={self.X},Y={self.Y}"

    def __iter__(self):
        return iter((self.X, self.Y))


class DeepSeaAction(IntEnum):
    """
    The actions available in the DeepSea MDP.
    """

    LEFT = 0
    """Move towards the left."""
    RIGHT = 1
    """Move towards the right."""


class DeepSeaMDP(BaseMDP, abc.ABC):
    """
    The base class for the DeepSea family.
    """

    @staticmethod
    def get_unique_symbols() -> List[str]:
        return ["A", " "]

    @staticmethod
    def does_seed_change_MDP_structure() -> bool:
        return False

    @staticmethod
    def sample_mdp_parameters(
        n: int, is_episodic: bool, seed: int = None
    ) -> List[Dict[str, Any]]:
        rng = np.random.RandomState(np.random.randint(10_000) if seed is None else seed)
        samples = []
        for _ in range(n):
            sample = dict(
                size=int(
                    (1 + np.minimum((800 / (100 * rng.random() + 35)), 25))
                    * (0.8 if is_episodic else 1)
                ),
                p_rand=min(2 / (8 * rng.random() + 3), 0.95),
                make_reward_stochastic=rng.choice([True, False]),
                reward_variance_multiplier=2 * rng.random() + 0.005,
            )
            sample["p_rand"] = None if sample["p_rand"] < 0.01 else sample["p_rand"]

            if sample["make_reward_stochastic"]:
                sample["sub_optimal_distribution"] = (
                    "beta",
                    (
                        sample["reward_variance_multiplier"],
                        sample["reward_variance_multiplier"]
                        * (sample["size"] / 0.5 - 1),
                    ),
                )
                sample["optimal_distribution"] = (
                    "beta",
                    (
                        sample["reward_variance_multiplier"] * (sample["size"] / 1 - 1),
                        sample["reward_variance_multiplier"],
                    ),
                )
                sample["other_distribution"] = (
                    "beta",
                    (
                        sample["reward_variance_multiplier"],
                        sample["reward_variance_multiplier"]
                        * 10
                        * (sample["size"] / 0.5 - 1),
                    ),
                )
            else:
                sample["sub_optimal_distribution"] = (
                    "deterministic",
                    (1.0 / (sample["size"] ** 2),),
                )
                sample["optimal_distribution"] = ("deterministic", (1.0,))
                sample["other_distribution"] = ("deterministic", (0.0,))

            samples.append(rounding_nested_structure(sample))

        return samples

    @staticmethod
    def get_node_class() -> Type[DeepSeaNode]:
        return DeepSeaNode

    def get_gin_parameters(self, index: int) -> str:
        prms = dict(
            size=self._size,
            make_reward_stochastic=self._make_reward_stochastic,
            reward_variance_multiplier=self._reward_variance_multiplier,
            sub_optimal_distribution=(
                self._sub_optimal_distribution.dist.name,
                self._sub_optimal_distribution.args,
            ),
            optimal_distribution=(
                self._optimal_distribution.dist.name,
                self._optimal_distribution.args,
            ),
            other_distribution=(
                self._other_distribution.dist.name,
                self._other_distribution.args,
            ),
        )
        if self._p_rand is not None:
            prms["p_rand"] = self._p_rand

        return DeepSeaMDP.produce_gin_file_from_mdp_parameters(
            prms, type(self).__name__, index
        )

    @property
    def n_actions(self) -> int:
        return len(DeepSeaAction)

    def __init__(
        self,
        seed: int,
        size: int,
        optimal_return: float = 1.0,
        suboptimal_return: float = 0.5,
        optimal_distribution: Union[Tuple, rv_continuous] = None,
        sub_optimal_distribution: Union[Tuple, rv_continuous] = None,
        other_distribution: Union[Tuple, rv_continuous] = None,
        make_reward_stochastic=False,
        reward_variance_multiplier: float = 1.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        seed : int
            The seed used for sampling rewards and next states.
        size : int
            The size of the grid.
        optimal_return : float
            If the rewards are made stochastic, this parameter controls the mean reward for the optimal trajectory.
            By default, it is set to 1.
        suboptimal_return: float
            If the rewards are made stochastic, this parameter controls the mean reward for suboptimal trajectories.
            By default, it is set to 0.5.
        optimal_distribution : Union[Tuple, rv_continuous]
            The distribution of the highly rewarding state. It can be either passed as a tuple containing Beta parameters
            or as a rv_continuous object.
        sub_optimal_distribution : Union[Tuple, rv_continuous]
            The distribution of the suboptimal rewarding states. It can be either passed as a tuple containing Beta
            parameters or as a rv_continuous object.
        other_distribution : Union[Tuple, rv_continuous]
            The distribution of the other states. It can be either passed as a tuple containing Beta parameters or as a
            rv_continuous object.
        make_reward_stochastic : bool
            If True, the rewards of the MDP will be stochastic. By default, it is set to False.
        reward_variance_multiplier : float
            A constant that can be used to increase the variance of the reward distributions without changing their means.
            The lower the value, the higher the variance. By default, it is set to 1.
        """

        if type(sub_optimal_distribution) == tuple:
            sub_optimal_distribution = get_dist(
                sub_optimal_distribution[0], sub_optimal_distribution[1]
            )
        if type(optimal_distribution) == tuple:
            optimal_distribution = get_dist(
                optimal_distribution[0], optimal_distribution[1]
            )
        if type(other_distribution) == tuple:
            other_distribution = get_dist(other_distribution[0], other_distribution[1])

        self._size = size
        self._optimal_return = optimal_return
        self._suboptimal_return = suboptimal_return
        self._optimal_distribution = optimal_distribution
        self._sub_optimal_distribution = sub_optimal_distribution
        self._other_distribution = other_distribution

        dists = [
            sub_optimal_distribution,
            optimal_distribution,
            other_distribution,
        ]
        if dists.count(None) == 0:
            self._sub_optimal_distribution = sub_optimal_distribution
            self._optimal_distribution = optimal_distribution
            self._other_distribution = other_distribution
        else:
            if make_reward_stochastic:
                self._sub_optimal_distribution = beta(
                    reward_variance_multiplier,
                    reward_variance_multiplier * (size / self._suboptimal_return - 1),
                )
                self._optimal_distribution = beta(
                    reward_variance_multiplier * (size / self._optimal_return - 1),
                    reward_variance_multiplier,
                )
                self._other_distribution = beta(
                    reward_variance_multiplier,
                    reward_variance_multiplier
                    * 10
                    * (size / self._suboptimal_return - 1),
                )
            else:
                self._sub_optimal_distribution = deterministic(1.0 / (size ** 2))
                self._optimal_distribution = deterministic(1.0)
                self._other_distribution = deterministic(0.0)

        super(DeepSeaMDP, self).__init__(
            seed=seed,
            reward_variance_multiplier=reward_variance_multiplier,
            make_reward_stochastic=make_reward_stochastic,
            **kwargs,
        )

    @property
    def _possible_starting_nodes(self) -> List[DeepSeaNode]:
        return [DeepSeaNode(0, self._size - 1)]

    def _get_next_nodes_parameters(
        self, node: "NODE_TYPE", action: "ACTION_TYPE"
    ) -> Tuple[Tuple[dict, float], ...]:
        if node.Y == 0:
            return ((dict(X=0, Y=self._size - 1), 1.0),)

        return (
            (
                dict(
                    X=min(node.X + 1, self._size - 1)
                    if action == DeepSeaAction.RIGHT
                    else max(node.X - 1, 0),
                    Y=max(0, node.Y - 1),
                ),
                1.0,
            ),
        )

    def _get_reward_distribution(
        self, node: "NODE_TYPE", action: "ACTION_TYPE", next_node: "NODE_TYPE"
    ) -> rv_continuous:
        return (
            self._optimal_distribution
            if node.X == self._size - 1
            and node.Y == 0
            and action == DeepSeaAction.RIGHT
            else (
                self._sub_optimal_distribution
                if action == DeepSeaAction.LEFT
                else self._other_distribution
            )
        )

    def _get_starting_node_sampler(self) -> NextStateSampler:
        return NextStateSampler(next_nodes=self._possible_starting_nodes)

    def _check_parameters_in_input(self):
        super(DeepSeaMDP, self)._check_parameters_in_input()

        assert self._size > 1

        # No lazy mechanic for DeepSea
        assert self._p_lazy is None

        assert self._suboptimal_return < self._optimal_return - 0.1

        dists = [
            self._sub_optimal_distribution,
            self._optimal_distribution,
            self._other_distribution,
        ]
        check_distributions(
            dists,
            self._make_reward_stochastic,
        )

    def _get_grid_representation(self, node: "NODE_TYPE") -> np.ndarray:
        grid = np.zeros((self._size, self._size), dtype=str)
        grid[:, :] = " "
        grid[node.Y, node.X] = "A"
        return grid[::-1, :]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            **super(DeepSeaMDP, self).parameters,
            **dict(
                size=self._size,
                optimal_return=self._optimal_return,
                suboptimal_return=self._suboptimal_return,
                optimal_distribution=self._optimal_distribution,
                sub_optimal_distribution=self._sub_optimal_distribution,
                other_distribution=self._other_distribution,
            ),
        }
