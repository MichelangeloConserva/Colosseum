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
class RiverSwimNode:
    """
    The node for the RiverSwim MDP.
    """

    X: int
    """x coordinate."""

    def __str__(self):
        return f"X={self.X}"

    def __iter__(self):
        return iter((self.X, self.X))


class RiverSwimAction(IntEnum):
    """
    The actions available in the RiverSwim MDP.
    """

    LEFT = 0
    RIGHT = 1


class RiverSwimMDP(BaseMDP, abc.ABC):
    """
    The base class for the RiverSwim family.
    """

    @staticmethod
    def get_unique_symbols() -> List[str]:
        return [" ", "A", "S", "G"]

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
            p_rand, p_lazy, _ = 0.9 * rng.dirichlet([0.2, 0.2, 5])
            sample = dict(
                size=int(np.minimum(2.5 + (200 / (45 * rng.random() + 11)), 25))
                if is_episodic
                else int((6 * rng.random() + 2) ** 2.2),
                make_reward_stochastic=rng.choice([True, False]),
                p_rand=p_rand,
                p_lazy=p_lazy,
                reward_variance_multiplier=2 * rng.random() + 0.005,
            )
            sample["p_rand"] = None if sample["p_rand"] < 0.01 else sample["p_rand"]
            sample["p_lazy"] = None if sample["p_lazy"] < 0.01 else sample["p_lazy"]

            if sample["make_reward_stochastic"]:
                sample["sub_optimal_distribution"] = (
                    "beta",
                    (
                        sample["reward_variance_multiplier"],
                        sample["reward_variance_multiplier"] * (1 / 0.2 - 1),
                    ),
                )
                sample["optimal_distribution"] = (
                    "beta",
                    (
                        sample["reward_variance_multiplier"],
                        sample["reward_variance_multiplier"] * (1 / 0.9 - 1),
                    ),
                )
                sample["other_distribution"] = (
                    "beta",
                    (
                        sample["reward_variance_multiplier"],
                        sample["reward_variance_multiplier"] * (10 / 0.2 - 1),
                    ),
                )
            else:
                sample["sub_optimal_distribution"] = (
                    "deterministic",
                    (round(5 / 1000, 3),),
                )
                sample["optimal_distribution"] = ("deterministic", (1.0,))
                sample["other_distribution"] = ("deterministic", (0.0,))

            samples.append(rounding_nested_structure(sample))
        return samples

    @staticmethod
    def get_node_class() -> Type[RiverSwimNode]:
        return RiverSwimNode

    def get_gin_parameters(self, index: int) -> str:
        prms = dict(
            size=self._size,
            make_reward_stochastic=self._make_reward_stochastic,
            reward_variance_multiplier=self._reward_variance_multiplier,
            optimal_distribution=(
                self._optimal_distribution.dist.name,
                self._optimal_distribution.args,
            ),
            other_distribution=(
                self._other_distribution.dist.name,
                self._other_distribution.args,
            ),
            sub_optimal_distribution=(
                self._sub_optimal_distribution.dist.name,
                self._sub_optimal_distribution.args,
            ),
        )

        if self._p_rand is not None:
            prms["p_rand"] = self._p_rand
        if self._p_lazy is not None:
            prms["p_lazy"] = self._p_lazy

        return RiverSwimMDP.produce_gin_file_from_mdp_parameters(
            prms, type(self).__name__, index
        )

    @property
    def n_actions(self) -> int:
        return len(RiverSwimAction)

    def __init__(
        self,
        seed: int,
        size: int,
        optimal_mean_reward: float = 0.9,
        sub_optimal_mean_reward: float = 0.2,
        sub_optimal_distribution: Union[Tuple, rv_continuous] = None,
        optimal_distribution: Union[Tuple, rv_continuous] = None,
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
            The length of the chain.
        optimal_mean_reward : float
            If the rewards are made stochastic, this parameter controls the mean reward for the highly rewarding states.
            By default, it is set to 0.9.
        sub_optimal_mean_reward : float
            If the rewards are made stochastic, this parameter controls the mean reward for the suboptimal states.
            By default, it is set to 0.2.
        sub_optimal_distribution : Union[Tuple, rv_continuous]
            The distribution of the suboptimal rewarding states. It can be either passed as a tuple containing Beta
            parameters or as a rv_continuous object.
        optimal_distribution : Union[Tuple, rv_continuous]
            The distribution of the highly rewarding state. It can be either passed as a tuple containing Beta parameters
            or as a rv_continuous object.
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
        self._optimal_mean_reward = optimal_mean_reward
        self._sub_optimal_mean_reward = sub_optimal_mean_reward
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
                if self.is_episodic():
                    sub_optimal_mean_reward /= self._size
                self._sub_optimal_distribution = beta(
                    reward_variance_multiplier,
                    reward_variance_multiplier * (1 / sub_optimal_mean_reward - 1),
                )
                self._optimal_distribution = beta(
                    reward_variance_multiplier,
                    reward_variance_multiplier * (1 / optimal_mean_reward - 1),
                )
                self._other_distribution = beta(
                    reward_variance_multiplier,
                    reward_variance_multiplier * (10 / sub_optimal_mean_reward - 1),
                )
            else:
                self._sub_optimal_distribution = deterministic(5 / 1000)
                self._optimal_distribution = deterministic(1.0)
                self._other_distribution = deterministic(0.0)

        super(RiverSwimMDP, self).__init__(
            seed=seed,
            reward_variance_multiplier=reward_variance_multiplier,
            make_reward_stochastic=make_reward_stochastic,
            **kwargs,
        )

    def _get_next_nodes_parameters(
        self, node: "NODE_TYPE", action: "ACTION_TYPE"
    ) -> Tuple[Tuple[dict, float], ...]:
        return (
            (
                dict(
                    X=min(node.X + 1, self._size - 1)
                    if action == RiverSwimAction.RIGHT
                    else max(node.X - 1, 0),
                ),
                1.0,
            ),
        )

    def _get_reward_distribution(
        self, node: "NODE_TYPE", action: "ACTION_TYPE", next_node: "NODE_TYPE"
    ) -> rv_continuous:
        return (
            self._optimal_distribution
            if node.X == self._size - 1 and action == RiverSwimAction.RIGHT
            else (
                self._sub_optimal_distribution
                if node.X == 0 and action == RiverSwimAction.LEFT
                else self._other_distribution
            )
        )

    def _get_starting_node_sampler(self) -> NextStateSampler:
        return NextStateSampler(next_nodes=self._possible_starting_nodes)

    def _check_parameters_in_input(self):
        super(RiverSwimMDP, self)._check_parameters_in_input()

        assert self._size > 1
        assert self._optimal_mean_reward - 0.1 > self._sub_optimal_mean_reward

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
        grid = np.zeros((1, self._size), dtype=str)
        grid[:, :] = " "
        grid[0, 0] = "S"
        grid[0, -1] = "G"
        grid[0, node.X] = "A"
        return grid

    @property
    def _possible_starting_nodes(self) -> List["NODE_TYPE"]:
        return [RiverSwimNode(0)]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            **super(RiverSwimMDP, self).parameters,
            **dict(
                size=self._size,
                optimal_mean_reward=self._optimal_mean_reward,
                sub_optimal_mean_reward=self._sub_optimal_mean_reward,
                optimal_distribution=self._optimal_distribution,
                sub_optimal_distribution=self._sub_optimal_distribution,
                other_distribution=self._other_distribution,
            ),
        }
