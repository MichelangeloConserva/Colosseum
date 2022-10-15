import abc
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, Union

import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map
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
class FrozenLakeNode:
    """
    The node for the FrozenLake MDP.
    """

    X: int
    """x coordinate."""
    Y: int
    """y coordinate."""

    def __str__(self):
        return f"X={self.X},Y={self.Y}"

    def __iter__(self):
        return iter((self.X, self.Y))


class FrozenLakeAction(IntEnum):
    """The action available in the FrozenLake MDP."""

    UP = 0
    """Move up."""
    RIGHT = 1
    """Move towards the right."""
    DOWN = 2
    """Move down."""
    LEFT = 3
    """Move towards the left."""


class FrozenLakeMDP(BaseMDP, abc.ABC):
    """
    The base class for the FrozenLake family.
    """

    @staticmethod
    def get_unique_symbols() -> List[str]:
        return ["A", "F", "H", "G"]

    @staticmethod
    def does_seed_change_MDP_structure() -> bool:
        return True

    @staticmethod
    def sample_mdp_parameters(
        n: int, is_episodic: bool, seed: int = None
    ) -> List[Dict[str, Any]]:
        rng = np.random.RandomState(np.random.randint(10_000) if seed is None else seed)
        samples = []
        for _ in range(n):
            p_rand, p_lazy, _ = 0.9 * rng.dirichlet([0.2, 0.2, 5])
            sample = dict(
                size=rng.choice(range(5, 7), None, True, [0.665, 0.335])
                if is_episodic
                else int((2.5 + np.minimum((400 / (150 * rng.random() + 35)), 15))),
                p_frozen=min((0.55 * rng.random() + 0.45) ** 0.3, 0.95),
                p_rand=p_rand,
                p_lazy=p_lazy,
                make_reward_stochastic=rng.choice([True, False]),
                reward_variance_multiplier=2 * rng.random() + 0.005,
            )
            sample["p_rand"] = None if sample["p_rand"] < 0.01 else sample["p_rand"]
            sample["p_lazy"] = None if sample["p_lazy"] < 0.01 else sample["p_lazy"]

            if sample["make_reward_stochastic"]:
                sample["default_r"] = (
                    "beta",
                    (
                        sample["reward_variance_multiplier"],
                        sample["reward_variance_multiplier"]
                        * (sample["size"] ** 2 / 0.1 - 1),
                    ),
                )
                sample["goal_r"] = (
                    "beta",
                    (
                        sample["reward_variance_multiplier"]
                        * (sample["size"] ** 2 - 1),
                        sample["reward_variance_multiplier"],
                    ),
                )
            else:
                sample["default_r"] = ("deterministic", (0.0,))
                sample["goal_r"] = ("deterministic", (1.0,))

            samples.append(rounding_nested_structure(sample))
        return samples

    @staticmethod
    def get_node_class() -> Type["NODE_TYPE"]:
        return FrozenLakeNode

    def get_gin_parameters(self, index: int) -> str:
        prms = dict(
            size=self._size,
            p_frozen=self._p_frozen,
            make_reward_stochastic=self._make_reward_stochastic,
            reward_variance_multiplier=self._reward_variance_multiplier,
            default_r=(
                self._default_r.dist.name,
                self._default_r.args,
            ),
            goal_r=(
                self._goal_r.dist.name,
                self._goal_r.args,
            ),
        )

        if self._p_rand is not None:
            prms["p_rand"] = self._p_rand
        if self._p_lazy is not None:
            prms["p_lazy"] = self._p_lazy

        return FrozenLakeMDP.produce_gin_file_from_mdp_parameters(
            prms, type(self).__name__, index
        )

    @property
    def n_actions(self) -> int:
        return len(FrozenLakeAction)

    def _next_positions(self, x, y, a):
        if self.lake[x, y] == "G":
            return dict(X=0, Y=0)

        if a == FrozenLakeAction.LEFT:
            next_x, next_y = x, min(y + 1, self._size - 1)
        if a == FrozenLakeAction.DOWN:
            next_x, next_y = min(x + 1, self._size - 1), y
        if a == FrozenLakeAction.RIGHT:
            next_x, next_y = x, max(y - 1, 0)
        if a == FrozenLakeAction.UP:
            next_x, next_y = max(x - 1, 0), y
        next_pos = self.lake[next_x, next_y]
        if next_pos == "H":
            return dict(X=0, Y=0)
        else:
            return dict(X=next_x, Y=next_y)

    def _get_next_nodes_parameters(
        self, node: "NODE_TYPE", action: "ACTION_TYPE"
    ) -> Tuple[Tuple[dict, float], ...]:
        p = 0.5 if self._is_slippery else 1.0
        next_nodes_prms = []
        next_nodes_prms.append((self._next_positions(node.X, node.Y, action), p))
        if self._is_slippery:
            for a in [(action - 1) % 4, (action + 1) % 4]:
                next_nodes_prms.append((self._next_positions(node.X, node.Y, a), p / 2))
        return next_nodes_prms

    def _get_reward_distribution(
        self, node: "NODE_TYPE", action: "ACTION_TYPE", next_node: "NODE_TYPE"
    ) -> rv_continuous:
        if self.lake[next_node.X, next_node.Y] == "G":
            return self._goal_r
        return self._default_r

    def _get_starting_node_sampler(self) -> NextStateSampler:
        return NextStateSampler(next_nodes=self._possible_starting_nodes)

    def _check_parameters_in_input(self):
        super(FrozenLakeMDP, self)._check_parameters_in_input()

        assert self._p_frozen >= 0.1
        assert self._size > 2

        assert self._suboptimal_return + 0.2 < self._optimal_return

        dists = [
            self._goal_r,
            self._default_r,
        ]
        check_distributions(
            dists,
            self._make_reward_stochastic,
        )

    def _get_grid_representation(self, node: "NODE_TYPE") -> np.ndarray:
        grid = self.lake.copy()
        grid[0, 0] = "F"
        grid[node.X, node.Y] = "A"
        return grid.T[::-1, :]

    @property
    def _possible_starting_nodes(self) -> List["NODE_TYPE"]:
        return [FrozenLakeNode(0, 0)]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            **super(FrozenLakeMDP, self).parameters,
            **dict(
                size=self._size,
                p_frozen=self._p_frozen,
                optimal_return=self._optimal_return,
                suboptimal_return=self._suboptimal_return,
                is_slippery=self._is_slippery,
                goal_r=self._goal_r,
                default_r=self._default_r,
            ),
        }

    def __init__(
        self,
        seed: int,
        size: int,
        p_frozen: float,
        optimal_return: float = 1.0,
        suboptimal_return: float = 0.1,
        is_slippery: bool = True,
        goal_r: Union[Tuple, rv_continuous] = None,
        default_r: Union[Tuple, rv_continuous] = None,
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
        p_frozen : float
            The probability that a tile of the lake is frozen and does not contain a hole.
        optimal_return: float
            If the rewards are made stochastic, this parameter controls the mean reward for the optimal trajectory.
            By default, it is set to 1.
        suboptimal_return: float
            If the rewards are made stochastic, this parameter controls the mean reward for suboptimal trajectories.
            By default, it is set to 0.1.
        is_slippery : bool
            If True, the outcome of the action is stochastic due to the frozen tiles being slippery. By default, it is
            set to True.
        goal_r : Union[Tuple, rv_continuous]
            The distribution of the highly rewarding state. It can be either passed as a tuple containing Beta parameters
            or as a rv_continuous object.
        default_r : Union[Tuple, rv_continuous]
            The distribution of the other states. It can be either passed as a tuple containing Beta parameters or as a
            rv_continuous object.
        make_reward_stochastic : bool
            If True, the rewards of the MDP will be stochastic. By default, it is set to False.
        reward_variance_multiplier : float
            A constant that can be used to increase the variance of the reward distributions without changing their means.
            The lower the value, the higher the variance. By default, it is set to 1.
        """

        if type(goal_r) == tuple:
            goal_r = get_dist(goal_r[0], goal_r[1])
        if type(default_r) == tuple:
            default_r = get_dist(default_r[0], default_r[1])

        self._size = size
        self._p_frozen = p_frozen
        self._optimal_return = optimal_return
        self._suboptimal_return = suboptimal_return
        self._is_slippery = is_slippery
        self._goal_r = goal_r
        self._default_r = default_r

        np.random.seed(seed)
        self.lake = np.array(
            list(
                map(
                    lambda x: list(x),
                    generate_random_map(size=self._size, p=self._p_frozen),
                )
            )
        )

        if (default_r, goal_r).count(None) == 0:
            self._default_r = default_r
            self._goal_r = goal_r
        else:
            if make_reward_stochastic:
                self._default_r = beta(
                    reward_variance_multiplier,
                    reward_variance_multiplier
                    * (size ** 2 / self._suboptimal_return - 1),
                )
                self._goal_r = beta(
                    reward_variance_multiplier * (size ** 2 / self._optimal_return - 1),
                    reward_variance_multiplier,
                )
            else:
                self._default_r = deterministic(0.0)
                self._goal_r = deterministic(1.0)

        super(FrozenLakeMDP, self).__init__(
            seed=seed,
            reward_variance_multiplier=reward_variance_multiplier,
            make_reward_stochastic=make_reward_stochastic,
            **kwargs,
        )
