import abc
from dataclasses import dataclass
from enum import IntEnum
from itertools import product
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, Union

import numpy as np
from scipy.stats import beta, rv_continuous

from colosseum.mdp import BaseMDP
from colosseum.mdp.utils.custom_samplers import NextStateSampler
from colosseum.utils.miscellanea import check_distributions, deterministic, get_dist

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE, NODE_TYPE


class MiniGridEmptyAction(IntEnum):
    """The action available in the MiniGridEmpty MDP."""

    MoveForward = 0
    TurnRight = 1
    TurnLeft = 2


class MiniGridEmptyDirection(IntEnum):
    """The possible agent directions in the MiniGridEmpty MDP."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


@dataclass(frozen=True)
class MiniGridEmptyNode:
    X: int
    Y: int
    Dir: MiniGridEmptyDirection

    def __str__(self):
        return f"X={self.X},Y={self.Y},Dir={self.Dir.name}"


class MiniGridEmptyMDP(BaseMDP, abc.ABC):
    @staticmethod
    def does_seed_change_MDP_structure() -> bool:
        return True

    @staticmethod
    def _sample_parameters(
        n: int, is_episodic: bool, seed: int = None
    ) -> List[Dict[str, Any]]:
        rng = np.random.RandomState(np.random.randint(10_000) if seed is None else seed)
        samples = []
        for _ in range(n):
            p_rand, p_lazy, _ = 0.9 * np.random.dirichlet([0.2, 0.2, 5])
            sample = dict(
                size=int(np.minimum(5 + (14 / (8 * np.random.random() + 1.0)), 20))
                if is_episodic
                else int(
                    1.5 * np.minimum(5 + (14 / (8 * np.random.random() + 1.0)), 20)
                ),
                n_starting_states=rng.randint(1, 5),
                p_rand=p_rand,
                p_lazy=p_lazy,
                make_reward_stochastic=rng.choice([True, False]),
                variance_multipliers=2 * rng.random() + 0.005,
            )
            sample["p_rand"] = None if sample["p_rand"] < 0.01 else sample["p_rand"]
            sample["p_lazy"] = None if sample["p_lazy"] < 0.01 else sample["p_lazy"]

            if sample["make_reward_stochastic"]:
                sample["optimal_distribution"] = beta(
                    sample["variance_multipliers"],
                    sample["variance_multipliers"] * (sample["size"] ** 2 - 1),
                )
                sample["other_distribution"] = beta(
                    sample["variance_multipliers"] * (sample["size"] ** 2 - 1),
                    sample["variance_multipliers"],
                )
            else:
                sample["optimal_distribution"] = deterministic(1.0)
                sample["other_distribution"] = deterministic(0.0)
            samples.append(sample)
        return samples

    @staticmethod
    def get_node_class() -> Type["NODE_TYPE"]:
        return MiniGridEmptyNode

    @property
    def n_actions(self) -> int:
        return len(MiniGridEmptyAction)

    def _get_next_nodes_parameters(
        self, node: "NODE_TYPE", action: "ACTION_TYPE"
    ) -> Tuple[Tuple[dict, float], ...]:
        d = node.Dir
        if action == MiniGridEmptyAction.TurnRight:
            return (
                (
                    dict(X=node.X, Y=node.Y, Dir=MiniGridEmptyDirection((d + 1) % 4)),
                    1.0,
                ),
            )
        if action == MiniGridEmptyAction.TurnLeft:
            return (
                (
                    dict(X=node.X, Y=node.Y, Dir=MiniGridEmptyDirection((d - 1) % 4)),
                    1.0,
                ),
            )
        if action == MiniGridEmptyAction.MoveForward:
            if d == MiniGridEmptyDirection.UP:
                return (
                    (dict(X=node.X, Y=min(node.Y + 1, self._size - 1), Dir=d), 1.0),
                )
            if d == MiniGridEmptyDirection.RIGHT:
                return (
                    (dict(X=min(self._size - 1, node.X + 1), Y=node.Y, Dir=d), 1.0),
                )
            if d == MiniGridEmptyDirection.DOWN:
                return ((dict(X=node.X, Y=max(node.Y - 1, 0), Dir=d), 1.0),)
            if d == MiniGridEmptyDirection.LEFT:
                return ((dict(X=max(0, node.X - 1), Y=node.Y, Dir=d), 1.0),)

    def _get_reward_distribution(
        self, node: "NODE_TYPE", action: "ACTION_TYPE", next_node: "NODE_TYPE"
    ) -> rv_continuous:
        return (
            self._optimal_distribution
            if next_node.X == self.goal_position[0]
            and next_node.Y == self.goal_position[1]
            else self._other_distribution
        )

    def get_positions_on_side(self, side: int) -> List[Tuple[int, int]]:
        nodes = []
        for i in range(self._size):
            for j in range(self._size):
                if side == 0:  # Starting from the left
                    nodes.append((i, j))
                elif side == 1:  # Starting from the south
                    nodes.append((j, i))
                elif side == 2:  # Starting from the right
                    nodes.append((self._size - 1 - i, self._size - 1 - j))
                else:  # Starting from the north
                    nodes.append((self._size - 1 - j, self._size - 1 - i))
                # if len(nodes) == N:
                #     return nodes
        return nodes

    @property
    def _possible_starting_nodes(self) -> List["NODE_TYPE"]:
        return [
            MiniGridEmptyNode(x, y, MiniGridEmptyDirection(d))
            for (x, y), d in product(self.__possible_starting_nodes, range(4))
        ]

    def _get_starting_node_sampler(self) -> NextStateSampler:
        self.side_start = self._rng.randint(4)
        self.goal_position = self.get_positions_on_side((self.side_start + 2) % 4)[
            : self._size
        ][self._rng.randint(self._size)]
        self.__possible_starting_nodes = self.get_positions_on_side(self.side_start)[
            : self._size
        ]
        self._rng.shuffle(self.__possible_starting_nodes)
        starting_nodes = self.__possible_starting_nodes[: self._n_starting_states]
        return NextStateSampler(
            next_nodes=[
                MiniGridEmptyNode(x, y, MiniGridEmptyDirection(self._rng.randint(4)))
                for x, y in starting_nodes
            ],
            probs=[1 / len(starting_nodes) for _ in range(len(starting_nodes))],
            seed=self._produce_random_seed(),
        )

    def _check_parameters_in_input(self):
        super(MiniGridEmptyMDP, self)._check_parameters_in_input()

        assert self._size > 2, f"the size should be greater than 2"
        assert self._n_starting_states > 0

        dists = [
            self._optimal_distribution,
            self._other_distribution,
        ]
        check_distributions(
            dists,
            self._make_reward_stochastic,
        )

    def _get_grid_representation(self, node: "NODE_TYPE"):
        grid = np.zeros((self._size, self._size), dtype=str)
        grid[:, :] = " "
        grid[self.goal_position[1], self.goal_position[0]] = "G"
        if self.cur_node.Dir == MiniGridEmptyDirection.UP:
            grid[self.cur_node.Y, self.cur_node.X] = "^"
        elif self.cur_node.Dir == MiniGridEmptyDirection.RIGHT:
            grid[self.cur_node.Y, self.cur_node.X] = ">"
        elif self.cur_node.Dir == MiniGridEmptyDirection.DOWN:
            grid[self.cur_node.Y, self.cur_node.X] = "v"
        elif self.cur_node.Dir == MiniGridEmptyDirection.LEFT:
            grid[self.cur_node.Y, self.cur_node.X] = "<"
        return grid[::-1, :]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            **super(MiniGridEmptyMDP, self).parameters,
            **dict(
                size=self._size,
                n_starting_states=self._n_starting_states,
                optimal_distribution=self._optimal_distribution,
                other_distribution=self._other_distribution,
            ),
        }

    def __init__(
        self,
        seed: int,
        size: int,
        n_starting_states: int = 1,
        optimal_distribution: Union[Tuple, rv_continuous] = None,
        other_distribution: Union[Tuple, rv_continuous] = None,
        make_reward_stochastic=False,
        variance_multipliers: float = 1.0,
        **kwargs,
    ):

        if type(optimal_distribution) == tuple:
            optimal_distribution = get_dist(
                optimal_distribution[0], optimal_distribution[1:]
            )
        if type(other_distribution) == tuple:
            other_distribution = get_dist(other_distribution[0], other_distribution[1:])

        self._n_starting_states = n_starting_states
        self._size = size

        dists = [
            optimal_distribution,
            other_distribution,
        ]
        if dists.count(None) == 0:
            self._optimal_distribution = optimal_distribution
            self._other_distribution = other_distribution
        else:
            if make_reward_stochastic:
                self._other_distribution = beta(
                    variance_multipliers, variance_multipliers * (size ** 2 - 1)
                )
                self._optimal_distribution = beta(
                    variance_multipliers * (size ** 2 - 1), variance_multipliers
                )
            else:
                self._optimal_distribution = deterministic(1.0)
                self._other_distribution = deterministic(0.0)

        super(MiniGridEmptyMDP, self).__init__(
            seed=seed,
            variance_multipliers=variance_multipliers,
            make_reward_stochastic=make_reward_stochastic,
            **kwargs,
        )
