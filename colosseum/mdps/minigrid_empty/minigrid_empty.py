from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from itertools import product
from typing import Any, Dict, List, Tuple, Type, Union

try:
    from functools import cached_property
except:
    from backports.cached_property import cached_property

import numpy as np
from scipy.stats import beta, rv_continuous

from colosseum.mdps.base_mdp import MDP, NextStateSampler
from colosseum.utils.mdps import check_distributions
from colosseum.utils.random_vars import deterministic, get_dist


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


class MiniGridEmptyMDP(MDP):
    @staticmethod
    def testing_parameters() -> Dict[str, Tuple]:
        t_params = MDP.testing_parameters()
        t_params["size"] = (5, 8, 10)
        t_params["make_reward_stochastic"] = (True, False)
        t_params["n_starting_states"] = (1, 4)
        return t_params

    @staticmethod
    def get_node_class() -> Type[MiniGridEmptyNode]:
        return MiniGridEmptyNode

    def __init__(
        self,
        seed: int,
        size: int,
        lazy: Union[None, float] = None,
        randomize_actions: bool = True,
        make_reward_stochastic=False,
        n_starting_states: int = 2,
        optimal_distribution: Union[Tuple, rv_continuous] = None,
        other_distribution: Union[Tuple, rv_continuous] = None,
        **kwargs,
    ):

        """

        Parameters
        ----------
        seed : int
            the seed used for sampling rewards and next states.
        randomize_actions : bool, optional
            whether the effect of the actions changes for every node. It is particularly important to set this value to
             true when doing experiments to avoid immediately reaching highly rewarding states in some MDPs by just
             selecting the same action repeatedly. By default, it is set to true.
        lazy : float
            the probability of an action not producing any effect on the MDP.
        size : int
            the size of the grid.
        make_reward_stochastic : bool, optional
            checks whether the rewards are to be made stochastic. By default, it is set to False.
        n_starting_states : int, optional
            the number of states in the starting distribution. By default, it is set to two.
        optimal_distribution : Union[Tuple, rv_continuous], optional
            The distribution of the highly rewarding state. It can be either passed as a tuple containing Beta parameters
            or as a rv_continuous object.
        other_distribution : Union[Tuple, rv_continuous]
            The distribution of the non highly rewarding states. It can be either passed as a tuple containing Beta parameters
            or as a rv_continuous object.
        """

        if type(optimal_distribution) == tuple:
            optimal_distribution = get_dist(
                optimal_distribution[0], optimal_distribution[1:]
            )
        if type(other_distribution) == tuple:
            other_distribution = get_dist(other_distribution[0], other_distribution[1:])

        self.n_starting_states = n_starting_states
        self.size = size
        self.make_reward_stochastic = make_reward_stochastic

        dists = [
            optimal_distribution,
            other_distribution,
        ]
        if dists.count(None) == 0:
            self.optimal_distribution = optimal_distribution
            self.other_distribution = other_distribution
        else:
            if make_reward_stochastic:
                self.other_distribution = beta(1, size ** 2 - 1)
                self.optimal_distribution = beta(size ** 2 - 1, 1)
            else:
                self.optimal_distribution = deterministic(1.0)
                self.other_distribution = deterministic(0.0)

        super().__init__(
            seed=seed,
            lazy=lazy,
            randomize_actions=randomize_actions,
            **kwargs,
        )

    @property
    def num_actions(self):
        return len(MiniGridEmptyAction)

    def _calculate_next_nodes_prms(
        self, node, action
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
                return ((dict(X=node.X, Y=min(node.Y + 1, self.size - 1), Dir=d), 1.0),)
            if d == MiniGridEmptyDirection.RIGHT:
                return ((dict(X=min(self.size - 1, node.X + 1), Y=node.Y, Dir=d), 1.0),)
            if d == MiniGridEmptyDirection.DOWN:
                return ((dict(X=node.X, Y=max(node.Y - 1, 0), Dir=d), 1.0),)
            if d == MiniGridEmptyDirection.LEFT:
                return ((dict(X=max(0, node.X - 1), Y=node.Y, Dir=d), 1.0),)

    def _calculate_reward_distribution(
        self,
        node: MiniGridEmptyNode,
        action: IntEnum,
        next_node: MiniGridEmptyNode,
    ) -> rv_continuous:
        return (
            self.optimal_distribution
            if next_node.X == self.goal_position[0]
            and next_node.Y == self.goal_position[1]
            else self.other_distribution
        )

    def _check_input_parameters(self):
        super(MiniGridEmptyMDP, self)._check_input_parameters()

        assert (
            self.size > 2
        ), f"the size should be greater than 2, you selected:{self.size}"

        assert self.n_starting_states > 0

        # Don't be too lazy
        if self.lazy:
            assert self.lazy <= 0.9
        dists = [
            self.optimal_distribution,
            self.other_distribution,
        ]
        check_distributions(
            dists,
            self.make_reward_stochastic,
        )

    @cached_property
    def possible_starting_nodes(self) -> List[MiniGridEmptyNode]:
        return [
            MiniGridEmptyNode(x, y, MiniGridEmptyDirection(d))
            for (x, y), d in product(self._possible_starting_nodes, range(4))
        ]

    def _instantiate_starting_node_sampler(self) -> NextStateSampler:
        self.side_start = self._rng.randint(4)
        self.goal_position = self.get_positions_on_side((self.side_start + 2) % 4)[
            : self.size
        ][self._rng.randint(self.size)]
        self._possible_starting_nodes = self.get_positions_on_side(self.side_start)[
            : self.size
        ]
        self._rng.shuffle(self._possible_starting_nodes)
        starting_nodes = self._possible_starting_nodes[: self.n_starting_states]
        return NextStateSampler(
            next_states=[
                MiniGridEmptyNode(x, y, MiniGridEmptyDirection(self._rng.randint(4)))
                for x, y in starting_nodes
            ],
            probs=[1 / len(starting_nodes) for _ in range(len(starting_nodes))],
            seed=self._next_seed(),
        )

    def calc_grid_repr(self, node) -> np.array:
        grid = np.zeros((self.size, self.size), dtype=str)
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
                size=self.size,
                n_starting_states=self.n_starting_states,
                optimal_distribution=self.optimal_distribution,
                other_distribution=self.other_distribution,
            ),
        }

    def get_positions_on_side(self, side: int) -> List[Tuple[int, int]]:
        nodes = []
        for i in range(self.size):
            for j in range(self.size):
                if side == 0:  # Starting from the left
                    nodes.append((i, j))
                elif side == 1:  # Starting from the south
                    nodes.append((j, i))
                elif side == 2:  # Starting from the right
                    nodes.append((self.size - 1 - i, self.size - 1 - j))
                else:  # Starting from the north
                    nodes.append((self.size - 1 - j, self.size - 1 - i))
                # if len(nodes) == N:
                #     return nodes
        return nodes
