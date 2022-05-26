from abc import ABC
from dataclasses import dataclass
from enum import IntEnum

from colosseum.utils.random_vars import deterministic, get_dist

try:
    from functools import cached_property
except:
    from backports.cached_property import cached_property

from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map
from scipy.stats import beta, rv_continuous

from colosseum.mdps import MDP
from colosseum.mdps.base_mdp import NextStateSampler
from colosseum.utils.mdps import check_distributions


@dataclass(frozen=True)
class FrozenLakeNode:
    X: int
    Y: int

    def __str__(self):
        return f"X={self.X},Y={self.Y}"

    def __iter__(self):
        return iter((self.X, self.Y))


class FrozenLakeAction(IntEnum):
    """The action available in the FrozenLake MDP."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class FrozenLakeMDP(MDP, ABC):
    @staticmethod
    def testing_parameters() -> Dict[str, Tuple]:
        t_params = MDP.testing_parameters()
        t_params["size"] = (5, 8, 10)
        t_params["p_frozen"] = (0.7, 0.8)
        t_params["make_reward_stochastic"] = (True, False)
        t_params["is_slippery"] = (True, False)
        return t_params

    @staticmethod
    def get_node_class() -> Type[FrozenLakeNode]:
        return FrozenLakeNode

    def __init__(
        self,
        seed: int,
        size: int,
        p_frozen: float,
        lazy: float = None,
        randomize_actions: bool = True,
        make_reward_stochastic=False,
        suboptimal_return: float = 0.1,
        optimal_return: float = 1.0,
        is_slippery: bool = True,
        goal_r: Union[Tuple, rv_continuous] = None,
        default_r: Union[Tuple, rv_continuous] = None,
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
        p_frozen : float
            the probability of a tile to be frozen and not to be a hole.
        make_reward_stochastic : bool, optional
            checks whether the rewards are to be made stochastic. By default, it is set to False.
        suboptimal_return: float, optional
            if the rewards are made stochastic, this parameter controls the mean reward for suboptimal trajectories.
            By default, it is set to 0.1.
        optimal_return: float, optional
            if the rewards are made stochastic, this parameter controls the mean reward for the optimal trajectory.
            By default, it is set to 1.
        is_slippery : bool, optional
            checks whether the frozen tiles are slippery and thus can produce random unexpected movements. By deafult,
            it is set to True.
        goal_r: Union[Tuple, rv_continuous]
            The reward distribution for reaching the highly rewarding state. It can be either passed as a tuple containing
            Beta parameters or as a rv_continuous object.
        default_r: Union[Tuple, rv_continuous]
            The reward distribution for all the other states. It can be either passed as a tuple containing Beta
            parameters or as a rv_continuous object.
        """

        if type(goal_r) == tuple:
            goal_r = get_dist(goal_r[0], goal_r[1:])
        if type(default_r) == tuple:
            default_r = get_dist(default_r[0], default_r[1:])

        self.optimal_return = optimal_return
        self.suboptimal_return = suboptimal_return
        self.is_slippery = is_slippery
        self.p_frozen = p_frozen
        self.goal_r = goal_r
        self.default_r = default_r
        self.size = size
        self.make_reward_stochastic = make_reward_stochastic

        if (default_r, goal_r).count(None) == 0:
            self.default_r = default_r
            self.goal_r = goal_r
        else:
            if make_reward_stochastic:
                self.default_r = beta(1, size ** 2 / self.suboptimal_return - 1)
                self.goal_r = beta(size ** 2 / self.optimal_return - 1, 1)
            else:
                self.default_r = deterministic(0.0)
                self.goal_r = deterministic(1.0)

        super().__init__(
            seed=seed,
            randomize_actions=randomize_actions,
            lazy=lazy,
            size=size,
            p_frozen=p_frozen,
            goal_r=goal_r,
            default_r=default_r,
            is_slippery=is_slippery,
            **kwargs,
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            **super(FrozenLakeMDP, self).parameters,
            **dict(
                size=self.size,
                optimal_return=self.optimal_return,
                suboptimal_return=self.suboptimal_return,
                is_slippery=self.is_slippery,
                p_frozen=self.p_frozen,
                goal_r=self.goal_r,
                default_r=self.default_r,
            ),
        }

    @cached_property
    def possible_starting_nodes(self) -> List[FrozenLakeNode]:
        return [FrozenLakeNode(0, 0)]

    @cached_property
    def lake(self):
        np.random.seed(self._next_seed())
        return np.array(
            list(
                map(
                    lambda x: list(x),
                    generate_random_map(size=self.size, p=self.p_frozen),
                )
            )
        )

    def _next_postions(self, x, y, a):
        if self.lake[x, y] == "G":
            return dict(X=0, Y=0)

        if a == FrozenLakeAction.LEFT:
            next_x, next_y = x, min(y + 1, self.size - 1)
        if a == FrozenLakeAction.DOWN:
            next_x, next_y = min(x + 1, self.size - 1), y
        if a == FrozenLakeAction.RIGHT:
            next_x, next_y = x, max(y - 1, 0)
        if a == FrozenLakeAction.UP:
            next_x, next_y = max(x - 1, 0), y
        next_pos = self.lake[next_x, next_y]
        if next_pos == "H":
            return dict(X=0, Y=0)
        else:
            return dict(X=next_x, Y=next_y)

    @property
    def num_actions(self):
        return len(FrozenLakeAction)

    def _calculate_next_nodes_prms(
        self, node: FrozenLakeNode, action: int
    ) -> List[Tuple[dict, float]]:
        p = 0.5 if self.is_slippery else 1.0
        next_nodes_prms = []
        next_nodes_prms.append((self._next_postions(node.X, node.Y, action), p))
        if self.is_slippery:
            for a in [(action - 1) % 4, (action + 1) % 4]:
                next_nodes_prms.append((self._next_postions(node.X, node.Y, a), p / 2))
        return next_nodes_prms

    def _calculate_reward_distribution(
        self, node: FrozenLakeNode, action: IntEnum, next_node: FrozenLakeNode
    ) -> rv_continuous:
        if self.lake[next_node.X, next_node.Y] == "G":
            return self.goal_r
        return self.default_r

    def _check_input_parameters(self):
        super(FrozenLakeMDP, self)._check_input_parameters()

        assert self.p_frozen >= 0.5
        assert self.size > 2

        assert self.suboptimal_return + 0.2 < self.optimal_return

        dists = [
            self.goal_r,
            self.default_r,
        ]
        check_distributions(
            dists,
            self.make_reward_stochastic,
        )

    def _instantiate_starting_node_sampler(self) -> NextStateSampler:
        return NextStateSampler(next_states=self.possible_starting_nodes)

    def calc_grid_repr(self, node: FrozenLakeNode) -> np.array:
        grid = self.lake.copy()
        grid[node.X, node.Y] = "A"
        return grid.T[::-1, :]
