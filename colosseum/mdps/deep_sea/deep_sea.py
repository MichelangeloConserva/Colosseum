from abc import ABC
from dataclasses import dataclass
from enum import IntEnum
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


@dataclass(frozen=True)
class DeepSeaNode:
    X: int
    Y: int

    def __str__(self):
        return f"X={self.X},Y={self.Y}"

    def __iter__(self):
        return iter((self.X, self.Y))


class DeepSeaAction(IntEnum):
    """The action available in the Chain MDP."""

    LEFT = 0
    RIGHT = 1


class DeepSeaMDP(MDP, ABC):
    @staticmethod
    def testing_parameters() -> Dict[str, Tuple]:
        t_params = MDP.testing_parameters()
        t_params["size"] = (5, 8, 10)
        t_params["lazy"] = (None,)
        t_params["make_reward_stochastic"] = (True, False)
        return t_params

    @staticmethod
    def get_node_class() -> Type[DeepSeaNode]:
        return DeepSeaNode

    def __init__(
        self,
        seed: int,
        size: int,
        randomize_actions: bool = True,
        make_reward_stochastic=False,
        lazy: Union[None, float] = None,
        suboptimal_return: float = 0.5,
        optimal_return: float = 1.0,
        sub_optimal_distribution: Union[Tuple, rv_continuous] = None,
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
        size : int
            the size of the grid.
        make_reward_stochastic : bool, optional
            checks whether the rewards are to be made stochastic. By default, it is set to False.
        lazy: float, optional
            the probability of an action not producing any effect on the MDP. By default, it is set to zero.
        suboptimal_return: float, optional
            if the rewards are made stochastic, this parameter controls the mean reward for suboptimal trajectories.
            By default, it is set to 0.5.
        optimal_return: float, optional
            if the rewards are made stochastic, this parameter controls the mean reward for the optimal trajectory.
            By default, it is set to 1.
        sub_optimal_distribution : Union[Tuple, rv_continuous], optional
            The distribution of the suboptimal rewarding states. It can be either passed as a tuple containing Beta
            parameters or as a rv_continuous object.
        optimal_distribution : Union[Tuple, rv_continuous], optional
            The distribution of the highly rewarding state. It can be either passed as a tuple containing Beta parameters
            or as a rv_continuous object.
        other_distribution : Union[Tuple, rv_continuous], optional
            The distribution of the other states. It can be either passed as a tuple containing Beta parameters or as a
            rv_continuous object.
        """

        if type(sub_optimal_distribution) == tuple:
            sub_optimal_distribution = get_dist(
                sub_optimal_distribution[0], sub_optimal_distribution[1:]
            )
        if type(optimal_distribution) == tuple:
            optimal_distribution = get_dist(
                optimal_distribution[0], optimal_distribution[1:]
            )
        if type(other_distribution) == tuple:
            other_distribution = get_dist(other_distribution[0], other_distribution[1:])

        self.other_distribution = other_distribution
        self.optimal_distribution = optimal_distribution
        self.sub_optimal_distribution = sub_optimal_distribution
        self.size = size
        self.suboptimal_return = suboptimal_return
        self.optimal_return = optimal_return
        self.make_reward_stochastic = make_reward_stochastic

        dists = [
            sub_optimal_distribution,
            optimal_distribution,
            other_distribution,
        ]
        if dists.count(None) == 0:
            self.sub_optimal_distribution = sub_optimal_distribution
            self.optimal_distribution = optimal_distribution
            self.other_distribution = other_distribution
        else:
            if make_reward_stochastic:
                self.sub_optimal_distribution = beta(
                    1, size / self.suboptimal_return - 1
                )
                self.optimal_distribution = beta(size / self.optimal_return - 1, 1)
                self.other_distribution = beta(
                    1, 10 * (size / self.suboptimal_return - 1)
                )
            else:
                self.sub_optimal_distribution = deterministic(1.0 / (size ** 2))
                self.optimal_distribution = deterministic(1.0)
                self.other_distribution = deterministic(0.0)

        super(DeepSeaMDP, self).__init__(
            seed=seed,
            randomize_actions=randomize_actions,
            lazy=lazy,
            size=size,
            sub_optimal_distribution=sub_optimal_distribution,
            optimal_distribution=optimal_distribution,
            other_distribution=other_distribution,
            **kwargs,
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            **super(DeepSeaMDP, self).parameters,
            **dict(
                size=self.size,
                suboptimal_return=self.suboptimal_return,
                optimal_return=self.optimal_return,
                sub_optimal_distribution=self.sub_optimal_distribution,
                optimal_distribution=self.optimal_distribution,
                other_distribution=self.other_distribution,
            ),
        }

    @cached_property
    def possible_starting_nodes(self) -> List[DeepSeaNode]:
        return [DeepSeaNode(0, self.size - 1)]

    def _check_input_parameters(self):
        super(DeepSeaMDP, self)._check_input_parameters()

        assert self.size > 1

        # Don't be lazy
        assert self.lazy is None

        assert self.suboptimal_return < self.optimal_return - 0.1

        dists = [
            self.sub_optimal_distribution,
            self.optimal_distribution,
            self.other_distribution,
        ]
        check_distributions(
            dists,
            self.make_reward_stochastic,
        )

    @property
    def num_actions(self):
        return len(DeepSeaAction)

    def _calculate_next_nodes_prms(
        self, node, action
    ) -> Tuple[Tuple[dict, float], ...]:
        if node.Y == 0:
            return ((dict(X=0, Y=self.size - 1), 1.0),)

        return (
            (
                dict(
                    X=min(node.X + 1, self.size - 1)
                    if action == DeepSeaAction.RIGHT
                    else max(node.X - 1, 0),
                    Y=max(0, node.Y - 1),
                ),
                1.0,
            ),
        )

    def _instantiate_starting_node_sampler(self) -> NextStateSampler:
        return NextStateSampler(next_states=self.possible_starting_nodes)

    def calc_grid_repr(self, node) -> np.array:
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[:, :] = " "
        grid[node.Y, node.X] = "A"
        return grid[::-1, :]

    def _calculate_reward_distribution(
        self,
        node: DeepSeaNode,
        action: IntEnum,
        next_node: DeepSeaNode,
    ) -> rv_continuous:
        return (
            self.optimal_distribution
            if node.X == self.size - 1 and node.Y == 0 and action == DeepSeaAction.RIGHT
            else (
                self.sub_optimal_distribution
                if action == DeepSeaAction.LEFT
                else self.other_distribution
            )
        )
