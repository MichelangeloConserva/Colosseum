from abc import ABC
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
from scipy.stats import beta, rv_continuous

from colosseum.mdps.base_mdp import MDP, NextStateSampler
from colosseum.utils.mdps import check_distributions
from colosseum.utils.random_vars import deterministic, get_dist


@dataclass(frozen=True)
class RiverSwimNode:
    X: int

    def __str__(self):
        return f"X={self.X}"

    def __iter__(self):
        return iter((self.X, self.X))


class RiverSwimAction(IntEnum):
    """The action available in the Chain MDP."""

    LEFT = 0
    RIGHT = 1


class RiverSwimMDP(MDP, ABC):
    @staticmethod
    def testing_parameters() -> Dict[str, Tuple]:
        t_params = MDP.testing_parameters()
        t_params["size"] = (5, 8, 30)
        t_params["make_reward_stochastic"] = (True, False)
        return t_params

    @staticmethod
    def get_node_class() -> Type[RiverSwimNode]:
        return RiverSwimNode

    def __init__(
        self,
        seed: int,
        size: int,
        lazy: Union[None, float] = None,
        randomize_actions: bool = True,
        make_reward_stochastic=False,
        optimal_mean_reward: float = 0.9,
        sub_optimal_mean_reward: float = 0.2,
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
        lazy : float
            the probability of an action not producing any effect on the MDP.
        size : int
            the size of the chain.
        make_reward_stochastic : bool, optional
            checks whether the rewards are to be made stochastic. By default, it is set to False.
        optimal_mean_reward : float, optional
            if the rewards are made stochastic, this parameter controls the mean reward for the highly rewarding states.
            By default, it is set to 0.9.
        sub_optimal_mean_reward: float, optional
            if the rewards are made stochastic, this parameter controls the mean reward for the suboptimal states.
            By default, it is set to 0.2.
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

        self.optimal_mean_reward = optimal_mean_reward
        self.sub_optimal_mean_reward = sub_optimal_mean_reward
        self.other_distribution = other_distribution
        self.optimal_distribution = optimal_distribution
        self.sub_optimal_distribution = sub_optimal_distribution
        self.size = size
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
                if self.is_episodic():
                    sub_optimal_mean_reward /= self.size
                self.sub_optimal_distribution = beta(1, 1 / sub_optimal_mean_reward - 1)
                self.optimal_distribution = beta(1, 1 / optimal_mean_reward - 1)
                self.other_distribution = beta(1, 10 / sub_optimal_mean_reward - 1)
            else:
                self.sub_optimal_distribution = deterministic(5 / 1000)
                self.optimal_distribution = deterministic(1.0)
                self.other_distribution = deterministic(0.0)

        super(RiverSwimMDP, self).__init__(
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
            **super(RiverSwimMDP, self).parameters,
            **dict(
                size=self.size,
                optimal_mean_reward=self.optimal_mean_reward,
                sub_optimal_mean_reward=self.sub_optimal_mean_reward,
                sub_optimal_distribution=self.sub_optimal_distribution,
                optimal_distribution=self.optimal_distribution,
                other_distribution=self.other_distribution,
            ),
        }

    @property
    def possible_starting_nodes(self) -> List[RiverSwimNode]:
        return [RiverSwimNode(0)]

    @property
    def num_actions(self):
        return len(RiverSwimAction)

    def _calculate_next_nodes_prms(
        self, node, action
    ) -> Tuple[Tuple[dict, float], ...]:
        return (
            (
                dict(
                    X=min(node.X + 1, self.size - 1)
                    if action == RiverSwimAction.RIGHT
                    else max(node.X - 1, 0),
                ),
                1.0,
            ),
        )

    def _calculate_reward_distribution(
        self, node: RiverSwimNode, action: IntEnum, next_node: RiverSwimNode
    ) -> rv_continuous:
        return (
            self.optimal_distribution
            if node.X == self.size - 1 and action == RiverSwimAction.RIGHT
            else (
                self.sub_optimal_distribution
                if node.X == 0 and action == RiverSwimAction.LEFT
                else self.other_distribution
            )
        )

    def _check_input_parameters(self):
        super(RiverSwimMDP, self)._check_input_parameters()

        assert self.size > 1
        assert self.optimal_mean_reward - 0.1 > self.sub_optimal_mean_reward

        # Don't be too lazy
        if self.lazy:
            assert self.lazy <= 0.9
        dists = [
            self.sub_optimal_distribution,
            self.optimal_distribution,
            self.other_distribution,
        ]
        check_distributions(
            dists,
            self.make_reward_stochastic,
        )

    def _instantiate_starting_node_sampler(self) -> NextStateSampler:
        return NextStateSampler(next_states=self.possible_starting_nodes)

    def calc_grid_repr(self, node) -> np.array:
        grid = np.zeros((1, self.size), dtype=str)
        grid[:, :] = " "
        grid[0, 0] = "S"
        grid[0, -1] = "G"
        grid[0, node.X] = "A"
        return grid
