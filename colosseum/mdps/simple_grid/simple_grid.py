from abc import ABC
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Tuple, Type, Union

import gin
import numpy as np
from scipy.stats import beta, rv_continuous

from colosseum.mdps.base_mdp import MDP, NextStateSampler
from colosseum.utils.mdps import check_distributions
from colosseum.utils.random_vars import deterministic, get_dist


@dataclass(frozen=True)
class SimpleGridNode:
    X: int
    Y: int

    def __str__(self):
        return f"X={self.X},Y={self.Y}"

    def __iter__(self):
        return iter((self.X, self.Y))


class SimpleGridAction(IntEnum):
    """The action available in the SimpleGrid MDP."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    NO_OP = 4


@gin.constants_from_enum
class SimpleGridReward(IntEnum):
    """The reward types available in the SimpleGrid MDP."""

    AND = 0
    NAND = 1
    OR = 2
    XOR = 3


class SimpleGridMDP(MDP, ABC):
    @staticmethod
    def testing_parameters() -> Dict[str, Tuple]:
        t_params = MDP.testing_parameters()
        t_params["size"] = (5, 8, 10)
        t_params["make_reward_stochastic"] = (True, False)
        t_params["reward_type"] = (
            SimpleGridReward.AND,
            SimpleGridReward.NAND,
            SimpleGridReward.OR,
            SimpleGridReward.XOR,
        )
        t_params["number_starting_states"] = (1, 5)
        return t_params

    @staticmethod
    def get_node_class() -> Type[SimpleGridNode]:
        return SimpleGridNode

    def __init__(
        self,
        seed: int,
        size: int,
        randomize_actions: bool = True,
        lazy: float = None,
        reward_type: SimpleGridReward = SimpleGridReward.XOR,
        make_reward_stochastic=False,
        number_starting_states: int = 1,
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
            the size of the grid.
        reward_type : SimpleGridReward
            determines the reward patter given at the corners of the grid. The available patterns are AND, NAND, OR and
            XOR.
        make_reward_stochastic : bool, optional
            checks whether the rewards are to be made stochastic. By default, it is set to False.
        number_starting_states : int, optional
            the number of states in the starting distribution. By default, it is set to two.
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

        self.make_reward_stochastic = make_reward_stochastic
        self.sub_optimal_mean_reward = sub_optimal_mean_reward
        self.optimal_mean_reward = optimal_mean_reward
        dists = [
            sub_optimal_distribution,
            optimal_distribution,
            other_distribution,
        ]

        if dists.count(None) == 0:
            self.sub_optimal_distribution = sub_optimal_distribution
            self.optimal_distribution = optimal_distribution
            self.other_distribution = other_distribution
        # elif 3 > dists.count(None) > 0:
        #     raise ValueError("Y")
        else:
            if make_reward_stochastic:
                self.sub_optimal_distribution = beta(
                    1, 10 / sub_optimal_mean_reward - 1
                )
                self.optimal_distribution = beta(1, 1 / optimal_mean_reward - 1)
                self.other_distribution = beta(1, 1 / sub_optimal_mean_reward - 1)
            else:
                self.sub_optimal_distribution = deterministic(0.0)
                self.optimal_distribution = deterministic(1.0)
                self.other_distribution = deterministic(0.5)

        self.size = size
        self.n_starting_states = number_starting_states
        self.reward_type = SimpleGridReward(reward_type)

        super().__init__(
            seed=seed,
            randomize_actions=randomize_actions,
            lazy=lazy,
            size=size,
            n_starting_states=number_starting_states,
            optimal_distribution=optimal_distribution,
            other_distribution=other_distribution,
            sub_optimal_distribution=sub_optimal_distribution,
            **kwargs,
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            **super(SimpleGridMDP, self).parameters,
            **dict(
                size=self.size,
                reward_type=self.reward_type.name,
                n_starting_states=self.n_starting_states,
                optimal_mean_reward=self.optimal_mean_reward,
                sub_optimal_mean_reward=self.sub_optimal_mean_reward,
                sub_optimal_distribution=self.sub_optimal_distribution,
                optimal_distribution=self.optimal_distribution,
                other_distribution=self.other_distribution,
            ),
        }

    @property
    def possible_starting_nodes(self) -> List[SimpleGridNode]:
        return self._possible_starting_nodes

    @property
    def num_actions(self):
        return len(SimpleGridAction)

    def _calculate_next_nodes_prms(
        self, node, action
    ) -> Tuple[Tuple[dict, float], ...]:
        if action == SimpleGridAction.UP:
            return ((dict(X=node.X, Y=min(node.Y + 1, self.size - 1)), 1.0),)
        if action == SimpleGridAction.RIGHT:
            return ((dict(X=min(node.X + 1, self.size - 1), Y=node.Y), 1.0),)
        if action == SimpleGridAction.DOWN:
            return ((dict(X=node.X, Y=max(node.Y - 1, 0)), 1.0),)
        if action == SimpleGridAction.LEFT:
            return ((dict(X=max(node.X - 1, 0), Y=node.Y), 1.0),)
        if action == SimpleGridAction.NO_OP:
            return ((dict(X=node.X, Y=node.Y), 1.0),)

    @staticmethod
    def _is_corner_loop(node, next_node, size):
        return (
            node.X == next_node.X
            and node.Y == next_node.Y
            and node.X in [0, size - 1]
            and node.Y in [0, size - 1]
        )

    def _calculate_reward_distribution(
        self,
        node: SimpleGridNode,
        action: IntEnum,
        next_node: SimpleGridNode,
    ) -> rv_continuous:
        # Corner nodes
        if SimpleGridMDP._is_corner_loop(node, next_node, self.size):
            if (
                (self.reward_type == SimpleGridReward.AND and (node.X and node.Y))
                or (
                    self.reward_type == SimpleGridReward.NAND
                    and not (node.X and node.Y)
                )
                or (self.reward_type == SimpleGridReward.OR and (node.X | node.Y))
                or (self.reward_type == SimpleGridReward.XOR and (node.X ^ node.Y))
            ):
                return self.optimal_distribution
            else:
                return self.sub_optimal_distribution
        else:
            return self.other_distribution

    def _check_input_parameters(self):
        super(SimpleGridMDP, self)._check_input_parameters()

        assert self.n_starting_states <= (self.size - 1) ** 2
        assert self.optimal_mean_reward - 0.1 > self.sub_optimal_mean_reward

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

    def _calculate_starting_nodes(self):
        center = np.array(((self.size - 1) / 2, (self.size - 1) / 2))
        distances = np.empty((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                distances[x, y] = ((np.array((x, y)) - center) ** 2).sum()

        batch: list = np.array(np.where(distances == distances.min())).T.tolist()
        self._rng.shuffle(batch)
        while not np.all(distances == np.inf):
            distances[batch[0][0], batch[0][1]] = np.inf
            yield batch[0]
            batch.pop(0)
            if len(batch) == 0:
                batch: list = np.array(
                    np.where(distances == distances.min())
                ).T.tolist()

    def _instantiate_starting_node_sampler(self) -> NextStateSampler:
        starting_nodes_iter = self._calculate_starting_nodes()
        self._possible_starting_nodes = [
            self.node_class(*next(starting_nodes_iter))
            for _ in range((self.size - 1) ** 2)
        ]
        starting_nodes = self._possible_starting_nodes[: self.n_starting_states]
        self._rng.shuffle(starting_nodes)
        if len(starting_nodes) == 1:
            return NextStateSampler(next_states=starting_nodes)
        return NextStateSampler(
            next_states=starting_nodes,
            probs=[1 / self.n_starting_states for _ in range(self.n_starting_states)],
            seed=self._next_seed(),
        )

    def calc_grid_repr(self, node) -> np.array:
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[:, :] = " "

        # Corner nodes
        if self.reward_type == SimpleGridReward.AND:
            grid[0, 0] = "-"
            grid[0, -1] = "-"
            grid[-1, 0] = "-"
            grid[-1, -1] = "+"
        elif self.reward_type == SimpleGridReward.NAND:
            grid[0, 0] = "+"
            grid[0, -1] = "+"
            grid[-1, 0] = "+"
            grid[-1, -1] = "-"
        elif self.reward_type == SimpleGridReward.OR:
            grid[0, 0] = "-"
            grid[0, -1] = "+"
            grid[-1, 0] = "+"
            grid[-1, -1] = "+"
        else:
            grid[0, 0] = "-"
            grid[0, -1] = "+"
            grid[-1, 0] = "+"
            grid[-1, -1] = "-"

        grid[node.Y, node.X] = "A"
        return grid[::-1, :]
