import abc
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, Union

import gin
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
class SimpleGridNode:
    """
    The node for the SimpleGrid MDP.
    """

    X: int
    """x coordinate."""
    Y: int
    """y coordinate."""

    def __str__(self):
        return f"X={self.X},Y={self.Y}"

    def __iter__(self):
        return iter((self.X, self.Y))


class SimpleGridAction(IntEnum):
    """
    The actions available in the SimpleGrid MDP.
    """

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    NO_OP = 4


@gin.constants_from_enum
class SimpleGridReward(IntEnum):
    """
    The reward types available in the SimpleGrid MDP. It controls the rewards for the corner states.
    """

    AND = 0
    NAND = 1
    OR = 2
    XOR = 3


class SimpleGridMDP(BaseMDP, abc.ABC):
    """
    The base class for the SimpleGrid family.
    """

    @staticmethod
    def get_action_class():
        return SimpleGridAction

    @staticmethod
    def get_unique_symbols() -> List[str]:
        return [" ", "A", "+", "-"]

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
                size=int(
                    (
                        1
                        + np.minimum((800 / (100 * rng.random() + 35)), 25)
                        * (0.8 if is_episodic else 1)
                    )
                ),
                n_starting_states=rng.randint(1, 5),
                p_rand=p_rand,
                p_lazy=p_lazy,
                make_reward_stochastic=rng.choice([True, False]),
                reward_variance_multiplier=2 * rng.random() + 0.005,
            )
            sample["p_rand"] = None if sample["p_rand"] < 0.01 else sample["p_rand"]
            sample["p_lazy"] = None if sample["p_lazy"] < 0.01 else sample["p_lazy"]

            sample["reward_type"] = rng.randint(4)

            if sample["make_reward_stochastic"]:
                sample["sub_optimal_distribution"] = (
                    "beta",
                    (
                        sample["reward_variance_multiplier"],
                        sample["reward_variance_multiplier"] * (10 / 0.2 - 1),
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
                        sample["reward_variance_multiplier"] * (1 / 0.2 - 1),
                    ),
                )
            else:
                sample["sub_optimal_distribution"] = ("deterministic", (0.0,))
                sample["optimal_distribution"] = ("deterministic", (1.0,))
                sample["other_distribution"] = ("deterministic", (0.5,))

            samples.append(rounding_nested_structure(sample))
        return samples

    @staticmethod
    def get_node_class() -> Type["NODE_TYPE"]:
        return SimpleGridNode

    def get_gin_parameters(self, index: int) -> str:
        prms = dict(
            size=self._size,
            n_starting_states=self._n_starting_states,
            reward_type=int(self._reward_type),
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

        return SimpleGridMDP.produce_gin_file_from_mdp_parameters(
            prms, type(self).__name__, index
        )

    @property
    def n_actions(self) -> int:
        return len(SimpleGridAction)

    def _get_next_nodes_parameters(
        self, node: "NODE_TYPE", action: "ACTION_TYPE"
    ) -> Tuple[Tuple[dict, float], ...]:
        if action == SimpleGridAction.UP:
            return ((dict(X=node.X, Y=min(node.Y + 1, self._size - 1)), 1.0),)
        if action == SimpleGridAction.RIGHT:
            return ((dict(X=min(node.X + 1, self._size - 1), Y=node.Y), 1.0),)
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

    def _get_reward_distribution(
        self, node: "NODE_TYPE", action: "ACTION_TYPE", next_node: "NODE_TYPE"
    ) -> rv_continuous:
        # Corner nodes
        if SimpleGridMDP._is_corner_loop(node, next_node, self._size):
            if (
                (self._reward_type == SimpleGridReward.AND and (node.X and node.Y))
                or (
                    self._reward_type == SimpleGridReward.NAND
                    and not (node.X and node.Y)
                )
                or (self._reward_type == SimpleGridReward.OR and (node.X | node.Y))
                or (self._reward_type == SimpleGridReward.XOR and (node.X ^ node.Y))
            ):
                return self._optimal_distribution
            else:
                return self._sub_optimal_distribution
        else:
            return self._other_distribution

    def _calculate_starting_nodes(self):
        center = np.array(((self._size - 1) / 2, (self._size - 1) / 2))
        distances = np.empty((self._size, self._size))
        for x in range(self._size):
            for y in range(self._size):
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

    def _get_starting_node_sampler(self) -> NextStateSampler:
        starting_nodes_iter = self._calculate_starting_nodes()
        self.__possible_starting_nodes = [
            self.get_node_class()(*next(starting_nodes_iter))
            for _ in range((self._size - 1) ** 2)
        ]
        starting_nodes = self._possible_starting_nodes[: self._n_starting_states]
        self._rng.shuffle(starting_nodes)
        if len(starting_nodes) == 1:
            return NextStateSampler(next_nodes=starting_nodes)
        return NextStateSampler(
            next_nodes=starting_nodes,
            probs=[1 / self._n_starting_states for _ in range(self._n_starting_states)],
            seed=self._produce_random_seed(),
        )

    def _check_parameters_in_input(self):
        super(SimpleGridMDP, self)._check_parameters_in_input()

        assert self._n_starting_states <= (self._size - 1) ** 2
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
        grid = np.zeros((self._size, self._size), dtype=str)
        grid[:, :] = " "

        # Corner nodes
        if self._reward_type == SimpleGridReward.AND:
            grid[0, 0] = "-"
            grid[0, -1] = "-"
            grid[-1, 0] = "-"
            grid[-1, -1] = "+"
        elif self._reward_type == SimpleGridReward.NAND:
            grid[0, 0] = "+"
            grid[0, -1] = "+"
            grid[-1, 0] = "+"
            grid[-1, -1] = "-"
        elif self._reward_type == SimpleGridReward.OR:
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

    @property
    def _possible_starting_nodes(self) -> List["NODE_TYPE"]:
        return self.__possible_starting_nodes

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            **super(SimpleGridMDP, self).parameters,
            **dict(
                size=self._size,
                reward_type=self._reward_type,
                n_starting_states=self._n_starting_states,
                optimal_mean_reward=self._optimal_mean_reward,
                sub_optimal_mean_reward=self._sub_optimal_mean_reward,
                optimal_distribution=self._optimal_distribution,
                sub_optimal_distribution=self._sub_optimal_distribution,
                other_distribution=self._other_distribution,
            ),
        }

    def __init__(
        self,
        seed: int,
        size: int,
        reward_type: SimpleGridReward = SimpleGridReward.XOR,
        n_starting_states: int = 1,
        optimal_mean_reward: float = 0.9,
        sub_optimal_mean_reward: float = 0.2,
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
        reward_type : SimpleGridReward
            The type of reward for the MDP. By default, the XOR type is used.
        n_starting_states : int
            The number of possible starting states.
        optimal_mean_reward : float
            If the rewards are made stochastic, this parameter controls the mean reward for the optimal trajectory.
            By default, it is set to 0.9.
        sub_optimal_mean_reward : float
            If the rewards are made stochastic, this parameter controls the mean reward for suboptimal trajectories.
            By default, it is set to 0.2.
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
        self._reward_type = SimpleGridReward(reward_type)
        self._n_starting_states = n_starting_states
        self._optimal_mean_reward = optimal_mean_reward
        self._sub_optimal_mean_reward = sub_optimal_mean_reward
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
                    reward_variance_multiplier * (10 / sub_optimal_mean_reward - 1),
                )
                self._optimal_distribution = beta(
                    reward_variance_multiplier,
                    reward_variance_multiplier * (1 / optimal_mean_reward - 1),
                )
                self._other_distribution = beta(
                    reward_variance_multiplier,
                    reward_variance_multiplier * (1 / sub_optimal_mean_reward - 1),
                )
            else:
                self._sub_optimal_distribution = deterministic(0.0)
                self._optimal_distribution = deterministic(1.0)
                self._other_distribution = deterministic(0.5)

        super(SimpleGridMDP, self).__init__(
            seed=seed,
            reward_variance_multiplier=reward_variance_multiplier,
            make_reward_stochastic=make_reward_stochastic,
            **kwargs,
        )
