import abc
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, Union

import gin
import numpy as np
from scipy.stats import beta, rv_continuous

from colosseum.mdp import BaseMDP
from colosseum.mdp.utils.custom_samplers import NextStateSampler
from colosseum.utils.miscellanea import check_distributions, deterministic, get_dist

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE, NODE_TYPE


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


class SimpleGridMDP(BaseMDP, abc.ABC):
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
                size=int(
                    (
                        1
                        + np.minimum((800 / (100 * np.random.random() + 35)), 25)
                        * (0.8 if is_episodic else 1)
                    )
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
                sample["sub_optimal_distribution"] = beta(
                    sample["variance_multipliers"],
                    sample["variance_multipliers"] * (10 / 0.2 - 1),
                )
                sample["optimal_distribution"] = beta(
                    sample["variance_multipliers"],
                    sample["variance_multipliers"] * (1 / 0.9 - 1),
                )
                sample["other_distribution"] = beta(
                    sample["variance_multipliers"],
                    sample["variance_multipliers"] * (1 / 0.2 - 1),
                )
            else:
                sample["sub_optimal_distribution"] = deterministic(0.0)
                sample["optimal_distribution"] = deterministic(1.0)
                sample["other_distribution"] = deterministic(0.5)
            samples.append(sample)
        return samples

    @staticmethod
    def get_node_class() -> Type["NODE_TYPE"]:
        return SimpleGridNode

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

    def _get_grid_representation(self, node: "NODE_TYPE"):
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
        variance_multipliers: float = 1.0,
        **kwargs,
    ):
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
                    variance_multipliers,
                    variance_multipliers * (10 / sub_optimal_mean_reward - 1),
                )
                self._optimal_distribution = beta(
                    variance_multipliers,
                    variance_multipliers * (1 / optimal_mean_reward - 1),
                )
                self._other_distribution = beta(
                    variance_multipliers,
                    variance_multipliers * (1 / sub_optimal_mean_reward - 1),
                )
            else:
                self._sub_optimal_distribution = deterministic(0.0)
                self._optimal_distribution = deterministic(1.0)
                self._other_distribution = deterministic(0.5)

        super(SimpleGridMDP, self).__init__(
            seed=seed,
            variance_multipliers=variance_multipliers,
            make_reward_stochastic=make_reward_stochastic,
            **kwargs,
        )
