import abc
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, Union

import numpy as np
from scipy.stats import beta, rv_continuous

from colosseum.mdp import BaseMDP
from colosseum.mdp.utils.custom_samplers import NextStateSampler
from colosseum.utils.miscellanea import check_distributions, deterministic, get_dist

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE, NODE_TYPE


@dataclass(frozen=True)
class RiverSwimNode:
    X: int

    def __str__(self):
        return f"X={self.X}"

    def __iter__(self):
        return iter((self.X, self.X))


class RiverSwimAction(IntEnum):
    LEFT = 0
    RIGHT = 1


class RiverSwimMDP(BaseMDP, abc.ABC):
    @staticmethod
    def does_seed_change_MDP_structure() -> bool:
        return False

    @staticmethod
    def _sample_parameters(
        n: int, is_episodic: bool, seed: int = None
    ) -> List[Dict[str, Any]]:
        rng = np.random.RandomState(np.random.randint(10_000) if seed is None else seed)
        samples = []
        for _ in range(n):
            p_rand, p_lazy, _ = 0.9 * np.random.dirichlet([0.2, 0.2, 5])
            sample = dict(
                size=np.minimum(2.5 + (200 / (45 * np.random.random() + 11)), 25)
                if is_episodic
                else int((6 * rng.random() + 2) ** 2.2),
                make_reward_stochastic=rng.choice([True, False]),
                p_rand=p_rand,
                p_lazy=p_lazy,
                variance_multipliers=2 * rng.random() + 0.005,
            )
            sample["p_rand"] = None if sample["p_rand"] < 0.01 else sample["p_rand"]
            sample["p_lazy"] = None if sample["p_lazy"] < 0.01 else sample["p_lazy"]

            if sample["make_reward_stochastic"]:
                sample["sub_optimal_distribution"] = beta(
                    sample["variance_multipliers"],
                    sample["variance_multipliers"] * (1 / 0.2 - 1),
                )
                sample["optimal_distribution"] = beta(
                    sample["variance_multipliers"],
                    sample["variance_multipliers"] * (1 / 0.9 - 1),
                )
                sample["other_distribution"] = beta(
                    sample["variance_multipliers"],
                    sample["variance_multipliers"] * (10 / 0.2 - 1),
                )
            else:
                sample["sub_optimal_distribution"] = deterministic(5 / 1000)
                sample["optimal_distribution"] = deterministic(1.0)
                sample["other_distribution"] = deterministic(0.0)
            samples.append(sample)
        return samples

    @staticmethod
    def get_node_class() -> Type[RiverSwimNode]:
        return RiverSwimNode

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
        variance_multipliers: float = 1.0,
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
        variance_multipliers : float, optional
            A constant that can be used to increase the variance of the reward distributions without changing their means.
            The lower the value, the higher the variance. By default, it is set to 1.
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
                    variance_multipliers,
                    variance_multipliers * (1 / sub_optimal_mean_reward - 1),
                )
                self._optimal_distribution = beta(
                    variance_multipliers,
                    variance_multipliers * (1 / optimal_mean_reward - 1),
                )
                self._other_distribution = beta(
                    variance_multipliers,
                    variance_multipliers * (10 / sub_optimal_mean_reward - 1),
                )
            else:
                self._sub_optimal_distribution = deterministic(5 / 1000)
                self._optimal_distribution = deterministic(1.0)
                self._other_distribution = deterministic(0.0)

        super(RiverSwimMDP, self).__init__(
            seed=seed,
            variance_multipliers=variance_multipliers,
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

    def _get_grid_representation(self, node: "NODE_TYPE"):
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
