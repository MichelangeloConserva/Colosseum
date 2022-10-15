import abc
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import IntEnum
from itertools import product
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, Union

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


class TaxiAction(IntEnum):
    """
    The actions available in the Taxi MDP.
    """

    MoveSouth = 0
    MoveNorth = 1
    MoveEast = 2
    MoveWest = 3
    PickUpPassenger = 4
    DropOffPassenger = 5


@dataclass(frozen=True)
class TaxiNode:
    """
    The node for the Taxi MDP.
    """

    X: int
    """x coordinate of the taxi."""
    Y: int
    """y coordinate of the taxi."""
    XPass: int
    """x coordinate of the passenger, -1 if it is on board."""
    YPass: int
    """y coordinate of the taxi, -1 if it is on board."""
    XDest: int
    """x coordinate of the destination."""
    YDest: int
    """y coordinate of the destination."""

    def __str__(self):
        return f"X={self.X},Y={self.Y},XPass={self.XPass},YPass={self.YPass},XDest={self.XDest},YDest={self.YDest}"


class TaxiMDP(BaseMDP, abc.ABC):
    """
    The base class for the Taxi family.
    """

    @staticmethod
    def get_unique_symbols() -> List[str]:
        return [" ", "A", "X", "D", "P"]

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
            p_rand, p_lazy, _ = 0.5 * rng.dirichlet([0.2, 0.2, 5])
            sample = dict(
                size=5
                if is_episodic
                else rng.choice(range(5, 8), None, True, [0.525, 0.325, 0.15]),
                p_rand=p_rand * (0.8 if is_episodic else 1),
                p_lazy=p_lazy * (0.8 if is_episodic else 1),
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
                        sample["reward_variance_multiplier"] * (1 / 0.2 - 1),
                    ),
                )
                sample["successfully_delivery_r"] = (
                    "beta",
                    (
                        sample["reward_variance_multiplier"],
                        sample["reward_variance_multiplier"] * (1 / 0.9 - 1),
                    ),
                )
                sample["failure_delivery_r"] = (
                    "beta",
                    (
                        sample["reward_variance_multiplier"],
                        sample["reward_variance_multiplier"] * (10 / 0.2 - 1),
                    ),
                )
            else:
                sample["default_r"] = ("deterministic", (0.1,))
                sample["successfully_delivery_r"] = ("deterministic", (1.0,))
                sample["failure_delivery_r"] = ("deterministic", (0.0,))

            samples.append(rounding_nested_structure(sample))
        return samples

    @property
    def _quadrant_width(self):
        return self._size / int(self._n_locations ** 0.5) / 2

    @property
    def _admissible_coordinate(self):
        rows = []
        j = 0
        while len(rows) < self._size:
            if j % 2 != 0:
                row = []
            else:
                row = [0] * int((self._width + self._space) // 2)
            i = 0
            while len(row) < self._size:
                row.append(int(i % (1 + self._space) == 0))
                if row[-1] == 1:
                    for _ in range(self._width - 1):
                        if len(row) == self._size:
                            break
                        row.append(1)
                i += 1
            for _ in range(self._length):
                if len(rows) == self._size:
                    break
                rows.append(row)
            if len(rows) < self._size:
                rows.append([0] * self._size)
            j += 1
        return np.vstack(np.where(np.array(rows) == 0)).T.tolist()

    @property
    def _quadrants(self):
        quadrants = np.zeros((self._size, self._size))
        split = np.array_split(range(self._size), int(self._n_locations ** 0.5))
        for i, (x, y) in enumerate(product(split, split)):
            for q_coo_x, q_coo_y in product(x, y):
                quadrants[q_coo_x, q_coo_y] = i
        quadrants = [
            list(
                filter(
                    lambda x: x in self._admissible_coordinate,
                    np.vstack(np.where(quadrants == i)).T.tolist(),
                )
            )
            for i in range(self._n_locations)
        ]

        assert all(len(q) != 0 for q in quadrants)
        return quadrants

    @property
    def locations(self):
        if len(self._locations) == 0:
            re_sample = True
            min_distance = max(self._quadrant_width, 2)
            while re_sample:
                locations = [
                    self._quadrants[i][self._rng.randint(len(self._quadrants[i]))]
                    for i in range(self._n_locations)
                ]
                re_sample = False
                nplocations = np.array(locations)
                for i in range(self._n_locations):
                    for j in range(1 + i, self._n_locations):
                        diff = np.sqrt(((nplocations[i] - nplocations[j]) ** 2).sum())
                        if diff <= min_distance:
                            re_sample = True
                            break
                    if re_sample:
                        break
            self._rng.shuffle(locations)
            self._locations = locations[: self.n_locations]
        return self._locations

    @staticmethod
    def get_node_class() -> Type["NODE_TYPE"]:
        return TaxiNode

    def get_gin_parameters(self, index: int) -> str:
        prms = dict(
            size=self._size,
            make_reward_stochastic=self._make_reward_stochastic,
            reward_variance_multiplier=self._reward_variance_multiplier,
            default_r=(
                self._default_r.dist.name,
                self._default_r.args,
            ),
            successfully_delivery_r=(
                self._successfully_delivery_r.dist.name,
                self._successfully_delivery_r.args,
            ),
            failure_delivery_r=(
                self._failure_delivery_r.dist.name,
                self._failure_delivery_r.args,
            ),
        )
        if self._p_rand is not None:
            prms["p_rand"] = self._p_rand

        return TaxiMDP.produce_gin_file_from_mdp_parameters(
            prms, type(self).__name__, index
        )

    @property
    def n_actions(self) -> int:
        return len(TaxiAction)

    def _get_next_nodes_parameters(
        self, node: "NODE_TYPE", action: "ACTION_TYPE"
    ) -> Tuple[Tuple[dict, float], ...]:
        next_node_prms = asdict(node)

        if action == TaxiAction.DropOffPassenger:
            # we have the passenger and we are dropping time(er/im) in the right place
            if node.XPass == -1 and node.X == node.XDest and node.Y == node.YDest:
                next_nodes_prms = []

                n = 0
                for pass_loc in filter(
                    lambda loc: loc != [node.X, node.Y],
                    self.locations,
                ):
                    n += len(list(filter(lambda loc: loc != pass_loc, self.locations)))
                p = 1.0 / n

                for pass_loc in filter(
                    lambda loc: loc != [node.X, node.Y],
                    self.locations,
                ):
                    admissible_destinations = list(
                        filter(lambda loc: loc != pass_loc, self.locations)
                    )

                    for destination in admissible_destinations:
                        cur_next_node_prms: dict = deepcopy(next_node_prms)
                        (
                            cur_next_node_prms["XPass"],
                            cur_next_node_prms["YPass"],
                        ) = pass_loc
                        (
                            cur_next_node_prms["XDest"],
                            cur_next_node_prms["YDest"],
                        ) = destination
                        next_nodes_prms.append((cur_next_node_prms, p))
                return tuple(next_nodes_prms)

        if action == TaxiAction.PickUpPassenger:
            if node.XPass != -1 and node.X == node.XPass and node.Y == node.YPass:
                next_node_prms["XPass"] = -1
                next_node_prms["YPass"] = -1

        if action == TaxiAction.MoveNorth:
            next_coord = [node.X, node.Y + 1]
        elif action == TaxiAction.MoveEast:
            next_coord = [node.X + 1, node.Y]
        elif action == TaxiAction.MoveSouth:
            next_coord = [node.X, node.Y - 1]
        elif action == TaxiAction.MoveWest:
            next_coord = [node.X - 1, node.Y]
        else:
            next_coord = [node.X, node.Y]
        if next_coord in self._admissible_coordinate:
            next_node_prms["X"] = next_coord[0]
            next_node_prms["Y"] = next_coord[1]

        return ((next_node_prms, 1.0),)

    def _get_reward_distribution(
        self, node: "NODE_TYPE", action: "ACTION_TYPE", next_node: "NODE_TYPE"
    ) -> rv_continuous:
        if action == TaxiAction.PickUpPassenger:
            if next_node.XPass != -1 or node.XPass == -1:
                # We don't have the passenger
                return self._failure_delivery_r
        if action == TaxiAction.DropOffPassenger:
            if next_node.XPass == -1 or node.XPass != -1:
                # We didn't drop the passenger in the destination
                return self._failure_delivery_r
            elif node.XPass == -1 and next_node.XPass != -1:
                return self._successfully_delivery_r
        return self._default_r

    def _get_starting_node_sampler(self) -> NextStateSampler:
        starting_nodes = []
        for (
            (pass_loc_x, pass_loc_y),
            (destination_x, destination_y),
            (taxi_x, taxi_y),
        ) in product(self.locations, self.locations, self._admissible_coordinate):
            if (pass_loc_x, pass_loc_y) == (destination_x, destination_y):
                continue

            starting_nodes.append(
                TaxiNode(
                    taxi_x, taxi_y, pass_loc_x, pass_loc_y, destination_x, destination_y
                )
            )
        self._rng.shuffle(starting_nodes)

        return NextStateSampler(
            next_nodes=starting_nodes,
            probs=[1 / len(starting_nodes) for _ in range(len(starting_nodes))],
            seed=self._produce_random_seed(),
        )

    def _check_parameters_in_input(self):
        super(TaxiMDP, self)._check_parameters_in_input()

        assert (
            self._failure_delivery_r.mean()
            < self._default_r.mean()
            < self._successfully_delivery_r.mean()
        )
        assert self._size > 3
        assert self.n_locations > (1 if self.is_episodic() else 2)
        assert self._size > self._length
        assert self._size > self._width
        assert self._size > self._space / 2
        assert self._size > 2 * self.n_locations ** 0.5
        assert self._optimal_mean_reward - 0.1 > self._sub_optimal_mean_reward

        dists = [
            self._default_r,
            self._failure_delivery_r,
            self._successfully_delivery_r,
        ]
        check_distributions(
            dists,
            self._make_reward_stochastic,
        )

    def _get_grid_representation(self, node: "NODE_TYPE") -> np.ndarray:
        grid = np.zeros((self._size, self._size), dtype=str)
        grid[:, :] = "X"
        for coo_x, coo_y in self._admissible_coordinate:
            grid[coo_x, coo_y] = " "

        grid[node.XDest, node.YDest] = "D"
        if node.XPass != -1:
            grid[node.XPass, node.YPass] = "P"
        grid[node.X, node.Y] = "A"
        return grid[::-1, :]

    @property
    def _possible_starting_nodes(self) -> List["NODE_TYPE"]:
        return self._starting_node_sampler.next_states

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            **super(TaxiMDP, self).parameters,
            **dict(
                size=self._size,
                length=self._length,
                width=self._width,
                space=self._space,
                n_locations=self._n_locations,
                optimal_mean_reward=self._optimal_mean_reward,
                sub_optimal_mean_reward=self._sub_optimal_mean_reward,
                default_r=self._default_r,
                successfully_delivery_r=self._successfully_delivery_r,
                failure_delivery_r=self._failure_delivery_r,
            ),
        }

    def __init__(
        self,
        seed: int,
        size: int,
        length=2,
        width=1,
        space=1,
        n_locations=2 ** 2,
        optimal_mean_reward: float = 0.9,
        sub_optimal_mean_reward: float = 0.2,
        default_r: Union[Tuple, rv_continuous] = None,
        successfully_delivery_r: Union[Tuple, rv_continuous] = None,
        failure_delivery_r: Union[Tuple, rv_continuous] = None,
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
        length : int
            The length of the walls.
        width : int
            The width of the walls.
        space : int
            The space between walls.
        n_locations : int
            The number of possible spawn locations. It must be a squared number.
        optimal_mean_reward : float
            If the rewards are made stochastic, this parameter controls the mean reward for the optimal trajectory.
            By default, it is set to 0.9.
        sub_optimal_mean_reward: float
            If the rewards are made stochastic, this parameter controls the mean reward for suboptimal trajectories.
            By default, it is set to 0.1.
        default_r
        successfully_delivery_r : Union[Tuple, rv_continuous]
            The reward distribution for successfully delivering a passenger. It can be either passed as a tuple
            containing Beta parameters or as a rv_continuous object.
        failure_delivery_r
            The reward distribution for failing to deliver a passenger. It can be either passed as a tuple containing
            Beta parameters or as a rv_continuous object.
        make_reward_stochastic : bool
            If True, the rewards of the MDP will be stochastic. By default, it is set to False.
        reward_variance_multiplier : float
            A constant that can be used to increase the variance of the reward distributions without changing their means.
            The lower the value, the higher the variance. By default, it is set to 1.
        """

        if type(successfully_delivery_r) == tuple:
            successfully_delivery_r = get_dist(
                successfully_delivery_r[0], successfully_delivery_r[1]
            )
        if type(failure_delivery_r) == tuple:
            failure_delivery_r = get_dist(failure_delivery_r[0], failure_delivery_r[1])

        if type(default_r) == tuple:
            default_r = get_dist(default_r[0], default_r[1])

        self._size = size
        self._length = length
        self._width = width
        self._space = space
        self.n_locations = n_locations
        self._n_locations = int(np.ceil(n_locations ** 0.5) ** 2)
        self._optimal_mean_reward = optimal_mean_reward
        self._sub_optimal_mean_reward = sub_optimal_mean_reward
        self._locations = []

        dists = [default_r, successfully_delivery_r, failure_delivery_r]
        if dists.count(None) == 0:
            self._default_r = default_r
            self._successfully_delivery_r = successfully_delivery_r
            self._failure_delivery_r = failure_delivery_r
        else:
            if make_reward_stochastic:
                self._default_r = beta(
                    reward_variance_multiplier,
                    reward_variance_multiplier * (1 / sub_optimal_mean_reward - 1),
                )
                self._successfully_delivery_r = beta(
                    reward_variance_multiplier,
                    reward_variance_multiplier * (1 / optimal_mean_reward - 1),
                )
                self._failure_delivery_r = beta(
                    reward_variance_multiplier,
                    reward_variance_multiplier * (10 / sub_optimal_mean_reward - 1),
                )
            else:
                self._default_r = deterministic(0.1)
                self._successfully_delivery_r = deterministic(1)
                self._failure_delivery_r = deterministic(0)

        kwargs[
            "randomize_actions"
        ] = False  # TODO : double check whether this is actually necessary or not

        super(TaxiMDP, self).__init__(
            seed=seed,
            reward_variance_multiplier=reward_variance_multiplier,
            make_reward_stochastic=make_reward_stochastic,
            **kwargs,
        )
