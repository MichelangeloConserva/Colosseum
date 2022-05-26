from abc import ABC
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import IntEnum

from colosseum.utils.random_vars import deterministic, get_dist

try:
    from functools import cached_property
except:
    from backports.cached_property import cached_property

from itertools import product
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
from scipy.stats import beta, rv_continuous

from colosseum.mdps import MDP
from colosseum.mdps.base_mdp import NextStateSampler
from colosseum.utils.mdps import check_distributions


class TaxiAction(IntEnum):
    """The action available in the MiniGridEmpty MDP."""

    MoveSouth = 0
    MoveNorth = 1
    MoveEast = 2
    MoveWest = 3
    PickUpPassenger = 4
    DropOffPassenger = 5


@dataclass(frozen=True)
class TaxiNode:
    X: int
    Y: int
    XPass: int
    YPass: int
    XDest: int
    YDest: int

    def __str__(self):
        return f"X={self.X},Y={self.Y},XPass={self.XPass},YPass={self.YPass},XDest={self.XDest},YDest={self.YDest}"


class TaxiMDP(MDP, ABC):
    @staticmethod
    def testing_parameters() -> Dict[str, Tuple]:
        t_params = MDP.testing_parameters()
        t_params["size"] = (8, 10)
        t_params["length"] = (1, 2)
        t_params["width"] = (1, 2)
        t_params["space"] = (1, 2)
        t_params["n_locations"] = (4, 9)
        t_params["make_reward_stochastic"] = (True, False)

        return t_params

    @staticmethod
    def get_node_class() -> Type[TaxiNode]:
        return TaxiNode

    def __init__(
        self,
        seed: int,
        size: int,
        lazy: float = None,
        randomize_actions: bool = True,
        make_reward_stochastic=False,
        length=2,
        width=1,
        space=1,
        n_locations=2 ** 2,
        optimal_mean_reward: float = 0.9,
        sub_optimal_mean_reward: float = 0.2,
        default_r: rv_continuous = None,
        successfully_delivery_r: Union[Tuple, rv_continuous] = None,
        failure_delivery_r: Union[Tuple, rv_continuous] = None,
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
        length : int, optional
            the length of the walls. By default, it is set to two.
        width : int, optional
            the width of the walls. By default, it is set to one.
        space : int, optional
            the space between walls. By default, it is set to one.
        n_locations : int, optional
            the number of locations in which the passenger and the destination can spawn. By default, it is set to four.
        optimal_mean_reward : float, optional
            if the rewards are made stochastic, this parameter controls the mean reward for the highly rewarding states.
            By default, it is set to 0.9.
        sub_optimal_mean_reward: float, optional
            if the rewards are made stochastic, this parameter controls the mean reward for the suboptimal states.
            By default, it is set to 0.2.
        default_r: Union[Tuple, rv_continuous]
            The reward distribution for moving and picking up passengers. It can be either passed as a tuple containing
            Beta parameters or as a rv_continuous object.
        successfully_delivery_r: Union[Tuple, rv_continuous]
            The reward distribution for successfully delivering a passenger. It can be either passed as a tuple
            containing Beta parameters or as a rv_continuous object.
        failure_delivery_r: Union[Tuple, rv_continuous]
            The reward distribution for failing to deliver a passenger.It can be either passed as a tuple containing
            Beta parameters or as a rv_continuous object.
        """

        if type(successfully_delivery_r) == tuple:
            successfully_delivery_r = get_dist(
                successfully_delivery_r[0], successfully_delivery_r[1:]
            )
        if type(failure_delivery_r) == tuple:
            failure_delivery_r = get_dist(failure_delivery_r[0], failure_delivery_r[1:])

        if type(default_r) == tuple:
            default_r = get_dist(default_r[0], default_r[1:])

        randomize_actions = False  # There is a bug when this is set to False. Not very important since there are many actions.

        self.sub_optimal_mean_reward = sub_optimal_mean_reward
        self.optimal_mean_reward = optimal_mean_reward
        self.n_locations = n_locations
        self._n_locations = int(np.ceil(n_locations ** 0.5) ** 2)
        self.space = space
        self.width = width
        self.length = length
        self.size = size
        self.make_reward_stochastic = make_reward_stochastic

        dists = [default_r, successfully_delivery_r, failure_delivery_r]
        if dists.count(None) == 0:
            self.default_r = default_r
            self.successfully_delivery_r = successfully_delivery_r
            self.failure_delivery_r = failure_delivery_r
        else:
            if make_reward_stochastic:
                self.default_r = beta(1, 1 / sub_optimal_mean_reward - 1)
                self.successfully_delivery_r = beta(1, 1 / optimal_mean_reward - 1)
                self.failure_delivery_r = beta(1, 10 / sub_optimal_mean_reward - 1)
            else:
                self.default_r = deterministic(0.1)
                self.successfully_delivery_r = deterministic(1)
                self.failure_delivery_r = deterministic(0)

        super().__init__(
            seed=seed,
            lazy=lazy,
            randomize_actions=randomize_actions,
            size=size,
            **kwargs,
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            **super(TaxiMDP, self).parameters,
            **dict(
                size=self.size,
                length=self.length,
                width=self.width,
                space=self.space,
                n_locations=self.n_locations,
                optimal_mean_reward=self.optimal_mean_reward,
                sub_optimal_mean_reward=self.sub_optimal_mean_reward,
                default_r=self.default_r,
                successfully_delivery_r=self.successfully_delivery_r,
                failure_delivery_r=self.failure_delivery_r,
            ),
        }

    @property
    def _quadrant_width(self):
        return self.size / int(self._n_locations ** 0.5) / 2

    @cached_property
    def _admissible_coordinate(self):
        rows = []
        j = 0
        while len(rows) < self.size:
            if j % 2 != 0:
                row = []
            else:
                row = [0] * int((self.width + self.space) // 2)
            i = 0
            while len(row) < self.size:
                row.append(int(i % (1 + self.space) == 0))
                if row[-1] == 1:
                    for _ in range(self.width - 1):
                        if len(row) == self.size:
                            break
                        row.append(1)
                i += 1
            for _ in range(self.length):
                if len(rows) == self.size:
                    break
                rows.append(row)
            if len(rows) < self.size:
                rows.append([0] * self.size)
            j += 1
        return np.vstack(np.where(np.array(rows) == 0)).T.tolist()

    @cached_property
    def _quadrants(self):
        quadrants = np.zeros((self.size, self.size))
        split = np.array_split(range(self.size), int(self._n_locations ** 0.5))
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

    @cached_property
    def locations(self):
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
        return locations[: self.n_locations]

    @property
    def possible_starting_nodes(self) -> List[TaxiNode]:
        return self.starting_node_sampler.next_states

    @property
    def num_actions(self):
        return len(TaxiAction)

    def _calculate_next_nodes_prms(
        self, node: TaxiNode, action: int
    ) -> Tuple[Tuple[dict, float], ...]:
        next_node_prms = asdict(node)

        if action == TaxiAction.DropOffPassenger:
            # we have the passenger and we are dropping h(er/im) in the right place
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

    def _calculate_reward_distribution(
        self,
        node: TaxiNode,
        action: IntEnum,
        next_node: TaxiNode,
    ) -> rv_continuous:
        if action == TaxiAction.PickUpPassenger:
            if next_node.XPass != -1 or node.XPass == -1:
                # We don't have the passenger
                return self.failure_delivery_r
        if action == TaxiAction.DropOffPassenger:
            if next_node.XPass == -1 or node.XPass != -1:
                # We didn't drop the passenger in the destination
                return self.failure_delivery_r
            elif node.XPass == -1 and next_node.XPass != -1:
                return self.successfully_delivery_r
        return self.default_r

    def _check_input_parameters(self):
        super(TaxiMDP, self)._check_input_parameters()

        assert (
            self.failure_delivery_r.mean()
            < self.default_r.mean()
            < self.successfully_delivery_r.mean()
        )
        assert self.size > 3
        assert self.n_locations > (1 if self.is_episodic() else 2)
        assert self.size > self.length
        assert self.size > self.width
        assert self.size > self.space / 2
        assert self.size > 2 * self.n_locations ** 0.5
        assert self.optimal_mean_reward - 0.1 > self.sub_optimal_mean_reward

        if self.lazy:
            assert self.lazy <= 0.9
        dists = [
            self.default_r,
            self.failure_delivery_r,
            self.successfully_delivery_r,
        ]
        check_distributions(
            dists,
            self.make_reward_stochastic,
        )

    def _instantiate_starting_node_sampler(self) -> NextStateSampler:
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
            next_states=starting_nodes,
            probs=[1 / len(starting_nodes) for _ in range(len(starting_nodes))],
            seed=self._next_seed(),
        )

    def calc_grid_repr(self, node: Any) -> np.array:
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[:, :] = "X"
        for coo_x, coo_y in self._admissible_coordinate:
            grid[coo_x, coo_y] = " "

        grid[node.XDest, node.YDest] = "D"
        if node.XPass != -1:
            grid[node.XPass, node.YPass] = "P"
        grid[node.X, node.Y] = "A"
        return grid[::-1, :]
