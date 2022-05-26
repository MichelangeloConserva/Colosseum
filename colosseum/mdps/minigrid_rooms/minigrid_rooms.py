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

from colosseum.mdps.base_mdp import MDP, NextStateSampler
from colosseum.utils.mdps import check_distributions


class MiniGridRoomsAction(IntEnum):
    """The action available in the MiniGridRooms MDP."""

    MoveForward = 0
    TurnRight = 1
    TurnLeft = 2


class MiniGridRoomsDirection(IntEnum):
    """The possible agent_directions in the MiniGridRooms MDP."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def grid_movement(self) -> np.array:
        """:returns the effect caused by each action in grid space."""
        if self == MiniGridRoomsDirection.UP:
            return np.array((0, 1))
        elif self == MiniGridRoomsDirection.DOWN:
            return np.array((0, -1))
        elif self == MiniGridRoomsDirection.RIGHT:
            return np.array((1, 0))
        else:
            return np.array((-1, 0))


@dataclass(frozen=True)
class NodeGridRooms:
    X: int
    Y: int
    Dir: MiniGridRoomsDirection

    def __str__(self):
        return f"X={self.X},Y={self.Y},Dir={MiniGridRoomsDirection(self.Dir).name}"


class MiniGridRoomsMDP(MDP):
    @staticmethod
    def testing_parameters() -> Dict[str, Tuple]:
        t_params = MDP.testing_parameters()
        t_params["room_size"] = (3, 4, 5)
        t_params["n_rooms"] = (4, 9)
        t_params["make_reward_stochastic"] = (True, False)
        t_params["n_starting_states"] = (1, 4)
        return t_params

    @staticmethod
    def get_node_class() -> Type[NodeGridRooms]:
        return NodeGridRooms

    @staticmethod
    def get_positions_coords_in_room(
        room_size: int, room_coord: Tuple[int, int]
    ) -> np.array:
        x_room_coord, y_room_coord = room_coord
        nodes = np.zeros((room_size, room_size), dtype=object)
        for i in range(room_size):
            for j in range(room_size):
                nodes[j, i] = (
                    i + (room_size + 1) * x_room_coord,
                    j + (room_size + 1) * y_room_coord,
                )
        nodes = nodes[::-1]
        return nodes

    def __init__(
        self,
        seed: int,
        room_size: int,
        randomize_actions: bool = True,
        lazy: float = None,
        make_reward_stochastic=False,
        n_rooms: int = 4,
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
        room_size : int
            the size of the rooms.
        lazy : float
            the probability of an action not producing any effect on the MDP. By default, it is set to zero.
        make_reward_stochastic : bool, optional
            checks whether the rewards are to be made stochastic. By default, it is set to False.
        n_rooms : int
            the number of rooms.
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
        self.room_size = room_size
        self.n_rooms = n_rooms
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
                size = int(room_size * n_rooms ** 0.5)
                self.other_distribution = beta(1, size ** 2 - 1)
                self.optimal_distribution = beta(size ** 2 - 1, 1)
            else:
                self.optimal_distribution = deterministic(1.0)
                self.other_distribution = deterministic(0.0)

        super().__init__(
            seed=seed,
            randomize_actions=randomize_actions,
            lazy=lazy,
            n_starting_states=n_starting_states,
            optimal_distribution=optimal_distribution,
            other_distribution=other_distribution,
            room_size=room_size,
            n_rooms=n_rooms,
            **kwargs,
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            **super(MiniGridRoomsMDP, self).parameters,
            **dict(
                room_size=self.room_size,
                n_rooms=self.n_rooms,
                n_starting_states=self.n_starting_states,
                optimal_distribution=self.optimal_distribution,
                other_distribution=self.other_distribution,
            ),
        }

    @cached_property
    def _admissible_coordinate(self) -> list:
        rooms_per_row = int(np.sqrt(self.n_rooms))

        vertical_checkers = [
            j * (self.room_size) + j + int(np.floor(self.room_size / 2))
            for j in range(rooms_per_row)
        ]
        horizontal_checkers = [
            j * self.room_size + j - 1 for j in range(1, rooms_per_row)
        ]
        door_positions = list(product(horizontal_checkers, vertical_checkers)) + list(
            product(vertical_checkers, horizontal_checkers)
        )
        rooms_coordinates = []
        for room_coord in product(range(rooms_per_row), range(rooms_per_row)):
            room = self.get_positions_coords_in_room(self.room_size, room_coord)
            for c in room.ravel().tolist():
                rooms_coordinates.append(tuple(c))
        return rooms_coordinates + door_positions

    @property
    def possible_starting_nodes(self) -> List[NodeGridRooms]:
        return self._possible_starting_nodes

    @property
    def num_actions(self):
        return len(MiniGridRoomsAction)

    def _calculate_next_nodes_prms(
        self, node: NodeGridRooms, action
    ) -> Tuple[Tuple[dict, float], ...]:
        d = node.Dir
        if action == MiniGridRoomsAction.TurnRight:
            return ((dict(X=node.X, Y=node.Y, Dir=((d + 1) % 4)), 1.0),)
        if action == MiniGridRoomsAction.TurnLeft:
            return ((dict(X=node.X, Y=node.Y, Dir=((d - 1) % 4)), 1.0),)
        if action == MiniGridRoomsAction.MoveForward:
            if d == MiniGridRoomsDirection.UP:
                next_coord = (node.X, node.Y + 1)
            if d == MiniGridRoomsDirection.RIGHT:
                next_coord = node.X + 1, node.Y
            if d == MiniGridRoomsDirection.DOWN:
                next_coord = node.X, node.Y - 1
            if d == MiniGridRoomsDirection.LEFT:
                next_coord = node.X - 1, node.Y
            if next_coord in self._admissible_coordinate:
                return ((dict(X=next_coord[0], Y=next_coord[1], Dir=d), 1.0),)
            return ((asdict(node), 1.0),)

    def _calculate_reward_distribution(
        self,
        node: NodeGridRooms,
        action: IntEnum,
        next_node: NodeGridRooms,
    ) -> rv_continuous:
        return (
            self.optimal_distribution
            if next_node.X == self.goal_position[0]
            and next_node.Y == self.goal_position[1]
            else self.other_distribution
        )

    def _check_input_parameters(self):
        super(MiniGridRoomsMDP, self)._check_input_parameters()

        assert self.n_rooms >= 4, "There should be at least 4 rooms"
        assert self.room_size >= 2, "The room size must be at least 2"
        assert int(np.sqrt(self.n_rooms)) == np.sqrt(
            self.n_rooms
        ), "Please provide a number of rooms with perfect square."

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

    def _instantiate_starting_node_sampler(self) -> NextStateSampler:
        rooms_per_row = int(np.sqrt(self.n_rooms))
        rooms = list(product(range(rooms_per_row), range(rooms_per_row)))
        # noinspection PyAttributeOutsideInit

        corner_rooms = list(product((0, int(self.n_rooms ** 0.5) - 1), repeat=2))
        sr = self._fast_rng.randint(0, len(corner_rooms) - 1)
        self.starting_room = corner_rooms[sr]
        corner_rooms.pop(sr)
        self.goal_room = corner_rooms[self._fast_rng.randint(0, len(corner_rooms) - 1)]
        assert self.goal_room != self.starting_room

        # Random goal position from a random room
        goal_positions = (
            self.get_positions_coords_in_room(self.room_size, self.goal_room)
            .ravel()
            .tolist()
        )
        self._rng.shuffle(goal_positions)
        # noinspection PyAttributeOutsideInit
        self.goal_position = goal_positions[0]

        # Random starting position from a random room
        starting_nodes = [
            NodeGridRooms(x, y, MiniGridRoomsDirection(d))
            for x, y in self.get_positions_coords_in_room(
                self.room_size, self.starting_room
            )
            .ravel()
            .tolist()
            for d in range(4)
        ]
        self._rng.shuffle(starting_nodes)
        self._possible_starting_nodes = starting_nodes

        return NextStateSampler(
            next_states=self._possible_starting_nodes[: self.n_starting_states],
            probs=[1 / self.n_starting_states for _ in range(self.n_starting_states)],
            seed=self._next_seed(),
        )

    def calc_grid_repr(self, node) -> np.array:
        rooms_per_row = int(np.sqrt(self.n_rooms))
        door_positions = [
            int(self.room_size // 2) + i * (self.room_size + 1) + 1
            for i in range(rooms_per_row)
        ]
        grid_size = rooms_per_row * self.room_size + rooms_per_row - 1
        grid = np.zeros((grid_size, grid_size), dtype=str)

        for x in range(1, grid_size + 1):
            for y in range(1, grid_size + 1):
                if (
                    x != 0
                    and x != (grid_size)
                    and x % (self.room_size + 1) == 0
                    and not y in door_positions
                ):
                    grid[y - 1, x - 1] = "W"
                    continue
                elif (
                    y != 0
                    and y != (grid_size)
                    and y % (self.room_size + 1) == 0
                    and not x in door_positions
                ):
                    grid[y - 1, x - 1] = "W"
                    continue
                else:
                    grid[y - 1, x - 1] = " "

        grid[self.goal_position[1], self.goal_position[0]] = "G"

        if self.cur_node.Dir == MiniGridRoomsDirection.UP:
            grid[self.cur_node.Y, self.cur_node.X] = "^"
        elif self.cur_node.Dir == MiniGridRoomsDirection.RIGHT:
            grid[self.cur_node.Y, self.cur_node.X] = ">"
        elif self.cur_node.Dir == MiniGridRoomsDirection.DOWN:
            grid[self.cur_node.Y, self.cur_node.X] = "v"
        elif self.cur_node.Dir == MiniGridRoomsDirection.LEFT:
            grid[self.cur_node.Y, self.cur_node.X] = "<"

        return grid[::-1, :]
