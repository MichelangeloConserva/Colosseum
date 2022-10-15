import abc
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


class MiniGridRoomsAction(IntEnum):
    """
    The actions available in the MiniGridRooms MDP.
    """

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
class MiniGridRoomsNode:
    """
    The node for the MiniGrid-Rooms MDP.
    """

    X: int
    """x coordinate."""
    Y: int
    """y coordinate."""
    Dir: MiniGridRoomsDirection
    """The direction the agent is facing."""

    def __str__(self):
        return f"X={self.X},Y={self.Y},Dir={MiniGridRoomsDirection(self.Dir).name}"


class MiniGridRoomsMDP(BaseMDP, abc.ABC):
    """
    The base class for the MiniGrid-Rooms family.
    """

    @staticmethod
    def get_unique_symbols() -> List[str]:
        return [" ", ">", "<", "v", "^", "G", "W"]

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
            n_rooms, room_size, _ = rng.dirichlet([0.2, 0.2, 1])
            n_rooms = min(9, (2 * n_rooms + 2).astype(int) ** 2)
            room_size = min(9, (7.0 * room_size + 3).astype(int))
            if is_episodic:
                room_size = max(room_size - 3, 3)
            sample = dict(
                room_size=room_size,
                n_rooms=n_rooms,
                n_starting_states=rng.randint(1, 5),
                p_rand=p_rand,
                p_lazy=p_lazy,
                make_reward_stochastic=rng.choice([True, False]),
                reward_variance_multiplier=2 * rng.random() + 0.005,
            )
            sample["p_rand"] = None if sample["p_rand"] < 0.01 else sample["p_rand"]
            sample["p_lazy"] = None if sample["p_lazy"] < 0.01 else sample["p_lazy"]

            if sample["make_reward_stochastic"]:
                size = int(sample["room_size"] * sample["n_rooms"] ** 0.5)
                sample["optimal_distribution"] = (
                    "beta",
                    (
                        sample["reward_variance_multiplier"],
                        sample["reward_variance_multiplier"] * (size ** 2 - 1),
                    ),
                )
                sample["other_distribution"] = (
                    "beta",
                    (
                        sample["reward_variance_multiplier"] * (size ** 2 - 1),
                        sample["reward_variance_multiplier"],
                    ),
                )
            else:
                sample["optimal_distribution"] = ("deterministic", (1.0,))
                sample["other_distribution"] = ("deterministic", (0.0,))

            samples.append(rounding_nested_structure(sample))
        return samples

    @staticmethod
    def get_node_class() -> Type["NODE_TYPE"]:
        return MiniGridRoomsNode

    def get_gin_parameters(self, index: int) -> str:
        prms = dict(
            room_size=self._room_size,
            n_rooms=self._n_rooms,
            n_starting_states=self._n_starting_states,
            make_reward_stochastic=self._make_reward_stochastic,
            reward_variance_multiplier=self._reward_variance_multiplier,
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
        if self._p_lazy is not None:
            prms["p_lazy"] = self._p_lazy

        return MiniGridRoomsMDP.produce_gin_file_from_mdp_parameters(
            prms, type(self).__name__, index
        )

    @property
    def n_actions(self) -> int:
        return len(MiniGridRoomsAction)

    @property
    def _admissible_coordinate(self) -> list:
        rooms_per_row = int(np.sqrt(self._n_rooms))

        vertical_checkers = [
            j * (self._room_size) + j + int(np.floor(self._room_size / 2))
            for j in range(rooms_per_row)
        ]
        horizontal_checkers = [
            j * self._room_size + j - 1 for j in range(1, rooms_per_row)
        ]
        door_positions = list(product(horizontal_checkers, vertical_checkers)) + list(
            product(vertical_checkers, horizontal_checkers)
        )
        rooms_coordinates = []
        for room_coord in product(range(rooms_per_row), range(rooms_per_row)):
            room = self.get_positions_coords_in_room(self._room_size, room_coord)
            for c in room.ravel().tolist():
                rooms_coordinates.append(tuple(c))
        return rooms_coordinates + door_positions

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

    def _get_next_nodes_parameters(
        self, node: "NODE_TYPE", action: "ACTION_TYPE"
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

    def _get_reward_distribution(
        self, node: "NODE_TYPE", action: "ACTION_TYPE", next_node: "NODE_TYPE"
    ) -> rv_continuous:
        return (
            self._optimal_distribution
            if next_node.X == self.goal_position[0]
            and next_node.Y == self.goal_position[1]
            else self._other_distribution
        )

    def _get_starting_node_sampler(self) -> NextStateSampler:
        rooms_per_row = int(np.sqrt(self._n_rooms))
        rooms = list(product(range(rooms_per_row), range(rooms_per_row)))

        corner_rooms = list(product((0, int(self._n_rooms ** 0.5) - 1), repeat=2))
        sr = self._fast_rng.randint(0, len(corner_rooms) - 1)
        self.starting_room = corner_rooms[sr]
        corner_rooms.pop(sr)
        self.goal_room = corner_rooms[self._fast_rng.randint(0, len(corner_rooms) - 1)]
        assert self.goal_room != self.starting_room

        # Random goal position from a random room
        goal_positions = (
            self.get_positions_coords_in_room(self._room_size, self.goal_room)
            .ravel()
            .tolist()
        )
        self._rng.shuffle(goal_positions)
        self.goal_position = goal_positions[0]

        # Random starting position from a random room
        starting_nodes = [
            MiniGridRoomsNode(x, y, MiniGridRoomsDirection(d))
            for x, y in self.get_positions_coords_in_room(
                self._room_size, self.starting_room
            )
            .ravel()
            .tolist()
            for d in range(4)
        ]
        self._rng.shuffle(starting_nodes)
        self.__possible_starting_nodes = starting_nodes

        return NextStateSampler(
            next_nodes=self._possible_starting_nodes[: self._n_starting_states],
            probs=[1 / self._n_starting_states for _ in range(self._n_starting_states)],
            seed=self._produce_random_seed(),
        )

    def _check_parameters_in_input(self):
        super(MiniGridRoomsMDP, self)._check_parameters_in_input()

        assert self._n_rooms >= 4, "There should be at least 4 rooms"
        assert self._room_size >= 2, "The room size must be at least 2"
        assert int(np.sqrt(self._n_rooms)) == np.sqrt(
            self._n_rooms
        ), "Please provide a number of rooms with perfect square."

        assert self._n_starting_states > 0

        dists = [
            self._optimal_distribution,
            self._other_distribution,
        ]
        check_distributions(
            dists,
            self._make_reward_stochastic,
        )

    def _get_grid_representation(self, node: "NODE_TYPE") -> np.ndarray:
        rooms_per_row = int(np.sqrt(self._n_rooms))
        door_positions = [
            int(self._room_size // 2) + i * (self._room_size + 1) + 1
            for i in range(rooms_per_row)
        ]
        grid_size = rooms_per_row * self._room_size + rooms_per_row - 1
        grid = np.zeros((grid_size, grid_size), dtype=str)

        for x in range(1, grid_size + 1):
            for y in range(1, grid_size + 1):
                if (
                    x != 0
                    and x != (grid_size)
                    and x % (self._room_size + 1) == 0
                    and not y in door_positions
                ):
                    grid[y - 1, x - 1] = "W"
                    continue
                elif (
                    y != 0
                    and y != (grid_size)
                    and y % (self._room_size + 1) == 0
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

    @property
    def _possible_starting_nodes(self) -> List["NODE_TYPE"]:
        return self.__possible_starting_nodes

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            **super(MiniGridRoomsMDP, self).parameters,
            **dict(
                room_size=self._room_size,
                n_rooms=self._n_rooms,
                n_starting_states=self._n_starting_states,
                optimal_distribution=self._optimal_distribution,
                other_distribution=self._other_distribution,
            ),
        }

    def __init__(
        self,
        seed: int,
        room_size: int,
        n_rooms: int = 4,
        n_starting_states: int = 2,
        optimal_distribution: Union[Tuple, rv_continuous] = None,
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
        room_size : int
            The size of the roorms.
        n_rooms : int
            The number of rooms. This must be a squared number.
        n_starting_states : int
            The number of possible starting states.
        optimal_distribution : Union[Tuple, rv_continuous]
            The distribution of the highly rewarding state. It can be either passed as a tuple containing Beta parameters
            or as a rv_continuous object.
        other_distribution : Union[Tuple, rv_continuous]
            The distribution of the other states. It can be either passed as a tuple containing Beta parameters or as a
            rv_continuous object.
        make_reward_stochastic : bool
            If True, the rewards of the MDP will be stochastic. By default, it is set to False.
        reward_variance_multiplier : float
            A constant that can be used to increase the variance of the reward distributions without changing their means.
            The lower the value, the higher the variance. By default, it is set to 1.
        """

        if type(optimal_distribution) == tuple:
            optimal_distribution = get_dist(
                optimal_distribution[0], optimal_distribution[1]
            )
        if type(other_distribution) == tuple:
            other_distribution = get_dist(other_distribution[0], other_distribution[1])

        self._n_starting_states = n_starting_states
        self._room_size = room_size
        self._n_rooms = n_rooms

        dists = [
            optimal_distribution,
            other_distribution,
        ]
        if dists.count(None) == 0:
            self._optimal_distribution = optimal_distribution
            self._other_distribution = other_distribution
        else:
            if make_reward_stochastic:
                size = int(room_size * n_rooms ** 0.5)
                self._other_distribution = beta(
                    reward_variance_multiplier, reward_variance_multiplier * (size ** 2 - 1)
                )
                self._optimal_distribution = beta(
                    reward_variance_multiplier * (size ** 2 - 1), reward_variance_multiplier
                )
            else:
                self._optimal_distribution = deterministic(1.0)
                self._other_distribution = deterministic(0.0)

        super(MiniGridRoomsMDP, self).__init__(
            seed=seed,
            reward_variance_multiplier=reward_variance_multiplier,
            make_reward_stochastic=make_reward_stochastic,
            **kwargs,
        )
