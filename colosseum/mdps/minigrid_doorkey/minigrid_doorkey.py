from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import IntEnum

from colosseum.utils.random_vars import deterministic, get_dist

try:
    from functools import cached_property
except:
    from backports.cached_property import cached_property

from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
from scipy.stats import beta, rv_continuous

from colosseum.mdps import MDP
from colosseum.mdps.base_mdp import NextStateSampler
from colosseum.mdps.minigrid_rooms.continuous.mdp import MiniGridRoomsContinuous
from colosseum.utils.mdps import check_distributions


class MiniGridDoorKeyAction(IntEnum):
    """The action available in the MiniGridDoorKey MDP."""

    MoveForward = 0
    TurnRight = 1
    TurnLeft = 2
    PickObject = 3
    DropObject = 4
    UseObject = 5


class MiniGridDoorKeyDirection(IntEnum):
    """The possible agent direction in the MiniGridDoorKey MDP."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


@dataclass(frozen=True)
class MiniGridDoorKeyNode:
    X: int
    Y: int
    Dir: MiniGridDoorKeyDirection
    XKey: int
    YKey: int
    IsDoorOpened: bool

    def __str__(self):
        return f"X={self.X},Y={self.Y},Dir={MiniGridDoorKeyDirection(self.Dir).name},XKey={self.XKey},YKey={self.YKey},IsDoorOpened{self.IsDoorOpened}"


class MiniGridDoorKeyMDP(MDP):
    @staticmethod
    def testing_parameters() -> Dict[str, Tuple]:
        t_params = MDP.testing_parameters()
        t_params["size"] = (3, 5, 7)
        t_params["make_reward_stochastic"] = (True, False)
        t_params["n_starting_states"] = (1, 4)
        return t_params

    @staticmethod
    def get_node_class() -> Type[MiniGridDoorKeyNode]:
        return MiniGridDoorKeyNode

    def __init__(
        self,
        seed: int,
        size: int,
        randomize_actions: bool = True,
        lazy: float = None,
        make_reward_stochastic=False,
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
        lazy : float
            the probability of an action not producing any effect on the MDP.
        size : int
            the size of the grid.
        make_reward_stochastic : bool, optional
            checks whether the rewards are to be made stochastic. By default, it is set to False.
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
        self.size = size
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
                self.other_distribution = beta(1, size ** 2 - 1)
                self.optimal_distribution = beta(size ** 2 - 1, 1)
            else:
                self.optimal_distribution = deterministic(1.0)
                self.other_distribution = deterministic(0.0)

        super().__init__(
            seed=seed,
            randomize_actions=randomize_actions,
            lazy=lazy,
            **kwargs,
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            **super(MiniGridDoorKeyMDP, self).parameters,
            **dict(
                size=self.size,
                n_starting_states=self.n_starting_states,
                optimal_distribution=self.optimal_distribution,
                other_distribution=self.other_distribution,
            ),
        }

    @property
    def possible_starting_nodes(self) -> List[MiniGridDoorKeyNode]:
        return self._possible_starting_nodes

    @cached_property
    def coordinates_available(self):
        coords = (
            MiniGridRoomsContinuous.get_positions_coords_in_room(self.size, (0, 0))
            .ravel()
            .tolist()
        )
        for i in range(self.size):
            if self.is_wall_horizontal:
                coords.remove((i, self.wall_position))
            else:
                coords.remove((self.wall_position, i))
        return tuple(coords)

    @property
    def num_actions(self):
        return len(MiniGridDoorKeyAction)

    def _calculate_next_nodes_prms(
        self, node: MiniGridDoorKeyNode, action: int
    ) -> Tuple[Tuple[dict, float], ...]:
        newnode_prms = deepcopy(asdict(node))
        if action == MiniGridDoorKeyAction.TurnRight:
            newnode_prms["Dir"] = (node.Dir + 1) % 4
        if action == MiniGridDoorKeyAction.TurnLeft:
            newnode_prms["Dir"] = (node.Dir - 1) % 4
        if action == MiniGridDoorKeyAction.MoveForward:
            if node.Dir == MiniGridDoorKeyDirection.UP:
                next_coord = (node.X, node.Y + 1)
            if node.Dir == MiniGridDoorKeyDirection.RIGHT:
                next_coord = node.X + 1, node.Y
            if node.Dir == MiniGridDoorKeyDirection.DOWN:
                next_coord = node.X, node.Y - 1
            if node.Dir == MiniGridDoorKeyDirection.LEFT:
                next_coord = node.X - 1, node.Y
            if next_coord in self.coordinates_available or (
                node.IsDoorOpened and next_coord == self.door_position
            ):
                newnode_prms["X"], newnode_prms["Y"] = next_coord
        if action == MiniGridDoorKeyAction.PickObject:
            if node.X == node.XKey and node.Y == node.YKey:
                newnode_prms["XKey"] = newnode_prms["YKey"] = -1
        if node.XKey == -1 and not node.IsDoorOpened:
            if action == MiniGridDoorKeyAction.DropObject:
                newnode_prms["XKey"] = node.X
                newnode_prms["YKey"] = node.Y
            if action == MiniGridDoorKeyAction.UseObject:
                if node.Dir == MiniGridDoorKeyDirection.UP:
                    next_coord = (node.X, node.Y + 1)
                if node.Dir == MiniGridDoorKeyDirection.RIGHT:
                    next_coord = node.X + 1, node.Y
                if node.Dir == MiniGridDoorKeyDirection.DOWN:
                    next_coord = node.X, node.Y - 1
                if node.Dir == MiniGridDoorKeyDirection.LEFT:
                    next_coord = node.X - 1, node.Y
                if next_coord == self.door_position:
                    newnode_prms["IsDoorOpened"] = True
        return ((newnode_prms, 1.0),)

    def _calculate_reward_distribution(
        self, node: Any, action: IntEnum, next_node: Any
    ) -> rv_continuous:
        return (
            self.optimal_distribution
            if next_node.X == self.goal_position[0]
            and next_node.Y == self.goal_position[1]
            else self.other_distribution
        )

    def _check_input_parameters(self):
        super(MiniGridDoorKeyMDP, self)._check_input_parameters()

        assert self.size >= 3

        check_distributions(
            [
                self.optimal_distribution,
                self.other_distribution,
            ],
            self.make_reward_stochastic,
        )

    def _instantiate_starting_node_sampler(self) -> NextStateSampler:
        # noinspection PyAttributeOutsideInit
        self.wall_position = self._rng.randint(self.size - 2) + 1
        # noinspection PyAttributeOutsideInit
        self.is_wall_horizontal = self._rng.rand() > 0.5

        if self.is_wall_horizontal:
            self.door_position = self._rng.randint(self.size), self.wall_position
        else:
            self.door_position = self.wall_position, self._rng.randint(self.size)

        self.is_goal_before = self._rng.rand() > 0.5

        coords = MiniGridRoomsContinuous.get_positions_coords_in_room(self.size, (0, 0))

        goal_positions = []
        starting_positions = []
        for i, j in coords.ravel():
            if (
                i < self.wall_position
                if self.is_goal_before
                else i > self.wall_position
            ):
                goal_positions.append((j, i) if self.is_wall_horizontal else (i, j))
            elif (
                i > self.wall_position
                if self.is_goal_before
                else i < self.wall_position
            ):
                starting_positions.append((j, i) if self.is_wall_horizontal else (i, j))

        possible_starting_positions = deepcopy(starting_positions)

        self._rng.shuffle(goal_positions)
        self.goal_position = goal_positions[0]

        self._rng.shuffle(starting_positions)
        self.start_key_position = starting_positions.pop(0)

        starting_positions = [
            (x, y, dir)
            for x, y in starting_positions
            for dir in MiniGridDoorKeyDirection
        ]
        assert self.n_starting_states < len(starting_positions)
        self._possible_starting_nodes = [
            MiniGridDoorKeyNode(
                x,
                y,
                dir.value,
                *self.start_key_position,
                False,
            )
            for x, y, dir in starting_positions
        ]
        return NextStateSampler(
            next_states=self._possible_starting_nodes[: self.n_starting_states],
            probs=[1 / self.n_starting_states for _ in range(self.n_starting_states)],
            seed=self._next_seed(),
        )

    def calc_grid_repr(self, node: Any) -> np.array:
        grid_size = self.size
        door_position = self.door_position
        wall_position = self.wall_position
        is_wall_horizontal = self.is_wall_horizontal
        grid = np.zeros((grid_size, grid_size), dtype=str)
        grid[:, :] = " "

        grid[self.goal_position[1], self.goal_position[0]] = "G"

        if self.cur_node.XKey != -1:
            grid[self.cur_node.YKey, self.cur_node.XKey] = "K"

        for i in range(grid_size):
            if not is_wall_horizontal:
                grid[i, wall_position] = "W_en"
            else:
                grid[wall_position, i] = "W_en"
        grid[door_position[1], door_position[0]] = (
            "O" if self.cur_node.IsDoorOpened else "C"
        )

        if self.cur_node.Dir == MiniGridDoorKeyDirection.UP:
            grid[self.cur_node.Y, self.cur_node.X] = "^"
        elif self.cur_node.Dir == MiniGridDoorKeyDirection.RIGHT:
            grid[self.cur_node.Y, self.cur_node.X] = ">"
        elif self.cur_node.Dir == MiniGridDoorKeyDirection.DOWN:
            grid[self.cur_node.Y, self.cur_node.X] = "v"
        elif self.cur_node.Dir == MiniGridDoorKeyDirection.LEFT:
            grid[self.cur_node.Y, self.cur_node.X] = "<"

        return grid[::-1, :]
