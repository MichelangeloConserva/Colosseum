"""
The module containing the MDP classes.
"""

from typing import Union

import numpy as np
from colosseum.mdp.base import BaseMDP
from colosseum.mdp.base_finite import EpisodicMDP
from colosseum.mdp.base_infinite import ContinuousMDP
from colosseum.mdp.deep_sea.base import DeepSeaNode as _DeepSeaNode
from colosseum.mdp.frozen_lake.base import FrozenLakeNode as _FrozenLakeNode
from colosseum.mdp.minigrid_empty.base import MiniGridEmptyNode as _MiniGridEmptyNode
from colosseum.mdp.minigrid_rooms.base import MiniGridRoomsNode as _NodeGridRooms
from colosseum.mdp.river_swim.base import RiverSwimNode as _RiverSwimNode
from colosseum.mdp.simple_grid.base import SimpleGridNode as _SimpleGridNode
from colosseum.mdp.taxi.base import TaxiNode as _TaxiNode
from colosseum.mdp.custom_mdp import CustomNode as _CustomNode

OBSERVATION_TYPE = Union[int, np.ndarray]
ACTION_TYPE = Union[int, float, np.ndarray]
REWARD_TYPE = Union[int, float, np.ndarray]

NODE_TYPE = Union[
    _CustomNode,
    _RiverSwimNode,
    _DeepSeaNode,
    _FrozenLakeNode,
    _SimpleGridNode,
    _MiniGridEmptyNode,
    _NodeGridRooms,
    _TaxiNode,
]
