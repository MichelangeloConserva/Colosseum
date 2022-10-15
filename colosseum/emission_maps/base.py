import abc
import importlib
import re
from typing import TYPE_CHECKING, Dict, Any, Type, Tuple

import numpy as np

from colosseum import config

if TYPE_CHECKING:
    from colosseum.mdp import NODE_TYPE
    from colosseum.mdp.base import BaseMDP
    from colosseum.noises.base import Noise


class EmissionMap(abc.ABC):
    """
    The base class to define emission maps that transform tabular MDPs into non-tabular versions.
    """

    @property
    @abc.abstractmethod
    def is_tabular(self) -> bool:
        """
        Returns
        -------
        bool
            The boolean for whether the emission map is tabular.
        """

    @abc.abstractmethod
    def node_to_observation(
        self, node: "NODE_TYPE", in_episode_time: int = None
    ) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            The non-tabular representation corresponding to the state given in input. Episodic MDPs also requires the
            current in-episode time step.
        """

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        Tuple[int, ...]
            The shape of the non-tabular representation.
        """
        if self._shape is None:
            self._shape = self.node_to_observation(self._mdp.starting_nodes[0], 0).shape
        return self._shape

    @property
    def all_observations(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            The numpy array containing all the non-tabular representation for the states in the MDP. In the episodic
            case, it is episode length by number of state by number of action. In the continuous case, the dimension for
            the episode is dropped.
        """
        if self._observations is None:
            if self._mdp.is_episodic():
                self._observations = np.empty(
                    (self._mdp.H, self._mdp.n_states, *self.shape), np.float32
                )
            else:
                self._observations = np.empty(
                    (self._mdp.n_states, *self.shape), np.float32
                )

            for i, n in enumerate(self._mdp.G.nodes):
                if self._mdp.is_episodic():
                    for h in range(self._mdp.H):
                        self._observations[h, i] = self.node_to_observation(n, h)
                else:
                    self._observations[i] = self.node_to_observation(n, None)
        return self._observations

    def __init__(
        self,
        mdp: "BaseMDP",
        noise_class: Type["Noise"],
        noise_kwargs: Dict[str, Any],
    ):
        """
        Parameters
        ----------
        mdp : BaseMDP
            The tabular MDP.
        noise_class : Type["Noise"]
            The noise that renders the emission map stochastic.
        noise_kwargs : Dict[str, Any]
            The parameters for the noise class.
        """

        self._mdp = mdp
        self._cached_obs = dict()
        self._observations = None
        self._shape = None

        if noise_class is not None:
            self._noise_map = noise_class(shape_f=lambda: self.shape, **noise_kwargs)
        else:
            self._noise_map = None

    def get_observation(
        self, state: "NODE_TYPE", in_episode_time: int = None
    ) -> np.ndarray:
        """
        computes the observation numpy array corresponding to the state in input.

        Parameters
        ----------
        state : NODE_TYPE
            The state for which we are computing the observation.
        in_episode_time : int
            The in episode time. It is ignored in the continuous setting, and, by default, it is set to None.

        Returns
        -------
        np.ndarray
            The observation.
        """

        if self._mdp.is_episodic():
            if in_episode_time is None:
                in_episode_time = self._mdp.h
            if in_episode_time >= self._mdp.H:
                return np.zeros(self.shape, np.float32)
        if not self._mdp.is_episodic():
            in_episode_time = None
        obs = self.all_observations[in_episode_time, self._mdp.node_to_index[state]]
        if self._noise_map is not None:
            return obs + next(self._noise_map)
        return obs


class StateLinear(EmissionMap, abc.ABC):
    """
    The base class for the emission map such that the non-tabular representation is a vector for which the value function
    of a given policy is linear.
    """

    def __init__(
        self,
        mdp: "BaseMDP",
        noise_class: Type["Noise"],
        noise_kwargs: Dict[str, Any],
        d: int = None,
    ):
        """
        Parameters
        ----------
        mdp : BaseMDP
            The tabular MDP.
        noise_class : Type["Noise"]
            The noise that renders the emission map stochastic.
        noise_kwargs : Dict[str, Any]
            The parameters for the noise class.
        d : int
            The dimensionality of the non-tabular representation vector.
        """

        self._features = None
        self._d = (
            max(config.get_min_linear_feature_dim(), int(mdp.n_states * 0.1))
            if d is None
            else d
        )

        super(StateLinear, self).__init__(mdp, noise_class, noise_kwargs)

    @property
    def is_tabular(self) -> bool:
        return False

    @property
    @abc.abstractmethod
    def V(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            The value function w.r.t. which the non-tabular representation is linear.
        """

    def _sample_features(self):
        self._features = _sample_linear_value_features(
            self.V, self._d, self._mdp.H if self._mdp.is_episodic() else None
        ).astype(np.float32)

    def node_to_observation(
        self, node: "NODE_TYPE", in_episode_time: int = None
    ) -> np.ndarray:
        if self._features is None:
            self._sample_features()
        if in_episode_time is not None and self._mdp.is_episodic():
            return self._features[in_episode_time, self._mdp.node_to_index[node]]
        return self._features[self._mdp.node_to_index[node]]


def get_emission_map_from_name(emission_map_name: str) -> Type[EmissionMap]:
    """
    Returns
    -------
    EmissionMap
        The EmissionMap class corresponding to the name of the emission map given in input.
    """
    return importlib.import_module(
        f"colosseum.emission_maps.{re.sub(r'(?<!^)(?=[A-Z])', '_', emission_map_name).lower()}"
    ).__getattribute__(emission_map_name)


def _sample_linear_value_features(v: np.ndarray, d: int, H: int = None):
    psi = np.random.randn(v.size, d)
    psi[:, 0] = 1
    psi[:, 1] = v

    P = psi @ np.linalg.inv(psi.T @ psi) @ psi.T

    W = np.random.randn(v.size, d)
    W[:, 0] = 1

    W_p = P @ W
    features = W_p / np.linalg.norm(W_p, axis=0, keepdims=True)
    if H is not None:
        features = features.reshape(H + 1, -1, d)
    return features


def _get_symbol_mapping(mdp: "BaseMDP") -> Dict[str, int]:
    symbols = mdp.get_unique_symbols()
    return dict(zip(symbols, range(len(symbols))))
