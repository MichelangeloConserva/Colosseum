import random
from typing import TYPE_CHECKING, Iterable, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from colosseum.mdp import NODE_TYPE


class NextStateSampler:
    """
    The `NextStateSampler` handles the sampling of states.
    """

    @property
    def next_nodes_and_probs(self) -> Iterable[Tuple["NODE_TYPE", float]]:
        """
        Returns
        -------
        Iterable[Tuple["NODE_TYPE", float]]
            The iterable over next state and probability pairs.
        """
        return zip(self.next_nodes, self.probs)

    def __init__(
        self, next_nodes: List["NODE_TYPE"], seed: int = None, probs: List[float] = None
    ):
        """
        Parameters
        ----------
        next_nodes : List["NODE_TYPE"]
            The next nodes from which the sampler will sample.
        seed : int
            The random seed.
        probs : List[float]
            The probabilities corresponding to each state.
        """
        assert len(next_nodes) > 0
        self.next_nodes = next_nodes
        self._probs = dict()

        # Deterministic sampler
        if len(next_nodes) == 1:
            assert probs is None or len(probs) == 1
            self.next_state = next_nodes[0]
            self.probs = [1.0]
            self.is_deterministic = True
        # Stochastic sampler
        else:
            assert seed is not None
            self.probs = probs
            self._rng = random.Random(seed)
            self.n = len(next_nodes)
            self.is_deterministic = False
            self.cached_states = self._rng.choices(
                self.next_nodes, weights=self.probs, k=5000
            )

    def sample(self) -> "NODE_TYPE":
        """
        Returns
        -------
        NODE_TYPE
            A sample of the next state distribution.
        """
        if self.is_deterministic:
            return self.next_state
        if len(self.cached_states) == 0:
            self.cached_states = self._rng.choices(
                self.next_nodes, weights=self.probs, k=5000
            )
        return self.cached_states.pop(0)

    def mode(self) -> "NODE_TYPE":
        """
        Returns
        -------
        NODE_TYPE
            The most probable next state.
        """
        if self.is_deterministic:
            return self.next_state
        return self.next_nodes[np.argmax(self.probs)]

    def prob(self, n: "NODE_TYPE") -> float:
        """
        Returns
        -------
        float
            The probability of sampling the given state.
        """
        if n not in self._probs:
            if n not in self.next_nodes:
                self._probs[n] = 0.0
            else:
                self._probs[n] = self.probs[self.next_nodes.index(n)]
        return self._probs[n]
