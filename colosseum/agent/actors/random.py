from typing import TYPE_CHECKING

import dm_env

from colosseum.agent.actors import BaseActor
from colosseum.utils.acme.specs import DiscreteArray, MDPSpec

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE


class RandomActor(BaseActor):
    """
    The `RandomActor` component acts uniformly randomly.
    """

    def __init__(self, seed: int, mdp_specs: MDPSpec, cache_size=50_000):
        """
        Parameters
        ----------
        seed : int
            The random seed.
        mdp_specs : MDPSpec
            The full specification of the MDP.
        cache_size : int
            The cache size for the randomly sampled actions.
        """

        super(RandomActor, self).__init__(seed, mdp_specs)

        self._cached_actions = []
        self._cache_size = cache_size

    def _fill_cache(self):
        if type(self._mdp_spec.actions) == DiscreteArray:
            self._cached_actions = self._rng.randint(
                0, self._mdp_spec.actions.num_values, self._cache_size
            ).tolist()
        else:
            raise NotImplementedError(
                "The random actor has been implemented only for discrete action spaces."
            )

    def select_action(self, ts: dm_env.TimeStep, time: int) -> "ACTION_TYPE":
        if len(self._cached_actions) == 0:
            self._fill_cache()
        return self._cached_actions.pop(0)
