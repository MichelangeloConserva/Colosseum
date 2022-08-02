from typing import TYPE_CHECKING

import dm_env

from colosseum.agent.actors import BaseActor
from colosseum.utils.acme.specs import DiscreteArray, MDPSpec

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE, OBSERVATION_TYPE


class RandomActor(BaseActor):
    def __init__(self, seed: int, environment_spec: MDPSpec, cache_size=50_000):
        super(RandomActor, self).__init__(seed, environment_spec)

        self._cached_actions = []
        self._cache_size = cache_size

    def _fill_cache(self):
        """
        creates a cache of random actions.
        """
        if type(self._environment_spec.actions) == DiscreteArray:
            self._cached_actions = self._rng.randint(
                0, self._environment_spec.actions.num_values, self._cache_size
            ).tolist()
        else:
            raise NotImplementedError(
                "The random actor has been implemented only for discrete action spaces."
            )

    def select_action(
        self, ts: dm_env.TimeStep, time_step: int
    ) -> "ACTION_TYPE":
        if len(self._cached_actions) == 0:
            self._fill_cache()
        return self._cached_actions.pop(0)
