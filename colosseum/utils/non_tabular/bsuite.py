import abc
from typing import TYPE_CHECKING

import dm_env
from bsuite.baselines.base import Agent as BAgent

from colosseum.agent.agents.base import BaseAgent
from colosseum.emission_maps import EmissionMap
from colosseum.utils.acme.specs import MDPSpec

if TYPE_CHECKING:
    from colosseum.mdp import ACTION_TYPE


class NonTabularBsuiteAgentWrapper(BaseAgent, abc.ABC):
    """
    A simple wrapper for `bsuite` agents.
    """

    @staticmethod
    def is_emission_map_accepted(emission_map: "EmissionMap") -> bool:
        return not emission_map.is_tabular

    def is_episode_end(
        self,
        ts_t: dm_env.TimeStep,
        a_t: "ACTION_TYPE",
        ts_tp1: dm_env.TimeStep,
        time: int,
    ) -> bool:
        return False

    def __init__(
        self,
        seed: int,
        agent: BAgent,
        mdp_specs: MDPSpec,
    ):
        self._agent = agent
        self._mdp_spec = mdp_specs
        self.emission_map = mdp_specs.emission_map

        super(NonTabularBsuiteAgentWrapper, self).__init__(
            seed, mdp_specs, None, None, None
        )

    def select_action(self, ts: dm_env.TimeStep, time: int) -> "ACTION_TYPE":
        return self._agent.select_action(ts)

    def step_update(
        self, ts_t: dm_env.TimeStep, a_t: "ACTION_TYPE", ts_tp1: dm_env.TimeStep, h: int
    ):
        self._agent.update(ts_t, a_t, ts_tp1)

    def update_models(self):
        pass

    def _before_new_episode(self):
        pass

    def episode_end_update(self):
        pass

    def before_start_interacting(self):
        pass
