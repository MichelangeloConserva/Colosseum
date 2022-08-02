import os

from colosseum.agent.agents.infinite_horizon.posterior_sampling import (
    PSRLContinuous,
)
from colosseum.agent.agents.infinite_horizon.q_learning import (
    QLearningContinuous,
)
from colosseum.agent.agents.infinite_horizon.ucrl2 import UCRL2Continuous

INFINITE_HORIZON_AGENT_CLASSES = [QLearningContinuous, PSRLContinuous, UCRL2Continuous]

files = os.listdir(__file__[: __file__.rfind(os.sep)])
n_agents = len(list(filter(lambda x: ".py" in x, files)))
assert (
    n_agents == len(INFINITE_HORIZON_AGENT_CLASSES) + 1
), f"The INFINITE_HORIZON_AGENT_CLASSES list in {__file__} is incomplete."
