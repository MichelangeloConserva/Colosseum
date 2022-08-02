import os

from colosseum.agent.agents.episodic.posterior_sampling import PSRLEpisodic
from colosseum.agent.agents.episodic.q_learning import QLearningEpisodic

EPISODIC_AGENT_CLASSES = [QLearningEpisodic, PSRLEpisodic]

files = os.listdir(__file__[: __file__.rfind(os.sep)])
n_agents = len(list(filter(lambda x: ".py" in x, files)))
assert (
    n_agents == len(EPISODIC_AGENT_CLASSES) + 1
), f"The EPISODIC_AGENT_CLASSES list in {__file__} is incomplete."
