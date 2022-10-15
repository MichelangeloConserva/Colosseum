"""
The MDP model components of reinforcement learning agents contains the knowledge of the regarding the MDPs.
Their main role is to provide estimates about quantities related to the MDP that can be used by a `BaseActor` to select
actions.
"""

from colosseum.agent.mdp_models.bayesian_model import BayesianMDPModel

MODEL_TYPES = [BayesianMDPModel]
