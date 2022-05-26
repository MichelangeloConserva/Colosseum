The _DeepSea_ MDP has been introduced by [1] as a _deep exploration_ challenge.
The MDP is a pyramid of states in which the agent starts at the top and dives a step down at each time step.
Depending on the action chosen, the agent can either dive to the right or to the left.
Once the agent reaches the base of the pyramid, it restarts from the top.
The agent is rewarded when diving to the left, but a large reward can be obtained by reaching the bottom rightmost state.
The main difficulty of _DeepSea_ is that the agent will not be able to reach the highly rewarding state before it restarts if it chooses a single _wrong_ action.
We removed the p_lazy parameter from _DeepSea_ due to the particular structure of this MDP.
In the episodic setting, staying in the same state even once would make reaching the goal state impossible.

[ [1](https://arxiv.org/pdf/1703.07608.pdf) ] Osband, Ian, et al. "Deep Exploration via Randomized Value Functions." J. Mach. Learn. Res. 20.124 (2019): 1-62.