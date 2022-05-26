The _RiverSwim_ MDP has been introduced by [1] as a simple but challenging MDP.
This MDP is a chain of states where the agent can only move between adjacent states.
The agent starts in the leftmost state.
We have removed the _current_ mechanism proposed by [1],
which increases the difficulty of moving right, since we provide more general controls (namely, p_lazy and p_rand).
The agent is given a small reward for staying in the initial state, but it can obtain a large reward in
the rightmost state. The challenge of _RiverSwim_ is that the agent has to travel all the states in the 
chain in order to discover the highly rewarding state.


[ [1](https://www.sciencedirect.com/science/article/pii/S0022000008000767) ] A. L. Strehl and M. L. Littman. An analysis of model-based interval estimation for markov
decision processes. Journal of Computer and System Sciences, 74(8):1309–1331, 2008.