Gym MiniGrid (MG) [1] is an important testbed for non-tabular reinforcement learning agents.
It presents several families of MDPs that produce different challenges.
The base structure is a grid world where an agent can move by going forward, rotating left, and rotating right.
Depending on the MDP family, the agent can have further access to actions such a pick, drop, and interact.
The goal is always to reach a highly rewarding state.

The _MG-Rooms_ is a collection of grids connected with narrow passages.
The presence of such bottlenecks produces a significantly higher challenge for exploration when compared to open
grids, especially when p_rand is non-zero.

[ [1](https://github.com/maximecb/gym-minigrid) ] Chevalier-Boisvert, M., Willems, L., and Pal, S. (2018). Minimalistic gridworld environment
for OpenAI gym. https://github.com/maximecb/gym-minigrid.