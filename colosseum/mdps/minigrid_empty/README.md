Gym MiniGrid (MG) [1] is an important testbed for non-tabular reinforcement learning agents.
It presents several families of MDPs that produce different challenges.
The base structure is a grid world where an agent can move by going forward, rotating left, and rotating right.
Depending on the MDP family, the agent can have further access to actions such a pick, drop, and interact.
The goal is always to reach a highly rewarding state.

The _MG-Empty_ MDP contains only the basic structure of an MG environment.
The starting states are selected from a randomly selected border of the grid and the highly rewarding state is located on the border at the opposite side of the grid.
Although, _MG-Empty_ appears similar to \texttt{SimpleGrid}, it implements a more complex mechanism to move between states that results in a completely different transition structure.

[ [1](https://github.com/maximecb/gym-minigrid) ] Chevalier-Boisvert, M., Willems, L., and Pal, S. (2018). Minimalistic gridworld environment
for OpenAI gym. https://github.com/maximecb/gym-minigrid.