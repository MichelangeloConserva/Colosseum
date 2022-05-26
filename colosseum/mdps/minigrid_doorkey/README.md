Gym MiniGrid (MG) [1] is an important testbed for non-tabular reinforcement learning agents.
It presents several families of MDPs that produce different challenges.
The base structure is a grid world where an agent can move by going forward, rotating left, and rotating right.
Depending on the MDP family, the agent can have further access to actions such a pick, drop, and interact.
The goal is always to reach a highly rewarding state.

In the _MG-DoorKey_ environment, the grid world is divided into two rooms separated by a wall with a door
that can be opened using a key.
The key is positioned in the same room where the agent starts.
Differently from _Empty_ and _Rooms_, in this case, the agent has all the five actions available to
allow it to interact with the key and the door.
_MG-DoorKey_ is a particularly challenging MDP for several reasons.
First, the agent has to take a very long sequence of actions before reaching the highly rewarding state.
Further, the action that picks the key produces effect only in the very few states in which the agent is correctly
positioned in front of the key, and the action that opens the door only has effect in the single state in which
the agent has the key and is correctly positioned in front of the door.

[ [1](https://github.com/maximecb/gym-minigrid) ] Chevalier-Boisvert, M., Willems, L., and Pal, S. (2018). Minimalistic gridworld environment
for OpenAI gym. https://github.com/maximecb/gym-minigrid.