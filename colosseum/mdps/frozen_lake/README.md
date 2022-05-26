The _FrozenLake_ MDP is a grid world where the agent has to walk over a frozen lake to reach a highly rewarding state.
Some tiles of the grid are walkable, whereas others represent holes in which the agent may fall (leading back to the starting point).
The agent can move in the cardinal directions. However, movement on the walkable tiles is not entirely deterministic, and so the agent risks falling into holes if it walks too close to them.
The agent receives a small reward at each time step and zero reward when it falls into a hole.
The starting position of the agent is the bottom left state, which is the one farthest away from the goal position.
The challenge presented by _FrozenLake_ is the high stochasticity of the movement.
A successful agent has to learn to balance the risk of falling into holes with reaching the goal quickly.

The Colosseum implementation of the frozen lake MDP has been inspired by the code available at [gym.openai.com/envs/FrozenLake-v0/](https://gym.openai.com/envs/FrozenLake-v0/).
