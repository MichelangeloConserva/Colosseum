The _Taxi_ MDP is a grid world where the agent has to pick up and drop off passengers [1].
Each time a passenger is taken to the correct location a new passenger and destination appears.
The agent has six actions available, which correspond to the cardinal directions, picking a passenger,
and dropping off a passenger.  The agent receives a large reward when it drops off a passenger in the
correct destination and zero reward when it tries to drop off a passenger in an incorrect destination.
At every other time step, it receives a small reward.

[ [1](https://arxiv.org/pdf/cs/9905014.pdf) ] Dietterich, Thomas G. "Hierarchical reinforcement learning with the MAXQ value function decomposition." Journal of artificial intelligence research 13 (2000): 227-303.
