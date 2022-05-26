The *posterior sampling for reinforcement learning* agent for episodic MDPs is taken from [1]
and inspired by the code available at [github.com/iosband/TabulaRL](https://github.com/iosband/TabulaRL).
A high probability regret upper bound of `O(HS\sqrt{AT})` ignoring logarithmic terms is given in [1].
`H` is the episode length, `S` is the number of states, `A` is the number of action and `T` is the optimization time horizon.

[ [1](https://proceedings.neurips.cc/paper/2013/file/6a5889bb0190d0211a991f47bb19a777-Paper.pdf) ] Osband, Ian, Daniel Russo, and Benjamin Van Roy. "(More) efficient reinforcement learning via posterior sampling." arXiv preprint arXiv:1306.0940 (2013).