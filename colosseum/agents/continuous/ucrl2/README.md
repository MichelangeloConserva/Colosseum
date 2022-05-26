The *UCRL2* agent for infinite horizon with undiscounted rewards MDPs is taken from [1] and heavily inspired by the code
available at [github.com/RonanFR/UCRL](https://github.com/RonanFR/UCRL).
[1] prove a high probability regret upper bound of `O(DS\sqrt{AT})` ignoring logarithmic terms for any communicating
MDP with unknown but finite diameter.
`D` is the diameter, `S` is the number of states, `A` is the number of action and `T` is the optimization time horizon.

[ [1](http://www.jmlr.org/papers/volume11/jaksch10a/jaksch10a.pdf) ] Jaksch, Ortner, and Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research, 11:1563–1600, 2010.
