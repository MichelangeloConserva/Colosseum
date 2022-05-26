## Diameter
The diameter was proposed as a measure of hardness for MDPs by [1].
Intuitively, the diameter of an MDP is the maximum number of time steps necessary to go from s to s' when following
the best policy for that purpose.

Although the diameter of weakly-communicating MDPs should be infinite, we assume that the only interesting states are
in the recurrent class, so we calculate the diameter in the recurrent class rather than setting it to infinity.

The diameter is always finite in the episodic setting (where every state is reachable within an episode).

## Value-norm
The value-norm was proposed by [2].
Note that each policy has a different distribution norm so, as suggested in the paper, we use the distribution-norm
of the optimal policy as a measure of hardness.

The distribution-norm is null for fully deterministic MDPs.

<br>

[ [1](https://papers.nips.cc/paper/2014/file/2ab56412b1163ee131e1246da0955bd1-Paper.pdf) ] Maillard, Odalric-Ambrym, Timothy A. Mann, and Shie Mannor. "How hard is my MDP?" The distribution-norm to the rescue"." Advances in Neural Information Processing Systems 27 (2014): 1835-1843.
<br>[ [2](http://www.jmlr.org/papers/volume11/jaksch10a/jaksch10a.pdf) ] Jaksch, Ortner, and Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research, 11:1563–1600, 2010.
