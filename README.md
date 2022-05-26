# Colosseum

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)


Colosseum is a pioneering Python package that creates a bridge between theory and practice in tabular reinforcement
learning.
Its two main functionalities are MDPs hardness investigation and running principled benchmarks.

### Hardness analysis
Colosseum provides practical tools to investigate hardness in a theoretically sound way.
More specifically, the core capabilities of the package are the following:

- The computation of three theoretical measures of hardness, the _diameter_ [1], the _sum of the reciprocals of the sub-optimality
gaps_ [2] and the _environmental value norm_ [3].
- The first efficient implementations of the algorithms to identify the communication class of MDPs [4].
- The computation of the exact expected regret in the continuous and episodic settings.
- Visual representations of the MDPs, such as state/state-action pairs visitation counts and state/state-action pairs value functions.

### Principled benchmarks
Colosseum implements principled benchmarks for the four most widely studied tabular reinforcement learning settings,
_episodic ergodic_, _episodic communicating_, _continuous ergodic_ and _continuous communicating_.
The environments were selected to maximize diversity with respect to two important measures of hardness,
the *diameter* and the *environmental value norm*, thus providing a varied set of challenges for which a 
*precise* characterization of hardness is available. 

### Acknowledgements
We are grateful for the extraordinary tools developed by the open-source Python community.
We particularly thank the authors of Jupyter Notebook, Matplotlib, Numpy, Pandas, Scipy, NetworkX, Seaborn, and Numba.

## References
[ [1](http://www.jmlr.org/papers/volume11/jaksch10a/jaksch10a.pdf) ] Jaksch, Ortner, and Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research, 11:1563–1600, 2010.
<br>[ [2](https://proceedings.neurips.cc/paper/2019/file/10a5ab2db37feedfdeaab192ead4ac0e-Paper.pdf) ] Simchowitz, Max, and Kevin G. Jamieson. "Non-asymptotic gap-dependent regret bounds for tabular mdps." Advances in Neural Information Processing Systems 32 (2019).
<br>[ [3](https://papers.nips.cc/paper/2014/file/2ab56412b1163ee131e1246da0955bd1-Paper.pdf) ] Maillard, Odalric-Ambrym, Timothy A. Mann, and Shie Mannor. "How hard is my MDP?" The distribution-norm to the rescue"." Advances in Neural Information Processing Systems 27 (2014): 1835-1843.
<br>[ [4](https://link.springer.com/chapter/10.1007/978-1-4613-0265-0_9) ] Kallenberg, L. C. M. "Classification problems in MDPs." Markov processes and controlled Markov chains. Springer, Boston, MA, 2002. 151-165.


## Example gallery

The code to produce the following plots is available in the examples folder.

### Hardness analysis

<p>
  <img src="imgs/ha.svg"  style="width:25%" title="hover text">
</p>

### Agent MDP interaction

<p>
  <img src="imgs/regret.svg"  style="width:50%" title="hover text">
</p>

### MDP visual representations

<p>
  <img src="imgs/deep_sea.svg"  style="width:20%" title="hover text">
  <img src="imgs/frozen_lake.svg"  style="width:20%" title="hover text">
  <img src="imgs/mge.svg"  style="width:20%" title="hover text">
  <img src="imgs/mgr.svg"  style="width:20%" title="hover text">
</p>

### Markov chain visual representations

<p>
  <img src="imgs/deep_sea_mc.svg"  style="width:20%" title="hover text">
  <img src="imgs/frozen_lake_mc.svg"  style="width:20%" title="hover text">
  <img src="imgs/mge_mc.svg"  style="width:20%" title="hover text">
  <img src="imgs/mgr_mc.svg"  style="width:20%" title="hover text">
</p>

### Visitation counts

<p >
  <img src="imgs/deep_sea_vc.svg"  style="width:20%" title="hover text">
  <img src="imgs/frozen_lake_vc.svg"  style="width:20%" title="hover text">
  <img src="imgs/deep_sea_vc2.svg"  style="width:20%" title="hover text">
  <img src="imgs/frozen_lake_vc2.svg"  style="width:20%" title="hover text">
</p>

<p >
  <img src="imgs/mge_vc.svg"  style="width:20%" title="hover text">
  <img src="imgs/mgr_vc.svg"  style="width:20%" title="hover text">
  <img src="imgs/mge_vc2.svg"  style="width:20%" title="hover text">
  <img src="imgs/mgr_vc2.svg"  style="width:20%" title="hover text">
</p>