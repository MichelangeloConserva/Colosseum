---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# What is Colosseum?

{{col}} is a pioneering Python package that creates a bridge between theory and practice in tabular reinforcement learning with an eye on the non-tabular setting.

If this is your first time on the {{col}} documentation, or you have not read the accompanying {{paper}}, please have a look at the {doc}`Motivation <introduction/manifesto>` section.
The available tutorials are the best way to get started.
Although we reccomend to go through the tutorials in order, each tutorial is self-contained.
Have a look at the [_Tutorials overview_ section](./tutorials/introduction) for further details.

**Core capabilities**  
- The computation of three theoretical measures of hardness for any given MDP.
- Empirical study of the properties of hardness measures.
- Principled benchmarking for tabular algorithms with rigorous hyperparameters optimization.
- Non-tabular versions of the tabular benchmark for which tabular hardness measures can be computed.
- Extensive visualizations for MDPs and analysis tools for the agents' performances.

[//]: # (- Efficient implementations to identify the communication class of an MDP {cite}`kallenberg2002classification`.)


If you use {{col}} in your research, please cite the accompanying {{paper}}.

``` {code-block} bibtex
@inproceedings{conserva2022hardness,
  title={Hardness in Markov Decision Processes: Theory and Practice},
  author={Conserva, Michelangelo and Rauber, Paulo},
  year={2022},
  booktitle={Advances in Neural Information Processing Systems},
}
```

**Acknowledgements**  
The authors would like to thank the open-source Python community for the fundamental tools this research has been built upon.
In particular, the authors would like to thank the authors of 
$\texttt{dm_env}$,
Gin Config,
Jupyter Notebook, 
Matplotlib, 
NetworkX, 
Numba,
Numpy, 
OpenAi Gym,
Pandas, 
Scipy, 
Seaborn,
TensorFlow, and
tqdm.
