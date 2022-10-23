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
Have a look at the {doc}`Motivation <./mds/manifesto>` section for a brief overview of the motivation behind the
{{col}} project.

<h4> Getting Started </h4>

- [Installation](./mds/installation)
- [Quickstart](./mds/quickstart)
- **Tutorials**:
  - [Configure {{col}}](./mds/colosseum-configurations.md)
  - [Inspect Agents Performances](./mds/agent-performance-analysis.md)
  - [Inspect Markov Decision Processes](./mds/mdp-functionalities)
  - [Visualize Markov Decision Processes](./mds/mdp-visual-representations)
  - [The {{col}} Benchmark](./mds/benchmark-introduction.md)
  - [Analyse Benchmarking Results](./mds/benchmark-analysis.md)
  - [Create Custom Benchmarks](./mds/benchmark-custom.md)
  - [Benchmarking Agents](./mds/benchmark-running.md)
  - [Hyperparameters Optimization](./mds/hyperopt.md)
  - [Scale Benchmarking to a Cluster](./mds/benchmark-running.md)
  - [Empirical Hardness Analysis](./mds/hardness-analysis.md)
  - [Non-Tabular Benchmarking](./mds/non-tabular.md)
- [API Documentation](./mds/api-reference.md)
- [Contributions](./mds/contributions.md)
- [Discord channel](https://discord.gg/JBEezJgxGY)

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
