# Motivation

{{Rl}} has attracted significant interest in recent years after striking performances obtained in board games
{cite}`silver2018general` and video games {cite}`vinyals2019grandmaster, berner2019dota`.
Solving these grand challenges constitutes an important milestone in the field.
However, the corresponding agents require efficient simulators due to their high **sample complexity**[^sc].
Outside of games, many important applications, e.g. healthcare, can also be naturally formulated as {{rl}} problems. 
However, simulators for these scenarios may **not** be available, reliable, or efficient.

The development of {{rl}} methods that explore efficiently has long been considered (one of) the most crucial efforts to reduce sample complexity.
Meticulously evaluating the strengths and weaknesses of such methods is essential to assess progress and inspire new developments in the field.
Such empirical evaluations must be performed using _benchmarks_ composed of a selection of environments and _evaluation criteria_.

{{col}} is the brainchild of the aspiration of its authors to develop a rigorous benchmarking methodology for {{rl}} algorithms.
We argue that the environment selection should be based on theoretically principled reasoning that considers the _hardness_ of the environments and the soundness of the evaluation criteria.

In non-tabular {{rl}}, there is no theory of hardness except for a few restricted settings.
Consequently, the selection of environments in current benchmarks {cite}`osband2020bsuite, rajan2021mdp` relies solely on the experience of their authors.
Although such benchmarks are certainly valuable, there is no guarantee that they contain a sufficiently diverse range of environments and that they are effectively able to quantify the agents' capabilities.
In contrast, in tabular {{rl}}, a rich theory of hardness of environments is available.
{{col}} leverages such theory to develop a principle benchmarking procedure.
Accordingly, the environments are selected to maximize diversity with respect to two important measures of hardness, providing a varied set of challenges for which a _precise_ characterization of hardness is available, and the evaluation criterion is the exact cumulative regret, which {{col}} efficiently computes.

Further details can be found in the accompanying {{paper}}.

[^sc]: the number of observations that they require to optimize a reward-based criterion in an unknown environment.

```{bibliography}
:filter: docname in docnames
```