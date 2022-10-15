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
# Benchmark analysis


`````{margin}
````{dropdown} Necessary imports
```{code-block} python
from colosseum.analysis.plots import plot_indicator_in_hardness_space, agent_performances_per_mdp_plot
from colosseum.analysis.tables import get_latex_table_of_average_indicator, get_latex_table_of_indicators
```
````
`````
```{code-cell}
:tags: [remove-input, remove-output]
import os
import shutil
from myst_nb import glue

from colosseum.analysis.plots import plot_indicator_in_hardness_space, agent_performances_per_mdp_plot
from colosseum.analysis.tables import get_latex_table_of_average_indicator, get_latex_table_of_indicators

benchmark_log_folder = "./experiments" + os.sep + "tabular" + os.sep + "benchmarking" + os.sep + "paper_results" + os.sep + "benchmark_continuous_communicating"
```

This tutorial explains how to analyse the results of {{col}} benchmarking procedures.
The running example is the result for tabular agents in the continuous communicating setting presented in the accompanying {{paper}}.

## Visualization tools

Two types of visualizations are available to visualize agents' performances, tables and plots.
The tables aim to summarise the information whereas plots provide more detailed.
Note that $\LaTeX$ code for the tables is automatically generated.

### Tables

````{margin}
```{important} 
Remember to load the $\texttt{booktabs}$, $\texttt{colortbl}$, and  $\texttt{xcolor}$ packages to compile the $\LaTeX$ tables.
```
````

#### _Summary table_
The [`get_latex_table_of_average_indicator`](../pdoc_files/colosseum/analysis/tables.html#get_latex_table_of_average_indicator)
function produces a table that summarises the agents' performances in terms of a single indicator.

```{code-cell}
# The tex variable contains a LaTex ready version, whereas pd_table is a Pandas table
tex, pd_table = get_latex_table_of_average_indicator(
    benchmark_log_folder,
    "normalized_cumulative_regret",
    print_table=True,
    return_table=True,
)
```
The $\LaTeX$ summary table is provided below.
```{code-cell}
:tags: [hide-output, remove-input]
print(tex)
```


#### Indicators table
The [`get_latex_table_of_indicators`](../pdoc_files/colosseum/analysis/tables.html#get_latex_table_of_indicators)
function produces a large table that can include multiple indicators. It also reports the the number of seeds that the agent was able to complete in the given training time limit.

```{code-cell}
# The tex variable contains a LaTex ready version
tex = get_latex_table_of_indicators(
    benchmark_log_folder,
    ["normalized_cumulative_regret", "steps_per_second"],
    show_prm_mdp=True,
    print_table=True,
)
```
The $\LaTeX$ indicators table is provided below.
```{code-cell}
:tags: [hide-output, remove-input]
print(tex)
```

### Plots

#### _Hardness space_ plot
The [`plot_indicator_in_hardness_space`](../pdoc_files/colosseum/analysis/plots.html#plot_indicator_in_hardness_space) 
function produces a plot that places the average cumulative regret obtained by each agent in the benchmark MDPs in the 
position corresponding to the diameter and environmental value norm of the corresponding MDP.
This plot enables investigating which kind of complexity impacts the performance of the agents most, and thus it helps gain a better understanding of their strengths and weaknesses.


```{code-cell}
:tags: [remove-output]
fig = plot_indicator_in_hardness_space(benchmark_log_folder, fontsize=24, savefig_folder = None)
```

```{code-cell}
:tags: [remove-output, remove-input]
glue("hardness_space", fig, display=False)
```
````{div} full-width
```{glue:} hardness_space
```
````


#### Online agents' performances
The [`agent_performances_per_mdp_plot`](../pdoc_files/colosseum/analysis/plots.html#agent_performances_per_mdp_plot) function
produces a plot that shows the values for a given indicator during the agent/MDP interactions.
This plot enables easily comparing agents' performances in the benchmark MDPs and provides an intuitive overview of the critical moments of the agent/MDP interaction, e.g., when the agent runs out of time or reaches the optimal policy.

```{code-cell}
:tags: [remove-output]
fig = agent_performances_per_mdp_plot(
    benchmark_log_folder,
    "normalized_cumulative_regret",
    figsize_scale=5,
    standard_error=True,
    n_rows=7,
    savefig_folder = None
)
```

```{code-cell}
:tags: [remove-output, remove-input]
glue("performances_per_mdp_plot", fig, display=False)
```

`````{div} full-width
```{glue:} performances_per_mdp_plot
```
`````

```{code-cell}
:tags: [remove-input, remove-output]
from colosseum import config
shutil.rmtree("tmp", ignore_errors=True)
shutil.rmtree(config.get_hardness_measures_cache_folder(), ignore_errors=True)
```
