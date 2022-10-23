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
# Analyse Benchmarking Results

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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import os
import shutil
from myst_nb import glue

from colosseum.analysis.plots import plot_indicator_in_hardness_space, agent_performances_per_mdp_plot
from colosseum.analysis.tables import get_latex_table_of_average_indicator, get_latex_table_of_indicators

benchmark_log_folder = "./experiments" + os.sep + "tabular" + os.sep + "benchmarking" + os.sep + "paper_results" + os.sep + "benchmark_continuous_communicating"
```

We'll reproduce the analysis of the benchmark results of the tabular agents in the continuous communicating setting presented in the accompanying {{paper}}.

<h3> Visualization tools </h3>

Two types of visualizations are available, tables and plots.
Note that $\LaTeX$ code for the tables is automatically generated.

<h4> Tables </h4>

````{margin}
```{important} 
The _booktabs_, _colortbl_, and  _xcolor_ packages are required to compile the $\LaTeX$ tables.
```
````

**Summary table**

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

**Indicators table**

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


<h4> Plots </h4>

**Hardness space plot**

The [`plot_indicator_in_hardness_space`](../pdoc_files/colosseum/analysis/plots.html#plot_indicator_in_hardness_space) 
function produces a plot that places the average cumulative regret obtained by each agent in the benchmark MDPs in the 
position corresponding to the diameter and environmental value norm of the corresponding MDP.
This plot enables investigating which kind of complexity impacts the performance of the agents most.


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


**Online agents' performances**

The [`agent_performances_per_mdp_plot`](../pdoc_files/colosseum/analysis/plots.html#agent_performances_per_mdp_plot) function
produces a plot that shows the values for a given indicator during the agent/MDP interactions.
This plot enables easily comparing agents' performances in the benchmark MDPs and provides an intuitive overview of the critical moments of the agent/MDP interaction, e.g., when an agent runs out of time or reaches the optimal policy.

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
