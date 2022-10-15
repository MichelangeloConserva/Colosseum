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
# Hyperparameters optimization


`````{margin}
````{dropdown} Necessary imports
```{code-block} python
from dataclasses import asdict

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from myst_nb import glue
from scipy.stats import beta

from colosseum import config
from colosseum.agent.agents.episodic import QLearningEpisodic
from colosseum.agent.agents.infinite_horizon import QLearningContinuous
from colosseum.benchmark.run import instantiate_and_get_exp_instances_from_agents_and_benchmarks_for_hyperopt
from colosseum.experiment.experiment_instances import run_experiment_instances
from colosseum.hyperopt import DEFAULT_HYPEROPT_CONF, DEFAULT_HYPEROPT_CONF_NONTABULAR, SMALL_HYPEROPT_CONF, \
    SMALL_HYPEROPT_CONF_NONTABULAR, HyperOptConfig
from colosseum.hyperopt.selection import get_optimal_agents_configs
from colosseum.hyperopt.utils import sample_agent_configs_and_benchmarks_for_hyperopt
from colosseum.mdp.custom_mdp import CustomEpisodic
from colosseum.mdp.river_swim import RiverSwimEpisodic
from colosseum.mdp.simple_grid import SimpleGridContinuous

experiments_folder = "tutorial"
experiment_name = "bench_run"

config.set_experiments_folder(experiments_folder, experiment_name)
config.set_hyperopt_folder(experiments_folder, experiment_name)

config.enable_verbose_logging()

seed = 42
```
````
`````
```{code-cell}
:tags: [remove-output, remove-input]
import shutil
from dataclasses import asdict

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from myst_nb import glue
from scipy.stats import beta

from colosseum import config
from colosseum.agent.agents.episodic import QLearningEpisodic
from colosseum.agent.agents.infinite_horizon import QLearningContinuous
from colosseum.benchmark.run import instantiate_and_get_exp_instances_from_agents_and_benchmarks_for_hyperopt
from colosseum.experiment.experiment_instances import run_experiment_instances
from colosseum.hyperopt import DEFAULT_HYPEROPT_CONF, DEFAULT_HYPEROPT_CONF_NONTABULAR, SMALL_HYPEROPT_CONF, \
    SMALL_HYPEROPT_CONF_NONTABULAR, HyperOptConfig
from colosseum.hyperopt.selection import retrieve_best_agent_config_from_hp_folder
from colosseum.hyperopt.utils import sample_agent_configs_and_benchmarks_for_hyperopt
from colosseum.mdp.custom_mdp import CustomEpisodic
from colosseum.mdp.river_swim import RiverSwimEpisodic
from colosseum.mdp.simple_grid import SimpleGridContinuous


def pretty_print(hpc):
  index = [
      "Default tabular",
      "Small tabular",
      "Default non-tabular",
      "Small non-tabular",
  ]
  columns = list(asdict(DEFAULT_HYPEROPT_CONF))
  data = [
      list(map(str, asdict(DEFAULT_HYPEROPT_CONF).values())),
      list(map(str, asdict(SMALL_HYPEROPT_CONF).values())),
      list(
          map(
              lambda x: x.__name__ if "emission_maps" in str(x) else str(x),
              asdict(DEFAULT_HYPEROPT_CONF_NONTABULAR).values(),
          )
      ),
      list(
          map(
              lambda x: x.__name__ if "emission_maps" in str(x) else str(x),
              asdict(SMALL_HYPEROPT_CONF_NONTABULAR).values(),
          )
      ),
  ]
  
  return pd.DataFrame(data, index, columns).T
  
experiments_folder = "tutorial"
experiment_name = "hyperopt"

config.set_hyperopt_folder(experiments_folder, experiment_name)
```

As the [_Agent implementation_ tutorial](../tutorials/agent-implementation.md) explains, agent classes must define a sampling space for its hyperparameters.
These sampling spaces are used to conduct a random search optimization procedure with the objective of minimizing the cumulative regret across a set of randomly sampled environments.
The random sampling procedure for environments is defined for each {{col}} MDP class, and aims to provide a varied set of MDPs of up to moderate scale.


## Hyperparameters optimization configurations

The [`HyperOptConfig`](../pdoc_files/colosseum/hyperopt/config.html#HyperOptConfig) class controls the parameters of the hyperparameter optimization procedure.
There are four default hyperparameters optimization configurations available in the package, which are reported in the table below.
[`DEFAULT_HYPEROPT_CONF`](../pdoc_files/colosseum/hyperopt/config.html#DEFAULT_HYPEROPT_CONF) is the default hyperparameters optimization configuration for tabular agents,
[`SMALL_HYPEROPT_CONF`](../pdoc_files/colosseum/hyperopt/config.html#SMALL_HYPEROPT_CONF) is a quick hyperparameters optimization configuration for tabular agents that can be used for quick testing,
[`DEFAULT_HYPEROPT_CONF_NONTABULAR`](../pdoc_files/colosseum/hyperopt/config.html#DEFAULT_HYPEROPT_CONF_NONTABULAR) is the default hyperparameters optimization configuration for non-tabular agents, and
[`SMALL_HYPEROPT_CONF_NONTABULAR`](../pdoc_files/colosseum/hyperopt/config.html#SMALL_HYPEROPT_CONF_NONTABULAR) is the default hyperparameters optimization configuration for non-tabular agents that can be used for quick testing.

```{code-cell}
:tags: [remove-input]
pretty_print(DEFAULT_HYPEROPT_CONF).style.set_table_attributes('style="font-size: 14px"')
```

## Hyperparameters optimization

Running the hyperparameters optimization procedure is very similar to running a benchmark.
The only difference is that the benchmarks are automatically sampled.

```{code-cell}
# Define a custom small scale hyperparameters optimization procedure
hpoc = HyperOptConfig(
    seed=42,
    n_timesteps=20_000,
    max_interaction_time_s=40,
    n_samples_agents=1,
    n_samples_mdps=1,
    log_every=500,
    n_seeds=1,
)

# Take the q-learning agents as running example
agent_cls = [QLearningContinuous, QLearningEpisodic]

# Create the benchmarks for the given agents classes and hyperparameters optimzation configuration
hyperopt_agents_and_benchmarks = sample_agent_configs_and_benchmarks_for_hyperopt(agent_cls, hpoc)

# Obtain the instances and run them locally
hp_exp_instances = instantiate_and_get_exp_instances_from_agents_and_benchmarks_for_hyperopt(
    hyperopt_agents_and_benchmarks
)
run_experiment_instances(hp_exp_instances)

# Compute the best hyperparameters, which, by default, minimize the average normalized cumulative regret
optimal_agent_configs = retrieve_best_agent_config_from_hp_folder()
```

```{code-cell}
:tags: [remove-input]
print(optimal_agent_configs[QLearningEpisodic])
print()
print(optimal_agent_configs[QLearningContinuous])
```

```{code-cell}
:tags: [remove-input, remove-output]
shutil.rmtree(config.get_hyperopt_folder())
```
