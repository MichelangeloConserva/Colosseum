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
# Custom benchmarks

`````{margin}
````{dropdown} Necessary imports
```{code-block} python
from dataclasses import dataclass
from typing import Type

from colosseum.emission_maps import EmissionMap
from colosseum import config
from colosseum.agent.agents.episodic import PSRLEpisodic
from colosseum.agent.agents.infinite_horizon import QLearningContinuous
from colosseum.agent.utils import sample_agent_gin_configs
from colosseum.benchmark.benchmark import ColosseumBenchmark
from colosseum.experiment import ExperimentConfig
from colosseum.utils.miscellanea import sample_mdp_gin_configs
from colosseum.utils.miscellanea import get_colosseum_mdp_classes
from colosseum.benchmark.utils import get_mdps_configs_from_mdps
from colosseum.mdp.deep_sea import DeepSeaEpisodic
from colosseum.mdp.frozen_lake import FrozenLakeEpisodic
from colosseum.mdp.minigrid_empty import MiniGridEmptyContinuous
from colosseum.mdp.simple_grid import SimpleGridContinuous
from colosseum.agent.utils import sample_agent_gin_configs_file
from colosseum.utils.miscellanea import sample_mdp_gin_configs_file
from colosseum.benchmark import ColosseumDefaultBenchmark

# Configuring the directories for the package
experiment_folder = "tutorial"
experiment_name = "custom_benchmark"
config.set_experiments_folder(experiment_folder, experiment_name)
config.set_hyperopt_folder(experiment_folder, experiment_name)
seed = 42
```
````
`````
```{code-cell}
:tags: [remove-output, remove-input]
from dataclasses import dataclass
from typing import Type

from colosseum.emission_maps import EmissionMap
from colosseum import config
from colosseum.agent.agents.episodic import PSRLEpisodic
from colosseum.agent.agents.infinite_horizon import QLearningContinuous
from colosseum.agent.utils import sample_agent_gin_configs
from colosseum.benchmark.benchmark import ColosseumBenchmark
from colosseum.experiment import ExperimentConfig
from colosseum.utils.miscellanea import sample_mdp_gin_configs
from colosseum.utils.miscellanea import get_colosseum_mdp_classes
from colosseum.benchmark.utils import get_mdps_configs_from_mdps
from colosseum.mdp.deep_sea import DeepSeaEpisodic
from colosseum.mdp.frozen_lake import FrozenLakeEpisodic
from colosseum.mdp.minigrid_empty import MiniGridEmptyContinuous
from colosseum.mdp.simple_grid import SimpleGridContinuous
from colosseum.agent.utils import sample_agent_gin_configs_file
from colosseum.utils.miscellanea import sample_mdp_gin_configs_file
from colosseum.benchmark import ColosseumDefaultBenchmark

# Configuring the directories for the package
experiment_folder = "tutorial"
experiment_name = "custom_benchmark"
config.set_experiments_folder(experiment_folder, experiment_name)
config.set_hyperopt_folder(experiment_folder, experiment_name)
seed = 42
```


In addition to the default benchmark, it is possible to create custom benchmarks that still support all the analysis capabilities of the package.


## Benchmark experimental configuration

In order to create a custom benchmark, we need to instantiate a
[`ColosseumBenchmark`](../pdoc_files/colosseum/benchmark/benchmark.html#ColosseumBenchmark) object, which requires
an [`ExperimentConfig`](../pdoc_files/colosseum/experiment/config.html#ExperimentConfig) and the parameters for the
MDPs.

The `ExperimentConfig` regulates the agent/MDP interaction.


In order to keep the scale of the experiments small, we define a configuration that results in a small number of short agent/MDP interactions.

```{code-cell}
:tags: [remove-output]
experiment_config = ExperimentConfig(
    n_seeds=1,
    n_steps=5_000,
    max_interaction_time_s=30,
    log_performance_indicators_every=1000,
)
```

## MDP configurations

There are three ways to create MDP configurations that can be used to form a benchmark.

### Random sampling

Each {{col}} MDP class implements a function to randomly sample parameters, whose main function is for the agent hyperparameters optimization procedure (see the corresponding {doc}`tutorial <../tutorials/hyperopt>`).
Nonetheless, we can sample such configurations to create our custom benchmark.

```{code-cell}
# Get all the episodic MDP Colosseum classes
episodic_mdp_classes = get_colosseum_mdp_classes(episodic=True)

mdps_configs = dict()
for cl in episodic_mdp_classes:
    # For each episodic MDP class, we sample a single configuration
    mdps_configs[cl] = sample_mdp_gin_configs_file(cl, n=1, seed=seed)

# We define the benchmark object with the sampled MDP configs and the previously defined experiment config
benchmark = ColosseumBenchmark(
    name="episodic_randomly_sampled", mdps_gin_configs=mdps_configs, experiment_config=experiment_config
)
```

[//]: # (To get an idea of how a sampling procedure for an MDP looks like, you can have a look at the )

[//]: # ([`sample_mdp_parameters`]&#40;../pdoc_files/colosseum/mdp/deep_sea/base.html#DeepSeaMDP.sample_mdp_parameters&#41; function of the)

[//]: # (DeepSea family.)

### Default {{col}} benchmark instances

We can also borrow the MDP instances from the default benchmark, and maybe modify them.

```{code-cell}
# Instantiate the episodic ergodic benchmark and take its MDP configurations
mdps_configs = ColosseumDefaultBenchmark.EPISODIC_ERGODIC.get_benchmark().mdps_gin_configs

# Save the configurations in a new ColosseumBenchmark object with a custom name and the previously defined experiment config
benchmark = ColosseumBenchmark("borrowing_from_default", mdps_configs, experiment_config)
```


### Obtain configurations from MDP instances

Finally, we can obtain the configuration from MDP instances we have defined ourselves.
This can be particularly useful when the goal is to analyse a set of MDPs with particular characteristics, e.g. MDPs with small number of states but large visitation complexity.

```{code-cell}
# Define a list of MDP instance
mdps = [
    DeepSeaEpisodic(seed=4, size=10, p_rand=0.4),
    FrozenLakeEpisodic(seed=4, size=5, p_frozen=0.8),
]
# from which we can obtain the configurations from
mdps_configs = get_mdps_configs_from_mdps(mdps)

benchmark = ColosseumBenchmark("custom_mdp_instances", mdps_configs, experiment_config)
```


```{code-cell}
:tags: [remove-input, remove-output]
import shutil
shutil.rmtree(config.get_experiments_folder())
```
