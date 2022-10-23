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
# Create Custom Benchmarks

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

In addition to the default benchmark, it is possible to create custom benchmarks.

To create a custom benchmark, we instantiate a
[`ColosseumBenchmark`](../pdoc_files/colosseum/benchmark/benchmark.html#ColosseumBenchmark) object, which requires
the parameters for the environments that will constitute the benchmark and
the settings that regulates the agent/MDP interactions, which are stored in
an [`ExperimentConfig`](../pdoc_files/colosseum/experiment/config.html#ExperimentConfig) object.

We define a configuration that results in a small number of short agent/MDP interactions.
```{code-cell}
:tags: [remove-output]
experiment_config = ExperimentConfig(
    n_seeds=1,
    n_steps=5_000,
    max_interaction_time_s=30,
    log_performance_indicators_every=1000,
)
```

<h4> MDP configurations </h4>

There are three ways to create environments configurations that can be used to create a custom benchmark.

<h5> Random sampling </h5>

Each {{col}} environment class implements a function to randomly sample parameters that are mainly used for the hyperparameters optimization procedure of the agents.
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

<h5> Default benchmark instances </h5>

We can also borrow the MDP instances from the default benchmark, and maybe modify them.

```{code-cell}
# Instantiate the episodic ergodic benchmark and take its MDP configurations
mdps_configs = ColosseumDefaultBenchmark.EPISODIC_ERGODIC.get_benchmark().mdps_gin_configs

# Save the configurations in a new ColosseumBenchmark object with a custom name and the previously defined experiment config
benchmark = ColosseumBenchmark("borrowing_from_default", mdps_configs, experiment_config)
```


<h5> Configurations from MDP instances </h5>

Finally, we can obtain environment configurations directly from instances.

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
