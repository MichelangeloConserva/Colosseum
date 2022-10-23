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

# Benchmarking Agents

`````{margin}
````{dropdown} Necessary imports
```{code-block} python
from colosseum import config
from colosseum.agent.agents.episodic import PSRLEpisodic, QLearningEpisodic
from colosseum.agent.agents.infinite_horizon import QLearningContinuous, UCRL2Continuous
from colosseum.agent.utils import sample_agent_gin_configs_file
from colosseum.benchmark.benchmark import ColosseumDefaultBenchmark
from colosseum.benchmark.run import instantiate_and_get_exp_instances_from_agents_and_benchmarks
from colosseum.experiment import ExperimentConfig
from colosseum.experiment.experiment_instances import run_experiment_instances

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
from colosseum import config
from colosseum.agent.agents.episodic import PSRLEpisodic, QLearningEpisodic
from colosseum.agent.agents.infinite_horizon import QLearningContinuous, UCRL2Continuous
from colosseum.agent.utils import sample_agent_gin_configs_file
from colosseum.benchmark.benchmark import ColosseumDefaultBenchmark
from colosseum.benchmark.run import instantiate_and_get_exp_instances_from_agents_and_benchmarks
from colosseum.experiment import ExperimentConfig
from colosseum.experiment.experiment_instances import run_experiment_instances

experiments_folder = "tutorial"
experiment_name = "bench_run"
config.set_experiments_folder(experiments_folder, experiment_name)
config.set_hyperopt_folder(experiments_folder, experiment_name)
config.enable_verbose_logging()

seed = 42
```

We'll shortly explain how to benchmark agents.

```{code-block} python
# Define a small scale experiment config
experiment_config = ExperimentConfig(
    n_seeds=1,
    n_steps=5_000,
    max_interaction_time_s=1 * 30,
    log_performance_indicators_every=1000,
)

# Take the default colosseum benchmark for the episodic ergodic and the continuous communicating settings
b_e = ColosseumDefaultBenchmark.EPISODIC_QUICK_TEST.get_benchmark()
b_e.experiment_config = experiment_config
b_c = ColosseumDefaultBenchmark.CONTINUOUS_QUICK_TEST.get_benchmark()
b_c.experiment_config = experiment_config

# Randomly sample some episodic agents
agents_configs_e = {
    PSRLEpisodic : sample_agent_gin_configs_file(PSRLEpisodic, n=1, seed=seed),
    QLearningEpisodic : sample_agent_gin_configs_file(QLearningEpisodic, n=1, seed=seed),
}

# Randomly sample some continuous agents
agents_configs_c = {
    QLearningContinuous : sample_agent_gin_configs_file(QLearningContinuous, n=1, seed=seed),
    UCRL2Continuous : sample_agent_gin_configs_file(UCRL2Continuous, n=1, seed=seed),
}

# Obtain the experiment instances for the agents configurations and the benchmark
agents_and_benchmarks = [
    (agents_configs_e, b_e),
    (agents_configs_c, b_c),
]
experiment_instances = instantiate_and_get_exp_instances_from_agents_and_benchmarks(agents_and_benchmarks)

# Run the experiment instances
# Note that if multiprocessing is enabled, Colosseum will take advantage of it
run_experiment_instances(experiment_instances)
```

```{code-cell}
:tags: [remove-input, remove-output]
import shutil
shutil.rmtree(config.get_experiments_folder())
```