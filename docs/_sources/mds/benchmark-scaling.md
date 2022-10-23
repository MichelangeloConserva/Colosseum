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
# Scale Benchmarking to a Cluster

`````{margin}
````{dropdown} Necessary imports
```{code-block} python
import os

from colosseum import config
from colosseum.agent.agents.episodic import QLearningEpisodic
from colosseum.agent.agents.infinite_horizon import QLearningContinuous
from colosseum.benchmark import ColosseumDefaultBenchmark
from colosseum.benchmark.run import instantiate_and_get_exp_instances_from_agents_and_benchmarks_for_hyperopt
from colosseum.benchmark.run import instantiate_and_get_exp_instances_from_agents_and_benchmarks
from colosseum.experiment.experiment_instances import save_instances_to_folder
from colosseum.hyperopt import SMALL_HYPEROPT_CONF
from colosseum.hyperopt.selection import retrieve_best_agent_config_from_hp_folder
from colosseum.hyperopt.utils import sample_agent_configs_and_benchmarks_for_hyperopt

# Set an experiment name that briefly describes the aim of the experiments
experiments_folder = "experiments" + os.sep + "tabular"
experiment_name = "tutorial"

exp_instances_hpo_folder = config.get_hyperopt_folder() + "experiment_instances"

config.set_experiments_folder(experiments_folder, experiment_name)
config.set_hyperopt_folder(experiments_folder, experiment_name)
```
````
`````
```{code-cell}
:tags: [remove-output, remove-input]
import os

from colosseum import config
from colosseum.agent.agents.episodic import QLearningEpisodic
from colosseum.agent.agents.infinite_horizon import QLearningContinuous
from colosseum.benchmark import ColosseumDefaultBenchmark
from colosseum.benchmark.run import instantiate_and_get_exp_instances_from_agents_and_benchmarks_for_hyperopt
from colosseum.benchmark.run import instantiate_and_get_exp_instances_from_agents_and_benchmarks
from colosseum.experiment.experiment_instances import save_instances_to_folder
from colosseum.hyperopt import SMALL_HYPEROPT_CONF
from colosseum.hyperopt.selection import retrieve_best_agent_config_from_hp_folder
from colosseum.hyperopt.utils import sample_agent_configs_and_benchmarks_for_hyperopt

agent_cls = [QLearningContinuous, QLearningEpisodic]

# Set an experiment name that briefly describes the aim of the experiments
experiments_folder = "experiments" + os.sep + "tabular"
experiment_name = "tutorial"

exp_instances_hpo_folder = config.get_hyperopt_folder() + "experiment_instances"
exp_instances_bench_folder = config.get_experiments_folder() + "experiment_instances"

config.set_experiments_folder(experiments_folder, experiment_name)
config.set_hyperopt_folder(experiments_folder, experiment_name)
```


Scaling up {{col}} benchmarking to run on a cluster is particularly straightforward.
Each agent/MDP interaction configuration can be stored as a [`ExperimentInstance`](../pdoc_files/colosseum/experiment/experiment_instance.html#ExperimentInstance) object, which can be easily pickled, uploaded to the cluster server, and run.
To properly execute the instances, it is also necessary to upload the benchmark folders containing the gin configurations to the cluster as shown below.

**Cluster jobs**  
The main task of the cluster jobs is to run the experiment instances using the following functions.

- [`run_experiment_instance`](../pdoc_files/colosseum/experiment/experiment_instances.html#run_experiment_instance) takes as input a ``ExperimentInstance`` object or a string containing a path to a file of a pickled ``ExperimentInstance``object, and runs the corresponding agent/MDP interaction.
- [`run_experiment_instances`](../pdoc_files/colosseum/experiment/experiment_instances.html#run_experiment_instances) takes as input a list of ``ExperimentInstance`` or a list of strings containing paths to the pickled ``ExperimentInstance`` objects. This function allows to group and to run multiples experiment instances using a single core or multiple cores depending on whether the multiprocessing is enabled or not.

<h4> Step 1: Hyperparameters optimization </h4>

```{code-block} python
# Assume we want to benchmark the following agent classes
agent_cls = [QLearningContinuous, QLearningEpisodic]

# Obtain the MDP configuration files and instantiate them locally
hyperopt_benchmarks = sample_agent_configs_and_benchmarks_for_hyperopt(agent_cls, SMALL_HYPEROPT_CONF)

# Create the corresponding ExperimentInstance objects
hp_exp_instances = instantiate_and_get_exp_instances_from_agents_and_benchmarks_for_hyperopt(
    hyperopt_benchmarks
)

# Pickle the experiment instances
exp_instance_paths = save_instances_to_folder(hp_exp_instances, exp_instances_hpo_folder)
```

We have now instantiated locally all the files we need to run the hyperparameters optimization procedure: the agents gin configurations, the MDPs gin configurations, and the pickled ExperimentInstances.
Note that, to simplify the entire procedure, it is important that the logging files are downloaded in the folders that were previously created when instantiating the agents and MDPs gin configuration files.

A suggestion of how to upload/download the necessary directories to the cluster server is reported below,
where `upload_folder` recursively copies a folder to the same path in the cluster and
`download_folder` recursively downloads a folder from the cluster to the same path in your local machine.

```{code-block} python
# Upload
for _, b in hyperopt_benchmarks:
    upload_folder(b.get_hyperopt_benchmark_log_folder())
upload_folder(exp_instances_hpo_folder)

# Let the cluster jobs run

# Download the results
for _, b in hyperopt_benchmarks:
    download_folder(cluster_ssh_path + b.get_hyperopt_benchmark_log_folder())
```

After the logging files have been downloaded and are available locally, we can proceed to the hyperparameters selection, which, by default, minimises the average normalized cumulative regret.
```{code-block} python
# Obtain the best hyperparameters given the performances stored in the loggings
agents_configs = retrieve_best_agent_config_from_hp_folder(agent_cls)
```

<h4> Step 2: Agents benchmarking </h4>

````{margin}
```{tip}
You can substitute the default benchmark with custom benchmarks here (see [Create Custom Bechmark tutorial](../mds/benchmark-custom.md)).
```
````

The first step of the {{col}} benchmarking procedure is completed, we now proceed to benchmark the best agent configurations on the default benchmark.

```{code-block} python
# Store the episodic and continuous agents configs separately.
agents_configs_episodic = {cl : agents_configs[cl] for cl in agents_configs if cl.is_episodic()}
agents_configs_continuous = {cl : agents_configs[cl] for cl in agents_configs if not cl.is_episodic()}

# Instantiate the benchmark for the different settings
b_cc = ColosseumDefaultBenchmark.CONTINUOUS_COMMUNICATING.get_benchmark()
b_ce = ColosseumDefaultBenchmark.CONTINUOUS_ERGODIC.get_benchmark()
b_ec = ColosseumDefaultBenchmark.EPISODIC_COMMUNICATING.get_benchmark()
b_ee = ColosseumDefaultBenchmark.EPISODIC_ERGODIC.get_benchmark()

# Prepare the input for the ExperimentInstance creator function
agents_and_benchmarks = [
    (agents_configs_continuous, b_cc),
    (agents_configs_continuous, b_ce),
    (agents_configs_episodic, b_ec),
    (agents_configs_episodic, b_ee),
]

# Instantiate the experiment instances (note the different function compared to the hyperoptimzation procedure)
experiment_instances = instantiate_and_get_exp_instances_from_agents_and_benchmarks(agents_and_benchmarks)
experiment_instances_paths = save_instances_to_folder(
    experiment_instances, exp_instances_bench_folder
)
```

Uploading and running the instances to the cluster should be done in the same way as we did for the hyperparameters optimization procedure.
Note the different function used to obtain the folder of the benchmark.
```{code-block} python
for _, b in agents_and_benchmarks:
    upload_folder(b.get_experiments_benchmark_log_folder())
upload_folder(exp_instances_bench_folder)
```

After downloading the results of the benchmarking procedure, you can proceed to analyse the results as explained in the
[Analyse Benchmarking Results tutorial](../mds/benchmark-analysis.md).
