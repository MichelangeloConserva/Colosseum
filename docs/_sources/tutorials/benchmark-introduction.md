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

# Colosseum benchmark

`````{margin}
````{dropdown} Necessary imports
```{code-block} python
import seedir

from colosseum.benchmark import ColosseumDefaultBenchmark
from colosseum.benchmark.utils import instantiate_benchmark_folder
```
````
`````
```{code-cell}
:tags: [remove-input, remove-output]
import os
import shutil
from glob import glob

import seedir

from colosseum.benchmark import ColosseumDefaultBenchmark
from colosseum.benchmark.utils import instantiate_benchmark_folder

def print_benchmark_configurations():
    bdir = glob("benchmark_*")[0]
    print("Default experiment configuration in the 'experiment_config.yml' files.")
    with open(bdir + os.sep + "experiment_config.yml") as f:
        print("".join(map(lambda x: "\t" + x, sorted(f.readlines(), key=len))))
```



This tutorial introduces the default {{col}} benchmark, which targets the four most widely studies setting of {{rl}}: episodic ergodic, episodic communicating, continuous ergodic, and continuous communicating.
Note that the continuous setting is also known as infinite horizon.

As explained in the {{paper}}, the environment selections is the result of the theoretical analysis presented in the paper, which has been validated through an empirical study of the behaviour of the hardness measures.
Concretely, for each of the aforementioned settings, $20$ environments have been selected to be as diverse as possible with respect to the diameter and the environmental value norm, which act as proxies for the visitation complexity and the estimation complexity.
The candidate environments have been sampled from a set of parameters such that their diameter is less than $100$ and the environmental value norm is less than $3.5$ to guarantee a sufficient challenge while limiting the scale of the environments.

````` {margin}
```` {admonition} MDP families APIs
<a href="../pdoc_files/colosseum/mdp/deep_sea/base.html">``DeepSea``</a>,
<a href="../pdoc_files/colosseum/mdp/frozen_lake/base.html">``FrozenLake``</a>,
<a href="../pdoc_files/colosseum/mdp/minigrid_empty/base.html">``MiniGridEmpty``</a>,
<a href="../pdoc_files/colosseum/mdp/minigrid_rooms/base.html">``MiniGridRooms``</a>,
<a href="../pdoc_files/colosseum/mdp/river_swim/base.html">``RiverSwim``</a>,
<a href="../pdoc_files/colosseum/mdp/simple_grid/base.html">``SimpleGrid``</a>, and
<a href="../pdoc_files/colosseum/mdp/taxi/base.html">``Taxi``</a>.
````
`````

## The {{col}} benchmark MDPs

The tables below report the environments in the benchmark along with their parameters.
We briefly describe the parameters that are common to all environment below, and we refer to the API of the agent classes for
specific parameters.

The `size` parameter controls the number of states,
the `make_reward_stochastic` parameter checks whether the rewards are stochastic,
the `p_lazy` parameter is the probability that an MDP stays in the same state instead of executing the action selected by an agent, and
the `p_rand` parameter is the probability that an MDP executes a random action instead of the action selected by an agent.


### Episodic ergodic

`````{div} full-width
````{tab-set}
```{tab-item} DeepSea
1. DeepSea(size=10, p_rand=0.4, make_reward_stochastic=False)
2. DeepSea(size=13, p_rand=0.3, make_reward_stochastic=True)
```
```{tab-item} FrozenLake
1. FrozenLake(size=4, make_reward_stochastic=True, p_lazy=0.03, p_frozen=0.98, p_rand=0.001)
```
```{tab-item} MiniGridEmpty
1. MiniGridEmpty(size=10, make_reward_stochastic=False, p_lazy=None, p_rand=0.05, n_starting_states=3)
2. MiniGridEmpty(size=10, make_reward_stochastic=True, p_lazy=None, p_rand=0.3, n_starting_states=3)
3. MiniGridEmpty(size=10, make_reward_stochastic=False, p_lazy=0.05, p_rand=0.2, n_starting_states=3)
4. MiniGridEmpty(size=8, make_reward_stochastic=False, p_lazy=None, p_rand=0.3, n_starting_states=3)
5. MiniGridEmpty(size=8, make_reward_stochastic=True, p_lazy=0.02, p_rand=0.4, n_starting_states=3)
6. MiniGridEmpty(size=6, make_reward_stochastic=True, p_lazy=0.1, p_rand=0.05, n_starting_states=3)
7. MiniGridEmpty(size=4, make_reward_stochastic=False, p_lazy=0.1, p_rand=0.4, n_starting_states=3)
```
```{tab-item} MiniGridRooms
1. MiniGridRooms(make_reward_stochastic=True, room_size=3, n_rooms=4, p_rand=0.001, p_lazy=None)
2. MiniGridRooms(make_reward_stochastic=True, room_size=3, n_rooms=9, p_rand=0.1, p_lazy=0.01)
3. MiniGridRooms(make_reward_stochastic=True, room_size=4, n_rooms=4, p_rand=0.1, p_lazy=0.01)
```
```{tab-item} RiverSwim
1. RiverSwim(make_reward_stochastic=True, size=5, p_lazy=0.1, p_rand=0.01, sub_optimal_distribution=("beta",(2.4, 24.0)), optimal_distribution=("beta",(0.01, 0.11)), other_distribution=("beta",(2.4, 249.0)))
2. RiverSwim(make_reward_stochastic=True, size=30, p_lazy=0.01, p_rand=0.01, sub_optimal_distribution=("beta",(14.9, 149.0)), optimal_distribution=("beta",(0.01, 0.11)), other_distribution=("beta",(14.9, 1499.0)))
```
```{tab-item} SimpleGrid
1. SimpleGrid(reward_type=3, size=10, make_reward_stochastic=True, p_lazy=0.2, p_rand=0.01)
2. SimpleGrid(reward_type=3, size=16, make_reward_stochastic=True, p_lazy=0.2, p_rand=0.2)
3. SimpleGrid(reward_type=3, size=16, make_reward_stochastic=False, p_lazy=0.01, p_rand=0.2)
```
```{tab-item} Taxi
1. Taxi(make_reward_stochastic=True, p_lazy=0.001, size=4, length=1, width=1, space=1, n_locations=3, p_rand=0.01, default_r=("beta",(0.8, 20.0)), successfully_delivery_r=("beta",(1.0, 0.1)), failure_delivery_r=("beta",(0.8, 50.0)))
2. Taxi(make_reward_stochastic=True, p_lazy=0.01, size=4, length=1, width=1, space=1, n_locations=3, p_rand=0.2, default_r=("beta",(0.8, 6.0)), successfully_delivery_r=("beta",(1.0, 0.1)), failure_delivery_r=("beta",(0.8, 40.0)))
```
````
`````

### Episodic communicating


`````{div} full-width
````{tab-set}
```{tab-item} DeepSea
1. DeepSea(size=5, p_rand=None, make_reward_stochastic=True)
2. DeepSea(size=25, p_rand=None, make_reward_stochastic=True)
```
```{tab-item} FrozenLake
1. FrozenLake(size=3, make_reward_stochastic=True, p_lazy=None, p_frozen=0.9, p_rand=None)
```
```{tab-item} MiniGridEmpty
1. MiniGridEmpty(size=6, make_reward_stochastic=True, p_lazy=None, p_rand=None, n_starting_states=3)
2. MiniGridEmpty(size=6, make_reward_stochastic=True, p_lazy=0.35, p_rand=None, n_starting_states=5)
3. MiniGridEmpty(size=6, make_reward_stochastic=True, p_lazy=0.25, p_rand=0.1, n_starting_states=3)
4. MiniGridEmpty(size=10, make_reward_stochastic=False, p_lazy=0.1, p_rand=None, n_starting_states=5)
5. MiniGridEmpty(size=10, make_reward_stochastic=False, p_lazy=0.15, p_rand=0.1, n_starting_states=3)
```
```{tab-item} MiniGridRooms
1. MiniGridRooms(make_reward_stochastic=True, room_size=4, n_rooms=4, p_rand=None, p_lazy=None)
2. MiniGridRooms(make_reward_stochastic=False, room_size=3, n_rooms=9, p_rand=None, p_lazy=0.05)
3. MiniGridRooms(make_reward_stochastic=False, room_size=3, n_rooms=9, p_rand=None, p_lazy=0.1)
4. MiniGridRooms(make_reward_stochastic=False, room_size=3, n_rooms=4, p_rand=None, p_lazy=0.15)
```
```{tab-item} RiverSwim
1. RiverSwim(make_reward_stochastic=True, size=25, p_lazy=0.05, p_rand=None, sub_optimal_distribution=("beta",(12.4, 124.0)), optimal_distribution=("beta",(0.01, 0.11)), other_distribution=("beta",(12.4, 1249.0)))
2. RiverSwim(make_reward_stochastic=False, size=40, p_lazy=None, p_rand=None, sub_optimal_distribution=None, optimal_distribution=None, other_distribution=None)
```
```{tab-item} SimpleGrid
1. SimpleGrid(size=10, make_reward_stochastic=True, p_lazy=0.5)
2. SimpleGrid(size=13, make_reward_stochastic=True, p_lazy=0.4)
3. SimpleGrid(size=13, make_reward_stochastic=False, p_lazy=None)
4. SimpleGrid(size=20, make_reward_stochastic=True, p_lazy=0.4)
```
```{tab-item} Taxi
1. Taxi(make_reward_stochastic=True, p_lazy=0.01, size=4, length=1, width=1, space=1, n_locations=3, p_rand=None, default_r=("beta",(0.7, 30.0)), successfully_delivery_r=("beta",(0.4, 0.1)), failure_delivery_r=("beta",(0.8, 50.0)))
2. Taxi(make_reward_stochastic=True, p_lazy=0.2, size=5, length=1, width=1, space=1, n_locations=3, p_rand=None, default_r=("beta",(0.7, 30.0)), successfully_delivery_r=("beta", (0.4, 0.1)), failure_delivery_r=("beta",(0.8, 50.0)))
```
````
`````


### Continuous ergodic


`````{div} full-width
````{tab-set}
```{tab-item} DeepSea
1. DeepSea(size=20, p_rand=0.1, make_reward_stochastic=False)
```
```{tab-item} FrozenLake
1. FrozenLake(size=5, make_reward_stochastic=True, p_lazy=0.01, p_frozen=0.95, p_rand=0.05)
```
```{tab-item} MiniGridEmpty
1. MiniGridEmpty(size=12, make_reward_stochastic=True, p_lazy=0.05, p_rand=0.495, n_starting_states=3)
2. MiniGridEmpty(size=12, make_reward_stochastic=True, p_lazy=0.1, p_rand=0.395, n_starting_states=3)
3. MiniGridEmpty(size=10, make_reward_stochastic=False, p_lazy=0.02, p_rand=0.7, n_starting_states=3)
4. MiniGridEmpty(size=14, make_reward_stochastic=True, p_lazy=0.02, p_rand=0.6, n_starting_states=3)
5. MiniGridEmpty(size=10, make_reward_stochastic=True, p_lazy=0.1, p_rand=0.21, n_starting_states=3, optimal_distribution=("beta",(1.0, 0.11)), other_distribution=("beta",(1.0, 4.0)))
6. MiniGridEmpty(size=14, make_reward_stochastic=True, p_lazy=0.02, p_rand=0.4, n_starting_states=3, optimal_distribution=("beta",(0.5, 0.11)), other_distribution=("beta",(1.5, 4.0)))
7. MiniGridEmpty(size=14, make_reward_stochastic=True, p_lazy=0.05, p_rand=0.31, n_starting_states=3, optimal_distribution=("beta",(1.0, 0.11)), other_distribution=("beta",(1.0, 4.0)))
8. MiniGridEmpty(size=14, make_reward_stochastic=True, p_lazy=0.1, p_rand=0.6, n_starting_states=3, optimal_distribution=("beta",(0.3, 0.11)), other_distribution=("beta",(2.0, 4.0)))
```
```{tab-item} MiniGridRooms
1. MiniGridRooms(make_reward_stochastic=True, room_size=3, n_rooms=16, p_rand=0.1, p_lazy=0.4)
2. MiniGridRooms(make_reward_stochastic=False, room_size=5, n_rooms=9, p_rand=0.3, p_lazy=0.4)
```
```{tab-item} RiverSwim
1. RiverSwim(make_reward_stochastic=False, size=30, p_lazy=0.1, p_rand=0.2)
2. RiverSwim(make_reward_stochastic=True, size=50, p_lazy=0.1, p_rand=0.1)
3. RiverSwim(make_reward_stochastic=True, size=80, p_lazy=0.001, p_rand=0.2)
4. RiverSwim(make_reward_stochastic=False, size=80, p_lazy=0.1, p_rand=0.01)
```
```{tab-item} SimpleGrid
1. SimpleGrid(size=15, make_reward_stochastic=True, p_lazy=0.4, p_rand=0.4)
2. SimpleGrid(size=10, make_reward_stochastic=False, p_lazy=0.2, sub_optimal_distribution=None, optimal_distribution=None, other_distribution=None, p_rand=0.01)
3. SimpleGrid(size=20, make_reward_stochastic=False, p_lazy=0.1, sub_optimal_distribution=None, optimal_distribution=None, other_distribution=None, p_rand=0.1)
```
```{tab-item} Taxi
1. Taxi(make_reward_stochastic=True, p_lazy=0.01, size=4, length=1, width=1, space=1, n_locations=3, p_rand=0.1, default_r=("beta",(0.8, 20.0)), successfully_delivery_r=("beta",(1.0, 0.1)), failure_delivery_r=("beta",(0.8, 50.0)))
```
````
`````


### Continuous communicating


`````{div} full-width
````{tab-set}
```{tab-item} DeepSea
1. DeepSea(size=40, p_rand=None, make_reward_stochastic=True)
2. DeepSea(size=40, p_rand=None, make_reward_stochastic=False)
3. DeepSea(size=35, p_rand=None, make_reward_stochastic=True)
```
```{tab-item} FrozenLake
1. FrozenLake(size=4, make_reward_stochastic=True, p_lazy=0.01, p_frozen=0.95, p_rand=None)
2. FrozenLake(size=5, make_reward_stochastic=True, p_lazy=0.35, p_frozen=0.9, p_rand=None)
```
```{tab-item} MiniGridEmpty
1. MiniGridEmpty(size=12, make_reward_stochastic=True, p_lazy=0.25, p_rand=None, n_starting_states=3, optimal_distribution=("beta",(1.0, 0.11)), other_distribution=("beta",(1.0, 4.0)))
2. MiniGridEmpty(size=12, make_reward_stochastic=False, p_lazy=0.3, p_rand=None, n_starting_states=3, optimal_distribution=None, other_distribution=None)
3. MiniGridEmpty(size=8, make_reward_stochastic=False, p_lazy=0.3, p_rand=None, n_starting_states=3, optimal_distribution=None, other_distribution=None)
4. MiniGridEmpty(size=8, make_reward_stochastic=True, p_lazy=0.7, p_rand=None, n_starting_states=3, optimal_distribution=("beta",(1.0, 0.11)), other_distribution=("beta",(1.0, 4.0)))
5. MiniGridEmpty(size=12, make_reward_stochastic=True, p_lazy=0.7, p_rand=None, n_starting_states=3, optimal_distribution=("beta",(1.0, 0.11)), other_distribution=("beta",(1.0, 4.0)))
```
```{tab-item} MiniGridRooms
1. MiniGridRooms(make_reward_stochastic=True, room_size=5, n_rooms=9, p_rand=None, p_lazy=0.3)
2. MiniGridRooms(make_reward_stochastic=True, room_size=3, n_rooms=9, p_rand=None, p_lazy=0.5)
3. MiniGridRooms(make_reward_stochastic=True, room_size=5, n_rooms=9, p_rand=None, p_lazy=0.5)
```
```{tab-item} RiverSwim
1. RiverSwim(make_reward_stochastic=True, size=25, p_lazy=0.1, p_rand=None)
2. RiverSwim(make_reward_stochastic=True, size=90, p_lazy=0.03, p_rand=None)
```
```{tab-item} SimpleGrid
1. SimpleGrid(size=15, make_reward_stochastic=True, p_lazy=0.2, p_rand=None, sub_optimal_distribution=("beta",(0.3, 49.0)), optimal_distribution=("beta",(2.0, 0.11)), other_distribution=("beta",(0.3, 4.0)))
2. SimpleGrid(size=16, make_reward_stochastic=False, p_lazy=0.065, p_rand=None, sub_optimal_distribution=None, optimal_distribution=None, other_distribution=None)
3. SimpleGrid(size=25, make_reward_stochastic=True, p_lazy=0.1, p_rand=None, sub_optimal_distribution=("beta",(0.3, 49.0)), optimal_distribution=("beta",(2.0, 0.11)), other_distribution=("beta",(0.3, 4.0)))
4. SimpleGrid(size=25, make_reward_stochastic=False, p_lazy=0.3, p_rand=None, sub_optimal_distribution=None, optimal_distribution=None, other_distribution=None)
```
```{tab-item} Taxi
1. Taxi(make_reward_stochastic=True, p_lazy=0.01, size=4, length=1, width=1, space=1, n_locations=3, p_rand=None, default_r=("beta",(0.7, 30.0)), successfully_delivery_r=("beta",(0.4, 0.1)), failure_delivery_r=("beta",(0.8, 50.0)))
```
````
`````

### {{col}} environments hardness

In order to show that the environments in the benchmark are effectively diverse according to the hardness measures,
{numref}`Figure %s<hardness_space>` places the benchmark environments according to their diameter and value norm.

```{figure} ../../images/space_of_hardness.svg
:name: hardness_space

Positions in measure of hardness space of the MDPs in the benchmark.
```

## Instantiate the {{col}} benchmark

A benchmark in {{col}} can be instantiated using the [`ColosseumBenchmark`](../pdoc_files/colosseum/benchmark/benchmark.html#ColosseumBenchmark)
class.
A `ColosseumBenchmark` object contains the parameters of the MDP and an [`ExperimentConfig`](../pdoc_files/colosseum/experiment/config.html#ExperimentConfig), which regulates the agent/MDP interactions.
The default benchmark can be accesses through the [`ColosseumDefaultBenchmark`](../pdoc_files/colosseum/benchmark/benchmark.html#ColosseumDefaultBenchmark)
enumeration, which also allow to retrieve the benchmark as shown below.

```{code-cell}
# Locally instantiate the episodic ergodic benchmark with folder name "benchmark_er"
instantiate_benchmark_folder(
    ColosseumDefaultBenchmark.EPISODIC_ERGODIC.get_benchmark(), "benchmark_er"
)

# Locally instantiate the episodic communicating benchmark with folder name "benchmark_ec"
instantiate_benchmark_folder(
    ColosseumDefaultBenchmark.EPISODIC_COMMUNICATING.get_benchmark(), "benchmark_ec"
)

# Locally instantiate the continuous ergodic benchmark with folder name "benchmark_ce"
instantiate_benchmark_folder(
    ColosseumDefaultBenchmark.CONTINUOUS_ERGODIC.get_benchmark(), "benchmark_ce"
)

# Locally instantiate the continuous communicating benchmark with folder name "benchmark_cc"
instantiate_benchmark_folder(
    ColosseumDefaultBenchmark.CONTINUOUS_COMMUNICATING.get_benchmark(), "benchmark_cc"
)

# Print the folder structure of the benchmark
for bdir in glob("benchmark_*"):
    seedir.seedir(bdir, style="emoji")
print("-" * 70)
# Print the benchmark configurations
print_benchmark_configurations()
```

The ```mdp_configs``` folder contains the Gin files with the configurations of the MDPs that were previously presented.  
The ```experiment_config.yml``` file contains the default `ExperimentConfig`, which prescribes to run the agent/MDP interaction for a total of $500\ 000$ time steps with a maximum interaction time of $10$ minutes for $20$ seeds.
The performance indicators are computed every $100$ interactions to keep their computational cost low.


```{code-cell}
:tags: [remove-input, remove-output]
shutil.rmtree("benchmark_er")
shutil.rmtree("benchmark_ec")
shutil.rmtree("benchmark_cc")
shutil.rmtree("benchmark_ce")
shutil.rmtree("tmp", ignore_errors=True)
```
