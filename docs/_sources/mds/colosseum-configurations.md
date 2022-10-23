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
# Configure Colosseum

`````{margin}
````{dropdown} Necessary imports
```{code-block} python
from colosseum import config
```
````
`````
```{code-cell}
:tags: [remove-output, remove-input]
from colosseum import config
```


{{col}} allows configuring global directories for the hyperparameters optimization and the benchmarking procedures,
the settings regarding multiprocessing, and some other functionalities.
The <a href="../pdoc_files/colosseum/config.html">``config``</a> module provide the configuring functions.


<h3> Configuring directories </h3>

**Benchmarking**

As a running example, we assume that the goal is to run the **tabular** benchmark separately for some **model-free** and 
**model-based** agents.
So we'll create a main experiment folder called tabular and two sub-folders for the two types of agents.
```{code-cell}
main_experiments_folder = "tabular"
```

When we benchmark the model-free agents, we set the name of the experiment as `model-free` and communicate it to {{col}}.
```{code-cell}
current_experiment_folder_name = "model_free"

# Set the experiment folder and the related hyperoptimization folder
config.set_experiments_folder(main_experiments_folder, current_experiment_folder_name)
config.set_hyperopt_folder(main_experiments_folder, current_experiment_folder_name)

# Show the folder structure for the benchmarking results and for the hyperparameters optimizations results
print("Model-free experiment folder: ", config.get_experiments_folder())
print("Model-free hyperoptimization folder", config.get_hyperopt_folder())

# Code for benchamrking the model-free agents goes here
```

When instead we benchmark the model-based agents, we set the name of the experiment as `model-based` and similarly communicate it to the package.
```{code-cell}
current_experiment_folder_name = "model_based"

# Set the experiment folder and the related hyperoptimization folder
config.set_experiments_folder(main_experiments_folder, current_experiment_folder_name)
config.set_hyperopt_folder(main_experiments_folder, current_experiment_folder_name)

# Show the folder structure for the benchmarking results and for the hyperparameters optimizations results
print("Model-based experiment folder: ", config.get_experiments_folder())
print("Model-based hyperoptimization folder", config.get_hyperopt_folder())

# Code for benchmarking the model-based agents goes here
```

**Hardness analysis**

The package includes cached values of the hardness measures for the benchmark environments and automatically caches the
values for new environments locally by creating a copy of the cached folder from the package.

```{code-cell}
print("Default hardness measures cache folder: ", config.get_hardness_measures_cache_folder())

# If you prefer, you can set a different folder path
config.set_hardness_measures_cache_folder("my_cached_hardness_measures_folder")
print("Custom hardness measures cache folder: ", config.get_hardness_measures_cache_folder())
```


<h3> Verbosity </h3>

{{col}} can provide verbose logging for the agent/MDP interaction, computing the hardness measures, and some time-consuming visualizations.
Note that verbosity is turned off by default.

```{code-cell}
# Enable verbosity
config.enable_verbose_logging()
# Disable verbosity
config.disable_verbose_logging()
```


<h3> Multiprocessing </h3>

{{col}} can leverage multiple cores for benchmarking agents and computing hardness measures.
Note that multiprocessing is turned off by default.

When multiprocessing is enabled, {{col}} sets the number of available cores to the total number of cores available minus two.
```{code-cell}
config.enable_multiprocessing()
print("Number of cores available to the package: ", config.get_available_cores())
```

However, it is possible to manually set the number of cores the package will use.
```{code-cell}
config.set_available_cores(5)
print("Number of cores available to the package: ", config.get_available_cores())
```

Once multiprocessing has been enabled, it can be disabled.
```{code-cell}
# Disable multiprocessing
config.disable_multiprocessing()
print("Number of cores available to the package: ", config.get_available_cores())
```

```{code-cell}
:tags: [remove-input, remove-output]
import shutil
shutil.rmtree("tabular")
shutil.rmtree(config.get_hardness_measures_cache_folder())
```

