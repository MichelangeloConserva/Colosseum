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
# Configurations

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

## {{col}} directories

When running {{col}} experiments, hyperparameter optimization procedures, and hardness analysis, it can be useful to set
up the directories in which logging files will be stored for future analysis.

### Benchmarking

As a running example, we assume that the goal is to run the tabular benchmark on different classes of algorithms, more
specifically on model-free and model-based algorithms, and analyze them separately.

In order to do so, we set the name of the main experiment folder as `tabular`.
```{code-cell}
experiments_folder = "tabular"
```

When we benchmark the model-free agents, we set the name of the experiment as `model-free` and communicate it to {{col}}.
```{code-cell}
experiment_name = "model_free"

# Set the experiment folder and the related hyperoptimization folder
config.set_experiments_folder(experiments_folder, experiment_name)
config.set_hyperopt_folder(experiments_folder, experiment_name)

print("Model-free experiment folder: ", config.get_experiments_folder())
print("Model-free hyperoptimization folder", config.get_hyperopt_folder())
```

When instead we benchmark the model-based agents, we set the name of the experiment as `model-based` and similarly communicate it to the package.
```{code-cell}
experiment_name = "model_based"

# Set the experiment folder and the related hyperoptimization folder
config.set_experiments_folder(experiments_folder, experiment_name)
config.set_hyperopt_folder(experiments_folder, experiment_name)

print("Model-based experiment folder: ", config.get_experiments_folder())
print("Model-based hyperoptimization folder", config.get_hyperopt_folder())
```

### Hardness analysis

The package already includes cached valued of the implemented hardness measures for the benchmark environments, and it also allows the user to cache the values of the measures for other environments locally.
In order to keep the hardness measures in the package separated from the user's new computations, a local folder is created and the {{col}} cached measures are copied therein.

```{code-cell}
print("Default hardness measures cache folder: ", config.get_hardness_measures_cache_folder())

# If you prefer, you can set the folder yourself
config.set_hardness_measures_cache_folder("my_cached_hardness_measures_folder")
print("Custom hardness measures cache folder: ", config.get_hardness_measures_cache_folder())
```

## Verbosity

{{col}} can provide verbose logging for the agent/MDP interaction, computing the hardness measures, and time-consuming visualizations.
Note that verbosity is turned off by default.

```{code-cell}
# Enable verbosity
config.enable_verbose_logging()
# Disable verbosity
config.disable_verbose_logging()
```


## Multiprocessing

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

