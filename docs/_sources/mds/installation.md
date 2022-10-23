# Installation

We recommend to set up a virtual environment using [Conda](https://docs.conda.io/en/latest/).
The graph-based visualizations require [Graphviz](https://www.graphviz.org/), which is automatically installed with [PyGraphviz](https://pygraphviz.github.io/documentation/stable/install.html) through conda.
```
conda create --name colosseum
conda activate colosseum
conda install --channel conda-forge pygraphviz
pip install git+https://github.com/MichelangeloConserva/Colosseum
```

If you prefer to use a Python virtual environment, you need to install Graphviz separately.
```
python3 -m venv colosseum
source colosseum/bin/activate

# Ubuntu and Debian 
# sudo apt-get install graphviz graphviz-dev

# Fedora and Red Hat
# sudo dnf install graphviz graphviz-devel

# MacOS
# brew install graphviz

pip install git+https://github.com/MichelangeloConserva/Colosseum
```

We refer to the PyGraphviz [installation tutorial](https://pygraphviz.github.io/documentation/stable/install.html) for installation of Graphviz in WIndows.
